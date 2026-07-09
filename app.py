import html
import re

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, TextIteratorStreamer
from threading import Thread

MODEL_ID = "ibm-granite/granite-switch-4.1-8b-preview"

# Must import before AutoModelForCausalLM to register GraniteSwitchForCausalLM
import granite_switch.hf  # noqa: F401, E402

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Adapter names must match the model's chat template exactly — the mixed
# hyphen/underscore usage is upstream's, not a typo. Authoritative list:
# https://github.com/generative-computing/granite-switch/blob/main/docs/adapter_catalog.html
# (mirrored in tests/adapter_catalog.json, enforced by tests/test_app.py).
#
# This demo offers only the Core and Guardian (safety) libraries. Each
# interaction is two turns: a standard generation first, then the selected
# adapter is automatically invoked on that generation via an adapter-specific
# follow-up prompt that appears in the chat history.

ADAPTER_CHOICES = [
    # Core library
    "requirement-check",
    "uncertainty",
    "context-attribution",
    # Guardian library
    "guardian-core",
    "factuality-detection",
    "factuality-correction",
    "policy-guardrails",
]

# Adapters whose second turn is grounded on a documents/context input
DOC_ADAPTERS = {
    "context-attribution",
    "factuality-detection",
    "factuality-correction",
}

# Adapters that need a free-text rule set (requirements or a policy)
RULES_ADAPTERS = {"requirement-check", "policy-guardrails"}

ADAPTER_DESCRIPTIONS = {
    "requirement-check": "After the response is generated, checks whether it satisfies your stated requirements. Returns {\"score\": \"yes\"|\"no\"}.",
    "uncertainty": "After the response is generated, scores how certain the model is about it.",
    "context-attribution": "After the response is generated, identifies which parts of the provided context support it.",
    "guardian-core": "After the response is generated, screens it for harm, unsafe content, or bias.",
    "factuality-detection": "After the response is generated, flags factual errors in it against the context documents.",
    "factuality-correction": "After the response is generated, rewrites it to fix factual errors against the context documents.",
    "policy-guardrails": "After the response is generated, checks it against your natural-language policy.",
}

# Fixed instruction the requirement-check adapter was trained to read after
# the constraints (verbatim from the granite-switch protocol; see
# requirement_check.py, the reference CLI implementation).
EVALUATION_PROMPT = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)

FOLLOWUP_PROMPTS = {
    "uncertainty": "How certain are you that your previous response is correct?",
    "context-attribution": "Which parts of the provided documents support your previous response?",
    "guardian-core": "Is the previous assistant response harmful, unsafe, or biased?",
    "factuality-detection": "Identify any factual errors in the previous assistant response against the provided documents.",
    "factuality-correction": "Rewrite the previous assistant response to correct any factual errors, based on the provided documents.",
}


def adapter_followup(adapter_name, rules):
    """The user turn that automatically invokes the adapter on the last response."""
    if adapter_name == "requirement-check":
        return f"<requirements> {rules.strip()}\n{EVALUATION_PROMPT}"
    if adapter_name == "policy-guardrails":
        return (
            f"Policy: {rules.strip()}\n"
            "Does the previous assistant response comply with this policy?"
        )
    return FOLLOWUP_PROMPTS[adapter_name]


# --------------------------------------------------------------- KV caching
# Both turns of an interaction run in one GPU call sharing a DynamicCache.
# The adapter turn reuses the KV of its common token prefix with turn 1
# (prompt + first generation) instead of re-prefilling it — sound for
# Granite Switch's aLoRA adapters, whose weights only apply after the
# activation token. The reuse ratio is published under each user message.
# Turn 1 is always a cold start: ZeroGPU releases the GPU (and the cache)
# between interactions.

CACHE_NOTE_PREFIX = "\n\n⚡ `KV cache: "


def cache_note(hits, total):
    pct = 100 * hits / total if total else 0.0
    return f"{CACHE_NOTE_PREFIX}{hits}/{total} prompt tokens reused ({pct:.0f}% hit)`"


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Gradio 6 block format
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def strip_cache_note(content):
    return _content_text(content).split(CACHE_NOTE_PREFIX)[0]


# The adapter's prompt and response bubbles are tinted purple: their content
# is wrapped in marker spans the CSS targets via :has(). The wrappers are
# UI-only and must never reach the model (see _clean_content).
ADAPTER_SPAN_RE = re.compile(
    r'^<span class="adapter-(?:prompt|response)">(.*)</span>$', re.S
)


def followup_display(followup):
    return f'<span class="adapter-prompt">{html.escape(followup)}</span>'


def adapter_response_display(text):
    return f'<span class="adapter-response">{html.escape(text)}</span>'


def _clean_content(content):
    text = strip_cache_note(content)
    m = ADAPTER_SPAN_RE.match(text)
    if m:
        text = html.unescape(m.group(1))
    return text


def _common_prefix_len(a, b):
    n = min(a.shape[-1], b.shape[-1])
    if n == 0:
        return 0
    neq = (a[:n] != b[:n]).nonzero()
    return int(neq[0]) if len(neq) else n


def _crop_cache(cache, n):
    cache.crop(n)


def _render_text(messages, adapter_name, context_docs, add_generation_prompt=True):
    template_kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    if adapter_name:
        template_kwargs["adapter_name"] = adapter_name
    if context_docs.strip():
        template_kwargs["documents"] = [{"doc_id": "1", "text": context_docs.strip()}]
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def _render(messages, adapter_name, context_docs):
    prompt = _render_text(messages, adapter_name, context_docs)
    return tokenizer(prompt, return_tensors="pt").input_ids


def _extend_sequence(seq1, turn1_text, gen1_text, full2_text):
    """Turn-2 input ids that reuse turn 1's exact tokens.

    Re-tokenizing the full turn-2 render does not reproduce the token ids
    the model actually generated (BPE decode->encode is not the identity),
    which zeroes out KV reuse of the decode tokens. Instead, extend the
    real turn-1 sequence tensor with only the tokenized suffix (follow-up
    turn + adapter generation prompt). Returns None when the turn-2 render
    is not a textual extension of what the model saw (e.g. different
    documents), in which case the caller re-tokenizes from scratch.
    """
    prefix_text = turn1_text + gen1_text + tokenizer.eos_token
    if not full2_text.startswith(prefix_text):
        return None
    base = seq1
    if int(base[-1]) != tokenizer.eos_token_id:  # hit max_new_tokens, no EOS
        base = torch.cat([base, base.new_tensor([tokenizer.eos_token_id])])
    suffix = tokenizer(
        full2_text[len(prefix_text):], return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(base.device)
    return torch.cat([base, suffix]).unsqueeze(0)


def _stream_one(input_ids, cache, max_new_tokens, temperature, holder):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        past_key_values=cache,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
    )

    def run():
        holder["sequence"] = model.generate(**gen_kwargs)[0]

    thread = Thread(target=run)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial
    thread.join()
    holder["text"] = partial


@spaces.GPU(duration=120)
def two_turn_generate(messages, adapter_name, followup, turn1_docs, turn2_docs,
                      max_new_tokens, temperature):
    """One interaction on one GPU slot: standard turn, then the adapter turn.

    Yields ("stats", turn, hits, prompt_tokens) before each generation and
    ("partial", turn, text) while it streams.
    """
    cache = DynamicCache()

    turn1_text = _render_text(messages, None, turn1_docs)
    ids1 = tokenizer(turn1_text, return_tensors="pt").input_ids.to(model.device)
    yield ("stats", 1, 0, int(ids1.shape[1]))  # cold start, nothing cached
    holder = {}
    for partial in _stream_one(ids1, cache, max_new_tokens, temperature, holder):
        yield ("partial", 1, partial)
    seq1 = holder["sequence"]

    messages2 = messages + [
        {"role": "assistant", "content": holder["text"]},
        {"role": "user", "content": followup},
    ]
    full2_text = _render_text(messages2, adapter_name, turn2_docs)
    ids2 = None
    if turn1_docs.strip() == turn2_docs.strip():
        ids2 = _extend_sequence(seq1, turn1_text, holder["text"], full2_text)
        if ids2 is not None:
            ids2 = ids2.to(model.device)
    if ids2 is None:  # docs differ or render isn't an extension — start over
        ids2 = tokenizer(full2_text, return_tensors="pt").input_ids.to(model.device)
    total2 = int(ids2.shape[1])
    hits = max(0, min(
        _common_prefix_len(seq1.to(ids2.device), ids2[0]),
        cache.get_seq_length(),
        total2 - 1,  # generate() must have at least one new token to process
    ))
    _crop_cache(cache, hits)
    yield ("stats", 2, hits, total2)
    for partial in _stream_one(ids2, cache, max_new_tokens, 0.0, {}):
        yield ("partial", 2, partial)


def _as_messages(history):
    # history is Gradio 6 messages format: [{"role": ..., "content": ...}];
    # cache notes and adapter-prompt styling are UI decoration and must
    # never reach the prompt
    out = []
    for m in history:
        content = _clean_content(m["content"])
        if content:
            out.append({"role": m["role"], "content": content})
    return out


def user_submit(message, history):
    # Single-shot demo: hide the input and Send button as soon as a message
    # is submitted; only Clear remains once the responses have generated.
    return (
        gr.update(value="", visible=False),
        history + [{"role": "user", "content": message}],
        gr.update(visible=False),
    )


def bot_respond(history, adapter_choice, context_docs, rules, max_new_tokens, temperature):
    """Two-turn flow: standard generation, then automatic adapter invocation.

    Turn 1 is a plain (no adapter) generation of the user's prompt, grounded
    on the context documents when provided. Turn 2 appends the adapter's
    follow-up prompt to the chat as a user message — visible in the UI
    without user action — and streams the adapter's response. The adapter
    turn is greedy (temperature 0): it is a judge, not a generator.

    Each generation's KV-cache reuse (hit rate) is appended as a note under
    the response it produced. The adapter's prompt is displayed in purple
    italics (see followup_display / the Blocks css).
    """
    # Hidden Gradio textboxes arrive as None, not ""
    context_docs = context_docs or ""
    rules = rules or ""

    followup = adapter_followup(adapter_choice, rules)
    adapter_docs = context_docs if adapter_choice in DOC_ADAPTERS else ""

    events = two_turn_generate(
        _as_messages(history), adapter_choice, followup,
        context_docs, adapter_docs, max_new_tokens, temperature,
    )
    notes = {}
    for event in events:
        if event[0] == "stats":
            _, turn, hits, total = event
            notes[turn] = cache_note(hits, total)
            if turn == 2:
                # turn 1 is complete — publish its note under its response,
                # then the adapter's prompt appears in the chat automatically
                history[-1]["content"] += notes[1]
                history = history + [{"role": "user", "content": followup_display(followup)}]
                yield history
            history = history + [{"role": "assistant", "content": ""}]
            yield history
        else:
            _, turn, partial = event
            history[-1]["content"] = (
                adapter_response_display(partial) if turn == 2 else partial
            )
            yield history
    history[-1]["content"] += notes[2]
    yield history


def get_adapter_description(adapter_choice):
    return ADAPTER_DESCRIPTIONS.get(adapter_choice, "")


def update_visibility(adapter_choice):
    show_docs = adapter_choice in DOC_ADAPTERS
    show_rules = adapter_choice in RULES_ADAPTERS
    return (
        gr.update(visible=show_docs),
        gr.update(visible=show_rules),
    )


CSS = """
/* the whole bubble turns purple; inner elements stay transparent so the
   text never shows its own background patch */
.message:has(.adapter-prompt),
.message:has(.adapter-response) {
    background-color: #c4b5fd !important;
    border-color: #a78bfa !important;
    color: #1f2937 !important;
}
.adapter-prompt, .adapter-response,
.message:has(.adapter-prompt) *,
.message:has(.adapter-response) * {
    background: transparent !important;
    color: inherit !important;
}
.adapter-prompt { font-style: italic; }
"""

with gr.Blocks(title="Granite Switch 4.1 8B Demo") as demo:
    gr.Markdown(
        """
# 🪨 Granite Switch 4.1 8B — ZeroGPU Demo

[`ibm-granite/granite-switch-4.1-8b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
is a single 8B checkpoint with **embedded LoRA adapters**. This demo shows the
**Core** (requirement checking, certainty, contextual attribution) and
**Guardian** (safety, factuality, policy) libraries in a two-turn flow:

1. Pick an adapter and submit a prompt.
2. The model first answers normally (no adapter).
3. The adapter's follow-up prompt (shown in *purple italics*) then appears
   in the chat automatically, and the adapter's verdict on that answer
   streams back. Use **Clear** to start over.

Under each response a ⚡ note reports the **KV-cache hit rate** for the
generation that produced it. The adapter turn reuses the cached
conversation prefix (Granite Switch's aLoRA adapters only apply after
their activation token, so the base KV is valid); the first turn is
always a cold start because ZeroGPU releases the GPU between interactions.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480)
            user_input = gr.Textbox(
                placeholder="Type your message…",
                label="Your message",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Switch Configuration")
            adapter_dropdown = gr.Dropdown(
                choices=ADAPTER_CHOICES,
                value="requirement-check",
                label="Adapter",
                info="Runs automatically on the model's first answer.",
            )
            adapter_desc = gr.Markdown(
                value=ADAPTER_DESCRIPTIONS["requirement-check"],
                label="",
            )
            context_box = gr.Textbox(
                label="Context / Documents",
                placeholder="Paste document text here (grounds the answer and the adapter check)…",
                lines=5,
                visible=False,
            )
            rules_box = gr.Textbox(
                label="Requirements / Policy",
                placeholder="Requirements the response must satisfy, or the policy to check against…",
                lines=4,
                visible=True,
            )
            gr.Markdown("### Generation")
            max_tokens = gr.Slider(
                minimum=64, maximum=2048, value=512, step=64, label="Max new tokens"
            )
            temperature = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.7, step=0.05, label="Temperature"
            )

    adapter_dropdown.change(
        fn=get_adapter_description,
        inputs=adapter_dropdown,
        outputs=adapter_desc,
    )
    adapter_dropdown.change(
        fn=update_visibility,
        inputs=adapter_dropdown,
        outputs=[context_box, rules_box],
    )

    for trigger in (submit_btn.click, user_input.submit):
        trigger(
            fn=user_submit,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, submit_btn],
            queue=False,
        ).then(
            fn=bot_respond,
            inputs=[chatbot, adapter_dropdown, context_box, rules_box, max_tokens, temperature],
            outputs=chatbot,
        )

    clear_btn.click(
        fn=lambda: ([], gr.update(value="", visible=True), gr.update(visible=True)),
        outputs=[chatbot, user_input, submit_btn],
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
