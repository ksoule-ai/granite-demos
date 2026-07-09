import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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


@spaces.GPU(duration=120)
def generate(messages, adapter_name, context_docs, max_new_tokens, temperature):
    """Stream one completion for `messages`, optionally through an adapter."""
    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
    )
    if adapter_name:
        template_kwargs["adapter_name"] = adapter_name
    if context_docs.strip():
        template_kwargs["documents"] = [{"doc_id": "1", "text": context_docs.strip()}]

    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


def _as_messages(history):
    # history is Gradio 6 messages format: [{"role": ..., "content": ...}]
    return [
        {"role": m["role"], "content": m["content"]} for m in history if m["content"]
    ]


def user_submit(message, history):
    return "", history + [{"role": "user", "content": message}]


def bot_respond(history, adapter_choice, context_docs, rules, max_new_tokens, temperature):
    """Two-turn flow: standard generation, then automatic adapter invocation.

    Turn 1 is a plain (no adapter) generation of the user's prompt, grounded
    on the context documents when provided. Turn 2 appends the adapter's
    follow-up prompt to the chat as a user message — visible in the UI
    without user action — and streams the adapter's response. The adapter
    turn is greedy (temperature 0): it is a judge, not a generator.
    """
    # Hidden Gradio textboxes arrive as None, not ""
    context_docs = context_docs or ""
    rules = rules or ""

    # Turn 1: standard generation of the user's prompt
    history = history + [{"role": "assistant", "content": ""}]
    for partial in generate(
        _as_messages(history[:-1]), None, context_docs, max_new_tokens, temperature
    ):
        history[-1]["content"] = partial
        yield history

    # Turn 2: the adapter-specific prompt appears in the chat automatically
    followup = adapter_followup(adapter_choice, rules)
    history = history + [{"role": "user", "content": followup}]
    yield history

    adapter_docs = context_docs if adapter_choice in DOC_ADAPTERS else ""
    history = history + [{"role": "assistant", "content": ""}]
    for partial in generate(
        _as_messages(history[:-1]), adapter_choice, adapter_docs, max_new_tokens, 0.0
    ):
        history[-1]["content"] = partial
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
3. The adapter's follow-up prompt then appears in the chat automatically,
   and the adapter's verdict on that answer streams back.
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
            outputs=[user_input, chatbot],
            queue=False,
        ).then(
            fn=bot_respond,
            inputs=[chatbot, adapter_dropdown, context_box, rules_box, max_tokens, temperature],
            outputs=chatbot,
        )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])


if __name__ == "__main__":
    demo.launch()
