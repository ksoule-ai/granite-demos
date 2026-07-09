import html

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-switch-4.1-8b-preview"

# Must import before AutoModelForCausalLM to register GraniteSwitchForCausalLM
import granite_switch.hf  # noqa: F401, E402

from mellea import MelleaSession
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components.intrinsic import core as core_intrinsics
from mellea.stdlib.components.intrinsic import guardian as guardian_intrinsics
from mellea.stdlib.requirements import ALoraRequirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

from switch_backend import SwitchBackend

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Mellea drives all generation. SwitchBackend (switch_backend.py) teaches
# mellea's HF backend to activate the checkpoint's *embedded* adapters via
# the chat template's control tokens instead of PEFT loading.
backend = SwitchBackend(MODEL_ID, custom_config=(tokenizer, model, model.device))
backend.register_embedded_adapters()

# Adapter names must match the model's adapter_index.json exactly — the mixed
# hyphen/underscore usage is upstream's, not a typo. Authoritative list:
# https://github.com/generative-computing/granite-switch/blob/main/docs/adapter_catalog.html
# (mirrored in tests/adapter_catalog.json, enforced by tests/test_app.py).
#
# This demo offers only the Core and Guardian (safety) libraries.
ADAPTER_CHOICES = [
    # Core library
    "requirement-check",
    "uncertainty",
    # Guardian library
    "guardian-core",
]

ADAPTER_DESCRIPTIONS = {
    "requirement-check": (
        "Drives a Mellea **instruct–validate–repair** loop: each draft answer is "
        "validated by the requirement-check aLoRA and regenerated until it passes "
        "(or the attempt budget runs out)."
    ),
    "uncertainty": "After the final answer, scores how certain the model is about it (0–1).",
    "guardian-core": "After the final answer, screens it for harm and reports a risk score (0–1).",
}

# Adapters that judge the final answer after it is produced (requirement-check
# instead steers generation through the IVR loop).
JUDGE_ADAPTERS = ["uncertainty", "guardian-core"]


# ------------------------------------------------------------------ rendering
# Adapter/verdict bubbles are tinted purple: their content is wrapped in a
# marker span the CSS targets via :has(). The wrappers are UI-only.
def verdict_display(text):
    return f'<span class="adapter-response">{html.escape(text)}</span>'


def status_display(text):
    return f'<span class="adapter-prompt">{html.escape(text)}</span>'


ATTEMPT_NOTE = {True: "✅ requirement satisfied", False: "❌ requirement not satisfied"}


# ---------------------------------------------------------------- generation
@spaces.GPU(duration=300)
def run_switch(prompt, adapters, rules, max_new_tokens, temperature, loop_budget):
    """One interaction on one GPU slot, driven end-to-end by Mellea.

    If requirement-check is selected (and requirements were given), the answer
    is produced by ``m.instruct`` under a RejectionSamplingStrategy: generate,
    validate with the embedded requirement-check aLoRA, retry on failure.
    Otherwise a single plain generation runs. Each remaining selected adapter
    then judges the final answer via its Mellea intrinsic. Yields events:

      ("status", text)                      — progress notes for the UI
      ("attempt", i, text, passed|None)     — a generation attempt (passed is
                                              None outside the IVR loop)
      ("final", index, success, attempts)   — which attempt was selected
      ("verdict", adapter, text)            — a judge adapter's verdict
    """
    gen_options = {"max_new_tokens": int(max_new_tokens)}
    if temperature > 0:
        gen_options["do_sample"] = True
        gen_options["temperature"] = float(temperature)
    else:
        gen_options["do_sample"] = False

    m = MelleaSession(backend, ctx=ChatContext())
    use_ivr = "requirement-check" in adapters and rules.strip()

    if use_ivr:
        yield (
            "status",
            f"Running instruct → validate → repair (budget: {int(loop_budget)} attempts, "
            "validated by the requirement-check aLoRA)…",
        )
        result = m.instruct(
            prompt,
            requirements=[ALoraRequirement(rules.strip())],
            strategy=RejectionSamplingStrategy(loop_budget=int(loop_budget)),
            return_sampling_results=True,
            model_options=gen_options,
        )
        for i, (gen, validations) in enumerate(
            zip(result.sample_generations, result.sample_validations), start=1
        ):
            passed = all(bool(v) for _, v in validations)
            yield ("attempt", i, str(gen), passed)
        yield (
            "final",
            result.result_index,
            result.success,
            len(result.sample_generations),
        )
    else:
        yield ("status", "Generating…")
        output = m.instruct(prompt, strategy=None, model_options=gen_options)
        yield ("attempt", 1, str(output), None)

    for adapter in adapters:
        if adapter == "uncertainty":
            yield ("status", "uncertainty aLoRA is scoring the answer…")
            certainty = core_intrinsics.check_certainty(m.ctx, backend)
            verdict = "confident" if certainty >= 0.5 else "not confident"
            yield (
                "verdict",
                "uncertainty",
                f"uncertainty → certainty {certainty:.2f}: the model is {verdict} in this answer.",
            )
        elif adapter == "guardian-core":
            yield ("status", "guardian-core aLoRA is screening the answer…")
            risk = guardian_intrinsics.guardian_check(m.ctx, backend, criteria="harm")
            verdict = "⚠️ risk detected" if risk > 0.5 else "no harm detected"
            yield (
                "verdict",
                "guardian-core",
                f"guardian-core → harm risk {risk:.2f}: {verdict}.",
            )


# ------------------------------------------------------------------ UI logic
def user_submit(message, history):
    # Single-shot demo: hide the input and Send button as soon as a message
    # is submitted; only Clear remains once the responses have generated.
    return (
        gr.update(value="", visible=False),
        history + [{"role": "user", "content": message}],
        gr.update(visible=False),
    )


def bot_respond(history, adapter_choices, rules, max_new_tokens, temperature, loop_budget):
    """Render run_switch's event stream into the chat history.

    Attempts appear as normal assistant bubbles (failed IVR attempts carry a
    ❌ note, the selected one a ✅). Adapter verdicts and progress notes are
    purple (see verdict_display / status_display / the css). A status bubble
    is replaced by the next real event.
    """
    # Hidden Gradio textboxes arrive as None, not ""
    rules = rules or ""
    if isinstance(adapter_choices, str):  # tolerate a bare adapter name
        adapter_choices = [adapter_choices]
    adapters = list(adapter_choices or [])

    prompt = history[-1]["content"]
    status_pending = False

    def drop_status(h):
        return h[:-1] if status_pending else h

    for event in run_switch(prompt, adapters, rules, max_new_tokens, temperature, loop_budget):
        kind = event[0]
        if kind == "status":
            history = drop_status(history) + [
                {"role": "assistant", "content": status_display(event[1])}
            ]
            status_pending = True
        elif kind == "attempt":
            _, i, text, passed = event
            if passed is not None:
                text = f"{text}\n\n*attempt {i}: {ATTEMPT_NOTE[passed]}*"
            history = drop_status(history) + [{"role": "assistant", "content": text}]
            status_pending = False
        elif kind == "final":
            _, index, success, attempts = event
            note = (
                f"✅ IVR loop converged on attempt {index + 1} of {attempts}."
                if success
                else f"⚠️ Attempt budget exhausted after {attempts} tries; showing attempt {index + 1}."
            )
            history = drop_status(history) + [
                {"role": "assistant", "content": verdict_display(note)}
            ]
            status_pending = False
        elif kind == "verdict":
            text = event[2]
            history = drop_status(history) + [
                {"role": "assistant", "content": verdict_display(text)}
            ]
            status_pending = False
        yield history
    if status_pending:
        yield drop_status(history)


def get_adapter_description(adapter_choices):
    return "\n\n".join(
        f"**{a}**: {ADAPTER_DESCRIPTIONS[a]}"
        for a in (adapter_choices or []) if a in ADAPTER_DESCRIPTIONS
    )


def update_visibility(adapter_choices):
    selected = set(adapter_choices or [])
    show_rules = "requirement-check" in selected
    return gr.update(visible=show_rules), gr.update(visible=show_rules)


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
# 🪨 Granite Switch 4.1 8B — Mellea IVR Demo (ZeroGPU)

[`ibm-granite/granite-switch-4.1-8b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
is a single 8B checkpoint with **embedded LoRA adapters**. This demo drives it
with [Mellea](https://docs.mellea.ai)'s HuggingFace backend:

1. Pick adapters, optionally state **requirements**, and submit a prompt.
2. With **requirement-check** selected, Mellea runs an
   **instruct → validate → repair** loop: every draft is judged by the
   embedded requirement-check aLoRA and regenerated until it passes or the
   attempt budget runs out. Each attempt appears in the chat with its verdict.
3. **uncertainty** and **guardian-core** then judge the final answer; their
   verdicts appear in *purple*. Use **Clear** to start over.

The adapters are activated by control tokens spliced in by the model's chat
template — no separate adapter weights are loaded. Judged turns are always
greedy; only the drafts use your temperature.
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
                value=["requirement-check"],
                multiselect=True,
                label="Adapters",
                info="requirement-check steers generation (IVR); the others judge the result.",
            )
            adapter_desc = gr.Markdown(
                value=get_adapter_description(["requirement-check"]),
                label="",
            )
            rules_box = gr.Textbox(
                label="Requirements",
                placeholder="Requirements the response must satisfy…",
                lines=4,
                visible=True,
            )
            loop_budget = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="IVR attempt budget",
                info="Max generate→validate cycles for requirement-check.",
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
        outputs=[rules_box, loop_budget],
    )

    for trigger in (submit_btn.click, user_input.submit):
        trigger(
            fn=user_submit,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, submit_btn],
            queue=False,
        ).then(
            fn=bot_respond,
            inputs=[chatbot, adapter_dropdown, rules_box, max_tokens, temperature, loop_budget],
            outputs=chatbot,
        )

    clear_btn.click(
        fn=lambda: ([], gr.update(value="", visible=True), gr.update(visible=True)),
        outputs=[chatbot, user_input, submit_btn],
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
