import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_ID = "ibm-granite/granite-switch-4.1-30b-preview"

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

# Adapters that take a documents/context input
DOC_ADAPTERS = {
    "answerability",
    "hallucination_detection",
    "citations",
    "context-attribution",
    "factuality-detection",
    "factuality-correction",
}

# Adapters that need a requirements field
REQUIREMENTS_ADAPTERS = {"requirement-check"}

ADAPTER_CHOICES = [
    "None (standard chat)",
    # RAG library
    "query_rewrite",
    "query_clarification",
    "answerability",
    "hallucination_detection",
    "citations",
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

ADAPTER_DESCRIPTIONS = {
    "None (standard chat)": "Standard Granite 4.1 30B instruct chat — no adapter active.",
    "query_rewrite": "Rewrites the latest user query into a standalone form for better retrieval.",
    "query_clarification": "Asks a clarifying question when the user query is ambiguous.",
    "answerability": "Judges whether the question is answerable from the provided documents.",
    "hallucination_detection": "Detects whether the assistant response hallucinates beyond the provided documents.",
    "citations": "Generates inline citations for a response grounded in documents.",
    "requirement-check": "Checks whether a response satisfies stated requirements. Returns {\"score\": \"yes\"|\"no\"}.",
    "uncertainty": "Scores how certain the model is about its answer.",
    "context-attribution": "Identifies which parts of the context support the answer.",
    "guardian-core": "Core safety guardian — screens inputs or outputs for harm.",
    "factuality-detection": "Flags factual errors in the assistant response against context documents.",
    "factuality-correction": "Rewrites the response to fix factual errors against context documents.",
    "policy-guardrails": "Checks a scenario against a natural-language policy.",
}


@spaces.GPU(duration=120)
def respond(message, history, adapter_choice, context_docs, requirements, max_new_tokens, temperature):
    adapter_name = None if adapter_choice == "None (standard chat)" else adapter_choice

    # requirement-check protocol: the final user turn carries the constraints
    # in a <requirements> block (same as requirement_check.py, the CLI demo).
    if adapter_name in REQUIREMENTS_ADAPTERS and requirements.strip():
        message = f"<requirements> {requirements.strip()}\n{message}"

    # history is Gradio 6 messages format: [{"role": ..., "content": ...}]
    messages = [
        {"role": m["role"], "content": m["content"]} for m in history if m["content"]
    ]
    messages.append({"role": "user", "content": message})

    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
    )
    if adapter_name:
        template_kwargs["adapter_name"] = adapter_name
    if adapter_name in DOC_ADAPTERS and context_docs.strip():
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


def get_adapter_description(adapter_choice):
    return ADAPTER_DESCRIPTIONS.get(adapter_choice, "")


def update_visibility(adapter_choice):
    show_docs = adapter_choice in DOC_ADAPTERS
    show_reqs = adapter_choice in REQUIREMENTS_ADAPTERS
    return (
        gr.update(visible=show_docs),
        gr.update(visible=show_reqs),
    )


with gr.Blocks(title="Granite Switch 4.1 30B Demo") as demo:
    gr.Markdown(
        """
# 🪨 Granite Switch 4.1 30B — ZeroGPU Demo

[`ibm-granite/granite-switch-4.1-30b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview)
is a single 30B checkpoint with **12 embedded LoRA adapters** across three Granite Libraries:
**RAG** (query rewriting, answerability, hallucination detection, citation),
**Core** (requirement checking, certainty, contextual attribution), and
**Guardian** (safety, bias, factuality). Select an adapter below to switch capabilities
without loading a different model.
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
                value="None (standard chat)",
                label="Adapter",
                info="Select which capability to activate.",
            )
            adapter_desc = gr.Markdown(
                value=ADAPTER_DESCRIPTIONS["None (standard chat)"],
                label="",
            )
            context_box = gr.Textbox(
                label="Context / Documents",
                placeholder="Paste document text here (required for RAG adapters)…",
                lines=5,
                visible=False,
            )
            requirements_box = gr.Textbox(
                label="Requirements",
                placeholder="List the requirements the response must satisfy…",
                lines=4,
                visible=False,
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
        outputs=[context_box, requirements_box],
    )

    def user_submit(message, history):
        return "", history + [{"role": "user", "content": message}]

    def bot_respond(history, adapter_choice, context_docs, requirements, max_new_tokens, temperature):
        message = history[-1]["content"]
        prior_history = history[:-1]
        history = history + [{"role": "assistant", "content": ""}]
        for partial in respond(message, prior_history, adapter_choice, context_docs, requirements, max_new_tokens, temperature):
            history[-1]["content"] = partial
            yield history

    for trigger in (submit_btn.click, user_input.submit):
        trigger(
            fn=user_submit,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(
            fn=bot_respond,
            inputs=[chatbot, adapter_dropdown, context_box, requirements_box, max_tokens, temperature],
            outputs=chatbot,
        )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])


if __name__ == "__main__":
    demo.launch()
