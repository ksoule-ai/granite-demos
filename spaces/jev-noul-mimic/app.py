# SPDX-License-Identifier: Apache-2.0
"""Granite Switch + Mellea vs. Jev: side-by-side nouls.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of state with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes.

This Space mimics that primitive with an open model running on ZeroGPU:

1. Granite Switch's base model answers the question, constrained by Mellea
   to exactly "yes" or "no".
2. Granite Switch's embedded ``uncertainty`` adapter (via Mellea's
   ``core.check_certainty``) scores how likely that answer is correct.
3. The certainty is folded into a noul:
   ``noul = certainty if answer == "yes" else 1 - certainty``.

The same state and questions go to the real Jev model via OpenRouter (nouls
only), and the two sets of numbers are shown side by side.
"""

import os

# ZeroGPU doesn't support torch.compile, and llguidance (Mellea's constrained
# decoding) compiles its token-mask kernel. Disabling Dynamo makes that a no-op
# so the kernel runs eagerly. Must be set before torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import granite_switch.hf  # noqa: F401  (registers the granite_switch architecture)
import gradio as gr
import spaces
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mellea import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B_PREVIEW
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

MODEL_ID = IBM_GRANITE_SWITCH_4_1_3B_PREVIEW.hf_model_name
# Jev is reached through OpenRouter's System One API, which the TypeSafe SDK
# speaks natively; a bare model id like "jev-1.13" routes to typesafe/jev-1.13.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13")
MAX_QUESTIONS = 8

# ZeroGPU: place the model on cuda at module level (CUDA is emulated outside
# @spaces.GPU and real inside it). Loading it ourselves lets us pick bf16, then
# Mellea wraps it via custom_config and registers the embedded adapters.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
backend = LocalHFBackend(
    model_id=IBM_GRANITE_SWITCH_4_1_3B_PREVIEW,
    custom_config=(tokenizer, model, torch.device("cuda")),
    load_embedded_adapters=True,
)


class YesNo(BaseModel):
    answer: Literal["yes", "no"]


def _question_prompt(state: str, question: str) -> str:
    return f"{state}\n\nQuestion: {question}\nAnswer yes or no."


def _parse_state(text: str):
    """Jev accepts text or JSON state; pass JSON through as JSON when given."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return value if isinstance(value, (dict, list)) else text


def _parse_questions(text: str) -> list[str]:
    questions = [line.strip() for line in text.splitlines() if line.strip()]
    if not questions:
        raise gr.Error("Enter at least one yes/no question (one per line).")
    if len(questions) > MAX_QUESTIONS:
        raise gr.Error(f"At most {MAX_QUESTIONS} questions per run.")
    return questions


# --------------------------------------------------------------------------- #
# Granite Switch + Mellea
# --------------------------------------------------------------------------- #


def _granite_duration(state: str, questions: list[str]) -> int:
    return 15 + 8 * len(questions)


@spaces.GPU(duration=_granite_duration)
def granite_nouls(state: str, questions: list[str]) -> tuple[list[dict], float]:
    """Return one {answer, certainty, noul} per question, plus GPU seconds."""
    start = time.perf_counter()
    results = []
    for question in questions:
        prompt = _question_prompt(state, question)

        # 1. Base model answers, constrained to the YesNo schema.
        reply, _ = mfuncs.chat(
            prompt,
            ChatContext(),
            backend,
            format=YesNo,
            model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 16},
        )
        answer = YesNo.model_validate_json(reply.content).answer

        # 2. Uncertainty adapter scores that answer. Present it as a plain
        #    "Yes."/"No." turn so the adapter sees an answer, not JSON.
        ctx = (
            ChatContext()
            .add(Message("user", prompt))
            .add(Message("assistant", "Yes." if answer == "yes" else "No."))
        )
        certainty = float(core.check_certainty(ctx, backend))

        # 3. Fold into a noul: P(answer is yes).
        noul = certainty if answer == "yes" else 1.0 - certainty
        results.append({"answer": answer, "certainty": certainty, "noul": noul})
    return results, time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Jev (real API)
# --------------------------------------------------------------------------- #


def jev_nouls(state, questions: list[str]) -> tuple[list[float] | None, float, str]:
    """Return (nouls, seconds, status). nouls is None when the call fails."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, 0.0, "OPENROUTER_API_KEY is not set on this Space."
    keys = [f"q{i}" for i in range(1, len(questions) + 1)]
    start = time.perf_counter()
    try:
        with TypeSafeClient(
            api_key=api_key, base_url=OPENROUTER_BASE_URL, model=JEV_MODEL, timeout=30.0
        ) as client:
            response = client.system_one(
                state=state,
                questions={k: Noul(instructions=q) for k, q in zip(keys, questions)},
            )
    except TypeSafeError as e:
        return None, time.perf_counter() - start, f"Jev call failed: {e}"
    elapsed = time.perf_counter() - start
    return (
        [response.nouls[k].noul for k in keys],
        elapsed,
        f"`{response.model}` via OpenRouter",
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def compare(state_text: str, questions_text: str):
    if not state_text.strip():
        raise gr.Error("Enter some state for the model to decide about.")
    questions = _parse_questions(questions_text)
    jev_state = _parse_state(state_text)

    # Jev runs over the network while Granite runs on the GPU.
    with ThreadPoolExecutor(max_workers=1) as pool:
        jev_future = pool.submit(jev_nouls, jev_state, questions)
        granite, granite_s = granite_nouls(state_text, questions)
        jev, jev_s, jev_status = jev_future.result()

    rows = []
    for i, (question, g) in enumerate(zip(questions, granite)):
        j = jev[i] if jev is not None else None
        rows.append(
            [
                question,
                None if j is None else round(j, 3),
                round(g["noul"], 3),
                round(g["certainty"], 3),
            ]
        )

    jev_line = (
        f"**Jev:** {jev_s * 1000:.0f} ms end to end · {jev_status}"
        if jev is not None
        else f"**Jev:** unavailable · {jev_status}"
    )
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch + Mellea:** {granite_s * 1000:.0f} ms GPU time "
        f"for {len(questions)} question(s) · `{MODEL_ID}` · "
        f"{2 * len(questions)} forward passes (answer + uncertainty adapter each)"
    )
    return rows, timing


EXAMPLES = [
    [
        "I was charged twice for my subscription this month and nobody has "
        "answered my last two emails. Please fix this today or cancel my account.",
        "Is this about billing?\nIs the customer angry?\n"
        "Does the customer want a refund?\nIs this a bug report?",
    ],
    [
        '{"order_id": "A-1042", "status": "delivered", "delivered_at": '
        '"2026-09-20", "customer_message": "The box arrived crushed and the '
        'mug inside is in pieces."}',
        "Was the item damaged?\nHas the order been delivered?\n"
        "Is the customer asking to change the shipping address?",
    ],
    [
        "The Eiffel Tower was completed in 1889 and is located in Berlin.",
        "Is this statement entirely accurate?\nDoes the statement mention a year?\n"
        "Is the Eiffel Tower in France?",
    ],
]

with gr.Blocks(title="Granite Switch nouls vs Jev") as demo:
    gr.Markdown(
        "# Granite Switch nouls vs. Jev\n"
        "[Jev](https://docs.typesafe.ai) is TypeSafe AI's *System One* model. A "
        "**noul** is its yes/no primitive: one calibrated probability that the "
        "answer is yes. This Space rebuilds that primitive from open parts, "
        "[Granite Switch](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) "
        "and [Mellea](https://mellea.ai), and compares the two on the same input.\n\n"
        "**How the Granite side works:** Mellea constrains the base model to answer "
        "`yes` or `no`. Granite Switch's embedded `uncertainty` adapter then scores "
        "that answer, and the result becomes a noul: "
        "`certainty` if the answer is yes, `1 − certainty` if it's no."
    )
    with gr.Row():
        with gr.Column():
            state = gr.Textbox(
                label="State (text or JSON)",
                lines=6,
                placeholder="The input the models decide about.",
            )
            questions = gr.Textbox(
                label=f"Yes/no questions (one per line, up to {MAX_QUESTIONS})",
                lines=5,
            )
            run = gr.Button("Compare", variant="primary")
        with gr.Column():
            table = gr.Dataframe(
                headers=[
                    "Question",
                    "Jev noul",
                    "Granite noul",
                    "Granite certainty",
                ],
                datatype=["str", "number", "number", "number"],
                interactive=False,
                wrap=True,
            )
            timing = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[state, questions])
    gr.Markdown(
        "Note: the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), and "
        "Mellea returns the probability-weighted average of those bins, so Granite "
        "nouls always fall between 0.05 and 0.95. Granite time is GPU compute only; it doesn't "
        "include ZeroGPU queueing. Jev time is the full API round trip."
    )
    run.click(compare, inputs=[state, questions], outputs=[table, timing])

if __name__ == "__main__":
    demo.launch()
