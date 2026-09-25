# SPDX-License-Identifier: Apache-2.0
"""Granite Switch + Mellea vs. Jev: side-by-side nouls.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of state with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes.

This Space mimics that primitive with Granite Switch served by vLLM on a
Hugging Face Inference Endpoint:

1. Granite Switch's base model answers the question, constrained by Mellea
   to exactly "yes" or "no".
2. Granite Switch's embedded ``uncertainty`` adapter (via Mellea's
   ``core.check_certainty``) scores how likely that answer is correct.
3. The certainty is folded into a noul:
   ``noul = certainty if answer == "yes" else 1 - certainty``.

All questions are sent to the endpoint concurrently so vLLM batches them.
The same state and questions go to the real Jev model via OpenRouter (nouls
only), and the two sets of numbers are shown side by side.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import gradio as gr
import httpx
from pydantic import BaseModel

from mellea import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

# Granite Switch endpoint (vLLM, OpenAI-compatible). HF_ENDPOINT_URL ends in /v1.
ENDPOINT_URL = os.environ["HF_ENDPOINT_URL"].rstrip("/")
HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_ID = os.environ.get("MODEL_ID", "ibm-granite/granite-switch-4.1-3b-preview")
# The endpoint scales to zero; the first request after idle wakes it.
WAKE_TIMEOUT_S = 420

# Jev is reached through OpenRouter's System One API, which the TypeSafe SDK
# speaks natively; a bare model id like "jev-1.13" routes to typesafe/jev-1.13.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13")
MAX_QUESTIONS = 8

# load_embedded_adapters fetches only the adapters' I/O configs from the Hub;
# the adapter weights are already inside the served model.
backend = OpenAIBackend(
    model_id=MODEL_ID,
    formatter=TemplateFormatter(model_id=MODEL_ID),
    base_url=ENDPOINT_URL,
    api_key=HF_TOKEN,
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


def wait_for_endpoint() -> float:
    """Block until the endpoint answers /models; return seconds spent waiting."""
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    while True:
        try:
            if httpx.get(f"{ENDPOINT_URL}/models", headers=headers, timeout=10).is_success:
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        if time.perf_counter() - start > WAKE_TIMEOUT_S:
            raise gr.Error("The Granite Switch endpoint didn't wake up in time. Try again shortly.")
        time.sleep(10)


def granite_noul(state: str, question: str) -> dict:
    """Answer one question and fold the adapter's certainty into a noul."""
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
    return {"answer": answer, "certainty": certainty, "noul": noul}


def granite_nouls(state: str, questions: list[str]) -> tuple[list[dict], float]:
    """Run every question concurrently; return results and wall-clock seconds.

    Mellea's sync calls share one background event loop, so these threads put
    overlapping requests on the wire and vLLM batches them.
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        results = list(pool.map(lambda q: granite_noul(state, q), questions))
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

    # Jev runs while the Granite endpoint wakes (if needed) and answers.
    with ThreadPoolExecutor(max_workers=1) as pool:
        jev_future = pool.submit(jev_nouls, jev_state, questions)
        wake_s = wait_for_endpoint()
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
    wake_note = f" · endpoint was asleep, woke in {wake_s:.0f} s (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch + Mellea:** {granite_s * 1000:.0f} ms end to end "
        f"for {len(questions)} question(s), batched · `{MODEL_ID}` on vLLM · "
        f"{2 * len(questions)} requests (answer + uncertainty adapter each){wake_note}"
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
        "nouls always fall between 0.05 and 0.95. Both times are full round trips. "
        "Granite makes two dependent requests per question, sent concurrently across "
        "questions; Jev answers every question in one call. The Granite endpoint "
        "scales to zero, so the first run after idle waits for it to wake."
    )
    run.click(compare, inputs=[state, questions], outputs=[table, timing])

if __name__ == "__main__":
    demo.launch()
