# SPDX-License-Identifier: Apache-2.0
"""Granite Switch + Mellea vs. Jev: side-by-side nouls.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of state with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes.

This Space mimics that primitive with Granite Switch served by vLLM on a
Hugging Face Inference Endpoint, without ever asking Granite for its own
answer:

1. Prefill the assistant turn with "Yes." and run Granite Switch's embedded
   ``uncertainty`` adapter through Mellea.
2. Its certainty that "Yes." is correct, c(yes), is P(yes): the noul.

The same state and questions go to the real Jev model via OpenRouter (nouls
only), and the two sets of numbers are shown side by side.

Batching on the endpoint
------------------------
The prompt puts the shared state first, then the instruction, then the
question, so every question shares one long prefix. The uncertainty adapter
is an aLoRA: it only activates at its invocation token, so it reads the base
model's KV cache for everything before that. vLLM's prefix cache can then
compute the state once and reuse it for every question:

1. **Prime:** send the first question alone. vLLM computes and caches the
   shared prefix. (Requests scheduled in the same step can't share blocks
   that are still being computed, so priming beats sending everything at once.)
2. **Fan out:** send the remaining questions concurrently. vLLM batches them,
   and each prefills only its own question plus the "Yes." turn.

vLLM's ``cached_tokens`` usage is reported so the cache hits are visible.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import gradio as gr
import httpx

from mellea import ModelOption
from mellea.backends.adapters import AdapterType
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic
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
UNCERTAINTY = backend.resolve_adapter("uncertainty")


def _question_prompt(state: str, question: str) -> str:
    # Shared state + instruction first so every question reuses one cached prefix.
    return f"{state}\n\nAnswer the following question with 'yes' or 'no'.\n{question}"


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


@dataclass
class Certainty:
    noul: float
    prompt_tokens: int
    cached_tokens: int


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


def granite_noul(state: str, question: str) -> Certainty:
    """c(yes) for one question: the uncertainty adapter on a prefilled "Yes."

    This is the same call ``core.check_certainty`` makes, done through
    ``mfuncs.act`` directly so the model output's token usage (including
    vLLM's ``cached_tokens``) isn't thrown away.
    """
    ctx = (
        ChatContext()
        .add(Message("user", _question_prompt(state, question)))
        .add(Message("assistant", "Yes."))
    )
    out, _ = mfuncs.act(
        Intrinsic("uncertainty", adapter_types=(AdapterType.ALORA, AdapterType.LORA)),
        ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        tool_calls=True,
        strategy=None,
    )
    usage = out.generation.usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return Certainty(
        noul=float(UNCERTAINTY.io_contract.parse(out.value)["certainty"]),
        prompt_tokens=usage.get("prompt_tokens", 0),
        cached_tokens=details.get("cached_tokens") or 0,
    )


def granite_nouls(state: str, questions: list[str]) -> tuple[list[Certainty], dict]:
    """Prime the shared prefix with one question, then fan out the rest.

    Mellea's sync calls share one background event loop, so the fan-out
    threads put their requests on the wire together and vLLM batches them.
    """
    t0 = time.perf_counter()
    first = granite_noul(state, questions[0])
    t1 = time.perf_counter()
    rest: list[Certainty] = []
    if len(questions) > 1:
        with ThreadPoolExecutor(max_workers=len(questions) - 1) as pool:
            rest = list(pool.map(lambda q: granite_noul(state, q), questions[1:]))
    t2 = time.perf_counter()
    return [first, *rest], {"prime_s": t1 - t0, "fanout_s": t2 - t1, "total_s": t2 - t0}


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
        granite, g_time = granite_nouls(state_text, questions)
        jev, jev_s, jev_status = jev_future.result()

    rows = []
    for i, (question, g) in enumerate(zip(questions, granite)):
        j = jev[i] if jev is not None else None
        rows.append([question, None if j is None else round(j, 3), round(g.noul, 3)])

    jev_line = (
        f"**Jev:** {jev_s * 1000:.0f} ms end to end · {jev_status}"
        if jev is not None
        else f"**Jev:** unavailable · {jev_status}"
    )
    fanout = granite[1:]
    fanout_prompt = sum(g.prompt_tokens for g in fanout)
    fanout_cached = sum(g.cached_tokens for g in fanout)
    cache_line = (
        f"fan-out reused {fanout_cached} of {fanout_prompt} prompt tokens from cache "
        f"({fanout_cached / fanout_prompt:.0%})"
        if fanout_prompt
        else "single question, nothing to share"
    )
    wake_note = f" · endpoint was asleep, woke in {wake_s:.0f} s (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch + Mellea:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {len(questions)} question(s) · `{MODEL_ID}` on vLLM{wake_note}  \n"
        f"Batching: prime 1 question {g_time['prime_s'] * 1000:.0f} ms "
        f"({granite[0].prompt_tokens} prompt tokens), then {len(fanout)} in parallel "
        f"{g_time['fanout_s'] * 1000:.0f} ms · {cache_line}"
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
        "**How the Granite side works:** Granite never answers the question itself. "
        "Instead, Mellea prefills the answer as \"Yes.\" and runs Granite Switch's "
        "embedded `uncertainty` adapter on it. The adapter's certainty that yes is "
        "correct, c(yes), is the noul."
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
                ],
                datatype=["str", "number", "number"],
                interactive=False,
                wrap=True,
            )
            timing = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[state, questions])
    gr.Markdown(
        "Note: the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), and "
        "Mellea returns the probability-weighted average of those bins, so the "
        "Granite noul always falls between 0.05 and 0.95. Both times are full round "
        "trips. The Granite endpoint scales to zero, so the first run after idle "
        "waits for it to wake; that wait isn't counted."
    )
    run.click(compare, inputs=[state, questions], outputs=[table, timing])

if __name__ == "__main__":
    demo.launch()
