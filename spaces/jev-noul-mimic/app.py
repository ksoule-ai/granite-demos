# SPDX-License-Identifier: Apache-2.0
"""Thinking Fast and Slow with Granite: Granite Switch vs. Jev, side-by-side nouls and free-form answers.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of state with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes.

This Space mimics that primitive with Granite Switch served by vLLM on a
Hugging Face Inference Endpoint, without ever asking Granite for its own
answer:

1. Prefill the assistant turn with "Yes." and run Granite Switch's embedded
   ``uncertainty`` adapter on it.
2. Its certainty that "Yes." is correct, c(yes), is P(yes): the noul.

The adapter call is the conversation Mellea's ``check_certainty`` sends, trimmed
so vLLM generates one token (the score digit) instead of the full JSON reply;
see ``granite_noul``.

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
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import gradio as gr
import httpx
import yaml
from huggingface_hub import hf_hub_download
from openai import OpenAI
from pydantic import BaseModel
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

from mellea import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

from examples import EXAMPLE_INPUTS, EXAMPLE_LABELS

# Granite Switch endpoint (vLLM, OpenAI-compatible). HF_ENDPOINT_URL ends in /v1.
ENDPOINT_URL = os.environ["HF_ENDPOINT_URL"].rstrip("/")
HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_ID = os.environ.get("MODEL_ID", "ibm-granite/granite-switch-4.1-3b-preview")
# The endpoint's GPU, shown under results. Update if the endpoint's hardware changes.
GRANITE_HARDWARE = os.environ.get("GRANITE_HARDWARE", "a single NVIDIA L4 GPU (24 GB)")
# The endpoint scales to zero after 15 idle minutes; the first request after
# that wakes it (a cold start takes ~3-5 minutes).
WAKE_TIMEOUT_S = 420
# Treat the endpoint as warm for this long after its last use, comfortably
# inside the 15-minute scale-to-zero window.
READY_TTL_S = 10 * 60

# Jev is reached through OpenRouter's System One API, which the TypeSafe SDK
# speaks natively; a bare model id like "jev-1.13" routes to typesafe/jev-1.13.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13")
MAX_QUESTIONS = 8

client = OpenAI(base_url=ENDPOINT_URL, api_key=HF_TOKEN)

# The uncertainty adapter's I/O config ships in the model repo (it's the same
# file Mellea's check_certainty reads). It defines the invocation text, the
# JSON field the adapter writes, and how each score digit maps to a certainty.
_io = yaml.safe_load(open(hf_hub_download(MODEL_ID, "io_configs/uncertainty/io.yaml")))
_likelihood = next(t for t in _io["transformations"] if t["type"] == "likelihood")
UNCERTAINTY_INVOCATION = _io["instruction"]  # "<certainty>"
SCORE_VALUES = {str(k): float(v) for k, v in _likelihood["categories_to_values"].items()}
# The adapter always answers {"score": "<digit>"}. Prefilling everything up to
# the digit means the model generates exactly one token.
SCORE_PREFIX = '{"' + _likelihood["input_path"][0] + '": "'
TOP_LOGPROBS = 10  # same as Mellea's likelihood decoding


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
        raise gr.Error("Enter at least one question (one per line).")
    if len(questions) > MAX_QUESTIONS:
        raise gr.Error(f"At most {MAX_QUESTIONS} questions per run.")
    return questions


# --------------------------------------------------------------------------- #
# Granite Switch
# --------------------------------------------------------------------------- #


@dataclass
class Certainty:
    noul: float
    prompt_tokens: int
    cached_tokens: int


class EndpointWarmer:
    """Wakes and warms the scale-to-zero endpoint once, shared by every visitor.

    States: idle -> checking -> (waking) -> warming -> ready, or error.
    "Waking" means the endpoint answered 503 and is cold-starting. "Warming"
    sends one throwaway uncertainty-adapter call so the first timed request
    doesn't pay for the new connection and first adapter call. "Ready" goes
    stale READY_TTL_S after the last use, ahead of HF's 15-minute scale-down,
    so the next visitor re-checks instead of trusting an endpoint that slept.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = "idle"
        self.detail = ""
        self.started_at = 0.0
        self.last_used = 0.0

    def is_ready(self) -> bool:
        return self.state == "ready" and time.time() - self.last_used < READY_TTL_S

    def touch(self) -> None:
        self.last_used = time.time()

    def ensure(self) -> None:
        """Start a wake/warm pass unless one is running or the endpoint is fresh."""
        with self._lock:
            if self.is_ready() or (self._thread is not None and self._thread.is_alive()):
                return
            self.state, self.detail, self.started_at = "checking", "", time.time()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        while True:
            try:
                if httpx.get(f"{ENDPOINT_URL}/models", headers=headers, timeout=10).is_success:
                    break
            except httpx.HTTPError:
                pass
            # Any request to a scaled-to-zero endpoint starts it; keep polling.
            self.state = "waking"
            if time.time() - self.started_at > WAKE_TIMEOUT_S:
                self.state, self.detail = "error", "didn't wake up within 7 minutes"
                return
            time.sleep(10)
        self.state = "warming"
        try:
            granite_noul("This is a warm-up request.", "Is this a warm-up request?")
        except Exception as e:  # surface any failure in the status line
            self.state, self.detail = "error", f"warm-up call failed: {e}"
            return
        self.detail = f"{time.time() - self.started_at:.0f} s"
        self.touch()
        self.state = "ready"

    def wait_ready(self) -> float:
        """Block until ready (starting a pass if needed); return seconds waited."""
        start = time.time()
        self.ensure()
        while not self.is_ready():
            if self.state == "error":
                raise gr.Error(f"Granite Switch endpoint unavailable: {self.detail}. Try again shortly.")
            if time.time() - start > WAKE_TIMEOUT_S + 60:
                raise gr.Error("The Granite Switch endpoint didn't wake up in time. Try again shortly.")
            time.sleep(1)
        return time.time() - start

    def status(self) -> str:
        elapsed = time.time() - self.started_at
        if self.is_ready():
            return "**Granite endpoint:** ● ready"
        return {
            "idle": "**Granite endpoint:** not checked yet",
            "checking": "**Granite endpoint:** checking…",
            "waking": f"**Granite endpoint:** ○ asleep, waking up ({elapsed:.0f} s so far; "
            "a cold start takes about 3–5 min)",
            "warming": "**Granite endpoint:** ◐ up, sending a warm-up request…",
            "error": f"**Granite endpoint:** ✕ {self.detail}",
            # "ready" but stale: it may have scaled to zero since.
            "ready": "**Granite endpoint:** idle for a while; will re-check on next use",
        }[self.state]


warmer = EndpointWarmer()


def granite_noul(state: str, question: str) -> Certainty:
    """c(yes) for one question: the uncertainty adapter on a prefilled "Yes."

    Same conversation Mellea's ``check_certainty`` sends (question, "Yes.",
    then the ``<certainty>`` invocation), with one change: the adapter's reply
    is prefilled up to the score digit and continued, so vLLM generates a
    single token instead of the full ``{"score": "N"}`` (7 tokens). The
    certainty is then decoded from that token's top logprobs the way Mellea's
    likelihood rule does it: keep the digit candidates, renormalize, and take
    the expected value of their mapped certainties.
    """
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "user", "content": _question_prompt(state, question)},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": UNCERTAINTY_INVOCATION},
            {"role": "assistant", "content": SCORE_PREFIX},
        ],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        extra_body={
            "chat_template_kwargs": {"adapter_name": "uncertainty"},
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    )
    top = response.choices[0].logprobs.content[0]
    candidates = [(top.token, top.logprob)] + [
        (t.token, t.logprob) for t in top.top_logprobs if t.token != top.token
    ]
    weighted = [
        (SCORE_VALUES[tok.strip()], math.exp(lp))
        for tok, lp in candidates
        if tok.strip() in SCORE_VALUES
    ]
    if not weighted:
        raise ValueError(f"Uncertainty adapter returned no score digit (got {top.token!r}).")
    total = sum(p for _, p in weighted)
    usage = response.usage
    details = usage.prompt_tokens_details if usage else None
    return Certainty(
        noul=sum(v * p for v, p in weighted) / total,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        cached_tokens=(details.cached_tokens or 0) if details else 0,
    )


def granite_nouls(state: str, questions: list[str]) -> tuple[list[Certainty], dict]:
    """Prime the shared prefix with one question, then fan out the rest.

    The fan-out threads share one OpenAI client (one HTTP connection pool), so
    their requests reach vLLM together and it batches them.
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
# Granite Switch, thinking slow: free-form answers from the base model
# --------------------------------------------------------------------------- #

SLOW_MAX_TOKENS = 512


def _slow_prompt(state: str, question: str) -> str:
    # Just the state, then the question. The state comes first, as in the fast
    # prompt, so both modes share the cached state prefix on the endpoint.
    return f"{state}\n\n{question}"


@dataclass
class SlowAnswer:
    text: str = ""
    done: bool = False
    error: str = ""
    first_token_s: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0


def granite_answer(state: str, question: str, slot: SlowAnswer, t0: float) -> None:
    """Stream the base model's answer (no adapter) into `slot`."""
    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": _slow_prompt(state, question)}],
            max_tokens=SLOW_MAX_TOKENS,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if slot.first_token_s is None:
                    slot.first_token_s = time.perf_counter() - t0
                slot.text += chunk.choices[0].delta.content
            if chunk.usage:
                slot.prompt_tokens = chunk.usage.prompt_tokens
                slot.completion_tokens = chunk.usage.completion_tokens
                details = chunk.usage.prompt_tokens_details
                slot.cached_tokens = (details.cached_tokens or 0) if details else 0
    except Exception as e:  # show the failure in that answer's cell
        slot.error = str(e)
    finally:
        slot.done = True


def _slow_rows(questions: list[str], slots: list[SlowAnswer]) -> list[list[str]]:
    rows = []
    for question, slot in zip(questions, slots):
        if slot.error:
            answer = f"(error: {slot.error})"
        else:
            answer = slot.text.strip() or ("…" if not slot.done else "(no answer)")
        rows.append([question, "N/A", answer])
    return rows


def think_slow(state_text: str, questions: list[str]):
    """Stream free-form answers: prime question 1, then fan out the rest.

    The first question runs alone until its first token arrives, which means
    the shared state prefix has been computed and cached. The others then go
    out together, so vLLM batches them on that cached prefix.
    """
    wake_s = warmer.wait_ready()
    slots = [SlowAnswer() for _ in questions]
    t0 = time.perf_counter()

    def start(i: int) -> None:
        threading.Thread(
            target=granite_answer, args=(state_text, questions[i], slots[i], t0), daemon=True
        ).start()

    start(0)
    while slots[0].first_token_s is None and not slots[0].done:
        time.sleep(0.02)
    for i in range(1, len(questions)):
        start(i)

    jev_line = "**Jev:** N/A. Jev returns decisions only; it doesn't generate text."
    while not all(s.done for s in slots):
        yield _slow_rows(questions, slots), f"{jev_line}  \n**Granite Switch (Thinking Slow):** writing…"
        time.sleep(0.25)
    total_s = time.perf_counter() - t0
    warmer.touch()

    generated = sum(s.completion_tokens for s in slots)
    rest = slots[1:]
    rest_prompt = sum(s.prompt_tokens for s in rest)
    rest_cached = sum(s.cached_tokens for s in rest)
    cache_line = (
        f" · answers 2–{len(slots)} reused {rest_cached} of {rest_prompt} prompt tokens "
        f"from cache ({rest_cached / rest_prompt:.0%})"
        if rest_prompt
        else ""
    )
    first = f"{slots[0].first_token_s * 1000:.0f} ms" if slots[0].first_token_s else "n/a"
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch (Thinking Slow):** {total_s:.1f} s for {len(questions)} "
        f"answer(s), {generated} tokens generated · first token {first} · `{MODEL_ID}` "
        f"base model, same endpoint, on {GRANITE_HARDWARE}{wake_note}  \n"
        f"Batching: question 1 first, the rest in parallel once its prefix was cached{cache_line}"
    )
    yield _slow_rows(questions, slots), timing


# --------------------------------------------------------------------------- #
# Granite Switch, compound thinking (Mellea): think slow, then score certainty
# --------------------------------------------------------------------------- #

# Mellea drives the same endpoint. load_embedded_adapters registers Granite
# Switch's embedded adapters (only their I/O configs are fetched; the weights
# are already in the served model), so core.check_certainty can call the
# uncertainty adapter by name.
mellea_backend = OpenAIBackend(
    model_id=MODEL_ID,
    formatter=TemplateFormatter(model_id=MODEL_ID),
    base_url=ENDPOINT_URL,
    api_key=HF_TOKEN,
    load_embedded_adapters=True,
)
COMPOUND_OPTIONS = {ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: SLOW_MAX_TOKENS}


class CompoundAnswer(BaseModel):
    """One compound-thinking result: the written answer plus its certainty."""

    question: str
    answer: str
    certainty: float


def compound_answer(state: str, question: str) -> tuple[CompoundAnswer, float, float]:
    """Think slow, then score the answer. Returns (result, answer_s, certainty_s).

    mfuncs.chat returns the context with both the question and the model's
    answer in it, which is exactly what check_certainty scores. The adapter is
    an aLoRA, so this second call reuses the KV cache for the question *and*
    the answer and only has to generate the score.
    """
    t0 = time.perf_counter()
    reply, ctx = mfuncs.chat(
        _slow_prompt(state, question), ChatContext(), mellea_backend, model_options=COMPOUND_OPTIONS
    )
    t1 = time.perf_counter()
    certainty = core.check_certainty(ctx, mellea_backend)
    t2 = time.perf_counter()
    result = CompoundAnswer(question=question, answer=reply.content.strip(), certainty=round(certainty, 3))
    return result, t1 - t0, t2 - t1


def _compound_json(questions: list[str], results: list[CompoundAnswer | None]) -> dict:
    return {
        "mode": "compound",
        "model": MODEL_ID,
        "results": [
            r.model_dump() if r else {"question": q, "answer": None, "certainty": None}
            for q, r in zip(questions, results)
        ],
    }


def think_compound(state_text: str, questions: list[str]):
    """Prefill the shared state once, then run every question in parallel.

    Mellea's chat call isn't streamed here, so instead of waiting for question
    1's first token (as slow mode does), a 1-token Mellea call caches the
    state prefix. All questions then go out together and vLLM batches them.
    Results fill into the JSON as each question finishes.
    """
    wake_s = warmer.wait_ready()
    t0 = time.perf_counter()
    mfuncs.chat(
        _slow_prompt(state_text, questions[0]),
        ChatContext(),
        mellea_backend,
        model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 1},
    )
    prefill_s = time.perf_counter() - t0

    results: list[CompoundAnswer | None] = [None] * len(questions)
    answer_s, certainty_s = [], []
    jev_line = "**Jev:** N/A. Jev returns decisions only; it doesn't generate text."
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {pool.submit(compound_answer, state_text, q): i for i, q in enumerate(questions)}
        pending = set(futures)
        while pending:
            done = {f for f in pending if f.done()}
            for f in done:
                result, a_s, c_s = f.result()
                results[futures[f]] = result
                answer_s.append(a_s)
                certainty_s.append(c_s)
            pending -= done
            if pending:
                yield _compound_json(questions, results), (
                    f"{jev_line}  \n**Granite Switch (Compound Thinking, Mellea):** "
                    f"{len(questions) - len(pending)} of {len(questions)} done…"
                )
                time.sleep(0.25)
    total_s = time.perf_counter() - t0
    warmer.touch()

    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch (Compound Thinking, Mellea):** {total_s:.1f} s for "
        f"{len(questions)} question(s) · `{MODEL_ID}`, same endpoint, on {GRANITE_HARDWARE}{wake_note}  \n"
        f"Prefill of the shared state {prefill_s * 1000:.0f} ms, then all questions in "
        f"parallel · per question: answer {max(answer_s):.1f} s max, certainty "
        f"{sum(certainty_s) / len(certainty_s) * 1000:.0f} ms avg (aLoRA on the cached answer)"
    )
    yield _compound_json(questions, results), timing


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
        wake_s = warmer.wait_ready()
        granite, g_time = granite_nouls(state_text, questions)
        warmer.touch()
        jev, jev_s, jev_status = jev_future.result()

    granite_rows = [[q, round(g.noul, 3)] for q, g in zip(questions, granite)]
    jev_rows = [[q, None if jev is None else round(jev[i], 3)] for i, q in enumerate(questions)]

    jev_timing = (
        f"{jev_s * 1000:.0f} ms end to end · {jev_status}"
        if jev is not None
        else f"Unavailable · {jev_status}"
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
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"**Granite Switch:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {len(questions)} question(s) · `{MODEL_ID}` on vLLM, {GRANITE_HARDWARE}{wake_note}  \n"
        f"Batching: prime 1 question {g_time['prime_s'] * 1000:.0f} ms "
        f"({granite[0].prompt_tokens} prompt tokens), then {len(fanout)} in parallel "
        f"{g_time['fanout_s'] * 1000:.0f} ms · {cache_line}"
    )
    return granite_rows, jev_rows, timing, jev_timing, warmer.status()


QUESTIONS_LABEL = f"Questions (one per line, up to {MAX_QUESTIONS})"
QUESTIONS_INFO = (
    "Thinking Fast needs yes/no questions. Thinking Slow and Compound Thinking "
    "take any question."
)


OUTPUT_HEADERS = {
    "fast": "### ⚡ Thinking Fast\nGranite Switch nouls: the uncertainty adapter's P(yes) "
    "for each question, one generated token each.",
    "slow": "### 🐢 Thinking Slow\nGranite Switch's base model writes an answer to each question.",
    "compound": "### 🧠 Compound Thinking\nGranite Switch writes each answer, then scores its "
    "certainty in it. Built with Mellea.",
}
IDLE_HEADER = "### Results\nPick a kind of thinking to start."
JEV_REFERENCE_HEADER = (
    "#### Reference: Jev\nTypeSafe AI's *System One* model, called via OpenRouter. "
    "Shown for comparison only; it isn't part of the Granite stack."
)


def think(mode: str, state_text: str, questions_text: str):
    """Run one mode, started by its button.

    The first update sets the header to the chosen kind of thinking, shows that
    mode's results (hiding the others) and clears the last run's timing, so the
    page switches the moment the button is clicked. Outputs, in order: header,
    Granite nouls, Jev reference box, Jev table, Jev timing, slow answers,
    compound JSON, Granite timing, endpoint status.
    """
    fast = mode == "fast"
    yield (
        OUTPUT_HEADERS[mode],
        gr.update(visible=fast, value=None),
        gr.update(visible=fast),
        gr.update(value=None),
        "",
        gr.update(visible=mode == "slow"),
        gr.update(visible=mode == "compound"),
        "",
        gr.skip(),
    )
    skip5 = (gr.skip(),) * 5
    if fast:
        granite_rows, jev_rows, timing_md, jev_timing, status = compare(state_text, questions_text)
        yield gr.skip(), granite_rows, gr.skip(), jev_rows, jev_timing, gr.skip(), gr.skip(), timing_md, status
        return
    if not state_text.strip():
        raise gr.Error("Enter some state for the model to think about.")
    questions = _parse_questions(questions_text)
    if mode == "slow":
        for rows, timing_md in think_slow(state_text, questions):
            yield *skip5, rows, gr.skip(), timing_md, warmer.status()
    else:
        for result_json, timing_md in think_compound(state_text, questions):
            yield *skip5, gr.skip(), result_json, timing_md, warmer.status()


def think_fast(state_text: str, questions_text: str):
    yield from think("fast", state_text, questions_text)


def think_slow_mode(state_text: str, questions_text: str):
    yield from think("slow", state_text, questions_text)


def think_compound_mode(state_text: str, questions_text: str):
    yield from think("compound", state_text, questions_text)


def on_page_load() -> str:
    """Start waking the endpoint as soon as someone opens the page."""
    warmer.ensure()
    return warmer.status()


# The Jev reference box: set apart from Granite's results with a dashed border
# and muted background. Uses our own elem_classes, not Gradio internals.
CSS = """
.jev-reference {
  border: 1px dashed var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
  padding: var(--spacing-lg);
  margin-top: var(--spacing-lg);
  opacity: 0.9;
}
"""


with gr.Blocks(title="Thinking Fast and Slow with Granite") as demo:
    gr.Markdown(
        "# 🧠 Thinking Fast and Slow with Granite\n"
        "Some decisions don't need reasoning out loud. They need a fast, calibrated "
        "gut call. That's the idea behind *System One* models like TypeSafe AI's "
        "[Jev](https://docs.typesafe.ai): instead of text, Jev answers a yes/no "
        "question with a **noul**, the probability that the answer is yes.\n\n"
        "One open 3B model, Granite Switch, does both from a single endpoint:\n"
        "- **⚡ Thinking Fast:** a yes/no call as a noul, one generated token per "
        "question, side by side with Jev.\n"
        "- **🐢 Thinking Slow:** a written answer, like any LLM. Jev returns "
        "decisions only; it doesn't generate text.\n"
        "- **🧠 Compound Thinking:** a written answer plus Granite's certainty in "
        "it, returned as JSON, built with [Mellea](https://mellea.ai).\n\n"
        "### Technologies inside\n"
        "- **Granite Switch.** One checkpoint that bundles IBM's Granite 4.1 base "
        "model with 12 embedded adapter functions (RAG, safety, uncertainty and "
        "more), each selected per request by name. "
        "[Model card](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) · "
        "[GitHub](https://github.com/generative-computing/granite-switch) · "
        "[Adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html)\n"
        "- **Uncertainty quantification (UQ) adapter.** A calibrated adapter that "
        "scores how likely an answer is to be correct: of the answers it scores at "
        "X%, about X% are right. Score a prefilled \"Yes.\" and you get a noul; "
        "score Granite's own answer and you get Compound Thinking. "
        "[Adapter README](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/uncertainty/README.md)\n"
        "- **aLoRA (activated LoRA).** Adapters that switch on at a trigger token "
        "and reuse the base model's KV cache for everything before it, so Granite "
        "reads your input once and every question, fast or slow, reuses that work. "
        "[Paper (NeurIPS 2025)](https://arxiv.org/abs/2504.12397) · "
        "[Code](https://github.com/IBM/activated-lora)\n"
        "- **Optimized vLLM kernels.** Granite Switch's vLLM integration, with "
        "kernels optimized by the Granite team, applies adapter weights per token "
        "position rather than per request, so adapter and base-model requests "
        "share batches and one KV cache. "
        "[aLoRA vs LoRA live race](https://generative-computing.github.io/granite-switch/race_live.html) · "
        "[vLLM](https://github.com/vllm-project/vllm)"
    )
    with gr.Row():
        with gr.Column():
            state = gr.Textbox(
                label="State (text or JSON)",
                lines=6,
                max_lines=18,
                placeholder="The input the models think about.",
            )
            questions = gr.Textbox(
                label=QUESTIONS_LABEL,
                info=QUESTIONS_INFO,
                lines=5,
                placeholder="e.g. Is the customer asking for a refund?",
            )
            with gr.Row():
                fast_btn = gr.Button("⚡ Thinking Fast", variant="primary")
                slow_btn = gr.Button("🐢 Thinking Slow", variant="primary")
                compound_btn = gr.Button("🧠 Compound Thinking", variant="primary")
            endpoint_status = gr.Markdown(warmer.status())
        with gr.Column():
            output_header = gr.Markdown(IDLE_HEADER)
            fast_table = gr.Dataframe(
                headers=["Question", "Granite noul"],
                datatype=["str", "number"],
                column_widths=["75%", "25%"],
                interactive=False,
                wrap=True,
                visible=False,
            )
            slow_table = gr.Dataframe(
                headers=["Question", "Jev", "Granite answer"],
                datatype=["str", "str", "str"],
                column_widths=["22%", "8%", "70%"],
                interactive=False,
                wrap=True,
                visible=False,
            )
            compound_json = gr.JSON(label="Compound Thinking (JSON)", visible=False)
            timing = gr.Markdown()
            with gr.Column(visible=False, elem_classes="jev-reference") as jev_box:
                gr.Markdown(JEV_REFERENCE_HEADER)
                jev_table = gr.Dataframe(
                    headers=["Question", "Jev noul"],
                    datatype=["str", "number"],
                    column_widths=["75%", "25%"],
                    interactive=False,
                    wrap=True,
                )
                jev_timing = gr.Markdown()
    gr.Examples(EXAMPLE_INPUTS, inputs=[state, questions], example_labels=EXAMPLE_LABELS)
    gr.Markdown(
        "Note: the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), and "
        "the noul is the probability-weighted average of those bins, so the "
        "Granite noul always falls between 0.05 and 0.95.\n\n"
        f"**Hardware:** Granite Switch is served by vLLM on {GRANITE_HARDWARE}, on a "
        "Hugging Face Inference Endpoint. All three modes run on that one GPU. Both "
        "times are full round trips. The endpoint scales to zero after 15 idle "
        "minutes; opening this page starts waking it, and any wait isn't counted in "
        "Granite's time."
    )
    # Each button starts its own kind of thinking straight away.
    run_outputs = [
        output_header, fast_table, jev_box, jev_table, jev_timing,
        slow_table, compound_json, timing, endpoint_status,
    ]
    fast_btn.click(think_fast, [state, questions], run_outputs, api_name="think_fast")
    slow_btn.click(think_slow_mode, [state, questions], run_outputs, api_name="think_slow")
    compound_btn.click(think_compound_mode, [state, questions], run_outputs, api_name="think_compound")
    # Wake on page load, and keep the status line current while it wakes.
    demo.load(on_page_load, outputs=endpoint_status, show_progress="hidden")
    gr.Timer(3).tick(warmer.status, outputs=endpoint_status, show_progress="hidden", queue=False)

if __name__ == "__main__":
    demo.launch(css=CSS)
