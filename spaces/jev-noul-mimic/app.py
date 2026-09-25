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
        yield _slow_rows(questions, slots), f"{jev_line}  \n**Granite Switch (thinking slow):** writing…"
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
        f"**Granite Switch (thinking slow):** {total_s:.1f} s for {len(questions)} "
        f"answer(s), {generated} tokens generated · first token {first} · `{MODEL_ID}` "
        f"base model, same endpoint{wake_note}  \n"
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
                    f"{jev_line}  \n**Granite Switch (compound thinking, Mellea):** "
                    f"{len(questions) - len(pending)} of {len(questions)} done…"
                )
                time.sleep(0.25)
    total_s = time.perf_counter() - t0
    warmer.touch()

    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch (compound thinking, Mellea):** {total_s:.1f} s for "
        f"{len(questions)} question(s) · `{MODEL_ID}`, same endpoint{wake_note}  \n"
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
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"{jev_line}  \n"
        f"**Granite Switch:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {len(questions)} question(s) · `{MODEL_ID}` on vLLM{wake_note}  \n"
        f"Batching: prime 1 question {g_time['prime_s'] * 1000:.0f} ms "
        f"({granite[0].prompt_tokens} prompt tokens), then {len(fanout)} in parallel "
        f"{g_time['fanout_s'] * 1000:.0f} ms · {cache_line}"
    )
    return rows, timing, warmer.status()


FAST_QUESTIONS_LABEL = f"Yes/no questions (one per line, up to {MAX_QUESTIONS})"
SLOW_QUESTIONS_LABEL = f"Questions, yes/no or free-form (one per line, up to {MAX_QUESTIONS})"


def think(mode: str, state_text: str, questions_text: str):
    """Run the selected mode.

    Fast: nouls vs Jev. Slow: Granite writes answers. Compound: Granite writes
    answers and scores each one's certainty, returned as JSON.
    """
    if mode == "fast":
        rows, timing_md, status = compare(state_text, questions_text)
        yield rows, gr.skip(), gr.skip(), timing_md, status
        return
    if not state_text.strip():
        raise gr.Error("Enter some state for the model to think about.")
    questions = _parse_questions(questions_text)
    if mode == "slow":
        for rows, timing_md in think_slow(state_text, questions):
            yield gr.skip(), rows, gr.skip(), timing_md, warmer.status()
    else:
        for result_json, timing_md in think_compound(state_text, questions):
            yield gr.skip(), gr.skip(), result_json, timing_md, warmer.status()


BUTTON_LABELS = {"fast": "Compare", "slow": "Think slow", "compound": "Think compound"}


def set_mode(mode: str):
    fast = mode == "fast"
    return (
        gr.update(
            label=FAST_QUESTIONS_LABEL if fast else SLOW_QUESTIONS_LABEL,
            placeholder="e.g. Is the customer asking for a refund?"
            if fast
            else "e.g. Is the customer asking for a refund? Or: What does the customer want?",
        ),
        gr.update(value=BUTTON_LABELS[mode]),
        gr.update(visible=mode == "fast"),  # nouls table
        gr.update(visible=mode == "slow"),  # answers table
        gr.update(visible=mode == "compound"),  # compound JSON
        "",  # clear the previous run's timing
    )


def on_page_load() -> str:
    """Start waking the endpoint as soon as someone opens the page."""
    warmer.ensure()
    return warmer.status()


# Style the Thinking radio as a segmented toggle: one pill, equal segments,
# the selected one in the theme's primary color. The native radio circles are
# hidden visually but stay in the DOM, so keyboard and screen readers still work.
CSS = """
#thinking-toggle .wrap {
  display: flex; flex-wrap: nowrap; gap: 0; padding: 4px;
  border: 1px solid var(--border-color-primary); border-radius: 999px;
  background: var(--background-fill-secondary);
}
#thinking-toggle label {
  flex: 1; justify-content: center; border: none; box-shadow: none;
  border-radius: 999px; background: transparent; padding: 8px 16px;
  font-weight: 600; transform: none;
}
#thinking-toggle label:hover { background: var(--background-fill-primary); }
#thinking-toggle label.selected {
  background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color);
}
#thinking-toggle input[type=radio] {
  position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0;
}
#thinking-toggle label > span { margin-left: 0; }
#thinking-toggle label:has(input:focus-visible) {
  outline: 2px solid var(--color-accent); outline-offset: 2px;
}
"""


with gr.Blocks(title="Thinking Fast and Slow with Granite") as demo:
    gr.Markdown(
        "# 🧠 Thinking Fast and Slow with Granite\n"
        "Some decisions don't need reasoning out loud. They need a fast, calibrated "
        "gut call. That's the idea behind *System One* models like TypeSafe AI's "
        "[Jev](https://docs.typesafe.ai): instead of text, Jev answers a yes/no "
        "question with a **noul**, the probability that the answer is yes.\n\n"
        "Here, an open 3B model, "
        "[Granite Switch](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview), "
        "makes the same kind of call in one generated token. Give both models the "
        "same input and questions and compare.\n\n"
        "**What makes it possible:** the **uncertainty (UQ) adapter**. Granite "
        "Switch has a built-in, calibrated uncertainty-quantification adapter that "
        "scores how likely an answer is to be correct. Prefill the answer \"Yes.\", "
        "ask the adapter, and its certainty *is* the noul.\n\n"
        "**What makes it fast**\n"
        "- **aLoRA KV-cache reuse.** The UQ adapter is an activated LoRA: it switches "
        "on only at its trigger token and reuses the base model's KV cache for "
        "everything before it. Granite reads your input once, and every question "
        "reuses that work.\n"
        "- **Optimized vLLM kernels.** Granite Switch runs on vLLM with kernels the "
        "Granite team optimized for switching between embedded adapters.\n"
        "- **One token per question.** Only the adapter's score digit is generated; "
        "its probabilities give the certainty.\n\n"
        "**And it can think slow, too.** The same endpoint, with the same weights, "
        "also serves Granite as a regular LLM that explains, drafts and works "
        "through problems in text. Fast gut calls and slow reasoning from one "
        "deployment. Jev returns decisions only; it doesn't generate text. Switch "
        "to **🐢 Thinking slow** below to try it.\n\n"
        "**Or both at once.** **🧠 Compound** thinking writes the answer, then has "
        "the UQ adapter score its certainty in that answer, returned together as "
        "JSON. This mode is built with [Mellea](https://mellea.ai)."
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                choices=[
                    ("⚡ Thinking fast", "fast"),
                    ("🐢 Thinking slow", "slow"),
                    ("🧠 Compound", "compound"),
                ],
                value="fast",
                label="Thinking",
                info=(
                    "Fast: yes/no questions, nouls vs Jev. Slow: Granite writes answers. "
                    "Compound: Granite writes answers and scores its certainty in each, as JSON."
                ),
                elem_id="thinking-toggle",
            )
            state = gr.Textbox(
                label="State (text or JSON)",
                lines=6,
                max_lines=18,
                placeholder="The input the models think about.",
            )
            questions = gr.Textbox(
                label=FAST_QUESTIONS_LABEL,
                lines=5,
                placeholder="e.g. Is the customer asking for a refund?",
            )
            run = gr.Button("Compare", variant="primary")
            endpoint_status = gr.Markdown(warmer.status())
        with gr.Column():
            fast_table = gr.Dataframe(
                headers=["Question", "Jev noul", "Granite noul"],
                datatype=["str", "number", "number"],
                interactive=False,
                wrap=True,
            )
            slow_table = gr.Dataframe(
                headers=["Question", "Jev", "Granite answer"],
                datatype=["str", "str", "str"],
                column_widths=["22%", "8%", "70%"],
                interactive=False,
                wrap=True,
                visible=False,
            )
            compound_json = gr.JSON(label="Compound thinking (JSON)", visible=False)
            timing = gr.Markdown()
    gr.Examples(EXAMPLE_INPUTS, inputs=[state, questions], example_labels=EXAMPLE_LABELS)
    gr.Markdown(
        "Note: the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), and "
        "the noul is the probability-weighted average of those bins, so the "
        "Granite noul always falls between 0.05 and 0.95. Both times are full round "
        "trips. The Granite endpoint scales to zero after 15 idle minutes; opening "
        "this page starts waking it, and any wait isn't counted in Granite's time."
    )
    mode.change(
        set_mode,
        inputs=mode,
        outputs=[questions, run, fast_table, slow_table, compound_json, timing],
        show_progress="hidden",
    )
    run.click(
        think,
        inputs=[mode, state, questions],
        outputs=[fast_table, slow_table, compound_json, timing, endpoint_status],
    )
    # Wake on page load, and keep the status line current while it wakes.
    demo.load(on_page_load, outputs=endpoint_status, show_progress="hidden")
    gr.Timer(3).tick(warmer.status, outputs=endpoint_status, show_progress="hidden", queue=False)

if __name__ == "__main__":
    demo.launch(css=CSS)
