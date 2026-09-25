# SPDX-License-Identifier: Apache-2.0
"""Thinking Fast and Slow with Granite: Granite Switch vs. Jev, side-by-side nouls and free-form answers.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of state with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes.

This Space mimics that primitive with Granite Switch served by vLLM on a
Hugging Face Inference Endpoint, without ever asking Granite for its own
answer:

1. Ask with "Reply with exactly one word, 'Yes' or 'No'.", then run Granite
   Switch's embedded ``uncertainty`` adapter twice: on a prefilled "Yes"
   (c(yes)) and on a prefilled "No" (c(no)).
2. The noul is c(yes) / (c(yes) + c(no)). Normalizing by c(no), which tracks
   the adapter's per-input drift, puts the yes/no boundary back at 0.5.

The adapter call is the conversation Mellea's ``check_certainty`` sends, trimmed
so vLLM generates one token (the score digit) instead of the full JSON reply;
see ``_certainty``.

The same state and questions go to the real Jev model via OpenRouter (nouls
only), and the two sets of numbers are shown side by side.

Batching on the endpoint
------------------------
The prompt puts the shared state first, then the instruction, then the
question, so every question shares one long prefix. The uncertainty adapter
is an aLoRA: it only activates at its invocation token, so it reads the base
model's KV cache for everything before that. vLLM's prefix cache can then
compute the state once and reuse it for every question:

1. **Prime:** send question 1's "Yes" call alone. vLLM computes and caches the
   shared prefix. (Requests scheduled in the same step can't share blocks
   that are still being computed, so priming beats sending everything at once.)
2. **Fan out:** send the other 2N - 1 adapter calls ("Yes" and "No" for every
   question) concurrently. vLLM batches them, and each prefills only its own
   question plus the prefilled answer.

vLLM's ``cached_tokens`` usage is reported so the cache hits are visible.
"""

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import gradio as gr
import httpx
import pandas as pd
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
MAX_QUESTIONS = 50

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


FAST_INSTRUCTION = "Reply with exactly one word, 'Yes' or 'No'."


def _question_prompt(state: str, question: str) -> str:
    # Shared state + instruction first so every question reuses one cached prefix.
    return f"{state}\n\n{FAST_INSTRUCTION}\n{question}"


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
    noul: float  # c(yes) / (c(yes) + c(no))
    c_yes: float
    c_no: float
    prompt_tokens: int  # summed over both adapter calls
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


def _certainty(state: str, question: str, answer: str) -> tuple[float, int, int]:
    """One uncertainty-adapter call: its certainty that `answer` is correct.

    Same conversation Mellea's ``check_certainty`` sends (question, answer,
    then the ``<certainty>`` invocation), with one change: the adapter's reply
    is prefilled up to the score digit and continued, so vLLM generates a
    single token instead of the full ``{"score": "N"}`` (7 tokens). The
    certainty is decoded from that token's top logprobs the way Mellea's
    likelihood rule does it: keep the digit candidates, renormalize, and take
    the expected value of their mapped certainties.

    Returns (certainty, prompt_tokens, cached_tokens).
    """
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "user", "content": _question_prompt(state, question)},
            {"role": "assistant", "content": answer},
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
    return (
        sum(v * p for v, p in weighted) / total,
        usage.prompt_tokens if usage else 0,
        (details.cached_tokens or 0) if details else 0,
    )


def _combine(yes: tuple[float, int, int], no: tuple[float, int, int]) -> Certainty:
    """noul = c(yes) / (c(yes) + c(no)).

    The adapter's certainty drifts from one input to another, so c(yes) alone
    has no fixed yes/no boundary. c(no) on the same input barely depends on
    the true answer but tracks that drift, so dividing by c(yes) + c(no)
    cancels it and puts the boundary back at 0.5.
    """
    c_yes, c_no = yes[0], no[0]
    return Certainty(
        noul=c_yes / (c_yes + c_no),
        c_yes=c_yes,
        c_no=c_no,
        prompt_tokens=yes[1] + no[1],
        cached_tokens=yes[2] + no[2],
    )


def granite_noul(state: str, question: str) -> Certainty:
    """One question's noul: the adapter on a prefilled "Yes", then on "No"."""
    return _combine(_certainty(state, question, "Yes"), _certainty(state, question, "No"))


def granite_nouls(state: str, questions: list[str]) -> tuple[list[Certainty], dict]:
    """Prime the shared prefix with one call, then fan out every other call.

    Each question needs two adapter calls (prefilled "Yes" and "No"). Question
    1's "Yes" call runs alone to cache the shared state; the remaining 2N - 1
    calls then go out together from threads sharing one OpenAI client (one
    HTTP connection pool), so vLLM batches them on the cached prefix.
    """
    t0 = time.perf_counter()
    first_yes = _certainty(state, questions[0], "Yes")
    t1 = time.perf_counter()
    calls = [(questions[0], "No")] + [(q, a) for q in questions[1:] for a in ("Yes", "No")]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        results = list(pool.map(lambda qa: _certainty(state, *qa), calls))
    t2 = time.perf_counter()
    nouls = [_combine(first_yes, results[0])]
    for i in range(1, len(questions)):
        nouls.append(_combine(results[2 * i - 1], results[2 * i]))
    return nouls, {"prime_s": t1 - t0, "fanout_s": t2 - t1, "total_s": t2 - t0, "fanout_calls": len(calls)}


# --------------------------------------------------------------------------- #
# Granite Switch, thinking slow (Mellea): write an answer, then score certainty
# --------------------------------------------------------------------------- #

SLOW_MAX_TOKENS = 512


def _slow_prompt(state: str, question: str) -> str:
    # Just the state, then the question. The state comes first, as in the fast
    # prompt, so both modes share the cached state prefix on the endpoint.
    return f"{state}\n\n{question}"


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
SLOW_OPTIONS = {ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: SLOW_MAX_TOKENS}


class SlowAnswer(BaseModel):
    """One Thinking Slow result: the written answer plus its certainty."""

    question: str
    answer: str
    certainty: float


def slow_answer(state: str, question: str) -> tuple[SlowAnswer, float, float]:
    """Think slow, then score the answer. Returns (result, answer_s, certainty_s).

    mfuncs.chat returns the context with both the question and the model's
    answer in it, which is exactly what check_certainty scores. The adapter is
    an aLoRA, so this second call reuses the KV cache for the question *and*
    the answer and only has to generate the score.
    """
    t0 = time.perf_counter()
    reply, ctx = mfuncs.chat(
        _slow_prompt(state, question), ChatContext(), mellea_backend, model_options=SLOW_OPTIONS
    )
    t1 = time.perf_counter()
    certainty = core.check_certainty(ctx, mellea_backend)
    t2 = time.perf_counter()
    result = SlowAnswer(question=question, answer=reply.content.strip(), certainty=round(certainty, 3))
    return result, t1 - t0, t2 - t1


def think_slow(state_text: str, questions: list[str]):
    """Prefill the shared state once, then run every question in parallel.

    Mellea's chat call isn't streamed here, so instead of waiting for question
    1's first token (as slow mode does), a 1-token Mellea call caches the
    state prefix. All questions then go out together and vLLM batches them.
    Yields (results so far, timing markdown, total seconds or None) as each
    question finishes; the caller turns results into the table.
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

    results: list[SlowAnswer | None] = [None] * len(questions)
    answer_s, certainty_s = [], []
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {pool.submit(slow_answer, state_text, q): i for i, q in enumerate(questions)}
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
                yield list(results), (
                    f"**Granite Switch (Thinking Slow, Mellea):** "
                    f"{len(questions) - len(pending)} of {len(questions)} done…"
                ), None
                time.sleep(0.25)
    total_s = time.perf_counter() - t0
    warmer.touch()

    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"**Granite Switch (Thinking Slow, Mellea):** {total_s:.1f} s for "
        f"{len(questions)} question(s) · `{MODEL_ID}`, same endpoint, on {GRANITE_HARDWARE}{wake_note}  \n"
        f"Prefill of the shared state {prefill_s * 1000:.0f} ms, then all questions in "
        f"parallel · per question: answer {max(answer_s):.1f} s max, certainty "
        f"{sum(certainty_s) / len(certainty_s) * 1000:.0f} ms avg (aLoRA on the cached answer)"
    )
    yield list(results), timing, total_s


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

    table = pd.DataFrame(
        {
            "Question": questions,
            "Granite noul": [round(g.noul, 3) for g in granite],
            # object dtype keeps a missing Jev value as None (valid JSON), not NaN
            JEV_COLUMN: pd.Series(
                [None if jev is None else round(jev[i], 3) for i in range(len(questions))],
                dtype=object,
            ),
        }
    )
    styled = table.style.format(
        lambda v: "—" if pd.isna(v) else f"{v:g}", subset=["Granite noul", JEV_COLUMN]
    )

    jev_timing = _jev_timing_md(jev, jev_s, jev_status)
    fanout = granite[1:]
    fanout_prompt = sum(g.prompt_tokens for g in fanout)
    fanout_cached = sum(g.cached_tokens for g in fanout)
    cache_line = (
        f"questions 2–{len(questions)} reused {fanout_cached} of {fanout_prompt} prompt "
        f"tokens from cache ({fanout_cached / fanout_prompt:.0%})"
        if fanout_prompt
        else "single question, nothing to share"
    )
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"**Granite Switch:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {len(questions)} question(s) · `{MODEL_ID}` on vLLM, {GRANITE_HARDWARE}{wake_note}  \n"
        f"Batching: 2 adapter calls per question (prefilled Yes, prefilled No) · prime 1 call "
        f"{g_time['prime_s'] * 1000:.0f} ms, then {g_time['fanout_calls']} in parallel "
        f"{g_time['fanout_s'] * 1000:.0f} ms · {cache_line}"
    )
    jev_ms = jev_s * 1000 if jev is not None else None
    agreement = _agreement([g.noul > 0.5 for g in granite], jev)
    return styled, timing, jev_timing, g_time["total_s"] * 1000, jev_ms, agreement, warmer.status()


QUESTIONS_LABEL = f"Questions (one per line, up to {MAX_QUESTIONS})"
QUESTIONS_INFO = "Thinking Fast needs yes/no questions. Thinking Slow takes any question."


OUTPUT_HEADERS = {
    "fast": "### ⚡ Thinking Fast\nGranite Switch nouls: the uncertainty adapter's "
    "certainty('yes')/(certainty('yes')+certainty('no')) for each question.",
    "slow": "### 🐢 Thinking Slow\nGranite Switch writes each answer, then scores its "
    "certainty in it. Built with Mellea.",
}
IDLE_HEADER = "### Results\nPick a kind of thinking to start."
JEV_COLUMN = "Jev noul (reference)"


def _fmt_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _verdict(text: str | None) -> bool | None:
    """Yes/no from a written response's opening word; None if it has neither."""
    m = re.match(r"\W*(yes|no)\b", text or "", re.IGNORECASE)
    return None if m is None else m.group(1).lower() == "yes"


def _agreement(granite_yes: list[bool | None], jev: list[float] | None) -> dict | None:
    """How often Granite's yes/no matches Jev's (Jev yes = noul > 0.5).

    Questions where Granite gave no clear yes/no are skipped and counted.
    """
    if jev is None:
        return None
    pairs = [(g, j > 0.5) for g, j in zip(granite_yes, jev) if j is not None and g is not None]
    jev_yes = [g for g, j in pairs if j]
    jev_no = [g for g, j in pairs if not j]
    return {
        "yes_match": sum(jev_yes), "jev_yes": len(jev_yes),
        "no_match": sum(not g for g in jev_no), "jev_no": len(jev_no),
        "skipped": sum(g is None for g in granite_yes),
    }


def _stats_html(
    granite_ms: float | None,
    jev_ms: float | None = None,
    show_jev: bool = False,
    agreement: dict | None = None,
    show_agreement: bool = False,
) -> str:
    """Big tiles: Granite and Jev end-to-end times, and agreement with Jev."""
    tiles = [
        '<div class="e2e-tile granite"><div class="e2e-value">'
        f"{_fmt_time(granite_ms) if granite_ms is not None else '…'}</div>"
        '<div class="e2e-label">Granite Switch · end to end</div></div>'
    ]
    if show_jev:
        tiles.append(
            '<div class="e2e-tile reference"><div class="e2e-value">'
            f"{_fmt_time(jev_ms) if jev_ms is not None else '—'}</div>"
            '<div class="e2e-label">Jev · reference · end to end</div></div>'
        )
    if show_agreement:
        if agreement and agreement["jev_yes"] + agreement["jev_no"]:
            a = agreement
            matched, total = a["yes_match"] + a["no_match"], a["jev_yes"] + a["jev_no"]
            value = f"{matched / total:.0%}"
            label = (
                f"agreement with Jev · Jev yes: {a['yes_match']}/{a['jev_yes']} · "
                f"Jev no: {a['no_match']}/{a['jev_no']}"
            )
            if a["skipped"]:
                label += f" · {a['skipped']} without a clear yes/no"
        else:
            value, label = ("…" if agreement is None else "—"), "agreement with Jev"
        tiles.append(
            f'<div class="e2e-tile agreement"><div class="e2e-value">{value}</div>'
            f'<div class="e2e-label">{label}</div></div>'
        )
    return f'<div class="e2e-row">{"".join(tiles)}</div>'


def _slow_table(questions: list[str], results: list, jev: list[float] | None):
    """Thinking Slow table: response, its certainty, and Jev's noul as a reference."""
    table = pd.DataFrame(
        {
            "Question": questions,
            "Granite response": [r.answer if r else "…" for r in results],
            "Granite Certainty": pd.Series([r.certainty if r else None for r in results], dtype=object),
            JEV_COLUMN: pd.Series(
                [None if jev is None else jev[i] for i in range(len(questions))], dtype=object
            ),
        }
    )
    return table.style.format(
        lambda v: "…" if v is None or pd.isna(v) else f"{v:g}", subset=["Granite Certainty", JEV_COLUMN]
    )


def _jev_timing_md(jev: list[float] | None, jev_s: float, jev_status: str) -> str:
    return (
        f"**Jev (reference):** {jev_s * 1000:.0f} ms end to end · {jev_status}"
        if jev is not None
        else f"**Jev (reference):** unavailable · {jev_status}"
    )


def think(mode: str, state_text: str, questions_text: str):
    """Run one mode, started by its button.

    The first update sets the header to the chosen kind of thinking, shows that
    mode's results (hiding the others) and clears the last run, so the page
    switches the moment the button is clicked. Outputs, in order: header,
    tiles, Thinking Fast table, Jev timing, Thinking Slow table, Granite
    timing, endpoint status.
    """
    fast = mode == "fast"
    yield (
        OUTPUT_HEADERS[mode],
        _stats_html(None, show_jev=True, show_agreement=True),
        gr.update(visible=fast, value=None),
        "",
        gr.update(visible=not fast, value=None),
        "",
        gr.skip(),
    )
    if fast:
        styled, timing_md, jev_timing, granite_ms, jev_ms, agreement, status = compare(
            state_text, questions_text
        )
        yield (
            gr.skip(), _stats_html(granite_ms, jev_ms, True, agreement, True), styled,
            jev_timing, gr.skip(), timing_md, status,
        )
        return
    if not state_text.strip():
        raise gr.Error("Enter some state for the model to think about.")
    questions = _parse_questions(questions_text)
    # Jev answers the same questions as a reference while Granite thinks slow.
    with ThreadPoolExecutor(max_workers=1) as pool:
        jev_future = pool.submit(jev_nouls, _parse_state(state_text), questions)
        for results, timing_md, total_s in think_slow(state_text, questions):
            if total_s is None:
                jev_now = jev_future.result()[0] if jev_future.done() else None
                yield (
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    _slow_table(questions, results, jev_now), timing_md, warmer.status(),
                )
                continue
            jev, jev_s, jev_status = jev_future.result()
            agreement = _agreement([_verdict(r.answer) for r in results], jev)
            yield (
                gr.skip(),
                _stats_html(total_s * 1000, jev_s * 1000 if jev is not None else None, True, agreement, True),
                gr.skip(),
                _jev_timing_md(jev, jev_s, jev_status),
                _slow_table(questions, results, jev),
                timing_md,
                warmer.status(),
            )


def think_fast(state_text: str, questions_text: str):
    yield from think("fast", state_text, questions_text)


def think_slow_mode(state_text: str, questions_text: str):
    yield from think("slow", state_text, questions_text)


def on_page_load() -> str:
    """Start waking the endpoint as soon as someone opens the page."""
    warmer.ensure()
    return warmer.status()


# Light blue example boxes and the big end-to-end time tiles. Selectors are
# our own ids and classes, not Gradio internals.
CSS = """
/* Example boxes: a light shade of blue so they stand out (muted in dark mode). */
:root { --example-bg: #e8f2fc; --example-bg-hover: #d6e8fa; --example-border: #c5dcf3; }
.dark { --example-bg: #1f2d3d; --example-bg-hover: #27394d; --example-border: #34506e; }
#examples button {
  background: var(--example-bg) !important;
  border: 1px solid var(--example-border) !important;
}
#examples button:hover { background: var(--example-bg-hover) !important; }
.e2e-row { display: flex; gap: var(--spacing-lg); margin: var(--spacing-md) 0; }
.e2e-tile {
  flex: 1; padding: var(--spacing-lg) var(--spacing-xl);
  border-radius: var(--radius-lg); background: var(--background-fill-secondary);
  border: 2px solid var(--color-accent);
}
.e2e-tile.reference, .e2e-tile.agreement { border: 2px solid var(--border-color-primary); }
.e2e-value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.e2e-label { font-size: 0.9rem; margin-top: 2px; }
"""


with gr.Blocks(title="Thinking Fast and Slow with Granite") as demo:
    gr.Markdown(
        "# 🧠 Thinking Fast and Slow with Granite\n"
        "Some decisions don't need reasoning out loud. They need a fast, calibrated "
        "gut call. That's the idea behind *System One* models like TypeSafe AI's "
        "[Jev](https://docs.typesafe.ai): instead of text, Jev answers a yes/no "
        "question with a **noul**, the probability that the answer is yes.\n\n"
        "One open 3B model, Granite Switch, does both from a single endpoint:\n"
        "- **⚡ Thinking Fast:** a yes/no call as a noul, calculated by assessing "
        "the certainty('yes')/(certainty('yes')+certainty('no'))\n"
        "- **🐢 Thinking Slow:** a written answer plus Granite's certainty in it, "
        "built with [Mellea](https://mellea.ai). Jev returns decisions only; it "
        "doesn't generate text.\n\n"
        "### Technologies inside\n"
        "- **Granite Switch.** One checkpoint that bundles IBM's Granite 4.1 base "
        "model with 12 embedded adapter functions (RAG, safety, uncertainty and "
        "more), each selected per request by name. "
        "[Model card](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) · "
        "[GitHub](https://github.com/generative-computing/granite-switch) · "
        "[Adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html)\n"
        "- **Uncertainty quantification (UQ) adapter.** A calibrated adapter that "
        "scores how likely an answer is to be correct: of the answers it scores at "
        "X%, about X% are right. "
        "[Adapter README](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/uncertainty/README.md)\n"
        "- **aLoRA (activated LoRA).** Adapters that switch on at a trigger token "
        "and reuse the base model's KV cache for everything before it, so Granite "
        "reads your input once and every question, fast or slow, reuses that work. "
        "[Paper (NeurIPS 2025)](https://arxiv.org/abs/2504.12397) · "
        "[Code](https://github.com/IBM/activated-lora)\n"
        "- **Optimized vLLM kernels.** Granite Switch's vLLM integration, with "
        "kernels optimized by the Granite team, applies adapter weights per token "
        "position rather than per request, so adapter and base-model requests "
        "share batches and one KV cache."
    )
    # The inputs are created first but placed further down, so the examples can
    # sit right under the intro. They start with the first example filled in.
    state = gr.Textbox(
        label="State (text or JSON)",
        value=EXAMPLE_INPUTS[0][0],
        lines=6,
        max_lines=18,
        placeholder="The input the models think about.",
        render=False,
    )
    questions = gr.Textbox(
        label=QUESTIONS_LABEL,
        info=QUESTIONS_INFO,
        value=EXAMPLE_INPUTS[0][1],
        lines=5,
        placeholder="e.g. Is the customer asking for a refund?",
        render=False,
    )
    gr.Examples(
        EXAMPLE_INPUTS, inputs=[state, questions], example_labels=EXAMPLE_LABELS, elem_id="examples"
    )
    with gr.Row():
        with gr.Column():
            state.render()
            questions.render()
            with gr.Row():
                fast_btn = gr.Button("⚡ Thinking Fast", variant="primary")
                slow_btn = gr.Button("🐢 Thinking Slow", variant="primary")
            endpoint_status = gr.Markdown(warmer.status())
        with gr.Column():
            output_header = gr.Markdown(IDLE_HEADER)
            stats = gr.HTML()
            fast_table = gr.Dataframe(
                elem_id="fast-table",
                headers=["Question", "Granite noul", JEV_COLUMN],
                datatype=["str", "number", "number"],
                column_widths=["56%", "20%", "24%"],
                interactive=False,
                wrap=True,
                visible=False,
            )
            slow_table = gr.Dataframe(
                elem_id="slow-table",
                headers=["Question", "Granite response", "Granite Certainty", JEV_COLUMN],
                datatype=["str", "str", "number", "number"],
                column_widths=["22%", "48%", "12%", "18%"],
                interactive=False,
                wrap=True,
                visible=False,
            )
            timing = gr.Markdown()
            jev_timing = gr.Markdown()
    gr.Markdown(
        "Note: the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), and "
        "the noul is the probability-weighted average of those bins, so the "
        "Granite noul always falls between 0.05 and 0.95.\n\n"
        f"**Hardware:** Granite Switch is served by vLLM on {GRANITE_HARDWARE}, on a "
        "Hugging Face Inference Endpoint. Both modes run on that one GPU. Both "
        "times are full round trips. The endpoint scales to zero after 15 idle "
        "minutes; opening this page starts waking it, and any wait isn't counted in "
        "Granite's time."
    )
    # Each button starts its own kind of thinking straight away.
    run_outputs = [
        output_header, stats, fast_table, jev_timing,
        slow_table, timing, endpoint_status,
    ]
    fast_btn.click(think_fast, [state, questions], run_outputs, api_name="think_fast")
    slow_btn.click(think_slow_mode, [state, questions], run_outputs, api_name="think_slow")
    # Wake on page load, and keep the status line current while it wakes.
    demo.load(on_page_load, outputs=endpoint_status, show_progress="hidden")
    gr.Timer(3).tick(warmer.status, outputs=endpoint_status, show_progress="hidden", queue=False)

if __name__ == "__main__":
    demo.launch(css=CSS)
