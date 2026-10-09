# SPDX-License-Identifier: Apache-2.0
"""Noul Race: Granite Switch vs. Jev vs. GPT Luna on yes/no questions about any context.

Jev (TypeSafe AI's "System One" model) answers yes/no questions about a
piece of context with a *noul*: one calibrated number in [0, 1], the
probability that the answer is yes. OpenAI's GPT Luna does the same through
the Decisions API, where a "predicate" question returns a probability. This
Space makes the same kind of call with Granite Switch, served by vLLM on a
Hugging Face Inference Endpoint, and scores all three against an answer key.

The flow
--------
1. The user provides a context: their own text or JSON, or a random Wikipedia
   article pulled through the MediaWiki API by the "Random Wikipedia article"
   button.
2. The questions box holds one yes/no question per line, each followed by its
   answer: "...? Yes" or "...? No". The user writes them, or "Generate
   questions" asks gpt-oss-120b, on OpenRouter, for 10 of them.
3. "Race the Models" sends the context and questions to all three models.
4. Each noul above 0.5 counts as a yes and is scored against the answer on
   its line. A question with no answer still gets nouls but isn't scored.
   Six tiles show each model's end-to-end latency and accuracy.

That is the "Sprint" tab. The "Obstacle Course" tab is the same page with a mix
of question types: a line ending "...? Freeform" is an open-ended question.
Yes/no questions still go to each model's System One call (the uncertainty
adapter for Granite, the Decisions API for Luna); freeform questions go to a
chat completion. Each model takes the questions one at a time, in order: a
question isn't sent until the one before it has come back, and the table fills
in as the answers arrive. Jev returns decisions only, so it isn't run there
and shows greyed out.

The Granite noul, without ever asking Granite for its own answer
---------------------------------------------------------------
1. Ask with "Reply with exactly one word, 'Yes' or 'No'.", prefill the answer
   as "Yes", and run Granite Switch's embedded ``uncertainty`` adapter (an
   aLoRA) on it.
2. The noul is c(yes): the adapter's certainty that "Yes" is the correct
   answer.

The adapter call is the conversation Mellea's ``check_certainty`` sends, trimmed
so vLLM generates one token (the score digit) instead of the full JSON reply;
see ``_certainty``.

Batching on the endpoint
------------------------
The prompt puts the shared context first, then the instruction, then the
question, so every question shares one long prefix. The uncertainty adapter
is an aLoRA: it only activates at its invocation token, so it reads the base
model's KV cache for everything before that. vLLM's prefix cache can then
compute the context once and reuse it for every question:

1. **Prime:** send question 1 alone. vLLM computes and caches the shared
   prefix. (Requests scheduled in the same step can't share blocks that are
   still being computed, so priming beats sending everything at once.)
2. **Fan out:** send the other questions concurrently. vLLM batches them, and
   each prefills only its own question plus the prefilled "Yes".

vLLM's ``cached_tokens`` usage is reported so the cache hits are visible.
"""

import json
import math
import random
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass

import gradio as gr
import httpx
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from openai import OpenAI, OpenAIError
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

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

# GPT Luna is reached through OpenAI's Decisions API (public beta). It's called
# over plain HTTP so the Space doesn't depend on a recent OpenAI SDK.
OPENAI_DECISIONS_URL = "https://api.openai.com/v1/decisions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
LUNA_MODEL = os.environ.get("LUNA_MODEL", "gpt-6-luna")
# The model for Luna's freeform answers on the Obstacle Course (chat completions).
LUNA_CHAT_MODEL = os.environ.get("LUNA_CHAT_MODEL", LUNA_MODEL)
MAX_QUESTIONS = 50

# A random Wikipedia article is cut at a paragraph break to at most this many
# characters (roughly 5k tokens) so it fits the endpoint's context window with
# room to spare. Text the user pastes is sent as is.
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "20000"))
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Most random articles are stubs. Draw a batch and keep the ones with at least
# this much wikitext, so there's enough to ask ten questions about.
RANDOM_BATCH = 20
MIN_ARTICLE_BYTES = 6000
MIN_ARTICLE_CHARS = 2000
# Wikimedia asks API clients to identify themselves.
WIKIPEDIA_USER_AGENT = "NoulRace/1.0 (Hugging Face Space demo; https://huggingface.co/spaces)"

# The questions and their answer key are written by gpt-oss-120b, reached
# through OpenRouter's chat completions API.
QUESTION_MODEL = os.environ.get("QUESTION_MODEL", "openai/gpt-oss-120b")
NUM_QUESTIONS = 10
# The Obstacle Course's generated mix.
OBSTACLE_YES_NO = 10
OBSTACLE_FREEFORM = 5
# 15 short lines are about 300 tokens; the rest is headroom for a model that
# reasons before it answers.
GENERATION_MAX_TOKENS = 8192
# Sent to OpenRouter with the request, to keep generation quick: route to the
# provider with the highest throughput for this model (the hosts differ a lot),
# and have a reasoning model think only briefly before it writes.
GENERATION_OPTIONS = {"provider": {"sort": "throughput"}, "reasoning": {"effort": "low"}}

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
# Obstacle Course: freeform questions get a short written answer.
FREEFORM_INSTRUCTION = "Answer in a few words."
FREEFORM_MAX_TOKENS = 20
FREEFORM = "freeform"  # a question's answer when it's open-ended
PENDING = "…"  # an Obstacle Course result that hasn't come back yet


def _question_prompt(state: str, question: str) -> str:
    # Shared state + instruction first so every question reuses one cached prefix.
    return f"{state}\n\n{FAST_INSTRUCTION}\n{question}"


def _freeform_prompt(state: str, question: str) -> str:
    # Same layout as the yes/no prompt, so both kinds share the cached state.
    return f"{state}\n\n{FREEFORM_INSTRUCTION}\n{question}"


def _parse_state(text: str):
    """Jev accepts text or JSON state; pass JSON through as JSON when given."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return value if isinstance(value, (dict, list)) else text


def _wikipedia(params: dict) -> dict:
    response = httpx.get(
        WIKIPEDIA_API,
        params={"action": "query", "format": "json", "formatversion": 2, **params},
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()["query"]


def random_article() -> tuple[str, str]:
    """Pick a random English Wikipedia article with some substance.

    Returns (context, status line). Draws a batch of random articles, keeps
    the ones long enough to ask questions about, and pulls one's plain text.
    """
    try:
        for _ in range(3):
            batch = _wikipedia({
                "generator": "random", "grnnamespace": 0, "grnlimit": RANDOM_BATCH, "prop": "info",
            })["pages"]
            candidates = [page for page in batch if page.get("length", 0) >= MIN_ARTICLE_BYTES]
            random.shuffle(candidates)
            for candidate in candidates:
                page = _wikipedia({
                    "prop": "extracts", "explaintext": 1, "pageids": candidate["pageid"],
                })["pages"][0]
                full = (page.get("extract") or "").strip()
                if len(full) >= MIN_ARTICLE_CHARS:
                    return _cut_article(page["title"], full)
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        raise gr.Error(f"Couldn't fetch an article from Wikipedia: {e}")
    raise gr.Error("Didn't find a long enough random article. Try again.")


def _cut_article(title: str, full: str) -> tuple[str, str]:
    text = full
    if len(text) > MAX_CONTEXT_CHARS:
        # Cut at the last paragraph break inside the limit, when there is one.
        text = text[:MAX_CONTEXT_CHARS]
        text = (text[: text.rfind("\n")] if "\n" in text else text).strip()
    link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    note = f"**[{title}]({link})** from Wikipedia: {len(text):,} characters"
    if len(text) < len(full):
        note += f" (the first part of {len(full):,}; the rest is left out)"
    return text, note + "."


# One question line, as typed or generated: an optional bullet or number, the
# question up to its question mark, then optionally its answer: Yes, No, or
# Freeform for an open-ended question. A comma, "|" or other punctuation
# between the two is tolerated.
_QA_LINE = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?P<q>.+?\?)[\s|,:*\-–—]*(?P<a>yes|no|free[\s-]?form)?\W*$",
    re.IGNORECASE,
)


def _parse_line(line: str) -> tuple[str, bool | str | None]:
    """Split one line into (question, answer).

    The answer is True for Yes, False for No, FREEFORM for an open-ended
    question, and None when the line has no answer.
    """
    match = _QA_LINE.match(line)
    if not match:
        return line.strip(), None
    answer = (match["a"] or "").lower()
    if answer in ("yes", "no"):
        return match["q"].strip(), answer == "yes"
    return match["q"].strip(), FREEFORM if answer else None


def _parse_questions(text: str) -> tuple[list[str], list[bool | str | None]]:
    """The questions box as (questions, answers), one per non-empty line."""
    pairs = [_parse_line(line) for line in text.splitlines() if line.strip()]
    if not pairs:
        raise gr.Error("Enter at least one question (one per line), or click Generate questions.")
    if len(pairs) > MAX_QUESTIONS:
        raise gr.Error(f"At most {MAX_QUESTIONS} questions per run.")
    return [q for q, _ in pairs], [a for _, a in pairs]


# --------------------------------------------------------------------------- #
# Granite Switch
# --------------------------------------------------------------------------- #


@dataclass
class Certainty:
    noul: float  # c(yes): the adapter's certainty in a prefilled "Yes"
    prompt_tokens: int
    cached_tokens: int


@dataclass
class Reply:
    text: str  # Granite's written answer to a freeform question
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


def granite_noul(state: str, question: str) -> Certainty:
    """One question's noul: the adapter's certainty in a prefilled "Yes"."""
    c_yes, prompt_tokens, cached_tokens = _certainty(state, question, "Yes")
    return Certainty(noul=c_yes, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)


def granite_freeform(state: str, question: str) -> Reply:
    """One freeform question's answer: a plain chat completion on the base model."""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": _freeform_prompt(state, question)}],
        max_tokens=FREEFORM_MAX_TOKENS,
        temperature=0.0,
    )
    usage = response.usage
    details = usage.prompt_tokens_details if usage else None
    return Reply(
        text=(response.choices[0].message.content or "").strip(),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        cached_tokens=(details.cached_tokens or 0) if details else 0,
    )


def _prime_then_fan_out(jobs: list) -> tuple[list, dict]:
    """Run the first job alone to cache the shared prefix, then the rest together.

    The fan-out threads share one OpenAI client (one HTTP connection pool), so
    their requests reach vLLM together and it batches them on the cached prefix.
    """
    t0 = time.perf_counter()
    first = jobs[0]()
    t1 = time.perf_counter()
    rest: list = []
    if len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs) - 1) as pool:
            rest = list(pool.map(lambda job: job(), jobs[1:]))
    t2 = time.perf_counter()
    return [first, *rest], {"prime_s": t1 - t0, "fanout_s": t2 - t1, "total_s": t2 - t0}


def granite_nouls(state: str, questions: list[str]) -> tuple[list[Certainty], dict]:
    """Every question as a noul: one adapter call each, primed then fanned out."""
    return _prime_then_fan_out([partial(granite_noul, state, q) for q in questions])


def granite_course(state: str, questions: list[str], freeform: list[bool], results: list, timing: dict) -> None:
    """The Obstacle Course on Granite: one question at a time, in order.

    An adapter call for a yes/no question, a chat completion for a freeform
    one; the next isn't sent until the last has come back. Each result is
    written into `results` as it arrives, so the page can show it. Fills
    `timing` with the seconds waited for the endpoint and the seconds taken.
    """
    timing["wake_s"] = warmer.wait_ready()
    start = timing["start"] = time.perf_counter()
    for i, (question, is_freeform) in enumerate(zip(questions, freeform)):
        results[i] = (granite_freeform if is_freeform else granite_noul)(state, question)
    timing["total_s"] = time.perf_counter() - start
    warmer.touch()


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
# GPT Luna (OpenAI Decisions API)
# --------------------------------------------------------------------------- #


class LunaError(Exception):
    """An OpenAI call for Luna failed; the message is short enough to show on the page."""


def _luna_post(url: str, api_key: str, payload: dict) -> dict:
    try:
        response = httpx.post(
            url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise LunaError(f"HTTP {e.response.status_code} {e.response.text.strip()[:300]}")
    except (httpx.HTTPError, ValueError) as e:
        raise LunaError(str(e))


def _luna_decisions(api_key: str, context: str, questions: list[str]) -> list[float | None]:
    """Every question in one Decisions request, each as a "predicate".

    A predicate's answer carries the probability that it's true. A question
    Luna refuses to answer comes back without a probability and is left as None.
    """
    keys = [f"q{i}" for i in range(1, len(questions) + 1)]
    body = _luna_post(
        OPENAI_DECISIONS_URL,
        api_key,
        {
            "model": LUNA_MODEL,
            "input": context,
            "questions": [
                {"type": "predicate", "name": k, "instructions": q} for k, q in zip(keys, questions)
            ],
        },
    )
    try:
        answers = {a["name"]: a for a in body["answers"]}
    except (KeyError, TypeError) as e:
        raise LunaError(f"unexpected reply: {e}")
    return [answers.get(k, {}).get("probability") for k in keys]


def _luna_chat(api_key: str, context: str, question: str) -> str:
    """One freeform question's answer from a chat completion."""
    body = _luna_post(
        OPENAI_CHAT_URL,
        api_key,
        {
            "model": LUNA_CHAT_MODEL,
            "messages": [{"role": "user", "content": _freeform_prompt(context, question)}],
            "max_completion_tokens": FREEFORM_MAX_TOKENS,
            # Luna reasons at "medium" unless told otherwise; Granite's base
            # model doesn't reason, so turn it off for a like-for-like answer.
            "reasoning_effort": "none",
        },
    )
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LunaError(f"unexpected reply: {e}")


def luna_nouls(context: str, questions: list[str]) -> tuple[list[float | None] | None, float, str]:
    """Return (nouls, seconds, status). nouls is None when the call fails."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, 0.0, "OPENAI_API_KEY is not set on this Space."
    start = time.perf_counter()
    try:
        nouls = _luna_decisions(api_key, context, questions)
    except LunaError as e:
        return None, time.perf_counter() - start, f"Luna call failed: {e}"
    elapsed = time.perf_counter() - start
    refused = sum(n is None for n in nouls)
    status = f"`{LUNA_MODEL}` via OpenAI's Decisions API"
    if refused:
        status += f" · {refused} question(s) refused"
    return nouls, elapsed, status


def luna_course(
    context: str, questions: list[str], freeform: list[bool], results: list, timing: dict
) -> None:
    """The Obstacle Course on Luna: one question at a time, in order.

    A Decisions request for a yes/no question, a chat completion for a
    freeform one; the next isn't sent until the last has come back. Each
    result (a noul, a written answer, or None for a refusal or a failed call)
    is written into `results` as it arrives. Fills `timing` with the seconds
    taken and any errors.
    """
    timing["errors"] = []
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        timing["unavailable"] = "OPENAI_API_KEY is not set on this Space."
        results[:] = [None] * len(questions)
        return
    start = timing["start"] = time.perf_counter()
    for i, (question, is_freeform) in enumerate(zip(questions, freeform)):
        try:
            if is_freeform:
                results[i] = _luna_chat(api_key, context, question)
            else:
                results[i] = _luna_decisions(api_key, context, [question])[0]
        except LunaError as e:
            results[i] = None
            timing["errors"].append(str(e))
    timing["total_s"] = time.perf_counter() - start
    if len(timing["errors"]) == len(questions):
        timing["unavailable"] = f"Luna calls failed: {timing['errors'][0]}"


# --------------------------------------------------------------------------- #
# Question generation (gpt-oss-120b on OpenRouter)
# --------------------------------------------------------------------------- #

QUESTION_PROMPT = """\
Read the document below and write {n} yes/no questions about it, each with its correct answer.

Rules:
- Every question must be answerable from the document alone, with one clear, unambiguous answer.
- About half of the answers must be Yes and half No, in mixed order.
- Each question must make sense on its own and be different from the others.
- Output exactly {n} lines and nothing else: no numbering, no headings, no commentary.
- Each line is a question ending in a question mark, then a space, then Yes or No. For example:
Was the invoice paid on time? No

Document:
{context}"""

OBSTACLE_PROMPT = """\
Read the document below and write {n} questions about it: {yes_no} yes/no questions, \
each with its correct answer, and {freeform} open-ended questions.

Rules:
- Every question must be answerable from the document alone.
- Each yes/no question has one clear, unambiguous answer. About half of those answers must be Yes and half No.
- Each open-ended question starts with Who, What, When, Where, Why or How, cannot be answered \
with yes or no, and can be answered in a few words.
- Each question must make sense on its own and be different from the others.
- Output exactly {n} lines and nothing else: no numbering, no headings, no commentary.
- A yes/no line is the question ending in a question mark, then a space, then Yes or No. \
An open-ended line is the question ending in a question mark, then a space, then Freeform. For example:
Was the invoice paid on time? No
Why was the invoice disputed? Freeform

Document:
{context}"""

def _parse_generated(text: str) -> list[tuple[str, bool | str]]:
    """Pull (question, answer) pairs out of the model's reply, skipping repeats and unanswered lines."""
    pairs: dict[str, bool | str] = {}
    for line in text.splitlines():
        question, answer = _parse_line(line)
        if answer is not None:
            pairs.setdefault(question, answer)
    return list(pairs.items())


def generate_questions(prompt: str):
    """Have the question model write the questions and answers, streaming them back.

    Yields the (question, answer) pairs parsed so far each time another line
    is complete, so the page can show the questions as they're written. The
    last value yielded is the full set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise gr.Error("OPENROUTER_API_KEY is not set on this Space.")
    openrouter = OpenAI(base_url=f"{OPENROUTER_BASE_URL}/v1", api_key=api_key, timeout=120.0)
    content, finish_reason, shown = "", None, 0
    try:
        stream = openrouter.chat.completions.create(
            model=QUESTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=GENERATION_MAX_TOKENS,
            stream=True,
            extra_body=GENERATION_OPTIONS,
        )
        for chunk in stream:
            # Reasoning arrives in its own field and is ignored; a chunk can also have no choices.
            if not chunk.choices:
                continue
            finish_reason = chunk.choices[0].finish_reason or finish_reason
            content += chunk.choices[0].delta.content or ""
            # Only whole lines are parsed, so a question is never shown half-written.
            pairs = _parse_generated(content[: content.rfind("\n") + 1])
            if len(pairs) > shown:
                shown = len(pairs)
                yield pairs
    except OpenAIError as e:
        raise gr.Error(f"Couldn't generate questions: {e}")
    pairs = _parse_generated(content)
    if not pairs:
        raise gr.Error(
            f"`{QUESTION_MODEL}` didn't return any usable questions "
            f"(finish reason: {finish_reason}; reply began: {content[:200]!r})."
        )
    yield pairs


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

GENERATE_LABEL = "Generate questions"
RANDOM_LABEL = "🎲 Random Wikipedia article"
MODEL_NAMES = ["Granite Switch", "Jev", "GPT Luna"]
EMPTY_RESULTS = [(name, None, None) for name in MODEL_NAMES]
GREY = "#9ca3af"  # the Jev column on the Obstacle Course, where Jev isn't run

# What differs between the two tabs. `sits_out` names the models that aren't run.
PAGES = {
    "race": {
        "tab": "Sprint",
        "run_label": "🏁 Race the Models",
        "api_prefix": "",
        "sits_out": (),
        "header": (
            "### Results\n"
            "A noul above 0.5 counts as a yes, and ✓ / ✗ marks it against the answer "
            "on the question's line. Granite noul: c('yes'), the UQ aLoRA's certainty in a prefilled 'Yes'."
        ),
        "table_headers": ["Question", "Answer", "Granite noul", "Jev noul", "Luna noul"],
        "column_widths": ["43%", "9%", "16%", "16%", "16%"],
        "questions_label": (
            f"Yes/no questions, one per line (up to {MAX_QUESTIONS}), each "
            "followed by its answer: “…? Yes” or “…? No”"
        ),
        "questions_placeholder": (
            "Is the customer asking for a refund? Yes\n"
            "Was the order delivered on time? No\n\n"
            f"Write your own, or click {GENERATE_LABEL}."
        ),
    },
    "obstacle": {
        "tab": "Obstacle Course",
        "run_label": "🚧 Run the Obstacle Course",
        "api_prefix": "obstacle_",
        "sits_out": ("Jev",),
        "header": (
            "### Results\n"
            "Yes/no questions go to each model's System One call and come back as a noul: "
            "above 0.5 counts as a yes, and ✓ / ✗ marks it against the answer on the "
            "question's line. Freeform questions go to a chat completion and come back as a "
            "written answer, which isn't scored. Each model takes the questions one at a "
            "time, in order, and the table fills in as answers come back. Jev returns "
            "decisions only and can't take freeform questions, so it isn't run here."
        ),
        "table_headers": ["Question", "Answer", "Granite Switch", "Jev", "GPT Luna"],
        "column_widths": ["26%", "10%", "28%", "8%", "28%"],
        "questions_label": (
            f"Questions, one per line (up to {MAX_QUESTIONS}). Yes/no: “…? Yes” or "
            "“…? No”. Open-ended: “…? Freeform”"
        ),
        "questions_placeholder": (
            "Is the customer asking for a refund? Yes\n"
            "What went wrong with the order? Freeform\n\n"
            f"Write your own, or click {GENERATE_LABEL}."
        ),
    },
}


def _question_line(question: str, answer: bool | str) -> str:
    return f"{question} Freeform" if answer == FREEFORM else f"{question} {'Yes' if answer else 'No'}"


def on_generate(context: str, obstacle: bool = False):
    """Fill the questions box with generated questions, each followed by its answer.

    The questions appear in the box as the model writes them. On the Obstacle
    Course the set is a mix of yes/no and open-ended ("...? Freeform")
    questions, put in a random order once they're all in. Yields (questions,
    status).
    """
    if not context.strip():
        raise gr.Error(f"Enter some context first, or click {RANDOM_LABEL}.")
    if obstacle:
        n_yes_no, n_freeform = OBSTACLE_YES_NO, OBSTACLE_FREEFORM
        prompt = OBSTACLE_PROMPT.format(
            n=n_yes_no + n_freeform, yes_no=n_yes_no, freeform=n_freeform, context=context
        )
    else:
        n_yes_no, n_freeform = NUM_QUESTIONS, 0
        prompt = QUESTION_PROMPT.format(n=NUM_QUESTIONS, context=context)

    def wanted(generated: list) -> tuple[list, list]:
        yes_no = [(q, a) for q, a in generated if a != FREEFORM][:n_yes_no]
        freeform = [(q, a) for q, a in generated if a == FREEFORM][:n_freeform]
        return yes_no, freeform

    def lines(pairs: list) -> str:
        return "\n".join(_question_line(q, a) for q, a in pairs)

    start = time.perf_counter()
    yield "", f"Asking `{QUESTION_MODEL}` for {n_yes_no + n_freeform} questions…"
    generated: list = []
    for generated in generate_questions(prompt):
        yes_no, freeform = wanted(generated)
        yield lines(yes_no + freeform), (
            f"Writing questions… {len(yes_no) + len(freeform)} of {n_yes_no + n_freeform}"
        )
    seconds = time.perf_counter() - start
    yes_no, freeform = wanted(generated)
    pairs = yes_no + freeform
    if not pairs:
        raise gr.Error(f"`{QUESTION_MODEL}` didn't return any usable questions. Try again.")
    if obstacle:
        random.shuffle(pairs)  # which positions are freeform is random
    yes = sum(a is True for _, a in yes_no)
    counts = f"{yes} yes, {len(yes_no) - yes} no"
    if obstacle:
        counts = f"{len(yes_no)} yes/no: {counts}; {len(freeform)} freeform"
    status = (
        f"Generated {len(pairs)} questions ({counts}) with "
        f"`{QUESTION_MODEL}` via OpenRouter in {seconds:.1f} s."
    )
    if len(pairs) < n_yes_no + n_freeform:
        status += f" The model returned fewer than the {n_yes_no + n_freeform} requested."
    yield lines(pairs), status


def on_generate_race(context: str):
    yield from on_generate(context, obstacle=False)


def on_generate_obstacle(context: str):
    yield from on_generate(context, obstacle=True)


def _fmt_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _accuracy(results: list | None, expected: list[bool | None]) -> tuple[int, int] | None:
    """(correct, scored) over the questions with a yes/no answer and a noul."""
    if results is None:
        return None
    scored = [(n > 0.5) == e for n, e in zip(results, expected) if e is not None and n is not None]
    return (sum(scored), len(scored)) if scored else None


def _noul_cell(noul: float | None, expected: bool | None) -> str:
    if noul is None:
        return "—"
    if expected is None:
        return f"{noul:.3f}"
    return f"{noul:.3f} {'✓' if (noul > 0.5) == expected else '✗'}"


def _answer_cell(answer: bool | str | None) -> str:
    if answer == FREEFORM:
        return "Freeform"
    return "—" if answer is None else "Yes" if answer else "No"


STOPWATCH_TICK_S = 0.1  # how often the running stopwatches are redrawn


def _stats_html(
    results: list[tuple[str, float | None, tuple[int, int] | None]],
    running: bool = False,
    sits_out: tuple[str, ...] = (),
    finished: tuple[str, ...] = (),
) -> str:
    """Two big tiles per model: end-to-end latency, with its accuracy beneath.

    One column per model: the top row is latency, the bottom row accuracy.
    `results` is one (name, milliseconds, (correct, scored)) per model.

    While `running`, the latency tiles are stopwatches: each shows the
    model's elapsed time so far, and accuracy waits as "…". Once the run is
    over, the tiles show the final figures ("—" for a model with none).

    The latency tile of the fastest model that has finished is green. While
    running, only the models named in `finished` are in contention, so the
    tile turns green the moment the first model is done, without waiting for
    the others. A model named in `sits_out` isn't run: its tiles are greyed
    out and read N/A.
    """
    blank = "…" if running else "—"
    done = [
        (ms, name)
        for name, ms, _ in results
        if ms is not None and name not in sits_out and (name in finished or not running)
    ]
    winner = min(done)[1] if done else None

    def tile(name: str, value: str, label: str, extra: str = "") -> str:
        out = name in sits_out
        return (
            f'<div class="e2e-tile{" sits-out" if out else extra}">'
            f'<div class="e2e-value">{"N/A" if out else value}</div>'
            f'<div class="e2e-label">{label}</div></div>'
        )

    def latency_value(ms: float | None) -> str:
        if ms is None:
            return blank
        # A stopwatch reads in tenths of a second so it doesn't jump between units.
        return f"{ms / 1000:.1f} s" if running else _fmt_time(ms)

    # The grid fills row by row, so all latency tiles come first, then accuracy.
    latency = [
        tile(name, latency_value(ms), f"{name} · end to end", " winner" if name == winner else "")
        for name, ms, _ in results
    ]
    accuracy = [
        tile(name, blank, f"{name} · accuracy")
        if acc is None
        else tile(name, f"{acc[0] / acc[1]:.0%}", f"{name} · accuracy · {acc[0]}/{acc[1]} correct")
        for name, _, acc in results
    ]
    return (
        f'<div class="e2e-grid" style="grid-template-columns: repeat({len(results)}, minmax(0, 1fr))">'
        f'{"".join(latency + accuracy)}</div>'
    )


def _stopwatches(clocks: dict[str, dict]) -> dict:
    """Arguments for `_stats_html` while the models run: readings and who's done.

    A clock is a dict that gains "start" when the model's run begins and
    "total_s" when it ends, so a finished model's stopwatch stops while the
    others keep going. A model that hasn't started yet reads None.
    """
    now = time.perf_counter()
    readings, finished = [], []
    for name in MODEL_NAMES:
        clock = clocks.get(name, {})
        if "total_s" in clock:
            readings.append((name, clock["total_s"] * 1000, None))
            finished.append(name)
        elif "start" in clock:
            readings.append((name, (now - clock["start"]) * 1000, None))
        else:
            readings.append((name, None, None))
    return {"results": readings, "finished": tuple(finished), "running": True}


def _clocked(clock: dict, run, *args):
    """Call `run`, noting in `clock` when it started and how long it took."""
    start = clock["start"] = time.perf_counter()
    try:
        return run(*args)
    finally:
        clock["total_s"] = time.perf_counter() - start


def _reference_timing_md(name: str, results: list | None, seconds: float, status: str) -> str:
    return (
        f"**{name}:** {seconds * 1000:.0f} ms end to end · {status}"
        if results is not None
        else f"**{name}:** unavailable · {status}"
    )


def _cache_line(granite: list, n: int) -> str:
    fanout = granite[1:]
    fanout_prompt = sum(g.prompt_tokens for g in fanout)
    fanout_cached = sum(g.cached_tokens for g in fanout)
    if not fanout_prompt:
        return "single question, nothing to share"
    return (
        f"questions 2–{n} reused {fanout_cached} of {fanout_prompt} prompt "
        f"tokens from cache ({fanout_cached / fanout_prompt:.0%})"
    )


def _granite_timing_md(g_time: dict, n: int, wake_s: float, batching: str, cache_line: str) -> str:
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    return (
        f"**Granite Switch:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {n} question(s) · `{MODEL_ID}` on vLLM, {GRANITE_HARDWARE}{wake_note}  \n"
        f"Batching: {batching} · prime 1 question "
        f"{g_time['prime_s'] * 1000:.0f} ms, then {n - 1} in parallel "
        f"{g_time['fanout_s'] * 1000:.0f} ms · {cache_line}"
    )


def race(context: str, questions_text: str):
    """Send the questions to every model and score them against their answers.

    The first update clears the last run, so the page responds the moment the
    button is clicked. While the models run, the latency tiles are stopwatches;
    when all three are back, the tiles show the final figures and the fastest
    model's latency tile turns green. Outputs, in order: tiles, table, Granite
    timing, Jev timing, Luna timing, endpoint status.
    """
    if not context.strip():
        raise gr.Error("Enter some context for the models to decide about.")
    questions, answers = _parse_questions(questions_text)
    # Every question is a yes/no question here; a Freeform marker just leaves it unscored.
    expected = [a if isinstance(a, bool) else None for a in answers]
    yield _stats_html(EMPTY_RESULTS, running=True), None, "", "", "", gr.skip()

    clocks: dict[str, dict] = {name: {} for name in MODEL_NAMES}

    def granite_run():
        # Granite's stopwatch starts once the endpoint is awake.
        waited = warmer.wait_ready()
        result = _clocked(clocks["Granite Switch"], granite_nouls, context, questions)
        warmer.touch()
        return waited, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        granite_future = pool.submit(granite_run)
        jev_future = pool.submit(_clocked, clocks["Jev"], jev_nouls, _parse_state(context), questions)
        luna_future = pool.submit(_clocked, clocks["GPT Luna"], luna_nouls, context, questions)
        runs = [granite_future, jev_future, luna_future]
        while not all(run.done() for run in runs):
            time.sleep(STOPWATCH_TICK_S)
            yield (
                _stats_html(**_stopwatches(clocks)),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
        wake_s, (granite, g_time) = granite_future.result()
        jev, jev_s, jev_status = jev_future.result()
        luna, luna_s, luna_status = luna_future.result()

    granite_nouls_ = [g.noul for g in granite]
    table = pd.DataFrame(
        {
            "Question": questions,
            "Answer": [_answer_cell(e) for e in expected],
            "Granite noul": [_noul_cell(n, e) for n, e in zip(granite_nouls_, expected)],
            "Jev noul": [
                _noul_cell(None if jev is None else jev[i], e) for i, e in enumerate(expected)
            ],
            "Luna noul": [
                _noul_cell(None if luna is None else luna[i], e) for i, e in enumerate(expected)
            ],
        }
    )
    timing = _granite_timing_md(
        g_time, len(questions), wake_s,
        "1 adapter call per question (prefilled Yes)", _cache_line(granite, len(questions)),
    )
    if not any(e is not None for e in expected):
        timing += (
            "\n\n**Not scored:** none of the questions has an answer. Put Yes or No "
            "after each question mark to score the models."
        )
    stats = _stats_html(
        [
            ("Granite Switch", g_time["total_s"] * 1000, _accuracy(granite_nouls_, expected)),
            ("Jev", jev_s * 1000 if jev is not None else None, _accuracy(jev, expected)),
            ("GPT Luna", luna_s * 1000 if luna is not None else None, _accuracy(luna, expected)),
        ]
    )
    yield (
        stats, table, timing,
        _reference_timing_md("Jev", jev, jev_s, jev_status),
        _reference_timing_md("GPT Luna", luna, luna_s, luna_status),
        warmer.status(),
    )


def _course_cell(result, expected: bool | None, is_freeform: bool) -> str:
    if result is PENDING:
        return PENDING
    if is_freeform:
        return "—" if result is None else str(result)
    return _noul_cell(result, expected)


def obstacle_course(context: str, questions_text: str):
    """Run a mix of yes/no and freeform questions on Granite and Luna, one at a time.

    A yes/no question goes to the model's System One call and comes back as a
    noul; a freeform one ("...? Freeform") goes to a chat completion and
    comes back as text. Each model works through the questions in order, and
    doesn't send one until the one before it has come back; the two models
    run side by side. While they run, the latency tiles are stopwatches and
    the table fills in as answers arrive. Once both have finished, the tiles
    show the final figures, the faster model's latency tile turns green, and
    the timing lines are filled in. Jev can't
    take freeform questions, so it isn't run: its tiles and column are greyed
    out. Same outputs as `race`.
    """
    sits_out = PAGES["obstacle"]["sits_out"]
    if not context.strip():
        raise gr.Error("Enter some context for the models to work from.")
    questions, answers = _parse_questions(questions_text)
    freeform = [a == FREEFORM for a in answers]
    expected = [a if isinstance(a, bool) else None for a in answers]
    granite: list = [PENDING] * len(questions)
    luna: list = [PENDING] * len(questions)
    g_time: dict = {}
    luna_time: dict = {}

    def table():
        granite_results = [
            g.text if isinstance(g, Reply) else g.noul if isinstance(g, Certainty) else g for g in granite
        ]
        frame = pd.DataFrame(
            {
                "Question": questions,
                "Answer": [_answer_cell(a) for a in answers],
                "Granite Switch": [
                    _course_cell(r, e, f) for r, e, f in zip(granite_results, expected, freeform)
                ],
                "Jev": ["N/A"] * len(questions),
                "GPT Luna": [_course_cell(r, e, f) for r, e, f in zip(luna, expected, freeform)],
            }
        )
        return granite_results, frame.style.set_properties(subset=["Jev"], color=GREY)

    yield _stats_html(EMPTY_RESULTS, running=True, sits_out=sits_out), table()[1], "", "", "", gr.skip()

    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [
            pool.submit(granite_course, context, questions, freeform, granite, g_time),
            pool.submit(luna_course, context, questions, freeform, luna, luna_time),
        ]
        clocks = {"Granite Switch": g_time, "GPT Luna": luna_time}
        shown = None
        while not all(run.done() for run in runs):
            time.sleep(STOPWATCH_TICK_S)
            arrived = sum(r is not PENDING for r in granite + luna)
            yield (
                _stats_html(**_stopwatches(clocks), sits_out=sits_out),
                table()[1] if arrived != shown else gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            shown = arrived
        for run in runs:
            run.result()  # surface a failed run

    granite_results, styled = table()
    n_free = sum(freeform)
    n = len(questions)
    cache_line = _cache_line(granite, n)
    wake_s = g_time.get("wake_s", 0.0)
    wake_note = f" · waited {wake_s:.0f} s for the endpoint to wake (not counted)" if wake_s > 5 else ""
    timing = (
        f"**Granite Switch:** {g_time['total_s'] * 1000:.0f} ms end to end "
        f"for {n} question(s) · `{MODEL_ID}` on vLLM, {GRANITE_HARDWARE}{wake_note}  \n"
        f"One at a time: {n - n_free} adapter call(s) for yes/no, {n_free} chat completion(s) "
        f"for freeform (up to {FREEFORM_MAX_TOKENS} tokens each) · {cache_line}"
    )
    if not any(e is not None for e in expected):
        timing += (
            "\n\n**Not scored:** none of the questions has a Yes or No answer, so there's "
            "no accuracy to show."
        )
    luna_ok = "unavailable" not in luna_time
    if luna_ok:
        luna_status = (
            f"one at a time: {n - n_free} yes/no via the Decisions API (`{LUNA_MODEL}`), "
            f"{n_free} freeform via chat completions (`{LUNA_CHAT_MODEL}`, reasoning off)"
        )
        if luna_time["errors"]:
            luna_status += f" · {len(luna_time['errors'])} call(s) failed, first: {luna_time['errors'][0]}"
        luna_md = f"**GPT Luna:** {luna_time['total_s'] * 1000:.0f} ms end to end · {luna_status}"
    else:
        luna_md = f"**GPT Luna:** unavailable · {luna_time['unavailable']}"
    stats = _stats_html(
        [
            ("Granite Switch", g_time["total_s"] * 1000, _accuracy(granite_results, expected)),
            ("Jev", None, None),
            ("GPT Luna", luna_time["total_s"] * 1000 if luna_ok else None, _accuracy(luna, expected) if luna_ok else None),
        ],
        sits_out=sits_out,
    )
    yield (
        stats,
        styled,
        timing,
        f'<span style="color: {GREY}">**Jev:** not run. Jev returns decisions only and '
        "can't take freeform questions.</span>",
        luna_md,
        warmer.status(),
    )


def on_page_load() -> str:
    """Start waking the endpoint as soon as someone opens the page."""
    warmer.ensure()
    return warmer.status()


# The big metric tiles. Selectors are our own classes, not Gradio internals.
CSS = """
.e2e-grid { display: grid; gap: var(--spacing-lg); margin: var(--spacing-md) 0; }
.e2e-tile {
  padding: var(--spacing-lg) var(--spacing-xl);
  border-radius: var(--radius-lg); background: var(--background-fill-secondary);
  border: 2px solid var(--border-color-primary);
}
.e2e-tile.sits-out { opacity: 0.4; }
/* The latency tile of the fastest model to finish. */
.e2e-tile.winner { background: #c9f0d6; border-color: #2e9e5b; }
.dark .e2e-tile.winner { background: #17512e; border-color: #3fb872; }
.e2e-value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.e2e-label { font-size: 0.9rem; margin-top: 2px; }
"""


def build_page(kind: str, endpoint_status: gr.Markdown) -> None:
    """Lay out one tab and wire its buttons. Both tabs share this layout."""
    page = PAGES[kind]
    obstacle = kind == "obstacle"
    prefix = page["api_prefix"]
    with gr.Row():
        with gr.Column():
            # Left column, top to bottom in the order it's used: context (typed
            # or a random article), questions (typed or generated), run.
            state = gr.Textbox(
                label="Context (text or JSON)",
                lines=10,
                max_lines=18,
                placeholder=f"Paste the document the models work from, or click {RANDOM_LABEL}.",
            )
            random_button = gr.Button(RANDOM_LABEL)
            random_status = gr.Markdown()
            questions = gr.Textbox(
                label=page["questions_label"],
                lines=8,
                max_lines=18,
                placeholder=page["questions_placeholder"],
            )
            generate = gr.Button(GENERATE_LABEL)
            generate_status = gr.Markdown()
            run = gr.Button(page["run_label"], variant="primary")
        with gr.Column():
            gr.Markdown(page["header"])
            stats = gr.HTML(_stats_html(EMPTY_RESULTS, sits_out=page["sits_out"]))
            table = gr.Dataframe(
                elem_id=f"{kind}-results-table",
                headers=page["table_headers"],
                datatype=["str", "str", "str", "str", "str"],
                column_widths=page["column_widths"],
                interactive=False,
                wrap=True,
            )
            timing = gr.Markdown()
            jev_timing = gr.Markdown()
            luna_timing = gr.Markdown()
    # A new article makes the old questions stale, so clear them.
    random_button.click(
        random_article, outputs=[state, random_status], api_name=f"{prefix}random_article"
    ).then(lambda: ("", ""), outputs=[questions, generate_status], show_progress="hidden", api_name=False)
    generate.click(
        on_generate_obstacle if obstacle else on_generate_race, state, [questions, generate_status],
        api_name=f"{prefix}generate",
    )
    run.click(
        obstacle_course if obstacle else race,
        [state, questions],
        [stats, table, timing, jev_timing, luna_timing, endpoint_status],
        api_name=f"{prefix}race",
    )


with gr.Blocks(title="Noul Race") as demo:
    gr.Markdown("# 🏁 Noul Race")
    # One endpoint status line under the tabs, shared by both.
    endpoint_status = gr.Markdown(warmer.status(), render=False)
    with gr.Tabs():
        # The Obstacle Course is the first tab, so it's the one the page opens on.
        for kind in ("obstacle", "race"):
            with gr.Tab(PAGES[kind]["tab"]):
                build_page(kind, endpoint_status)
    endpoint_status.render()
    gr.Markdown(
        f"**Random article:** the plain text of a random English Wikipedia article, "
        f"cut to the first {MAX_CONTEXT_CHARS:,} characters when it's longer.\n\n"
        f"**Generated questions and answers** are written by `{QUESTION_MODEL}` on "
        "OpenRouter, and can be wrong. Edit any answer in the box before running.\n\n"
        "**Granite noul:** the uncertainty adapter scores ten bins (0.05, 0.15, … 0.95), "
        "and the certainty is the probability-weighted average of those bins, so the "
        "Granite noul always falls between 0.05 and 0.95.\n\n"
        f"**Hardware:** Granite Switch is served by vLLM on {GRANITE_HARDWARE}, on a "
        "Hugging Face Inference Endpoint. All times are full round trips. The "
        "endpoint scales to zero after 15 idle minutes; opening this page starts "
        "waking it, and any wait isn't counted in Granite's time."
    )
    # Wake on page load, and keep the status line current while it wakes.
    demo.load(on_page_load, outputs=endpoint_status, show_progress="hidden")
    gr.Timer(3).tick(warmer.status, outputs=endpoint_status, show_progress="hidden", queue=False)

if __name__ == "__main__":
    demo.launch(css=CSS)
