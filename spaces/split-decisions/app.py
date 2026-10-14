# SPDX-License-Identifier: Apache-2.0
"""Decisions Switching Costs: five models on a mix of yes/no and brainstorming questions.

Some questions want a fast yes/no call; others want a written answer. This
Space gives five models a list that mixes the two and times each one end to
end. It grew out of Noul Race's Obstacle Course.

The flow
--------
1. When the page loads, it pulls a random Wikipedia article and asks
   gpt-oss-120b, on OpenRouter, for 10 yes/no and 5 brainstorming questions about
   it, shown in a random order. "New article" does the same again.
2. "Run the models" gives the article and the questions to every model. Each
   model takes the questions one at a time, in order: a question isn't sent
   until the answer to the one before it has come back.

How each model answers
----------------------
- A yes/no question (⚡) comes back as a *noul*: the probability that the
  answer is yes.
- A freeform question (💬) asks the model to come up with something, and
  comes back as a one-word answer, with reasoning off.

Nothing is scored; the page compares speed.

==============  ==============================  ==========================
Model           Yes/no                          Freeform
==============  ==============================  ==========================
Granite Switch  uncertainty adapter, c(yes)     base model, same endpoint
JEV-9B          decision adapter, P("true")     base model, same endpoint
Gemma Decider   decider's letter readout        same model, chat completion
GPT Luna        gpt-6-luna-decisions            gpt-6-luna
Jev + Qwen      jev-1.13                        qwen3.8-27b
==============  ==============================  ==========================

The last two rows run on OpenRouter, which sends each chat request to the
fastest provider it has for the model. Jev returns decisions only, so it is
paired with Qwen, which writes its freeform answers.

The page
--------
A time chart, with a bar per model that grows question by question, split
into decision time and chat time, and medals for the first three to finish.
Under it, a compact table: a row per question, a column per model, filling
downwards as the answers arrive.

The code
--------
Each model is a `Racer` with two methods, `noul` and `freeform`. One function,
`run_course`, takes any racer through the questions and records what happens in
a `Run`; the board is drawn from the runs.
"""

import html
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import httpx
import yaml
from huggingface_hub import hf_hub_download
from openai import OpenAI, OpenAIError

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# Granite Switch endpoint (vLLM, OpenAI-compatible). HF_ENDPOINT_URL ends in /v1.
ENDPOINT_URL = os.environ["HF_ENDPOINT_URL"].rstrip("/")
HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_ID = os.environ.get("MODEL_ID", "ibm-granite/granite-switch-4.1-3b-preview")
# The endpoint's GPU, shown in the notes. Update if the endpoint's hardware changes.
GRANITE_HARDWARE = os.environ.get("GRANITE_HARDWARE", "a single NVIDIA L4 GPU (24 GB)")

# JEV-9B is AutoTrust's open reproduction of Jev (not a TypeSafe model): the
# Qwen3.5-9B base model plus a decision adapter, served by stock vLLM on its own
# Inference Endpoint (scripts/create_jev_endpoint.py). Without the URL the model
# shows as unavailable. It takes the same token as the Granite endpoint.
JEV9B = "JEV-9B"  # its name on the page
JEV9B_REPO = os.environ.get("JEV9B_MODEL_ID", "autotrust/JEV-9B")
JEV9B_ENDPOINT_URL = os.environ.get("JEV9B_ENDPOINT_URL", "").rstrip("/")
if JEV9B_ENDPOINT_URL and not JEV9B_ENDPOINT_URL.endswith("/v1"):
    JEV9B_ENDPOINT_URL += "/v1"
JEV9B_HARDWARE = os.environ.get("JEV9B_HARDWARE", "a single NVIDIA L40S GPU (48 GB)")
JEV9B_DECISION_MODEL = "jev-decision"  # the adapter's name on the endpoint

# Gemma Decider: Mapika/decider-chat-gemma4-31b, the unmodified Gemma 4 31B plus
# a decider config, served by stock vLLM on its own Inference Endpoint
# (scripts/create_gemma_decider_endpoint.py). Without the URL the model shows
# as unavailable. It takes the same token as the Granite endpoint.
GEMMA = "Gemma Decider"  # its name on the page
GEMMA_REPO = os.environ.get("GEMMA_DECIDER_MODEL_ID", "Mapika/decider-chat-gemma4-31b")
GEMMA_ENDPOINT_URL = os.environ.get("GEMMA_DECIDER_ENDPOINT_URL", "").rstrip("/")
if GEMMA_ENDPOINT_URL and not GEMMA_ENDPOINT_URL.endswith("/v1"):
    GEMMA_ENDPOINT_URL += "/v1"
GEMMA_HARDWARE = os.environ.get("GEMMA_DECIDER_HARDWARE", "a single NVIDIA A100 GPU (80 GB)")
# decider-ai builds the decision prompts. Only its pure-Python prompt modules
# are used, so it's installed without its dependencies (torch and more) at
# start-up; see _load_decider.
DECIDER_PACKAGE = "decider-ai==1.9.0"
# The endpoint's --max-model-len is 8192; this leaves room for the template,
# the question and the options after the article.
GEMMA_MAX_STATE_TOKENS = 6144

# The endpoints scale to zero after 15 idle minutes; the first request after
# that wakes them (a cold start takes ~3-5 minutes).
WAKE_TIMEOUT_S = 420
# Treat an endpoint as warm for this long after its last use, comfortably
# inside the 15-minute scale-to-zero window.
READY_TTL_S = 10 * 60

# Everything else runs on OpenRouter, with one key. Decision models (Jev, Luna
# Decisions) take a System One request; the rest take a chat completion.
OPENROUTER_SYSTEM_ONE_URL = "https://openrouter.ai/api/v1/systemone"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter_id(name: str, default: str) -> str:
    """A model id from the environment, with the default's vendor prefix when it has none."""
    value = os.environ.get(name) or default
    return value if "/" in value else f"{default.split('/')[0]}/{value}"


JEV_MODEL = _openrouter_id("JEV_MODEL", "typesafe/jev-1.13")
# Jev returns decisions only, so this model writes its freeform answers.
JEV_PARTNER_MODEL = _openrouter_id("JEV_PARTNER_MODEL", "qwen/qwen3.8-27b")
# GPT Luna is two model ids on OpenRouter: one for decisions, one for text.
LUNA_DECISIONS_MODEL = _openrouter_id("LUNA_DECISIONS_MODEL", "openai/gpt-6-luna-decisions")
LUNA_MODEL = _openrouter_id("LUNA_MODEL", "openai/gpt-6-luna")
# Sent with every OpenRouter chat request: route to the provider with the
# highest throughput for the model (the hosts differ a lot).
FASTEST = {"provider": {"sort": "throughput"}}
# Added to a freeform request, so no model reasons before it writes. Luna
# reasons at "medium" and Qwen thinks unless told otherwise; the two vLLM
# models don't reason, so this keeps the answers like for like.
NO_REASONING = {"reasoning": {"effort": "none"}}
# OpenAI's fastest processing tier, passed through by OpenRouter. It applies
# to Luna's chat requests only: OpenAI's Decisions API has no service tier
# (it rejects the parameter), OpenRouter ignores one on decision requests,
# and Jev's and Qwen's providers have no tiers.
PRIORITY = {"service_tier": "priority"}

# A random Wikipedia article is cut at a paragraph break to at most this many
# characters (roughly 5k tokens) so it fits the endpoints' context windows with
# room to spare.
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "20000"))
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Most random articles are stubs. Draw a batch and keep the ones with at least
# this much wikitext, so there's enough to ask fifteen questions about.
RANDOM_BATCH = 20
MIN_ARTICLE_BYTES = 6000
MIN_ARTICLE_CHARS = 2000
# Wikimedia asks API clients to identify themselves.
WIKIPEDIA_USER_AGENT = "SplitDecisions/1.0 (Hugging Face Space demo; https://huggingface.co/spaces)"

# The questions are written by gpt-oss-120b, which always reasons first; it is
# asked to do so only briefly.
QUESTION_MODEL = _openrouter_id("QUESTION_MODEL", "openai/gpt-oss-120b")
NUM_YES_NO = 10
NUM_FREEFORM = 5
# 15 short lines are about 300 tokens; the rest is headroom for its reasoning.
GENERATION_MAX_TOKENS = 8192

YES_NO_INSTRUCTION = "Reply with exactly one word, 'Yes' or 'No'."
FREEFORM_INSTRUCTION = "Answer in one word."
FREEFORM_MAX_TOKENS = 20


def _prompt(context: str, instruction: str, question: str) -> str:
    # Context first, so every question on a vLLM endpoint reuses one cached prefix.
    return f"{context}\n\n{instruction}\n{question}"


# --------------------------------------------------------------------------- #
# Context: a random Wikipedia article
# --------------------------------------------------------------------------- #


def _wikipedia(params: dict) -> dict:
    response = httpx.get(
        WIKIPEDIA_API,
        params={"action": "query", "format": "json", "formatversion": 2, **params},
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()["query"]


def random_article() -> dict:
    """Pick a random English Wikipedia article with some substance.

    Returns its title, link, text (the plain text the models work from) and
    full_chars (its length before any cut). Draws a batch of random articles,
    keeps the ones long enough to ask questions about, and pulls one's text.
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


def _cut_article(title: str, full: str) -> dict:
    text = full
    if len(text) > MAX_CONTEXT_CHARS:
        # Cut at the last paragraph break inside the limit, when there is one.
        text = text[:MAX_CONTEXT_CHARS]
        text = (text[: text.rfind("\n")] if "\n" in text else text).strip()
    link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return {"title": title, "link": link, "text": text, "full_chars": len(full)}


# --------------------------------------------------------------------------- #
# Questions: generating a set
# --------------------------------------------------------------------------- #

GENERATION_PROMPT = """\
Read the document below and write {n} questions about it: {yes_no} yes/no questions \
and {freeform} brainstorming questions.

Rules:
- Each yes/no question must be answerable from the document alone, with one clear answer. \
For about half of them the answer is yes, and for the rest no.
- Each brainstorming question asks the reader to come up with something based on the \
document, not to look a fact up: a fitting word, label, name, adjective or theme. It has no \
single right answer, cannot be answered with yes or no, and asks for exactly one word.
- Each question must make sense on its own and be different from the others.
- Output exactly {n} lines and nothing else: no numbering, no headings, no commentary, no answers.
- A yes/no line is the question ending in a question mark, then a space, then Binary. \
A brainstorming line is the question ending in a question mark, then a space, then Freeform. For example:
Was the invoice paid on time? Binary
In one word, how would the customer have felt about the late invoice? Freeform
What one-word nickname would suit the delivery company? Freeform

Document:
{context}"""

# One generated line: an optional bullet or number, the question up to its
# question mark, then its kind. A model that slips and writes the answer
# ("Yes", "No") in place of "Binary" has still written a yes/no question.
_QUESTION_LINE = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?P<q>.+?\?)[\s|,:*(\-–—]*(?P<kind>binary|yes|no|free[\s-]?form)\b",
    re.IGNORECASE,
)


def _parse_generated(text: str) -> list[tuple[str, bool]]:
    """The (question, is_freeform) pairs in the model's reply, capped at the numbers asked for.

    Repeats and lines that don't fit the format are skipped. Yes/no questions
    come first, then freeform ones.
    """
    kinds: dict[str, bool] = {}
    for line in text.splitlines():
        match = _QUESTION_LINE.match(line)
        if match:
            kinds.setdefault(match["q"].strip(), match["kind"].lower().startswith("free"))
    yes_no = [(q, False) for q, is_freeform in kinds.items() if not is_freeform][:NUM_YES_NO]
    freeform = [(q, True) for q, is_freeform in kinds.items() if is_freeform][:NUM_FREEFORM]
    return yes_no + freeform


def generate_questions(context: str):
    """Have the question model write the questions, streaming them back.

    Yields the pairs parsed so far each time another line is complete, so the
    page can show the questions as they're written. The last value yielded is
    the full set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise gr.Error("OPENROUTER_API_KEY is not set on this Space.")
    openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=120.0)
    prompt = GENERATION_PROMPT.format(
        n=NUM_YES_NO + NUM_FREEFORM, yes_no=NUM_YES_NO, freeform=NUM_FREEFORM, context=context
    )
    # The model now and then leaves out a whole kind of question; ask once more when it does.
    for _ in range(2):
        content, finish_reason, shown = "", None, 0
        try:
            stream = openrouter.chat.completions.create(
                model=QUESTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GENERATION_MAX_TOKENS,
                stream=True,
                extra_body={**FASTEST, "reasoning": {"effort": "low"}},
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
        if len({is_freeform for _, is_freeform in pairs}) == 2:
            break
    if not pairs:
        raise gr.Error(
            f"`{QUESTION_MODEL}` didn't return any usable questions "
            f"(finish reason: {finish_reason}; reply began: {content[:200]!r})."
        )
    yield pairs


# --------------------------------------------------------------------------- #
# What every model has in common
# --------------------------------------------------------------------------- #


class ModelError(Exception):
    """One call to a model failed; the message is short enough to show on the page."""


class Unavailable(Exception):
    """A model can't run at all: a missing key, or an endpoint that won't wake."""


@dataclass
class Answer:
    """One question's result, with the token counts the API reported (None when it didn't).

    `value` is a noul for a yes/no question and text for a freeform one. It is
    None when the call failed or the model declined to answer.
    """

    value: float | str | None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None


def _usage(body: dict) -> dict:
    """The token counts in a response body's usage, as keyword arguments for `Answer`.

    Reads both namings: chat completions' prompt_tokens(_details) and the
    Decisions and System One APIs' input_tokens(_details).
    """
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "cached_tokens": details.get("cached_tokens"),
    }


def _post(http: httpx.Client, url: str, payload: dict, **kwargs) -> dict:
    """POST JSON and return the reply's JSON."""
    try:
        response = http.post(url, json=payload, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise ModelError(f"HTTP {e.response.status_code} {e.response.text.strip()[:300]}")
    except (httpx.HTTPError, ValueError) as e:
        raise ModelError(str(e))


class Racer:
    """One model's way of answering. A new one is made for each run."""

    name = ""
    about = ""  # where it runs, for its timing line
    noul_call = ""  # what one yes/no request is called, for its timing line
    freeform_call = "chat completion"
    yes_no_how = ""  # what answers each kind of question, in a couple of words, for its heading
    freeform_how = ""

    def __init__(self, context: str, stop: threading.Event | None = None) -> None:
        self.context = context
        # Set when the run is stopped (a restart, or the page closing), so the
        # racer stops sending questions and stops waiting for an endpoint.
        self.stop = stop or threading.Event()

    def start(self) -> float:
        """Get ready, before the clock starts. Returns the seconds spent waking an endpoint."""
        return 0.0

    def noul(self, question: str) -> Answer:
        raise NotImplementedError

    def freeform(self, question: str) -> Answer:
        raise NotImplementedError

    def finish(self) -> None:
        """Called once the run is over, however it ended."""


class EndpointWarmer:
    """Wakes and warms a scale-to-zero endpoint once, shared by every visitor.

    `name` is the model's name in messages, `url` the endpoint's /v1 URL, and
    `warm_up` a throwaway call to it.

    States: idle -> checking -> (waking) -> warming -> ready, or error.
    "Waking" means the endpoint answered 503 and is cold-starting. "Warming"
    runs `warm_up` so the first timed request doesn't pay for the new
    connection and first adapter call. "Ready" goes stale READY_TTL_S after
    the last use, ahead of HF's 15-minute scale-down, so the next visitor
    re-checks instead of trusting an endpoint that slept.
    """

    def __init__(self, name: str, url: str, warm_up) -> None:
        self.name, self.url, self.warm_up = name, url, warm_up
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
                if httpx.get(f"{self.url}/models", headers=headers, timeout=10).is_success:
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
            self.warm_up()
        except Exception as e:  # surface any failure in the status line
            self.state, self.detail = "error", f"warm-up call failed: {e}"
            return
        self.detail = f"{time.time() - self.started_at:.0f} s"
        self.touch()
        self.state = "ready"

    def wait_ready(self, stop: threading.Event | None = None) -> float:
        """Block until ready (starting a pass if needed); return seconds waited.

        Gives up when `stop` is set: the run that was waiting has been stopped.
        The wake-up itself carries on, for the next run.
        """
        start = time.time()
        self.ensure()
        while not self.is_ready():
            if stop is not None and stop.is_set():
                raise Unavailable("the run was stopped before the endpoint woke up.")
            if self.state == "error":
                raise Unavailable(f"endpoint unavailable: {self.detail}. Try again shortly.")
            if time.time() - start > WAKE_TIMEOUT_S + 60:
                raise Unavailable("the endpoint didn't wake up in time. Try again shortly.")
            time.sleep(1)
        return time.time() - start

    def status(self) -> str:
        """The endpoint's state in a few words, for the status line."""
        if self.is_ready():
            return f"● {self.name} ready"
        elapsed = time.time() - self.started_at
        return {
            "idle": f"○ {self.name} not checked yet",
            "checking": f"○ {self.name} checking…",
            "waking": f"○ {self.name} waking up, {elapsed:.0f} s so far (a cold start takes 3–6 min)",
            "warming": f"◐ {self.name} up, warming…",
            "error": f"✕ {self.name}: {self.detail}",
            # "ready" but stale: it may have scaled to zero since.
            "ready": f"○ {self.name} idle, re-checks on next use",
        }[self.state]


WARM_UP = ("This is a warm-up request.", "Is this a warm-up request?")

# --------------------------------------------------------------------------- #
# Granite Switch
# --------------------------------------------------------------------------- #

granite_client = OpenAI(base_url=ENDPOINT_URL, api_key=HF_TOKEN)

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


def _granite_chat(messages: list[dict], **kwargs):
    try:
        return granite_client.chat.completions.create(
            model=MODEL_ID, messages=messages, temperature=0.0, **kwargs
        )
    except OpenAIError as e:
        raise ModelError(str(e))


def _granite_usage(response) -> dict:
    usage = response.usage
    details = usage.prompt_tokens_details if usage else None
    return {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "cached_tokens": (details.cached_tokens or 0) if details else 0,
    }


def granite_noul(context: str, question: str) -> Answer:
    """One yes/no question's noul: c(yes), the uncertainty adapter's certainty in a prefilled "Yes".

    Granite never answers the question itself. This is the conversation
    Mellea's ``check_certainty`` sends (question, answer, then the
    ``<certainty>`` invocation), with one change: the adapter's reply is
    prefilled up to the score digit and continued, so vLLM generates a single
    token instead of the full ``{"score": "N"}`` (7 tokens). The certainty is
    decoded from that token's top logprobs the way Mellea's likelihood rule
    does it: keep the digit candidates, renormalize, and take the expected
    value of their mapped certainties.
    """
    response = _granite_chat(
        [
            {"role": "user", "content": _prompt(context, YES_NO_INSTRUCTION, question)},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": UNCERTAINTY_INVOCATION},
            {"role": "assistant", "content": SCORE_PREFIX},
        ],
        max_tokens=1,
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
        (SCORE_VALUES[token.strip()], math.exp(logprob))
        for token, logprob in candidates
        if token.strip() in SCORE_VALUES
    ]
    if not weighted:
        raise ModelError(f"the uncertainty adapter returned no score digit (got {top.token!r})")
    certainty = sum(value * p for value, p in weighted) / sum(p for _, p in weighted)
    return Answer(certainty, **_granite_usage(response))


def granite_freeform(context: str, question: str) -> Answer:
    """One freeform question's answer: a plain chat completion on the base model."""
    response = _granite_chat(
        [{"role": "user", "content": _prompt(context, FREEFORM_INSTRUCTION, question)}],
        max_tokens=FREEFORM_MAX_TOKENS,
    )
    return Answer((response.choices[0].message.content or "").strip(), **_granite_usage(response))


granite_warmer = EndpointWarmer("Granite Switch", ENDPOINT_URL, lambda: granite_noul(*WARM_UP))


class Granite(Racer):
    name = "Granite Switch"
    about = "on its own vLLM endpoint"
    noul_call = f"uncertainty adapter call (`{MODEL_ID}`)"
    freeform_call = "chat completion (base model)"
    yes_no_how, freeform_how = "uncertainty adapter", "base model"

    def start(self) -> float:
        return granite_warmer.wait_ready(self.stop)

    def noul(self, question: str) -> Answer:
        return granite_noul(self.context, question)

    def freeform(self, question: str) -> Answer:
        return granite_freeform(self.context, question)

    def finish(self) -> None:
        granite_warmer.touch()


# --------------------------------------------------------------------------- #
# JEV-9B (its own vLLM endpoint)
# --------------------------------------------------------------------------- #

# JEV-9B's decision head, from its model repo: the token ids of a noul's two
# options ("false", "true"), the head's bias for each, and the temperature that
# calibrates a noul. The bias and temperature are applied here, as the model
# card's recipe does.
_jev9b_head = json.load(open(hf_hub_download(JEV9B_REPO, "adapter_vllm/decision_head.json")))
_noul_slot = _jev9b_head["slots"]["ranges"]["noul"][0]
JEV9B_NOUL_IDS = _jev9b_head["verbalizer_ids"][_noul_slot : _noul_slot + 2]
JEV9B_NOUL_BIAS = _jev9b_head["bias"][_noul_slot : _noul_slot + 2]
JEV9B_NOUL_TEMPERATURE = json.load(open(hf_hub_download(JEV9B_REPO, "calibration.json")))["per_kind"]["noul"]

jev9b_http = httpx.Client(headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=60.0)


def jev9b_noul(context: str, question: str) -> Answer:
    """One yes/no question as a noul from JEV-9B's decision head.

    The model card's recipe: the decision prompt goes to the adapter as a plain
    completion of one token, limited to the two option tokens, and their
    log-probabilities come back. Adding the head's bias and dividing by the
    temperature gives the calibrated distribution; the noul is its "true".
    """
    body = _post(
        jev9b_http,
        f"{JEV9B_ENDPOINT_URL}/completions",
        {
            "model": JEV9B_DECISION_MODEL,
            "prompt": f"[kind] noul\n[state] {context}\n[question] {question}\n[options]\nfalse\ntrue\n[decision]:",
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 2,
            "allowed_token_ids": JEV9B_NOUL_IDS,
            "add_special_tokens": False,
            "return_tokens_as_token_ids": True,
        },
    )
    try:
        top = body["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs = {int(token.split(":")[1]): value for token, value in top.items()}
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ModelError(f"unexpected reply: {e}")
    logits = [
        (logprobs.get(token_id, -1e9) + bias) / JEV9B_NOUL_TEMPERATURE
        for token_id, bias in zip(JEV9B_NOUL_IDS, JEV9B_NOUL_BIAS)
    ]
    weights = [math.exp(z - max(logits)) for z in logits]
    return Answer(weights[1] / sum(weights), **_usage(body))


def jev9b_freeform(context: str, question: str) -> Answer:
    """A short written answer from JEV-9B's base model (no adapter), thinking off."""
    body = _post(
        jev9b_http,
        f"{JEV9B_ENDPOINT_URL}/chat/completions",
        {
            "model": JEV9B_REPO,
            "messages": [{"role": "user", "content": _prompt(context, FREEFORM_INSTRUCTION, question)}],
            "max_tokens": FREEFORM_MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    try:
        return Answer((body["choices"][0]["message"]["content"] or "").strip(), **_usage(body))
    except (KeyError, IndexError, TypeError) as e:
        raise ModelError(f"unexpected reply: {e}")


# None when the Space has no JEV-9B endpoint to call.
jev9b_warmer = (
    EndpointWarmer(JEV9B, JEV9B_ENDPOINT_URL, lambda: jev9b_noul(*WARM_UP)) if JEV9B_ENDPOINT_URL else None
)


class Jev9b(Racer):
    name = JEV9B
    about = "on its own vLLM endpoint"
    noul_call = f"decision adapter call (`{JEV9B_REPO}`)"
    freeform_call = "chat completion (base model, thinking off)"
    yes_no_how, freeform_how = "decision adapter", "base model"

    def start(self) -> float:
        if jev9b_warmer is None:
            raise Unavailable("JEV9B_ENDPOINT_URL is not set on this Space.")
        return jev9b_warmer.wait_ready(self.stop)

    def noul(self, question: str) -> Answer:
        return jev9b_noul(self.context, question)

    def freeform(self, question: str) -> Answer:
        return jev9b_freeform(self.context, question)

    def finish(self) -> None:
        if jev9b_warmer is not None:
            jev9b_warmer.touch()


# --------------------------------------------------------------------------- #
# Gemma Decider (its own vLLM endpoint)
# --------------------------------------------------------------------------- #


class _Decider:
    """decider's prompt code and the model's tokenizer, as scripts/gemma_decider_client.py sets them up.

    The model has no adapter: a decision is the base model's next-token
    distribution over the option letters at the "Answer: (" slot of decider's
    chat-layout prompt, softmaxed at the temperature decider_config.json gives
    for that number of options.
    """

    def __init__(self) -> None:
        try:
            import decider  # noqa: F401
        except ImportError:
            # decider-ai declares torch and other heavy dependencies, none of
            # which the prompt modules use, so install it without them.
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", DECIDER_PACKAGE], check=True
            )
        from decider import systemone, temperature
        from decider.prompt import chat_for, label_table, load_decider_config
        from decider.prompt_fast import build_rows
        from transformers import AutoTokenizer

        config = load_decider_config(GEMMA_REPO)
        self.s1, self.tt, self.build_rows = systemone, temperature, build_rows
        self.tok = AutoTokenizer.from_pretrained(GEMMA_REPO)
        self.chat = chat_for(self.tok, config)  # the chat layout's template head and tail, thinking off
        self.letter_ids = label_table(self.tok)[1]
        (self.temperature, self.temperature_map), _ = temperature.from_config(config)
        self.isolated = bool(config.get("isolated_levels", False))


def _load_decider() -> tuple[_Decider | None, str]:
    """(the decider setup, "") when the endpoint is configured and it loads, else (None, why not)."""
    if not GEMMA_ENDPOINT_URL:
        return None, "GEMMA_DECIDER_ENDPOINT_URL is not set on this Space."
    try:
        return _Decider(), ""
    except Exception as e:  # shown on the page; the rest of the race runs without Gemma
        return None, f"couldn't set up decider's prompt code: {e}"


gemma_decider, GEMMA_UNAVAILABLE = _load_decider()
gemma_http = httpx.Client(headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=60.0)


def gemma_noul(context: str, question: str) -> Answer:
    """One yes/no question as a noul, read the way decider reads it.

    decider's own code builds the prompt rows as token ids, so they match
    decider token for token. Each row is a one-token completion limited to
    the question's option letters, at temperature 0 (the model's generation
    config sets top_k and top_p, which would cut unlikely letters; greedy
    requests skip both). The endpoint returns the letters' log-probabilities
    renormalised over the letters, so a softmax at decider's temperature gives
    the probabilities, and decider's code turns them into the noul.
    """
    d = gemma_decider
    questions = {"q1": d.s1.render_question({"type": "noul", "instructions": question})}
    flat, index = d.s1.plan_rows(questions, d.isolated)
    rows, _ = d.build_rows(
        d.tok,
        d.s1.render_state(context),
        [[(row["question"], list(row["options"]))] for row in flat],
        max_ctx_tokens=GEMMA_MAX_STATE_TOKENS,
        chat=d.chat,
    )
    for row, kind in zip(rows, d.s1.row_types(questions, index)):
        row["types"] = [kind]
    temperatures = d.tt.for_items(d.temperature, d.temperature_map, rows)
    probabilities, prompt_tokens, cached_tokens = [], 0, 0
    for i, row in enumerate(rows):
        letters = d.letter_ids[: row["nopts"][0]]
        body = _post(
            gemma_http,
            f"{GEMMA_ENDPOINT_URL}/completions",
            {
                "model": GEMMA_REPO,
                "prompt": row["ids"],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": len(letters),
                "allowed_token_ids": letters,
                "return_tokens_as_token_ids": True,
            },
        )
        try:
            top = body["choices"][0]["logprobs"]["top_logprobs"][0]
            by_id = {int(token.split(":")[1]): value for token, value in top.items()}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ModelError(f"unexpected reply: {e}")
        logprobs = [by_id.get(token_id, -1e9) for token_id in letters]
        t = temperatures[i][0] if isinstance(temperatures, list) else temperatures
        weights = [math.exp((x - max(logprobs)) / t) for x in logprobs]
        probabilities.append([w / sum(weights) for w in weights])
        usage = _usage(body)
        prompt_tokens += usage["prompt_tokens"] or 0
        cached_tokens += usage["cached_tokens"] or 0
    try:
        noul = d.s1.assemble(questions, index, probabilities)["q1"]["noul"]
    except (KeyError, TypeError) as e:
        raise ModelError(f"couldn't read the decision: {e}")
    return Answer(noul, prompt_tokens, cached_tokens)


def gemma_freeform(context: str, question: str) -> Answer:
    """A one-word answer from the same model as an ordinary chat completion, thinking off."""
    body = _post(
        gemma_http,
        f"{GEMMA_ENDPOINT_URL}/chat/completions",
        {
            "model": GEMMA_REPO,
            "messages": [{"role": "user", "content": _prompt(context, FREEFORM_INSTRUCTION, question)}],
            "max_tokens": FREEFORM_MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    try:
        return Answer((body["choices"][0]["message"]["content"] or "").strip(), **_usage(body))
    except (KeyError, IndexError, TypeError) as e:
        raise ModelError(f"unexpected reply: {e}")


# None when Gemma can't be called (no endpoint URL, or decider didn't load).
gemma_warmer = (
    EndpointWarmer("Gemma Decider", GEMMA_ENDPOINT_URL, lambda: gemma_noul(*WARM_UP))
    if gemma_decider is not None
    else None
)


class Gemma(Racer):
    name = GEMMA
    about = "on its own vLLM endpoint"
    noul_call = f"decider readout call (`{GEMMA_REPO}`)"
    freeform_call = "chat completion (same model, thinking off)"
    yes_no_how, freeform_how = "decider readout", "gemma-4-31b"

    def start(self) -> float:
        if gemma_warmer is None:
            raise Unavailable(GEMMA_UNAVAILABLE)
        return gemma_warmer.wait_ready(self.stop)

    def noul(self, question: str) -> Answer:
        return gemma_noul(self.context, question)

    def freeform(self, question: str) -> Answer:
        return gemma_freeform(self.context, question)

    def finish(self) -> None:
        if gemma_warmer is not None:
            gemma_warmer.touch()


# --------------------------------------------------------------------------- #
# The OpenRouter models: GPT Luna, and Jev with Qwen
# --------------------------------------------------------------------------- #

# One client, so every request after the first reuses a connection.
openrouter_http = httpx.Client(timeout=60.0)


def _openrouter(url: str, payload: dict) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    return _post(openrouter_http, url, payload, headers={"Authorization": f"Bearer {api_key}"})


def openrouter_noul(model: str, context: str, question: str) -> Answer:
    """One yes/no question as a noul from a decision model, through the System One API.

    A question the model declines comes back without a noul, and the answer's
    value is None.
    """
    body = _openrouter(
        OPENROUTER_SYSTEM_ONE_URL,
        {"model": model, "state": context, "questions": {"q1": {"type": "noul", "instructions": question}}},
    )
    try:
        return Answer(body["answers"]["q1"].get("noul"), **_usage(body))
    except (KeyError, TypeError, AttributeError) as e:
        raise ModelError(f"unexpected reply: {e}")


def openrouter_freeform(model: str, context: str, question: str, **options) -> Answer:
    """A short written answer from a chat completion. `options` are added to the request."""
    body = _openrouter(
        OPENROUTER_CHAT_URL,
        {
            "model": model,
            "messages": [{"role": "user", "content": _prompt(context, FREEFORM_INSTRUCTION, question)}],
            **options,
        },
    )
    try:
        return Answer((body["choices"][0]["message"]["content"] or "").strip(), **_usage(body))
    except (KeyError, IndexError, TypeError) as e:
        raise ModelError(f"unexpected reply: {e}")


class OpenRouterRacer(Racer):
    """A decision model for yes/no questions and a chat model for freeform ones."""

    about = "on OpenRouter"
    noul_model = ""
    freeform_model = ""
    freeform_options: dict = {}  # added to every freeform request

    def start(self) -> float:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise Unavailable("OPENROUTER_API_KEY is not set on this Space.")
        return 0.0

    def noul(self, question: str) -> Answer:
        return openrouter_noul(self.noul_model, self.context, question)

    def freeform(self, question: str) -> Answer:
        return openrouter_freeform(
            self.freeform_model, self.context, question,
            max_tokens=FREEFORM_MAX_TOKENS, **FASTEST, **NO_REASONING, **self.freeform_options,
        )


def _openrouter_racer(
    racer_name: str, decisions: str, writer: str, options: dict | None = None, options_note: str = ""
) -> type[OpenRouterRacer]:
    """A racer for a decision model and a chat model. `options` go on its freeform requests."""

    class Pair(OpenRouterRacer):
        name = racer_name
        noul_model, freeform_model = decisions, writer
        freeform_options = options or {}
        noul_call = f"decision request (`{decisions}`)"
        freeform_call = f"chat completion (`{writer}`, fastest provider, reasoning off{options_note})"
        yes_no_how, freeform_how = decisions.split("/")[-1], writer.split("/")[-1]

    return Pair


# GPT Luna is two model ids on OpenRouter: one for decisions, one for text.
# Its text model runs on OpenAI's priority tier, the fastest it offers.
Luna = _openrouter_racer("GPT Luna", LUNA_DECISIONS_MODEL, LUNA_MODEL, PRIORITY, ", priority tier")
# Jev returns decisions only, so a partner model does the writing it can't.
JevTeam = _openrouter_racer("Jev + Qwen", JEV_MODEL, JEV_PARTNER_MODEL)

# The order of the board's columns.
RACERS = [Granite, Jev9b, Gemma, Luna, JevTeam]

# --------------------------------------------------------------------------- #
# Running the course
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    """One model's trip through the questions, filled in as it goes."""

    name: str
    answers: list  # one per question: an Answer, or None until it has come back
    started: float | None = None  # perf_counter() when the clock started
    total_s: float | None = None  # set when the model has been through every question
    wake_s: float = 0.0  # seconds spent waiting for an endpoint, before the clock started
    unavailable: str | None = None  # why the model couldn't run
    errors: list[str] = field(default_factory=list)
    # Seconds each question took, in order, and the question in progress as
    # (index, perf_counter() when it was sent). The time bar is drawn from these.
    durations: list[float] = field(default_factory=list)
    current: tuple[int, float] | None = None

    @property
    def finished(self) -> bool:
        return self.total_s is not None and self.unavailable is None

    def states(self) -> list[str]:
        """Each question's state on the progress strip: todo, running, done or out."""
        if self.unavailable is not None:
            return ["out"] * len(self.answers)
        states = ["todo" if answer is None else "done" for answer in self.answers]
        if self.started is not None and "todo" in states:
            states[states.index("todo")] = "running"
        return states


def run_course(racer: Racer, questions: list[tuple[str, bool]], run: Run) -> None:
    """Take one model through the questions, one at a time, in order.

    A yes/no question is one `noul` call and a freeform one is one `freeform`
    call; the next isn't sent until the last has come back. Each answer is
    written into `run` as it arrives, so the page can show it. A failed call
    leaves an empty answer and the run goes on. When the run is stopped, the
    model finishes the question it's on and sends no more.
    """
    try:
        run.wake_s = racer.start()
    except Unavailable as e:
        run.unavailable = str(e)
        return
    start = run.started = time.perf_counter()
    try:
        for i, (question, is_freeform) in enumerate(questions):
            if racer.stop.is_set():
                return
            sent = time.perf_counter()
            run.current = (i, sent)
            try:
                run.answers[i] = (racer.freeform if is_freeform else racer.noul)(question)
            except ModelError as e:
                run.answers[i] = Answer(None)
                run.errors.append(str(e))
            run.durations.append(time.perf_counter() - sent)
            run.current = None
        run.total_s = time.perf_counter() - start
    finally:
        racer.finish()
    if len(run.errors) == len(questions):
        run.unavailable = f"calls failed: {run.errors[0]}"


# --------------------------------------------------------------------------- #
# Drawing the page
# --------------------------------------------------------------------------- #

YES_NO_ICON, FREEFORM_ICON = "⚡", "💬"


def _icon(is_freeform: bool) -> str:
    return FREEFORM_ICON if is_freeform else YES_NO_ICON


def _short(model_id: str) -> str:
    """A model id without its vendor: openai/gpt-6-luna -> gpt-6-luna."""
    return model_id.split("/")[-1]


def _fmt_time(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _article_html(article: dict | None, message: str = "") -> str:
    """The Article panel: the title in a file-style row, with the text folded away under it.

    With no article yet, the row shows `message` in the title's place.
    """
    head = '<div class="sd-panel sd-article"><div class="sd-panel-title">Article (Context/State)</div>'
    if article is None:
        return f'{head}<div class="sd-file"><span class="sd-chip">WIKI</span><span class="sd-muted">{html.escape(message)}</span></div></div>'
    size = f"{len(article['text']):,} characters"
    if len(article["text"]) < article["full_chars"]:
        size += f", the first part of {article['full_chars']:,}"
    return (
        f'{head}<div class="sd-file"><span class="sd-chip">WIKI</span><div class="sd-file-name">'
        f'<a href="{html.escape(article["link"])}" target="_blank" rel="noopener" title="Open on Wikipedia">'
        f'{html.escape(article["title"])}</a><span>{size}</span></div></div>'
        "<details><summary>Read the article</summary>"
        f'<div class="sd-article-text">{html.escape(article["text"])}</div></details></div>'
    )


def _model_heading(racer: type[Racer], run: Run) -> str:
    """A model's column heading: its name, with what answers each kind of question as a tooltip."""
    tip = f"{YES_NO_ICON} yes/no: {racer.yes_no_how} · {FREEFORM_ICON} freeform: {racer.freeform_how}"
    return (
        f'<div class="sd-head{" out" if run.unavailable is not None else ""}" title="{html.escape(tip)}">'
        f"<b>{html.escape(run.name)}</b></div>"
    )


def _answer_html(run: Run, i: int, state: str) -> str:
    """One model's answer to question `i`: a noul with Yes or No after it, or the text it wrote."""
    if state != "done":
        return f'<div class="sd-cell {state}">{"…" if state == "running" else ""}</div>'
    value = run.answers[i].value
    if value is None:
        return '<div class="sd-cell done sd-muted" title="No answer">—</div>'
    if isinstance(value, str):
        return f'<div class="sd-cell done">{html.escape(value)}</div>'
    return (
        f'<div class="sd-cell done sd-noul" title="Probability that the answer is yes: {value:.3f}">'
        f'{value:.2f} ({"Yes" if value > 0.5 else "No"})</div>'
    )


# The time bars' scale grows through these, so a bar visibly lengthens as the
# run goes on instead of every bar being rescaled to the slowest model.
TIME_SCALES_S = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600)


def _segments(run: Run, questions: list[tuple[str, bool]]) -> list[tuple[bool, float, bool]]:
    """A model's time so far, question by question: (is_freeform, seconds, still running)."""
    segments = [(questions[i][1], seconds, False) for i, seconds in enumerate(run.durations)]
    current = run.current
    if current is not None:
        index, sent = current
        segments.append((questions[index][1], time.perf_counter() - sent, True))
    return segments


MEDALS = ("🥇", "🥈", "🥉")


def _places(runs: list[Run]) -> dict[str, int]:
    """Each finished model's place, 0 for the fastest, by its end-to-end time."""
    finished = sorted((run.total_s, run.name) for run in runs if run.finished)
    return {name: place for place, (_, name) in enumerate(finished)}


def _time_bars_html(runs: list[Run], questions: list[tuple[str, bool]]) -> str:
    """A bar per model that lengthens as its run goes on.

    Each question is a segment, in the order the model took them, coloured by
    kind to match its icon: decision time for yes/no questions, chat time for
    freeform ones. Beside the bar are the model's total time and its total
    cache reuse (see `_cache_totals`). A model that has finished gets a medal
    for its place. All bars share one
    scale, which steps up through TIME_SCALES_S as the slowest model passes
    each mark.

    The bars are in finishing order: as each model finishes it moves up under
    the ones that finished before it, so first place is always on top. Models
    still running follow in the usual order, and one that couldn't run is last.
    """
    all_segments = [_segments(run, questions) for run in runs]
    longest = max((sum(seconds for _, seconds, _ in segments) for segments in all_segments), default=0)
    scale = next((s for s in TIME_SCALES_S if s >= longest), longest)
    places = _places(runs)

    def standing(k: int) -> tuple[int, int]:
        run = runs[k]
        if run.name in places:
            return (0, places[run.name])
        return (2 if run.unavailable is not None else 1, k)

    rows = []
    for k in sorted(range(len(runs)), key=standing):
        run, segments = runs[k], all_segments[k]
        bar = "".join(
            f'<i class="{"chat" if is_freeform else "decide"}{" live" if live else ""}" '
            f'style="width: {100 * seconds / scale:.2f}%" '
            f'title="{"Chat" if is_freeform else "Decision"}: {_fmt_time(seconds)}"></i>'
            for is_freeform, seconds, live in segments
        )
        total = sum(seconds for _, seconds, _ in segments)
        place = places.get(run.name)
        medal = MEDALS[place] if place is not None and place < len(MEDALS) else ""
        if run.unavailable is not None:
            totals, state = "N/A", " out"
        else:
            cached, prompt, reported, answered = _cache_totals(run)
            if reported and prompt:
                tip = f"{cached:,} of {prompt:,} prompt tokens read from cache"
                if reported < answered:
                    tip += f", over the {reported} of {answered} requests whose API reports it"
                cache = f'<span class="cache" title="{tip}">{cached / prompt:.0%} cached</span>'
            elif answered:
                cache = '<span class="cache" title="This API doesn\'t report cache use">cache not reported</span>'
            else:
                cache = '<span class="cache">—</span>'
            totals = f"<b>{total:.1f} s</b>{cache}"
            state = " winner" if place == 0 else ""
        rows.append(
            f'<div class="sd-tl-name{state}"><span class="sd-medal">{medal}</span>{html.escape(run.name)}</div>'
            f'<div class="sd-tl-track{state}">{bar}</div><div class="sd-tl-totals{state}">{totals}</div>'
        )
    ticks = "".join(
        f'<span style="left: {100 * k / 5:.0f}%">{scale * k / 5:g} s</span>' for k in range(6)
    )
    return (
        '<div class="sd-panel sd-timeline"><div class="sd-panel-head"><div class="sd-panel-title">Total Latency (lower is better)</div>'
        '<div class="sd-tl-legend">'
        f'<span><i class="decide"></i>{YES_NO_ICON} decisions</span>'
        f'<span><i class="chat"></i>{FREEFORM_ICON} chat</span></div></div>'
        f'<div class="sd-tl-grid">{"".join(rows)}'
        f'<div></div><div class="sd-tl-ticks">{ticks}</div><div></div>'
        '<div></div><div class="sd-tl-axis">Time spent</div><div></div></div></div>'
    )


def _board_html(
    runs: list[Run] | None, questions: list[tuple[str, bool]], note: str = ""
) -> str:
    """The time bars, then the table: a row per question, a column per model.

    Each model's column fills downwards as its answers arrive, and the
    question it's working on is highlighted. `runs` is None before the first
    run, and `note` is shown under the last question while the questions are
    still being written.
    """
    if runs is None:
        runs = [Run(racer.name, [None] * len(questions)) for racer in RACERS]
    count = f"{len(questions)} question{'' if len(questions) == 1 else 's'}" if questions else "Questions"
    corner_tip = f"{YES_NO_ICON} yes/no question · {FREEFORM_ICON} one-word brainstorm"
    cells = [f'<div class="sd-head sd-corner" title="{corner_tip}"><b>{count}</b></div>']
    cells += [_model_heading(racer, run) for racer, run in zip(RACERS, runs)]
    states = [run.states() for run in runs]
    for i, (question, is_freeform) in enumerate(questions):
        cells.append(
            f'<div class="sd-question"><span class="sd-icon" title="{"freeform" if is_freeform else "yes/no"}">'
            f"{_icon(is_freeform)}</span><span>{html.escape(question)}</span></div>"
        )
        cells += [_answer_html(run, i, run_states[i]) for run, run_states in zip(runs, states)]
    if note:
        cells.append(f'<div class="sd-board-note">{html.escape(note)}</div>')
    return _time_bars_html(runs, questions) + (
        '<div class="sd-panel"><div class="sd-panel-title">Answers</div>'
        f'<div class="sd-board-wrap"><div class="sd-board" style="--sd-models: {len(runs)}">'
        f'{"".join(cells)}</div></div></div>'
    )


def _plural(count: int, noun: str) -> str:
    """A count and its noun, as in 3 adapter calls. The s goes before any bracketed note."""
    head, sep, tail = noun.partition(" (")
    return f"{count} {head}{'' if count == 1 else 's'}{sep}{tail}"


def _cache_totals(run: Run) -> tuple[int, int, int, int]:
    """A model's cache reuse so far: (cached tokens, prompt tokens, requests that report it, requests answered).

    Only the requests whose reply reports a cache count are counted:
    OpenRouter's decision requests (Jev's and Luna Decisions') don't.
    """
    answered = [a for a in run.answers if a is not None and a.prompt_tokens is not None]
    reported = [a for a in answered if a.cached_tokens is not None]
    return (
        sum(a.cached_tokens for a in reported),
        sum(a.prompt_tokens for a in reported),
        len(reported),
        len(answered),
    )


def _cache_note(run: Run) -> str:
    """The model's total cache reuse, in words, for its timing line. Shown even when it's 0."""
    cached, prompt, reported, answered = _cache_totals(run)
    if not reported:
        return "cache reuse not reported by the API"
    share = f" ({cached / prompt:.0%})" if prompt else ""
    note = f"{cached:,} of {prompt:,} prompt tokens came from cache{share}"
    if reported < answered:
        note += f", over the {reported} of {answered} requests whose API reports it"
    return note


def _timing_line(racer: Racer, run: Run, questions: list[tuple[str, bool]]) -> str:
    """One model's line in the details: how long it took and how it was called."""
    if run.unavailable is not None:
        return f"**{run.name}:** unavailable · {run.unavailable}"
    n_freeform = sum(is_freeform for _, is_freeform in questions)
    line = (
        f"**{run.name}:** {_fmt_time(run.total_s)} end to end for {len(questions)} question(s), {racer.about}"
        + (f" · waited {run.wake_s:.0f} s for the endpoint to wake (not counted)" if run.wake_s > 5 else "")
        + f"  \n{_plural(len(questions) - n_freeform, racer.noul_call)} for {YES_NO_ICON} yes/no, "
        f"{_plural(n_freeform, racer.freeform_call)} for {FREEFORM_ICON} freeform · {_cache_note(run)}"
    )
    if run.errors:
        line += f" · {len(run.errors)} call(s) failed, first: {run.errors[0]}"
    return line


# --------------------------------------------------------------------------- #
# The two things the page does: set up a round, and run it
# --------------------------------------------------------------------------- #

OFF, ON = {"interactive": False}, {"interactive": True}
RUN_LABEL = "▶ Run the models"
RESTART_LABEL = "↻ Restart"  # the run button while a run is going


def new_round():
    """Pull a random article and have questions written about it.

    Runs when the page loads and on "New article". The article's title shows
    first, then the questions as they're written; once they're all in they
    are put in a random order and the run button is switched on.

    Outputs: article card, board, the round (for `run_models`), the run
    button, and the timing lines, which are cleared.
    """
    yield (
        _article_html(None, "Finding a random article…"), _board_html(None, []), None,
        gr.update(value=RUN_LABEL, **OFF), "",
    )
    total = NUM_YES_NO + NUM_FREEFORM
    try:
        article = random_article()
        card = _article_html(article)
        writing = f"{_short(QUESTION_MODEL)} is writing {total} questions…"
        yield card, _board_html(None, [], note=writing), None, gr.skip(), gr.skip()
        questions: list = []
        for questions in generate_questions(article["text"]):
            yield card, _board_html(None, questions, note=writing), None, gr.skip(), gr.skip()
    except gr.Error:
        failed = _article_html(None, "Couldn't set up a round. Click New article to try again.")
        yield failed, gr.skip(), None, gr.skip(), gr.skip()
        raise
    random.shuffle(questions)
    round_ = {"context": article["text"], "questions": questions}
    yield card, _board_html(None, questions), round_, gr.update(**ON), gr.skip()


STOPWATCH_TICK_S = 0.1  # how often the board is redrawn during a run


# The stop signal of each browser session's run in progress, so a restart can
# end it. Keyed by Gradio's session hash.
_session_runs: dict[str, threading.Event] = {}
_session_runs_lock = threading.Lock()


def stop_run(request: gr.Request) -> None:
    """Stop this session's run, if one is going.

    Runs outside the queue when either button is clicked, ahead of the new
    round or run, so a restart never waits for the old run to finish. Each
    model finishes the question it's on and stops; its column stops updating.
    """
    with _session_runs_lock:
        stop = _session_runs.pop(request.session_hash, None)
    if stop is not None:
        stop.set()


def run_models(round_: dict | None, request: gr.Request):
    """Give the questions to every model and show the answers as they arrive.

    The models run side by side, each one question at a time. While they
    run, each model's time bar grows and its column of answers fills in, and
    the run button reads "Restart". Once all have finished, the timing lines
    are filled in.

    Either button stops the run (see `stop_run`): this generator then returns
    without touching the page again, so it can't draw over what replaced it.
    Gradio closing the generator (the page was closed) stops the run too.

    Outputs: board, timing lines, endpoint status, run button.
    """
    if not round_:
        raise gr.Error("The questions aren't ready yet.")
    context, questions = round_["context"], round_["questions"]
    stop = threading.Event()
    with _session_runs_lock:
        previous = _session_runs.get(request.session_hash)
        _session_runs[request.session_hash] = stop
    if previous is not None:
        previous.set()
    racers = [racer(context, stop) for racer in RACERS]
    runs = [Run(racer.name, [None] * len(questions)) for racer in racers]
    keep = gr.skip()

    pool = ThreadPoolExecutor(max_workers=len(racers))
    try:
        futures = [pool.submit(run_course, racer, questions, run) for racer, run in zip(racers, runs)]
        yield _board_html(runs, questions), "", keep, gr.update(value=RESTART_LABEL)
        while not all(future.done() for future in futures):
            time.sleep(STOPWATCH_TICK_S)
            if stop.is_set():
                return
            yield _board_html(runs, questions), keep, keep, keep
        for future in futures:
            future.result()  # surface an unexpected failure
    finally:
        # A stopped run's models finish their current question in the background.
        stop.set()
        pool.shutdown(wait=False)
        with _session_runs_lock:
            if _session_runs.get(request.session_hash) is stop:
                del _session_runs[request.session_hash]
    timing = "\n\n".join(_timing_line(racer, run, questions) for racer, run in zip(racers, runs))
    yield _board_html(runs, questions), timing, endpoints_status(), gr.update(value=RUN_LABEL)


def _warmers() -> list[EndpointWarmer]:
    """The scale-to-zero endpoints this Space can call."""
    return [w for w in (granite_warmer, jev9b_warmer, gemma_warmer) if w is not None]


def endpoints_status() -> str:
    """The Endpoints panel: a line per scale-to-zero endpoint, with a dot for its state."""
    lines = []
    for warmer in _warmers():
        if warmer.is_ready():
            level = "ready"
        elif warmer.state == "error":
            level = "error"
        elif warmer.state in ("checking", "waking", "warming"):
            level = "waking"
        else:
            level = "idle"
        # status() starts with its own symbol (● ○ ◐ ✕); the dot replaces it.
        lines.append(f'<li class="{level}"><i></i><span>{html.escape(warmer.status()[2:])}</span></li>')
    return f'<div class="sd-panel"><div class="sd-panel-title">Endpoints</div><ul class="sd-endpoints">{"".join(lines)}</ul></div>'


def on_page_load() -> str:
    """Start waking the endpoints as soon as someone opens the page."""
    for warmer in _warmers():
        warmer.ensure()
    return endpoints_status()


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

# The IBM logo in the top bar, as in IBM's Granite Vision demo (from that demo's assets).
IBM_LOGO = (Path(__file__).parent / "ibm-logo.svg").read_text()
BRAND = (
    f'<div class="sd-brand"><span class="sd-logo">{IBM_LOGO}</span><i class="sd-divider"></i>'
    '<b>Decisions Switching Costs</b><span class="sd-sub">Test how quickly models can move between snap decisions and natural language tasks</span></div>'
)
# Opens the About panel: a label for its checkbox, so it works without a round trip.
INFO_BUTTON = (
    '<label for="sd-about" class="sd-icon-btn" title="About this demo">'
    '<svg viewBox="0 0 32 32" width="18" height="18" aria-hidden="true">'
    '<circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" stroke-width="2"/>'
    '<path d="M16 14v9M16 9.5v1" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg></label>'
)


INTRO = (
    '<p class="sd-intro">Agents need to switch between quick decisions (is this a refund request? should '
    "this go to billing?) and natural-language tasks like tool-calling and answering. Each switch can cost "
    "time: another model, another request, a context read again from scratch. Granite Switch lets one model "
    "efficiently flip between deciding and chatting, reusing its prefix cache at every step. Test it out "
    "yourself, see how quickly models can switch between yes/no and freeform questions generated from a "
    "random Wikipedia article.</p>"
)


def _about_html() -> str:
    """The About panel, shown over the page from the info button. It starts closed.

    A hidden checkbox holds whether it's open; the info button, the close
    button and the backdrop are all labels for it.
    """
    question_types = [
        (YES_NO_ICON, "Yes/no decision", "Comes back as a noul: the probability that the answer is yes"),
        (FREEFORM_ICON, "One-word brainstorm", "Asks the model to come up with a word; reasoning off"),
    ]
    models = [
        ("GS", "Granite Switch", "Uncertainty aLoRA for decisions, base model for words"),
        ("J9", JEV9B, "Open reproduction of Jev: decision adapter plus base model"),
        ("GD", GEMMA, "Gemma 4 31B read through decider's answer-letter prompt"),
        ("GL", "GPT Luna", "Luna Decisions plus GPT-6 Luna on the priority tier"),
        ("JQ", "Jev + Qwen", "TypeSafe's Jev for decisions, Qwen3.8 27B for words"),
    ]

    def items(rows: list[tuple[str, str, str]], mono: bool = False) -> str:
        return "".join(
            f'<div class="sd-about-item"><span class="sd-about-icon{" mono" if mono else ""}">{icon}</span>'
            f"<div><b>{html.escape(title)}</b><p>{html.escape(text)}</p></div></div>"
            for icon, title, text in rows
        )

    pills = "".join(f"<span>{p}</span>" for p in ("Granite Switch", "aLoRA", "vLLM", "OpenRouter", "Wikipedia"))
    return (
        '<input type="checkbox" id="sd-about" class="sd-about-toggle">'
        '<div class="sd-about"><label for="sd-about" class="sd-about-backdrop" aria-label="Close"></label>'
        '<div class="sd-about-panel"><div class="sd-about-body"><label for="sd-about" class="sd-about-close" title="Close">✕</label>'
        "<h2>Decisions Switching Costs</h2><h3>with Granite Switch</h3>"
        "<p class=\"sd-about-lede\">Some questions want a fast yes/no call. Others want a word. Five models "
        "take a mix of both about a random Wikipedia article, one question at a time, starting with "
        f"<code>{html.escape(MODEL_ID.split('/')[-1])}</code>. Which gets through first?</p>"
        f'<div class="sd-about-section">Question types</div><div class="sd-about-grid">{items(question_types)}</div>'
        f'<div class="sd-about-section">The models</div><div class="sd-about-grid">{items(models, mono=True)}</div>'
        f'<div class="sd-pills">{pills}</div></div></div></div>'
    )


NOTES = (
    f"**The article** is the plain text of a random English Wikipedia article, cut to the "
    f"first {MAX_CONTEXT_CHARS:,} characters when it's longer. **The questions** are written "
    f"by `{QUESTION_MODEL}` on OpenRouter.\n\n"
    f"**{YES_NO_ICON} Yes/no** questions come back as a noul: the probability that the answer "
    "is yes. The board shows it with Yes (above 0.5) or No after it, as in 0.97 (Yes). "
    f"**{FREEFORM_ICON} Freeform** questions are brainstorming questions: they ask the model "
    "to come up with a fitting word, label or name, and the model is told to answer in one "
    f"word (up to {FREEFORM_MAX_TOKENS} tokens), with reasoning off. Each model takes the questions one at a time, in order. "
    "Nothing is scored.\n\n"
    f"**Granite Switch:** `{MODEL_ID}`, served by vLLM on {GRANITE_HARDWARE}, on a Hugging "
    "Face Inference Endpoint. Its noul is c(yes), the uncertainty adapter's certainty in a "
    "prefilled 'Yes', which always falls between 0.05 and 0.95. Freeform questions go to "
    "its base model.\n\n"
    f"**{JEV9B}:** [`{JEV9B_REPO}`](https://huggingface.co/{JEV9B_REPO}), AutoTrust's open "
    f"reproduction of Jev (not a TypeSafe model), served by vLLM on {JEV9B_HARDWARE} on its "
    "own Inference Endpoint. Its noul is its decision adapter's probability of 'true'. "
    "Freeform questions go to its base model.\n\n"
    f"**{GEMMA}:** [`{GEMMA_REPO}`](https://huggingface.co/{GEMMA_REPO}), the unmodified "
    f"Gemma 4 31B with a decider config, served by vLLM on {GEMMA_HARDWARE} on its own "
    "Inference Endpoint. There is no adapter: its noul is the model's own next-token "
    "probability for the answer letter, read with decider's prompt and temperature. "
    "Freeform questions go to the same model as a chat completion.\n\n"
    f"**{Luna.name}:** on OpenRouter. Yes/no questions go to `{LUNA_DECISIONS_MODEL}` and "
    f"freeform questions to `{LUNA_MODEL}` on OpenAI's priority tier, its fastest. OpenAI's "
    "Decisions API has no priority tier.\n\n"
    f"**{JevTeam.name}:** on OpenRouter. Jev (`{JEV_MODEL}`), TypeSafe's System One model, "
    f"returns decisions only, so `{JEV_PARTNER_MODEL}` writes the freeform answers. Neither "
    "has a priority tier on OpenRouter.\n\n"
    "**OpenRouter** sends every chat request to the fastest provider it has for that model "
    "(`provider.sort` set to `throughput`).\n\n"
    "**Race:** a bar per model, a segment per question in the order it took them, with its "
    "total time and total cache reuse beside it: "
    f"{YES_NO_ICON} decision time for yes/no questions and {FREEFORM_ICON} chat time for "
    "freeform ones.\n\n"
    "**Timing:** all times are full round trips. The three endpoints scale to zero after 15 "
    "idle minutes; opening this page starts waking them, and any wait isn't counted."
)

# IBM Carbon's look, as in IBM's Granite Vision demo: IBM Plex type, a cool-gray
# frame, white panels with a thin border, black and white pill buttons. Light
# mode only, like that demo. Selectors are our own classes, plus Gradio's
# footer, which that demo hides too.
CSS = """
:root {
  --sd-text: #161616; --sd-text-2: #525252; --sd-text-3: #6f6f6f;
  --sd-border: #dde1e6; --sd-layer: #f2f4f8; --sd-blue: #0f62fe;
  --sd-decide: #f1c21b; --sd-decide-text: #8e6a00; --sd-chat: #a2a9b0; --sd-chat-text: #697077;
  --sd-run-bg: #edf5ff; --sd-win: #24a148;
}
footer { display: none !important; }
.gradio-container { padding: 0 !important; }
/* Gradio pads every HTML block; the panels bring their own padding. */
.sd-flush .html-container { padding: 0 !important; }
/* The frame and its top bar. */
.sd-frame { background: var(--sd-layer); border-radius: 12px; padding: 0 12px 12px !important; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06); }
.sd-topbar { flex-wrap: nowrap !important; align-items: center !important; gap: 8px !important; padding: 10px 0 2px; }
.sd-topbar > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.sd-topbar > .sd-brand-wrap { flex: 1 1 auto !important; }
button.sd-pill {
  border-radius: 999px !important; height: 32px; min-width: 0 !important; padding: 0 16px !important;
  font-size: 14px !important; font-weight: 500 !important; box-shadow: none !important;
  background: #fff !important; color: var(--sd-text) !important; border: 1px solid var(--sd-border) !important;
}
button.sd-pill:hover { background: var(--sd-layer) !important; }
button.sd-pill.sd-dark { background: #161616 !important; color: #fff !important; border-color: #161616 !important; }
button.sd-pill.sd-dark:hover { background: #393939 !important; }
button.sd-pill:disabled { opacity: 0.35; cursor: not-allowed; }
.sd-body { gap: 12px !important; align-items: flex-start !important; }
.sd-side, .sd-main { gap: 12px !important; }
.sd-acc { border-radius: 12px !important; }

/* Everything drawn inside the HTML blocks. Nested under .gradio-container .prose so these rules
   outrank Gradio's own `.gradio-container-<version> .prose *` text colour, whichever stylesheet
   loads last (with server-side rendering, as on Spaces, Gradio's loads after this one). */
.gradio-container .prose {
  /* Gradio colours every element itself; let ours inherit from their parent instead,
     unless a rule below says otherwise. */
  .sd-panel *, .sd-about *, .sd-brand *, .sd-intro * { color: inherit; }
  .sd-intro { margin: 0 0 4px; max-width: 110ch; font-size: 14px; line-height: 1.55; color: var(--sd-text-2); }
  .sd-muted { color: var(--sd-text-3); }
  .sd-brand { display: flex; align-items: center; gap: 10px; font-size: 16px; white-space: nowrap; overflow: hidden; }
  .sd-logo { display: flex; flex: none; color: #161616; }
  .sd-logo svg { display: block; }
  .sd-divider { flex: none; width: 1px; height: 20px; margin: 0 4px; background: #c1c7cd; }
  .sd-brand b { font-weight: 600; }
  .sd-brand .sd-sub { color: var(--sd-text-2); font-weight: 300; overflow: hidden; text-overflow: ellipsis; }
  .sd-icon-btn {
    display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 50%;
    background: #fff; border: 1px solid var(--sd-border); color: var(--sd-text-2); cursor: pointer;
  }
  .sd-icon-btn:hover { background: var(--sd-layer); color: var(--sd-text); }
  /* Panels. */
  .sd-panel { background: #fff; border: 1px solid var(--sd-border); border-radius: 12px; padding: 14px 16px 16px; }
  .sd-panel + .sd-panel { margin-top: 12px; }
  .sd-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
  .sd-panel-title { font-size: 14px; font-weight: 600; color: var(--sd-text); margin-bottom: 10px; }
  .sd-panel-head .sd-panel-title { margin-bottom: 0; }
  /* Article panel: a file-style row, then the folded text. */
  .sd-file { display: flex; align-items: center; gap: 12px; padding: 12px; border-radius: 8px; background: var(--sd-layer); }
  .sd-chip { flex: none; padding: 2px 8px; border-radius: 999px; background: #fff; font: 600 11px/1.6 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: 0.04em; color: var(--sd-text-2); }
  .sd-file-name { display: flex; flex-direction: column; align-items: flex-start; min-width: 0; text-align: left; }
  .sd-file-name a { font-size: 15px; font-weight: 600; color: var(--sd-text); text-decoration: none; line-height: 1.3; text-align: left !important; }
  .sd-file-name a:hover { color: var(--sd-blue); text-decoration: underline; }
  .sd-file-name span { font-size: 12px; color: var(--sd-text-3); }
  .sd-article details { margin-top: 10px; }
  .sd-article summary { cursor: pointer; font-size: 13px; font-weight: 500; color: var(--sd-blue); }
  .sd-article-text {
    margin-top: 8px; max-height: 360px; overflow-y: auto; white-space: pre-wrap;
    font-size: 13px; line-height: 1.55; color: var(--sd-text-2);
  }
  /* Endpoints panel. */
  .sd-endpoints { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .sd-endpoints li { display: flex; gap: 8px; align-items: baseline; font-size: 13px; line-height: 1.35; color: var(--sd-text-2); }
  .sd-endpoints i { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--sd-chat); transform: translateY(-1px); }
  .sd-endpoints li.ready i { background: var(--sd-win); }
  .sd-endpoints li.waking i { background: var(--sd-decide); }
  .sd-endpoints li.error i { background: #da1e28; }
  /* Answers panel: a row per question, a column per model. */
  .sd-board-wrap { overflow-x: auto; }
  .sd-board {
    display: grid; min-width: 820px; column-gap: 12px;
    grid-template-columns: minmax(220px, 1.7fr) repeat(var(--sd-models), minmax(110px, 1fr));
  }
  .sd-head { display: flex; flex-direction: column; justify-content: flex-end; padding: 0 0 6px; border-bottom: 1px solid var(--sd-text); min-width: 0; }
  .sd-head b { font-size: 13px; font-weight: 600; }
  .sd-head.out { opacity: 0.4; }
  .sd-question, .sd-cell { padding: 5px 0; border-top: 1px solid var(--sd-border); font-size: 13px; line-height: 1.35; }
  .sd-head + .sd-question, .sd-head + .sd-cell { border-top: 0; }
  .sd-question { display: flex; gap: 6px; }
  .sd-icon { flex: none; width: 1.2em; text-align: center; font-size: 12px; }
  .sd-cell { overflow-wrap: anywhere; color: var(--sd-text); }
  .sd-cell.running { background: var(--sd-run-bg); color: var(--sd-blue); font-weight: 600; padding-left: 6px; }
  .sd-cell.out { background: repeating-linear-gradient(135deg, transparent 0 6px, var(--sd-border) 6px 7px); opacity: 0.6; }
  .sd-noul { font-variant-numeric: tabular-nums; }
  .sd-board-note { grid-column: 1 / -1; padding: 10px 0; border-top: 1px solid var(--sd-border); font-size: 13px; color: var(--sd-text-3); }
  /* Race panel: a bar per model, a segment per question, coloured by kind. */
  .sd-tl-legend { display: flex; gap: 16px; font-size: 12px; color: var(--sd-text-2); }
  .sd-tl-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .sd-tl-legend i { width: 12px; height: 12px; border-radius: 3px; }
  .sd-tl-grid { display: grid; grid-template-columns: minmax(150px, max-content) 1fr minmax(170px, max-content); gap: 8px 14px; align-items: center; }
  .sd-tl-name { font-weight: 500; font-size: 14px; white-space: nowrap; }
  .sd-medal { display: inline-block; width: 1.5em; }
  .sd-tl-track {
    display: flex; height: 20px; border-radius: 4px; overflow: hidden; background-color: var(--sd-layer);
    background-image: repeating-linear-gradient(to right, transparent 0 calc(20% - 1px), var(--sd-border) calc(20% - 1px) 20%);
  }
  .sd-tl-track i { flex: none; height: 100%; box-shadow: inset -1px 0 0 #fff; }
  .sd-timeline i.decide { background: var(--sd-decide); }
  .sd-timeline i.chat { background: var(--sd-chat); }
  .sd-tl-track i.live { opacity: 0.5; }
  .sd-tl-totals { display: flex; gap: 10px; font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .sd-tl-totals b { min-width: 3.4em; font-weight: 600; }
  .sd-tl-totals .cache { color: var(--sd-text-2); }
  .sd-tl-name.winner, .sd-tl-totals.winner b { color: var(--sd-win); font-weight: 600; }
  .sd-tl-track.winner { outline: 2px solid var(--sd-win); outline-offset: 1px; }
  .sd-tl-name.out, .sd-tl-totals.out, .sd-tl-track.out { opacity: 0.4; }
  .sd-tl-ticks { position: relative; height: 14px; font-size: 11px; color: var(--sd-text-3); font-variant-numeric: tabular-nums; }
  .sd-tl-ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
  .sd-tl-ticks span:first-child { transform: none; }
  .sd-tl-ticks span:last-child { transform: translateX(-100%); }
  .sd-tl-axis { text-align: center; font-size: 12px; font-weight: 500; color: var(--sd-text-2); margin-top: -4px; }
  /* About panel: black, as in IBM's Granite Vision demo. */
  .sd-about-toggle { display: none; }
  .sd-about-toggle:not(:checked) + .sd-about { display: none; }
  .sd-about { position: fixed; inset: 0; z-index: 1000; display: flex; align-items: center; justify-content: center; padding: 24px; }
  .sd-about-backdrop { position: absolute; inset: 0; background: rgba(22, 22, 22, 0.55); cursor: pointer; }
  .sd-about-panel {
    position: relative; width: min(860px, 100%); max-height: calc(100vh - 48px); overflow: auto;
    padding: 24px 28px; border-radius: 12px; background: #000; color: #f4f4f4;
  }
  .sd-about-body { position: relative; display: flex; flex-direction: column; }
  .sd-about-close {
    position: absolute; top: 4px; right: 0; display: flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 50%; border: 2px solid var(--sd-blue); color: #fff; cursor: pointer; font-size: 14px;
  }
  .sd-about h2 { margin: 6px 48px 4px 0; font-size: 28px; font-weight: 400; color: #fff; }
  .sd-about h3 { margin: 0 0 16px; font-size: 20px; font-weight: 300; color: #8d8d8d; }
  .sd-about-lede { margin: 0 0 20px; padding-bottom: 20px; border-bottom: 1px solid #393939; font-size: 16px; line-height: 1.6; color: #f4f4f4; }
  .sd-about code { padding: 2px 6px; border-radius: 6px; background: #262626; color: #fff; font: 15px "IBM Plex Mono", ui-monospace, monospace; }
  .sd-about-section { margin: 4px 0 14px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #f4f4f4; }
  .sd-about-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 28px; margin-bottom: 22px; }
  .sd-about-item { display: flex; gap: 12px; align-items: flex-start; }
  .sd-about-icon {
    flex: none; display: flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: #333; font-size: 14px;
  }
  .sd-about-icon.mono { font: 600 10px "IBM Plex Mono", ui-monospace, monospace; color: #fff; }
  .sd-about-item b { display: block; font-size: 15px; font-weight: 600; color: #fff; }
  .sd-about-item p { margin: 4px 0 0; font-size: 13px; line-height: 1.45; color: #c6c6c6; }
  .sd-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: auto; padding-top: 8px; }
  .sd-pills span { padding: 2px 8px; border-radius: 999px; background: #a7f0ba; color: #0e6027; font-size: 14px; }
  @media (max-width: 900px) {
    .sd-about-grid { grid-template-columns: 1fr; }
  }
}
"""

# IBM's Carbon palette (Cool Gray and Blue), as Gradio theme colours.
_COOL_GRAY = gr.themes.Color(
    c50="#f9f9fb", c100="#f2f4f8", c200="#dde1e6", c300="#c1c7cd", c400="#a2a9b0", c500="#878d96",
    c600="#697077", c700="#4d5358", c800="#393939", c900="#21272a", c950="#121619",
)
_BLUE = gr.themes.Color(
    c50="#edf5ff", c100="#edf5ff", c200="#d0e2ff", c300="#a6c8ff", c400="#78a9ff", c500="#4589ff",
    c600="#0f62fe", c700="#0043ce", c800="#002d9c", c900="#001d6c", c950="#001141",
)
THEME = gr.themes.Base(
    primary_hue=_BLUE,
    secondary_hue=_BLUE,
    neutral_hue=_COOL_GRAY,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#e8eaed",
    body_text_color="#161616",
    body_text_color_subdued="#6f6f6f",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f2f4f8",
    border_color_primary="#dde1e6",
    block_background_fill="#ffffff",
    block_border_color="#dde1e6",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    layout_gap="12px",
)
# Light mode only, like IBM's demo: Gradio follows the system setting otherwise.
FORCE_LIGHT_JS = """() => {
  const strip = () => [document.documentElement, document.body].forEach(e => e.classList.remove('dark'));
  strip();
  new MutationObserver(strip).observe(document.body, { attributes: true, attributeFilter: ['class'] });
}"""

with gr.Blocks(title="Decisions Switching Costs", fill_width=True) as demo:
    round_state = gr.State()
    with gr.Column(elem_classes="sd-frame"):
        with gr.Row(elem_classes="sd-topbar"):
            gr.HTML(BRAND, padding=False, elem_classes=["sd-brand-wrap", "sd-flush"], min_width=0)
            gr.HTML(INFO_BUTTON, padding=False, elem_classes="sd-flush", min_width=0)
            new_button = gr.Button("🎲 New article", elem_classes="sd-pill", scale=0, min_width=0)
            run_button = gr.Button(RUN_LABEL, elem_classes=["sd-pill", "sd-dark"], scale=0, min_width=0, interactive=False)
        gr.HTML(INTRO, padding=False, elem_classes="sd-flush")
        with gr.Row(elem_classes="sd-body"):
            with gr.Column(scale=1, min_width=280, elem_classes="sd-side"):
                article_card = gr.HTML(_article_html(None, "Finding a random article…"), padding=False, elem_classes="sd-flush")
                endpoint_status = gr.HTML(endpoints_status(), padding=False, elem_classes="sd-flush")
            with gr.Column(scale=4, elem_classes="sd-main"):
                board = gr.HTML(_board_html(None, []), padding=False, elem_classes="sd-flush")
                with gr.Accordion("How each model was called", open=False, elem_classes="sd-acc"):
                    timing = gr.Markdown()
                with gr.Accordion("About the models", open=False, elem_classes="sd-acc"):
                    gr.Markdown(NOTES)
    gr.HTML(_about_html(), padding=False, elem_classes="sd-flush")

    round_outputs = [article_card, board, round_state, run_button, timing]
    # A round is set up when the page opens, so there's something to run straight away.
    demo.load(new_round, outputs=round_outputs, show_progress="hidden", api_name=False)
    # Both buttons work mid-run: the run in progress is stopped first, outside
    # the queue, so the new round or run starts straight away.
    new_button.click(stop_run, queue=False, show_progress="hidden", api_name=False).then(
        new_round, outputs=round_outputs, show_progress="hidden", api_name="new_round"
    )
    run_button.click(stop_run, queue=False, show_progress="hidden", api_name=False).then(
        run_models, round_state, [board, timing, endpoint_status, run_button],
        show_progress="hidden", api_name="run",
    )
    # Wake the endpoints on page load, and keep the status line current while they wake.
    demo.load(on_page_load, outputs=endpoint_status, show_progress="hidden", api_name=False)
    gr.Timer(3).tick(endpoints_status, outputs=endpoint_status, show_progress="hidden", queue=False)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, js=FORCE_LIGHT_JS)
