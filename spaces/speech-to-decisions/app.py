# SPDX-License-Identifier: Apache-2.0
"""Speech to Decisions: models size up what you're saying, while you're saying it.

Speak into the microphone. NVIDIA's Parakeet turns the speech into text. About
once a second, two models look at the last few words: OpenAI's open-weight
gpt-oss-120b says what the speaker is doing in 1 to 5 words, and TypeSafe AI's
Jev (a "System One" model: it returns decisions, not text) screens the words
for harmful content and answers a list of yes/no questions. All three models
are called through OpenRouter with one API key.

The flow
--------
1. The page (static/, a hand-built page with no framework) captures the
   microphone at 16 kHz and posts it to /api/chunk every half second. Each
   chunk is appended to the session's open *segment* of audio.
2. Parakeet has no streaming mode on OpenRouter: a request takes a whole clip
   and returns its text. So about once a second the open segment is sent again
   in full, and the text that comes back replaces the segment's *live* text
   (shown in grey). A request for 10 seconds of audio takes about 0.4 s.
3. When the speaker pauses, the segment's text is *committed* to the
   transcript and a new segment starts, so no request ever carries more than
   MAX_SEGMENT_S of audio. A speaker who doesn't pause is cut at the quietest
   moment of the segment's last few seconds.
4. Each time the text changes, its last RECENT_WORDS words go to two models
   (after QUIET_AFTER_S of silence the judgments are wiped, and the next ones
   are about the words said after the silence):
   - Jev, in one System One request: three safety Nouls (harm, profanity,
     violence) and one Noul per yes/no question on the page. A Noul is the
     probability that the answer is yes. That request takes about 0.2 s.
   - gpt-oss-120b, in one chat completion: what the speaker is doing, in 1 to
     5 words. It takes 0.2 to 0.8 s, so it runs on its own thread and never
     holds up the transcript or Jev.

The transcription and the Jev call run on a background thread per session, so
the microphone stream is never held up. Every /api/chunk reply carries the
session's latest results, and the page redraws from it.

The API
-------
GET  /api/config    what the page needs to draw itself
POST /api/session   set the yes/no questions (and make a session if needed)
POST /api/chunk     a chunk of 16-bit mono PCM; ?sid= and ?rate=
POST /api/stop      recording stopped: finish with the audio that's left
POST /api/clear     drop the session and start a new one
Every POST returns the session's state as JSON. Everything else under / is
served from static/.
"""

import base64
import io
import os
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError

# Every model is reached through OpenRouter. Jev goes through its System One
# API, which the TypeSafe SDK speaks natively; a bare model id like "jev-1.13"
# routes to typesafe/jev-1.13.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
STT_URL = f"{OPENROUTER_BASE_URL}/v1/audio/transcriptions"
CHAT_URL = f"{OPENROUTER_BASE_URL}/v1/chat/completions"
# The model that says what the speaker is doing, in a few words.
DESCRIBE_MODEL = os.environ.get("DESCRIBE_MODEL", "openai/gpt-oss-120b")
# Sent with that request to keep it quick: route to the provider with the
# highest throughput for the model (the hosts differ a lot), and have a
# reasoning model think only briefly before it writes.
DESCRIBE_OPTIONS = {"provider": {"sort": "throughput"}, "reasoning": {"effort": "low"}}
# The label is a few tokens; the rest is headroom for the model's reasoning.
DESCRIBE_MAX_TOKENS = 400
DESCRIBE_MAX_WORDS = 5
# Most replies take 0.2 to 0.8 s, but now and then one takes many seconds. So a
# new request doesn't wait for the last one: up to this many run at once, the
# newest answer wins, and a request that takes this long is given up on.
DESCRIBE_IN_FLIGHT = 3
DESCRIBE_TIMEOUT_S = 5.0
DESCRIBE_PROMPT = (
    "You label live speech. You are given the last few words of a transcript, which may be "
    "cut off mid-sentence. Reply with 1 to 5 words that say what the speaker is doing, "
    "starting with a verb ending in -ing, for example 'Reporting an outage'. No punctuation, "
    "no quotes, nothing else."
)
STT_MODEL = os.environ.get("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13")
# Parakeet detects the language when this is empty.
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")

SAMPLE_RATE = 16000  # what the audio is resampled to before it's sent
STREAM_EVERY_S = 0.5  # how often the page sends a chunk
MAX_CHUNK_BYTES = 2_000_000  # a chunk is about 16 kB; this is far above any real one
STATIC_DIR = Path(__file__).parent / "static"
# Transcribe again once this much new audio has arrived.
MIN_NEW_AUDIO_S = float(os.environ.get("MIN_NEW_AUDIO_S", "1.0"))
# A segment is committed at a pause once it's at least MIN_COMMIT_S long, and
# cut at its quietest recent moment once it reaches MAX_SEGMENT_S.
MIN_COMMIT_S = 4.0
MAX_SEGMENT_S = 15.0
PAUSE_S = 0.7
# After this many seconds with no speech the board is wiped. The next judgments
# are about what's said after the silence only, not the words before it.
QUIET_AFTER_S = float(os.environ.get("QUIET_AFTER_S", "10"))
# A 100 ms frame louder than this (RMS, as a fraction of full scale) counts as
# speech. Audio with no such frame isn't sent to Parakeet at all.
VOICE_RMS = float(os.environ.get("VOICE_RMS", "0.01"))
FRAME = SAMPLE_RATE // 10

# The models judge what's being said right now: the last RECENT_WORDS words of
# the transcript, so the board follows the speaker from one subject to the next.
RECENT_WORDS = 25
MAX_NOULS = 50
SESSION_TTL_S = 30 * 60

# The safety check: Jev answers one Noul per category about the same words.
# key -> (name on the page, the question Jev is asked). A category above
# SAFETY_THRESHOLD flags the text.
SAFETY = {
    "harm": (
        "Harm",
        "Is what the speaker says harmful: could it hurt someone, or does it encourage, plan or "
        "explain something dangerous, abusive or illegal?",
    ),
    "profanity": ("Profanity", "Does the speaker use profane or vulgar language?"),
    "violence": (
        "Violence",
        "Does the speaker threaten, encourage or describe wanting to commit violence or physical harm?",
    ),
}
SAFETY_THRESHOLD = 0.5

DEFAULT_NOULS = [
    "Is the speaker asking a question?",
    "Is the speaker asking someone to do something?",
    "Is something going wrong?",
    "Is the speaker making a promise or a commitment?",
    "Does the speaker mention money or a price?",
    "Does the speaker mention a person by name?",
    "Does the speaker mention a date, a time or a deadline?",
    "Does the speaker mention a place?",
    "Is the speaker talking about themselves?",
    "Is the speaker joking or being playful?",
    "Is the speaker sharing private or sensitive information?",
    "Does the speaker sound unsure of what they're saying?",
]

# One HTTP connection pool for each API, shared by every session, so a request
# doesn't pay for a new TLS handshake.
_http = httpx.Client(timeout=30.0)
_jev: TypeSafeClient | None = None
_jev_lock = threading.Lock()


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def _jev_client() -> TypeSafeClient:
    global _jev
    with _jev_lock:
        if _jev is None:
            _jev = TypeSafeClient(
                api_key=_api_key(), base_url=OPENROUTER_BASE_URL, model=JEV_MODEL, timeout=15.0
            )
        return _jev


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def to_mono_16k(rate: int, samples: np.ndarray) -> np.ndarray:
    """A microphone chunk as mono 16 kHz float32 in [-1, 1]."""
    y = np.asarray(samples)
    scale = 1.0 if y.dtype.kind == "f" else float(np.iinfo(y.dtype).max)
    y = y.astype(np.float32) / scale
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate == SAMPLE_RATE or not len(y):
        return y
    if rate % SAMPLE_RATE == 0:
        # A whole-number ratio (48 kHz is 3x): average each group of samples,
        # which also filters out what 16 kHz can't represent.
        k = rate // SAMPLE_RATE
        return y[: len(y) // k * k].reshape(-1, k).mean(axis=1)
    n = int(len(y) * SAMPLE_RATE / rate)
    return np.interp(np.arange(n) * (rate / SAMPLE_RATE), np.arange(len(y)), y).astype(np.float32)


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    """The loudness of each whole 100 ms frame."""
    n = len(audio) // FRAME
    if not n:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((audio[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))


def has_voice(audio: np.ndarray) -> bool:
    return bool((_frame_rms(audio) > VOICE_RMS).any())


def trailing_silence_s(audio: np.ndarray) -> float:
    """Seconds of quiet at the end of the audio."""
    loud = np.nonzero(_frame_rms(audio) > VOICE_RMS)[0]
    frames = len(audio) // FRAME
    return (frames - 1 - loud[-1]) * 0.1 if len(loud) else frames * 0.1


def quietest_cut(audio: np.ndarray, window_s: float = 4.0) -> int:
    """The sample at the middle of the quietest frame in the last `window_s` seconds."""
    rms = _frame_rms(audio)
    first = max(0, len(rms) - int(window_s * 10))
    return (first + int(np.argmin(rms[first:]))) * FRAME + FRAME // 2


def wav_base64(audio: np.ndarray) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Parakeet, Jev and gpt-oss-120b (OpenRouter)
# --------------------------------------------------------------------------- #


class ModelError(Exception):
    """A model call failed; the message is short enough to show on the page."""


def transcribe(audio: np.ndarray) -> tuple[str, float, float]:
    """Return (text, seconds, cost in dollars) for a clip of 16 kHz audio."""
    payload = {"model": STT_MODEL, "input_audio": {"data": wav_base64(audio), "format": "wav"}}
    if STT_LANGUAGE:
        payload["language"] = STT_LANGUAGE
    start = time.perf_counter()
    try:
        response = _http.post(STT_URL, headers={"Authorization": f"Bearer {_api_key()}"}, json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as e:
        raise ModelError(f"Parakeet call failed ({e.response.status_code}): {e.response.text[:200]}") from e
    except (httpx.HTTPError, ValueError) as e:
        raise ModelError(f"Parakeet call failed: {e}") from e
    cost = float((body.get("usage") or {}).get("cost") or 0.0)
    return (body.get("text") or "").strip(), time.perf_counter() - start, cost


def build_questions(nouls: list[str]) -> dict:
    """The safety check and the yes/no questions, as one System One question set."""
    questions = {f"s_{key}": Noul(instructions=question) for key, (_, question) in SAFETY.items()}
    for i, question in enumerate(nouls):
        questions[f"n{i}"] = Noul(instructions=question)
    return questions


def classify(text: str, nouls: list[str]) -> tuple[dict, float, float, str]:
    """Ask Jev everything about `text`. Return (answers, seconds, cost, model)."""
    start = time.perf_counter()
    try:
        response = _jev_client().system_one(
            state={"transcript_of_what_the_speaker_said": text}, questions=build_questions(nouls)
        )
    except TypeSafeError as e:
        raise ModelError(f"Jev call failed: {e}") from e
    elapsed = time.perf_counter() - start
    # The SDK's usage has no cost field, so read the raw reply for it.
    try:
        cost = float((response.raw_http_response.json().get("usage") or {}).get("cost") or 0.0)
    except (TypeSafeError, ValueError, AttributeError):
        cost = 0.0
    answers = {
        "safety": {key: response.nouls[f"s_{key}"].noul for key in SAFETY},
        "nouls": [(q, response.nouls[f"n{i}"].noul) for i, q in enumerate(nouls)],
    }
    return answers, elapsed, cost, response.model


def describe(text: str) -> tuple[str, float, float]:
    """What the speaker is doing, in 1 to 5 words. Return (label, seconds, cost)."""
    payload = {
        "model": DESCRIBE_MODEL,
        "messages": [
            {"role": "system", "content": DESCRIBE_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": DESCRIBE_MAX_TOKENS,
        "temperature": 0,
        **DESCRIBE_OPTIONS,
    }
    start = time.perf_counter()
    try:
        response = _http.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {_api_key()}"},
            json=payload,
            timeout=DESCRIBE_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"].get("content") or ""
    except httpx.HTTPStatusError as e:
        raise ModelError(f"{DESCRIBE_MODEL} call failed ({e.response.status_code}): {e.response.text[:200]}") from e
    except httpx.TimeoutException as e:
        raise ModelError(f"{DESCRIBE_MODEL} took longer than {DESCRIBE_TIMEOUT_S:g} s.") from e
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise ModelError(f"{DESCRIBE_MODEL} call failed: {e}") from e
    cost = float((body.get("usage") or {}).get("cost") or 0.0)
    return clean_label(content), time.perf_counter() - start, cost


def clean_label(content: str) -> str:
    """The reply's first line, without quotes or a full stop, held to the word limit."""
    lines = content.strip().splitlines()
    label = lines[0].strip(" \"'.") if lines else ""
    # Now and then the model writes its label twice, run together.
    half = len(label) // 2
    if half and label[:half] == label[half:]:
        label = label[:half]
    return " ".join(label.split()[:DESCRIBE_MAX_WORDS])


def clean_nouls(questions: list[str]) -> list[str]:
    """The yes/no questions to ask: blank lines and repeats are dropped."""
    seen, kept = set(), []
    for line in questions:
        line = line.strip()[:300]
        if line and line.lower() not in seen:
            seen.add(line.lower())
            kept.append(line)
    return kept[:MAX_NOULS]


def recent(session: "Session") -> str:
    """What's being said right now: the last RECENT_WORDS words since the last long silence."""
    return " ".join(session.text().split()[session.fresh_from :][-RECENT_WORDS:])


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


@dataclass
class Session:
    """One visitor's audio, transcript and latest answers."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    nouls: list[str] = field(default_factory=lambda: list(DEFAULT_NOULS))  # the yes/no questions
    # The open segment: audio whose text isn't committed yet.
    segment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pending: int = 0  # samples that arrived since the last pass started
    transcribed: int = 0  # how many of the segment's samples `live` covers
    busy: bool = False  # a pass is running
    committed: list[str] = field(default_factory=list)
    live: str = ""
    answers: dict | None = None
    classified: tuple | None = None  # the (text, questions) Jev last answered
    description: str = ""  # what the speaker is doing, from gpt-oss-120b
    described: str | None = None  # the text `description` is about
    describe_asked: str | None = None  # the text of the newest description request
    describe_sent: int = 0  # description requests started; each one's number
    describe_shown: int = 0  # the number of the request `description` came from
    describing: int = 0  # description requests running now
    describe_s: float = 0.0
    describe_error: str = ""
    stt_s: float = 0.0
    stt_audio_s: float = 0.0
    jev_s: float = 0.0
    jev_model: str = ""
    updates: int = 0
    silent_s: float = 0.0  # how long it's been since a chunk had speech in it
    quiet: bool = False  # silent for QUIET_AFTER_S: the board is blank
    fresh_from: int = 0  # how many of the transcript's words came before the last long silence
    cost: float = 0.0
    error: str = ""
    last_seen: float = field(default_factory=time.monotonic)

    def text(self) -> str:
        return " ".join(self.committed + ([self.live] if self.live else []))


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(sid: str | None) -> tuple[str, Session]:
    """The session for this browser tab, made on first use. Idle ones are dropped."""
    now = time.monotonic()
    with _sessions_lock:
        for old in [k for k, s in _sessions.items() if now - s.last_seen > SESSION_TTL_S]:
            del _sessions[old]
        if sid not in _sessions:
            sid = uuid.uuid4().hex
            _sessions[sid] = Session()
        _sessions[sid].last_seen = now
        return sid, _sessions[sid]


def start_describe(session: Session, text: str) -> int | None:
    """Reserve a description request for `text`. Call with the session's lock held.

    Returns the request's number, or None when the newest request already
    asked about this text or too many are running.
    """
    if not text or text == session.describe_asked or session.describing >= DESCRIBE_IN_FLIGHT:
        return None
    session.describing += 1
    session.describe_sent += 1
    session.describe_asked = text
    return session.describe_sent


def run_describe(session: Session, text: str, number: int) -> None:
    """Ask gpt-oss-120b what the speaker is doing; keep the answer unless a newer one is in."""
    try:
        label, seconds, cost = describe(text)
        with session.lock:
            session.cost += cost
            if number > session.describe_shown:
                session.describe_shown, session.described = number, text
                session.describe_s, session.describe_error = seconds, ""
                if label:
                    session.description = label
    except ModelError as e:
        with session.lock:
            if number == session.describe_sent:
                # The newest request failed: show why, and let the next pass ask again.
                session.describe_error, session.describe_asked = str(e), None
    finally:
        with session.lock:
            session.describing -= 1


def run_pass(session: Session, final: bool = False) -> None:
    """Transcribe the open segment, commit it if it's time to, and ask the models.

    Only one pass runs per session at a time (`busy`). Audio that arrives
    while it runs is appended behind the part this pass works on.
    """
    try:
        with session.lock:
            audio = session.segment.copy()
            session.pending = 0
            covered, text = session.transcribed, session.live
            nouls = list(session.nouls)
        seconds = len(audio) / SAMPLE_RATE
        paused = trailing_silence_s(audio) >= PAUSE_S
        forced = seconds >= MAX_SEGMENT_S and not paused and not final
        commit = final or forced or (paused and seconds >= MIN_COMMIT_S)
        cut = quietest_cut(audio) if forced else len(audio)

        # Send the segment only if there's speech in it that `live` doesn't cover.
        if has_voice(audio[:cut]) and (forced or has_voice(audio[covered:])):
            text, stt_s, cost = transcribe(audio[:cut])
            with session.lock:
                session.stt_s, session.stt_audio_s = stt_s, cut / SAMPLE_RATE
                session.cost += cost

        with session.lock:
            if commit:
                if text:
                    session.committed.append(text)
                session.segment, session.live, session.transcribed = session.segment[cut:], "", 0
            elif not text and seconds > 2.0:
                # Nothing but quiet so far: keep only the last half second, so
                # the start of the next word isn't lost.
                keep = len(audio) - SAMPLE_RATE // 2
                session.segment, session.transcribed = session.segment[keep:], 0
            else:
                session.live, session.transcribed = text, len(audio)
            state = recent(session)
            asked = (state, tuple(nouls))
            stale = bool(state) and asked != session.classified
            # The description runs beside Jev, on its own thread.
            number = start_describe(session, state)

        if number:
            threading.Thread(target=run_describe, args=(session, state, number), daemon=True).start()
        if stale:
            answers, jev_s, cost, model = classify(state, nouls)
            with session.lock:
                session.answers, session.classified = answers, asked
                session.jev_s, session.jev_model = jev_s, model
                session.cost += cost
                session.updates += 1
        with session.lock:
            session.error = ""
    except ModelError as e:
        with session.lock:
            session.error = str(e)
    finally:
        with session.lock:
            session.busy = False


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


def state_json(sid: str, session: Session) -> dict:
    """Everything the page draws, as of now."""
    with session.lock:
        answers = session.answers
        return {
            "sid": sid,
            "ready": bool(_api_key()),
            "committed": " ".join(session.committed),
            "live": session.live,
            "description": session.description,
            "safety": answers["safety"] if answers else None,
            "nouls": [{"question": q, "noul": n} for q, n in answers["nouls"]] if answers else None,
            "questions": list(session.nouls),
            "timings": {
                "stt_s": session.stt_s,
                "stt_audio_s": session.stt_audio_s,
                "jev_s": session.jev_s,
                "decisions": len(SAFETY) + len(answers["nouls"]) if answers else 0,
                "describe_s": session.describe_s,
            },
            "updates": session.updates,
            "quiet": session.quiet,
            "cost": session.cost,
            "errors": [e for e in (session.error, session.describe_error) if e],
        }


def go_quiet(session: Session) -> None:
    """A long silence: wipe the judgments. Call with the session's lock held.

    The words so far are set aside too, so the judgments that come back when
    the speaker starts again are about the new words alone.
    """
    session.quiet = True
    session.fresh_from = len(session.text().split())
    session.answers, session.classified = None, None
    session.description, session.described, session.describe_asked = "", None, None
    # Any description request still out is from before the silence: ignore its answer.
    session.describe_shown = session.describe_sent


def add_audio(session: Session, audio: np.ndarray) -> None:
    """Append a chunk to the open segment and start a pass if one is due."""
    with session.lock:
        session.segment = np.concatenate([session.segment, audio])
        session.pending += len(audio)
        if has_voice(audio):
            session.silent_s, session.quiet = 0.0, False
        else:
            session.silent_s += len(audio) / SAMPLE_RATE
            if session.silent_s >= QUIET_AFTER_S and not session.quiet:
                go_quiet(session)
        start = not session.busy and session.pending >= MIN_NEW_AUDIO_S * SAMPLE_RATE
        if start:
            session.busy = True
    if start:
        threading.Thread(target=run_pass, args=(session,), daemon=True).start()


def _wait(session: Session, done, timeout_s: float = 20.0) -> bool:
    """Wait until `done()` is true, checked with the session's lock held."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with session.lock:
            if done():
                return True
        time.sleep(0.05)
    return False


def _claim_pass(session: Session) -> bool:
    if session.busy:
        return False
    session.busy = True
    return True


def settle(session: Session, final: bool) -> None:
    """Run one pass here, then make sure the description is about the same words."""
    if not _api_key():
        return
    if _wait(session, lambda: _claim_pass(session)):
        run_pass(session, final=final)
    # The pass started a description request if one was due; let it finish.
    _wait(session, lambda: session.describing == 0, DESCRIBE_TIMEOUT_S + 1)
    with session.lock:
        state = recent(session)
        number = start_describe(session, state) if state != session.described else None
    if number:
        run_describe(session, state, number)


class SessionBody(BaseModel):
    sid: str | None = None
    questions: list[str] | None = None


app = FastAPI(title="Speech to Decisions", docs_url=None, redoc_url=None)


@app.get("/api/config")
def get_config() -> dict:
    return {
        "ready": bool(_api_key()),
        "models": {"stt": STT_MODEL, "jev": JEV_MODEL, "describe": DESCRIBE_MODEL},
        "safety": [{"key": key, "label": label} for key, (label, _) in SAFETY.items()],
        "safety_threshold": SAFETY_THRESHOLD,
        "questions": DEFAULT_NOULS,
        "max_questions": MAX_NOULS,
        "recent_words": RECENT_WORDS,
        "quiet_after_s": QUIET_AFTER_S,
        "sample_rate": SAMPLE_RATE,
        "chunk_s": STREAM_EVERY_S,
    }


@app.post("/api/session")
def post_session(body: SessionBody) -> dict:
    """Set the yes/no questions. With nothing recording, ask Jev the new ones now."""
    sid, session = get_session(body.sid)
    if body.questions is not None:
        with session.lock:
            session.nouls = clean_nouls(body.questions)
            idle = not session.busy and bool(session.committed) and not len(session.segment)
        if idle:
            settle(session, final=False)
    return state_json(sid, session)


@app.post("/api/chunk")
async def post_chunk(request: Request, sid: str | None = None, rate: int = SAMPLE_RATE) -> dict:
    """A chunk of microphone audio: 16-bit little-endian mono PCM at `rate`."""
    body = await request.body()
    if len(body) > MAX_CHUNK_BYTES or not 8000 <= rate <= 192000:
        raise HTTPException(status_code=413 if len(body) > MAX_CHUNK_BYTES else 400)
    sid, session = get_session(sid)
    if body and _api_key():
        add_audio(session, to_mono_16k(rate, np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2")))
    return state_json(sid, session)


@app.post("/api/stop")
def post_stop(body: SessionBody) -> dict:
    """Recording stopped: finish with whatever audio hasn't been transcribed."""
    sid, session = get_session(body.sid)
    settle(session, final=True)
    return state_json(sid, session)


@app.post("/api/clear")
def post_clear(body: SessionBody) -> dict:
    """Drop the session and start a new one with the same questions."""
    with _sessions_lock:
        old = _sessions.pop(body.sid, None)
    sid, session = get_session(None)
    if body.questions is not None:
        session.nouls = clean_nouls(body.questions)
    elif old is not None:
        session.nouls = list(old.nouls)
    return state_json(sid, session)


# Everything else is the page.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
