---
title: Speech to Decisions
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Live speech, sized up by Jev and gpt-oss-120b
---

# Speech to Decisions

Talk into the microphone and watch models size up what you're saying, while
you're still saying it.

Three models do the work, all called through [OpenRouter](https://openrouter.ai)
with one API key:

- **Parakeet** ([`nvidia/parakeet-tdt-0.6b-v3`](https://openrouter.ai/nvidia/parakeet-tdt-0.6b-v3)),
  NVIDIA's 600M-parameter speech recognition model, turns the speech into text.
- **gpt-oss-120b** ([`openai/gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b)),
  OpenAI's open-weight model, says what the speaker is doing in 1 to 5 words.
- **Jev** ([`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13)),
  TypeSafe AI's *System One* model, screens the words for harmful content and
  answers the yes/no questions. Jev writes no text. It takes a state and a set
  of typed questions and returns a decision for each.

## What's on the page

The left column has the microphone and, under it, the live transcript. Tap the
microphone to start; the dots around it swell with your voice, and a dotted
meter and a timer run while it listens. The right column is the board, which
updates about once a second:

- **The speaker is.** A freeform label from gpt-oss-120b, such as "Reporting
  an outage" or "Inviting friends over".
- **Safety check.** Jev answers three questions about the words: is this
  harmful, is there profanity, and is there violence. Each answer is a
  **Noul**, the probability that the answer is yes. The box reads "Nothing
  harmful" while all three are at 0.5 or below, and turns red and reads
  "Flagged" when any is above it. The three bars show each noul.
- **Yes/No questions.** A bar per question, also a Noul from Jev. A bar turns
  blue above 0.5.

**The yes/no questions are editable until you start recording.** Each row's
question sits in a text field: change it, remove it with the × at the end of
its row, or type in the dashed empty row at the end to add one. Up to 50 are
sent. While the microphone is on
the questions are locked and the rows are plain text; they unlock when you
stop. If you edit a question after a recording, Jev answers the new list about
the words already on the page.

**Start over** in the header clears the transcript and the board and keeps
your questions. The **i** button opens a panel that explains the page.

**What's judged** is what's being said right now: the last 25 words of the
transcript. So the board follows the speaker when the subject changes, and a
harmful remark clears from the safety check about 25 words later.

**Silence blanks the board.** After 10 seconds with no speech while recording,
every judgment is wiped: the label, the safety check and the yes/no bars all
show a dash, and a "Waiting for speech" tag appears. When you start talking
again the board fills back in, and those judgments are about the words said
after the silence only. The transcript keeps everything.

The safety check and the yes/no questions go to Jev in one System One request:
3 safety Nouls and up to 50 more.

## The freeform label

Each time the text changes, its last 25 words also go to gpt-oss-120b in one
chat completion, with a prompt asking for 1 to 5 words starting with an "-ing"
verb. The request sets `provider.sort` to `throughput`, so OpenRouter routes it
to the fastest host, and `reasoning.effort` to `low`.

Most replies take 0.2 to 0.8 s, but now and then one takes many seconds. So
the label doesn't hold up anything else, and one slow reply doesn't hold up the
next:

- It runs on its own thread, beside the Jev request.
- A new request doesn't wait for the last one. Up to 3 run at once, and an
  answer is dropped if a newer one is already on the board.
- A request is given up on after 5 s.

The reply's first line is used, without quotes or a full stop, cut to 5 words.

## How live transcription works

Parakeet on OpenRouter takes a whole clip per request and returns its text;
there's no streaming mode. The Space gets a live transcript out of it like
this:

1. **Chunks.** The page captures the microphone at 16 kHz and posts it to the
   Space every half second as 16-bit PCM. Each chunk is appended to the
   session's open *segment*. A browser that can't capture at 16 kHz sends its
   own rate, and the Space resamples.
2. **Re-transcribe.** Once a second of new audio has arrived, the whole open
   segment is sent to Parakeet again. Its text replaces the segment's *live*
   text, shown in grey, which can still change as more audio gives Parakeet
   more to go on.
3. **Commit at a pause.** When the speaker has been quiet for 0.7 s and the
   segment is at least 4 s long, its text is committed to the transcript and a
   new segment starts. A speaker who doesn't pause is cut after 15 s, at the
   quietest moment of the segment's last 4 s.
4. **Skip the quiet.** Audio with no 100 ms frame above the loudness threshold
   isn't sent at all, so silence costs nothing and Parakeet isn't asked to
   find words in room noise.

Transcription and the Jev call run on a background thread per session, so the
microphone stream never waits on a model. If a request takes longer than a
second, the next one starts when it finishes and carries all the audio that
arrived in between.

## Speed and cost

Measured from a laptop, replaying a 19-second clip through the app's handlers:

| Call | Round trip |
|---|---|
| Parakeet, 9 s of audio | 0.26–0.65 s |
| Jev, 15 decisions | 0.15–0.22 s |
| gpt-oss-120b, one label | 0.2–0.8 s, with an occasional reply of several seconds |

The status line under the transcript shows all three timings for the latest
requests and a running cost, read from the `usage.cost` each API returns. Because
the open segment is sent again every second, a minute of speech is billed as
several minutes of audio. The 19-second clip cost about $0.005 in all.

## How it's built

The page is hand-built, with no front-end framework, and a small FastAPI server
sits behind it. The OpenRouter key stays on the server.

| File | What it does |
|---|---|
| [`app.py`](app.py) | The server: sessions, the transcription and judgment passes, and the API. |
| [`static/index.html`](static/index.html), [`static/style.css`](static/style.css) | The page and its styles. |
| [`static/app.js`](static/app.js) | Microphone capture, posting chunks, and drawing the board. |
| [`Dockerfile`](Dockerfile) | Runs the server with uvicorn on port 7860. |

The API is five routes. Every `POST` returns the session's whole state as JSON,
and the page redraws from it:

| Route | What it does |
|---|---|
| `GET /api/config` | Model names, safety categories and default questions. |
| `POST /api/session` | Sets the yes/no questions, and makes a session if there isn't one. |
| `POST /api/chunk` | Takes a chunk of audio. |
| `POST /api/stop` | Finishes with the audio that's left. |
| `POST /api/clear` | Drops the session and starts a new one. |

**The design** follows IBM's Granite demo pages, in particular
[Granite Speech Streaming WebGPU](https://huggingface.co/spaces/ibm-granite/granite-speech-streaming-webgpu)
(Apache-2.0): the Carbon design tokens, IBM Plex type, the app card, the record
panel with its cluster of dots, the dark info panel, and the tags and pills
are adapted from that Space's stylesheet. The judgment tiles and noul
bars are this demo's own, drawn with the same tokens. The header carries the
same IBM logo and "Research Demos" title as that Space; the demo's own name is
in the info panel.

The page's script is a classic script and loads its audio worklet from a
`blob:` URL. On a private Space the page sits in a cross-site iframe, where
module fetches aren't authenticated; the reference Space documents the same
constraint.

To run it locally:

```bash
pip install -r requirements.txt
OPENROUTER_API_KEY=... PORT=7860 python app.py   # http://localhost:7860
```

## Setup

1. Create a Docker Space (free CPU hardware is enough) and push this folder to
   it, or run
   [`scripts/create_speech_to_decisions_space.py`](../../scripts/create_speech_to_decisions_space.py)
   from the repo root.
2. **Settings → Variables and secrets**, add:
   - `OPENROUTER_API_KEY`: your OpenRouter API key, used for all three models.
     Without it the page loads and says so.
   - `JEV_MODEL` (optional): defaults to `jev-1.13` (routed to
     `typesafe/jev-1.13`).
   - `DESCRIBE_MODEL` (optional): the OpenRouter chat model that writes the
     label. Defaults to `openai/gpt-oss-120b`.
   - `STT_MODEL` (optional): defaults to `nvidia/parakeet-tdt-0.6b-v3`. Any
     OpenRouter transcription model that accepts WAV works.
   - `STT_LANGUAGE` (optional): defaults to `en`. Set it to an empty string
     and Parakeet detects the language.
   - `VOICE_RMS` (optional): the loudness that counts as speech, as a fraction
     of full scale. Defaults to `0.01`. Lower it for a quiet microphone.
   - `QUIET_AFTER_S` (optional): how many seconds of silence blank the board.
     Defaults to `10`.
   - `MIN_NEW_AUDIO_S` (optional): how much new audio triggers the next
     transcription. Defaults to `1.0`.

The browser asks for microphone access, which needs HTTPS. A Space has it; so
does `localhost`.

## Caveats

- The grey text is provisional. Words near the end of the open segment often
  change on the next pass, and Jev's answers change with them.
- The models see 25 words with no earlier context, so a pronoun or a
  half-finished sentence can swing an answer.
- The safety check is three yes/no questions put to a general decision model,
  with a 0.5 cut-off that hasn't been tuned. It isn't a moderation system. The
  categories overlap: in testing, a remark about suicide scored 0.80 on harm
  and 0.78 on violence, and hate speech scored 0.91 on harm with violence
  below the cut-off. Profanity on its own (swearing at a printer) flags the
  box through the profanity bar alone.
- The label is written fresh each time, so its wording can change from one
  second to the next while the speaker's meaning hasn't.
- A segment cut at 15 s without a pause can split a word across two segments.
- The loudness threshold is fixed. In a noisy room, background sound can count
  as speech and get sent to Parakeet. It also resets the 10-second silence
  count, so a noisy room may never blank the board.
- Visitors share the Space's OpenRouter key, so a public Space spends the
  owner's credits.
