---
title: Noul Race
emoji: 🏁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - ibm-granite/granite-switch-4.1-3b-preview
short_description: Granite's UQ aLoRA nouls raced against Jev and GPT Luna
---

# Noul Race

Some decisions don't need reasoning out loud. They need a fast, calibrated gut
call. That's the idea behind *System One* models like TypeSafe AI's
[Jev](https://docs.typesafe.ai): instead of text, Jev answers a yes/no question
with a **noul**, the probability that the answer is yes.

OpenAI's **GPT Luna** (`gpt-6-luna`) makes the same kind of call through the
[Decisions API](https://developers.openai.com/api/docs/guides/decisions), where
a *predicate* question returns the probability that it's true.

Noul Race puts one open 3B model, Granite Switch, next to Jev (via
[OpenRouter](https://openrouter.ai/typesafe)) and GPT Luna on the same context
and the same questions, and scores all three against an answer key.

The page has two tabs. **Obstacle Course**, the one it opens on, mixes yes/no
and open-ended questions, and runs Granite Switch and GPT Luna only. **Sprint**
is all yes/no questions across all three models.

## The flow (Sprint)

1. **Provide a context.** Paste your own text or JSON, or click **Random
   Wikipedia article** to pull one. The button draws random English articles
   through the MediaWiki API, skips stubs, and fills the context box with one
   article's plain text. Articles longer than 20,000 characters are cut to the
   first 20,000, at a paragraph break.
2. **Provide the questions.** One yes/no question per line, each followed by
   its answer after the question mark: `…? Yes` or `…? No`. Write them
   yourself, or click **Generate questions** and OpenAI's open-weight
   gpt-oss-120b, on OpenRouter
   ([`openai/gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b)),
   writes 10 from the context. The answers are right there in the box, so
   they can be checked and edited before the race.
3. **Race the Models.** The context and questions go to Granite Switch, Jev
   and GPT Luna at the same time.
4. **Scoring.** A noul above 0.5 counts as a yes. The table shows the answer
   from each line and each model's noul with ✓ or ✗.

Six tiles summarize the race, one column per model: **end-to-end latency** on
top and, beneath it, **accuracy** against the answer key. While the models run,
each latency tile is a stopwatch showing that model's elapsed time; a model's
stopwatch stops when its answers are back. The latency tile of the first model
to finish turns green straight away, without waiting for the others. Both tabs
work this way.

A question with no answer after it still gets nouls but isn't scored.

Granite Switch is served by vLLM on **a single NVIDIA L4 GPU (24 GB)**, on a
Hugging Face Inference Endpoint.

## Obstacle Course

This tab is the same page as Sprint with a mix of question types, to show a model
that has to switch between a System One call and a written answer.

- **Questions.** A line ending `…? Yes` or `…? No` is a yes/no question. A line
  ending `…? Freeform` is an open-ended question. **Generate questions** asks
  for 10 yes/no questions and 5 open-ended ones from the context and puts them
  in a random order, so which positions are freeform changes every time.
- **Yes/no questions** go to each model's System One call and come back as a
  noul: Granite's uncertainty adapter (c(yes)), and Luna's Decisions API.
- **Freeform questions** go to a chat completion and come back as a written
  answer of up to 20 tokens: Granite's base model on the same endpoint, and
  OpenAI's chat completions API for Luna, with `reasoning_effort` set to
  `none` (Luna's default is `medium`; Granite's base model doesn't reason).
- **Jev isn't run.** It returns decisions only and can't take freeform
  questions. It stays in the tiles and the table, greyed out and marked N/A.
- **Scoring.** Accuracy counts the yes/no questions only. Written answers are
  shown in the table and aren't graded.

**One at a time.** Each model works through the questions in order, and a
question isn't sent until the answer to the one before it has come back. So
every question is its own request: an adapter call or a chat completion for
Granite, and a Decisions request or a chat completion for Luna. The two models
run side by side, and each one's latency is the time its own run took. The
table fills in as answers arrive; the latency and accuracy tiles are worked out
at the end. On Granite, every request after the first still reuses the cached
context.

### Technologies inside

- **Granite Switch.** One checkpoint that bundles IBM's Granite 4.1 base model
  with 12 embedded adapter functions (RAG, safety, uncertainty and more), each
  selected per request by name.
  [Model card](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) ·
  [GitHub](https://github.com/generative-computing/granite-switch) ·
  [Adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html)
- **Uncertainty quantification (UQ) adapter.** A calibrated adapter that scores
  how likely an answer is to be correct: of the answers it scores at X%, about
  X% are right.
  [Adapter README](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/uncertainty/README.md)
- **aLoRA (activated LoRA).** Adapters that switch on at a trigger token and
  reuse the base model's KV cache for everything before it, so Granite reads
  the context once and every question reuses that work.
  [Paper (NeurIPS 2025)](https://arxiv.org/abs/2504.12397) ·
  [Code](https://github.com/IBM/activated-lora)
- **Optimized vLLM kernels.** Granite Switch's vLLM integration, with kernels
  optimized by the Granite team, applies adapter weights per token position
  rather than per request, so adapter and base-model requests share batches and
  one KV cache.

## How questions are generated

The Space sends one chat request to OpenRouter's chat completions API, using
the OpenAI client and the model `openai/gpt-oss-120b`. The prompt asks
for exactly 10 lines, each a question, then a space, then `Yes` or `No`, with
about half of each answer, every question answerable from the document alone.

Three things keep it quick:

- **Fastest provider.** OpenRouter serves this model from several hosts with
  very different speeds. The request sets `provider.sort` to `throughput`, so
  it goes to the fastest one.
- **Low reasoning effort.** gpt-oss-120b reasons before it answers. The request
  sets `reasoning.effort` to `low`, which is enough for this task.
- **Streaming.** The reply is streamed, and each question shows up in the box
  as soon as its line is complete.

The reply is parsed line by line; numbering, bullets and repeated questions are
dropped, so a run can end up with fewer than 10. The status line under the
button says how many questions were produced and how long it took. On the
Obstacle Course, the questions stream in the order the model writes them and
are shuffled once they're all in.

## How Granite produces a noul


Granite Switch runs on vLLM on a Hugging Face Inference Endpoint (see the
[granite-demos README](https://github.com/ksoule-ai/granite-demos#readme) for
how that endpoint is built). The Space calls it with the OpenAI client and
selects the embedded adapter by name through `chat_template_kwargs`.

Granite never answers the question itself. The prompt ends with the
instruction "Reply with exactly one word, 'Yes' or 'No'.", the answer is
prefilled as **"Yes"**, and Granite Switch's embedded `uncertainty` adapter
scores it once.

The noul is **c(yes)**: the adapter's certainty that "Yes" is the correct
answer.

The adapter scores ten bins (0.05, 0.15, … 0.95) for how likely a prefilled
answer is to be correct, and the certainty is the probability-weighted average
of those bins, so the noul always falls between 0.05 and 0.95.

**How it compares** on 82 hand-labeled questions across all seven examples
(scored at a 0.5 cutoff; AUC = how well the scores rank yes above no; Brier =
how well calibrated the probabilities are, lower is better):

| Granite noul | Correct | AUC | Brier |
|---|---|---|---|
| **UQ adapter, c(yes) on a prefilled "Yes."** (this demo's method) | **52/82** | **0.83** | **0.198** |
| UQ adapter, c(yes) / (c(yes) + c(no)) | 74/82 | 0.93 | 0.222 |
| Answer-token P(yes) / (P(yes) + P(no)) (Thinking Fast and Slow with Granite) | 77/82 | 0.99 | 0.041 |
| Jev (reference) | 82/82 | 1.00 | 0.001 |

Those rows were measured in the original demo. c(yes) ranks yes above no
reasonably well (AUC 0.83), but the adapter's certainty drifts from one input
to another, so 0.5 isn't a reliable yes/no boundary, and accuracy at that
cutoff suffers. Dividing by c(yes) + c(no) corrects for the drift at the cost
of a second adapter call per question.

## One generated token per adapter call

The adapter always replies `{"score": "N"}`, where the digit N is the only
part that carries information. That reply is 7 tokens, and on an L4 each
generated token costs about 33 ms. So the request prefills the adapter's reply
up to the digit (`{"score": "`) and asks vLLM to continue it
(`continue_final_message`). The model generates exactly one token, and its
top-10 logprobs give the certainty: keep the digit candidates, renormalize,
and take the expected value of their mapped certainties.

This is the same conversation and decoding that Mellea's
`core.check_certainty` uses, minus the 6 fixed tokens. The adapter's settings
(invocation text, score field, digit-to-certainty mapping) are read from the
model repo's `io_configs/uncertainty/io.yaml`, the file Mellea reads too.
Measured on an L4 over 34 questions:

| | Mellea `check_certainty` path | 1-token path |
|---|---|---|
| Median latency per call | 281 ms | 72 ms |
| Certainty difference | — | median 0.0014, max 0.0088 |

## Batching on the endpoint

The prompt is laid out so every question shares one long prefix:

```
<context>

Reply with exactly one word, 'Yes' or 'No'.
<question>
```

The uncertainty adapter is an aLoRA: it activates only at its invocation
token and reads the base model's KV cache for everything before it. That lets
vLLM's prefix cache compute the state once and reuse it for every question:

1. **Prime.** Question 1 goes alone, so vLLM computes and caches the shared
   prefix. Requests scheduled in the same step can't share blocks that are
   still being computed, so priming beats sending everything at once.
2. **Fan out.** The other questions go concurrently, one thread each, sharing
   one OpenAI client. vLLM batches them, and each only reads its own question
   and the prefilled "Yes".

The page reports vLLM's `cached_tokens` for the fan-out, so the reuse is
visible. vLLM caches in 16-token blocks, so a state shorter than one block
gets no reuse.

## Warm start

The endpoint scales to zero after 15 idle minutes (HF's minimum), and a cold
start takes about 3–5 minutes. So that visitors don't sit through that after
pressing **Race the Models**, one shared background warmer handles it:

1. **Wake on page load.** Opening the page probes the endpoint's `/models`
   route. If it's asleep, that request starts it, and the warmer keeps polling
   while the visitor reads and types.
2. **Warm-up request.** Once the endpoint answers, the warmer sends one
   throwaway uncertainty-adapter call, so the first timed run doesn't pay for
   opening the connection or the first adapter call.
3. **Status line.** The page shows *checking / asleep, waking / warming /
   ready*, refreshed every 3 seconds.

"Ready" expires 10 minutes after the last use, safely inside the 15-minute
scale-down window. After that the next visitor re-checks rather than trusting
an endpoint that may have gone to sleep. The run button waits on the same warmer,
and any wait is reported separately, not counted in Granite's time. Only one
wake/warm pass runs at a time, however many tabs are open.

The Jev side calls `TypeSafeClient.system_one(...)` with one `Noul` per
question and reads `response.nouls[key].noul`. All questions go in one
request. The client points at OpenRouter's System One API
(`base_url="https://openrouter.ai/api"`), which the TypeSafe SDK supports
as-is.

The GPT Luna side sends one `POST https://api.openai.com/v1/decisions` request
with the context as `input` and every question as a `predicate`, then reads
each answer's `probability` as the noul. The Decisions API is in public beta,
so it's called over plain HTTP rather than through the SDK. A question Luna
refuses comes back without a probability; it shows as "—" and isn't scored.

## Setup

1. Create a Gradio Space (free CPU hardware is enough) and push this folder
   to it.
2. **Settings → Variables and secrets**, add these secrets:
   - `HF_ENDPOINT_URL`: the Granite Switch endpoint URL, ending in `/v1`.
   - `HF_TOKEN`: a token allowed to call that endpoint.
   - `OPENROUTER_API_KEY`: your OpenRouter API key, used for both Jev and
     question generation. Without it, questions can't be generated and the Jev
     column shows as unavailable.
   - `QUESTION_MODEL` (optional): the OpenRouter model that writes the
     questions. Defaults to `openai/gpt-oss-120b`.
   - `OPENAI_API_KEY`: your OpenAI API key, for GPT Luna. Without it, the Luna
     column shows as unavailable.
   - `LUNA_MODEL` (optional): defaults to `gpt-6-luna`.
   - `LUNA_CHAT_MODEL` (optional): the OpenAI model for Luna's freeform answers
     on the Obstacle Course. Defaults to `LUNA_MODEL`.
   - `MAX_CONTEXT_CHARS` (optional): the cut-off for a random article's text.
     Defaults to `20000`.
   - `MODEL_ID` (optional): defaults to
     `ibm-granite/granite-switch-4.1-3b-preview`.
   - `JEV_MODEL` (optional): defaults to `jev-1.13` (routed to
     `typesafe/jev-1.13`).

## Caveats

- Wikipedia articles are public, so the hosted models may answer some
  questions from what they already know rather than from the context.
- Only the first part of a long random article is used.
- Accuracy is measured against the answers in the questions box. Generated
  answers come from gpt-oss-120b and can be wrong or ambiguous. Treat the
  result as a comparison between the models on the same questions, not as a
  benchmark score.
- The Granite noul is an emulation: a certainty score from an adapter built to
  judge whether an answer is correct, not a model trained to output calibrated
  decisions. How it compares with Jev is exactly what this demo is meant to
  test. Its yes/no boundary isn't reliably at 0.5, and it always falls between
  0.05 and 0.95.
- Priming adds one round trip before the fan-out. With one question there's
  nothing to share. Jev answers every question in one call.
- The warmer shortens cold starts but can't skip them. If someone clicks
  the run button within the first few minutes of opening the page after idle,
  they still wait for the rest of the wake-up (up to ~7 minutes).
