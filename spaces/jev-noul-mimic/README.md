---
title: Thinking Fast and Slow with Granite
emoji: 🧠
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
short_description: Fast yes/no calls and slow answers from one Granite endpoint
---

# Thinking Fast and Slow with Granite

Some decisions don't need reasoning out loud. They need a fast, calibrated gut
call. That's the idea behind *System One* models like TypeSafe AI's
[Jev](https://docs.typesafe.ai): instead of text, Jev answers a yes/no question
with a **noul**, the probability that the answer is yes.

One open 3B model, Granite Switch, does both from a single endpoint, next to
Jev (via [OpenRouter](https://openrouter.ai/typesafe)) on the same input:

- **⚡ Thinking Fast:** a yes/no call as a noul, from two one-token adapter
  calls per question, side by side with Jev.
- **🐢 Thinking Slow:** a written answer, like any LLM. Jev returns decisions
  only; it doesn't generate text.
- **🧠 Compound Thinking:** a written answer plus Granite's certainty in it,
  returned as JSON, built with [Mellea](https://mellea.ai).

Granite Switch is served by vLLM on **a single NVIDIA L4 GPU (24 GB)**, on a
Hugging Face Inference Endpoint. All three modes run on that one GPU.

### Technologies inside

- **Granite Switch.** One checkpoint that bundles IBM's Granite 4.1 base model
  with 12 embedded adapter functions (RAG, safety, uncertainty and more), each
  selected per request by name.
  [Model card](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) ·
  [GitHub](https://github.com/generative-computing/granite-switch) ·
  [Adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html)
- **Uncertainty quantification (UQ) adapter.** A calibrated adapter that scores
  how likely an answer is to be correct: of the answers it scores at X%, about
  X% are right. Score a prefilled "Yes" and a prefilled "No", and
  c(yes) / (c(yes) + c(no)) is a noul; score Granite's own written answer and
  you get Compound Thinking.
  [Adapter README](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/uncertainty/README.md)
- **aLoRA (activated LoRA).** Adapters that switch on at a trigger token and
  reuse the base model's KV cache for everything before it, so Granite reads the
  input once and every question, fast or slow, reuses that work.
  [Paper (NeurIPS 2025)](https://arxiv.org/abs/2504.12397) ·
  [Code](https://github.com/IBM/activated-lora)
- **Optimized vLLM kernels.** Granite Switch's vLLM integration, with kernels
  optimized by the Granite team, applies adapter weights per token position
  rather than per request, so adapter and base-model requests share batches and
  one KV cache.
  [aLoRA vs LoRA live race](https://generative-computing.github.io/granite-switch/race_live.html) ·
  [vLLM](https://github.com/vllm-project/vllm)

## How the Granite side produces a noul

Granite Switch runs on vLLM on a Hugging Face Inference Endpoint (see the
[granite-demos README](https://github.com/ksoule-ai/granite-demos#readme) for
how that endpoint is built). The Space calls it with the OpenAI client and
selects the embedded adapter by name through `chat_template_kwargs`.

Granite never answers the question itself. The prompt ends with the
instruction "Reply with exactly one word, 'Yes' or 'No'.", and for each
question Granite Switch's embedded `uncertainty` adapter runs twice:

1. On a prefilled **"Yes"**: c(yes), its certainty that yes is correct.
2. On a prefilled **"No"**: c(no), its certainty that no is correct.

The noul is **c(yes) / (c(yes) + c(no))**.

Why normalize: the adapter's certainty drifts from one input to another, so
c(yes) alone has no fixed yes/no boundary, and a single global rescaling
fitted on one set of inputs made others worse. c(no) on the same input barely
depends on the true answer, but it tracks that drift, so dividing by
c(yes) + c(no) cancels it and puts the boundary back at 0.5.

The adapter scores ten bins (0.05, 0.15, … 0.95) for how likely a prefilled
answer is to be correct, and each certainty is the probability-weighted average
of those bins, so c(yes), c(no) and the noul all fall between 0.05 and 0.95.

**How it compares** on 82 hand-labeled questions across all seven examples
(scored at a 0.5 cutoff; AUC = how well the scores rank yes above no):

| Granite noul | Correct | AUC |
|---|---|---|
| c(yes) alone, prefilled "Yes." (earlier version) | 52/82 | 0.83 |
| **c(yes) / (c(yes) + c(no))** (this demo) | **74/82** | **0.93** |
| Jev (reference) | 82/82 | 1.00 |

The normalized noul gets the direction right far more often, but its values
stay fairly close to 0.5, so read them as a ranking and a lean rather than as
sharply calibrated probabilities.

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
<state>

Answer the following question with 'yes' or 'no'.
<question>
```

The uncertainty adapter is an aLoRA: it activates only at its invocation
token and reads the base model's KV cache for everything before it. That lets
vLLM's prefix cache compute the state once and reuse it for every question:

1. **Prime.** Question 1's "Yes" call goes alone, so vLLM computes and caches
   the shared prefix. Requests scheduled in the same step can't share blocks
   that are still being computed, so priming beats sending everything at once.
2. **Fan out.** The other 2N − 1 adapter calls (the rest of the "Yes" and "No"
   calls) go concurrently, one thread each, sharing one OpenAI client. vLLM
   batches them, and each only reads its own question and prefilled answer.

The page reports vLLM's `cached_tokens` for the fan-out, so the reuse is
visible. vLLM caches in 16-token blocks, so a state shorter than one block
gets no reuse.

## Thinking Slow

Three buttons under the questions each start a different kind of thinking on
the same state, straight away: **⚡ Thinking Fast**, **🐢 Thinking Slow** and
**🧠 Compound Thinking** (below). The results panel switches to match, under a
header naming the kind of thinking that ran. Large tiles under the header show
the end-to-end time: Granite's, plus Jev's (light grey) in Thinking Fast, side
by side so the two are easy to compare.

- **⚡ Thinking Fast**: yes/no questions. One table shows Granite's noul and,
  in a **Jev noul (reference)** column, Jev's. Everything Jev (that column, its
  end-to-end tile and its timing line) is light grey, so it reads as the
  comparison baseline rather than part of the Granite stack.
- **🐢 Thinking Slow**: the same yes/no questions, or any free-form question you type.
  Granite's base model, with no adapter, writes an answer to each, streamed as
  it's generated, so you can read its reasoning next to the noul it gave in
  fast mode. Jev isn't shown here: it returns decisions only and doesn't
  generate text.

All preset examples use yes/no questions, so the same example works with every
button.

Both modes hit the same endpoint and the same weights. The slow prompt is just
the state, then the question, with no instruction added. The state still comes
first, so it shares the cached state prefix with fast mode. It batches
the same way: question 1 runs alone until its first token arrives (so the
prefix is cached), then the rest stream in parallel. Answers are capped at 512
tokens.

## Compound Thinking

**🧠 Compound Thinking** combines the two: Granite thinks slow, then scores its own
answer. It's built with [Mellea](https://mellea.ai), driving the same endpoint
through `OpenAIBackend(load_embedded_adapters=True)`. For each question:

1. `mfuncs.chat` runs the slow prompt (state, then question) on the base model
   and returns a `ChatContext` holding the question and Granite's answer.
2. `core.check_certainty` runs the UQ adapter on that context: the model's
   certainty that *its own answer* is correct (not a prefilled "Yes" or "No").
3. The result is a Pydantic `CompoundAnswer`, and the page returns the whole
   response as JSON:

```json
{
  "mode": "compound",
  "model": "ibm-granite/granite-switch-4.1-3b-preview",
  "results": [
    {"question": "…", "answer": "…", "certainty": 0.83}
  ]
}
```

The certainty call is cheap because the adapter is an aLoRA. It reuses the KV
cache for the question *and* the answer, and only generates the score.
Mellea's chat call isn't streamed here, so instead of priming on question 1's
first token, a 1-token Mellea call caches the shared state prefix first. Then
every question runs in parallel, and the JSON fills in as each one finishes.

## Warm start

The endpoint scales to zero after 15 idle minutes (HF's minimum), and a cold
start takes about 3–5 minutes. So that visitors don't sit through that after
clicking **Compare**, one shared background warmer handles it:

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
an endpoint that may have gone to sleep. **Compare** waits on the same warmer,
and any wait is reported separately, not counted in Granite's time. Only one
wake/warm pass runs at a time, however many tabs are open.

The Jev side calls `TypeSafeClient.system_one(...)` with one `Noul` per
question and reads `response.nouls[key].noul`. All questions go in one
request. The client points at OpenRouter's System One API
(`base_url="https://openrouter.ai/api"`), which the TypeSafe SDK supports
as-is.

## Setup

1. Create a Gradio Space (free CPU hardware is enough) and push this folder
   to it.
2. **Settings → Variables and secrets**, add these secrets:
   - `HF_ENDPOINT_URL`: the Granite Switch endpoint URL, ending in `/v1`.
   - `HF_TOKEN`: a token allowed to call that endpoint.
   - `OPENROUTER_API_KEY`: your OpenRouter API key. Without it, the Granite
     column still works and the Jev column shows as unavailable.
   - `MODEL_ID` (optional): defaults to
     `ibm-granite/granite-switch-4.1-3b-preview`.
   - `JEV_MODEL` (optional): defaults to `jev-1.13` (routed to
     `typesafe/jev-1.13`).

## Caveats

- The Granite noul is an emulation. The uncertainty adapter was trained to
  judge whether an answer is correct, not to produce yes/no probabilities, so
  its calibration against Jev is exactly what this demo is meant to test.
- Priming adds one round trip before the fan-out. With one question there's
  nothing to share. Jev answers every question in one call.
- The warmer shortens cold starts but can't skip them. If someone clicks
  **Compare** within the first few minutes of opening the page after idle,
  they still wait for the rest of the wake-up (up to ~7 minutes).
