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

- **⚡ Thinking Fast:** a yes/no call as a noul, calculated from Granite's
  one-token answer as P('yes')/(P('yes')+P('no'))
- **🐢 Thinking Slow:** a written answer plus Granite's certainty in it, built
  with [Mellea](https://mellea.ai). Jev returns decisions only; it doesn't
  generate text.

Granite Switch is served by vLLM on **a single NVIDIA L4 GPU (24 GB)**, on a
Hugging Face Inference Endpoint. Both modes run on that one GPU.

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
  reuse the base model's KV cache for everything before it, so checking the
  certainty of an answer reuses the work Granite already did reading the input
  and writing that answer.
  [Paper (NeurIPS 2025)](https://arxiv.org/abs/2504.12397) ·
  [Code](https://github.com/IBM/activated-lora)
- **Optimized vLLM kernels.** Granite Switch's vLLM integration, with kernels
  optimized by the Granite team, applies adapter weights per token position
  rather than per request, so adapter and base-model requests share batches and
  one KV cache.

## How Thinking Fast produces a noul

Granite Switch runs on vLLM on a Hugging Face Inference Endpoint (see the
[granite-demos README](https://github.com/ksoule-ai/granite-demos#readme) for
how that endpoint is built). The Space calls it with the OpenAI client.

The prompt ends with the instruction "Reply with exactly one word, 'Yes' or
'No'.", so the first token Granite generates *is* its answer. The request
asks for just that one token, with its probabilities (`logprobs`, top 20
alternatives):

1. Convert each candidate token's log-probability to a probability.
2. Sum the spellings: "Yes", "yes" and " yes" are separate tokens, and so are
   the "No" variants.
3. Renormalize over the two: **noul = P(yes) / (P(yes) + P(no))**.

In practice virtually all the probability lands on yes or no tokens, so
nothing meaningful is lost by keeping only the top 20.

**How it compares** on 82 hand-labeled questions across all seven examples
(scored at a 0.5 cutoff; AUC = how well the scores rank yes above no; Brier =
how well calibrated the probabilities are, lower is better):

| Granite noul | Correct | AUC | Brier |
|---|---|---|---|
| UQ adapter, c(yes) on a prefilled "Yes." (earlier version) | 52/82 | 0.83 | 0.198 |
| UQ adapter, c(yes) / (c(yes) + c(no)) (earlier version) | 74/82 | 0.93 | 0.222 |
| **Answer-token P(yes) / (P(yes) + P(no))** (this demo) | **77/82** | **0.99** | **0.041** |
| Jev (reference) | 82/82 | 1.00 | 0.001 |

The uncertainty adapter is built to judge whether an answer is correct, and it
does that well on Granite's written answers (see Thinking Slow). Scoring a bare
prefilled "Yes" or "No" was a weaker yes/no signal than the answer token
itself, which is also faster: one generated token per question.

## Batching on the endpoint

The prompt is laid out so every question shares one long prefix:

```
<state>

Reply with exactly one word, 'Yes' or 'No'.
<question>
```

vLLM's prefix cache computes the state once and reuses it for every question:

1. **Prime.** Question 1 goes alone, so vLLM computes and caches the shared
   prefix. Requests scheduled in the same step can't share blocks that are
   still being computed, so priming beats sending everything at once.
2. **Fan out.** The other questions go concurrently, one thread each, sharing
   one OpenAI client. vLLM batches them, and each only reads its own question.

The page reports vLLM's `cached_tokens` for the fan-out, so the reuse is
visible. vLLM caches in 16-token blocks, so a state shorter than one block
gets no reuse.

## The two buttons

Two buttons under the questions each start a different kind of thinking on the
same state, straight away: **⚡ Thinking Fast** and **🐢 Thinking Slow**. The
results panel switches to match, under a header naming the kind of thinking
that ran. Large tiles under the header show the end-to-end times (Granite's
and Jev's) side by side, plus **agreement with Jev**: of the
questions Jev calls yes (noul > 0.5), how many Granite also calls yes, and the
same for no. Granite's yes/no is its noul > 0.5 in Thinking Fast, and the
opening "Yes"/"No" of its written response in Thinking Slow; responses without
a clear yes or no are left out of the count and noted on the tile.

- **⚡ Thinking Fast**: yes/no questions. One table shows Granite's noul and,
  in a **Jev noul (reference)** column, Jev's, as the comparison baseline
  rather than part of the Granite stack.
- **🐢 Thinking Slow**: the same yes/no questions, or any free-form question
  you type. Granite writes a response, then scores its own certainty in it
  (below). The table shows **Question | Granite response | Granite Certainty |
  Jev noul (reference)**; Jev answers the same questions in parallel as a
  reference column, since it returns decisions only and doesn't generate
  text.

All preset examples use yes/no questions, so the same example works with both
buttons.

## Thinking Slow

**🐢 Thinking Slow** has Granite write an answer, then score its own certainty
in that answer. It's built with [Mellea](https://mellea.ai), driving the same
endpoint through `OpenAIBackend(load_embedded_adapters=True)`. For each
question:

1. `mfuncs.chat` runs the slow prompt on the base model and returns a
   `ChatContext` holding the question and Granite's answer. The prompt is just
   the state, then the question, with no instruction added. The state still
   comes first, so it shares the cached state prefix with Thinking Fast.
2. `core.check_certainty` runs the UQ adapter on that context: the model's
   certainty that *its own answer* is correct (not a prefilled "Yes" or "No").
3. Each result (question, answer, certainty) becomes a row in the Thinking
   Slow table.

The certainty call is cheap because the adapter is an aLoRA. It reuses the KV
cache for the question *and* the answer, and only generates the score. A
1-token Mellea call caches the shared state prefix first; then every question
runs in parallel, and the table fills in as each one finishes. Answers are
capped at 512 tokens.

On the 50-question stress example, Granite's written answers were right on
44 of 50, and the adapter's certainty separated them: 0.69 on average when
right, 0.30 when wrong (only 2 wrong answers, so treat that as indicative).

## Warm start

The endpoint scales to zero after 15 idle minutes (HF's minimum), and a cold
start takes about 3–5 minutes. So that visitors don't sit through that after
clicking a button, one shared background warmer handles it:

1. **Wake on page load.** Opening the page probes the endpoint's `/models`
   route. If it's asleep, that request starts it, and the warmer keeps polling
   while the visitor reads and types.
2. **Warm-up request.** Once the endpoint answers, the warmer sends one
   throwaway Thinking Fast call, so the first timed run doesn't pay for
   opening the connection.
3. **Status line.** The page shows *checking / asleep, waking / warming /
   ready*, refreshed every 3 seconds.

"Ready" expires 10 minutes after the last use, safely inside the 15-minute
scale-down window. After that the next visitor re-checks rather than trusting
an endpoint that may have gone to sleep. Both buttons wait on the same warmer,
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

- The Granite noul is an emulation: the probability a general-purpose LLM
  puts on its one-word answer, not a model trained to output calibrated
  decisions. How it compares with Jev is exactly what this demo is meant to
  test. It tends to be confident, so when it's wrong it can be confidently
  wrong.
- Priming adds one round trip before the fan-out. With one question there's
  nothing to share. Jev answers every question in one call.
- The warmer shortens cold starts but can't skip them. If someone clicks
  a button within the first few minutes of opening the page after idle,
  they still wait for the rest of the wake-up (up to ~7 minutes).
