---
title: Granite Switch nouls vs Jev
emoji: ⚖️
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
short_description: Open-model nouls (Granite Switch) next to Jev's
---

# Granite Switch nouls vs. Jev

[Jev](https://docs.typesafe.ai) is TypeSafe AI's *System One* model. A **noul** is
its yes/no primitive: one calibrated number in [0, 1], the probability that the
answer is yes.

This Space rebuilds that primitive from open parts and shows it next to the real
Jev model, called through [OpenRouter](https://openrouter.ai/typesafe), on the
same input.

## How the Granite side produces a noul

Granite Switch runs on vLLM on a Hugging Face Inference Endpoint (see the
[granite-demos README](https://github.com/ksoule-ai/granite-demos#readme) for
how that endpoint is built). The Space calls it with the OpenAI client and
selects the embedded adapter by name through `chat_template_kwargs`.

Granite never answers the question itself. For each question it makes one
call: the assistant turn is prefilled with "Yes." and Granite Switch's
embedded `uncertainty` adapter scores it. That certainty, c(yes), is the
probability of yes, which is the noul.

The adapter scores ten bins (0.05, 0.15, … 0.95) for how likely the
prefilled answer is to be correct, and the noul is the probability-weighted
average of those bins, so it always falls between 0.05 and 0.95.

## One generated token per question

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

1. **Prime.** The first question goes alone, so vLLM computes and caches the
   shared prefix. Requests scheduled in the same step can't share blocks that
   are still being computed, so priming beats sending everything at once.
2. **Fan out.** The remaining questions go concurrently (one thread each,
   sharing one OpenAI client). vLLM batches them, and each only reads its own
   question and the "Yes." turn.

The page reports vLLM's `cached_tokens` for the fan-out, so the reuse is
visible. vLLM caches in 16-token blocks, so a state shorter than one block
gets no reuse.

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
