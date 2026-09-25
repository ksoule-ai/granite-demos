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
short_description: Open-model nouls (Granite Switch + Mellea) next to Jev's
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
how that endpoint is built). Mellea talks to it through
`OpenAIBackend(load_embedded_adapters=True)`, which selects each embedded
adapter by name on the request.

Granite never answers the question itself. For each question it makes one
call: the assistant turn is prefilled with "Yes." and Granite Switch's
embedded `uncertainty` adapter scores it. That certainty, c(yes), is the
probability of yes, which is the noul.

The adapter scores ten bins (0.05, 0.15, … 0.95) for how likely the
prefilled answer is to be correct, and Mellea returns the probability-weighted
average of those bins, so the noul always falls between 0.05 and 0.95.

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
2. **Fan out.** The remaining questions go concurrently (one thread each;
   Mellea's sync calls share a single background event loop, so the requests
   overlap). vLLM batches them, and each only reads its own question and the
   "Yes." turn.

The page reports vLLM's `cached_tokens` for the fan-out, so the reuse is
visible. vLLM caches in 16-token blocks, so a state shorter than one block
gets no reuse.

The adapter call goes through `mfuncs.act` with an `Intrinsic`, the same path
`core.check_certainty` uses, so the token usage on the model output is kept.

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
- The endpoint scales to zero. The first run after idle waits (up to ~7
  minutes) for it to wake; that wait is reported separately and not counted
  in Granite's time.
