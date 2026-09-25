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
how that endpoint is built). Mellea talks
to it through `OpenAIBackend(load_embedded_adapters=True)`, which selects each
embedded adapter by name on the request. For each question:

1. **Answer.** Mellea's `mfuncs.chat(..., format=YesNo)` constrains
   `granite-switch-4.1-3b-preview` to answer exactly `yes` or `no`.
2. **Score.** Mellea's `core.check_certainty` runs Granite Switch's embedded
   `uncertainty` adapter over that question-and-answer pair. The adapter
   scores ten bins (0.05, 0.15, … 0.95) for how likely the answer is to be
   correct, and Mellea returns the probability-weighted average of those
   bins, so certainty always falls between 0.05 and 0.95.
3. **Fold.** `noul = certainty` if the answer is yes, otherwise `1 − certainty`.

**Batching:** all questions run concurrently, one thread each. Mellea's sync
calls share a single background event loop, so the requests overlap on the
wire and vLLM batches them together. Each question's two steps stay in
order, because the adapter scores the answer from step 1.

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
- Granite makes two dependent requests per question (answer, then adapter).
  Questions run in parallel, but each still takes two round trips. Jev
  answers every question in one call.
- The endpoint scales to zero. The first run after idle waits (up to ~7
  minutes) for it to wake; that wait is reported separately and not counted
  in Granite's time.
