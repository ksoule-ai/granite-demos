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

Granite never answers the question itself. For each question, on ZeroGPU,
`granite-switch-4.1-3b-preview` makes two separate calls:

1. **c(yes).** Prefill the assistant turn with "Yes." and run Mellea's
   `core.check_certainty`, which calls Granite Switch's embedded
   `uncertainty` adapter.
2. **c(no).** Prefill "No." and run the adapter again.
3. **Normalize.** `noul = c(yes) / (c(yes) + c(no))`.

The adapter scores ten bins (0.05, 0.15, … 0.95) for how likely the
prefilled answer is to be correct, and Mellea returns the probability-weighted
average of those bins. Each certainty therefore falls between 0.05 and 0.95,
and so does the noul. The two checks are independent, so c(yes) and c(no)
needn't sum to 1; normalizing turns them into a single probability of yes.

The Jev side calls `TypeSafeClient.system_one(...)` with one `Noul` per
question and reads `response.nouls[key].noul`. All questions go in one
request. The client points at OpenRouter's System One API
(`base_url="https://openrouter.ai/api"`), which the TypeSafe SDK supports
as-is.

## Setup

1. Create a Gradio Space and push this folder to it.
2. **Settings → Hardware → ZeroGPU.** This needs a PRO account, or an
   organization on a Team or Enterprise plan.
3. **Settings → Secrets:**
   - `OPENROUTER_API_KEY`: your OpenRouter API key. Without it, the Granite
     column still works and the Jev column shows as unavailable.
   - `JEV_MODEL` (optional): defaults to `jev-1.13` (routed to
     `typesafe/jev-1.13`).

## Caveats

- The Granite noul is an emulation. The uncertainty adapter was trained to
  judge whether an answer is correct, not to produce yes/no probabilities, so
  its calibration against Jev is exactly what this demo is meant to test.
- Granite makes two adapter calls per question (prefilled yes, then no), so
  more questions take longer. Jev answers every question in one call.
- Granite time is GPU compute only and doesn't include ZeroGPU queueing. Jev
  time is the full API round trip.
