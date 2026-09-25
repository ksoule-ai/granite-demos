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
Jev API on the same input.

## How the Granite side produces a noul

For each question, on ZeroGPU:

1. **Answer.** Mellea's `mfuncs.chat(..., format=YesNo)` constrains
   `granite-switch-4.1-3b-preview` to answer exactly `yes` or `no`.
2. **Score.** Mellea's `core.check_certainty` runs Granite Switch's embedded
   `uncertainty` adapter over that question-and-answer pair. It returns a
   calibrated probability that the answer is correct, in ten bins
   (0.05, 0.15, … 0.95).
3. **Fold.** `noul = certainty` if the answer is yes, otherwise `1 − certainty`.

The Jev side calls `TypeSafeClient.system_one(...)` with one `Noul` per
question and reads `response.nouls[key].noul`. All questions go in one
request.

## Setup

1. Create a Gradio Space and push this folder to it.
2. **Settings → Hardware → ZeroGPU.** This needs a PRO account, or an
   organization on a Team or Enterprise plan.
3. **Settings → Secrets:**
   - `TYPESAFE_API_KEY`: your Jev API key. Without it, the Granite column
     still works and the Jev column shows as unavailable.
   - `JEV_MODEL` (optional): defaults to `jev-latest`.

## Caveats

- The Granite noul is an emulation. The uncertainty adapter was trained to
  judge whether an answer is correct, not to produce yes/no probabilities, so
  its calibration against Jev is exactly what this demo is meant to test.
- Granite makes two forward passes per question (answer, then adapter), so
  more questions take longer. Jev answers every question in one call.
- Granite time is GPU compute only and doesn't include ZeroGPU queueing. Jev
  time is the full API round trip.
