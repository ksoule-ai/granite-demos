---
title: ZeroGPU Smoke Test
emoji: 🔦
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - zerogpu
---

# ZeroGPU Smoke Test

Minimal Space to verify ZeroGPU allocation before deploying the full
[Granite Switch 4.1 30B demo](https://github.com/ksoule-ai/granite-demos).
One button: grabs a ZeroGPU slot, reports device name / VRAM / CUDA version,
and times a matmul.

Interpreting the result for the 30B demo:

| Reported VRAM | Meaning |
|---|---|
| ~96 GiB | `xlarge` — runs the 30B in BF16 as-is |
| ~48 GiB | `large` — 30B needs 8-bit/4-bit quantization |
| FAIL / no CUDA | ZeroGPU not enabled on this Space (check Settings → Hardware) |
