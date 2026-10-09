---
title: Granite Switch 4.1 8B Demo
emoji: 🪨
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - granite
  - ibm
  - granite-switch
  - zerogpu
  - mellea
---

# granite-switch demos

Demos built on IBM's [Granite Switch](https://github.com/generative-computing/granite-switch)
preview checkpoints and their embedded adapters. There are five, in two groups.

## What's in this repo

**Serving Granite Switch** (documented in full below):

| Demo | What it shows | Where it lives |
|---|---|---|
| [Activity 1: Inference Endpoint](#activity-1--granite-switch-on-a-hugging-face-inference-endpoint) | Hosts `granite-switch-4.1-3b-preview` on a Hugging Face Inference Endpoint with a custom vLLM container, behind an OpenAI-compatible API, and calls the **requirement-check** adapter from raw Python. A proven deployment recipe for a preview architecture that stock vLLM can't load. | [`Dockerfile`](Dockerfile), [`docker/`](docker/), [`requirement_check.py`](requirement_check.py), [`ENDPOINT.md`](ENDPOINT.md) |
| [Activity 2: ZeroGPU Space](#activity-2--zerogpu-space-demo-mellea--hf-transformers-backend) | Drives `granite-switch-4.1-8b-preview` in-process on ZeroGPU through [Mellea](https://docs.mellea.ai), with an instruct–validate–repair loop, token streaming, and live KV-cache metrics. This is the app the Space frontmatter above points at. | [`app.py`](app.py), [`switch_backend.py`](switch_backend.py), [`tests/`](tests/) |

**Hugging Face Spaces** (each has its own README; summarized in
[Spaces](#spaces)):

| Demo | What it shows | Where it lives |
|---|---|---|
| [Thinking Fast and Slow with Granite](spaces/jev-noul-mimic/README.md) | One 3B model doing two kinds of thinking next to TypeSafe AI's Jev. **Thinking Fast** answers yes/no questions with a *noul* (the probability the answer is yes) read from Granite's one-token answer. **Thinking Slow** writes an answer, then scores its own certainty with the uncertainty (UQ) adapter through Mellea. | [`spaces/jev-noul-mimic/`](spaces/jev-noul-mimic/) |
| [Noul Race](spaces/noul-race/README.md) | Races Granite Switch against Jev and OpenAI's GPT Luna on the same context and questions, with live stopwatches and accuracy against an answer key. An **Obstacle Course** tab mixes yes/no and freeform questions, one at a time; a **Race** tab runs yes/no questions in parallel. | [`spaces/noul-race/`](spaces/noul-race/), [`scripts/create_noul_race_space.py`](scripts/create_noul_race_space.py) |
| [ZeroGPU smoke test](spaces/zerogpu-smoke/README.md) | A one-button Space that grabs a ZeroGPU slot and reports the device, VRAM and CUDA version. Used to check ZeroGPU allocation before deploying Activity 2. It doesn't call the endpoint. | [`spaces/zerogpu-smoke/`](spaces/zerogpu-smoke/), [`scripts/create_smoke_space.py`](scripts/create_smoke_space.py) |

Activities 1 and 2 use the same architecture; they differ in **serving model** (persistent
vLLM server vs. in-process Transformers) and therefore in what's easy: Activity 1 gives you
a standing OpenAI endpoint other apps can call, which is what the two noul Spaces do;
Activity 2 fits a single-checkpoint demo onto free ZeroGPU hardware.

## Background: what Granite Switch is

Granite Switch is **one checkpoint bundling ~12 embedded LoRA/aLoRA adapters** (RAG,
Core, Guardian). You select one per request **by name**, and the model's chat template
splices in a control token that activates that adapter's weights at inference — no
separate adapter files are loaded. For example, `requirement-check` judges whether an
assistant response satisfies user-specified constraints and returns
`{"score": "yes"|"no"}`; `uncertainty` scores the model's confidence; `guardian-core`
screens for harm.

Its model type, `granite_switch`, is **not** in stock vLLM or Transformers — it ships
only in the separate [`granite-switch`](https://github.com/generative-computing/granite-switch)
pip package. Both activities below install that package; `--trust-remote-code` does **not**
help, because the architecture lives in the package, not as remote code in the model repo.

---

# Activity 1 — Granite Switch on a Hugging Face Inference Endpoint

Host [`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)
on a Hugging Face Inference Endpoint (vLLM backend) and call its **requirement-check**
adapter from Python.

> **Already deployed?** See [ENDPOINT.md](ENDPOINT.md) for the live endpoint's
> URL, auth, and how to point other projects at it.

## Why this needs a custom container (read first)

If you deploy with HF's managed vLLM engine (or a plain `vllm/vllm-openai` image),
the endpoint builds but **fails to start** with:

```
Value error, The checkpoint you are trying to load has model type `granite_switch`
but Transformers does not recognize this architecture.
```

So a **custom container that installs `granite-switch` is required**, not optional.

## 1a. Build the custom container

The image is just stock vLLM plus the `granite-switch` package. See
[`docker/Dockerfile`](docker/Dockerfile):

```dockerfile
FROM vllm/vllm-openai:v0.19.1
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
ARG GRANITE_SWITCH_REF=756f946640d571a5beef2a49d4cab6614c61f18a
RUN git clone https://github.com/generative-computing/granite-switch.git /opt/granite-switch \
 && git -C /opt/granite-switch checkout "$GRANITE_SWITCH_REF" \
 && pip install "/opt/granite-switch[vllm]"
```

**Version note:** the `[vllm]` extra pins **vLLM 0.19.1** (CUDA 12.x), which matches
HF's L4 hosts. The `[vllm20]` extra needs **CUDA 13+**, which those hosts don't have
— don't use it here.

**Pin note:** the clone is pinned to granite-switch commit `756f946`. Commit
`6013c7f` (2026-09-10) dropped the SingleSwitch format that the
`granite-switch-4.1-3b-preview` checkpoint uses. The package version didn't change,
so an unpinned build fails at endpoint startup with a SingleSwitch/MultiSwitch
error. Move the pin forward only once you're serving a MultiSwitch checkpoint.

**Build with GitHub Actions, not locally.** A Mac builds `arm64`, which Inference
Endpoints rejects (it requires `linux/amd64`). GitHub's runners are `x86`, so the
image comes out correct automatically. This repo ships
[`.github/workflows/build.yml`](.github/workflows/build.yml), which builds on every
push touching `docker/` and pushes to GHCR.

After the workflow runs:

1. Open the repo's **Packages** (profile or repo sidebar) → the
   `granite-switch-vllm` image.
2. **Package settings → Change visibility → Public.** HF needs this to pull
   without registry credentials.

Your image URL is then:

```
ghcr.io/<your-lowercase-username>/granite-switch-vllm:latest
```

> GHCR image names must be lowercase. If your GitHub username has capitals, the tag
> in the workflow must use the lowercase form.

## 1b. Create the endpoint

1. [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co) → **New** →
   **Model repository** → `ibm-granite/granite-switch-4.1-3b-preview`. (HF mounts the
   selected model at `/repository` inside the container.)
2. **Hardware:** a single **L4 (24 GB)** is plenty for this 3–4B BF16 model. An A10G
   also works.
3. **Container type:** **Custom Container**, with:
   - **Image URL:** `ghcr.io/<your-lowercase-username>/granite-switch-vllm:latest`
   - **Port:** `8000`
   - **Health route:** `/health`
   - **Container Arguments:**
     ```
     --model /repository --served-model-name ibm-granite/granite-switch-4.1-3b-preview --host 0.0.0.0 --chat-template /repository/chat_template.jinja --enable-prompt-tokens-details
     ```
     `--model /repository` points vLLM at the mounted weights (not the Hub repo id).
     `--served-model-name` lets callers use the friendly id.
     `--chat-template /repository/chat_template.jinja` was required to import the custom chat template correctly.
     `--enable-prompt-tokens-details` turns on `cached_tokens` reporting (see Notes).
4. **Authentication:** choose **Authenticated** (a.k.a. Protected). Reachable over
   the internet by anyone holding a valid HF token — i.e. you (for experiments) and a
   Space demo (token held server-side as a secret) — but closed to the anonymous
   public. Do **not** choose Private (VPC/PrivateLink only, unreachable from your
   laptop or a Space).
5. **Autoscaling / cost:** set **min replicas 0** (scale-to-zero) and **max replicas 1**.
   The token authenticates callers but does **not** change billing — endpoint compute
   always bills to the owner regardless of who calls, so min-0/max-1 + scale-to-zero
   is what actually caps cost. First call after idle pays a ~1–3 min cold start.
6. Deploy and wait for **Running**. Your endpoint URL looks like
   `https://xxxx.us-east-1.aws.endpoints.huggingface.cloud`.

## 1c. Sanity checks after it's up

- **Status reaches Running** and the startup logs show vLLM loading the
  `granite_switch` architecture and prefix caching enabled (no "does not recognize
  this architecture" — if you see that, the image didn't install `granite-switch`;
  check the Actions build log).
- `GET {url}/v1/models` returns the model id; use that (or the `--served-model-name`
  value) as `MODEL_ID`.
- Quick smoke test (append `/v1` to the endpoint URL):
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="https://xxxx.endpoints.huggingface.cloud/v1",
                  api_key="hf_...your-token")
  r = client.chat.completions.create(
      model="ibm-granite/granite-switch-4.1-3b-preview",
      messages=[{"role": "user", "content": "What is the square root of 4?"}],
  )
  print(r.choices[0].message.content)
  ```

## 1d. Run the requirement-check CLI demo

```bash
cp .env.example .env      # set HF_ENDPOINT_URL (append /v1) and HF_TOKEN
pip install -r requirements.txt
python requirement_check.py
```

Expected output:

```
[good] requirements satisfied: True  (43 words)
[ bad] requirements satisfied: False  (11 words)
```

## How the adapter is invoked (raw API)

vLLM applies the model's chat template server-side, so the adapter is selected by
passing `adapter_name` through `chat_template_kwargs`:

```python
client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "user", "content": user_task},
        {"role": "assistant", "content": response_under_review},
        {"role": "user", "content": f"<requirements> {constraints}\n{EVALUATION_PROMPT}"},
    ],
    max_tokens=15,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"adapter_name": "requirement-check"}},
)
```

The final user turn carries the constraints (in a `<requirements>` tag) plus a fixed
evaluation instruction. The adapter replies with `{"score": "yes"}` or `{"score": "no"}`.
Other adapters work the same way — swap the `adapter_name` and follow each adapter's
message protocol.

## Notes (Activity 1)

- This activity uses the raw OpenAI-compatible API so there's no extra dependency and
  the mechanics are visible. IBM's preferred high-level client is
  [Mellea](https://mellea.ai) — that's the path Activity 2 takes.
- **KV-cache reporting:** prefix caching (APC) is on by default in this vLLM, but the
  `cached_tokens` field in the usage response is gated behind
  `--enable-prompt-tokens-details` (included in the args above). Caching still works
  without the flag — only the reporting is gated. It counts **full 16-token blocks**
  only, so a short shared prefix can read 0 even on a real hit; test with a long shared
  prefix or a multi-turn exchange.
- **aLoRA KV sharing:** switching adapters over a shared context is *designed* to reuse
  the base model's KV cache. Whether raw, stateless OpenAI-API calls realize that
  cross-adapter reuse (vs. vLLM keying the cache per adapter id) is an open
  implementation detail — a Mellea-managed session is the reliable way to get it. Since
  this activity is raw-API, don't assume cross-adapter reuse without measuring prefill
  tokens / TTFT.

## Troubleshooting (Activity 1)

- **"does not recognize this architecture" at startup** → the image is stock vLLM, not
  the custom one, or the `granite-switch` install failed. Check the Actions build log
  and the endpoint's image URL.
- **KV-cache / out-of-memory error on load** → Granite 4.1's base context is large, so
  the full-context KV cache may not fit on 24 GB. Add `--max-model-len 16384` (or lower)
  to the Container Arguments to shrink it.
- **Endpoint builds but never goes healthy** → confirm Port `8000` and health route
  `/health` in the custom-container config, and that `--host 0.0.0.0` is in the args.
- **GitHub Actions build fails with "no space left on device"** → CUDA images are large;
  add a free-disk-space step at the top of the build job (e.g.
  `jlumbroso/free-disk-space`) or prune before building.

---

# Activity 2 — ZeroGPU Space demo (Mellea + HF Transformers backend)

The Space app ([`app.py`](app.py)) drives
[`ibm-granite/granite-switch-4.1-8b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
**in-process on ZeroGPU** through [Mellea](https://docs.mellea.ai)'s HuggingFace
backend — no separate server. It turns the checkpoint's adapters into an interactive
instruct–validate–repair (IVR) workflow.

Each interaction:

1. Pick adapters (all three by default), optionally state **requirements**, submit a prompt.
2. With **requirement-check** selected, Mellea runs an **instruct → validate → repair**
   loop: every draft is judged by the embedded requirement-check aLoRA and regenerated
   until it passes or the attempt budget runs out. Draft tokens **stream** into the chat.
3. **uncertainty** and **guardian-core** then judge the final answer, each in its own bubble.
4. Every generation reports a ⚡ **KV-cache hit rate**.

## Architecture

- **[`switch_backend.py`](switch_backend.py)** — `SwitchBackend(LocalHFBackend)`, the
  glue that teaches Mellea 0.6.0's HF backend to activate Granite Switch's *embedded*
  adapters via the chat template's control tokens. Mellea 0.6.0 only supports embedded
  adapters on its vLLM/OpenAI backend out of the box; this makes them work in-process.
  It also implements **per-interaction KV prefix reuse** with a measured hit-rate metric
  (adapter/retry turns reuse the cached conversation prefix; judge prompts are re-expressed
  on the draft's actual decode tokens so reuse extends into the generation, not just the
  prompt).
- **[`app.py`](app.py)** — the Gradio UI and the IVR orchestration. Requirements are
  validated on the **live conversation context** (working around a Mellea 0.6.0 issue
  where the stock sampling strategies validate against an empty context, so the aLoRA
  judges blind). Drafts identical to an already-failed attempt are not re-judged.

## Run / test the Space app (no GPU needed)

`tests/` verifies `app.py` before deploying it as a Space. Everything except the model
weights is real — the actual `mellea`, `transformers`, `torch`, `gradio`, and
`granite_switch` packages, and the **real tokenizer + chat template render every
prompt**; only `model.generate` is faked. Adapter names are validated against
`tests/adapter_catalog.json`, a mirror of the
[upstream adapter catalog](https://github.com/generative-computing/granite-switch/blob/main/docs/adapter_catalog.html).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests/ -v
```

Covered: embedded-adapter activation tokens reach the prompt (and never on plain
generation), the IVR loop (retry / converge / exhaust, greedy judges), token streaming,
KV-cache arithmetic and attribution, and UI rendering. Not covered: real weights and
generation quality — after deploying, send one message per adapter as a manual smoke test.

## Dependency notes (Activity 2)

- **Mellea is installed *without* its `[hf]` extra** (see [requirements.txt](requirements.txt)):
  that extra pins `transformers<5`, which conflicts with granite-switch's `>=5.5.1`
  requirement. The pieces of the extra the HF backend actually needs at inference time
  (`llguidance`, `xgrammar`) are listed explicitly instead.
- **Python 3.12** — ZeroGPU's builder maxes out at 3.12, and Mellea requires ≥3.11.

---

# Spaces

Three smaller demos live under [`spaces/`](spaces/), each a self-contained Gradio Space
folder with its own README, `app.py` and `requirements.txt`. The two noul demos call the
Activity 1 endpoint, so they run on free CPU hardware; the GPU work happens on the
endpoint.

## Thinking Fast and Slow with Granite

[`spaces/jev-noul-mimic/`](spaces/jev-noul-mimic/) · [README](spaces/jev-noul-mimic/README.md)

*System One* models like TypeSafe AI's [Jev](https://docs.typesafe.ai) answer a yes/no
question with a **noul**: one number in [0, 1], the probability that the answer is yes.
This Space makes the same kind of call with Granite Switch and shows Jev alongside as a
reference.

- **Thinking Fast.** The prompt ends with "Reply with exactly one word, 'Yes' or 'No'.",
  Granite generates one token, and the noul is P(yes) / (P(yes) + P(no)) from that
  token's log-probabilities. On 82 hand-labeled questions this got 77 right (AUC 0.99),
  against 74 for a ratio of two UQ-adapter scores and 52 for a single UQ score.
- **Thinking Slow.** Granite writes an answer, then the UQ adapter (an aLoRA, so it
  reuses the KV cache for the question and the answer) scores how likely that answer is
  to be correct. Built with Mellea's `core.check_certainty`.
- **Batching.** The shared state goes first in every prompt, one request primes vLLM's
  prefix cache, and the rest fan out in parallel on the cached prefix.

## Noul Race

[`spaces/noul-race/`](spaces/noul-race/) · [README](spaces/noul-race/README.md)

A fork of the demo above, turned into a three-way race between Granite Switch, Jev and
OpenAI's GPT Luna (`gpt-6-luna`, through the Decisions API).

- **Context.** Paste any text or JSON, or click **Random Wikipedia article**.
- **Questions.** One per line with the answer after the question mark (`…? Yes`,
  `…? No`). Write them yourself or click **Generate questions**, which streams them in
  from `openai/gpt-oss-120b` on OpenRouter.
- **Race tab.** 10 generated yes/no questions go to all three models at once. Granite's
  noul here is c(yes): the UQ adapter's certainty in a prefilled "Yes".
- **Obstacle Course tab.** 10 yes/no and 5 freeform (`…? Freeform`) questions in random
  order, run one at a time. Yes/no questions go to each model's System One call;
  freeform questions go to a chat completion capped at 20 tokens. Jev returns decisions
  only, so it isn't run and shows greyed out.
- **Metrics.** Each model has a latency tile that runs as a stopwatch, with an accuracy
  tile beneath it. The first model to finish gets a green latency tile.

Deploy it with [`scripts/create_noul_race_space.py`](scripts/create_noul_race_space.py),
which creates the Space (private unless `--public` is passed), sets its secrets from
`.env`, and uploads the folder:

```bash
pip install "huggingface_hub>=0.24" python-dotenv
python scripts/create_noul_race_space.py
```

It needs `HF_TOKEN` and `HF_ENDPOINT_URL` in `.env`, plus `OPENROUTER_API_KEY` (Jev and
question generation) and `OPENAI_API_KEY` (GPT Luna) for the full race.

## ZeroGPU smoke test

[`spaces/zerogpu-smoke/`](spaces/zerogpu-smoke/) · [README](spaces/zerogpu-smoke/README.md)

One button: it grabs a ZeroGPU slot, reports the device name, VRAM and CUDA version, and
times a matmul. Use it to confirm ZeroGPU allocation works on your account before
deploying Activity 2. Create it with
[`scripts/create_smoke_space.py`](scripts/create_smoke_space.py).
