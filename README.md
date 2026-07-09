---
title: Granite Switch 4.1 8B Demo
emoji: 🪨
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
python_version: "3.13"
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

**The Space app ([`app.py`](app.py))** drives
[`ibm-granite/granite-switch-4.1-8b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
on ZeroGPU through [Mellea](https://docs.mellea.ai)'s HuggingFace backend:
the **requirement-check** aLoRA powers an instruct–validate–repair loop, and
**uncertainty** / **guardian-core** judge the final answer.
[`switch_backend.py`](switch_backend.py) is the glue that teaches mellea's
`LocalHFBackend` to activate the checkpoint's *embedded* adapters via chat
template control tokens (mellea 0.6.0 only supports embedded adapters on its
vLLM/OpenAI backend out of the box). Tests: `.venv/bin/python -m pytest tests/`.

The rest of this README covers the separate **Inference Endpoint** deployment
and the reference requirement-check CLI.

# granite-switch endpoint and requirement-check test

Host [`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) on a Hugging Face Inference Endpoint (vLLM backend) and call its **requirement-check** adapter from Python.

> **Already deployed?** See [ENDPOINT.md](ENDPOINT.md) for the live endpoint's
> URL, auth, and how to point other projects at it.

Granite Switch is one checkpoint bundling ~12 embedded LoRA/aLoRA adapters
(RAG, Core, Guardian). You select one per request by name. The `requirement-check`
adapter judges whether an assistant response satisfies a set of user-specified
constraints and returns `{"score": "yes"|"no"}`.

---

## Why this needs a custom container (read first)

Granite Switch is a **preview architecture**. Its model type, `granite_switch`,
is **not** in stock vLLM or Transformers — it ships only in the separate
[`granite-switch`](https://github.com/generative-computing/granite-switch) package.

If you deploy with HF's managed vLLM engine (or a plain `vllm/vllm-openai` image),
the endpoint builds but **fails to start** with:

```
Value error, The checkpoint you are trying to load has model type `granite_switch`
but Transformers does not recognize this architecture.
```

`--trust-remote-code` does **not** fix this, because the architecture isn't shipped
as remote code in the model repo — it's in the pip package. So a **custom container
that installs `granite-switch` is required**, not optional. Section 1 builds it.

---

## 1. Deploy the endpoint

### 1a. Build the custom container

The image is just stock vLLM plus the `granite-switch` package. See
[`docker/Dockerfile`](docker/Dockerfile):

```dockerfile
FROM vllm/vllm-openai:v0.19.1
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/generative-computing/granite-switch.git /opt/granite-switch \
 && pip install "/opt/granite-switch[vllm]"
```

**Version note:** the `[vllm]` extra pins **vLLM 0.19.1** (CUDA 12.x), which matches
HF's L4 hosts. The `[vllm20]` extra needs **CUDA 13+**, which those hosts don't have
— don't use it here.

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

### 1b. Create the endpoint

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

### 1c. Sanity checks after it's up

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

---

## 2. Run the demo

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

---

## 3. Test the ZeroGPU Space app (no GPU needed)

`tests/` verifies `app.py` before deploying it as a Space. Everything except
the 60 GB checkpoint is real (gradio, transformers streaming, torch); the
model is faked, and adapter names are validated against
`tests/adapter_catalog.json` — a mirror of the
[upstream adapter catalog](https://github.com/generative-computing/granite-switch/blob/main/docs/adapter_catalog.html).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests/ -v
```

Covered: adapter names exist upstream, UI builds under the pinned gradio,
documents/requirements fields toggle and reach the prompt, streaming works
end-to-end through `TextIteratorStreamer`. Not covered: real weights and
generation quality — after deploying, send one message per adapter as a
manual smoke test.

---

## How the adapter is invoked

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

---

## Notes

- IBM's preferred high-level client is [Mellea](https://mellea.ai), which wraps
  adapter selection and constrained decoding. This demo uses the raw
  OpenAI-compatible API so there's no extra dependency and the mechanics are visible.
- Other embedded adapters (`guardian-core`, `uncertainty`, `context-attribution`,
  `factuality-detection`, …) work the same way — swap the `adapter_name` and follow
  each adapter's message protocol.
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
  this demo is raw-API, don't assume cross-adapter reuse without measuring prefill
  tokens / TTFT.

---

## Troubleshooting

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
