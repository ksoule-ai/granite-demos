# granite-switch requirement-check demo

Host [`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)
on a Hugging Face Inference Endpoint (vLLM backend) and call its
**requirement-check** adapter from Python.

Granite Switch is one checkpoint bundling ~12 embedded LoRA/aLoRA adapters
(RAG, Core, Guardian). You select one per request by name. The
`requirement-check` adapter judges whether an assistant response satisfies a
set of user-specified constraints and returns `{"score": "yes"|"no"}`.

## 1. Deploy the endpoint

The model is a **preview architecture**, so it needs recent runtimes:
vLLM ≥0.19.1 (<0.21.0) or Transformers ≥5.5.1. Use the **vLLM** container.

1. Log in to Hugging Face and open the [model page](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview); accept any gating terms.
2. Go to [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co) → **New** → **Model repository** → `ibm-granite/granite-switch-4.1-3b-preview`.
3. **Hardware:** a single **L4 (24 GB)** or **A10G** is plenty for a 3–4B BF16 model with long-context KV cache.
4. **Container:** select **vLLM** (not the default TGI). Pin an image tag in the **0.19.1–0.20.x** range so the Granite Switch architecture is recognized. If the picker only offers an older vLLM, use a **Custom** container with `vllm/vllm-openai:v0.20.x`.
5. Deploy and wait for **Running**. Your endpoint URL looks like
   `https://xxxx.us-east-1.aws.endpoints.huggingface.cloud`.

### Sanity checks after it's up
- Build log recognizes the architecture (no "unknown model type" — if so, bump the vLLM image tag).
- `GET {url}/v1/models` returns the model id; use that as `MODEL_ID`.

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

## How the adapter is invoked

vLLM applies the model's chat template server-side, so the adapter is
selected by passing `adapter_name` through `chat_template_kwargs`:

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

The final user turn carries the constraints (in a `<requirements>` tag) plus
a fixed evaluation instruction. The adapter replies with
`{"score": "yes"}` or `{"score": "no"}`.

## Notes
- IBM's preferred high-level client is [Mellea](https://mellea.ai), which
  wraps adapter selection and constrained decoding. This demo uses the raw
  OpenAI-compatible API so there's no extra dependency and the mechanics are
  visible.
- Other embedded adapters (`guardian-core`, `uncertainty`,
  `context-attribution`, `factuality-detection`, …) work the same way — swap
  the `adapter_name` and follow each adapter's message protocol.
