# Granite Switch Inference Endpoint — Reference

Portable reference for the deployed endpoint. Copy this file into any project
that needs to call it.

## Endpoint summary

| | |
|---|---|
| **Model** | [`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) |
| **What it is** | Granite 4.1 3B with 12 embedded LoRA/aLoRA adapters (RAG, Core, Guardian), selected per-request by name |
| **Host** | Hugging Face Inference Endpoints (AWS `us-east-1`) |
| **Backend** | vLLM, OpenAI-compatible API |
| **Base URL** | Not published — retrieve it from the endpoint dashboard (see below) |
| **Max context** | 131,072 tokens |
| **Auth** | `Authorization: Bearer <HF token>` — any request without it is rejected |
| **License** | Apache-2.0 |

## Configure a project's `.env`

Add these three variables to the consuming project's `.env` (create it if
absent, and make sure `.env` is in that project's `.gitignore` — never commit
it):

```bash
HF_ENDPOINT_URL=https://<endpoint-id>.us-east-1.aws.endpoints.huggingface.cloud/v1   # see below
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # see below
MODEL_ID=ibm-granite/granite-switch-4.1-3b-preview
```

Both values are deliberately not published in this repo. To obtain them:

**Endpoint URL (`HF_ENDPOINT_URL`):**
- Reuse the value from `granite-demos/.env` on this machine, **or**
- Open [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co)
  (owner account), select the Granite Switch endpoint, and copy its
  **Endpoint URL** — then **append `/v1`**.

**Token (`HF_TOKEN`):**
- Reuse the value from `granite-demos/.env` on this machine, **or**
- Mint a new one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  → *Create new token* → **Fine-grained** → enable *Make calls to Inference
  Providers / Inference Endpoints*. One token per project makes revocation
  painless.

Note the base URL ends in `/v1` — the OpenAI SDK appends `/chat/completions`
to it, so without `/v1` you'll get 404s.

## Smoke test

Verify the endpoint is up and the token works:

```bash
curl -sS -H "Authorization: Bearer $HF_TOKEN" "$HF_ENDPOINT_URL/models"
# → {"object":"list","data":[{"id":"ibm-granite/granite-switch-4.1-3b-preview",...}]}
```

## Calling it from tests

Any OpenAI-compatible client works. Python:

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["HF_ENDPOINT_URL"], api_key=os.environ["HF_TOKEN"])

# Plain chat (base model, no adapter)
r = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=50,
)
print(r.choices[0].message.content)
```

### Activating an embedded adapter

Adapters are selected by passing `adapter_name` through
`chat_template_kwargs` (vLLM applies the chat template server-side):

```python
r = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[...],          # message layout per the adapter's protocol
    max_tokens=15,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"adapter_name": "requirement-check"}},
)
```

Each adapter has its own message protocol and output schema — see the
[adapter reference table](https://huggingface.co/ibm-granite/granitelib-core-r1.0)
and the worked `requirement-check` example in
[`requirement_check.py`](requirement_check.py) (judges whether a response
satisfies constraints; returns `{"score": "yes"|"no"}`).

## Operational notes

- **Billing:** the endpoint bills per hour while running. If scale-to-zero is
  enabled, the first request after idle takes ~1–5 min to cold-start (expect
  a 503 with a "scaled to zero" message — retry with backoff in tests).
- **Management:** pause/resume/scale at
  [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co) (owner
  account: the one that deployed it).
- **If the URL changes** (endpoint recreated), only `HF_ENDPOINT_URL` in each
  project's `.env` needs updating — everything else stays the same.
