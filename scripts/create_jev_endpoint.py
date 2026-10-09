"""Create the JEV-9B Inference Endpoint (autotrust/JEV-9B on stock vLLM).

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_jev_endpoint.py

Reads HF_TOKEN from .env (or the environment); it needs permission to manage
Inference Endpoints. This starts a paid GPU instance. It scales to zero after
15 idle minutes, like the Granite Switch endpoint.

autotrust/JEV-9B is an open reproduction of TypeSafe's Jev: the unmodified
Qwen3.5-9B backbone (System 2) plus a LoRA adapter and decision head for
typed decisions (System 1). One vLLM engine serves both: ordinary requests
use the model name autotrust/JEV-9B, and decisions go to the LoRA module
"jev-decision" (see the model card for the decision prompt).

No custom image is needed: vLLM 0.31.0 has Qwen3.5, LoRA on lm_head,
--logprobs-mode and --mamba-cache-mode. Use the default (CUDA 13) tag, which
runs on Hugging Face's L40S hosts. The v0.31.0-cu129 tag crashes at start-up:
its torchcodec is built for CUDA 13 and fails to load (libnvrtc.so.13).

Pass --gpu to pick the instance type (default nvidia-l40s, 48 GB). The weights
are 18 GB in bf16, so a 24 GB card (nvidia-l4, nvidia-a10g) is a tight fit.
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO = "autotrust/JEV-9B"
ENDPOINT_NAME = "jev-9b"
IMAGE = "vllm/vllm-openai:v0.31.0"
# Sized for this demo's contexts (up to ~5k tokens). The model card uses 4096 for decisions alone.
MAX_MODEL_LEN = "8192"
VLLM_ARGS = [
    "--model", "/repository",  # Inference Endpoints mounts the repo here
    "--served-model-name", REPO,
    "--host", "0.0.0.0",
    "--enable-lora", "--max-lora-rank", "32",
    "--lora-modules", "jev-decision=/repository/adapter_vllm",
    # Required: makes the returned log-probabilities respect allowed_token_ids.
    "--logprobs-mode", "processed_logprobs",
    "--max-model-len", MAX_MODEL_LEN,
    # Many questions about one context share a prefix.
    "--enable-prefix-caching", "--mamba-cache-mode", "align",
    "--enable-prompt-tokens-details",  # report cached_tokens in usage
    "--max-logprobs", "256",
]


def main():
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not found — put it in .env or export it.")
    gpu = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "nvidia-l40s"

    api = HfApi(token=os.environ["HF_TOKEN"])
    revision = api.model_info(REPO).sha  # pin the weights the endpoint serves
    print(f"Creating endpoint {ENDPOINT_NAME}: {REPO}@{revision[:8]} on {gpu} ...")
    endpoint = api.create_inference_endpoint(
        ENDPOINT_NAME,
        repository=REPO,
        revision=revision,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        instance_type=gpu,
        instance_size="x1",
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=15,
        type="private",  # same as the Granite Switch endpoint: callers need the owner's token
        custom_image={"health_route": "/health", "port": 8000, "url": IMAGE},
        container_args=VLLM_ARGS,
    )
    print(f"Status: {endpoint.status}. Start-up takes several minutes (download, then CUDA-graph capture).")
    print("Watch it at https://ui.endpoints.huggingface.co — the URL appears once it is running.")


if __name__ == "__main__":
    main()
