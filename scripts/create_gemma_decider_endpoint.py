"""Create the Gemma 4 decider Inference Endpoint (Mapika/decider-chat-gemma4-31b on stock vLLM).

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_gemma_decider_endpoint.py

Reads HF_TOKEN from .env (or the environment); it needs permission to manage
Inference Endpoints. This starts a paid GPU instance. It scales to zero after
15 idle minutes, like the other endpoints.

Mapika/decider-chat-gemma4-31b is the unmodified google/gemma-4-31B-it plus a
decider_config.json: there is no adapter and no decision head. Decisions are
read from the base model's own next-token distribution at the "Answer: (" slot
of a fixed prompt, so one engine serves both modes with one model name:
ordinary requests are chat completions, and decisions are one-token
completions limited to the option letters (scripts/gemma_decider_client.py).

No custom image is needed. decider's own vLLM server (decider.serve_vllm) is
not used: it runs vLLM in-process, serves /v1/systemone only (no chat), and
pins vLLM 0.29.0. The readout it does is redone client-side instead.

Pass --gpu and --size to pick the hardware (default nvidia-a100 x1, 80 GB).
The weights are 62.5 GB in bf16, so a single 48 GB card can't hold them;
--gpu nvidia-l40s --size x2 splits them over two cards (tensor parallel).
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO = "Mapika/decider-chat-gemma4-31b"
ENDPOINT_NAME = "gemma4-decider"
IMAGE = "vllm/vllm-openai:v0.31.0"
# Sized like the JEV-9B endpoint (demo contexts up to ~5k tokens); it also leaves
# room for the KV cache next to 62.5 GB of weights on one 80 GB card.
MAX_MODEL_LEN = "8192"
VLLM_ARGS = [
    "--model", "/repository",  # Inference Endpoints mounts the repo here
    "--served-model-name", REPO,
    "--host", "0.0.0.0",
    # Gemma 4 is multimodal; text only skips the vision tower's memory (as decider.serve_vllm does).
    "--language-model-only",
    # Required: makes the returned log-probabilities respect allowed_token_ids.
    "--logprobs-mode", "processed_logprobs",
    "--max-model-len", MAX_MODEL_LEN,
    "--gpu-memory-utilization", "0.92",
    # Many questions about one state share a prefix.
    "--enable-prefix-caching",
    "--enable-prompt-tokens-details",  # report cached_tokens in usage
    # A choice question can have up to 255 options, one logprob each.
    "--max-logprobs", "256",
]


def main():
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not found — put it in .env or export it.")
    gpu = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "nvidia-a100"
    size = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "x1"
    gpus = int(size.lstrip("x"))

    api = HfApi(token=os.environ["HF_TOKEN"])
    revision = api.model_info(REPO).sha  # pin the weights the endpoint serves
    print(f"Creating endpoint {ENDPOINT_NAME}: {REPO}@{revision[:8]} on {gpu} {size} ...")
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
        instance_size=size,
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=15,
        type="private",  # same as the other endpoints: callers need the owner's token
        custom_image={"health_route": "/health", "port": 8000, "url": IMAGE},
        container_args=VLLM_ARGS + ["--tensor-parallel-size", str(gpus)],
    )
    print(f"Status: {endpoint.status}. Start-up takes several minutes (download, then CUDA-graph capture).")
    print("Watch it at https://ui.endpoints.huggingface.co — the URL appears once it is running.")


if __name__ == "__main__":
    main()
