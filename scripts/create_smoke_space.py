"""Create (or update) the ZeroGPU smoke-test Space from spaces/zerogpu-smoke/.

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_smoke_space.py

Reads HF_TOKEN from .env (or the environment). The token needs "write"
scope. ZeroGPU hardware requires a PRO account; if the hardware request is
rejected, the Space is still created on CPU and you can flip it to ZeroGPU
in the Space's Settings -> Hardware.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "zerogpu-smoke"
SPACE_NAME = "zerogpu-smoke-test"
ZEROGPU_HARDWARE = "zero-a10g"  # HF API name for ZeroGPU


def main():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found — put it in .env or export it.")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Authenticated as {user}; creating Space {repo_id} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    try:
        api.request_space_hardware(repo_id=repo_id, hardware=ZEROGPU_HARDWARE)
        print(f"Hardware set to {ZEROGPU_HARDWARE} (ZeroGPU).")
    except HfHubHTTPError as e:
        print(
            f"WARNING: could not set ZeroGPU hardware ({e}).\n"
            "Space stays on CPU — enable ZeroGPU manually in "
            f"https://huggingface.co/spaces/{repo_id}/settings (needs PRO)."
        )

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="ZeroGPU smoke test app",
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
