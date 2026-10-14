"""Create (or update) the Decisions Switching Costs Space from spaces/split-decisions/.

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_split_decisions_space.py

Reads from .env (or the environment):

- HF_TOKEN: needs "write" scope. Also stored as the Space's HF_TOKEN secret,
  so it must be allowed to call the Granite Switch endpoint.
- HF_ENDPOINT_URL: the Granite Switch endpoint URL, ending in /v1.
- JEV9B_ENDPOINT_URL (optional): the JEV-9B endpoint URL (see
  scripts/create_jev_endpoint.py). Without it JEV-9B shows as unavailable.
- GEMMA_DECIDER_ENDPOINT_URL (optional): the Gemma decider endpoint URL (see
  scripts/create_gemma_decider_endpoint.py). Without it Gemma Decider shows
  as unavailable.
- OPENROUTER_API_KEY: used for GPT Luna, for Jev and Qwen, and for
  generating the questions. Without it the page has no questions to run.
- MODEL_ID, GRANITE_HARDWARE, JEV9B_HARDWARE, GEMMA_DECIDER_HARDWARE, JEV_MODEL, JEV_PARTNER_MODEL,
  LUNA_MODEL, LUNA_DECISIONS_MODEL, QUESTION_MODEL, MAX_CONTEXT_CHARS
  (optional): passed through when set.

The Space runs on free CPU hardware and is created private; pass --public to
make it public.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "split-decisions"
SPACE_NAME = "split-decisions"
REQUIRED_SECRETS = ["HF_ENDPOINT_URL", "HF_TOKEN"]
OPTIONAL_SECRETS = [
    "JEV9B_ENDPOINT_URL", "GEMMA_DECIDER_ENDPOINT_URL", "OPENROUTER_API_KEY",
    "MODEL_ID", "GRANITE_HARDWARE", "JEV9B_HARDWARE", "GEMMA_DECIDER_HARDWARE", "JEV_MODEL", "JEV_PARTNER_MODEL",
    "LUNA_MODEL", "LUNA_DECISIONS_MODEL", "QUESTION_MODEL", "MAX_CONTEXT_CHARS",
]


def main():
    load_dotenv()
    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name)]
    if missing:
        sys.exit(f"{', '.join(missing)} not found — put them in .env or export them.")

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Authenticated as {user}; creating Space {repo_id} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private="--public" not in sys.argv,
        exist_ok=True,
    )

    # Secrets go in before the upload, so the first build starts with them set.
    for name in REQUIRED_SECRETS + OPTIONAL_SECRETS:
        if os.environ.get(name):
            api.add_space_secret(repo_id=repo_id, key=name, value=os.environ[name])
            print(f"Set secret {name}.")
        else:
            print(f"Skipped {name} (not set).")

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Decisions Switching Costs app",
        ignore_patterns=["__pycache__/*"],
        delete_patterns=["*.py"],  # drop modules that no longer exist locally
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
