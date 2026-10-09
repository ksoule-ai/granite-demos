"""Create (or update) the Noul Race Space from spaces/noul-race/.

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_noul_race_space.py

Reads from .env (or the environment):

- HF_TOKEN: needs "write" scope. Also stored as the Space's HF_TOKEN secret,
  so it must be allowed to call the Granite Switch endpoint.
- HF_ENDPOINT_URL: the Granite Switch endpoint URL, ending in /v1.
- OPENROUTER_API_KEY: used for Jev and for generating the questions.
- OPENAI_API_KEY (optional; OPEN_AI_KEY is also accepted): for GPT Luna.
  Without it the Luna column shows as unavailable.
- MODEL_ID, JEV_MODEL, LUNA_MODEL, LUNA_CHAT_MODEL, QUESTION_MODEL (optional): passed through when set.

The Space runs on free CPU hardware and is created private; pass --public to
make it public.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "noul-race"
SPACE_NAME = "noul-race"
REQUIRED_SECRETS = ["HF_ENDPOINT_URL", "HF_TOKEN"]
OPTIONAL_SECRETS = ["OPENROUTER_API_KEY", "MODEL_ID", "JEV_MODEL", "OPENAI_API_KEY", "LUNA_MODEL", "LUNA_CHAT_MODEL", "QUESTION_MODEL"]


def main():
    load_dotenv()
    if os.environ.get("OPEN_AI_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", os.environ["OPEN_AI_KEY"])
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
        commit_message="Noul Race app",
        ignore_patterns=["__pycache__/*"],
        delete_patterns=["*.py"],  # drop modules that no longer exist locally
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
