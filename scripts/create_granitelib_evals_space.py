"""Create (or update) the Granitelib Adapters vs Decision Models Space from spaces/granitelib-evals/.

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_granitelib_evals_space.py

Reads from .env (or the environment):

- HF_TOKEN: needs "write" scope.

It's a static Space (a single index.html), so it needs no secrets and no
hardware. The Space is created private; pass --public to make it public.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "granitelib-evals"
SPACE_NAME = "granitelib-evals"


def main():
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not found — put it in .env or export it.")

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Authenticated as {user}; creating Space {repo_id} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private="--public" not in sys.argv,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Granitelib Adapters vs Decision Models page",
        delete_patterns=["*.css", "*.js"],  # the static template's style.css, and any stale assets
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
