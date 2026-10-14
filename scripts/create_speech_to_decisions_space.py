"""Create (or update) the Speech to Decisions Space from spaces/speech-to-decisions/.

Run from the repo root on a machine with your .env present:

    pip install "huggingface_hub>=0.24" python-dotenv
    python scripts/create_speech_to_decisions_space.py

Reads from .env (or the environment):

- HF_TOKEN: needs "write" scope. It's used to create the Space and isn't
  stored on it.
- OPENROUTER_API_KEY: stored as a Space secret; used for Parakeet, Jev and
  gpt-oss-120b.
- JEV_MODEL, DESCRIBE_MODEL, STT_MODEL, STT_LANGUAGE, VOICE_RMS,
  MIN_NEW_AUDIO_S, QUIET_AFTER_S (optional): passed through when set.

The Space is a Docker Space (a FastAPI server and a hand-built page) on free
CPU hardware. It is created private; pass --public to make it public.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).parent.parent / "spaces" / "speech-to-decisions"
SPACE_NAME = "speech-to-decisions"
REQUIRED_SECRETS = ["OPENROUTER_API_KEY"]
OPTIONAL_SECRETS = ["JEV_MODEL", "DESCRIBE_MODEL", "STT_MODEL", "STT_LANGUAGE", "VOICE_RMS", "MIN_NEW_AUDIO_S", "QUIET_AFTER_S"]


def main():
    load_dotenv()
    missing = [name for name in ["HF_TOKEN"] + REQUIRED_SECRETS if not os.environ.get(name)]
    if missing:
        sys.exit(f"{', '.join(missing)} not found — put them in .env or export them.")

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Authenticated as {user}; creating Space {repo_id} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
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
        commit_message="Speech to Decisions app",
        ignore_patterns=["__pycache__/*", "**/__pycache__/*"],
        delete_patterns=["*.py", "static/*"],  # drop files that no longer exist locally
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
