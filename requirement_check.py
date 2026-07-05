#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Requirement-check adapter demo against a Granite Switch vLLM endpoint.

Granite Switch (ibm-granite/granite-switch-4.1-3b-preview) is a single
checkpoint carrying many embedded LoRA/aLoRA adapters. You pick one per
request by name. When the model is served by vLLM's OpenAI-compatible
server, the chat template runs server-side, so the adapter is selected by
passing ``adapter_name`` through ``chat_template_kwargs``.

The ``requirement-check`` adapter judges whether an assistant response
satisfies a set of user-specified constraints, returning ``{"score":
"yes"}`` or ``{"score": "no"}``.

Usage:
    cp .env.example .env   # fill in HF_ENDPOINT_URL and HF_TOKEN
    pip install -r requirements.txt
    python requirement_check.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

ADAPTER_NAME = "requirement-check"

# Fixed instruction the adapter was trained to read after the constraints.
# (Taken verbatim from the granite-switch requirement-check protocol.)
EVALUATION_PROMPT = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)


def make_client() -> tuple[OpenAI, str]:
    """Build an OpenAI client pointed at the HF Inference Endpoint."""
    load_dotenv()
    base_url = os.environ.get("HF_ENDPOINT_URL")
    token = os.environ.get("HF_TOKEN")
    model = os.environ.get("MODEL_ID", "ibm-granite/granite-switch-4.1-3b-preview")
    if not base_url or not token:
        sys.exit("Set HF_ENDPOINT_URL and HF_TOKEN in .env (see .env.example).")
    return OpenAI(base_url=base_url, api_key=token), model


def check_requirements(
    client: OpenAI,
    model: str,
    user_task: str,
    response: str,
    requirements: str,
) -> bool:
    """Return True iff `response` satisfies `requirements` for `user_task`.

    Builds the message layout the requirement-check adapter expects:
    the original task, the assistant response under review, then a user
    turn naming the constraints followed by the evaluation instruction.
    The adapter is activated via chat_template_kwargs.adapter_name.
    """
    eval_turn = f"<requirements> {requirements}\n{EVALUATION_PROMPT}"
    messages = [
        {"role": "user", "content": user_task},
        {"role": "assistant", "content": response},
        {"role": "user", "content": eval_turn},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=15,
        temperature=0.0,  # greedy — this is a judge, not a generator
        extra_body={"chat_template_kwargs": {"adapter_name": ADAPTER_NAME}},
    )

    raw = completion.choices[0].message.content.strip()
    try:
        score = json.loads(raw)["score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ValueError(f"Unexpected adapter output: {raw!r}")
    return score.lower() == "yes"


def main() -> None:
    client, model = make_client()

    user_task = (
        "Write a short climate-change paragraph for a science newsletter. "
        "It must be in a formal, professional tone, include at least 3 "
        "specific examples, cite sources or indicate uncertainty, and be "
        "under 100 words."
    )
    requirements = (
        "Response must be in formal professional tone; must include at "
        "least 3 specific examples; must cite sources or indicate "
        "uncertainty; must be under 100 words."
    )

    # A response that plausibly satisfies the constraints...
    good = (
        "Climate change affects biodiversity in several ways. Rising "
        "temperatures force species northward - many butterflies have "
        "shifted their ranges. Ocean acidification damages coral reefs, "
        "threatening the Great Barrier Reef. Changing precipitation "
        "affects amphibian breeding, as seen in the golden toad's "
        "extinction. These impacts are accelerating according to IPCC "
        "reports."
    )
    # ...and one that clearly does not (informal, no examples, no sources).
    bad = "lol climate is basically just weather being moody, who knows"

    for label, resp in [("good", good), ("bad", bad)]:
        ok = check_requirements(client, model, user_task, resp, requirements)
        print(f"[{label:>4}] requirements satisfied: {ok}  ({len(resp.split())} words)")


if __name__ == "__main__":
    main()
