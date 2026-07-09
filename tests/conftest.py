"""Test doubles for running app.py without a GPU or the 60 GB checkpoint.

Everything except the model weights is real: the actual `gradio`,
`transformers` (including TextIteratorStreamer), `torch`, and
`granite_switch` packages are imported. Only three things are faked,
patched in *before* app.py's module-level load:

* ``spaces``          — pass-through GPU decorator (no HF Spaces runtime here)
* ``AutoTokenizer``   — FakeTokenizer that mimics the Granite Switch chat
                        template contract: it validates ``adapter_name``
                        against the authoritative catalog
                        (tests/adapter_catalog.json) and inserts the
                        ``<|name|>`` control token, exactly as the real
                        template does. Unknown adapter names raise, as they
                        would in production.
* ``AutoModelForCausalLM`` — FakeModel whose ``generate()`` feeds token ids
                        through the *real* TextIteratorStreamer protocol
                        (prompt put, token puts, end), so app.py's streaming
                        path is exercised for real.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CATALOG = json.loads((Path(__file__).parent / "adapter_catalog.json").read_text())
VALID_ADAPTERS = set(CATALOG["adapters"])


class FakeTokenizer:
    eos_token = "<|end_of_text|>"
    eos_token_id = 10**9  # never produced by FakeModel's arange sequences

    def __init__(self):
        self.last_template_kwargs = None
        self.last_messages = None
        self.last_prompt = None

    def apply_chat_template(self, messages, **kwargs):
        adapter = kwargs.get("adapter_name")
        if adapter is not None and adapter not in VALID_ADAPTERS:
            # The real chat template only knows the embedded adapters'
            # control tokens; an unknown name fails there too.
            raise ValueError(f"unknown adapter_name: {adapter!r}")
        self.last_template_kwargs = dict(kwargs)
        self.last_messages = [dict(m) for m in messages]

        parts = []
        for m in messages:
            parts.append(f"<|start_of_role|>{m['role']}<|end_of_role|>{m['content']}<|end_of_text|>")
        docs = kwargs.get("documents")
        if docs:
            rendered = " ".join(d["text"] for d in docs)
            parts.insert(0, f"<|start_of_role|>documents<|end_of_role|>{rendered}<|end_of_text|>")
        if kwargs.get("add_generation_prompt"):
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
        if adapter:
            parts.append(f"<|{adapter}|>")
        prompt = "".join(parts)
        self.last_prompt = prompt
        return prompt

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Stable fake encoding: one id per whitespace-separated chunk.
        n = max(1, len(text.split()))
        return SimpleNamespace(input_ids=torch.arange(n).unsqueeze(0))

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # Trailing space makes TextIteratorStreamer flush on every put.
        return "".join(f"tok{i} " for i in token_ids)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.last_generate_kwargs = None

    def eval(self):
        return self

    def generate(self, input_ids=None, streamer=None, max_new_tokens=None,
                 past_key_values=None, **kwargs):
        self.last_generate_kwargs = dict(
            input_ids=input_ids, max_new_tokens=max_new_tokens,
            past_key_values=past_key_values, **kwargs
        )
        n_new = min(int(max_new_tokens or 8), 8)
        if past_key_values is not None:
            # Mimic generate(): KV exists for every processed position, i.e.
            # everything except the last generated token (never fed forward).
            processed = input_ids.shape[1] + n_new - 1
            add = processed - past_key_values.get_seq_length()
            if add > 0:
                kv = torch.zeros(1, 1, add, 4)
                past_key_values.update(kv, kv, 0)
        if streamer is not None:
            streamer.put(input_ids[0])  # prompt — skipped via skip_prompt=True
            base = input_ids.shape[1]
            for i in range(n_new):
                streamer.put(torch.tensor([base + i]))
            streamer.end()
        return torch.cat(
            [input_ids, torch.arange(input_ids.shape[1], input_ids.shape[1] + n_new).unsqueeze(0)],
            dim=1,
        )


@pytest.fixture(scope="session")
def app_module():
    """Import app.py once with the fakes installed."""
    spaces_stub = types.ModuleType("spaces")

    def gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]  # bare @spaces.GPU
        return lambda fn: fn  # @spaces.GPU(duration=...)

    spaces_stub.GPU = gpu
    sys.modules["spaces"] = spaces_stub

    import transformers

    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    orig_tok = transformers.AutoTokenizer.from_pretrained
    orig_model = transformers.AutoModelForCausalLM.from_pretrained
    transformers.AutoTokenizer.from_pretrained = classmethod(
        lambda cls, *a, **k: fake_tokenizer
    )
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(
        lambda cls, *a, **k: fake_model
    )
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import app  # noqa: E402  (module-level load happens here, against fakes)
    finally:
        transformers.AutoTokenizer.from_pretrained = orig_tok
        transformers.AutoModelForCausalLM.from_pretrained = orig_model

    app._fake_tokenizer = fake_tokenizer
    app._fake_model = fake_model
    return app
