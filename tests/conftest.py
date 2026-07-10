"""Test doubles for running app.py without a GPU or the 16 GB checkpoint.

Everything except the model weights is real: the actual `mellea`,
`transformers`, `torch`, `granite_switch`, and `gradio` packages are
imported, and the *real* Granite Switch tokenizer + chat template render
every prompt (so adapter activation tokens are exercised for real, all the
way through mellea's intrinsic pipeline). Only two things are faked, patched
in *before* app.py's module-level load:

* ``spaces``               — pass-through GPU decorator (no HF Spaces runtime)
* ``AutoModelForCausalLM`` — FakeSwitchModel whose ``generate()`` returns
                             scripted text: judge answers when an adapter
                             control token (ids 100352-100363) is present in
                             the prompt, base answers otherwise. It records
                             every call so tests can assert on the exact
                             token ids and generate kwargs the model saw.
"""

import sys
import types
from collections import deque

import pytest
import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

# Control token ids from the model's config.json / adapter_index.json.
CONTROL_TOKENS = {
    "citations": 100352,
    "query_rewrite": 100353,
    "query_clarification": 100354,
    "hallucination_detection": 100355,
    "answerability": 100356,
    "factuality-detection": 100357,
    "policy-guardrails": 100358,
    "factuality-correction": 100359,
    "guardian-core": 100360,
    "uncertainty": 100361,
    "requirement-check": 100362,
    "context-attribution": 100363,
}
_ID_TO_ADAPTER = {v: k for k, v in CONTROL_TOKENS.items()}

# Raw model outputs conforming to each adapter's io.yaml response_format.
DEFAULT_JUDGE_ANSWERS = {
    "requirement-check": '{"score": "yes"}',
    "uncertainty": '{"score": "8"}',
    "guardian-core": '{"score": "no"}',
}


class FakeSwitchModel:
    device = torch.device("cpu")
    vocab_size = 100364

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.calls = []  # (adapter_name | None, input id list, generate kwargs)
        self.cache_calls = []  # (adapter_name | None, cache_len_in | None, ids)
        self.judge_answers = {}  # adapter name -> deque of scripted raw outputs
        self.draft_answers = deque()  # scripted plain-generation outputs
        self.base_answer = "The moon is made of anorthosite rock."

    # ------------------------------------------------------------- scripting
    def script_judge(self, adapter, answers):
        self.judge_answers[adapter] = deque(answers)

    def script_drafts(self, answers):
        self.draft_answers = deque(answers)

    def reset(self):
        self.calls = []
        self.cache_calls = []
        self.judge_answers = {}
        self.draft_answers = deque()
        self.base_answer = "The moon is made of anorthosite rock."

    def calls_for(self, adapter):
        return [c for c in self.calls if c[0] == adapter]

    # ------------------------------------------------- transformers protocol
    def eval(self):
        return self

    # The Space installs mellea without the [hf] extra, so peft is absent and
    # transformers' PEFT integration raises this (not "No adapter loaded",
    # which is the only ValueError mellea's lock helper swallows). The glue
    # must never reach these on switch checkpoints — regression for the
    # "PEFT is not installed" crash seen on the Space.
    def active_adapters(self):
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    def set_adapter(self, *args, **kwargs):
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    def generate(self, inputs, **kwargs):
        input_ids = inputs if torch.is_tensor(inputs) else inputs["input_ids"]
        ids = input_ids[0].tolist()
        adapter = next((_ID_TO_ADAPTER[t] for t in ids if t in _ID_TO_ADAPTER), None)
        self.calls.append((adapter, ids, dict(kwargs)))

        # KV-cache contract of transformers 5.9 generate: with a cache, the
        # prompt must exceed the cached length, and prefix trimming only
        # happens when an attention_mask of the SAME length as input_ids is
        # present. Encode those preconditions as hard asserts so the backend
        # injection is verified on every call.
        past = kwargs.get("past_key_values")
        if past is not None:
            assert past.get_seq_length() < input_ids.shape[1], (
                "cache must be shorter than the prompt"
            )
            mask = kwargs.get("attention_mask")
            assert mask is not None and mask.shape == input_ids.shape, (
                "past_key_values requires a full-length attention_mask"
            )
        self.cache_calls.append(
            (adapter, past.get_seq_length() if past is not None else None, ids)
        )

        if adapter is not None:
            script = self.judge_answers.get(adapter)
            text = script.popleft() if script else DEFAULT_JUDGE_ANSWERS[adapter]
        elif self.draft_answers:
            text = self.draft_answers.popleft()
        else:
            text = self.base_answer
        new = self._tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        new = torch.cat([new, torch.tensor([[self._tokenizer.eos_token_id]])], dim=1)
        sequences = torch.cat([input_ids, new], dim=1)

        if past is not None:
            # Mimic generate's cache mutation: KV exists for every processed
            # position, i.e. everything except the last generated token
            # (never fed forward). Uniform (1, 1, seq, 1) shapes keep the
            # backend's later crop() calls working.
            grow = (sequences.shape[1] - 1) - past.get_seq_length()
            if grow > 0:
                kv = torch.zeros(1, 1, grow, 1)
                past.update(kv, kv, 0)

        streamer = kwargs.get("streamer")
        if streamer is not None:  # mellea streaming path (AsyncTextIteratorStreamer)
            streamer.put(input_ids[0])  # prompt — skipped via skip_prompt=True
            for token in new[0]:
                streamer.put(token.unsqueeze(0))
            streamer.end()

        scores = None
        if kwargs.get("output_scores"):
            scores = tuple(
                torch.full((1, self.vocab_size), -20.0).scatter(1, new[:, i:i + 1], 20.0)
                for i in range(new.shape[1])
            )
        if kwargs.get("return_dict_in_generate"):
            return GenerateDecoderOnlyOutput(sequences=sequences, scores=scores)
        return sequences


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
    from pathlib import Path

    fake_holder = {}

    def fake_model_from_pretrained(cls, *args, **kwargs):
        # The tokenizer is loaded (for real) before the model in app.py.
        return fake_holder["model"]

    orig_model = transformers.AutoModelForCausalLM.from_pretrained
    orig_tok = transformers.AutoTokenizer.from_pretrained

    def tok_from_pretrained(cls, *args, **kwargs):
        tokenizer = orig_tok(*args, **kwargs)
        fake_holder["model"] = FakeSwitchModel(tokenizer)
        return tokenizer

    transformers.AutoTokenizer.from_pretrained = classmethod(tok_from_pretrained)
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(
        fake_model_from_pretrained
    )
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import app  # noqa: E402  (module-level load happens here, against fakes)
    finally:
        transformers.AutoTokenizer.from_pretrained = orig_tok
        transformers.AutoModelForCausalLM.from_pretrained = orig_model

    app._fake_model = fake_holder["model"]
    return app


@pytest.fixture()
def fake_model(app_module):
    app_module._fake_model.reset()
    return app_module._fake_model
