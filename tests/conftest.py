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
        self.judge_answers = {}  # adapter name -> deque of scripted raw outputs
        self.base_answer = "The moon is made of anorthosite rock."

    # ------------------------------------------------------------- scripting
    def script_judge(self, adapter, answers):
        self.judge_answers[adapter] = deque(answers)

    def reset(self):
        self.calls = []
        self.judge_answers = {}
        self.base_answer = "The moon is made of anorthosite rock."

    def calls_for(self, adapter):
        return [c for c in self.calls if c[0] == adapter]

    # ------------------------------------------------- transformers protocol
    def eval(self):
        return self

    def active_adapters(self):
        raise ValueError("No adapter loaded. Please load an adapter first.")

    def set_adapter(self, *args, **kwargs):
        raise ValueError("No adapter loaded. Please load an adapter first.")

    def generate(self, inputs, **kwargs):
        input_ids = inputs if torch.is_tensor(inputs) else inputs["input_ids"]
        ids = input_ids[0].tolist()
        adapter = next((_ID_TO_ADAPTER[t] for t in ids if t in _ID_TO_ADAPTER), None)
        self.calls.append((adapter, ids, dict(kwargs)))

        if adapter is not None:
            script = self.judge_answers.get(adapter)
            text = script.popleft() if script else DEFAULT_JUDGE_ANSWERS[adapter]
        else:
            text = self.base_answer
        new = self._tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        new = torch.cat([new, torch.tensor([[self._tokenizer.eos_token_id]])], dim=1)
        sequences = torch.cat([input_ids, new], dim=1)

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
