"""Pre-deployment checks for the ZeroGPU Space (app.py).

Run with:  .venv/bin/python -m pytest tests/ -v

What this covers without a GPU:
  * every adapter name offered in the UI exists in the upstream catalog
  * the Gradio UI builds under the pinned gradio version
  * adapter-specific input fields (documents / requirements) toggle correctly
  * respond() streams through the real TextIteratorStreamer machinery
  * adapter control tokens, documents, and requirements all reach the prompt
What it cannot cover: real weights, real GPU generation quality. After the
Space is up, send one message per adapter as a manual smoke test.
"""

import json
from pathlib import Path

import gradio as gr
import pytest

CATALOG = json.loads((Path(__file__).parent / "adapter_catalog.json").read_text())
CATALOG_ADAPTERS = set(CATALOG["adapters"])
DOC_ADAPTERS_IN_CATALOG = {
    name for name, meta in CATALOG["adapters"].items() if meta.get("needs_documents")
}


# ---------------------------------------------------------------- adapter names

def test_every_ui_adapter_exists_upstream(app_module):
    """A wrong adapter_name fails only at request time on the Space — catch it here."""
    ui_adapters = [c for c in app_module.ADAPTER_CHOICES if not c.startswith("None")]
    unknown = [a for a in ui_adapters if a not in CATALOG_ADAPTERS]
    assert not unknown, f"UI offers adapters the model does not have: {unknown}"


def test_all_catalog_adapters_offered(app_module):
    """The demo's point is showing all 12 adapters — don't silently drop any."""
    ui_adapters = {c for c in app_module.ADAPTER_CHOICES if not c.startswith("None")}
    missing = CATALOG_ADAPTERS - ui_adapters
    assert not missing, f"Catalog adapters missing from the UI: {missing}"


def test_doc_adapters_match_catalog(app_module):
    assert app_module.DOC_ADAPTERS == DOC_ADAPTERS_IN_CATALOG


def test_every_adapter_has_description(app_module):
    for choice in app_module.ADAPTER_CHOICES:
        assert choice in app_module.ADAPTER_DESCRIPTIONS, f"no description for {choice}"


# ---------------------------------------------------------------- UI construction

def test_ui_builds(app_module):
    assert isinstance(app_module.demo, gr.Blocks)


def test_visibility_toggles(app_module):
    # documents box shows exactly for doc-needing adapters
    for adapter in CATALOG_ADAPTERS:
        docs_upd, reqs_upd = app_module.update_visibility(adapter)
        assert docs_upd["visible"] == (adapter in DOC_ADAPTERS_IN_CATALOG), adapter
        assert reqs_upd["visible"] == (adapter == "requirement-check"), adapter
    docs_upd, reqs_upd = app_module.update_visibility("None (standard chat)")
    assert not docs_upd["visible"] and not reqs_upd["visible"]


# ---------------------------------------------------------------- respond() streaming

def run_respond(app_module, message="Hello", history=None, adapter="None (standard chat)",
                docs="", requirements="", max_new_tokens=64, temperature=0.7):
    return list(
        app_module.respond(message, history or [], adapter, docs, requirements,
                           max_new_tokens, temperature)
    )


def test_standard_chat_streams_growing_text(app_module):
    chunks = run_respond(app_module)
    assert chunks, "respond() yielded nothing"
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.startswith(prev), "stream must be cumulative"
    assert chunks[-1].strip()
    # no adapter kwargs leaked into a plain chat
    assert "adapter_name" not in app_module._fake_tokenizer.last_template_kwargs


def test_history_becomes_messages(app_module):
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    run_respond(app_module, message="third", history=history)
    roles = [(m["role"], m["content"]) for m in app_module._fake_tokenizer.last_messages]
    assert roles == [("user", "first"), ("assistant", "second"), ("user", "third")]


@pytest.mark.parametrize("adapter", sorted(CATALOG_ADAPTERS))
def test_each_adapter_reaches_the_template(app_module, adapter):
    """End-to-end per adapter: FakeTokenizer raises on names the model lacks."""
    chunks = run_respond(app_module, adapter=adapter, docs="Paris is in France.",
                         requirements="Answer in one sentence.")
    assert chunks
    assert app_module._fake_tokenizer.last_template_kwargs["adapter_name"] == adapter
    assert f"<|{adapter}|>" in app_module._fake_tokenizer.last_prompt


@pytest.mark.parametrize("adapter", sorted(DOC_ADAPTERS_IN_CATALOG))
def test_doc_adapters_pass_documents(app_module, adapter):
    run_respond(app_module, adapter=adapter, docs="The sky is blue.")
    docs = app_module._fake_tokenizer.last_template_kwargs.get("documents")
    assert docs and docs[0]["text"] == "The sky is blue."


def test_docs_omitted_when_empty(app_module):
    run_respond(app_module, adapter="answerability", docs="   ")
    assert "documents" not in app_module._fake_tokenizer.last_template_kwargs


def test_requirement_check_injects_requirements(app_module):
    """The Requirements box must actually reach the prompt (protocol:
    the final user turn carries a <requirements> block — see
    requirement_check.py for the reference CLI implementation)."""
    run_respond(app_module, message="Check the last answer.",
                adapter="requirement-check",
                requirements="Must be under 50 words.")
    prompt = app_module._fake_tokenizer.last_prompt
    assert "<requirements>" in prompt
    assert "Must be under 50 words." in prompt


def test_requirements_ignored_for_other_adapters(app_module):
    run_respond(app_module, adapter="uncertainty", requirements="Must rhyme.")
    assert "Must rhyme." not in app_module._fake_tokenizer.last_prompt


def test_generation_params_forwarded(app_module):
    run_respond(app_module, max_new_tokens=128, temperature=0.0)
    gk = app_module._fake_model.last_generate_kwargs
    assert gk["max_new_tokens"] == 128
    assert gk["do_sample"] is False  # temperature 0 → greedy
