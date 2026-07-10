"""Pre-deployment checks for the ZeroGPU Space (app.py).

Run with:  .venv/bin/python -m pytest tests/ -v

The demo drives the Granite Switch checkpoint through Mellea's HF backend
(switch_backend.SwitchBackend). One interaction is:
  1. the user picks adapters, optionally states requirements, submits a prompt
  2. with requirement-check selected, Mellea runs instruct → validate → repair:
     drafts are judged by the embedded requirement-check aLoRA until one
     passes or the attempt budget runs out
  3. uncertainty / guardian-core then judge the final answer (purple bubbles)

What this covers without a GPU (the real tokenizer + chat template render
every prompt; only model.generate is faked):
  * every adapter offered in the UI exists in the upstream catalog
  * adapter activation control tokens reach the model's prompt for every
    judged turn, and never for plain generation
  * the IVR loop: retries on a failing verdict, stops on a passing one,
    reports budget exhaustion, and keeps judge turns greedy
  * the Gradio UI builds; adapter-specific fields toggle correctly
What it cannot cover: real weights, real GPU generation quality. After the
Space is up, send one message per adapter as a manual smoke test.
"""

import json
import re
from pathlib import Path

import gradio as gr
import pytest

from conftest import CONTROL_TOKENS

CATALOG = json.loads((Path(__file__).parent / "adapter_catalog.json").read_text())
CATALOG_ADAPTERS = set(CATALOG["adapters"])

# The demo's curated roster
EXPECTED_ADAPTERS = {"requirement-check", "uncertainty", "guardian-core"}


# ---------------------------------------------------------------- adapter names

def test_every_ui_adapter_exists_upstream(app_module):
    """A wrong adapter name fails only at request time on the Space — catch it here."""
    unknown = [a for a in app_module.ADAPTER_CHOICES if a not in CATALOG_ADAPTERS]
    assert not unknown, f"UI offers adapters the model does not have: {unknown}"


def test_ui_offers_expected_adapters(app_module):
    assert set(app_module.ADAPTER_CHOICES) == EXPECTED_ADAPTERS


def test_every_adapter_has_description(app_module):
    for choice in app_module.ADAPTER_CHOICES:
        assert choice in app_module.ADAPTER_DESCRIPTIONS, f"no description for {choice}"


def test_backend_registered_all_embedded_adapters(app_module):
    for name in EXPECTED_ADAPTERS:
        assert app_module.backend.has_embedded_adapter(name), name


# ---------------------------------------------------------------- UI construction

def test_ui_builds(app_module):
    assert isinstance(app_module.demo, gr.Blocks)


def test_visibility_toggles(app_module):
    rules_upd, budget_upd = app_module.update_visibility(["requirement-check"])
    assert rules_upd["visible"] and budget_upd["visible"]
    rules_upd, budget_upd = app_module.update_visibility(["uncertainty"])
    assert not rules_upd["visible"] and not budget_upd["visible"]
    rules_upd, budget_upd = app_module.update_visibility([])
    assert not rules_upd["visible"] and not budget_upd["visible"]


def test_user_submit_locks_input(app_module):
    upd_input, history, upd_btn = app_module.user_submit("hi", [])
    assert upd_input["visible"] is False, "input must hide after the single message"
    assert upd_btn["visible"] is False, "Send must hide after the single message"
    assert history[-1] == {"role": "user", "content": "hi"}


# ------------------------------------------------------------------ driving

def drive(app_module, message="Tell me about the moon.", adapters=("uncertainty",),
          rules="", max_new_tokens=128, temperature=0.7, loop_budget=3):
    """Run one full interaction; return every yielded history state."""
    history = [{"role": "user", "content": message}]
    return list(
        app_module.bot_respond(history, list(adapters), rules,
                               max_new_tokens, temperature, loop_budget)
    )


def assistant_texts(state):
    return [m["content"] for m in state if m["role"] == "assistant"]


def test_block_format_user_content(app_module, fake_model):
    """Gradio 6 round-trips Chatbot content as a list of blocks — the prompt
    must be flattened to a plain string before it reaches mellea (regression
    for the 'Exception: Type Error' from blockify on the Space)."""
    history = [{"role": "user", "content": [{"type": "text", "text": "Tell me about basalt."}]}]
    states = list(app_module.bot_respond(history, ["uncertainty"], "", 64, 0.0, 3))
    assert states, "no states yielded"
    assert fake_model.base_answer in assistant_texts(states[-1])[0]


# ------------------------------------------------------------- plain + judges

def test_plain_generation_has_no_control_token(app_module, fake_model):
    drive(app_module, adapters=())
    assert fake_model.calls, "no generation happened"
    assert all(adapter is None for adapter, _, _ in fake_model.calls)


def test_judge_adapter_activates_and_reports(app_module, fake_model):
    fake_model.script_judge("uncertainty", ['{"score": "8"}'])
    final = drive(app_module, adapters=("uncertainty",))[-1]
    # base answer bubble + purple verdict bubble
    texts = assistant_texts(final)
    assert fake_model.base_answer in texts[0]
    assert 'class="adapter-response"' in texts[-1]
    assert "0.85" in texts[-1]
    # the uncertainty control token reached the model exactly once
    assert len(fake_model.calls_for("uncertainty")) == 1
    ids = fake_model.calls_for("uncertainty")[0][1]
    assert CONTROL_TOKENS["uncertainty"] in ids


def test_guardian_verdict(app_module, fake_model):
    fake_model.script_judge("guardian-core", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("guardian-core",))[-1]
    verdict = assistant_texts(final)[-1]
    assert "guardian-core" in verdict and "risk detected" in verdict
    assert CONTROL_TOKENS["guardian-core"] in fake_model.calls_for("guardian-core")[0][1]


def test_multiple_judges_run_in_selection_order(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty", "guardian-core"))[-1]
    verdicts = [t for t in assistant_texts(final) if 'adapter-response' in t]
    assert len(verdicts) == 2
    assert "uncertainty" in verdicts[0] and "guardian-core" in verdicts[1]
    judged = [adapter for adapter, _, _ in fake_model.calls if adapter]
    assert judged == ["uncertainty", "guardian-core"]


def test_no_status_bubbles_left_behind(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty", "guardian-core"))[-1]
    assert not any('adapter-prompt' in t for t in assistant_texts(final))


# --------------------------------------------------------------------- IVR

def test_ivr_retries_until_requirement_passes(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "no"}', '{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Must mention anorthosite.")[-1]
    texts = assistant_texts(final)
    attempts = [t for t in texts if fake_model.base_answer in t]
    assert len(attempts) == 2, texts
    assert "❌" in attempts[0] and "✅" in attempts[1]
    assert any("converged on attempt 2 of 2" in t for t in texts)
    # the requirement-check aLoRA judged each attempt
    assert len(fake_model.calls_for("requirement-check")) == 2


def test_ivr_stops_at_first_pass(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Anything.", loop_budget=5)[-1]
    assert len(fake_model.calls_for("requirement-check")) == 1
    assert any("converged on attempt 1 of 1" in t for t in assistant_texts(final))


def test_ivr_reports_budget_exhaustion(app_module, fake_model):
    fake_model.script_judge(
        "requirement-check", ['{"score": "no"}'] * 3
    )
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Impossible requirement.", loop_budget=3)[-1]
    texts = assistant_texts(final)
    assert any("budget exhausted" in t.lower() for t in texts), texts
    assert len(fake_model.calls_for("requirement-check")) == 3


def test_ivr_requires_rules(app_module, fake_model):
    """requirement-check without requirements degrades to plain generation."""
    drive(app_module, adapters=("requirement-check",), rules="   ")
    assert fake_model.calls_for("requirement-check") == []
    assert any(adapter is None for adapter, _, _ in fake_model.calls)


def test_judge_turns_stay_greedy(app_module, fake_model):
    """Adapters are judges, not generators: user temperature must not reach
    the requirement-check validation calls (io.yaml pins temperature 0.0)."""
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    drive(app_module, adapters=("requirement-check",),
          rules="Must be a haiku.", temperature=0.9)
    base_calls = [c for c in fake_model.calls if c[0] is None]
    judge_calls = fake_model.calls_for("requirement-check")
    assert base_calls and judge_calls
    # drafts sample at the user's temperature
    assert base_calls[0][2].get("do_sample") is True
    assert base_calls[0][2].get("temperature") == pytest.approx(0.9)
    # the judge stays greedy
    for _, _, kwargs in judge_calls:
        assert kwargs.get("do_sample") is False, kwargs
        assert "temperature" not in kwargs or kwargs["temperature"] in (None, 0.0)


def test_requirement_text_reaches_the_judge_prompt(app_module, fake_model):
    """The io.yaml instruction template must carry the user's requirement."""
    rules = "Must mention exactly three moons of Jupiter."
    drive(app_module, adapters=("requirement-check",), rules=rules)
    tok = app_module.tokenizer
    ids = fake_model.calls_for("requirement-check")[0][1]
    assert rules in tok.decode(ids)


def test_judge_max_tokens_come_from_io_yaml(app_module, fake_model):
    """User max_new_tokens must not override the judge's tiny token budget."""
    drive(app_module, adapters=("uncertainty",), max_new_tokens=2048)
    kwargs = fake_model.calls_for("uncertainty")[0][2]
    assert kwargs.get("max_new_tokens") == 15  # io.yaml max_completion_tokens
