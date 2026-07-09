"""Pre-deployment checks for the ZeroGPU Space (app.py).

Run with:  .venv/bin/python -m pytest tests/ -v

The demo's user journey is two turns per interaction:
  1. the user picks a Core or Guardian adapter and submits a prompt
  2. turn 1 generates a standard (no adapter) response
  3. the adapter's follow-up prompt is auto-appended to the chat as a
     user message
  4. turn 2 generates the adapter's response to that follow-up

What this covers without a GPU:
  * every adapter name offered in the UI exists in the upstream catalog,
    and the UI offers exactly the Core + Guardian libraries
  * the Gradio UI builds under the pinned gradio version
  * adapter-specific input fields (documents / rules) toggle correctly
  * the two-turn flow: turn 1 has no adapter, the follow-up appears in the
    history, turn 2 activates the adapter and is greedy
  * documents and requirements reach the prompt where the protocol says
What it cannot cover: real weights, real GPU generation quality. After the
Space is up, send one message per adapter as a manual smoke test.
"""

import json
from pathlib import Path

import gradio as gr
import pytest

CATALOG = json.loads((Path(__file__).parent / "adapter_catalog.json").read_text())
CATALOG_ADAPTERS = set(CATALOG["adapters"])
CORE_GUARDIAN_ADAPTERS = {
    name for name, meta in CATALOG["adapters"].items()
    if meta["library"] in ("core", "guardian")
}
DOC_ADAPTERS_IN_CATALOG = {
    name for name in CORE_GUARDIAN_ADAPTERS
    if CATALOG["adapters"][name].get("needs_documents")
}


# ---------------------------------------------------------------- adapter names

def test_every_ui_adapter_exists_upstream(app_module):
    """A wrong adapter_name fails only at request time on the Space — catch it here."""
    unknown = [a for a in app_module.ADAPTER_CHOICES if a not in CATALOG_ADAPTERS]
    assert not unknown, f"UI offers adapters the model does not have: {unknown}"


def test_ui_offers_exactly_core_and_guardian(app_module):
    """The journey is scoped to the Core and Guardian libraries — no RAG, no gaps."""
    assert set(app_module.ADAPTER_CHOICES) == CORE_GUARDIAN_ADAPTERS


def test_doc_adapters_match_catalog(app_module):
    assert app_module.DOC_ADAPTERS == DOC_ADAPTERS_IN_CATALOG


def test_every_adapter_has_description(app_module):
    for choice in app_module.ADAPTER_CHOICES:
        assert choice in app_module.ADAPTER_DESCRIPTIONS, f"no description for {choice}"


def test_every_adapter_has_a_followup(app_module):
    for choice in app_module.ADAPTER_CHOICES:
        text = app_module.adapter_followup(choice, "some rule")
        assert text and text.strip(), f"empty follow-up for {choice}"


# aLoRA invocation markers from adapter_map in the model's
# chat_template.jinja: the template splices the activation token in where
# this text appears in the final user turn. Without the marker the adapter
# activates only at the generation boundary and answers in prose instead of
# its trained JSON protocol. (context-attribution is LoRA-flavored: no marker.)
INVOCATION_TEXT = {
    "requirement-check": "<requirements>",
    "uncertainty": "<certainty>",
    "guardian-core": "<guardian>",
    "factuality-detection": "<guardian>",
    "factuality-correction": "<guardian>",
    "policy-guardrails": "<guardian>",
}


@pytest.mark.parametrize("adapter", sorted(INVOCATION_TEXT))
def test_alora_followups_carry_their_invocation_marker(app_module, adapter):
    followup = app_module.adapter_followup(adapter, "some rule")
    assert INVOCATION_TEXT[adapter] in followup, (
        f"{adapter}'s follow-up must contain {INVOCATION_TEXT[adapter]} so the "
        "chat template can splice in the activation token"
    )


# ---------------------------------------------------------------- UI construction

def test_ui_builds(app_module):
    assert isinstance(app_module.demo, gr.Blocks)


def test_visibility_toggles(app_module):
    for adapter in CORE_GUARDIAN_ADAPTERS:
        docs_upd, rules_upd = app_module.update_visibility([adapter])
        assert docs_upd["visible"] == (adapter in DOC_ADAPTERS_IN_CATALOG), adapter
        assert rules_upd["visible"] == (
            adapter in {"requirement-check", "policy-guardrails"}
        ), adapter
    # any selected adapter needing a field is enough to show it
    docs_upd, rules_upd = app_module.update_visibility(
        ["uncertainty", "factuality-detection", "requirement-check"]
    )
    assert docs_upd["visible"] and rules_upd["visible"]
    docs_upd, rules_upd = app_module.update_visibility([])
    assert not docs_upd["visible"] and not rules_upd["visible"]


# ---------------------------------------------------------------- two-turn flow

def drive(app_module, message="Hello", adapter="uncertainty", docs="", rules="",
          max_new_tokens=64, temperature=0.7):
    """Run one full interaction; return every yielded history state."""
    adapters = [adapter] if isinstance(adapter, str) else adapter
    history = [{"role": "user", "content": message}]
    return list(
        app_module.bot_respond(history, adapters, docs, rules,
                               max_new_tokens, temperature)
    )


def test_two_turn_shape(app_module):
    states = drive(app_module)
    final = states[-1]
    assert [m["role"] for m in final] == ["user", "assistant", "user", "assistant"]
    assert final[1]["content"].strip(), "turn-1 generation is empty"
    assert final[3]["content"].strip(), "adapter response is empty"


def test_followup_appears_in_history_automatically(app_module):
    states = drive(app_module, adapter="uncertainty")
    final = states[-1]
    followup = app_module.adapter_followup("uncertainty", "")
    # displayed in purple italics via the adapter-prompt wrapper
    assert final[2]["content"] == app_module.followup_display(followup)
    assert 'class="adapter-prompt"' in final[2]["content"]
    # the follow-up must be yielded to the UI before the adapter response streams
    followup_first_seen = next(s for s in states if len(s) == 3)
    assert followup_first_seen[2]["role"] == "user"


def test_adapter_response_is_styled_and_round_trips(app_module):
    final = drive(app_module, adapter="uncertainty")[-1]
    assert 'class="adapter-response"' in final[3]["content"]
    clean = app_module._clean_content(final[3]["content"])
    assert clean and "adapter-response" not in clean


def test_followup_display_round_trips(app_module):
    """The styling wrapper must escape HTML-ish protocol text (e.g.
    <requirements>) for display, and _clean_content must recover the exact
    original for prompting."""
    followup = app_module.adapter_followup("requirement-check", "Must rhyme & scan.")
    shown = app_module.followup_display(followup)
    assert "<requirements>" not in shown  # escaped for display
    assert app_module._clean_content(shown) == followup


def test_user_submit_locks_input(app_module):
    upd_input, history, upd_btn = app_module.user_submit("hi", [])
    assert upd_input["visible"] is False, "input must hide after the single message"
    assert upd_btn["visible"] is False, "Send must hide after the single message"
    assert history[-1] == {"role": "user", "content": "hi"}


def test_turn1_has_no_adapter_turn2_has_adapter(app_module):
    gen = app_module.bot_respond(
        [{"role": "user", "content": "Hi"}], "guardian-core", "", "", 64, 0.7
    )
    for state in gen:
        # while turn 1 is streaming, the template must not have seen an adapter
        if len(state) == 2 and state[1]["content"]:
            assert "adapter_name" not in app_module._fake_tokenizer.last_template_kwargs
    assert app_module._fake_tokenizer.last_template_kwargs["adapter_name"] == "guardian-core"
    assert "<|guardian-core|>" in app_module._fake_tokenizer.last_prompt


def test_turn2_messages_are_task_response_followup(app_module):
    drive(app_module, message="my task", adapter="uncertainty")
    roles = [m["role"] for m in app_module._fake_tokenizer.last_messages]
    assert roles == ["user", "assistant", "user"]
    assert app_module._fake_tokenizer.last_messages[0]["content"] == "my task"
    assert app_module._fake_tokenizer.last_messages[2]["content"] == (
        app_module.adapter_followup("uncertainty", "")
    )


def test_streaming_is_cumulative_within_each_turn(app_module):
    states = drive(app_module)
    turn1 = [s[1]["content"] for s in states if len(s) == 2]
    # turn-2 partials are wrapped for styling; compare the unwrapped text
    turn2 = [app_module._clean_content(s[3]["content"]) for s in states if len(s) == 4]
    for chunk_list in (turn1, turn2):
        assert chunk_list, "no streamed states for a turn"
        for prev, cur in zip(chunk_list, chunk_list[1:]):
            assert cur.startswith(prev), "stream must be cumulative"


@pytest.mark.parametrize("adapter", sorted(CORE_GUARDIAN_ADAPTERS))
def test_each_adapter_reaches_the_template(app_module, adapter):
    """End-to-end per adapter: FakeTokenizer raises on names the model lacks."""
    states = drive(app_module, adapter=adapter, docs="Paris is in France.",
                   rules="Answer in one sentence.")
    assert states
    assert app_module._fake_tokenizer.last_template_kwargs["adapter_name"] == adapter
    assert f"<|{adapter}|>" in app_module._fake_tokenizer.last_prompt


@pytest.mark.parametrize("adapter", sorted(DOC_ADAPTERS_IN_CATALOG))
def test_doc_adapters_pass_documents_to_turn2(app_module, adapter):
    drive(app_module, adapter=adapter, docs="The sky is blue.")
    docs = app_module._fake_tokenizer.last_template_kwargs.get("documents")
    assert docs and docs[0]["text"] == "The sky is blue."


def test_non_doc_adapters_omit_documents_in_turn2(app_module):
    """Docs ground turn 1, but a non-doc adapter's turn must not receive them."""
    gen = app_module.bot_respond(
        [{"role": "user", "content": "Hi"}], "guardian-core",
        "The sky is blue.", "", 64, 0.7
    )
    for state in gen:
        # while turn 1 is streaming, its prompt should be grounded on the docs
        if len(state) == 2 and state[1]["content"]:
            docs = app_module._fake_tokenizer.last_template_kwargs.get("documents")
            assert docs and docs[0]["text"] == "The sky is blue."
    assert "documents" not in app_module._fake_tokenizer.last_template_kwargs


def test_hidden_textboxes_arrive_as_none(app_module):
    """Gradio passes None (not \"\") for hidden textboxes — regression for the
    AttributeError seen on the Space when docs/rules boxes were hidden."""
    history = [{"role": "user", "content": "Hi"}]
    states = list(
        app_module.bot_respond(history, "uncertainty", None, None, 64, 0.7)
    )
    assert [m["role"] for m in states[-1]] == ["user", "assistant", "user", "assistant"]


def test_docs_omitted_when_empty(app_module):
    drive(app_module, adapter="context-attribution", docs="   ")
    assert "documents" not in app_module._fake_tokenizer.last_template_kwargs


def test_requirement_check_followup_carries_requirements(app_module):
    """Protocol: the final user turn carries a <requirements> block plus the
    fixed evaluation instruction — see requirement_check.py, the reference
    CLI implementation."""
    drive(app_module, message="Write a haiku.", adapter="requirement-check",
          rules="Must be under 50 words.")
    followup = app_module._fake_tokenizer.last_messages[-1]["content"]
    assert followup.startswith("<requirements> Must be under 50 words.")
    assert app_module.EVALUATION_PROMPT in followup
    assert "<requirements>" in app_module._fake_tokenizer.last_prompt


def test_policy_guardrails_followup_carries_policy(app_module):
    drive(app_module, adapter="policy-guardrails", rules="No medical advice.")
    followup = app_module._fake_tokenizer.last_messages[-1]["content"]
    assert "No medical advice." in followup


def test_rules_ignored_for_other_adapters(app_module):
    drive(app_module, adapter="uncertainty", rules="Must rhyme.")
    assert "Must rhyme." not in app_module._fake_tokenizer.last_prompt


def test_multiple_adapters_run_sequentially(app_module):
    adapters = ["uncertainty", "guardian-core", "requirement-check"]
    final = drive(app_module, message="my task", adapter=adapters,
                  rules="Must be polite.")[-1]
    roles = [m["role"] for m in final]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant",
                     "user", "assistant"]
    # each adapter's own follow-up, in selection order, styled
    for i, adapter in enumerate(adapters):
        followup_msg = final[2 + 2 * i]["content"]
        assert 'class="adapter-prompt"' in followup_msg
        expected = app_module.adapter_followup(adapter, "Must be polite.")
        assert app_module._clean_content(followup_msg) == expected
    # every response (base + 3 adapters) carries a cache note; prompts don't
    for i, m in enumerate(final):
        has_note = bool(NOTE_RE.search(m["content"]))
        assert has_note == (m["role"] == "assistant"), f"message {i}"
    # the last adapter to run is the last one selected
    assert app_module._fake_tokenizer.last_template_kwargs["adapter_name"] == (
        "requirement-check"
    )


def test_each_adapter_turn_reuses_the_shared_prefix(app_module):
    final = drive(app_module, message="my task",
                  adapter=["uncertainty", "guardian-core"])[-1]
    _, t1, _ = map(int, NOTE_RE.search(final[1]["content"]).groups())
    for i in (3, 5):
        h, t, _ = map(int, NOTE_RE.search(final[i]["content"]).groups())
        assert h > t1, f"adapter response {i} must reuse turn 1's decode tokens"
        assert h < t


# ---------------------------------------------------------------- KV-cache notes

NOTE_RE = __import__("re").compile(r"⚡ `KV cache: (\d+)/(\d+) prompt tokens reused \((\d+)% hit\)`")


def test_cache_note_under_both_responses(app_module):
    final = drive(app_module, message="my task", adapter="uncertainty")[-1]
    for i in (1, 3):  # the notes belong to the responses...
        assert NOTE_RE.search(final[i]["content"]), f"no cache note on response {i}"
    for i in (0, 2):  # ...not to the prompts
        assert not NOTE_RE.search(final[i]["content"]), f"cache note on prompt {i}"


def test_turn1_is_cold_turn2_reuses_prefix(app_module):
    final = drive(app_module, message="my task", adapter="uncertainty")[-1]
    h1, t1, _ = map(int, NOTE_RE.search(final[1]["content"]).groups())
    h2, t2, _ = map(int, NOTE_RE.search(final[3]["content"]).groups())
    assert h1 == 0 and t1 > 0, "turn 1 must report a cold cache"
    assert 0 < h2 < t2, "adapter turn must reuse a proper prefix of its prompt"
    # decode tokens from turn 1 must be reused, not just its prompt: the
    # turn-2 input extends the actual turn-1 sequence tensor (re-tokenizing
    # would break at the first generated token — BPE re-encode != identity)
    assert h2 > t1, "adapter turn must also reuse turn 1's generated tokens"


def test_ui_decoration_never_reaches_the_prompt(app_module):
    # a second interaction whose incoming history carries notes and the
    # styled adapter prompt from the first (the UI is single-shot, but the
    # prompt path must stay clean regardless)
    final = drive(app_module, message="first question", adapter="uncertainty")[-1]
    history = final + [{"role": "user", "content": "second question"}]
    list(app_module.bot_respond(history, "uncertainty", "", "", 64, 0.7))
    for m in app_module._fake_tokenizer.last_messages:
        assert "KV cache:" not in m["content"]
        assert "adapter-prompt" not in m["content"]
        assert "adapter-response" not in m["content"]
    assert app_module._fake_tokenizer.last_messages[0]["content"] == "first question"
    # the styled follow-up from interaction 1 round-trips back to clean text
    assert app_module._fake_tokenizer.last_messages[2]["content"] == (
        app_module.adapter_followup("uncertainty", "")
    )


def test_adapter_turn_is_greedy(app_module):
    """The adapter is a judge, not a generator — turn 2 must be greedy even
    when the user's temperature slider is nonzero."""
    drive(app_module, temperature=0.9)
    gk = app_module._fake_model.last_generate_kwargs
    assert gk["do_sample"] is False


def test_generation_params_forwarded(app_module):
    drive(app_module, max_new_tokens=128)
    gk = app_module._fake_model.last_generate_kwargs
    assert gk["max_new_tokens"] == 128
