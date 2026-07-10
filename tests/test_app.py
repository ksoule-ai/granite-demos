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
    # the verdict bubble carries the parsed JSON, not just prose
    assert '{"certainty": 0.85}' in texts[-1]
    # the uncertainty control token reached the model exactly once
    assert len(fake_model.calls_for("uncertainty")) == 1
    ids = fake_model.calls_for("uncertainty")[0][1]
    assert CONTROL_TOKENS["uncertainty"] in ids


def test_guardian_verdict(app_module, fake_model):
    fake_model.script_judge("guardian-core", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("guardian-core",))[-1]
    verdict = assistant_texts(final)[-1]
    # bare verdict: JSON + reading, no "guardian-core (harm) →" prefix
    assert '"guardian"' in verdict and "risk detected" in verdict
    assert not verdict.replace('<span class="meta-note">', "").startswith("guardian-core")
    assert CONTROL_TOKENS["guardian-core"] in fake_model.calls_for("guardian-core")[0][1]


def test_multiple_judges_get_their_own_bubbles(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty", "guardian-core"))[-1]
    verdicts = [
        t for t in assistant_texts(final)
        if '"certainty"' in t or '"guardian"' in t
    ]
    assert len(verdicts) == 2, assistant_texts(final)
    assert '"certainty"' in verdicts[0]
    assert '"guardian"' in verdicts[1]
    judged = [adapter for adapter, _, _ in fake_model.calls if adapter]
    assert judged == ["uncertainty", "guardian-core"]


def test_meta_messages_purple_drafts_plain(app_module, fake_model):
    """Every non-draft assistant message carries the meta-note marker (light
    purple + italics via CSS); draft/answer bubbles stay plain markdown."""
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check", "uncertainty"),
                  rules="Anything.")[-1]
    texts = assistant_texts(final)
    drafts = [t for t in texts if fake_model.base_answer in t]
    metas = [t for t in texts if fake_model.base_answer not in t]
    assert drafts and metas, texts
    assert all("meta-note" not in t for t in drafts), drafts
    assert all(t.startswith('<span class="meta-note">') for t in metas), metas


def test_draft_streams_cumulatively_into_one_bubble(app_module, fake_model):
    """Draft tokens stream into a single assistant bubble: successive states
    grow the same bubble (each a prefix of the next), and the final state has
    exactly one answer bubble, not one per streamed chunk."""
    states = drive(app_module, adapters=("uncertainty",))
    versions = []
    for state in states:
        texts = [t for t in assistant_texts(state) if fake_model.base_answer.startswith(t)]
        if texts:
            versions.append(texts[0])
        assert len(texts) <= 1, "streaming must update one bubble, not append"
    assert len(versions) > 1, "no streamed intermediate states observed"
    for prev, cur in zip(versions, versions[1:]):
        assert cur.startswith(prev), "stream must be cumulative"
    assert versions[-1] == fake_model.base_answer


def test_no_status_bubbles_left_behind(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty", "guardian-core"))[-1]
    assert not any("⏳" in t for t in assistant_texts(final))


# --------------------------------------------------------------------- IVR

def test_ivr_retries_until_requirement_passes(app_module, fake_model):
    drafts = [f"{fake_model.base_answer} (v1)", f"{fake_model.base_answer} (v2)"]
    fake_model.script_drafts(drafts)
    fake_model.script_judge("requirement-check", ['{"score": "no"}', '{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Must mention anorthosite.")[-1]
    texts = assistant_texts(final)
    attempts = [t for t in texts if fake_model.base_answer in t]
    assert len(attempts) == 2, texts
    # drafts carry the italic attempt label plus a KV note; each checker
    # verdict is its own bubble
    assert attempts[0].startswith(f"*(Attempt 1)*\n{drafts[0]}")
    assert attempts[1].startswith(f"*(Attempt 2)*\n{drafts[1]}")
    assert all(KV_NOTE_RE.search(t) for t in attempts)
    checks = [t for t in texts if "requirement_check" in t]
    assert len(checks) == 2, texts
    # format: {json} — ✅/❌ note
    assert "— ❌ requirement not satisfied" in checks[0]
    assert "— ✅ requirement satisfied" in checks[1]
    assert all(t.index('{"requirement_check"') < t.index("—") for t in checks), checks
    # bubbles interleave: draft, check, draft, check
    assert texts.index(checks[0]) == texts.index(attempts[0]) + 1
    # the IVR outcome shares the final check's bubble rather than its own
    assert "converged on attempt 2 of 2" in checks[1]
    assert not any("converged" in t for t in texts if t not in checks)
    # the requirement-check aLoRA judged each attempt
    assert len(fake_model.calls_for("requirement-check")) == 2


def test_ivr_stops_at_first_pass(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Anything.", loop_budget=5)[-1]
    assert len(fake_model.calls_for("requirement-check")) == 1
    assert any("converged on attempt 1 of 1" in t for t in assistant_texts(final))


def test_ivr_reports_budget_exhaustion(app_module, fake_model):
    fake_model.script_drafts(
        [f"{fake_model.base_answer} (v{i})" for i in (1, 2, 3)]
    )
    fake_model.script_judge(
        "requirement-check", ['{"score": "no"}'] * 3
    )
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Impossible requirement.", loop_budget=3)[-1]
    texts = assistant_texts(final)
    exhausted = [t for t in texts if "budget exhausted" in t.lower()]
    assert exhausted, texts
    # new wording, merged into the last requirement-check bubble
    assert "will revert to attempt 1 for the rest of the conversation" in exhausted[0]
    assert '{"requirement_check"' in exhausted[0]
    assert len(fake_model.calls_for("requirement-check")) == 3


def test_attempt_prefix_only_in_ivr_mode(app_module, fake_model):
    """Drafts show '(Attempt X)' only when the requirement checker runs; the
    prefix is display-only (the judge prompt sees the clean draft), and it is
    present while the draft is still streaming."""
    states = drive(app_module, adapters=("requirement-check",), rules="Anything.")
    streamed = [t for s in states for t in assistant_texts(s) if t.startswith("*(Attempt 1)*")]
    assert streamed, "no prefixed draft states observed"
    assert any(t != streamed[-1] for t in streamed), "prefix missing during streaming"
    tok = app_module.tokenizer
    judge_prompt = tok.decode(fake_model.calls_for("requirement-check")[0][1])
    assert "(Attempt" not in judge_prompt
    fake_model.reset()
    final = drive(app_module, adapters=("uncertainty",))[-1]
    assert not any("(Attempt" in t for t in assistant_texts(final))


def test_identical_attempt_reuses_verdict_and_retries(app_module, fake_model):
    """A draft identical to an already-failed one is not re-judged — with KV
    reuse, bf16 prefill noise can flip a borderline verdict on the exact same
    text (identical answers scored fail then pass on the Space). The loop
    notes the repeat and moves on to the next attempt."""
    fake_model.script_judge("requirement-check", ['{"score": "no"}'])
    final = drive(app_module, adapters=("requirement-check",),
                  rules="Impossible requirement.", loop_budget=3)[-1]
    texts = assistant_texts(final)
    # all three drafts are identical -> only the first is judged
    assert len(fake_model.calls_for("requirement-check")) == 1
    attempts = [t for t in texts if fake_model.base_answer in t]
    assert len(attempts) == 3, texts
    repeats = [t for t in texts if "identical attempt — retrying" in t]
    assert len(repeats) == 2, texts
    # repeats follow their attempts; the exhaustion note still lands
    assert texts.index(repeats[0]) == texts.index(attempts[1]) + 1
    assert any("budget exhausted" in t.lower() for t in texts)


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


def test_ivr_judge_sees_the_draft_answer(app_module, fake_model):
    """mellea 0.6.0's stock sampling strategies validate with output=, whose
    SimpleContext has an empty generation view — the requirement-check aLoRA
    judged only the eval turn, never the draft (blind-judge upstream issue).
    The manual IVR loop validates on the live context, so every judge prompt
    must contain the draft answer being judged."""
    fake_model.script_judge("requirement-check", ['{"score": "no"}', '{"score": "yes"}'])
    drive(app_module, adapters=("requirement-check",), rules="Must mention rock.")
    tok = app_module.tokenizer
    judge_calls = fake_model.calls_for("requirement-check")
    assert judge_calls, "no judge calls recorded"
    for _, ids, _ in judge_calls:
        assert fake_model.base_answer in tok.decode(ids), (
            "judge prompt does not contain the draft answer — blind judging"
        )


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


# ------------------------------------------------------------------ KV cache

KV_NOTE_RE = re.compile(r"⚡ `?KV cache: (\d+)/(\d+) prompt tokens reused \((\d+)% hit\)`?")


def _prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _draft_identity(app_module, fake_model, draft_ids):
    """The token ids the backend records as the draft's cache identity:
    prompt ids + generated answer ids + eos (FakeSwitchModel's sequences)."""
    tok = app_module.tokenizer
    answer = tok(fake_model.base_answer, add_special_tokens=False).input_ids
    return list(draft_ids) + list(answer) + [tok.eos_token_id]


def test_kv_cold_draft_reports_zero_hits(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty",))[-1]
    adapter, cache_len, ids = fake_model.cache_calls[0]
    assert adapter is None and cache_len == 0
    draft = assistant_texts(final)[0]
    m = KV_NOTE_RE.search(draft)
    assert m, draft
    assert (int(m.group(1)), int(m.group(2)), m.group(3)) == (0, len(ids), "0")


def test_kv_judge_reuses_draft_prefix(app_module, fake_model):
    final = drive(app_module, adapters=("uncertainty",))[-1]
    draft_ids = fake_model.cache_calls[0][2]
    _, cache_len, judge_ids = next(
        c for c in fake_model.cache_calls if c[0] == "uncertainty"
    )
    identity = _draft_identity(app_module, fake_model, draft_ids)
    expected = max(0, min(
        _prefix_len(identity, judge_ids), len(identity) - 1, len(judge_ids) - 1
    ))
    assert cache_len == expected > 0
    bubble = next(t for t in assistant_texts(final) if '"certainty"' in t)
    assert f"<br>⚡ KV cache: {expected}/{len(judge_ids)}" in bubble
    assert "`" not in bubble  # no markdown inside meta spans


def test_kv_second_attempt_reuses_prompt_prefix(app_module, fake_model):
    fake_model.script_drafts(
        [f"{fake_model.base_answer} (v1)", f"{fake_model.base_answer} (v2)"]
    )
    fake_model.script_judge("requirement-check", ['{"score": "no"}', '{"score": "yes"}'])
    drive(app_module, adapters=("requirement-check",), rules="Must mention rock.")
    kinds = [c[0] for c in fake_model.cache_calls]
    assert kinds == [None, "requirement-check", None, "requirement-check"]
    # After a judge runs, the cache identity is the judge's INPUT ids; the
    # next draft can reuse their shared instruction prefix.
    judge1_ids = fake_model.cache_calls[1][2]
    _, cache_len2, attempt2_ids = fake_model.cache_calls[2]
    expected = max(0, min(
        _prefix_len(judge1_ids, attempt2_ids),
        len(judge1_ids),  # judge cache is longer; identity clamps reuse
        len(attempt2_ids) - 1,
    ))
    assert cache_len2 == expected > 0


def test_kv_stats_attribution_order(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    final = drive(app_module,
                  adapters=("requirement-check", "uncertainty", "guardian-core"),
                  rules="Anything.")[-1]
    notes = []
    for t in assistant_texts(final):
        notes.extend(KV_NOTE_RE.findall(t))
    assert len(notes) == len(fake_model.cache_calls) == 4
    for (hits, total, _pct), (adapter, cache_len, ids) in zip(
        notes, fake_model.cache_calls
    ):
        assert int(hits) == cache_len, adapter
        assert int(total) == len(ids), adapter


def test_kv_note_never_reaches_prompts(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    drive(app_module, adapters=("requirement-check", "uncertainty"), rules="Anything.")
    tok = app_module.tokenizer
    for _, ids, _ in fake_model.calls:
        assert "KV cache" not in tok.decode(ids)


def test_kv_math_failure_falls_back(app_module, fake_model, monkeypatch):
    """Cache-math errors must degrade to cache-less generation, not crash."""
    import switch_backend

    def boom(a, b):
        raise RuntimeError("synthetic cache-math failure")

    monkeypatch.setattr(switch_backend, "_common_prefix_len", boom)
    final = drive(app_module, adapters=("uncertainty",))[-1]
    # first draft never computes a prefix (no identity yet) — still cached
    assert fake_model.cache_calls[0][1] == 0
    # the judge's math failed: pristine call, no cache injected
    judge = next(c for c in fake_model.cache_calls if c[0] == "uncertainty")
    assert judge[1] is None
    bubble = next(t for t in assistant_texts(final) if '"certainty"' in t)
    assert f"⚡ KV cache: 0/{len(judge[2])}" in bubble


def test_kv_resets_between_interactions(app_module, fake_model):
    drive(app_module, adapters=("uncertainty",))
    drive(app_module, adapters=("uncertainty",))
    drafts = [c for c in fake_model.cache_calls if c[0] is None]
    assert len(drafts) == 2
    assert drafts[1][1] == 0, "second interaction must start cold"


def test_final_note_inserted_before_kv_footnote(app_module, fake_model):
    fake_model.script_judge("requirement-check", ['{"score": "yes"}'])
    final = drive(app_module, adapters=("requirement-check",), rules="Anything.")[-1]
    check = next(t for t in assistant_texts(final) if "requirement_check" in t)
    assert check.index("converged") < check.index("⚡ KV cache"), check
    assert check.endswith("</span>")
