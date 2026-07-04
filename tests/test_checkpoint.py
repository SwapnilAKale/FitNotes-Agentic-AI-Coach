"""
tests/test_checkpoint.py

Checkpoint/resume for quota-interrupted questions. No Gemini calls — every
LLM stage function is monkeypatched.

 a. 429 in grounding  → checkpoint saved with VERBATIM draft,
    completed_stage="draft", user-facing status contains NO draft text.
 b. Resume            → draft LLM NOT called again, grounding runs on the
    stored draft, checkpoint cleared.
 c. 429 during draft  → checkpoint has draft:null; resume runs draft.
 d. New question while checkpoint pending → slot discarded.
 e. Stale checkpoint (>48h) discarded on load.
 f. Operational: oversized tool result mechanically pruned on save with
    marker; ids in kept portions intact.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src import checkpoint as ckpt                      # noqa: E402
from src import coordinator as coordinator_mod          # noqa: E402
from src import analysis_agent as analysis_mod          # noqa: E402
from src.coordinator import Coordinator                 # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────────

ANALYTICAL_PARAMS = {
    "route": "analytical", "exercise_names": None, "muscle_groups": None,
    "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None,
}

FAKE_PKG = {"scope": "broad", "query_period_days": 90, "exercises": []}

RATE_LIMIT_MSG = "429 RESOURCE_EXHAUSTED. {'retryDelay': '30s'}"


@pytest.fixture()
def slot(tmp_path, monkeypatch):
    """Point the checkpoint slot at a temp file; return its path."""
    path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("CHECKPOINT_PATH", str(path))
    return path


@pytest.fixture()
def coord(monkeypatch):
    """Coordinator with no Gemini client, no agent, free package build."""
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)

    async def fake_classify(question):
        return dict(ANALYTICAL_PARAMS)
    monkeypatch.setattr(c, "_classify", fake_classify)

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: dict(FAKE_PKG))
    # Memory retrieval is environment-dependent — neutralize it
    try:
        import src.shared.memory as shared_memory
        monkeypatch.setattr(shared_memory, "retrieve_relevant_memories",
                            lambda q: None)
    except ImportError:
        pass
    return c


def _patch_stages(monkeypatch, analyze=None, ground=None):
    """Install fake analyze/ground_check on the analysis_agent module."""
    if analyze is not None:
        monkeypatch.setattr(analysis_mod, "analyze", analyze)
    if ground is not None:
        monkeypatch.setattr(analysis_mod, "ground_check", ground)


def _read_slot(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── a. 429 in grounding → verbatim draft saved, no draft text to user ─────────

DRAFT_TEXT = "DRAFT-SENTINEL: bench went from 100 lbs to 130 lbs over 8 weeks."


def test_a_grounding_429_saves_verbatim_draft(slot, coord, monkeypatch):
    async def fake_analyze(*a, **k):
        return DRAFT_TEXT

    async def fake_ground(*a, **k):
        raise RuntimeError(RATE_LIMIT_MSG)

    _patch_stages(monkeypatch, fake_analyze, fake_ground)

    with pytest.raises(ckpt.QuotaInterrupted) as ei:
        asyncio.run(coord.route("how is my bench progressing?"))

    e = ei.value
    assert e.checkpoint_saved is True
    # POLICY: the user never sees unverified draft text
    assert "DRAFT-SENTINEL" not in (e.user_message or "")
    assert "DRAFT-SENTINEL" not in str(e)
    assert "continue" in e.user_message.lower()

    cp = _read_slot(slot)
    assert cp["route"] == "analytical"
    assert cp["completed_stage"] == "draft"
    assert cp["draft"] == DRAFT_TEXT          # VERBATIM, never summarized
    assert cp["question"] == "how is my bench progressing?"
    assert cp["params"]["route"] == "analytical"


# ── b. Resume: draft NOT re-called, grounding runs on stored draft ────────────

def test_b_resume_skips_draft_and_clears(slot, coord, monkeypatch):
    ckpt.save_checkpoint(route="analytical",
                         question="how is my bench progressing?",
                         params=dict(ANALYTICAL_PARAMS),
                         completed_stage="draft", draft=DRAFT_TEXT)

    calls = {"analyze": 0, "ground": 0}

    async def fake_analyze(*a, **k):
        calls["analyze"] += 1
        return "SHOULD-NEVER-BE-USED"

    async def fake_ground(draft, pkg):
        calls["ground"] += 1
        assert draft == DRAFT_TEXT            # grounding sees the exact text
        return "VERIFIED: " + draft, [{"action": "qualified"}]

    _patch_stages(monkeypatch, fake_analyze, fake_ground)

    async def fake_coverage(question, answer):
        return answer, True
    monkeypatch.setattr(coord, "_coverage_check", fake_coverage)

    result = asyncio.run(coord.route("continue"))

    assert calls["analyze"] == 0              # completed stage NOT re-paid
    assert calls["ground"] == 1
    assert result["answer"] == "VERIFIED: " + DRAFT_TEXT
    assert result["route"] == "analytical"
    assert not slot.exists()                  # cleared on success


# ── c. 429 during draft → draft:null; resume runs the draft stage ─────────────

def test_c_draft_429_then_resume_runs_draft(slot, coord, monkeypatch):
    async def failing_analyze(*a, **k):
        raise RuntimeError(RATE_LIMIT_MSG)

    _patch_stages(monkeypatch, analyze=failing_analyze)

    with pytest.raises(ckpt.QuotaInterrupted) as ei:
        asyncio.run(coord.route("how is my bench progressing?"))

    assert "your question is saved" in ei.value.user_message.lower()
    assert "(countdown: 30s)" in ei.value.user_message

    cp = _read_slot(slot)
    assert cp["completed_stage"] == "classify"
    assert cp["draft"] is None

    # Resume: now the draft stage runs (once), then grounding + coverage
    calls = {"analyze": 0}

    async def ok_analyze(*a, **k):
        calls["analyze"] += 1
        return DRAFT_TEXT

    async def ok_ground(draft, pkg):
        return draft, []

    _patch_stages(monkeypatch, ok_analyze, ok_ground)

    async def fake_coverage(question, answer):
        return answer, True
    monkeypatch.setattr(coord, "_coverage_check", fake_coverage)

    result = asyncio.run(coord.route("continue"))
    assert calls["analyze"] == 1
    assert result["answer"] == DRAFT_TEXT
    assert not slot.exists()


# ── d. Confirm-before-discard: a NEW question never silently drops the slot ───

SAVED_Q = "how is my bench progressing?"
NEW_Q   = "how many sessions this month?"


def _awaiting_slot(saved_q=SAVED_Q, new_q=NEW_Q):
    """Create a live slot already awaiting a discard-confirmation reply."""
    cp = ckpt.save_checkpoint(route="analytical", question=saved_q,
                              params=dict(ANALYTICAL_PARAMS),
                              completed_stage="draft", draft=DRAFT_TEXT)
    return ckpt.mark_awaiting_discard(cp, new_q)


def _count_processing(coord, monkeypatch, op_route="operational"):
    """Patch both pipelines with counters; return the dict."""
    seen = {"op": 0, "an": 0, "last": None}

    async def fake_op(q, **kw):          # **kw: /log boundary flags (ignored here)
        seen["op"] += 1; seen["last"] = q
        return "OP:" + q
    monkeypatch.setattr(coord, "_run_operational", fake_op)

    async def fake_an(q, p, resume=None):
        seen["an"] += 1; seen["last"] = q
        return "AN:" + q, []
    monkeypatch.setattr(coord, "_run_analytical", fake_an)

    async def fake_classify(q):
        return {**ANALYTICAL_PARAMS, "route": op_route}
    monkeypatch.setattr(coord, "_classify", fake_classify)
    return seen


def test_d_new_question_prompts_keeps_slot(slot, coord, monkeypatch):
    ckpt.save_checkpoint(route="analytical", question=SAVED_Q,
                         params=dict(ANALYTICAL_PARAMS),
                         completed_stage="draft", draft=DRAFT_TEXT)
    seen = _count_processing(coord, monkeypatch)

    result = asyncio.run(coord.route(NEW_Q))

    # Confirm prompt returned; new question NOT processed; slot intact.
    assert result["route"] == "checkpoint_confirm"
    assert SAVED_Q in result["answer"]
    assert "continue" in result["answer"].lower()
    assert seen["op"] == 0 and seen["an"] == 0
    cp = _read_slot(slot)
    assert cp["awaiting_discard_confirm"] is True
    assert cp["pending_question"] == NEW_Q
    # No draft text ever leaks into the prompt
    assert "DRAFT-SENTINEL" not in result["answer"]


def test_d_a_continue_after_prompt_resumes_saved(slot, coord, monkeypatch):
    _awaiting_slot()
    calls = {"analyze": 0, "ground": 0}

    async def fake_analyze(*a, **k):
        calls["analyze"] += 1; return "SHOULD-NOT-RUN"

    async def fake_ground(draft, pkg):
        calls["ground"] += 1
        assert draft == DRAFT_TEXT
        return "VERIFIED: " + draft, []

    _patch_stages(monkeypatch, fake_analyze, fake_ground)

    async def fake_coverage(q, a):
        return a, True
    monkeypatch.setattr(coord, "_coverage_check", fake_coverage)

    result = asyncio.run(coord.route("continue"))
    assert calls["analyze"] == 0 and calls["ground"] == 1
    assert result["answer"] == "VERIFIED: " + DRAFT_TEXT
    assert not slot.exists()                  # resumed + cleared


def test_d_b_resend_question_discards_and_processes(slot, coord, monkeypatch):
    _awaiting_slot()
    seen = _count_processing(coord, monkeypatch)

    result = asyncio.run(coord.route(NEW_Q))     # re-send the new question
    assert not slot.exists()                     # discarded
    assert result["route"] == "operational"
    assert seen["op"] == 1 and seen["last"] == NEW_Q


def test_d_discard_keyword_processes_stashed(slot, coord, monkeypatch):
    _awaiting_slot()
    seen = _count_processing(coord, monkeypatch)

    result = asyncio.run(coord.route("new"))     # explicit discard keyword
    assert not slot.exists()
    # The stashed new question (not the bare "new") is what gets processed
    assert seen["op"] == 1 and seen["last"] == NEW_Q


def test_d_ambiguous_reply_reasks_keeps_slot(slot, coord, monkeypatch):
    _awaiting_slot()
    seen = _count_processing(coord, monkeypatch)

    result = asyncio.run(coord.route("ok"))      # neither continue nor new
    assert result["route"] == "checkpoint_confirm"
    assert seen["op"] == 0 and seen["an"] == 0
    assert slot.exists()                         # never silently discarded


def test_d_c_stale_slot_no_prompt_processes_directly(slot, coord, monkeypatch):
    cp = ckpt.save_checkpoint(route="analytical", question="old",
                              params=dict(ANALYTICAL_PARAMS),
                              completed_stage="draft", draft=DRAFT_TEXT)
    cp["created"] = (datetime.now() - timedelta(hours=49)).isoformat()
    with open(slot, "w", encoding="utf-8") as f:
        json.dump(cp, f)
    seen = _count_processing(coord, monkeypatch)

    result = asyncio.run(coord.route(NEW_Q))
    assert result["route"] == "operational"      # processed directly, no prompt
    assert seen["op"] == 1 and seen["last"] == NEW_Q
    assert not slot.exists()                      # stale slot dropped on load


def test_d2_continue_without_checkpoint_is_nothing_to_resume(slot, coord, monkeypatch):
    # Nothing-to-resume guard (replaces the old fall-through): "continue" with
    # no live slot must NOT classify/run a fresh question — it returns a notice.
    called = {"classify": 0}

    async def fake_classify(question):
        called["classify"] += 1
        return {**ANALYTICAL_PARAMS, "route": "operational"}
    monkeypatch.setattr(coord, "_classify", fake_classify)

    result = asyncio.run(coord.route("continue"))
    assert result["route"] == "none"
    assert "no saved question" in result["answer"].lower()
    assert called["classify"] == 0            # no LLM call — no double-pay


# ── e. Stale checkpoint discarded on load ─────────────────────────────────────

def test_e_stale_checkpoint_discarded(slot):
    cp = ckpt.save_checkpoint(route="analytical", question="q",
                              params=dict(ANALYTICAL_PARAMS),
                              completed_stage="draft", draft=DRAFT_TEXT)
    cp["created"] = (datetime.now() - timedelta(hours=49)).isoformat()
    with open(slot, "w", encoding="utf-8") as f:
        json.dump(cp, f)

    assert ckpt.load_checkpoint() is None
    assert not slot.exists()

    # Just-under-48h slot survives
    cp["created"] = (datetime.now() - timedelta(hours=47)).isoformat()
    with open(slot, "w", encoding="utf-8") as f:
        json.dump(cp, f)
    assert ckpt.load_checkpoint() is not None


# ── f. Operational pruning: mechanical head+tail, ids intact ──────────────────

def test_f_operational_prune_on_save(slot):
    big = "ID_HEAD_12345 weight=130.0 " + ("x" * 6000) + " ID_TAIL_67890 reps=8"
    messages = [
        {"role": "user", "content": "log my workout"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0_0", "type": "function",
             "function": {"name": "query_workout_data", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_0_0", "content": big},
        {"role": "tool", "tool_call_id": "call_0_1", "content": "short result"},
    ]
    ckpt.save_checkpoint(route="operational", question="log my workout",
                         messages=messages)

    cp = _read_slot(slot)
    pruned = cp["messages"][2]["content"]
    assert ckpt.PRUNE_MARKER in pruned
    assert len(pruned) <= ckpt.PRUNE_TOOL_RESULT_AT + len(ckpt.PRUNE_MARKER) + 2
    # Exact values in the kept head/tail untouched
    assert pruned.startswith("ID_HEAD_12345 weight=130.0")
    assert pruned.endswith("ID_TAIL_67890 reps=8")
    # Small tool results and non-tool messages untouched
    assert cp["messages"][3]["content"] == "short result"
    assert cp["messages"][0]["content"] == "log my workout"
    assert cp["messages"][1]["tool_calls"][0]["function"]["name"] == "query_workout_data"
    # The in-memory list passed by the caller is not mutated
    assert messages[2]["content"] == big


# ── extra: continue-intent matcher sanity ─────────────────────────────────────

def test_continue_intent_matcher():
    for msg in ("continue", "Continue", "  resume ", "carry on", "please continue",
                "ok continue", "Resume.", "continue!"):
        assert ckpt.is_continue_intent(msg), msg
    for msg in ("continue my bench analysis please and also add a set",
                "how is my bench?", "", "log 130 lbs", "carry the bar on"):
        assert not ckpt.is_continue_intent(msg), msg


# ══════════════════════════════════════════════════════════════════════════════
# Per-minute vs daily 429 — silent retry, classify-gap closure, nothing-to-resume
# ══════════════════════════════════════════════════════════════════════════════

_PER_MINUTE_BODY = (
    "429 RESOURCE_EXHAUSTED. {'quotaId': "
    "'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'retryDelay': '53s'}"
)
_DAILY_BODY = (
    "429 RESOURCE_EXHAUSTED. {'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'retryDelay': '30s'}"
)


class _Fake429(Exception):
    pass


def _per_minute_exc():
    return _Fake429(_PER_MINUTE_BODY)


def _daily_exc():
    return _Fake429(_DAILY_BODY)


def _no_sleep(monkeypatch):
    """Patch asyncio.sleep in the coordinator to a recording no-op."""
    waits = []
    async def fake_sleep(s):
        waits.append(s)
    monkeypatch.setattr(coordinator_mod.asyncio, "sleep", fake_sleep)
    return waits


# ── classifier: is_per_minute_quota / retry_delay_seconds ─────────────────────

def test_per_minute_classifier_and_delay():
    pm, dy = _per_minute_exc(), _daily_exc()
    assert ckpt.is_per_minute_quota(pm) is True
    assert ckpt.is_per_minute_quota(dy) is False        # daily → not per-minute
    assert ckpt.retry_delay_seconds(pm) == 53
    assert ckpt.retry_delay_seconds(dy) == 30
    # structured exc.details path: 'PerMinute' only in details, not in str(e)
    e = _Fake429("429 RESOURCE_EXHAUSTED")
    e.details = {"error": {"details": [{"violations": [{"quotaId": "FooPerMinuteBar"}]}]}}
    assert ckpt.is_per_minute_quota(e) is True


# ── per-minute 429 → silent retry, no checkpoint, no QuotaInterrupted ─────────

def test_per_minute_429_retries_silently(slot, coord, monkeypatch):
    waits = _no_sleep(monkeypatch)
    calls = {"n": 0}

    async def flaky_analyze(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _per_minute_exc()
        return DRAFT_TEXT

    async def ok_ground(draft, pkg):
        return draft, []

    _patch_stages(monkeypatch, flaky_analyze, ok_ground)

    async def fake_coverage(q, a):
        return a, True
    monkeypatch.setattr(coord, "_coverage_check", fake_coverage)

    result = asyncio.run(coord.route("how is my bench progressing?"))

    assert calls["n"] == 2                       # 429 once, retried once, succeeded
    assert result["answer"] == DRAFT_TEXT
    assert not slot.exists()                     # NO checkpoint written
    assert waits == [55]                         # retryDelay 53 + buffer 2, under cap 70


# ── per-minute 429 beyond max retries → falls through to daily checkpoint ─────

def test_per_minute_429_exhausts_to_daily_path(slot, coord, monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    async def always_per_minute(*a, **k):
        calls["n"] += 1
        raise _per_minute_exc()

    _patch_stages(monkeypatch, analyze=always_per_minute)

    with pytest.raises(ckpt.QuotaInterrupted):
        asyncio.run(coord.route("how is my bench progressing?"))

    # initial + PER_MINUTE_MAX_RETRIES attempts, then daily fallback
    assert calls["n"] == 1 + coordinator_mod.PER_MINUTE_MAX_RETRIES
    cp = _read_slot(slot)
    assert cp["completed_stage"] == "draft" or cp["completed_stage"] == "classify"
    # draft stage failed before producing text → no verbatim draft stored
    assert cp["draft"] is None


# ── classify-stage DAILY 429 → gap closed (checkpoint classify, draft null) ───

def test_classify_daily_429_checkpoints_and_resumes(slot, coord, monkeypatch):
    _no_sleep(monkeypatch)

    async def failing_classify(q):
        raise _daily_exc()
    monkeypatch.setattr(coord, "_classify", failing_classify)

    with pytest.raises(ckpt.QuotaInterrupted) as ei:
        asyncio.run(coord.route("how is my bench progressing?"))
    assert "your question is saved" in ei.value.user_message.lower()

    cp = _read_slot(slot)
    assert cp["completed_stage"] == "classify"
    assert cp["draft"] is None
    assert cp["params"] is None                  # true classify gap (route unknown)

    # Resume re-runs classify (cheap) then the rest of the pipeline.
    calls = {"classify": 0, "analyze": 0}

    async def ok_classify(q):
        calls["classify"] += 1
        return dict(ANALYTICAL_PARAMS)
    monkeypatch.setattr(coord, "_classify", ok_classify)

    async def ok_analyze(*a, **k):
        calls["analyze"] += 1
        return DRAFT_TEXT

    async def ok_ground(draft, pkg):
        return draft, []
    _patch_stages(monkeypatch, ok_analyze, ok_ground)

    async def fake_coverage(q, a):
        return a, True
    monkeypatch.setattr(coord, "_coverage_check", fake_coverage)

    result = asyncio.run(coord.route("continue"))
    assert calls["classify"] == 1 and calls["analyze"] == 1
    assert result["answer"] == DRAFT_TEXT
    assert not slot.exists()


# ── /resume path: nothing-to-resume guard (route('continue') with no slot) ────

def test_resume_with_no_slot_returns_notice(slot, coord, monkeypatch):
    spy = {"classify": 0, "analytical": 0, "operational": 0}

    async def spy_classify(q):
        spy["classify"] += 1
        return dict(ANALYTICAL_PARAMS)
    monkeypatch.setattr(coord, "_classify", spy_classify)

    async def spy_an(q, p, resume=None):
        spy["analytical"] += 1
        return "X", []
    monkeypatch.setattr(coord, "_run_analytical", spy_an)

    async def spy_op(q, **kw):           # **kw: /log boundary flags (ignored here)
        spy["operational"] += 1
        return "Y"
    monkeypatch.setattr(coord, "_run_operational", spy_op)

    result = asyncio.run(coord.route("continue"))      # what POST /resume sends

    assert result["route"] == "none"
    assert "no saved question" in result["answer"].lower()
    assert spy == {"classify": 0, "analytical": 0, "operational": 0}  # no LLM/work
