"""
Can the user be told a write happened when it did not?

WHY THIS EXISTS. A live write check ran the real flow against the real database:
a workout was staged, confirmed, and written; "delete that set" then produced a
`discard_staged_writes` call and the reply "has been removed from your staged
session, and the entire staged workout has been discarded" — while the row sat
untouched in training_log. Nothing was lost. The data stayed while the user was
told it was gone, which is worse in the way that matters: you stop looking.

Three layers failed in sequence, and this file pins all three.

  Row 1  the agent is never told its staged batch was committed, so it still
         believes the set is pending — reaching for discard is correct reasoning
         from a false premise, not tool confusion.
  Row 2  discard_staged_writes returns {"discarded": true} whether it dropped a
         batch or found an empty slot, so the agent cannot tell the difference.
  Row 3  the success-claim gate catches false "saved" claims but almost no other
         write verb.

THIS FILE IS A SAFETY NET, NOT A SPEC. It was written to pass against UNMODIFIED
code, so it cannot be a restatement of the behavior about to be introduced. Rows
the arc deliberately changes sit in PRE-CHANGE blocks carrying their current
value. A row that moves without such a label is a regression.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")

from src import coordinator as coordinator_mod                      # noqa: E402
from src.coordinator import (                                       # noqa: E402
    MSG_NO_WRITE_OCCURRED,
    MSG_STAGED_NOT_SAVED,
    Coordinator,
    _WRITE_COMPLETION_CLAIM_RE,
    _WRITE_SUCCESS_CLAIM_RE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Driving the REAL gate — a fake agent, the genuine _run_operational
# ══════════════════════════════════════════════════════════════════════════════

def _coord(monkeypatch, answer, *, db_write_effect=False,
           staging_reached_confirm=False, staged_this_turn=False,
           write_attempted=False):
    """Coordinator whose agent returns a chosen answer + structural facts.

    The gate under test lives inside _run_operational, so it is exercised for
    real rather than re-implemented here.
    """
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())

    class _FakeAgent:
        async def answer(self, question, resume_messages=None):
            return {
                "answer": answer,
                "db_write_effect": db_write_effect,
                "staging_reached_confirm": staging_reached_confirm,
                "staged_this_turn": staged_this_turn,
                "write_attempted": write_attempted,
            }

    c = Coordinator(agent_session=_FakeAgent())
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _run(coord, q="log my bench", **kw):
    return asyncio.run(coord._run_operational(q, **kw))


# ── The gate works where it was built to work ─────────────────────────────────

def test_unbacked_logged_claim_is_replaced(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.")
    assert _run(c, log_boundary=True) == MSG_NO_WRITE_OCCURRED


def test_staged_only_claim_becomes_the_truthful_staged_message(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.", staged_this_turn=True)
    assert _run(c, log_boundary=True).startswith(MSG_STAGED_NOT_SAVED[:40])


def test_a_backed_claim_is_left_alone(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.", db_write_effect=True)
    assert _run(c, log_boundary=True) == "Your workout has been logged."


def test_reaching_the_execute_gate_also_backs_the_claim(monkeypatch):
    c = _coord(monkeypatch, "✅ Your goal has been saved to your database.",
               staging_reached_confirm=True)
    assert "saved to your database" in _run(c)


def test_a_question_is_never_rewritten(monkeypatch):
    # No claim is being made — the flags alone must not trigger the gate.
    q = "Which date should I log that under?"
    c = _coord(monkeypatch, q)
    assert _run(c, log_boundary=True) == q


# ── The property the Row-3 fix must not break ─────────────────────────────────

RESEARCH_PROSE = [
    "The study removed participants who missed sessions, then re-ran the analysis.",
    "Two trials were deleted from the meta-analysis for poor blinding.",
    "The authors updated their conclusions in a later erratum.",
]


@pytest.mark.parametrize("prose", RESEARCH_PROSE)
def test_research_prose_is_never_rewritten(prose):
    """THE FALSE POSITIVE TO AVOID. _run_operational also serves research
    answers. Broadening the claim regexes without a structural scope would turn
    'the study removed participants' into a database warning. Asserted at the
    DETECTOR level so it holds regardless of how scoping is later wired."""
    assert not _WRITE_SUCCESS_CLAIM_RE.search(prose)


# ══════════════════════════════════════════════════════════════════════════════
# Row 3 — what the gate currently detects, verb by verb
#
# The loose detector only runs on logging-flow turns (log_boundary /
# fallback_write). On goal and set-edit turns ONLY the narrow one applies, which
# is why the goal/set rows below are effectively unguarded today.
# ══════════════════════════════════════════════════════════════════════════════

def _detected(claim: str, *, log_flow: bool) -> bool:
    return bool(_WRITE_SUCCESS_CLAIM_RE.search(claim)) or (
        log_flow and bool(_WRITE_COMPLETION_CLAIM_RE.search(claim)))


_CAUGHT_TODAY_ON_LOG_FLOWS = [
    "Your workout has been logged.",
    "I've logged your workout.",
    "3 sets were added to your database.",
    "Your bodyweight has been recorded.",
    "Successfully deleted the set.",
]


@pytest.mark.parametrize("claim", _CAUGHT_TODAY_ON_LOG_FLOWS)
def test_logging_claims_are_detected(claim):
    assert _detected(claim, log_flow=True)


# ── CHANGED by Row 3 ──────────────────────────────────────────────────────────
# These completion claims used to reach the user unchecked: the loose detector
# never ran on goal or set-edit turns, so only the narrow regex applied and
# caught one of thirteen. Both the verb list and the SCOPE are widened — the
# scope by a structural fact (a write tool was actually called this turn), which
# is what keeps research answers out.
#
# It matters exactly when the write did NOT happen — cancelled at the
# confirmation prompt, failed, or never attempted — and the agent says it did.

_WRITE_CLAIMS = [
    ("goal set",    "Your goal is now set."),
    ("goal set",    "I've created that goal for you."),
    ("goal update", "Your goal has been updated."),
    ("goal update", "I've updated your goal."),
    ("goal delete", "Your goal has been deleted."),
    ("goal delete", "I've removed that goal."),
    ("set update",  "The set has been updated to 12 reps."),
    ("set update",  "I've corrected that set."),
    ("set update",  "That set has been fixed."),
    ("set delete",  "The set has been deleted from your database."),
    ("set delete",  "I've deleted that set."),
    ("set delete",  "That set has been removed."),
]


@pytest.mark.parametrize("flow,claim", _WRITE_CLAIMS)
def test_row3_write_claims_are_detected_on_a_write_turn(flow, claim):
    assert _detected(claim, log_flow=True)


@pytest.mark.parametrize("flow,claim", _WRITE_CLAIMS)
def test_row3_the_same_claims_stay_inert_without_a_write(flow, claim):
    """The other direction, and the reason the scope is structural: identical
    words on a turn that called no write tool must not be rewritten. Only the
    narrow regex applies there, exactly as before."""
    assert _detected(claim, log_flow=False) == bool(
        _WRITE_SUCCESS_CLAIM_RE.search(claim))


def test_row3_the_observed_false_claim_is_now_detected():
    # The exact sentence the live check produced while the row remained.
    observed = ("The Flat Barbell Bench Press set of 101 lbs x 7 reps on "
                "2026-08-08 has been removed from your staged session, and the "
                "entire staged workout has been discarded.")
    assert _detected(observed, log_flow=True)


def test_row3_end_to_end_a_delete_claim_without_a_write_is_replaced(monkeypatch):
    """Through the real gate: a delete was attempted, nothing was written, and
    the agent claims removal. The user must not be told the row is gone."""
    c = _coord(monkeypatch, "I've deleted that set.", write_attempted=True)
    assert _run(c, "delete my bench set") == MSG_NO_WRITE_OCCURRED


def test_row3_a_backed_delete_claim_survives(monkeypatch):
    c = _coord(monkeypatch, "I've deleted that set.",
               write_attempted=True, db_write_effect=True)
    assert _run(c, "delete my bench set") == "I've deleted that set."


def test_row3_research_answer_is_untouched_without_a_write(monkeypatch):
    """THE FALSE POSITIVE, end to end. Same code path, research answer, no write
    tool called — the structural fact keeps the prose intact."""
    prose = "The study removed participants who missed sessions."
    c = _coord(monkeypatch, prose)
    assert _run(c, "what does science say about adherence?") == prose


def test_row3_no_write_message_serves_every_write_kind():
    """One message for all of them. Splitting it by flow was attempted and
    abandoned: the gate fires hardest when the agent called NO tool, and then
    nothing distinguishes a logging turn from a goal turn — the canonical live
    failure is a bare "Yes thats correct", which sets neither flow flag."""
    assert "saved or changed" in MSG_NO_WRITE_OCCURRED       # covers both
    assert "/log" in MSG_NO_WRITE_OCCURRED                   # tip survives
    assert "your logging request" not in MSG_NO_WRITE_OCCURRED


def test_row3_write_attempted_is_set_on_the_call_not_the_result():
    """A write that errors or returns garbage is still a write turn — otherwise
    the worst case (tool blew up, agent claims success anyway) is the one case
    the gate skips. Asserted by ORDER: the flag is set before the result parse,
    so a parse failure cannot skip it."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    set_at = src.index("self._turn_write_attempted = True")
    parse_at = src.index("parsed = json.loads(result)")
    assert set_at < parse_at, "write_attempted is set inside/after the parse"


# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — the discard tool's report
# ══════════════════════════════════════════════════════════════════════════════

def _server_src() -> str:
    return (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")


def test_every_host_discard_is_fire_and_forget():
    """Why the payload is safe to change: no caller reads it. Pinned so a future
    caller that starts parsing it is caught here rather than in production."""
    for host in ("cli.py", "server.py"):
        for line in (_ROOT / host).read_text(encoding="utf-8").splitlines():
            if "discard_staged_writes" in line and "call_tool" in line:
                assert "=" not in line.split("call_tool")[0].split("await")[-1], \
                    f"{host}: discard return is being captured: {line.strip()}"


# ── CHANGED by Row 2 ──────────────────────────────────────────────────────────
# The tool USED to answer identically whether it dropped a batch or found an
# empty slot, so an agent calling it on nothing was told "discarded" and passed
# that on. It now reports which happened — and in the empty case names the tools
# that DO delete saved data, so the tool result corrects the mistake instead of
# confirming it. Exercised for real, not asserted from source text.

def _discard_sync():
    import importlib
    m = importlib.import_module("mcp_servers.combined_server")
    return m


def test_row2_discard_reports_that_it_dropped_something():
    m = _discard_sync()
    m._staged_writes.clear()
    m._staged_writes.setdefault("workout", []).append({"dummy": True})
    out = json.loads(m._discard_staged_writes_sync())
    assert out["discarded"] is True and out["had_pending"] is True
    assert not m._staged_writes                      # really cleared


def test_row2_discard_reports_an_empty_slot_honestly():
    m = _discard_sync()
    m._staged_writes.clear()
    out = json.loads(m._discard_staged_writes_sync())
    assert out["discarded"] is False and out["had_pending"] is False


def test_row2_empty_discard_points_at_the_tools_that_delete_saved_data():
    # The observed failure was the agent calling discard for a SAVED row and
    # reporting removal. The empty-slot result now says the opposite outright.
    m = _discard_sync()
    m._staged_writes.clear()
    msg = json.loads(m._discard_staged_writes_sync())["message"]
    assert "delete_workout_set" in msg and "delete_goal" in msg
    assert "cannot remove data already" in msg


def test_row2_tool_description_declares_the_new_contract():
    src = _server_src()
    assert "had_pending" in src
    i = src.index('name="discard_staged_writes"')
    desc = src[i:i + 900]
    assert "had_pending=false" in desc
    assert "delete_workout_set" in desc


# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — nobody tells the agent the write landed
# ══════════════════════════════════════════════════════════════════════════════

def test_agent_history_is_what_the_model_actually_sees():
    """The seam Row 1 must write into: _op_prepare builds the model's context
    from _conversation_history, so anything appended there reaches the model."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    prep = src[src.index("def _op_prepare"):]
    prep = prep[:prep.index("def _op_agent_step")]
    assert "_conversation_history" in prep


# ── CHANGED by Row 1 ──────────────────────────────────────────────────────────
# After the host committed a staged batch the agent was never informed, so its
# last known state stayed "staged, awaiting confirmation". Both hosts now tell
# it — the same seam applied to both interfaces, because a fix landing on one
# and not the other is how these gaps survive.

def _agent_session():
    from src.agent import AgentSession
    return AgentSession("data/unused.fitnotes")


def test_row1_note_reaches_the_model_context():
    """It must land in _conversation_history — that is what _op_prepare feeds
    the model. Anywhere else (e.g. chat_history) the agent never sees it."""
    from src.agent import HOST_WRITE_NOTE_MARK
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 3 sets across 1 exercise(s).")
    flat = [m for ex in s._conversation_history for m in ex]
    assert len(flat) == 1
    assert flat[0]["role"] == "assistant"
    assert HOST_WRITE_NOTE_MARK in flat[0]["content"]
    assert "3 sets across 1 exercise(s)" in flat[0]["content"]


def test_row1_empty_outcome_records_nothing():
    s = _agent_session()
    s.note_host_write("")
    s.note_host_write(None)
    assert s._conversation_history == []


def test_row1_note_is_skipped_by_memory_extraction():
    """The note belongs in the model's context but must never be mined as a
    fact about the user — "3 sets across 1 exercise" is plumbing, not a
    preference."""
    from src.agent import HOST_WRITE_NOTE_MARK
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    body = src[src.index("async def _auto_extract_memories"):]
    body = body[:body.index("conversation_text +=") + 200]
    assert "HOST_WRITE_NOTE_MARK" in body
    assert "continue" in body


@pytest.mark.parametrize("host", ["cli.py", "server.py"])
def test_row1_both_hosts_notify_after_a_successful_execute(host):
    text = (_ROOT / host).read_text(encoding="utf-8")
    assert "execute_staged_workout" in text
    assert "note_host_write" in text, f"{host} executes but never tells the agent"


def test_row1_every_successful_execute_branch_notifies():
    """Counted, not spot-checked: each host branch that flips _staged_active on
    a successful execute must also notify. One unpatched branch reproduces the
    whole bug."""
    for host, expected in (("cli.py", 2), ("server.py", 1)):
        text = (_ROOT / host).read_text(encoding="utf-8")
        assert text.count("note_host_write") == expected, host
        assert text.count("_staged_active = False") >= expected, host


def test_row1_record_external_exchange_still_has_one_caller():
    # The new seam is deliberately separate: record_external_exchange feeds
    # memory extraction, which host notes must NOT enter.
    call = re.compile(r"\.record_external_exchange\s*\(")
    hits = []
    for p in list((_ROOT / "src").rglob("*.py")) + [_ROOT / "cli.py", _ROOT / "server.py"]:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if call.search(line):
                hits.append(f"{p.name}:{i}")
    assert len(hits) == 1, f"expected the analytical caller only, got {hits}"
    assert "coordinator" in hits[0]
