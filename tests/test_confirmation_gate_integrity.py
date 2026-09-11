"""
Confirmation-gate integrity + Issue 2 hardening — the ghost-panel fix and the
write-success claim-gate holes, one false-signal family.

Live-proven bug (two-run consistent): server._confirmation_handler arms
pending_confirmation at log_workout CALL time — a pre-call hook that cannot
see the outcome — so a refused call (unit guard / needs_clarification) left an
EMPTY slot yet /chat returned confirmation_required with raw-args fallback,
and the agent's ask was swallowed. Fix: the slot is the single source of
panel truth — PROVEN empty ⇒ ghost, suppress the panel, ship the answer;
UNKNOWN (read failed) ⇒ fail toward the panel. Same seam on the CLI
(_finalize_staged_workout must not verify/execute an empty slot).

Claim gate: hole B (loose phrasings, scoped to logging-flow turns via the
structural log_boundary/fallback_write facts) and hole A (staged-only turns
get the truthful staged-not-saved rewrite, never an untouched "saved!").

No live server, no Gemini — monkeypatched server module + stubbed session,
mirroring tests/test_write_verify.py's harness.
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src import coordinator as coordinator_mod            # noqa: E402
from src.coordinator import (                              # noqa: E402
    Coordinator,
    MSG_NO_WRITE_OCCURRED,
    MSG_STAGED_NOT_SAVED,
)

_ASK = ("Barbell Row is logged in lbs, not kg. (60 kg ≈ 132.3 lbs.) "
        "What's the weight in lbs?")
_SLOT_EMPTY = json.dumps({"staged_workouts": []})
_SLOT_REAL = json.dumps({"staged_workouts": [
    {"exercise_id": 1, "exercise_name": "Barbell Row", "date": "2026-07-09",
     "sets": [{"metric_weight": 60.0109, "reps": 8, "unit": 0}]}
]})
_PREVIEW = "Barbell Row — 2026-07-09\n  132.3 lbs × 8 reps"


# ══════════════════════════════════════════════════════════════════════════════
# (a) Server: panel truth = slot truth
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod, "session", None)
    # Hermetic checkpoints: the PASS branch saves and the FAIL branch clears —
    # neither may touch the real slot file from a test.
    monkeypatch.setattr(server_mod._ckpt, "save_checkpoint",
                        lambda **kw: None)
    monkeypatch.setattr(server_mod._ckpt, "clear_staged_checkpoint",
                        lambda: None)
    server_mod._state.update({"pending_confirmation": False,
                              "allow_execute": False, "staging_preview": "",
                              "confirmation_preview": "",
                              "pending_execute_kind": None,
                              "decomposed_answer": ""})
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


def _drive(srv, monkeypatch, *, answer_text, slot_response, formatter_response,
           verdict=None):
    """One /chat turn with pending armed the way _confirmation_handler's
    log_workout branch arms it (call-time). slot_response / formatter_response
    are either a raw string to return or an Exception to raise. Returns
    (body, calls, seen)."""
    calls: list = []
    seen: dict = {}

    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": answer_text, "route": "operational",
                "flagged_claims": [], "error": None, "log_boundary": True,
                "log_flow_turns": ["Log Barbell Row 60kg x 8", "Today"]}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            if isinstance(slot_response, Exception):
                raise slot_response
            return slot_response
        if name == "format_staged_workout_for_confirmation":
            if isinstance(formatter_response, Exception):
                raise formatter_response
            return formatter_response
        if name == "execute_staged_workout":
            return json.dumps({"success": True, "message": "written"})
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        seen["flow"] = flow
        seen["slot"] = slot_json
        seen["preview"] = preview
        return verdict or {"verdict": "PASS", "reason": ""}

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging))
    monkeypatch.setattr(srv, "session", SimpleNamespace(call_tool=call_tool, note_host_write=lambda *a, **k: None))
    body = _body(asyncio.run(srv._process_turn("Today")))
    return body, calls, seen


def test_ghost_empty_slot_suppresses_panel_and_ships_the_ask(srv, monkeypatch):
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text=_ASK,
        slot_response=_SLOT_EMPTY,
        formatter_response=json.dumps({"error": "No staged workout found."}))

    # The refusal ask reaches the user — never a panel for a batch that
    # doesn't exist, and never the raw-args fallback.
    assert body["type"] == "answer"
    assert body["text"] == _ASK
    assert "preview" not in body and "preview_source" not in body
    # A stray /confirm must find nothing to execute.
    assert srv._state["pending_execute_kind"] is None
    # Negative direction: the whole panel machinery stayed cold — no
    # formatter, no verify, no verdict record, no execute.
    names = [n for n, _ in calls]
    assert "format_staged_workout_for_confirmation" not in names
    assert "record_workout_verify" not in names
    assert "execute_staged_workout" not in names
    assert seen == {}                       # verify LLM never consulted


def test_real_batch_still_panels_with_slot_preview(srv, monkeypatch):
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text="Staged.",
        slot_response=_SLOT_REAL,
        formatter_response=json.dumps({"preview": _PREVIEW}),
        verdict={"verdict": "PASS", "reason": ""})

    assert body["type"] == "confirmation_required"
    assert body["preview"] == _PREVIEW
    assert body["preview_source"] == "slot"
    # The slot-first reorder must keep feeding verify the raw slot + preview.
    assert seen["slot"] == _SLOT_REAL
    assert seen["preview"] == _PREVIEW
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert records == [{"verdict": "PASS", "reason": ""}]


def test_formatter_failure_on_real_batch_keeps_args_fallback(srv, monkeypatch):
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text="Staged.",
        slot_response=_SLOT_REAL,
        formatter_response=RuntimeError("formatter down"))

    # Defensive path survives: a REAL batch never loses its panel.
    assert body["type"] == "confirmation_required"
    assert body["preview"] == "ARGS-BLOB"
    assert body["preview_source"] == "args_fallback"
    # No slot-rendered preview ⇒ verify skipped (ERROR, fail-open), recorded.
    assert seen == {}
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert len(records) == 1 and records[0]["verdict"] == "ERROR"


def test_slot_read_failure_fails_toward_panel(srv, monkeypatch):
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text="Staged.",
        slot_response=RuntimeError("mcp hiccup"),
        formatter_response=json.dumps({"preview": _PREVIEW}))

    # UNKNOWN emptiness must never suppress: the panel still gates the write,
    # and verify is skipped (never diffed against an empty string).
    assert body["type"] == "confirmation_required"
    assert seen == {}
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert len(records) == 1 and records[0]["verdict"] == "ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# (b) CLI: same seam — _finalize_staged_workout on an empty slot
# ══════════════════════════════════════════════════════════════════════════════

def _cli_session(calls, slot_response):
    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            if isinstance(slot_response, Exception):
                raise slot_response
            return slot_response
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "execute_staged_workout":
            return json.dumps({"success": True, "message": "written"})
        return json.dumps({"ok": True})
    return SimpleNamespace(call_tool=call_tool, _staged_active=True, note_host_write=lambda *a, **k: None)


def _cli_coordinator(verdict):
    async def verify_log_staging(flow, slot_json, preview):
        return verdict
    return SimpleNamespace(verify_log_staging=verify_log_staging)


def test_cli_finalize_empty_slot_is_a_silent_noop():
    import cli
    calls: list = []
    line = asyncio.run(cli._finalize_staged_workout(
        _cli_session(calls, _SLOT_EMPTY),
        _cli_coordinator({"verdict": "PASS", "reason": ""}),
        {"log_flow_turns": ["log bench 100x5"]}, "log bench 100x5"))

    assert line == ""                       # nothing extra printed
    names = [n for n, _ in calls]
    assert "execute_staged_workout" not in names       # never a no-op execute
    assert "record_workout_verify" not in names
    assert "format_staged_workout_for_confirmation" not in names


def test_cli_finalize_real_batch_still_executes():
    import cli
    calls: list = []
    line = asyncio.run(cli._finalize_staged_workout(
        _cli_session(calls, _SLOT_REAL),
        _cli_coordinator({"verdict": "PASS", "reason": ""}),
        {"log_flow_turns": ["log bench 100x5"]}, "log bench 100x5"))

    assert line.startswith("✅")
    assert "execute_staged_workout" in [n for n, _ in calls]


# ══════════════════════════════════════════════════════════════════════════════
# (c) Claim gate — hole B (loose phrasings, flow-scoped) + hole A (staged tier)
# ══════════════════════════════════════════════════════════════════════════════

class FakeAgent:
    """AgentSession stand-in (same shape as test_write_path_fixes.FakeAgent)."""

    def __init__(self, answer_text, staging_reached_confirm=False,
                 db_write_effect=False, staged_this_turn=False):
        self.answer_text = answer_text
        self.staging_reached_confirm = staging_reached_confirm
        self.db_write_effect = db_write_effect
        self.staged_this_turn = staged_this_turn

    async def answer(self, question):
        return {
            "question": question,
            "answer": self.answer_text,
            "tool_calls_made": 1,
            "error": None,
            "staging_reached_confirm": self.staging_reached_confirm,
            "db_write_effect": self.db_write_effect,
            "staged_this_turn": self.staged_this_turn,
        }


def _make_coord(monkeypatch, agent):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=agent)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


@pytest.mark.parametrize("claim", [
    "I've logged your workout for today. Nice session!",
    "You're all set!",
    "Your workout is saved.",
    "Done — I have added 3 sets of Barbell Row.",
])
def test_loose_claims_suppressed_on_log_flow_turns(monkeypatch, claim):
    # Hole B: on a logging-flow turn, an unqualified completion phrase with
    # all-False structural flags is a false claim — suppressed.
    agent = FakeAgent(answer_text=claim)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(
        coord._run_operational("log bench 100x5", log_boundary=True))

    assert answer == MSG_NO_WRITE_OCCURRED


def test_loose_phrases_ship_untouched_outside_log_flow(monkeypatch):
    # The looser detector is scoped by the structural flow fact: ordinary
    # speech on a non-logging turn must never be replaced.
    text = "You're all set for tomorrow — I've saved that note to memory."
    agent = FakeAgent(answer_text=text)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("remember my split"))

    assert answer == text


def test_staged_turn_saved_claim_becomes_staged_not_saved(monkeypatch):
    # Hole A: staged-only (no execute attempt, no write) + a completed-write
    # claim ⇒ the truthful staged-state rewrite, never an untouched "saved!".
    agent = FakeAgent(
        answer_text="Your Barbell Row set was successfully logged to your database.",
        staged_this_turn=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("log bench 100x5"))

    assert answer == MSG_STAGED_NOT_SAVED


def test_staged_turn_honest_staging_language_untouched(monkeypatch):
    # The rewrite only fires on COMPLETED-write claims — honest staging
    # language ("staged", "confirm") carries no claim and ships as-is.
    text = "I've staged your workout — please confirm to save it."
    agent = FakeAgent(answer_text=text, staged_this_turn=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(
        coord._run_operational("log bench 100x5", log_boundary=True))

    assert answer == text


def test_backed_claim_ships_even_with_loose_phrasing(monkeypatch):
    # A real write this turn backs any phrasing — gate stays silent.
    agent = FakeAgent(answer_text="You're all set — workout logged!",
                      db_write_effect=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(
        coord._run_operational("log bench 100x5", log_boundary=True))

    assert answer == "You're all set — workout logged!"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 (#12): panel shows staging only; /confirm delivers the merged answer
# in the CHAT together with the write outcome
# ══════════════════════════════════════════════════════════════════════════════

# A decomposed mixed turn's full answer carries BOTH the analytical part and
# the write chunk's staging-time text; the write-excluded merge (#19) carries
# the analytical part only. The panel stash / post-confirm prepend must use the
# latter so a committed write is never re-described as "not saved yet".
_ANALYSIS = "### Is my squat progressing?\n\nYour squat is up 5%."
_STAGING = ("### Log bench 100 lbs for 5 reps\n\n"
            "⚠️ That isn't saved yet — it's staged and needs your confirmation.")
_MERGED = f"{_ANALYSIS}\n\n{_STAGING}"      # full answer (used on the no-panel path)
_NONWRITE = _ANALYSIS                        # write-excluded (stash + prepend)


def _decomposed_turn(srv, monkeypatch, *, execute_response=None):
    """Drive one decomposed mixed turn to the confirm panel. The stubs stay
    installed so the test can follow up with srv.confirm(). Returns the panel
    body and the call_tool log."""
    calls: list = []
    if execute_response is None:
        execute_response = {"success": True, "message": "written"}

    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": _MERGED, "decomposed_nonwrite_answer": _NONWRITE,
                "route": "analytical", "decomposed": True,
                "flagged_claims": [], "error": None, "log_boundary": False,
                "log_flow_turns": ["Log bench 100 lbs for 5 reps today"]}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            return _SLOT_REAL
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "execute_staged_workout":
            return json.dumps(execute_response)
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        return {"verdict": "PASS", "reason": ""}

    async def answer(message):
        return {"answer": f"AGENT({message})", "error": None}

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging,
        _pending_log_carry=True))          # #13: /confirm must clear this
    monkeypatch.setattr(srv, "session", SimpleNamespace(
        call_tool=call_tool, chat_history=[], answer=answer, note_host_write=lambda *a, **k: None))
    body = _body(asyncio.run(srv._process_turn(
        "is my squat progressing and log bench 100x5")))
    return body, calls


def test_decomposed_panel_shows_staging_only_and_stashes_answer(srv, monkeypatch):
    body, _ = _decomposed_turn(srv, monkeypatch)

    assert body["type"] == "confirmation_required"
    assert body["preview"] == _PREVIEW                  # staged batch ONLY
    assert _NONWRITE not in body["preview"]
    assert "text" not in body                           # no dead payload field
    # #19: the stash holds the WRITE-EXCLUDED merge, never the staging text.
    assert srv._state["decomposed_answer"] == _NONWRITE
    assert "isn't saved yet" not in srv._state["decomposed_answer"]


def test_confirm_delivers_merged_answer_before_write_outcome(srv, monkeypatch):
    _decomposed_turn(srv, monkeypatch)

    body = _body(asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=True))))

    assert body["type"] == "answer"
    assert body["text"].startswith(_NONWRITE)           # analytical half first
    assert "✅" in body["text"]
    assert body["text"].index(_NONWRITE) < body["text"].index("✅")
    # #19 core: the committed write is described ONLY by the ✅ line — the stale
    # staging text never reaches the post-confirm reply.
    assert "isn't saved yet" not in body["text"]
    assert "staged" not in body["text"].lower()
    assert srv._state["decomposed_answer"] == ""        # consumed, never re-shipped
    assert srv.coordinator._pending_log_carry is False  # #13 defensive clear


def test_cancel_delivers_merged_answer_with_cancel_text(srv, monkeypatch):
    _decomposed_turn(srv, monkeypatch)

    body = _body(asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=False))))

    assert body["type"] == "answer"
    assert body["text"].startswith(_NONWRITE)
    assert "AGENT(Cancel that)" in body["text"]
    assert "isn't saved yet" not in body["text"]        # stale staging text gone
    assert srv._state["decomposed_answer"] == ""
    assert srv.coordinator._pending_log_carry is False  # #13 defensive clear


def test_confirm_execute_failure_still_delivers_merged_answer(srv, monkeypatch):
    _decomposed_turn(srv, monkeypatch,
                     execute_response={"success": False, "message": "db locked"})

    body = _body(asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=True))))

    assert body["type"] == "error"
    assert body["text"].startswith(_NONWRITE)           # not lost on a failed write
    assert "❌" in body["text"]
    assert srv._state["decomposed_answer"] == ""


def test_non_decomposed_panel_payload_unchanged(srv, monkeypatch):
    """A plain single-write turn's panel is byte-identical to before Stage 3
    (no prepend, no text field, nothing stashed)."""
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text="Staged.",
        slot_response=_SLOT_REAL,
        formatter_response=json.dumps({"preview": _PREVIEW}),
        verdict={"verdict": "PASS", "reason": ""})

    assert body["type"] == "confirmation_required"
    assert body["preview"] == _PREVIEW                  # no prepend
    assert "text" not in body
    assert srv._state["decomposed_answer"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# (f) EVERY execute tool is actually stopped by the gate
#
# LIVE-CAUGHT in Phase 3 of the write-safety check. A note on a set staged with
# requires_confirmation:true and then wrote to the database with NO panel:
# execute_staged_set_comment was absent from server.py's own EXECUTE_TOOLS, so
# _confirmation_handler fell through to its STAGING branch and returned True.
#
# Everything above this block tests the panel for log_workout and never asked
# whether the other executes were gated at all. Parametrising is what closes the
# class — the list test next door proves the string is present, this proves the
# write is stopped.
# ══════════════════════════════════════════════════════════════════════════════

from src.agent import EXECUTE_TOOLS                                # noqa: E402


@pytest.mark.parametrize("tool", sorted(EXECUTE_TOOLS))
def test_every_execute_tool_is_blocked_until_confirmed(srv, tool):
    from src.agent import CONFIRM_DEFERRED
    approved = asyncio.run(srv._confirmation_handler(tool, {}))
    # CHANGED from `is False`: still blocked (the safety property), but the
    # handler now says WHY — "the user is looking at this", not "the user said
    # no". False made the agent record the write as cancelled, and nothing
    # corrected that once the user confirmed. What must not move is that the
    # write does not run.
    assert approved is not True, f"{tool} executed without confirmation"
    assert approved == CONFIRM_DEFERRED, f"{tool} reported a refusal, not a deferral"
    assert srv._state["pending_confirmation"] is True, f"{tool} armed no panel"


@pytest.mark.parametrize("tool", sorted(EXECUTE_TOOLS))
def test_every_execute_tool_proceeds_once_confirmed(srv, tool):
    """The other direction — /confirm must still work for all of them."""
    srv._state["allow_execute"] = True
    assert asyncio.run(srv._confirmation_handler(tool, {})) is True


def test_a_staging_call_is_not_blocked(srv):
    """NEGATIVE, and the reason the fall-through branch exists: staging must run
    so the MCP server can store the payload the panel then previews. Only the
    execute is gated."""
    args = {"exercise_name": "Decline Barbell Bench Press", "date": "2026-07-22",
            "comment": "elbows flared on the last two"}
    assert asyncio.run(srv._confirmation_handler("set_set_comment", args)) is True
    assert srv._state["pending_confirmation"] is False
    assert "elbows flared" in srv._state["staging_preview"]


# ══════════════════════════════════════════════════════════════════════════════
# (g) A DEFERRAL IS NOT A CANCELLATION
#
# LIVE-CAUGHT in Phase 5. After a goal was created AND updated, the agent said
# "the previous actions were cancelled and no changes were made."
#
# On the web the panel is raised by the handler returning False — which the
# agent read as the user REFUSING, so it recorded "The write action was
# cancelled. Do not retry it." in its history. The user then confirms, the write
# runs, and nothing corrects that record. Workouts escape it because
# note_host_write repairs them afterwards; agent-driven executes (goal, set
# edit, comment) had no such repair, so the lie stood for the rest of the
# session.
#
# The failure direction is the dangerous one: "cancelled, nothing saved" invites
# the user to log the same thing twice.
# ══════════════════════════════════════════════════════════════════════════════

def _exec_one_tool(session, tool_name, handler):
    """Drive the real _op_exec_tools node through one confirmation-gated call."""
    session.confirmation_handler = handler
    called = []

    async def _call_tool(name, args):
        called.append(name)
        return json.dumps({"success": True})
    session.call_tool = _call_tool

    state = {
        "messages": [{"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": tool_name, "arguments": "{}"}}]}],
        "execute_attempted": False,
        "tool_calls_made": 0,
        "iteration": 0,
    }
    out = asyncio.run(session._op_exec_tools(state, SimpleNamespace(gemini_contents=[])))
    return state, out, called


def _session():
    from src.agent import AgentSession
    s = AgentSession.__new__(AgentSession)
    s._staged_active = False
    s._turn_write_effect = False
    s._turn_staged = False
    s._turn_write_attempted = False
    s._turn_write_block_reason = None
    s._conversation_history = []
    s.chat_history = []
    return s


async def _defer(tool_name, arguments):
    from src.agent import CONFIRM_DEFERRED
    return CONFIRM_DEFERRED


async def _decline(tool_name, arguments):
    return False


def test_a_deferral_still_blocks_the_write():
    """THE SAFETY PROPERTY. Everything else here is about wording; this is the
    gate itself, and it must not move."""
    s = _session()
    _, _, called = _exec_one_tool(s, "execute_staged_goal", _defer)
    assert called == [], "a deferred write actually ran"


def test_a_deferral_is_not_reported_as_cancelled():
    s = _session()
    state, out, _ = _exec_one_tool(s, "execute_staged_goal", _defer)
    tool_msg = [m for m in out["messages"] if m.get("role") == "tool"][-1]
    payload = json.loads(tool_msg["content"])
    # The FLAG is what the agent keys on; the prose may well contain the word
    # "cancelled" while denying it ("Nothing is cancelled"), so assert the
    # structure, not a substring.
    assert payload.get("deferred") is True
    assert "cancelled" not in payload
    assert "nothing is cancelled" in payload["message"].lower()
    assert out["write_deferred"] is True
    assert out["write_cancelled"] is True          # still ends the turn


def test_a_real_decline_still_says_cancelled():
    """The other direction — the CLI blocks on input() and genuinely knows the
    user said no, so its False must keep meaning exactly that."""
    s = _session()
    state, out, called = _exec_one_tool(s, "execute_staged_goal", _decline)
    assert called == []
    tool_msg = [m for m in out["messages"] if m.get("role") == "tool"][-1]
    payload = json.loads(tool_msg["content"])
    assert payload.get("cancelled") is True
    assert out["write_deferred"] is False


def _cancel_note(deferred):
    s = _session()
    state = {"question": "set a goal", "messages": [], "new_exchange_start": 0,
             "tool_calls_made": 1, "execute_attempted": False,
             "last_text": "", "write_deferred": deferred}
    s._op_finalize_cancelled(state)
    return [m for m in state["messages"] if m.get("role") == "user"][-1]["content"]


def test_the_deferred_history_note_denies_cancellation():
    """The note is the agent's LASTING record — this exact text is what made it
    tell the user a confirmed goal had been cancelled."""
    note = _cancel_note(True).lower()
    assert "not cancelled" in note
    assert "the write action was cancelled" not in note


def test_the_declined_history_note_is_unchanged():
    """The other direction: a real decline must still read exactly as before."""
    assert _cancel_note(False) == "The write action was cancelled. Do not retry it."


def test_a_deferral_never_tells_the_agent_not_to_retry():
    """"Do not retry it" is right for a decline and wrong for a deferral — the
    host retries it the moment the user confirms."""
    s = _session()
    state = {"question": "set a goal", "messages": [], "new_exchange_start": 0,
             "tool_calls_made": 1, "execute_attempted": False,
             "last_text": "", "write_deferred": True}
    s._op_finalize_cancelled(state)
    note = [m for m in state["messages"] if m.get("role") == "user"][-1]["content"]
    assert "nothing has been lost" in note.lower()


# ── bodyweight: a direct write with no staged slot ────────────────────────────

def test_bodyweight_is_gated_at_its_execute(srv):
    """CHANGED: body weight is properly STAGED now, so the gate falls on its
    execute like every other write, and the panel renders from the slot rather
    than from a JSON dump of the arguments. It used to be the lone direct
    writer, gated at the call itself.

    What must not move is that it cannot reach the database unconfirmed.
    """
    from src.agent import CONFIRM_DEFERRED, EXECUTE_TOOLS
    assert "execute_staged_bodyweight" in EXECUTE_TOOLS
    assert "execute_staged_bodyweight_delete" in EXECUTE_TOOLS
    approved = asyncio.run(srv._confirmation_handler(
        "execute_staged_bodyweight", {}))
    assert approved == CONFIRM_DEFERRED
    assert srv._state["pending_confirmation"] is True


def test_staging_a_bodyweight_is_not_blocked(srv):
    """Staging must run so the slot exists for the panel to preview."""
    assert asyncio.run(srv._confirmation_handler(
        "log_bodyweight", {"body_weight": 180, "unit": "lbs"})) is True


def test_bodyweight_proceeds_once_confirmed(srv):
    srv._state["allow_execute"] = True
    assert asyncio.run(srv._confirmation_handler(
        "execute_staged_bodyweight", {})) is True


# -- Row B: the OTHER stores, which nothing guarded ---------------------------

@pytest.mark.parametrize("tool,args", [
    ("delete_user_article", {"article_id": "abc123"}),
    ("delete_exercise_quirk", {"exercise_name": "Barbell Row"}),
    ("add_exercise_quirk", {"exercise_name": "Barbell Row", "quirk": "straps"}),
    ("update_exercise_quirk", {"exercise_name": "Barbell Row", "quirk": "no straps"}),
])
def test_the_other_stores_are_confirmed_too(srv, tool, args):
    """These write data/user_context.json and the Chroma index. None was in
    DB_WRITE_TOOLS, so the handler never saw them - delete_user_article
    irreversibly destroyed an uploaded document with no prompt at all."""
    from src.agent import CONFIRM_DEFERRED, CONFIRM_TOOLS
    assert tool in CONFIRM_TOOLS
    approved = asyncio.run(srv._confirmation_handler(tool, args))
    assert approved == CONFIRM_DEFERRED, f"{tool} wrote without confirmation"
    assert srv._state["pending_confirmation"] is True


def test_the_other_stores_are_not_counted_as_database_writes():
    """SCOPE. They must stay OUT of DB_WRITE_TOOLS: that set feeds
    db_write_effect, the fact the claim gate uses to decide whether a TRAINING
    DATABASE write happened. Counting them would let "I saved that" pass
    unchallenged on a turn that touched no training data."""
    from src.agent import DB_WRITE_TOOLS
    for tool in ("add_exercise_quirk", "update_exercise_quirk",
                 "delete_exercise_quirk", "delete_user_article"):
        assert tool not in DB_WRITE_TOOLS


def test_the_direct_write_panel_is_readable_not_a_json_blob(srv):
    """The last place a raw json.dumps was put in front of the user."""
    asyncio.run(srv._confirmation_handler(
        "delete_user_article", {"article_id": "abc123"}))
    preview = srv._state["confirmation_preview"]
    assert "{" not in preview and '":' not in preview
    assert "permanent" in preview.lower()
    assert "abc123" in preview


def test_a_read_only_tool_is_still_not_gated(srv):
    """The other direction - gating everything would make the panel meaningless."""
    assert asyncio.run(srv._confirmation_handler(
        "search_fitness_knowledge", {"query": "creatine"})) is True
    assert srv._state["pending_confirmation"] is False


# ══════════════════════════════════════════════════════════════════════════════
# (h) CONFIRMED MEANS WRITTEN — the server commits, not the agent
#
# LIVE-CAUGHT in the Phase 5 re-run, and it is the founding failure of this arc
# reaching the user again: "✅ The goal of 150 lbs ... has been deleted from your
# database" while Goal._id=2 sat in the table.
#
# The sequence: delete_goal staged, panel raised, user confirmed — and the agent
# then called verify_set_deleted (the wrong tool, for a GOAL) instead of
# execute_staged_goal_delete. That verifier answered "confirmed deleted" because
# no matching training_log SET existed, and the staged slot was never drained.
#
# A confirmed write must not depend on the model choosing to perform it. The
# staged key names exactly one execute tool.
# ══════════════════════════════════════════════════════════════════════════════

_SIBLING_KEYS = ["goal", "update_goal", "delete_goal",
                 "update_set", "delete_set", "set_comment"]


def _confirm_srv(srv, monkeypatch, staged_key, execute_result=None):
    """Drive /confirm for a sibling staged write, recording every tool call."""
    calls: list = []

    async def call_tool(name, args):
        calls.append(name)
        if name == "format_staged_write_for_confirmation":
            return json.dumps({"preview": "PREVIEW", "staged_key": staged_key}
                              if staged_key else {"error": "No staged write found."})
        if name.startswith("execute_"):
            return json.dumps(execute_result
                              or {"success": True, "verified": True,
                                  "message": "Saved and verified."})
        return json.dumps({"ok": True})

    async def answer(msg):
        calls.append("AGENT_ANSWER")
        return {"answer": "agent replied", "error": None}

    noted: list = []
    monkeypatch.setattr(srv, "session", SimpleNamespace(
        call_tool=call_tool, answer=answer, chat_history=[],
        _staged_active=True,
        note_host_write=lambda *a, **k: noted.append(a)))
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(_pending_log_carry=True))
    srv._state["pending_execute_kind"] = "goal"
    body = _body(asyncio.run(srv.confirm(SimpleNamespace(confirmed=True))))
    return body, calls, noted


@pytest.mark.parametrize("key", _SIBLING_KEYS)
def test_each_staged_key_drives_its_own_execute(srv, monkeypatch, key):
    expected = srv._SIBLING_EXECUTE[key]
    body, calls, _ = _confirm_srv(srv, monkeypatch, key)
    assert expected in calls, f"{key} did not execute {expected}: {calls}"
    assert body["type"] == "answer"
    # the agent is NOT the one committing it
    assert "AGENT_ANSWER" not in calls


@pytest.mark.parametrize("key", _SIBLING_KEYS)
def test_the_execute_result_is_the_outcome(srv, monkeypatch, key):
    """A failed execute must surface as a failure — never a success narrated by
    the agent on top of a write that did not land."""
    body, _, _ = _confirm_srv(
        srv, monkeypatch, key,
        execute_result={"success": False, "message": "write failed"})
    assert body["type"] == "error"
    assert "write failed" in body["text"]


def test_nothing_staged_falls_through_instead_of_guessing(srv, monkeypatch):
    """The other direction: with no staged key the server must not invent an
    execute — it hands back to the agent, the pre-existing behaviour."""
    body, calls, _ = _confirm_srv(srv, monkeypatch, None)
    assert not [c for c in calls if c.startswith("execute_")]
    assert "AGENT_ANSWER" in calls


def test_a_committed_sibling_tells_the_agent(srv, monkeypatch):
    """The 4b repair these flows never had — the server wrote it, so the agent
    must be told, or it goes on believing the edit is still pending."""
    _, _, noted = _confirm_srv(srv, monkeypatch, "delete_goal")
    assert noted, "the agent was never told its goal write committed"
    assert "PREVIEW" in noted[0][1]


def test_every_sibling_key_has_exactly_one_execute(srv):
    """No key may map to two tools, and no execute may serve two keys — that
    ambiguity is what an agent-driven commit had instead of a map."""
    tools = list(srv._SIBLING_EXECUTE.values())
    assert len(tools) == len(set(tools))
    assert set(srv._SIBLING_EXECUTE) == set(_SIBLING_KEYS)
