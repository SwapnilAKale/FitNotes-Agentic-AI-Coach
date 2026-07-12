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
                              "pending_execute_kind": None})
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
    monkeypatch.setattr(srv, "session", SimpleNamespace(call_tool=call_tool))
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
    return SimpleNamespace(call_tool=call_tool, _staged_active=True)


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
# Stage 3: a decomposed turn's merged answer survives the panel
# ══════════════════════════════════════════════════════════════════════════════

def test_decomposed_merged_answer_survives_confirmation_panel(srv, monkeypatch):
    """A mixed analytical+write turn stages a batch AND carries the merged
    non-write answer. The confirmation_required payload must ship both: the
    merged text (in `text` and prepended to the preview until the frontend
    renders `text`) and the staged preview."""
    merged = "### Is my squat progressing?\n\nYour squat is up 5%."
    calls: list = []

    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": merged, "route": "analytical", "decomposed": True,
                "flagged_claims": [], "error": None, "log_boundary": False,
                "log_flow_turns": ["Log bench 100 lbs for 5 reps today"]}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            return _SLOT_REAL
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        return {"verdict": "PASS", "reason": ""}

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging))
    monkeypatch.setattr(srv, "session", SimpleNamespace(
        call_tool=call_tool, chat_history=[]))     # route "analytical" mirrors
    body = _body(asyncio.run(srv._process_turn(
        "is my squat progressing and log bench 100x5")))

    assert body["type"] == "confirmation_required"
    assert body["text"] == merged                       # for the frontend
    assert merged in body["preview"]                    # visible NOW
    assert _PREVIEW in body["preview"]                  # staged lines intact
    assert body["preview"].index(merged) < body["preview"].index(_PREVIEW)


def test_non_decomposed_panel_payload_unchanged(srv, monkeypatch):
    """A plain single-write turn's panel is byte-identical to before Stage 3
    (no answer prepended, empty text field)."""
    body, calls, seen = _drive(
        srv, monkeypatch, answer_text="Staged.",
        slot_response=_SLOT_REAL,
        formatter_response=json.dumps({"preview": _PREVIEW}),
        verdict={"verdict": "PASS", "reason": ""})

    assert body["type"] == "confirmation_required"
    assert body["preview"] == _PREVIEW                  # no prepend
    assert body["text"] == ""
