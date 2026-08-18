"""
Stage 2 of the operational-write verification layer: verify-at-staging.

ONE LLM diff call per logging turn — the assembled /log-flow user turns
(Input A) against the staged slot JSON + its deterministic rendering
(Input B) — firing after staging completes and BEFORE the confirm panel
(web) / execute (CLI).

(a) Input-A assembly on the Coordinator: single-turn /log strips the prefix
    and peels the analytical tail (workout portion only); a multi-turn carry
    assembles turn-1's head + each carry reply in order; scope is one flow
    by construction (snapshot-and-clear at route() entry — no leak across
    flows).
(b) Coordinator.verify_log_staging: PASS/FAIL parsed; unparseable output,
    client exception, or a missing slot-rendered preview all yield ERROR
    (fail-open — the panel is itself a human check); no retries.
(c) Server panel branch: PASS/ERROR proceed to confirmation_required with
    the verdict recorded; FAIL suppresses the panel, clears
    pending_execute_kind, discards the slot IMMEDIATELY (before the reply),
    then records the verdict — and a stray /confirm afterwards can never
    reach execute_staged_workout.
(d) MCP side: record_workout_verify stores the verdict as a sibling key in
    _staged_writes (cleared for free by discard_staged_writes) and is NOT
    exposed in list_tools.
(e) CLI seam: _finalize_staged_workout verifies before executing — FAIL
    discards and returns the re-state message without executing.

No Gemini (client stubbed / verify faked), no server process, no MCP
subprocess.
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

from src import coordinator as coordinator_mod          # noqa: E402
from src.coordinator import (                            # noqa: E402
    Coordinator,
    MSG_VERIFY_RESTATE,
    format_verify_fail_message,
)
import mcp_servers.combined_server as cs                 # noqa: E402


# ── Fixtures / stubs (mirroring test_log_command.py) ─────────────────────────

class FakeAgent:
    """AgentSession stand-in: records the question _run_operational forwards and
    returns a scripted answer() dict incl. the staging_reached_confirm signal."""

    def __init__(self, answer_text="Staged.", staging_reached_confirm=True,
                 staged_this_turn=False):
        self.questions: list[str] = []
        self.answer_text = answer_text
        self.staging_reached_confirm = staging_reached_confirm
        self.staged_this_turn = staged_this_turn

    async def answer(self, question):
        self.questions.append(question)
        return {
            "question": question,
            "answer": self.answer_text,
            "tool_calls_made": 1,
            "error": None,
            "staging_reached_confirm": self.staging_reached_confirm,
            "staged_this_turn": self.staged_this_turn,
        }


def _make_coord(monkeypatch, agent):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=agent)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _client_returning(text):
    resp = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                        finish_reason=None)
    ])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


_VALID_ANALYTICAL = (
    '{"route":"analytical","exercise_names":null,"muscle_groups":null,'
    '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}'
)


def _stub_analytical(coord, monkeypatch):
    async def an(q, p, resume=None):
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)


# ══════════════════════════════════════════════════════════════════════════════
# (a) Input-A assembly — the coordinator's _log_flow_turns / log_flow_turns key
# ══════════════════════════════════════════════════════════════════════════════

def test_input_a_single_turn_strips_prefix_and_tail(monkeypatch):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route(
        "/log bench 100x5, 3 sets. Also, how is my back progressing?"))

    # Workout portion ONLY: prefix stripped AND the analytical tail peeled.
    assert result["log_flow_turns"] == ["bench 100x5, 3 sets."]


def test_input_a_multi_turn_carry_assembles_in_order(monkeypatch):
    agent = FakeAgent(staging_reached_confirm=False)   # turn 1 asks for the date
    coord = _make_coord(monkeypatch, agent)

    r1 = asyncio.run(coord.route("/log deadlift 3 sets of 5 at 200 lbs"))
    assert r1["log_flow_turns"] == ["deadlift 3 sets of 5 at 200 lbs"]
    assert coord._pending_log_carry is True

    agent.staging_reached_confirm = True               # turn 2 completes staging
    r2 = asyncio.run(coord.route("27 June"))
    assert r2["log_boundary"] is True
    assert r2["log_flow_turns"] == ["deadlift 3 sets of 5 at 200 lbs", "27 June"]


def test_carry_not_armed_on_staged_turn(monkeypatch):
    """#13: under Fix 5 the agent never calls execute for a workout — the
    SERVER does, after /confirm — so staging_reached_confirm is structurally
    False on EVERY successful staged write. staged_this_turn is the direct
    signal that the panel takes over: the carry must stay down. (This exact
    result shape wrongly armed the carry before the fix.)"""
    agent = FakeAgent(staging_reached_confirm=False, staged_this_turn=True)
    coord = _make_coord(monkeypatch, agent)

    asyncio.run(coord._run_operational("bench 100x5 today", log_boundary=True))

    assert coord._pending_log_carry is False


def test_carry_not_armed_on_agent_driven_execute(monkeypatch):
    """Sibling flows (goal/set edits) where the agent itself reaches the
    execute gate keep the original signal — carry stays down."""
    agent = FakeAgent(staging_reached_confirm=True, staged_this_turn=False)
    coord = _make_coord(monkeypatch, agent)

    asyncio.run(coord._run_operational("bench 100x5 today", log_boundary=True))

    assert coord._pending_log_carry is False


def test_carry_armed_only_on_clarification_turn(monkeypatch):
    """Nothing staged, gate never reached ⇒ the turn ended in a logging
    clarification — the carry arms so the next reply joins the flow."""
    agent = FakeAgent(staging_reached_confirm=False, staged_this_turn=False)
    coord = _make_coord(monkeypatch, agent)

    asyncio.run(coord._run_operational("bench today", log_boundary=True))

    assert coord._pending_log_carry is True


def test_input_a_fallback_write_is_the_single_message(monkeypatch):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route("log bench 100x5, 3 sets"))

    assert result["log_boundary"] is False             # regex fallback, not /log
    assert result["log_flow_turns"] == ["log bench 100x5, 3 sets"]


def test_input_a_absent_on_non_write_turn(monkeypatch):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    _stub_analytical(coord, monkeypatch)
    coord._client = _client_returning(_VALID_ANALYTICAL)

    result = asyncio.run(coord.route("how is my bench progressing"))

    assert result["route"] == "analytical"
    assert result["log_flow_turns"] is None


def test_no_leak_across_flows(monkeypatch):
    """User-required leak test: turn 1 /log flow, turn 2 carry-consume, turn 3
    fresh analytical → the list is EMPTY after turn 3 (cleared entering it,
    never rebuilt); a turn-4 /log does not inherit turn-1/2 content."""
    agent = FakeAgent(staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    _stub_analytical(coord, monkeypatch)
    coord._client = _client_returning(_VALID_ANALYTICAL)

    asyncio.run(coord.route("/log deadlift 3 sets of 5 at 200 lbs"))   # turn 1
    agent.staging_reached_confirm = True
    asyncio.run(coord.route("27 June"))                                # turn 2 (carry)

    r3 = asyncio.run(coord.route("how is my bench progressing"))      # turn 3
    assert r3["log_flow_turns"] is None
    assert coord._log_flow_turns == []          # scope ended by construction

    r4 = asyncio.run(coord.route("/log squat 5x5 at 225 lbs"))        # turn 4
    assert r4["log_flow_turns"] == ["squat 5x5 at 225 lbs"]           # no turn-1/2 content


# ══════════════════════════════════════════════════════════════════════════════
# (b) verify_log_staging — verdict parsing, fail-open ERROR paths, prompt shape
# ══════════════════════════════════════════════════════════════════════════════

_SLOT = json.dumps({"staged_workouts": [
    {"exercise_id": 1, "date": "2026-06-27",
     "sets": [{"metric_weight": 90.72, "reps": 5, "unit": 0, "distance": 0,
               "duration_seconds": 0, "is_personal_record": 0, "comment": None}]},
]})
_PREVIEW = "Staged workout — 2026-06-27\n\nDeadlift\n  Set 1: 200 lbs × 5 reps"


def test_verify_pass(monkeypatch):
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_returning('{"verdict": "PASS"}')
    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, _PREVIEW))
    assert v == {"verdict": "PASS", "reason": ""}


def test_verify_fail_carries_reason(monkeypatch):
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_returning(
        '{"verdict": "FAIL", "reason": "set 2 comment missing from the JSON"}')
    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, _PREVIEW))
    assert v["verdict"] == "FAIL"
    assert v["reason"] == "set 2 comment missing from the JSON"


def test_format_verify_fail_message_includes_preview():
    msg = format_verify_fail_message(_PREVIEW)
    assert _PREVIEW in msg
    assert "didn't match your request" in msg


def test_verify_unparseable_output_is_error(monkeypatch):
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_returning("the staged workout looks fine to me")
    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, _PREVIEW))
    assert v["verdict"] == "ERROR"


def test_verify_unrecognized_verdict_is_error(monkeypatch):
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_returning('{"verdict": "MAYBE"}')
    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, _PREVIEW))
    assert v["verdict"] == "ERROR"


def test_verify_client_exception_is_error(monkeypatch):
    coord = _make_coord(monkeypatch, FakeAgent())

    def boom(**kw):
        raise RuntimeError("503 UNAVAILABLE")
    coord._client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, _PREVIEW))
    assert v["verdict"] == "ERROR"
    assert "503" in v["reason"]


def test_verify_missing_preview_skips_llm_call(monkeypatch):
    """A name-blind diff (opaque exercise_ids, kg weights, no rendering) could
    spuriously FAIL a good batch — the LLM must not even be called."""
    coord = _make_coord(monkeypatch, FakeAgent())
    calls = {"n": 0}

    def gen(**kw):
        calls["n"] += 1
        raise AssertionError("LLM must not be called without a preview")
    coord._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

    v = asyncio.run(coord.verify_log_staging(["deadlift 200x5"], _SLOT, None))
    assert v["verdict"] == "ERROR"
    assert calls["n"] == 0


def test_verify_prompt_is_a_diff_with_turns_in_order(monkeypatch):
    """The call must carry the numbered turns IN ORDER plus both Input-B
    artifacts, under a diff-not-re-extract system instruction."""
    coord = _make_coord(monkeypatch, FakeAgent())
    captured = {}
    resp = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(
            parts=[SimpleNamespace(text='{"verdict": "PASS"}')]),
            finish_reason=None)
    ])

    def gen(**kw):
        captured.update(kw)
        return resp
    coord._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

    turns = ["deadlift 3 sets of 5 at 200 lbs", "27 June"]
    asyncio.run(coord.verify_log_staging(turns, _SLOT, _PREVIEW))

    text = captured["contents"][0].parts[0].text
    assert "1. deadlift 3 sets of 5 at 200 lbs" in text
    assert "2. 27 June" in text
    assert text.index("1. deadlift") < text.index("2. 27 June")
    assert "[STAGED JSON]" in text and _SLOT in text
    assert "[STAGED RENDERING]" in text and _PREVIEW in text
    sysins = captured["config"].system_instruction
    assert "DIFF, not a re-parse" in sysins
    assert "Do NOT re-derive" in sysins


# ══════════════════════════════════════════════════════════════════════════════
# (c) Server panel branch — PASS proceeds, FAIL suppresses + discards, fail-open
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod, "session", None)
    server_mod._state.update({"pending_confirmation": False,
                              "allow_execute": False, "staging_preview": "",
                              "pending_execute_kind": None})
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


def _drive_workout_turn(srv, monkeypatch, *, verdict=None, verify_raises=False,
                        formatter_result=None, flow_turns=None,
                        message="log my workout"):
    """Drive one /chat turn that stages a workout (pending_execute_kind set by
    the fake route, as _confirmation_handler's log_workout branch would).
    Returns (body, calls, seen): the recorded call_tool log and what the
    stubbed verify received."""
    calls: list = []
    seen: dict = {}
    if formatter_result is None:
        formatter_result = {"preview": _PREVIEW}

    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": "", "route": "operational", "flagged_claims": [],
                "error": None, "log_boundary": True,
                "log_flow_turns": flow_turns}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "format_staged_workout_for_confirmation":
            return json.dumps(formatter_result)
        if name == "read_staged_workout_slot":
            return _SLOT
        if name == "execute_staged_workout":
            return json.dumps({"success": True, "message": "written"})
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        seen["flow"] = flow
        seen["slot"] = slot_json
        seen["preview"] = preview
        if verify_raises:
            raise RuntimeError("verify blew up")
        return verdict

    async def answer(msg):
        calls.append(("session.answer", msg))
        return {"answer": "ok", "error": None}

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging))
    monkeypatch.setattr(srv, "session", SimpleNamespace(
        call_tool=call_tool, answer=answer, note_host_write=lambda *a, **k: None))
    body = _body(asyncio.run(srv._process_turn(message)))
    return body, calls, seen


def test_server_pass_panel_proceeds_and_verdict_recorded(srv, monkeypatch):
    body, calls, _ = _drive_workout_turn(
        srv, monkeypatch, verdict={"verdict": "PASS", "reason": ""})

    assert body["type"] == "confirmation_required"
    assert body["preview_source"] == "slot"
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert records == [{"verdict": "PASS", "reason": ""}]
    # Only the turn-start discard fired — the batch survives for /confirm.
    assert [n for n, _ in calls].count("discard_staged_writes") == 1


def test_server_fail_suppresses_panel_and_discards_immediately(srv, monkeypatch):
    body, calls, _ = _drive_workout_turn(
        srv, monkeypatch,
        verdict={"verdict": "FAIL", "reason": "second exercise missing"})

    # Panel suppressed — the reply is the re-state prompt, not a confirmation.
    assert body["type"] == "answer"
    assert body["text"] == format_verify_fail_message(_PREVIEW)
    assert "preview" not in body
    # A stray /confirm must find nothing: kind cleared, slot discarded NOW.
    assert srv._state["pending_execute_kind"] is None
    names = [n for n, _ in calls]
    assert names.count("discard_staged_writes") == 2   # turn-start + FAIL
    # Discard BEFORE record: discard clears the whole dict, so the FAIL verdict
    # is written after it (the surviving sibling key).
    fail_discard = len(names) - 1 - names[::-1].index("discard_staged_writes")
    record = names.index("record_workout_verify")
    assert fail_discard < record
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert records == [{"verdict": "FAIL", "reason": "second exercise missing"}]
    assert "execute_staged_workout" not in names
    # #13 defense in depth: a FAILed flow leaves no carry behind.
    assert srv.coordinator._pending_log_carry is False


def test_server_post_fail_stray_confirm_cannot_reach_execute(srv, monkeypatch):
    _, calls, _ = _drive_workout_turn(
        srv, monkeypatch, verdict={"verdict": "FAIL", "reason": "wrong date"})

    resp = asyncio.run(srv.confirm(SimpleNamespace(confirmed=True)))
    body = _body(resp)

    # pending_kind was None → the workout execute branch never fires; the
    # sibling path re-prompts the agent (harmless), and execute is NEVER called.
    assert "execute_staged_workout" not in [n for n, _ in calls]
    assert body["type"] in ("answer", "error")


def test_server_verify_exception_fails_open(srv, monkeypatch):
    body, calls, _ = _drive_workout_turn(srv, monkeypatch, verify_raises=True)

    assert body["type"] == "confirmation_required"     # panel still shows
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert len(records) == 1 and records[0]["verdict"] == "ERROR"


def test_server_args_fallback_preview_skips_verify(srv, monkeypatch):
    # Formatter returned an error payload → args-fallback preview → the LLM
    # verify is SKIPPED (ERROR verdict), never called with a name-blind diff.
    body, calls, seen = _drive_workout_turn(
        srv, monkeypatch, verdict={"verdict": "PASS", "reason": ""},
        formatter_result={"error": "No staged workout found."})

    assert body["type"] == "confirmation_required"
    assert body["preview_source"] == "args_fallback"
    assert seen == {}                                  # verify never invoked
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert len(records) == 1 and records[0]["verdict"] == "ERROR"


def test_server_verify_receives_assembled_flow_turns(srv, monkeypatch):
    flow = ["deadlift 3 sets of 5 at 200 lbs", "27 June"]
    _, _, seen = _drive_workout_turn(
        srv, monkeypatch, verdict={"verdict": "PASS", "reason": ""},
        flow_turns=flow)
    assert seen["flow"] == flow
    assert seen["slot"] == _SLOT
    assert seen["preview"] == _PREVIEW


def test_server_verify_never_receives_bare_message_without_flow_turns(srv, monkeypatch):
    # REWRITTEN (was: ...falls_back_to_raw_message...): the old [message]
    # fallback diffed the staged slot against the bare reply text and turned a
    # lost flow thread into a confidently-wrong FAIL that discarded a good
    # batch. The caller now passes the missing thread through as-is; the REAL
    # verify_log_staging skip-guards it into a fail-open ERROR (panel still
    # shows — pinned separately in test_write_path_fixes.py).
    _, _, seen = _drive_workout_turn(
        srv, monkeypatch, verdict={"verdict": "PASS", "reason": ""},
        flow_turns=None, message="log bench 100x5")
    assert seen["flow"] is None                # bare reply never becomes Input A


# ══════════════════════════════════════════════════════════════════════════════
# (d) MCP side — sibling-key storage, discard cleanup, unexposed tool
# ══════════════════════════════════════════════════════════════════════════════

def test_record_workout_verify_stores_sibling_key():
    cs._staged_writes.clear()
    out = asyncio.run(cs.call_tool(
        "record_workout_verify", {"verdict": "FAIL", "reason": "missing comment"}))
    assert json.loads(out[0].text) == {"recorded": True}
    assert cs._staged_writes["workout_verify"] == {
        "verdict": "FAIL", "reason": "missing comment"}


def test_discard_staged_writes_clears_workout_verify():
    cs._staged_writes.clear()
    cs._staged_writes["workout_verify"] = {"verdict": "PASS", "reason": ""}
    cs._staged_writes["workout"] = [{"exercise_id": 1, "date": "2026-06-27", "sets": []}]
    cs._discard_staged_writes_sync()
    assert "workout_verify" not in cs._staged_writes   # sibling-key cleanup for free
    assert "workout" not in cs._staged_writes


def test_record_workout_verify_dispatchable_but_unexposed():
    exposed = {t.name for t in asyncio.run(cs.list_tools())}
    assert "record_workout_verify" not in exposed


# ══════════════════════════════════════════════════════════════════════════════
# (e) CLI seam — _finalize_staged_workout verifies before executing
# ══════════════════════════════════════════════════════════════════════════════

import cli as cli_mod                                     # noqa: E402


def _cli_fakes(verdict, execute_result=None):
    calls: list = []

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            return _SLOT
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "execute_staged_workout":
            return json.dumps(execute_result or {"success": True, "message": "saved"})
        return json.dumps({"ok": True})

    session = SimpleNamespace(call_tool=call_tool, _staged_active=True, note_host_write=lambda *a, **k: None)

    async def verify(flow, slot_json, preview):
        return verdict
    coordinator = SimpleNamespace(verify_log_staging=verify)
    return session, coordinator, calls


def test_cli_fail_discards_and_skips_execute():
    session, coord_ns, calls = _cli_fakes(
        {"verdict": "FAIL", "reason": "set count mismatch"})
    line = asyncio.run(cli_mod._finalize_staged_workout(
        session, coord_ns, {"log_flow_turns": ["bench 100x5"]}, "bench 100x5"))

    assert line == format_verify_fail_message(_PREVIEW)
    names = [n for n, _ in calls]
    assert "execute_staged_workout" not in names
    assert names.index("discard_staged_writes") < names.index("record_workout_verify")
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert records == [{"verdict": "FAIL", "reason": "set count mismatch"}]


def test_cli_pass_records_then_executes():
    session, coord_ns, calls = _cli_fakes({"verdict": "PASS", "reason": ""})
    line = asyncio.run(cli_mod._finalize_staged_workout(
        session, coord_ns, {"log_flow_turns": ["bench 100x5"]}, "bench 100x5"))

    assert line.startswith("✅")
    names = [n for n, _ in calls]
    assert "discard_staged_writes" not in names
    assert names.index("record_workout_verify") < names.index("execute_staged_workout")
    assert session._staged_active is False
