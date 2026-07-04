"""
Write-path checkpoint-and-restore across the stage→verify→panel LLM boundaries.

(a) checkpoint.py: the three write-path fields round-trip; clear_staged_checkpoint
    clears ONLY a staged cp (guard negative pinned); enrich_checkpoint reads the
    slot RAW — no staleness gate can no-op it on a fresh (or even ancient) cp.
(b) Boundary 1: a QuotaInterrupted escaping agent.answer during a /log turn
    enriches the agent's just-written checkpoint with the staged slot + the
    assembled flow turns; an empty slot enriches nothing; the 429 always
    propagates.
(c) coordinator._resume three-way branch: slot+PASS restores and arms without
    any agent call; slot+no-verdict restores with restored_verify None; no
    slot → the existing agent.resume path; invalid ids → re-state + cp cleared.
(d) server: the restore signal arms the panel with NO agent call; a restored
    PASS verdict skips the verify LLM; case-2 runs verify with the RESTORED
    flow turns; checkpoint-2 is written on PASS with the real turns/question
    (never ["continue"]/"continue"); /confirm success+cancel clear ONLY a
    staged cp.
(e) MCP restore tool: id-validated against the current DB, refuses wholesale
    on a miss, dispatchable but unexposed.
(f) CLI: restored batches always pass the interactive gate (ERROR included —
    fail-open never skips the gate); FAIL re-states without prompting.

No Gemini, no server process, no MCP subprocess.
"""

import asyncio
import json
import os
import sqlite3
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
from src.coordinator import Coordinator, MSG_VERIFY_RESTATE, format_verify_fail_message  # noqa: E402
from src import checkpoint as ckpt                      # noqa: E402
import mcp_servers.combined_server as cs                # noqa: E402


# ── Fixtures / stubs ─────────────────────────────────────────────────────────

@pytest.fixture
def ckpt_path(tmp_path, monkeypatch):
    p = str(tmp_path / "checkpoint.json")
    monkeypatch.setenv("CHECKPOINT_PATH", p)
    return p


_SLOT = [{"exercise_id": 1, "date": "2026-06-27",
          "sets": [{"metric_weight": 90.72, "reps": 5, "unit": 0, "distance": 0,
                    "duration_seconds": 0, "is_personal_record": 0,
                    "comment": "grip felt strong"}]}]
_FLOW = ["deadlift 3 sets of 5 at 200 lbs", "27 June"]
_PASS = {"verdict": "PASS", "reason": ""}
_PREVIEW = "Staged workout — 2026-06-27\n\nDeadlift\n  Set 1: 200 lbs × 5 reps"
_QUESTION = "deadlift 3 sets of 5 at 200 lbs"


class RestoreFakeAgent:
    """AgentSession stand-in for resume tests: programmable call_tool with a
    call log; resume()/answer() raise unless explicitly allowed — restoring
    must never touch the agent loop."""

    def __init__(self, tool_responses=None, resume_result=None):
        self.calls: list = []
        self.tool_responses = tool_responses or {}
        self.resume_called = False
        self.resume_result = resume_result
        self._staged_active = False

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        r = self.tool_responses.get(name)
        if r is not None:
            return r(args) if callable(r) else r
        return json.dumps({"ok": True})

    async def resume(self, cp):
        if self.resume_result is None:
            raise AssertionError("agent.resume must not be called on a staged restore")
        self.resume_called = True
        return self.resume_result

    async def answer(self, question):
        raise AssertionError("agent.answer must not be called on resume")


def _make_coord(monkeypatch, agent):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    return Coordinator(agent_session=agent)


def _staged_cp(ckpt_path, verdict=_PASS, question=_QUESTION):
    return ckpt.save_checkpoint(
        route="operational", question=question,
        staged_slot=_SLOT, log_flow_turns=_FLOW, verify_verdict=verdict)


# ══════════════════════════════════════════════════════════════════════════════
# (a) checkpoint.py — fields, guarded clear, raw-read enrich
# ══════════════════════════════════════════════════════════════════════════════

def test_checkpoint_roundtrip_staged_fields(ckpt_path):
    _staged_cp(ckpt_path)
    cp = ckpt.load_checkpoint()
    assert cp["staged_slot"] == _SLOT
    assert cp["log_flow_turns"] == _FLOW
    assert cp["verify_verdict"] == _PASS
    # mark_awaiting_discard dict-copies — the write-path fields must survive.
    cp2 = ckpt.mark_awaiting_discard(cp, "new question")
    reloaded = ckpt.load_checkpoint()
    assert cp2["staged_slot"] == _SLOT
    assert reloaded["staged_slot"] == _SLOT
    assert reloaded["awaiting_discard_confirm"] is True


def test_clear_staged_checkpoint_guard(ckpt_path):
    _staged_cp(ckpt_path)
    assert ckpt.clear_staged_checkpoint() is True
    assert ckpt.load_checkpoint() is None
    # Negative: a checkpoint WITHOUT a staged slot is never destroyed by the
    # staged-clear (an unrelated interrupted question must survive /confirm).
    ckpt.save_checkpoint(route="analytical", question="how's my bench")
    assert ckpt.clear_staged_checkpoint() is False
    assert ckpt.load_checkpoint() is not None


def test_enrich_checkpoint_raw_read_no_staleness_gate(ckpt_path):
    # An ANCIENT cp (load_checkpoint's 48h gate would clear it and return
    # None) — enrich must still read/merge/write, proving it routes through a
    # raw file read, not the staleness gate.
    ckpt._write({"created": "2020-01-01T00:00:00", "route": "operational",
                 "question": _QUESTION})
    out = ckpt.enrich_checkpoint({"staged_slot": _SLOT, "log_flow_turns": _FLOW})
    assert out is not None and out["staged_slot"] == _SLOT
    with open(ckpt_path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["staged_slot"] == _SLOT
    assert raw["log_flow_turns"] == _FLOW
    assert raw["created"] == "2020-01-01T00:00:00"   # untouched, ungated


def test_enrich_checkpoint_missing_file_is_none(ckpt_path):
    assert ckpt.enrich_checkpoint({"staged_slot": _SLOT}) is None
    assert not os.path.exists(ckpt_path)             # nothing conjured


# ══════════════════════════════════════════════════════════════════════════════
# (b) Boundary 1 — QuotaInterrupted enrichment in _run_operational
# ══════════════════════════════════════════════════════════════════════════════

class QuotaFakeAgent(RestoreFakeAgent):
    """answer() mimics the real agent's on-429 behavior: save the transcript
    checkpoint, then raise QuotaInterrupted."""

    def __init__(self, slot_json):
        super().__init__(tool_responses={"read_staged_workout_slot": slot_json})

    async def answer(self, question):
        ckpt.save_checkpoint(route="operational", question=question,
                             messages=[{"role": "user", "content": question}])
        raise ckpt.QuotaInterrupted(Exception("429 RESOURCE_EXHAUSTED"), "interrupted")


def test_quota_interrupt_enriches_checkpoint_with_slot_and_flow(monkeypatch, ckpt_path):
    agent = QuotaFakeAgent(json.dumps({"staged_workouts": _SLOT}))
    coord = _make_coord(monkeypatch, agent)

    with pytest.raises(ckpt.QuotaInterrupted):
        asyncio.run(coord.route("/log deadlift 3 sets of 5 at 200 lbs"))

    cp = ckpt.load_checkpoint()
    assert cp["staged_slot"] == _SLOT
    assert cp["log_flow_turns"] == ["deadlift 3 sets of 5 at 200 lbs"]
    assert cp["messages"]                       # agent's transcript untouched


def test_quota_interrupt_empty_slot_no_enrichment(monkeypatch, ckpt_path):
    agent = QuotaFakeAgent(json.dumps({"staged_workouts": []}))
    coord = _make_coord(monkeypatch, agent)

    with pytest.raises(ckpt.QuotaInterrupted):
        asyncio.run(coord.route("/log deadlift 3 sets of 5 at 200 lbs"))

    cp = ckpt.load_checkpoint()
    assert cp.get("staged_slot") is None        # nothing staged → case 3 resume


# ══════════════════════════════════════════════════════════════════════════════
# (c) coordinator._resume — the three-way branch
# ══════════════════════════════════════════════════════════════════════════════

def test_resume_case1_restores_arms_skips_agent(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path, verdict=_PASS)
    agent = RestoreFakeAgent(tool_responses={
        "restore_staged_workout_slot": json.dumps({"restored": True, "workouts": 1}),
    })
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route("continue"))

    assert result["restore_staged"] is True
    assert result["restored_verify"] == _PASS
    assert result["log_flow_turns"] == _FLOW
    assert result["restored_question"] == _QUESTION
    assert agent._staged_active is True
    assert agent.resume_called is False
    restore_calls = [a for n, a in agent.calls if n == "restore_staged_workout_slot"]
    assert restore_calls == [{"staged_workouts": _SLOT, "verify": _PASS}]
    # Keep-until-confirm: the cp survives panel-arming.
    assert ckpt.load_checkpoint() is not None


def test_resume_case2_no_verdict_restores_without_verify(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path, verdict=None)
    agent = RestoreFakeAgent(tool_responses={
        "restore_staged_workout_slot": json.dumps({"restored": True, "workouts": 1}),
    })
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route("continue"))

    assert result["restore_staged"] is True
    assert result["restored_verify"] is None    # server/CLI verify NOW
    assert result["log_flow_turns"] == _FLOW
    restore_calls = [a for n, a in agent.calls if n == "restore_staged_workout_slot"]
    assert restore_calls == [{"staged_workouts": _SLOT, "verify": None}]


def test_resume_case3_no_slot_uses_agent_path(monkeypatch, ckpt_path):
    ckpt.save_checkpoint(route="operational", question="log my workout",
                         messages=[{"role": "user", "content": "log my workout"}])
    agent = RestoreFakeAgent(resume_result={
        "question": "log my workout", "answer": "resumed by agent",
        "tool_calls_made": 0, "error": None,
    })
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route("continue"))

    assert agent.resume_called is True          # unchanged existing behavior
    assert result["answer"] == "resumed by agent"
    assert not result.get("restore_staged")
    assert ckpt.load_checkpoint() is None       # existing clear-on-success


def test_resume_invalid_ids_restates_and_clears(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path)
    agent = RestoreFakeAgent(tool_responses={
        "restore_staged_workout_slot": json.dumps(
            {"error": "invalid_exercise_ids", "missing": [1]}),
    })
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route("continue"))

    assert "re-log" in result["answer"]
    assert not result.get("restore_staged")     # gate never armed
    assert ckpt.load_checkpoint() is None       # dead slot cleared


# ══════════════════════════════════════════════════════════════════════════════
# (d) server — arming, verify skip, checkpoint-2, /confirm clears
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod, "session", None)
    server_mod._state.update({"pending_confirmation": False,
                              "allow_execute": False, "staging_preview": "",
                              "confirmation_preview": "",
                              "pending_execute_kind": None})
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


def _drive_restore_turn(srv, monkeypatch, *, restored_verify, verify_verdict=None,
                        message="continue"):
    """Drive one /chat turn whose fake route returns a restore-signal result
    (as coordinator._resume_staged_workout would). Returns
    (body, calls, verify_seen, saved_checkpoints)."""
    calls: list = []
    seen: dict = {}
    saved: list = []

    async def route(msg):
        return {"answer": "Restored your staged workout — confirm below to save it.",
                "route": "operational", "flagged_claims": [], "error": None,
                "log_boundary": True, "log_flow_turns": _FLOW,
                "restore_staged": True, "restored_verify": restored_verify,
                "restored_question": _QUESTION}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "read_staged_workout_slot":
            return json.dumps({"staged_workouts": _SLOT})
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        seen["flow"] = flow
        seen["preview"] = preview
        return verify_verdict

    def save_checkpoint(**kw):
        saved.append(kw)
        return kw

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging))
    monkeypatch.setattr(srv, "session", SimpleNamespace(call_tool=call_tool))
    monkeypatch.setattr(srv._ckpt, "save_checkpoint", save_checkpoint)
    body = _body(asyncio.run(srv._process_turn(message)))
    return body, calls, seen, saved


def test_server_restore_case1_arms_panel_no_agent_no_verify(srv, monkeypatch):
    body, calls, seen, saved = _drive_restore_turn(
        srv, monkeypatch, restored_verify=_PASS)

    assert body["type"] == "confirmation_required"
    assert body["preview_source"] == "slot"
    assert seen == {}                            # verify LLM never called (case 1)
    records = [a for n, a in calls if n == "record_workout_verify"]
    assert records == [_PASS]
    # Q1: checkpoint-2 stores the REAL flow turns and original question — the
    # ["continue"]/"continue" fallback is structurally bypassed on restore.
    assert len(saved) == 1
    assert saved[0]["log_flow_turns"] == _FLOW
    assert saved[0]["question"] == _QUESTION
    assert saved[0]["staged_slot"] == _SLOT
    assert saved[0]["verify_verdict"] == _PASS
    assert "continue" not in saved[0]["log_flow_turns"]
    assert saved[0]["question"] != "continue"


def test_server_restore_case2_runs_verify_with_restored_flow(srv, monkeypatch):
    body, calls, seen, saved = _drive_restore_turn(
        srv, monkeypatch, restored_verify=None, verify_verdict=_PASS)

    assert body["type"] == "confirmation_required"
    assert seen["flow"] == _FLOW                 # restored Input A, not ["continue"]
    assert seen["preview"] == _PREVIEW
    assert len(saved) == 1 and saved[0]["verify_verdict"] == _PASS


def test_server_restore_case2_fail_restates_discards_clears(srv, monkeypatch):
    cleared = {"n": 0}
    monkeypatch.setattr(srv._ckpt, "clear_staged_checkpoint",
                        lambda: cleared.__setitem__("n", cleared["n"] + 1) or True)
    body, calls, seen, saved = _drive_restore_turn(
        srv, monkeypatch, restored_verify=None,
        verify_verdict={"verdict": "FAIL", "reason": "batch incomplete"})

    assert body["type"] == "answer"
    assert body["text"] == format_verify_fail_message(_PREVIEW)
    assert srv._state["pending_execute_kind"] is None
    names = [n for n, _ in calls]
    assert "discard_staged_writes" in names
    assert cleared["n"] == 1                     # restore loop closed
    assert saved == []                           # no checkpoint-2 on FAIL


def test_server_normal_pass_writes_checkpoint2(srv, monkeypatch):
    # Normal (non-restore) staged turn: PASS must checkpoint the verified
    # batch before the panel wait, keyed by the real message + flow turns.
    calls: list = []
    saved: list = []

    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": "", "route": "operational", "flagged_claims": [],
                "error": None, "log_boundary": True,
                "log_flow_turns": ["deadlift 3 sets of 5 at 200 lbs"]}

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "read_staged_workout_slot":
            return json.dumps({"staged_workouts": _SLOT})
        return json.dumps({"ok": True})

    async def verify_log_staging(flow, slot_json, preview):
        return dict(_PASS)

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging))
    monkeypatch.setattr(srv, "session", SimpleNamespace(call_tool=call_tool))
    monkeypatch.setattr(srv._ckpt, "save_checkpoint",
                        lambda **kw: saved.append(kw) or kw)

    body = _body(asyncio.run(srv._process_turn("/log deadlift 3 sets of 5 at 200 lbs")))

    assert body["type"] == "confirmation_required"
    assert len(saved) == 1
    assert saved[0]["staged_slot"] == _SLOT
    assert saved[0]["log_flow_turns"] == ["deadlift 3 sets of 5 at 200 lbs"]
    assert saved[0]["question"] == "/log deadlift 3 sets of 5 at 200 lbs"
    assert saved[0]["verify_verdict"]["verdict"] == "PASS"


def _confirm_session(calls, execute_result):
    async def call_tool(name, args):
        calls.append((name, args))
        if name == "execute_staged_workout":
            return json.dumps(execute_result)
        return json.dumps({"ok": True})

    async def answer(msg):
        return {"answer": "ok", "error": None}
    return SimpleNamespace(call_tool=call_tool, answer=answer,
                           _staged_active=True)


def test_confirm_success_clears_staged_checkpoint(srv, monkeypatch, ckpt_path):
    _staged_cp(ckpt_path)
    calls: list = []
    monkeypatch.setattr(srv, "session", _confirm_session(
        calls, {"success": True, "message": "saved"}))
    srv._state["pending_execute_kind"] = "workout"

    body = _body(asyncio.run(srv.confirm(SimpleNamespace(confirmed=True))))

    assert body["type"] == "answer" and body["text"].startswith("✅")
    assert ckpt.load_checkpoint() is None        # double-write closed


def test_confirm_cancel_clears_staged_checkpoint(srv, monkeypatch, ckpt_path):
    _staged_cp(ckpt_path)
    calls: list = []
    monkeypatch.setattr(srv, "session", _confirm_session(
        calls, {"success": True, "message": "saved"}))
    srv._state["pending_execute_kind"] = "workout"

    asyncio.run(srv.confirm(SimpleNamespace(confirmed=False)))

    assert ckpt.load_checkpoint() is None        # rejected batch not restorable
    assert "execute_staged_workout" not in [n for n, _ in calls]


def test_confirm_never_clears_non_staged_checkpoint(srv, monkeypatch, ckpt_path):
    # Negative guard: an unrelated interrupted-question checkpoint survives
    # both /confirm outcomes.
    ckpt.save_checkpoint(route="analytical", question="how's my bench")
    calls: list = []
    monkeypatch.setattr(srv, "session", _confirm_session(
        calls, {"success": True, "message": "saved"}))
    srv._state["pending_execute_kind"] = "workout"

    asyncio.run(srv.confirm(SimpleNamespace(confirmed=True)))
    assert ckpt.load_checkpoint() is not None

    srv._state["pending_execute_kind"] = "workout"
    asyncio.run(srv.confirm(SimpleNamespace(confirmed=False)))
    assert ckpt.load_checkpoint() is not None


# ══════════════════════════════════════════════════════════════════════════════
# (e) MCP restore tool — id validation, dispatch, unexposed
# ══════════════════════════════════════════════════════════════════════════════

def _make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (
            _id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER
        );
        INSERT INTO exercise (_id, name, category_id) VALUES (1, 'Test Press', 5);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = str(tmp_path / "w.fitnotes")
    _make_db(p)
    monkeypatch.setattr(cs, "DB_PATH", p)
    cs._staged_writes.clear()
    return p


def test_restore_tool_writes_slot_and_verdict(db):
    out = json.loads(cs._restore_staged_workout_slot_sync(
        {"staged_workouts": _SLOT, "verify": _PASS}))
    assert out == {"restored": True, "workouts": 1}
    assert cs._staged_writes["workout"] == _SLOT
    assert cs._staged_writes["workout_verify"] == _PASS


def test_restore_tool_no_verdict_leaves_verify_unset(db):
    out = json.loads(cs._restore_staged_workout_slot_sync(
        {"staged_workouts": _SLOT, "verify": None}))
    assert out["restored"] is True
    assert "workout_verify" not in cs._staged_writes


def test_restore_tool_invalid_id_refuses_wholesale(db):
    bad = [dict(_SLOT[0], exercise_id=999)]
    out = json.loads(cs._restore_staged_workout_slot_sync(
        {"staged_workouts": _SLOT + bad, "verify": _PASS}))
    assert out["error"] == "invalid_exercise_ids"
    assert out["missing"] == [999]
    assert "workout" not in cs._staged_writes    # nothing written on a miss
    assert "workout_verify" not in cs._staged_writes


def test_restore_tool_empty_payload_refuses(db):
    out = json.loads(cs._restore_staged_workout_slot_sync({"staged_workouts": []}))
    assert "error" in out
    assert "workout" not in cs._staged_writes


def test_restore_tool_dispatchable_but_unexposed(db):
    out = asyncio.run(cs.call_tool(
        "restore_staged_workout_slot", {"staged_workouts": _SLOT}))
    assert json.loads(out[0].text)["restored"] is True
    exposed = {t.name for t in asyncio.run(cs.list_tools())}
    assert "restore_staged_workout_slot" not in exposed


# ══════════════════════════════════════════════════════════════════════════════
# (f) CLI — restored batches always pass the interactive gate
# ══════════════════════════════════════════════════════════════════════════════

import cli as cli_mod                                     # noqa: E402


def _cli_restore_fakes(verify_verdict=None):
    calls: list = []

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "read_staged_workout_slot":
            return json.dumps({"staged_workouts": _SLOT})
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "execute_staged_workout":
            return json.dumps({"success": True, "message": "saved"})
        return json.dumps({"ok": True})

    session = SimpleNamespace(call_tool=call_tool, _staged_active=True)

    async def verify(flow, slot_json, preview):
        return verify_verdict
    return session, SimpleNamespace(verify_log_staging=verify), calls


def _restore_result(restored_verify):
    return {"restore_staged": True, "restored_verify": restored_verify,
            "log_flow_turns": _FLOW, "restored_question": _QUESTION}


def test_cli_restore_pass_gate_yes_executes(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path)
    session, coord_ns, calls = _cli_restore_fakes()
    prompts: list = []
    monkeypatch.setattr("builtins.input",
                        lambda p="": prompts.append(p) or "yes")

    line = asyncio.run(cli_mod._confirm_restored_workout(
        session, coord_ns, _restore_result(_PASS)))

    assert line.startswith("✅")
    assert prompts                               # gate WAS consulted
    assert "execute_staged_workout" in [n for n, _ in calls]
    assert ckpt.load_checkpoint() is None        # committed → cp cleared


def test_cli_restore_gate_no_discards_and_clears(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path)
    session, coord_ns, calls = _cli_restore_fakes()
    monkeypatch.setattr("builtins.input", lambda p="": "no")

    line = asyncio.run(cli_mod._confirm_restored_workout(
        session, coord_ns, _restore_result(_PASS)))

    assert line.startswith("❌")
    names = [n for n, _ in calls]
    assert "discard_staged_writes" in names
    assert "execute_staged_workout" not in names
    assert ckpt.load_checkpoint() is None        # rejected → cp cleared


def test_cli_restore_case2_error_still_gated(monkeypatch, ckpt_path):
    # Q3: verify machinery ERROR on the no-verdict branch must fall THROUGH
    # to the interactive gate — fail-open never skips the human gate, and
    # execute fires only after an explicit "yes".
    _staged_cp(ckpt_path, verdict=None)
    session, coord_ns, calls = _cli_restore_fakes(
        verify_verdict={"verdict": "ERROR", "reason": "503 UNAVAILABLE"})
    prompts: list = []
    monkeypatch.setattr("builtins.input",
                        lambda p="": prompts.append(p) or "yes")

    line = asyncio.run(cli_mod._confirm_restored_workout(
        session, coord_ns, _restore_result(None)))

    assert prompts                               # input() consulted BEFORE any write
    names = [n for n, _ in calls]
    assert "execute_staged_workout" in names     # only after "yes"
    assert names.index("record_workout_verify") < names.index("execute_staged_workout")
    assert line.startswith("✅")


def test_cli_restore_case2_fail_no_prompt_no_execute(monkeypatch, ckpt_path):
    _staged_cp(ckpt_path, verdict=None)
    session, coord_ns, calls = _cli_restore_fakes(
        verify_verdict={"verdict": "FAIL", "reason": "batch incomplete"})
    monkeypatch.setattr(
        "builtins.input",
        lambda p="": (_ for _ in ()).throw(AssertionError("no prompt on FAIL")))

    line = asyncio.run(cli_mod._confirm_restored_workout(
        session, coord_ns, _restore_result(None)))

    assert line == format_verify_fail_message(_PREVIEW)
    names = [n for n, _ in calls]
    assert "discard_staged_writes" in names
    assert "execute_staged_workout" not in names
    assert ckpt.load_checkpoint() is None        # restore loop closed
