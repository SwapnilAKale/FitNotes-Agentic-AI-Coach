"""
Chat history records every turn the moment it starts (live re-check, 2026-09-16).

WHY THIS EXISTS. A turn reached history only when it FINISHED, and only some
kinds did: the server added analytical turns and the agent added operational
ones. So a page reloaded mid-answer lost the prompt and the "…", unlocked the
input, and a message typed then was silently rejected as busy. Recall answers,
refusals, greetings, errors and quota messages were never recorded at all, and
neither was a "✅ logged" confirmation; a waiting confirmation panel had no way
back after a reload.

The contract (the page relies on it):
  • an accepted /chat turn adds the user's prompt AND a blank assistant entry
    marked `pending` at once;
  • when the turn ends that entry is filled — or, when the reply is a panel,
    removed (the panel is restored by its own endpoint);
  • an error reply is marked `kind: "error"`;
  • a turn rejected before it starts records nothing;
  • button actions (/resume, /disambiguate, /confirm) add no user entry.
The server is the only writer: anything the agent appends during a turn is
replaced, so nothing is recorded twice.
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

_PREVIEW = "Barbell Row — 2026-07-09\n  132.3 lbs × 8 reps"
_SLOT = json.dumps({"staged_workouts": [
    {"exercise_id": 1, "exercise_name": "Barbell Row", "date": "2026-07-09",
     "sets": [{"metric_weight": 60.0109, "reps": 8, "unit": 0}]}]})


@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod._ckpt, "save_checkpoint", lambda **kw: None)
    monkeypatch.setattr(server_mod._ckpt, "clear_staged_checkpoint", lambda: None)
    server_mod._state.update({"pending_confirmation": False, "allow_execute": False,
                              "staging_preview": "", "confirmation_preview": "",
                              "pending_execute_kind": None, "decomposed_answer": ""})
    server_mod._state.pop("panel", None)
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


def _install(srv, monkeypatch, route, *, verdict="PASS", execute_ok=True, agent_answer=None):
    session = SimpleNamespace(chat_history=[], note_host_write=lambda *a, **k: None,
                              _staged_active=True)

    async def call_tool(name, args):
        if name == "read_staged_workout_slot":
            return _SLOT
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        if name == "execute_staged_workout":
            return json.dumps({"success": execute_ok, "message": "written"})
        return json.dumps({"ok": True})

    async def answer(message):
        # The real agent records its own turn; the server must not end up with two.
        session.chat_history += [{"role": "user", "text": message},
                                 {"role": "assistant", "text": f"AGENT({message})"}]
        return {"answer": agent_answer or f"AGENT({message})", "error": None}

    async def verify_log_staging(flow, slot_json, preview):
        return {"verdict": verdict, "reason": ""}

    session.call_tool, session.answer = call_tool, answer
    monkeypatch.setattr(srv, "session", session)
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        route=route, verify_log_staging=verify_log_staging, _pending_log_carry=False))
    return session


def _simple(text, route="analytical"):
    async def route_fn(msg):
        return {"answer": text, "route": route, "flagged_claims": [], "error": None}
    return route_fn


def _pairs(history):
    return [(m["role"], m["text"]) for m in history]


def _panel_route(srv):
    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": "staged", "route": "operational", "flagged_claims": [],
                "error": None, "log_boundary": True, "log_flow_turns": ["Log row 60kg x 8"]}
    return route


# ── Recorded at once ──────────────────────────────────────────────────────────

def test_a_turn_is_in_history_while_it_is_being_answered(srv, monkeypatch):
    release = None

    async def route(msg):
        await release.wait()
        return {"answer": "done", "route": "analytical", "flagged_claims": [], "error": None}

    _install(srv, monkeypatch, route)

    async def scenario():
        nonlocal release
        release = asyncio.Event()
        turn = asyncio.create_task(srv._process_turn("how are my rows?"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        during = _body(await srv.history())["history"]
        release.set()
        await turn
        return during, _body(await srv.history())["history"]

    during, after = asyncio.run(scenario())
    assert [(m["role"], m["text"], m.get("pending")) for m in during] == [
        ("user", "how are my rows?", None), ("assistant", "", True)]
    assert _pairs(after) == [("user", "how are my rows?"), ("assistant", "done")]
    assert not any("pending" in m for m in after)


@pytest.mark.parametrize("route", ["analytical", "recall", "out_of_scope", "filler"])
def test_every_kind_of_answer_is_recorded_once(srv, monkeypatch, route):
    session = _install(srv, monkeypatch, _simple("the reply", route))
    asyncio.run(srv._process_turn("a question"))
    assert _pairs(session.chat_history) == [("user", "a question"), ("assistant", "the reply")]


def test_an_agent_that_records_its_own_turn_is_not_recorded_twice(srv, monkeypatch):
    async def route(msg):
        srv.session.chat_history += [{"role": "user", "text": msg},
                                     {"role": "assistant", "text": "op answer"}]
        return {"answer": "op answer", "route": "operational", "flagged_claims": [], "error": None}

    session = _install(srv, monkeypatch, route)
    asyncio.run(srv._process_turn("log it"))
    assert _pairs(session.chat_history) == [("user", "log it"), ("assistant", "op answer")]


# ── Errors are recorded as errors, and never left pending ─────────────────────

def test_a_quota_error_is_recorded_as_what_the_user_saw(srv, monkeypatch):
    async def route(msg):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    session = _install(srv, monkeypatch, route)
    resp = asyncio.run(srv._process_turn("plan me a week"))
    assert resp.status_code == 429
    shown = _body(resp)["message"]
    assert _pairs(session.chat_history) == [("user", "plan me a week"), ("assistant", shown)]
    assert session.chat_history[1].get("kind") == "error"
    assert "pending" not in session.chat_history[1]


def test_an_unexpected_failure_never_leaves_the_turn_pending(srv, monkeypatch):
    async def route(msg):
        return None                           # breaks the handler after routing

    session = _install(srv, monkeypatch, route)
    with pytest.raises(AttributeError):
        asyncio.run(srv._process_turn("anything"))
    last = session.chat_history[-1]
    assert last["role"] == "assistant" and "pending" not in last and last.get("kind") == "error"


# ── Panels, rejections, button actions ───────────────────────────────────────

def test_a_disambiguation_panel_leaves_only_the_prompt(srv, monkeypatch):
    async def route(msg):
        return {"answer": "fallback prose", "route": "operational", "flagged_claims": [],
                "error": None, "disambiguation": {"groups": [{"name": "row", "candidates": []}]}}

    session = _install(srv, monkeypatch, route)
    assert _body(asyncio.run(srv._process_turn("log row")))["type"] == "disambiguation_required"
    assert _pairs(session.chat_history) == [("user", "log row")]


def test_a_confirmation_panel_leaves_only_the_prompt(srv, monkeypatch):
    session = _install(srv, monkeypatch, _panel_route(srv))
    assert _body(asyncio.run(srv._process_turn("log row 60kg x 8")))["type"] == "confirmation_required"
    assert _pairs(session.chat_history) == [("user", "log row 60kg x 8")]


def test_a_turn_rejected_as_busy_records_nothing(srv, monkeypatch):
    session = _install(srv, monkeypatch, _simple("never"))

    async def scenario():
        async with srv.agent_lock:
            return await srv._process_turn("typed after reload")

    assert asyncio.run(scenario()).status_code == 429
    assert session.chat_history == []


def test_resume_records_its_reply_without_a_user_bubble(srv, monkeypatch):
    session = _install(srv, monkeypatch, _simple("There's no saved question to resume.", "none"))
    asyncio.run(srv.resume())
    assert _pairs(session.chat_history) == [("assistant", "There's no saved question to resume.")]


def test_a_logged_confirmation_is_recorded(srv, monkeypatch):
    session = _install(srv, monkeypatch, _panel_route(srv))
    asyncio.run(srv._process_turn("log row 60kg x 8"))
    body = _body(asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=True))))
    assert body["text"].startswith("✅")
    assert _pairs(session.chat_history) == [("user", "log row 60kg x 8"),
                                            ("assistant", body["text"])]


def test_a_cancelled_confirmation_is_recorded_once(srv, monkeypatch):
    session = _install(srv, monkeypatch, _panel_route(srv), agent_answer="Okay, cancelled.")
    asyncio.run(srv._process_turn("log row 60kg x 8"))
    asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=False)))
    assert _pairs(session.chat_history) == [("user", "log row 60kg x 8"),
                                            ("assistant", "Okay, cancelled.")]


# ── A turn is busy through its panel work ─────────────────────────────────────
#
# The panel work (slot read, staging-verify model call) runs INSIDE agent_lock.
# A second request there would start a turn whose first step discards the
# staged write still being checked — so it must be refused. This holds today;
# the test keeps it that way if the panel work is ever moved out of the lock.
# (A 2026-09-16 reading claimed the lock was released first. It was a misread
# indentation; this test is what settled it.)

def test_a_second_request_during_panel_work_is_busy(srv, monkeypatch):
    session = _install(srv, monkeypatch, _panel_route(srv))
    in_panel_work, release = None, None
    real_call_tool = session.call_tool

    async def call_tool(name, args):
        if name == "read_staged_workout_slot":
            in_panel_work.set()
            await release.wait()
        return await real_call_tool(name, args)

    session.call_tool = call_tool

    async def scenario():
        nonlocal in_panel_work, release
        in_panel_work, release = asyncio.Event(), asyncio.Event()
        first = asyncio.create_task(srv._process_turn("log row 60kg x 8"))
        await in_panel_work.wait()
        # Bounded: an ACCEPTED second turn would wait on this same panel work
        # forever — fail with a timeout instead of hanging the suite.
        second = await asyncio.wait_for(srv._process_turn("another question"), 5)
        confirm = await asyncio.wait_for(srv.confirm(srv.ConfirmRequest(confirmed=True)), 5)
        release.set()
        await first
        srv.coordinator.route = _simple("fine")
        after = await srv._process_turn("once it has ended")
        return second, confirm, after

    second, confirm, after = asyncio.run(scenario())
    assert second.status_code == 429 and _body(second) == {"error": "Agent is busy, please wait"}
    assert confirm.status_code == 429
    assert after.status_code == 200 and _body(after)["text"] == "fine"


# ── A quota stop that saved a checkpoint stays resumable after a reload ──────

def test_a_resumable_quota_stop_is_recorded_as_resumable(srv, monkeypatch):
    class QuotaStop(Exception):
        checkpoint_saved = True
        user_message = "Daily limit reached — your progress is saved."

    async def route(msg):
        raise QuotaStop("429 RESOURCE_EXHAUSTED")

    session = _install(srv, monkeypatch, route)
    resp = asyncio.run(srv._process_turn("plan me a week"))
    assert _body(resp)["checkpoint_saved"] is True
    entry = session.chat_history[-1]
    assert entry.get("kind") == "error" and entry.get("resumable") is True
    assert entry["text"] == "Daily limit reached — your progress is saved."


# ── A waiting confirmation panel can be restored after a reload ──────────────

def test_a_waiting_panel_is_restorable(srv, monkeypatch):
    _install(srv, monkeypatch, _panel_route(srv))
    panel = _body(asyncio.run(srv._process_turn("log row 60kg x 8")))
    restored = _body(asyncio.run(srv.pending_confirmation()))
    assert restored == {"preview": panel["preview"], "preview_source": panel["preview_source"]}


def test_no_panel_after_confirming(srv, monkeypatch):
    _install(srv, monkeypatch, _panel_route(srv))
    asyncio.run(srv._process_turn("log row 60kg x 8"))
    asyncio.run(srv.confirm(srv.ConfirmRequest(confirmed=True)))
    assert _body(asyncio.run(srv.pending_confirmation())) == {}


def test_no_panel_once_a_new_turn_starts(srv, monkeypatch):
    _install(srv, monkeypatch, _panel_route(srv))
    asyncio.run(srv._process_turn("log row 60kg x 8"))
    srv.coordinator.route = _simple("something else")
    asyncio.run(srv._process_turn("how are my rows?"))
    assert _body(asyncio.run(srv.pending_confirmation())) == {}


def test_no_panel_when_the_staging_check_fails(srv, monkeypatch):
    _install(srv, monkeypatch, _panel_route(srv), verdict="FAIL")
    asyncio.run(srv._process_turn("log row 60kg x 8"))
    assert _body(asyncio.run(srv.pending_confirmation())) == {}
