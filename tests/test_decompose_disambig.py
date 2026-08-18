"""
tests/test_decompose_disambig.py — up-front name resolution + the structured
disambiguation panel for decomposed turns.

Contract under test (the fix for the run-2 "lost write + carry hijack" bug):
  - a decomposed turn with ANY ambiguous exercise name resolves every name
    BEFORE any chunk runs: nothing dispatches, one full-turn slot is armed
    holding the whole `requests` array + all ambiguous groups;
  - the panel's structured picks patch every name into BOTH channels
    (exercise_names for the read lane; intent_text for the write lane) and the
    WHOLE turn resumes — analytical + write together;
  - "Other" free text re-resolves at the group's permissiveness (strict for a
    write); a still-ambiguous "Other" re-prompts just that group;
  - the /log carry never arms across a decomposed turn (no next-message hijack);
  - server: /disambiguate shares the confirm-panel tail, /pending-disambiguation
    exposes the live slot, /disambiguate/cancel clears it.

Real resolver against the bundled DB (partial names "squat"/"bench press" are
genuinely ambiguous); no Gemini — lanes are stubbed.
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
from src.coordinator import Coordinator                 # noqa: E402


@pytest.fixture()
def coord(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _client_returning(text):
    resp = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                        finish_reason=None)])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def _chunk(lane, intent, **over):
    entry = {"lane": lane, "intent_text": intent}
    entry.update(over)
    return entry


def _payload(chunks, route="analytical", **flat_over):
    flat = {"route": route, "display_intent": False, "exercise_names": None,
            "muscle_groups": None, "query_period_days": 90, "rep_target": None,
            "cardio_lock": None, "needs_custom_sql": False,
            "custom_sql_intent": None, "requests": chunks}
    flat.update(flat_over)
    return json.dumps(flat)


def _spy_lanes(coord, monkeypatch):
    calls: list = []

    async def an(q, p, resume=None):
        calls.append(("analytical", q, p))
        return f"AN({q})", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):
        calls.append(("operational", q, kw))
        return f"OP({q})"
    monkeypatch.setattr(coord, "_run_operational", op)

    async def rc(q):
        calls.append(("recall", q, None))
        return f"RC({q})"
    monkeypatch.setattr(coord, "_run_recall", rc)
    return calls


# Both names partial → ambiguous in the bundled DB.
AMBIG = [
    _chunk("analytical", "Is my squat progressing?", exercise_names=["squat"]),
    _chunk("operational", "Log bench press 100 lbs for 5 reps today",
           exercise_names=["bench press"]),
]


# ═══ Pre-resolution: hold the whole turn, run nothing ════════════════════════

def test_two_ambiguous_names_arm_two_groups_and_dispatch_nothing(coord, monkeypatch):
    coord._client = _client_returning(_payload(AMBIG))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route(
        "is my squat progressing and log bench press 100x5"))

    assert calls == []                                   # nothing ran
    slot = coord._pending_decomposition
    assert len(slot["params"]["requests"]) == 2          # full turn held
    names = [g["name"] for g in slot["groups"]]
    assert names == ["squat", "bench press"]             # ask order
    # write group resolved STRICT (no auto-pick) — real bench candidates offered
    bench = next(g for g in slot["groups"] if g["name"] == "bench press")
    assert "Flat Dumbbell Bench Press" in bench["candidates"]
    # structured payload surfaced for the server panel
    assert [g["name"] for g in result["disambiguation"]["groups"]] == \
        ["squat", "bench press"]


def test_one_ambiguous_one_exact_arms_single_group(coord, monkeypatch):
    chunks = [
        _chunk("analytical", "Is my Lat Pulldown progressing?",
               exercise_names=["Lat Pulldown"]),           # exact → resolves
        _chunk("operational", "Log bench press 100 lbs for 5 reps today",
               exercise_names=["bench press"]),             # ambiguous
    ]
    coord._client = _client_returning(_payload(chunks))
    _spy_lanes(coord, monkeypatch)

    asyncio.run(coord.route("compound"))

    slot = coord._pending_decomposition
    assert [g["name"] for g in slot["groups"]] == ["bench press"]


# ═══ Structured resolve → resume the whole turn ══════════════════════════════

def test_panel_picks_resume_both_lanes_with_resolved_names(coord, monkeypatch):
    coord._client = _client_returning(_payload(AMBIG))
    calls = _spy_lanes(coord, monkeypatch)
    asyncio.run(coord.route("compound"))                 # arm
    assert calls == []

    result = asyncio.run(coord.resolve_disambiguation([
        {"name": "squat", "choice": "Sumo Squats"},
        {"name": "bench press", "choice": "Flat Dumbbell Bench Press"},
    ]))

    # both lanes ran, in order, with the EXACT names bound into their intents
    assert [c[0] for c in calls] == ["analytical", "operational"]
    assert "Sumo Squats" in calls[0][1]                  # analytical intent rewritten
    assert "Flat Dumbbell Bench Press" in calls[1][1]    # write intent rewritten
    # the write chunk's exercise_names also patched (belt + suspenders)
    assert calls[0][2]["exercise_names"] == ["Sumo Squats"]
    # slot cleared, carry never armed → next message can't be hijacked
    assert coord._pending_decomposition is None
    assert coord._pending_log_carry is False
    assert result.get("decomposed") is True


def test_other_free_text_clean_match_is_used(coord, monkeypatch):
    coord._client = _client_returning(_payload(AMBIG))
    calls = _spy_lanes(coord, monkeypatch)
    asyncio.run(coord.route("compound"))

    # squat via a listed pick; bench press via "Other" naming an exact variant
    asyncio.run(coord.resolve_disambiguation([
        {"name": "squat", "choice": "Sumo Squats"},
        {"name": "bench press", "choice": "__other__",
         "other_text": "Incline Barbell Bench Press"},
    ]))

    assert "Incline Barbell Bench Press" in calls[1][1]
    assert coord._pending_decomposition is None


def test_other_still_ambiguous_reprompts_only_that_group(coord, monkeypatch):
    coord._client = _client_returning(_payload(AMBIG))
    calls = _spy_lanes(coord, monkeypatch)
    asyncio.run(coord.route("compound"))

    # squat resolved; bench press "Other" = another PARTIAL name → still ambiguous
    result = asyncio.run(coord.resolve_disambiguation([
        {"name": "squat", "choice": "Sumo Squats"},
        {"name": "bench press", "choice": "__other__",
         "other_text": "dumbbell bench"},
    ]))

    # nothing dispatched — the turn is still held, re-prompting the one group
    assert calls == []
    assert result["disambiguation"] is not None
    groups = result["disambiguation"]["groups"]
    assert len(groups) == 1
    assert coord._pending_decomposition is not None      # slot retained


def test_resolve_with_no_live_slot_returns_none(coord):
    assert asyncio.run(coord.resolve_disambiguation([])) is None


def test_cancel_clears_slot_and_carry(coord, monkeypatch):
    coord._client = _client_returning(_payload(AMBIG))
    _spy_lanes(coord, monkeypatch)
    asyncio.run(coord.route("compound"))
    assert coord._pending_decomposition is not None

    assert coord.cancel_disambiguation() is True
    assert coord._pending_decomposition is None
    assert coord._pending_log_carry is False


# ═══ Server endpoints ════════════════════════════════════════════════════════

@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod, "session", None)
    monkeypatch.setattr(server_mod._ckpt, "save_checkpoint", lambda **kw: None)
    monkeypatch.setattr(server_mod._ckpt, "clear_staged_checkpoint", lambda: None)
    server_mod._state.update({"pending_confirmation": False, "allow_execute": False,
                              "staging_preview": "", "confirmation_preview": "",
                              "pending_execute_kind": None, "decomposed_answer": ""})
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


_SLOT_REAL = json.dumps({"staged_workouts": [
    {"exercise_id": 1, "exercise_name": "Flat Dumbbell Bench Press",
     "date": "2026-07-16", "sets": [{"metric_weight": 45.3592, "reps": 5, "unit": 0}]}]})
_PREVIEW = "Flat Dumbbell Bench Press — 2026-07-16\n  100 lbs × 5 reps"


def test_disambiguate_endpoint_resolves_write_to_confirmation(srv, monkeypatch):
    """/disambiguate shares _process_turn's tail: a resolved write chunk lands
    on the confirm panel, with the write-EXCLUDED merge stashed (#12/#19) — the
    staging text never rides into the post-confirm reply."""
    nonwrite = "### Is my squat progressing?\n\nUp 4%."
    merged = nonwrite + "\n\n### Log bench\n\n⚠️ That isn't saved yet, please confirm."

    async def resolve_disambiguation(selections):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": merged, "decomposed_nonwrite_answer": nonwrite,
                "route": "analytical", "decomposed": True,
                "flagged_claims": [], "error": None,
                "log_flow_turns": ["Log Flat Dumbbell Bench Press 100 lbs x5"],
                "resolved_question": "compound"}

    async def verify_log_staging(flow, slot_json, preview):
        return {"verdict": "PASS", "reason": ""}

    async def call_tool(name, args):
        if name == "read_staged_workout_slot":
            return _SLOT_REAL
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": _PREVIEW})
        return json.dumps({"ok": True})

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        resolve_disambiguation=resolve_disambiguation,
        verify_log_staging=verify_log_staging, _pending_log_carry=False))
    monkeypatch.setattr(srv, "session", SimpleNamespace(
        call_tool=call_tool, chat_history=[], note_host_write=lambda *a, **k: None))

    body = _body(asyncio.run(srv.disambiguate(
        srv.DisambiguateRequest(selections=[
            srv.DisambiguateSelection(name="squat", choice="Sumo Squats"),
            srv.DisambiguateSelection(name="bench press",
                                      choice="Flat Dumbbell Bench Press")]))))

    assert body["type"] == "confirmation_required"
    assert body["preview"] == _PREVIEW
    # #19: the stash holds the write-excluded merge, not the staging text.
    assert srv._state["decomposed_answer"] == nonwrite
    assert "isn't saved yet" not in srv._state["decomposed_answer"]


def test_disambiguate_endpoint_reprompts_when_unresolved(srv, monkeypatch):
    async def resolve_disambiguation(selections):
        return {"answer": "", "route": "analytical", "flagged_claims": [],
                "error": None,
                "disambiguation": {"groups": [
                    {"name": "bench press", "candidates": ["Flat Dumbbell Bench Press"]}]}}

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        resolve_disambiguation=resolve_disambiguation))

    body = _body(asyncio.run(srv.disambiguate(
        srv.DisambiguateRequest(selections=[
            srv.DisambiguateSelection(name="bench press", choice="__other__",
                                      other_text="dumbbell bench")]))))

    assert body["type"] == "disambiguation_required"
    assert body["groups"][0]["name"] == "bench press"


def test_pending_disambiguation_endpoint(srv, monkeypatch):
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        _disambiguation_payload=lambda: {"groups": [
            {"name": "squat", "candidates": ["Sumo Squats"]}]}))
    body = _body(asyncio.run(srv.pending_disambiguation()))
    assert body["groups"][0]["name"] == "squat"

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        _disambiguation_payload=lambda: None))
    assert _body(asyncio.run(srv.pending_disambiguation()))["groups"] is None


def test_disambiguate_cancel_endpoint(srv, monkeypatch):
    cleared = {"v": False}

    def cancel_disambiguation():
        cleared["v"] = True
        return True

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(
        cancel_disambiguation=cancel_disambiguation))
    body = _body(asyncio.run(srv.disambiguate_cancel()))
    assert body["type"] == "answer"
    assert cleared["v"] is True
