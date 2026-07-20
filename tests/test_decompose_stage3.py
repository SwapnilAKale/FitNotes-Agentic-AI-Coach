"""
tests/test_decompose_stage3.py — Decomposition Arc Stage 3: per-chunk
execution + ordered merge.

Contract under test:
  - a ≥2-chunk classify with MIXED lanes dispatches the decomposed executor;
    each chunk runs its OWN lane with its OWN intent_text, in index order;
  - the merged answer carries "### <intent>" headers in ask-order;
  - out_of_scope chunks answer inline (zero lane spend);
  - a write chunk re-arms log_flow_turns to ITS intent (verify Input A);
  - per-chunk clean-fail: one broken chunk never kills its siblings;
  - an ambiguous analytical chunk contributes its ask AND arms the Stage-2
    slot with the CHUNK's question (chunk-scoped resume);
  - chunk cap: at most _DECOMP_CHUNK_CAP executed + a notice line;
  - write-safety: regex-caught writes whose classify is non-decomposable or
    parse-failed land operational-whole (distrust override).

No Gemini — classify stubbed via the fake client (same kit as stage-1 tests).
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
from src.coordinator import (                           # noqa: E402
    Coordinator, OUT_OF_SCOPE_REFUSAL, _DECOMP_CHUNK_CAP,
    DataAgentIntegrityError,
)


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
                        finish_reason=None)
    ])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def _chunk(lane, intent, **over):
    entry = {"lane": lane, "intent_text": intent}
    entry.update(over)
    return entry


def _payload(chunks, route="analytical", **flat_over):
    flat = {
        "route": route, "display_intent": False, "exercise_names": None,
        "muscle_groups": None, "query_period_days": 90, "rep_target": None,
        "cardio_lock": None, "needs_custom_sql": False,
        "custom_sql_intent": None, "requests": chunks,
    }
    flat.update(flat_over)
    return json.dumps(flat)


def _spy_lanes(coord, monkeypatch):
    """Spy all three lanes; record (lane, question) in call order."""
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


# Exact, unambiguous name so the pre-resolution gate passes straight through
# to per-chunk dispatch (these tests exercise dispatch mechanics, not name
# disambiguation — that has its own tests in test_decompose_disambig.py).
MIXED = [
    _chunk("analytical", "Is my Lat Pulldown progressing?",
           exercise_names=["Lat Pulldown"]),
    _chunk("operational", "Log bench 100 lbs for 5 reps today"),
]


# ═══ Trigger + dispatch ══════════════════════════════════════════════════════

def test_mixed_lanes_dispatch_decomposed_in_order(coord, monkeypatch):
    coord._client = _client_returning(_payload(MIXED))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route(
        "is my squat progressing and log bench 100x5"))

    assert [c[0] for c in calls] == ["analytical", "operational"]
    # each lane received its OWN self-contained intent_text
    assert calls[0][1] == "Is my Lat Pulldown progressing?"
    assert calls[1][1] == "Log bench 100 lbs for 5 reps today"
    # the analytical chunk ran with flat-shaped chunk params, requests=None
    chunk_params = calls[0][2]
    assert chunk_params["route"] == "analytical"
    assert chunk_params["requests"] is None
    assert chunk_params["exercise_names"] == ["Lat Pulldown"]
    # merged in ask-order under headers
    a = result["answer"]
    assert "### Is my Lat Pulldown progressing?" in a
    assert "### Log bench 100 lbs for 5 reps today" in a
    assert a.index("AN(") < a.index("OP(")
    assert result["decomposed"] is True


def test_recall_and_out_of_scope_chunks(coord, monkeypatch):
    chunks = [
        _chunk("out_of_scope", "Write a poem about my bench press"),
        _chunk("recall", "What was the number you mentioned?"),
        _chunk("analytical", "How is my deadlift trending?"),
    ]
    coord._client = _client_returning(_payload(chunks))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("poem and number and deadlift"))

    # out_of_scope answered inline — no lane call for it
    assert [c[0] for c in calls] == ["recall", "analytical"]
    a = result["answer"]
    assert OUT_OF_SCOPE_REFUSAL in a
    assert a.index(OUT_OF_SCOPE_REFUSAL) < a.index("RC(") < a.index("AN(")


def test_single_lane_multi_chunk_does_not_decompose(coord, monkeypatch):
    chunks = [
        _chunk("operational", "Log bench 100x5"),
        _chunk("operational", "Delete my deadlift goal"),
    ]
    coord._client = _client_returning(_payload(chunks, route="operational"))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("log bench and delete goal question"))

    # uniform lanes → flat dispatch, ONE operational run with the whole message
    assert [c[0] for c in calls] == ["operational"]
    assert result.get("decomposed", False) is False


# ═══ Write chunk mechanics ═══════════════════════════════════════════════════

def test_write_chunk_rearms_flow_turns_and_fallback(coord, monkeypatch):
    coord._client = _client_returning(_payload(MIXED))
    captured = {}

    async def an(q, p, resume=None):
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):
        captured["q"] = q
        captured["kw"] = kw
        captured["flow_turns"] = list(coord._log_flow_turns)
        return "OP"
    monkeypatch.setattr(coord, "_run_operational", op)

    result = asyncio.run(coord.route(
        "is my squat progressing and log bench 100 lbs for 5 reps"))

    # verify Input A = the chunk's own restatement, not the compound message
    assert captured["flow_turns"] == ["Log bench 100 lbs for 5 reps today"]
    assert captured["kw"].get("fallback_write") is True
    # envelope carries the WRITE CHUNK's flow turns (fallback_write armed)
    assert result["log_flow_turns"] == ["Log bench 100 lbs for 5 reps today"]


def test_nonwrite_merge_excludes_staged_write_part(coord, monkeypatch):
    # #19: the write chunk's part is staging-time text ("staged, needs
    # confirmation"). `answer` (full) carries both parts; the separate
    # `decomposed_nonwrite_answer` — what the panel stash / CLI finalize
    # prepend to the "✅ logged" line — carries the analytical part ONLY, so a
    # committed write is never re-described as unsaved.
    coord._client = _client_returning(_payload(MIXED))
    staged_text = "I've staged your workout — it isn't saved yet, please confirm."

    async def an(q, p, resume=None):
        return "Your Lat Pulldown is up 5%.", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):
        return staged_text
    monkeypatch.setattr(coord, "_run_operational", op)

    result = asyncio.run(coord.route(
        "is my lat pulldown progressing and log bench 100 lbs for 5 reps"))

    # full answer keeps both (used only on the no-panel path)
    assert "Your Lat Pulldown is up 5%." in result["answer"]
    assert staged_text in result["answer"]
    # write-excluded merge: analytical kept, staging text gone
    nonwrite = result["decomposed_nonwrite_answer"]
    assert "Your Lat Pulldown is up 5%." in nonwrite
    assert staged_text not in nonwrite
    assert "isn't saved yet" not in nonwrite
    assert "staged" not in nonwrite.lower()


def test_nonwrite_merge_equals_answer_when_no_write_chunk(coord, monkeypatch):
    # Mirror negative: a decomposed turn with NO write chunk excludes nothing —
    # the write-excluded merge is identical to the full answer.
    coord._client = _client_returning(_payload([
        _chunk("analytical", "Is my squat progressing?"),
        _chunk("recall", "What was that number?"),
    ]))
    _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("squat and that number"))

    assert result["decomposed_nonwrite_answer"] == result["answer"]


def test_write_safety_distrust_override_single_chunk(coord, monkeypatch):
    # Regex fires; classifier claims ONE analytical request → operational-whole.
    single = [_chunk("analytical", "Log-ish looking but classifier disagrees")]
    coord._client = _client_returning(_payload(single, route="analytical"))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("log my bench press 100x5"))

    assert [c[0] for c in calls] == ["operational"]
    assert calls[0][1] == "log my bench press 100x5"      # whole message
    assert result["route"] == "operational"


def test_write_safety_parse_fail_never_unparseable(coord, monkeypatch):
    coord._client = _client_returning("{broken json")
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("log my bench press 100x5"))

    assert result["route"] == "operational"               # never "unparseable"
    assert [c[0] for c in calls] == ["operational"]


# ═══ Per-chunk clean-fail + ambiguity + cap ══════════════════════════════════

def test_chunk_integrity_failure_spares_siblings(coord, monkeypatch):
    coord._client = _client_returning(_payload([
        _chunk("analytical", "Is my squat progressing?"),
        _chunk("recall", "What was that number?"),
    ]))

    class _V:
        invariant_id = "B9"
        message = "boom"

    async def an(q, p, resume=None):
        raise DataAgentIntegrityError([_V()])
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def rc(q):
        return "RC-OK"
    monkeypatch.setattr(coord, "_run_recall", rc)

    result = asyncio.run(coord.route("squat and that number"))
    a = result["answer"]
    assert "integrity check failed (B9)" in a
    assert "RC-OK" in a                                    # sibling survived
    assert a.index("B9") < a.index("RC-OK")                # order kept


def test_ambiguous_name_arms_full_turn_slot_before_any_chunk_runs(coord, monkeypatch):
    # Pre-resolution: an ambiguous exercise name (real DB: "squat" → 5 variants)
    # arms ONE full-turn slot and NO chunk executes — the whole turn is held
    # until the panel resolves the name (so a sibling write is never lost).
    coord._client = _client_returning(_payload([
        _chunk("analytical", "Is my squat progressing?",
               exercise_names=["squat"]),
        _chunk("recall", "What was that figure?"),
    ]))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("squat progress and that figure"))

    # nothing dispatched — the gate short-circuited before the chunk loop
    assert calls == []
    # full-turn slot: question is the ORIGINAL message, not a chunk intent,
    # and params keep the whole requests array (resumable to both chunks)
    slot = coord._pending_decomposition
    assert slot["question"] == "squat progress and that figure"
    assert len(slot["params"]["requests"]) == 2
    assert len(slot["groups"]) == 1
    assert slot["groups"][0]["name"] == "squat"
    assert "Sumo Squats" in slot["groups"][0]["candidates"]
    # structured payload surfaced for the server to raise the panel
    assert result["disambiguation"]["groups"][0]["name"] == "squat"
    assert "clarifying" in result["answer"].lower()


def test_chunk_cap_notice(coord, monkeypatch):
    chunks = ([_chunk("analytical", f"Part {i}?") for i in range(4)]
              + [_chunk("recall", "Part 5?")])
    coord._client = _client_returning(_payload(chunks))
    calls = _spy_lanes(coord, monkeypatch)

    result = asyncio.run(coord.route("five parts"))

    assert len(calls) == _DECOMP_CHUNK_CAP                 # 5th never ran
    assert f"first {_DECOMP_CHUNK_CAP} parts" in result["answer"]
