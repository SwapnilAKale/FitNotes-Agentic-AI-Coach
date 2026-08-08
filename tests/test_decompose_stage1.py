"""
tests/test_decompose_stage1.py — Decomposition Arc Stage 1 (emit-inert).

The classifier additionally emits params["requests"] — one entry per distinct
request in the message (index / lane / intent_text / per-chunk parameter
fields). Stage 1 contract under test:

  - requests is ALWAYS either None or a fully valid non-empty list — a
    partially valid array is dropped whole (never half-trusted).
  - The flat fields keep whole-message semantics and are NEVER touched by the
    sanitizer, even when requests is malformed.
  - Divergence between flat fields and the chunk array is LOG-ONLY.
  - Nothing consumes requests: routing behaves identically with or without
    it, and it rides through params to the analytical stage unchanged.

No Gemini — the classify client is stubbed (same kit as test_routing_step_c).
No server.
"""

import asyncio
import json
import logging
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
    Coordinator, _CLASSIFY_SYSTEM, _CHUNK_PARAM_DEFAULTS,
)


# ── Fixtures / stubs (same kit as test_routing_step_c) ────────────────────────

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


def _client_raising(exc):
    def boom(**kw):
        raise exc
    return SimpleNamespace(models=SimpleNamespace(generate_content=boom))


FLAT = {
    "route": "analytical",
    "display_intent": False,
    "exercise_names": ["Lat Pulldown"],
    "muscle_groups": None,
    "query_period_days": 90,
    "rep_target": None,
    "cardio_lock": None,
    "needs_custom_sql": False,
    "custom_sql_intent": None,
}


def _chunk(lane, intent, **over):
    entry = {"lane": lane, "intent_text": intent}
    entry.update(over)
    return entry


def _classify_json(coord, payload):
    coord._client = _client_returning(json.dumps(payload))
    return asyncio.run(coord._classify("stub question"))


# ── Parse-path tests ──────────────────────────────────────────────────────────

def test_multi_chunk_requests_parsed(coord):
    payload = dict(FLAT)
    payload["exercise_names"] = ["Lat Pulldown", "Flat Dumbbell Bench Press"]
    payload["requests"] = [
        _chunk("analytical", "Is my Lat Pulldown progressing?",
               index=5, exercise_names=["Lat Pulldown"]),
        _chunk("operational", "Log Flat Dumbbell Bench Press 100 lbs x 5 today",
               index=7, exercise_names=["Flat Dumbbell Bench Press"]),
    ]
    params = _classify_json(coord, payload)
    reqs = params["requests"]
    assert isinstance(reqs, list) and len(reqs) == 2
    # list order is authoritative — the model's index values are overwritten
    assert [c["index"] for c in reqs] == [0, 1]
    assert reqs[0]["lane"] == "analytical"
    assert reqs[1]["lane"] == "operational"
    # per-chunk defaults filled for keys the model omitted
    for c in reqs:
        for key, default_val in _CHUNK_PARAM_DEFAULTS.items():
            assert key in c
        assert c["query_period_days"] == 90
    # flat fields intact
    assert params["route"] == "analytical"
    assert params["exercise_names"] == ["Lat Pulldown", "Flat Dumbbell Bench Press"]
    assert "_parse_failed" not in params


def test_single_chunk_mirrors_flat(coord, caplog):
    payload = dict(FLAT)
    payload["requests"] = [
        _chunk("analytical", "How is my Lat Pulldown progressing?",
               exercise_names=["Lat Pulldown"]),
    ]
    with caplog.at_level(logging.INFO):
        params = _classify_json(coord, payload)
    assert len(params["requests"]) == 1
    assert params["requests"][0]["index"] == 0
    assert params["requests"][0]["exercise_names"] == ["Lat Pulldown"]
    # consistent single chunk → no divergence warning
    assert not [r for r in caplog.records if "divergence" in r.getMessage()]


def test_missing_requests_defaults_none(coord):
    # Legacy flat-only model output (e.g. the model ignores the new schema):
    # behavior must be exactly today's.
    params = _classify_json(coord, dict(FLAT))
    assert params["requests"] is None
    for key, val in FLAT.items():
        assert params[key] == val
    assert "_parse_failed" not in params


@pytest.mark.parametrize("bad_requests", [
    "two things",                                        # not a list
    {"lane": "analytical"},                              # dict, not list
    [],                                                  # empty list
    ["not a dict"],                                      # non-dict entry
    [_chunk("bogus_lane", "do a thing")],                # invalid lane
    [_chunk("analytical", "")],                          # empty intent_text
    [{"lane": "analytical"}],                            # missing intent_text
    [_chunk("analytical", "fine"),
     _chunk("operational", "")],                         # one bad entry poisons all
])
def test_malformed_requests_dropped_flat_survives(coord, bad_requests):
    payload = dict(FLAT)
    payload["requests"] = bad_requests
    params = _classify_json(coord, payload)
    assert params["requests"] is None          # whole-array drop, never partial
    for key, val in FLAT.items():
        assert params[key] == val              # flat fields untouched
    assert "_parse_failed" not in params       # a bad inert extra never fails classify


def test_chunk_setdefaults(coord):
    payload = dict(FLAT)
    payload["requests"] = [_chunk("analytical", "How consistent have I been?")]
    params = _classify_json(coord, payload)
    entry = params["requests"][0]
    for key, default_val in _CHUNK_PARAM_DEFAULTS.items():
        assert entry[key] == default_val


def test_truncated_json_falls_to_parse_failed_default(coord):
    # Output cut mid-array (the failure mode the 1024-token cap makes rare):
    # whole classify degrades to the _parse_failed analytical default.
    full = dict(FLAT)
    full["requests"] = [_chunk("analytical", "How is my Lat Pulldown progressing?")]
    raw = json.dumps(full)[:-40]
    coord._client = _client_returning(raw)
    params = asyncio.run(coord._classify("stub"))
    assert params["_parse_failed"] is True
    assert params["route"] == "analytical"
    assert params["requests"] is None


def test_divergence_logged_not_acted_on(coord, caplog):
    # Lane conflict: flat route analytical, sole chunk operational.
    payload = dict(FLAT)
    payload["requests"] = [
        _chunk("operational", "Log bench 100 lbs x 5 today",
               exercise_names=["Bench Press"]),
    ]
    with caplog.at_level(logging.WARNING):
        params = _classify_json(coord, payload)
    msgs = [r.getMessage() for r in caplog.records if "divergence" in r.getMessage()]
    assert any("route" in m for m in msgs)
    assert any("exercise_names" in m for m in msgs)
    # log-only: nothing mutated or dropped
    assert params["route"] == "analytical"
    assert params["requests"][0]["lane"] == "operational"
    assert params["requests"][0]["exercise_names"] == ["Bench Press"]
    assert "_parse_failed" not in params


def test_exception_default_includes_requests_none(coord):
    coord._client = _client_raising(RuntimeError("classify boom"))
    params = asyncio.run(coord._classify("how's my back"))
    assert params["_parse_failed"] is True
    assert params["requests"] is None


def test_prompt_integrity_smoke():
    # Guard against accidental truncation of the tuned prompt during the
    # Stage-1 edit: legacy anchors AND the new decomposition anchors.
    for anchor in ("out_of_scope", "MEDICAL", "RECALL —",
                   "Return ONLY valid JSON", "PARAMETER EXTRACTION",
                   "CUSTOM SQL"):
        assert anchor in _CLASSIFY_SYSTEM, f"legacy anchor missing: {anchor}"
    for anchor in ('"requests"', "intent_text", '"lane"'):
        assert anchor in _CLASSIFY_SYSTEM, f"new anchor missing: {anchor}"


# ── Routing behavior (Stage 3 superseded the Stage-1 inertness contract) ─────

def test_all_analytical_requests_stay_single_run(coord, monkeypatch):
    # DELIBERATE non-decompose case: a multi-chunk array whose lanes are all
    # analytical stays ONE analytical run (live-proven good; one pipeline is
    # cheaper than two). The array still rides through in params.
    payload = dict(FLAT)
    payload["requests"] = [
        _chunk("analytical", "Is my Lat Pulldown progressing?",
               exercise_names=["Lat Pulldown"]),
        _chunk("analytical", "Show me my last Lat Pulldown session.",
               exercise_names=["Lat Pulldown"], display_intent=True),
    ]
    coord._client = _client_returning(json.dumps(payload))

    seen = {"analytical": 0, "operational": 0, "params": None}

    async def an(q, p, resume=None):
        seen["analytical"] += 1
        seen["params"] = p
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):
        seen["operational"] += 1
        return "OP"
    monkeypatch.setattr(coord, "_run_operational", op)

    result = asyncio.run(coord.route("lat pulldown progress and last session"))
    assert result["route"] == "analytical"
    assert seen["analytical"] == 1
    assert seen["operational"] == 0
    assert len(seen["params"]["requests"]) == 2


def test_entry_boundary_log_synthesizes_write_regex_defers(coord):
    # /log boundary: still deterministic, classify-free synthetic params.
    out = coord._node_entry_boundary({"question": "/log bench 100x5"})
    assert out["params"] is not None
    assert out["params"]["route"] == "operational"
    assert "requests" not in out["params"]
    assert not out.get("write_intent_hint")

    # Regex write (Stage 3): params deferred to classify; hint armed so the
    # distrust override can land it operational if it doesn't decompose.
    out = coord._node_entry_boundary({"question": "log my bench press 100x5"})
    assert out["params"] is None
    assert out["write_intent_hint"] is True
    assert out["fallback_write"] is True
