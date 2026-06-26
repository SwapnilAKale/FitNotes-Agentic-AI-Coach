"""
6b — question→PR-parameter wiring: classifier schema (rep_target / cardio_lock),
deterministic unit normalization, coordinator threading, and the package fields
(pr_repfloor / pr_cardio_locked). No Gemini (classifier client stubbed), no server.
"""

import asyncio
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

from src import coordinator as coordinator_mod                 # noqa: E402
from src.coordinator import Coordinator, _normalize_cardio_lock  # noqa: E402
from src.data_agent import prepare_analysis_package            # noqa: E402


# ── Fixtures / stubs (mirror tests/test_routing_step_c.py) ───────────────────────

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


# ── Unit normalization (pure, deterministic — NOT the LLM) ───────────────────────

def test_normalize_duration_min_to_seconds():
    assert _normalize_cardio_lock({"field": "duration", "value": 10, "unit": "min"}) \
        == {"field": "duration", "value": 600}


def test_normalize_distance_km_passthrough():
    assert _normalize_cardio_lock({"field": "distance", "value": 5, "unit": "km"}) \
        == {"field": "distance", "value": 5}


def test_normalize_distance_other_units():
    assert _normalize_cardio_lock({"field": "distance", "value": 800, "unit": "m"}) \
        == {"field": "distance", "value": 0.8}


def test_normalize_none_and_invalid():
    assert _normalize_cardio_lock(None) is None
    assert _normalize_cardio_lock({"field": None, "value": None, "unit": None}) is None
    assert _normalize_cardio_lock({"field": "distance", "value": None}) is None
    assert _normalize_cardio_lock({"field": "bogus", "value": 5}) is None


# ── Classifier extraction (stubbed model JSON, no Gemini) ────────────────────────

def test_classify_extracts_rep_target(coord):
    coord._client = _client_returning(
        '{"route":"analytical","exercise_names":["Bench Press"],"muscle_groups":null,'
        '"query_period_days":90,"rep_target":5,"cardio_lock":null,'
        '"needs_custom_sql":false,"custom_sql_intent":null}')
    params = asyncio.run(coord._classify("what's my 5-rep bench PR"))
    assert params["rep_target"] == 5
    assert params["cardio_lock"] is None


def test_classify_extracts_cardio_lock(coord):
    coord._client = _client_returning(
        '{"route":"analytical","exercise_names":["Walking"],"muscle_groups":null,'
        '"query_period_days":null,"rep_target":null,'
        '"cardio_lock":{"field":"duration","value":10,"unit":"min"},'
        '"needs_custom_sql":false,"custom_sql_intent":null}')
    params = asyncio.run(coord._classify("most distance in 10 minutes walking"))
    assert params["cardio_lock"] == {"field": "duration", "value": 10, "unit": "min"}
    assert params["rep_target"] is None


def test_classify_defaults_absent_targets(coord):
    # A model that omits the new fields entirely → defaults to None (both → default PR).
    coord._client = _client_returning(
        '{"route":"analytical","exercise_names":["Bench Press"],"muscle_groups":null,'
        '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}')
    params = asyncio.run(coord._classify("what's my bench PR"))
    assert params["rep_target"] is None
    assert params["cardio_lock"] is None


# ── Coordinator threading boundary (capture prepare_analysis_package kwargs) ──────

def test_coordinator_threads_normalized_params(coord, monkeypatch):
    captured = {}

    def fake_prep(**kw):
        captured.update(kw)
        raise RuntimeError("STOP-after-capture")     # short-circuit before any LLM stage

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package", fake_prep)
    params = {
        "route": "analytical", "exercise_names": None, "muscle_groups": None,
        "query_period_days": 90, "rep_target": 5,
        "cardio_lock": {"field": "duration", "value": 10, "unit": "min"},
        "needs_custom_sql": False, "custom_sql_intent": None,
    }
    with pytest.raises(RuntimeError, match="STOP-after-capture"):
        asyncio.run(coord._run_analytical("5-rep PR; most distance in 10 min", params))
    assert captured.get("reps_floor") == 5
    assert captured.get("cardio_lock") == {"field": "duration", "value": 600}


# ── End-to-end package fields (direct prepare_analysis_package, no LLM) ───────────

def test_package_pr_repfloor_present_and_static_pr_unchanged():
    base = next(e for e in prepare_analysis_package(
        query_period_days=None, exercise_names=["Lat Pulldown"])["exercises"]
        if e["name"] == "Lat Pulldown")
    assert "pr_repfloor" not in base                      # absent without a target
    ex = next(e for e in prepare_analysis_package(
        query_period_days=None, exercise_names=["Lat Pulldown"], reps_floor=5)["exercises"]
        if e["name"] == "Lat Pulldown")
    assert ex["pr"] == base["pr"]                         # static PR unchanged
    assert isinstance(ex["pr_repfloor"], dict)
    assert ex["pr_repfloor"]["reps"] >= 5 and ex["pr_repfloor"]["reps_floor"] == 5


def test_package_pr_repfloor_null_when_no_qualifying_set():
    ex = next(e for e in prepare_analysis_package(
        query_period_days=None, exercise_names=["Lat Pulldown"], reps_floor=99)["exercises"]
        if e["name"] == "Lat Pulldown")
    assert "pr_repfloor" in ex and ex["pr_repfloor"] is None   # present + null, not error


def test_package_pr_cardio_locked_present():
    base = next(e for e in prepare_analysis_package(
        query_period_days=None, exercise_names=["Walking"])["exercises"]
        if e["name"] == "Walking")
    assert "pr_cardio_locked" not in base                # absent without a lock
    ex = next(e for e in prepare_analysis_package(
        query_period_days=None, exercise_names=["Walking"],
        cardio_lock={"field": "duration", "value": 600})["exercises"]
        if e["name"] == "Walking")
    assert "pr" in ex                                    # default cardio PR still present
    assert isinstance(ex["pr_cardio_locked"], dict)
    assert ex["pr_cardio_locked"]["lock"] == "duration"
