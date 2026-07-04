"""
Step A routing: OUT_OF_SCOPE gate (refuse non-fitness at classification, no
tool/agent/package/analysis spend), medical carve-out (medical is NOT
out_of_scope), and coach-character / medical-line prompt presence.

No Gemini — the classifier LLM is stubbed; we assert the routing DECISION and
that out_of_scope short-circuits before any downstream call.
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

from src import coordinator as coordinator_mod          # noqa: E402
from src.coordinator import Coordinator, OUT_OF_SCOPE_REFUSAL  # noqa: E402


# ── Coordinator with a stubbed classifier returning a chosen route ──────────

@pytest.fixture()
def coord(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)
    # No checkpoint slot interference
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _stub_classify(coord, monkeypatch, route, **extra):
    async def fake(question):
        return {"route": route, "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False,
                "custom_sql_intent": None, **extra}
    monkeypatch.setattr(coord, "_classify", fake)


def _spy_downstream(coord, monkeypatch):
    """Patch every downstream worker with a counter — out_of_scope must hit none."""
    seen = {"analytical": 0, "operational": 0, "pkg": 0}

    async def an(q, p, resume=None):
        seen["analytical"] += 1
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):               # **kw: /log boundary flags (ignored here)
        seen["operational"] += 1
        return "OP"
    monkeypatch.setattr(coord, "_run_operational", op)

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: seen.__setitem__("pkg", seen["pkg"] + 1) or {})
    return seen


# ── OUT_OF_SCOPE routing + no-spend proof ────────────────────────────────────

def test_out_of_scope_routes_and_spends_nothing(coord, monkeypatch):
    _stub_classify(coord, monkeypatch, "out_of_scope")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("write a 100 word essay on the Olympics"))
    assert result["route"] == "out_of_scope"
    assert result["answer"] == OUT_OF_SCOPE_REFUSAL
    assert result["error"] is None
    # NO package build, NO agent turn, NO analysis call
    assert seen == {"analytical": 0, "operational": 0, "pkg": 0}


@pytest.mark.parametrize("q", [
    "write a 100 word essay on the Olympics",
    "write me a python script",
    "write an SQL query to count rows",
    "who is the prime minister of France",
    "what does ad-hoc mean",
    "write a motivational poem for leg day",
    "100 ways to cook chicken",
])
def test_out_of_scope_examples_short_circuit(coord, monkeypatch, q):
    # The classifier decides out_of_scope; route() must short-circuit to refusal.
    _stub_classify(coord, monkeypatch, "out_of_scope")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "out_of_scope"
    assert seen["analytical"] == 0 and seen["operational"] == 0 and seen["pkg"] == 0


# ── IN-SCOPE examples still route normally (NOT out_of_scope) ────────────────

def test_in_scope_analytical_runs_pipeline(coord, monkeypatch):
    _stub_classify(coord, monkeypatch, "analytical")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("what is progressive overload"))
    assert result["route"] == "analytical"
    assert seen["analytical"] == 1            # pipeline ran, not refused


def test_in_scope_operational_runs(coord, monkeypatch):
    _stub_classify(coord, monkeypatch, "operational")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("best way to cook chicken keeping protein high"))
    assert result["route"] == "operational"
    assert seen["operational"] == 1


def test_medical_is_not_out_of_scope(coord, monkeypatch):
    # "what spinal injury do I have" must route normally (refusal comes from the
    # system prompt, NOT the scope gate). Stub classifier returns operational.
    _stub_classify(coord, monkeypatch, "operational")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("what spinal injury do I have"))
    assert result["route"] != "out_of_scope"
    assert seen["operational"] == 1


def test_medical_adapt_is_in_scope(coord, monkeypatch):
    _stub_classify(coord, monkeypatch, "operational")
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("wrist pain when I do biceps, what should I do"))
    assert result["route"] != "out_of_scope"
    assert seen["operational"] == 1


# ── Prompt-presence ──────────────────────────────────────────────────────────

def test_classifier_prompt_has_out_of_scope_taxonomy():
    from src.coordinator import _CLASSIFY_SYSTEM
    assert "OUT_OF_SCOPE" in _CLASSIFY_SYSTEM
    assert "out_of_scope" in _CLASSIFY_SYSTEM
    assert "FITNESS-CONNECTION test" in _CLASSIFY_SYSTEM
    assert "lean IN" in _CLASSIFY_SYSTEM
    # medical carve-out: medical is NOT out_of_scope
    assert "MEDICAL questions are NEVER out_of_scope" in _CLASSIFY_SYSTEM
    # in/out taxonomy markers
    assert "IN SCOPE" in _CLASSIFY_SYSTEM and "OUT OF SCOPE" in _CLASSIFY_SYSTEM


def test_analysis_prompt_has_character_and_medical():
    from src.analysis_agent import _ANALYSIS_SYSTEM
    assert "COACH CHARACTER" in _ANALYSIS_SYSTEM
    assert "USER HOLDS THE FINAL CALL" in _ANALYSIS_SYSTEM
    assert "BIAS TOWARD TRAINING" in _ANALYSIS_SYSTEM
    assert "MEDICAL LINE" in _ANALYSIS_SYSTEM
    assert "NEVER diagnose" in _ANALYSIS_SYSTEM


def test_operational_prompt_has_character_and_medical():
    from src.agent import SYSTEM_PROMPT
    assert "COACH CHARACTER" in SYSTEM_PROMPT
    assert "USER HOLDS THE FINAL CALL" in SYSTEM_PROMPT
    assert "BIAS TOWARD TRAINING" in SYSTEM_PROMPT
    assert "MEDICAL LINE" in SYSTEM_PROMPT
    assert "NEVER diagnose" in SYSTEM_PROMPT
