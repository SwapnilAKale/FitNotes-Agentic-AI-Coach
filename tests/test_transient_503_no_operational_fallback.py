"""
tests/test_transient_503_no_operational_fallback.py

A transient 503 (model overload) or a genuine pipeline bug must NEVER downgrade
an analytical question to the operational lane (operational has no read tools
post-strip and would fabricate an answer). No Gemini calls — exceptions are
simulated and every LLM stage is monkeypatched.

  1. _is_transient_server_error predicate: 503/UNAVAILABLE → True; 429, 500,
     generic → False (incl. the genai ServerError isinstance branch).
  2. _call_with_per_minute_retry retries a transient 503 a bounded number of
     times then succeeds; an unbounded 503 raises (no infinite loop).
  3. Fresh-route handling: 503 → clean busy message, generic bug → clean fail
     message, 429 → propagates — and _run_operational is called ZERO times on
     every analytical-pipeline failure class.
  4. Regression: the legitimate operational route still calls _run_operational.
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
from src.coordinator import Coordinator                 # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────────

ANALYTICAL_PARAMS = {
    "route": "analytical", "exercise_names": None, "muscle_groups": None,
    "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None,
}
OPERATIONAL_PARAMS = {**ANALYTICAL_PARAMS, "route": "operational"}

FAKE_PKG = {"scope": "broad", "query_period_days": 90, "exercises": []}

ERR_503     = RuntimeError("503 UNAVAILABLE: The model is overloaded.")
ERR_UNAVAIL = RuntimeError("Service temporarily UNAVAILABLE, please retry.")
ERR_429     = RuntimeError("429 RESOURCE_EXHAUSTED. {'retryDelay': '30s'}")
ERR_500     = RuntimeError("500 INTERNAL error in the pipeline.")
ERR_GENERIC = ValueError("boom — a genuine code bug")


class FakeServerError(Exception):
    """Stand-in for google.genai.errors.ServerError (carries .code/.status)."""
    def __init__(self, code, status):
        self.code = code
        self.status = status
        super().__init__(f"{code} {status}")


@pytest.fixture(autouse=True)
def _isolate_checkpoint(tmp_path, monkeypatch):
    """Point the checkpoint slot at a temp file so a stray real slot can't make
    route() prompt for discard instead of processing the question."""
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))


@pytest.fixture()
def coord(monkeypatch):
    """Coordinator with no Gemini client and a free (stubbed) package build."""
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)

    async def fake_classify(question):
        return dict(ANALYTICAL_PARAMS)
    monkeypatch.setattr(c, "_classify", fake_classify)

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: dict(FAKE_PKG))
    try:
        import src.shared.memory as shared_memory
        monkeypatch.setattr(shared_memory, "retrieve_relevant_memories",
                            lambda q: None)
    except ImportError:
        pass
    return c


def _no_sleep(monkeypatch):
    """Neutralize the in-request backoff wait so retry tests don't actually wait."""
    async def _instant(_seconds):
        return None
    monkeypatch.setattr(coordinator_mod.asyncio, "sleep", _instant)


def _count_operational(coord):
    """Replace _run_operational with a counter; return the counter dict."""
    calls = {"n": 0}

    async def fake_op(question, **kw):   # **kw: /log boundary flags (ignored here)
        calls["n"] += 1
        return "OPERATIONAL-ANSWER (should never appear on analytical failure)"

    coord._run_operational = fake_op   # bound-name override; instance attr wins
    return calls


# ── 1. Predicate ────────────────────────────────────────────────────────────────

def test_predicate_string_paths():
    f = coordinator_mod._is_transient_server_error
    assert f(ERR_503) is True
    assert f(ERR_UNAVAIL) is True
    assert f(ERR_429) is False
    assert f(ERR_500) is False          # 500 INTERNAL is a bug, not transient
    assert f(ERR_GENERIC) is False


def test_predicate_isinstance_branch(monkeypatch):
    monkeypatch.setattr(coordinator_mod, "_GenaiServerError", FakeServerError)
    f = coordinator_mod._is_transient_server_error
    assert f(FakeServerError(503, "UNAVAILABLE")) is True
    assert f(FakeServerError(200, "UNAVAILABLE")) is True   # status alone qualifies
    assert f(FakeServerError(500, "INTERNAL")) is False     # 500 → not transient


# ── 2. Bounded retry in _call_with_per_minute_retry ─────────────────────────────

def test_retry_503_then_success(coord, monkeypatch):
    _no_sleep(monkeypatch)
    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] <= coordinator_mod.TRANSIENT_MAX_RETRIES:
            raise ERR_503
        return "OK"

    result = asyncio.run(coord._call_with_per_minute_retry(flaky))
    assert result == "OK"
    # initial attempt + TRANSIENT_MAX_RETRIES retries == the success call
    assert state["n"] == coordinator_mod.TRANSIENT_MAX_RETRIES + 1


def test_retry_503_exhausts_and_raises(coord, monkeypatch):
    _no_sleep(monkeypatch)
    state = {"n": 0}

    async def always_503():
        state["n"] += 1
        raise ERR_503

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(coord._call_with_per_minute_retry(always_503))
    assert "503" in str(ei.value)
    # 1 initial + TRANSIENT_MAX_RETRIES retries, then it gives up (no infinite loop)
    assert state["n"] == coordinator_mod.TRANSIENT_MAX_RETRIES + 1


# ── 3. Fresh-route handling — the KEY tests ─────────────────────────────────────

def test_fresh_503_returns_busy_and_no_operational(coord, monkeypatch):
    calls = _count_operational(coord)

    async def boom_503(question, params, resume=None):
        raise ERR_503
    monkeypatch.setattr(coord, "_run_analytical", boom_503)

    result = asyncio.run(coord.route("how is my bench progressing?"))
    assert result["answer"] == coordinator_mod._MSG_MODEL_BUSY
    assert result["route"] == "analytical"
    assert result["error"]
    assert calls["n"] == 0                       # operational NEVER touched


def test_fresh_generic_bug_returns_clean_fail_and_no_operational(coord, monkeypatch):
    calls = _count_operational(coord)

    async def boom_bug(question, params, resume=None):
        raise ERR_GENERIC
    monkeypatch.setattr(coord, "_run_analytical", boom_bug)

    result = asyncio.run(coord.route("how is my bench progressing?"))
    assert result["answer"] == coordinator_mod._MSG_PIPELINE_ERROR
    assert result["route"] == "analytical"
    assert result["error"]
    assert calls["n"] == 0                       # operational NEVER touched


def test_fresh_429_propagates_and_no_operational(coord, monkeypatch):
    calls = _count_operational(coord)

    async def boom_429(question, params, resume=None):
        raise ERR_429
    monkeypatch.setattr(coord, "_run_analytical", boom_429)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(coord.route("how is my bench progressing?"))
    assert "429" in str(ei.value)
    assert calls["n"] == 0                       # operational NEVER touched


# ── 4. Regression — legitimate operational route still works ────────────────────

def test_operational_route_still_calls_operational(coord, monkeypatch):
    calls = _count_operational(coord)

    async def fake_classify(question):
        return dict(OPERATIONAL_PARAMS)
    monkeypatch.setattr(coord, "_classify", fake_classify)

    # A non-write question so it reaches _classify (not the write short-circuit)
    # and is routed operational by the stub.
    result = asyncio.run(coord.route("what does progressive overload mean?"))
    assert calls["n"] == 1                        # legitimate lane intact
    assert result["route"] == "operational"
