"""
Memory extraction wired into the analytical path.

Bug: `_auto_extract_memories` scans `AgentSession._conversation_history`, which is
populated only by the operational `answer()` loop. The Coordinator's analytical
path runs the analysis pipeline directly and never appended to it, so analytical/
coaching Q&A was invisible to extraction (coordinator marked it "NOT YET WIRED").

Fix: `_run_analytical` records the (question, FINAL STRIPPED answer) exchange via
`AgentSession.record_external_exchange`, in the same shape operational turns use.
out_of_scope / filler / parse-failure short-circuit before `_run_analytical`, so
they record nothing.

No Gemini — the LLM stages are stubbed; the recording is asserted directly.
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

from src import coordinator as coordinator_mod          # noqa: E402
from src.coordinator import Coordinator                 # noqa: E402
from src import analysis_agent                          # noqa: E402


# ── AgentSession.record_external_exchange (unit) ─────────────────────────────

def test_record_external_exchange_shape():
    from src.agent import AgentSession
    s = AgentSession(os.environ["FITNOTES_DB_PATH"])   # __init__ only; no MCP/Gemini
    s.record_external_exchange("how is my back?", "Your back volume is up 10%.")
    assert s._conversation_history == [[
        {"role": "user",      "content": "how is my back?"},
        {"role": "assistant", "content": "Your back volume is up 10%."},
    ]]
    # the exchange is the same shape _auto_extract_memories iterates (role/content)
    ex = s._conversation_history[0]
    assert all("role" in m and "content" in m for m in ex)


def test_record_external_exchange_empty_guard():
    from src.agent import AgentSession
    s = AgentSession(os.environ["FITNOTES_DB_PATH"])
    s.record_external_exchange("", "answer")
    s.record_external_exchange("question", "")
    s.record_external_exchange("question", None)  # type: ignore[arg-type]
    assert s._conversation_history == []


def test_record_external_exchange_counts_toward_threshold_and_caps():
    from src.agent import AgentSession
    s = AgentSession(os.environ["FITNOTES_DB_PATH"])
    for i in range(4):
        s.record_external_exchange(f"q{i}", f"a{i}")
    # each analytical turn = one exchange → 4 exchanges clears the <4 early-return
    assert len(s._conversation_history) == 4
    for i in range(4, 15):
        s.record_external_exchange(f"q{i}", f"a{i}")
    assert len(s._conversation_history) == 10          # capped (same as operational)


# ── Coordinator analytical path records the STRIPPED answer ──────────────────

class _FakeAgent:
    def __init__(self):
        self.recorded = []
    def record_external_exchange(self, q, a):
        self.recorded.append((q, a))


@pytest.fixture()
def coord_with_agent(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    agent = _FakeAgent()
    c = Coordinator(agent_session=agent)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c, agent


_BROAD = {"route": "analytical", "exercise_names": None, "muscle_groups": None,
          "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None}


def test_analytical_turn_records_stripped_answer(coord_with_agent, monkeypatch):
    coord, agent = coord_with_agent
    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: {"scope": "broad", "exercises": []})

    async def fake_analyze(*a, **k):
        # analyze returns a TAGGED draft; the coordinator strips before grounding
        return "Your PR is 130 lbs [[exercises|Lat Pulldown|pr.weight]]."
    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)

    async def fake_ground(draft, gctx):
        return draft, []          # echoes the STRIPPED draft it is given
    monkeypatch.setattr(analysis_agent, "ground_check", fake_ground)

    async def fake_cov(self, q, answer):
        return answer, True
    monkeypatch.setattr(Coordinator, "_coverage_check", fake_cov)

    answer, flagged = asyncio.run(
        coord._run_analytical("how's my lat pulldown?", dict(_BROAD)))

    assert agent.recorded == [("how's my lat pulldown?", "Your PR is 130 lbs.")]
    # recorded answer is the STRIPPED one — no citation tags
    recorded_q, recorded_a = agent.recorded[0]
    assert "[[" not in recorded_a and "]]" not in recorded_a
    assert recorded_a == answer        # exactly the final user-facing answer


# ── Exclusions: out_of_scope / filler / parse-failure record nothing ─────────

def _client_returning(text):
    resp = SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]), finish_reason=None)])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def test_filler_records_nothing(coord_with_agent):
    coord, agent = coord_with_agent
    result = asyncio.run(coord.route("ok"))
    assert result["route"] == "filler"
    assert agent.recorded == []


def test_parse_failure_records_nothing(coord_with_agent):
    coord, agent = coord_with_agent
    coord._client = _client_returning("this is not json")
    result = asyncio.run(coord.route("asdkjf qweoiruzxcv"))
    assert result["route"] == "unparseable"
    assert agent.recorded == []


def test_out_of_scope_records_nothing(coord_with_agent, monkeypatch):
    coord, agent = coord_with_agent
    async def fake_classify(q):
        return {"route": "out_of_scope", "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None}
    monkeypatch.setattr(coord, "_classify", fake_classify)
    result = asyncio.run(coord.route("write me a poem"))
    assert result["route"] == "out_of_scope"
    assert agent.recorded == []
