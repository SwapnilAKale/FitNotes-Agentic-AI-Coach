"""
Demographics — Stage B (follow-up asking mechanism).

Deterministic (no Gemini): when the user mentions a demographic in passing and
the anchor isn't stored, the agent appends ONE gentle answer-first follow-up;
the immediate next reply is parsed + written (via Stage A) or the follow-up is
dropped. The dropped/unanswered aside never enters memory-extraction history
(it's appended AFTER the clean answer is recorded).

Store isolated to a tmp file; no Chroma; the analytical pipeline is stubbed.
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

from src import demographic_followup as F          # noqa: E402
from src import coordinator as coordinator_mod      # noqa: E402
from src.coordinator import Coordinator             # noqa: E402
from src import analysis_agent                      # noqa: E402
import src.memory as memory                         # noqa: E402


# ── pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg,expected", [
    # "22-year-old male" — age cue fires (birthdate); the sex detector is
    # deliberately conservative (needs an explicit "as a / I'm a male" cue), and
    # only one follow-up is offered per turn regardless (birthdate has priority).
    ("Should I lift more for a 22-year-old male?", ["birthdate"]),
    ("I've been training 3 years, am I plateauing?", ["training_start_date"]),
    ("I'm 5'9, how's my squat?", ["height"]),
    ("176cm tall — good bench?", ["height"]),
    ("as a male, am I strong?", ["sex"]),
    ("what's my squat PR?", []),                 # no demographic cue
    ("log bench 3x5 at 100 lbs", []),            # reps/weight must NOT fire
    ("how many days did I train?", []),
])
def test_detect_mentions(msg, expected):
    assert F.detect_mentions(msg) == expected


@pytest.mark.parametrize("key,msg,kind,value,unit", [
    ("birthdate", "2003-06-18", "answer", "2003-06-18", None),
    ("birthdate", "March 1995", "answer", "1995-03-01", None),
    ("birthdate", "2003", "clarify", None, None),            # bare year → clarify
    ("birthdate", "how is my squat?", "not_answer", None, None),
    ("training_start_date", "2021", "answer", "2021-01-01", None),  # bare year → Jan-1
    ("training_start_date", "June 2020", "answer", "2020-06-01", None),
    ("sex", "male", "answer", "male", None),
    ("sex", "I'm a guy", "answer", "male", None),
    ("height", "176cm", "answer", 176.0, "cm"),
    ("height", "180", "clarify", None, None),                # bare number → clarify unit
    ("height", "what?", "not_answer", None, None),
    ("birthdate", "2003-13-40", "clarify", None, None),      # impossible date
])
def test_interpret_answer(key, msg, kind, value, unit):
    r = F.interpret_answer(key, msg)
    assert r["kind"] == kind
    if kind == "answer":
        assert r["value"] == value and r.get("unit") == unit


def test_height_feet_inches():
    r = F.interpret_answer("height", "5'9")
    assert r == {"kind": "answer", "value": 69.0, "unit": "in"}


# ── coordinator: post-step (trigger) with the inner route stubbed ───────────

class _FakeAgent:
    def __init__(self):
        self.recorded = []
    def record_external_exchange(self, q, a):
        self.recorded.append((q, a))


MAIN = "Your squat is up 10% this period."


@pytest.fixture()
def coord(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "mem.json")
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    c = Coordinator(agent_session=_FakeAgent())

    async def fake_inner(q, **kw):       # **kw: /log carry flag (ignored here)
        return {"answer": MAIN, "route": "analytical", "flagged_claims": [], "error": None}
    monkeypatch.setattr(c, "_route_with_checkpoint", fake_inner)
    return c


def test_mention_appends_single_followup_answer_first(coord):
    res = asyncio.run(coord.route("Should I lift more for a 22-year-old?"))
    assert res["answer"].startswith(MAIN)              # answer-first
    assert "\n\n" in res["answer"]                     # follow-up on a new line
    assert F.followup_text("birthdate") in res["answer"]
    assert coord._pending_followup == {"key": "birthdate", "clarified": False}


def test_no_followup_when_anchor_already_stored(coord):
    memory.set_demographic("birthdate", "2003-06-18")
    res = asyncio.run(coord.route("advice for a 22-year-old?"))
    assert res["answer"] == MAIN                       # nothing appended
    assert coord._pending_followup is None


def test_no_followup_when_no_relevant_mention(coord):
    res = asyncio.run(coord.route("how's my squat trending?"))
    assert res["answer"] == MAIN
    assert coord._pending_followup is None


def test_only_one_pending_at_a_time(coord):
    # a message mentioning two anchors sets exactly one pending (first by priority)
    asyncio.run(coord.route("as a 22-year-old male, am I strong?"))
    assert coord._pending_followup["key"] == "birthdate"   # one, not a stack


# ── coordinator: pre-step (consume / clarify / drop) ────────────────────────

def test_answer_parsed_and_written_with_ack(coord):
    coord._pending_followup = {"key": "birthdate", "clarified": False}
    res = asyncio.run(coord.route("2003-06-18"))
    assert res["route"] == "followup_ack"
    assert res["answer"] == F.ACK
    assert memory.get_demographic("birthdate")["value"] == "2003-06-18"
    assert coord._pending_followup is None


def test_sex_and_height_answers_written(coord):
    coord._pending_followup = {"key": "sex", "clarified": False}
    assert asyncio.run(coord.route("male"))["route"] == "followup_ack"
    assert memory.get_demographic("sex")["value"] == "male"

    coord._pending_followup = {"key": "height", "clarified": False}
    asyncio.run(coord.route("176cm"))
    h = memory.get_demographic("height")
    assert h["value"] == 176.0 and h["unit"] == "cm"


def test_start_date_bare_year_convention(coord):
    coord._pending_followup = {"key": "training_start_date", "clarified": False}
    asyncio.run(coord.route("2021"))
    assert memory.get_demographic("training_start_date")["value"] == "2021-01-01"


def test_clarify_once_then_give_up(coord):
    coord._pending_followup = {"key": "birthdate", "clarified": False}
    r1 = asyncio.run(coord.route("2003"))             # bare year → clarify
    assert r1["route"] == "followup_clarify"
    assert coord._pending_followup == {"key": "birthdate", "clarified": True}
    # a second malformed attempt → give up (drop, route normally), no loop
    r2 = asyncio.run(coord.route("2004"))
    assert r2["answer"] == MAIN                        # routed normally
    assert coord._pending_followup is None
    assert memory.get_demographic("birthdate") is None


def test_drop_if_unanswered_routes_normally(coord):
    coord._pending_followup = {"key": "birthdate", "clarified": False}
    res = asyncio.run(coord.route("how's my squat?"))  # a new question, not an answer
    assert res["answer"] == MAIN                        # routed normally
    assert res["route"] == "analytical"
    assert coord._pending_followup is None
    assert memory.get_demographic("birthdate") is None


def test_one_turn_life_late_bare_answer_not_captured(coord):
    # after a drop, pending is None; a later bare "2003-06-18" is NOT auto-written
    coord._pending_followup = None
    asyncio.run(coord.route("2003-06-18"))
    assert memory.get_demographic("birthdate") is None
    assert coord._pending_followup is None


# ── drop-before-record: the follow-up never enters extraction history ───────

def test_followup_not_in_recorded_history(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "mem.json")
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    agent = _FakeAgent()
    c = Coordinator(agent_session=agent)

    # real _run_analytical (records the CLEAN answer) with the LLM stages stubbed
    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: {"scope": "broad", "exercises": []})
    async def fake_analyze(*a, **k):
        return "Your squat is up 10%."
    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)
    async def fake_ground(draft, gctx):
        return draft, []
    monkeypatch.setattr(analysis_agent, "ground_check", fake_ground)
    async def fake_cov(self, q, answer):
        return answer, True
    monkeypatch.setattr(Coordinator, "_coverage_check", fake_cov)
    async def fake_classify(self, q):
        return {"route": "analytical", "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None}
    monkeypatch.setattr(Coordinator, "_classify", fake_classify)

    res = asyncio.run(c.route("Should I lift more as a 22-year-old?"))

    # the RETURNED answer carries the follow-up …
    assert F.followup_text("birthdate") in res["answer"]
    # … but the RECORDED exchange (for memory extraction) is the CLEAN answer only
    assert agent.recorded, "analytical turn should have recorded an exchange"
    recorded_answer = agent.recorded[-1][1]
    assert "By the way" not in recorded_answer
    assert F.followup_text("birthdate") not in recorded_answer
