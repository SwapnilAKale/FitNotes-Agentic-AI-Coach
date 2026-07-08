"""
Write-path fixes for the "Log Barbell Row 60kg x 8" live failure (four coupled
defects):

(1) Fallback-write turns (regex-inferred, no /log prefix) now ARM the log-flow
    carry when they end pending a logging clarification, so the reply turn
    joins the flow and verify receives the ORIGINATING request text.
(2) verify_log_staging treats a missing flow thread (no originating request
    text) as NON-VERIFIABLE — skip with ERROR (fail-open to the human gate),
    never a destructive FAIL against the bare reply text. A real mismatch with
    flow turns present still FAILs (no auto-approve).
(3) _log_workout_sync honors the per-set "unit" arg: a kg input on an
    lbs-native exercise stores metric_weight = weight (the kg figure — reads
    back as weight*2.2046 lbs, the same mass); lbs/absent and kg-native
    kg inputs stay typed/2.2046 byte-unchanged.
(4) A write-success claim can never ship unless a write/stage/execute actually
    happened this turn: structural flags decide, a claim-presence regex only
    detects that a claim is being made (a clarification question with the same
    flags is never touched).

No Gemini (client stubbed / agent faked), no server, no MCP subprocess.
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
from src.coordinator import (                            # noqa: E402
    Coordinator,
    MSG_NO_WRITE_OCCURRED,
)
import mcp_servers.combined_server as cs                 # noqa: E402


# ── Fixtures / stubs (same shape as test_log_command.py) ─────────────────────

class FakeAgent:
    """AgentSession stand-in returning a scripted answer() dict including the
    per-turn write-effect flags the real finalize nodes now export."""

    def __init__(self, answer_text="Staged.", staging_reached_confirm=True,
                 db_write_effect=False, staged_this_turn=False):
        self.questions: list[str] = []
        self.answer_text = answer_text
        self.staging_reached_confirm = staging_reached_confirm
        self.db_write_effect = db_write_effect
        self.staged_this_turn = staged_this_turn

    async def answer(self, question):
        self.questions.append(question)
        return {
            "question": question,
            "answer": self.answer_text,
            "tool_calls_made": 1,
            "error": None,
            "staging_reached_confirm": self.staging_reached_confirm,
            "db_write_effect": self.db_write_effect,
            "staged_this_turn": self.staged_this_turn,
        }


def _make_coord(monkeypatch, agent):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=agent)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _client_returning(text):
    resp = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                        finish_reason=None)
    ])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def _client_never_called():
    def _boom(**kw):
        raise AssertionError("LLM must not be called on this path")
    return SimpleNamespace(models=SimpleNamespace(generate_content=_boom))


_VALID_OPERATIONAL = (
    '{"route":"operational","exercise_names":null,"muscle_groups":null,'
    '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}'
)

_ORIG = "Log Barbell Row 60kg x 8"


# ══════════════════════════════════════════════════════════════════════════════
# Defect 1 — fallback write arms the carry; reply joins the flow
# ══════════════════════════════════════════════════════════════════════════════

def test_fallback_write_pending_clarification_arms_carry(monkeypatch):
    agent = FakeAgent(answer_text="What date was this workout?",
                      staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)

    result = asyncio.run(coord.route(_ORIG))

    assert result["route"] == "operational"
    assert coord._pending_log_carry is True            # the fix: armed on fallback
    assert result["log_flow_turns"] == [_ORIG]         # Input A starts here


def test_fallback_write_then_date_reply_carries_originating_text(monkeypatch):
    agent = FakeAgent(answer_text="What date was this workout?",
                      staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    coord._client = _client_never_called()             # neither turn may classify

    asyncio.run(coord.route(_ORIG))
    result = asyncio.run(coord.route("Today"))

    # The reply consumed the carry and joined the flow: verify Input A is the
    # assembled originating request, never the bare reply.
    assert result["log_boundary"] is True
    assert result["log_flow_turns"] == [_ORIG, "Today"]
    assert agent.questions == [_ORIG, "Today"]


def test_fallback_write_never_flows_bare_reply_as_input_a(monkeypatch):
    # Negative direction: whatever happens, ["Today"] alone must never be the
    # flow-turn list the verify would receive after a fallback-write boundary.
    agent = FakeAgent(answer_text="What date was this workout?",
                      staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    coord._client = _client_never_called()

    asyncio.run(coord.route(_ORIG))
    result = asyncio.run(coord.route("Today"))

    assert result["log_flow_turns"] != ["Today"]


def test_fallback_carry_unrelated_next_turn_clears_without_consuming(monkeypatch):
    agent = FakeAgent(answer_text="What date was this workout?",
                      staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)

    asyncio.run(coord.route(_ORIG))
    # A question is not a clarification answer — clears the carry, routes fresh.
    coord._client = _client_returning(_VALID_OPERATIONAL)
    result = asyncio.run(coord.route("what is my Barbell Row PR?"))

    assert result["log_boundary"] is False
    assert result.get("log_flow_turns") is None
    assert coord._pending_log_carry is False


def test_fallback_write_staging_complete_does_not_arm_carry(monkeypatch):
    agent = FakeAgent(answer_text="Staged.", staging_reached_confirm=True)
    coord = _make_coord(monkeypatch, agent)

    asyncio.run(coord.route(_ORIG))

    assert coord._pending_log_carry is False


# ══════════════════════════════════════════════════════════════════════════════
# Defect 2 — missing flow thread is non-verifiable (ERROR), never FAIL
# ══════════════════════════════════════════════════════════════════════════════

_PREVIEW = "2026-07-05 — Barbell Row: 60 kg × 8"
_SLOT = json.dumps({"staged_workouts": [{"exercise_id": 53, "date": "2026-07-05"}]})


@pytest.mark.parametrize("flow_turns", [None, []])
def test_verify_skips_without_flow_turns_no_llm_no_fail(monkeypatch, flow_turns):
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_never_called()

    verdict = asyncio.run(coord.verify_log_staging(flow_turns, _SLOT, _PREVIEW))

    assert verdict["verdict"] == "ERROR"               # fail-open, never FAIL
    assert "unavailable" in verdict["reason"]


def test_verify_with_flow_turns_still_fails_a_real_mismatch(monkeypatch):
    # No auto-approve: when the originating text IS present the diff runs and
    # a model FAIL verdict is returned unchanged.
    coord = _make_coord(monkeypatch, FakeAgent())
    coord._client = _client_returning(
        '{"verdict":"FAIL","reason":"weights do not match the request"}')

    verdict = asyncio.run(coord.verify_log_staging([_ORIG, "Today"], _SLOT, _PREVIEW))

    assert verdict["verdict"] == "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# Defect 3 — per-set unit honored at staging
# ══════════════════════════════════════════════════════════════════════════════

def _make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY,
            exercise_id INTEGER,
            date DATE,
            metric_weight REAL,
            reps INTEGER,
            unit INTEGER NOT NULL DEFAULT 0,
            is_personal_record INTEGER,
            is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0,
            duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO exercise (_id, name, category_id) VALUES
            (1, 'Barbell Row', 5),                  -- lbs-native
            (2, 'Seated Machine Curl (Kg)', 2),     -- kg-native (all history)
            (3, 'Deadlift', 6);                     -- kg-native from 2025-12-26
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


def _staged_metric(exercise, date, set_dict):
    out = json.loads(cs._log_workout_sync(
        {"exercise_name": exercise, "date": date, "sets": [set_dict]}))
    assert out.get("staged"), out
    return cs._staged_writes["workout"][-1]["sets"][0]["metric_weight"]


def test_kg_on_lbs_native_stores_kg_figure_directly(db):
    # 60 kg on Barbell Row (lbs-native): typed lbs = 60*2.2046 = 132.28, so
    # metric_weight = 132.28/2.2046 = 60.0 — recovers to 132.3 lbs ≡ 60 kg.
    m = _staged_metric("Barbell Row", "2026-07-05",
                       {"weight": 60, "unit": "kg", "reps": 8})
    assert m == pytest.approx(60.0)
    assert m * 2.2046 == pytest.approx(132.276, abs=0.01)   # read-back mass
    # Negative: the old lbs-assumed value (60/2.2046 = 27.2158) must be gone.
    assert m != pytest.approx(27.2158, abs=0.01)


def test_lbs_input_byte_unchanged(db):
    m = _staged_metric("Barbell Row", "2026-07-05",
                       {"weight": 60, "unit": "lbs", "reps": 8})
    assert m == 60 / 2.2046                                 # exact old arithmetic


def test_absent_unit_byte_unchanged(db):
    m = _staged_metric("Barbell Row", "2026-07-05",
                       {"weight": 60, "reps": 8})
    assert m == 60 / 2.2046


def test_kg_native_exercise_kg_input_unchanged(db):
    # kg-native: the typed number IS the kg figure — storage stays typed/2.2046
    # (verified against live DB: Deadlift typed 120 kg ↔ metric 54.4316).
    m = _staged_metric("Seated Machine Curl (Kg)", "2026-07-05",
                       {"weight": 60, "unit": "kg", "reps": 8})
    assert m == 60 / 2.2046


def test_deadlift_kg_era_kg_input_unchanged(db):
    m = _staged_metric("Deadlift", "2026-07-05",
                       {"weight": 120, "unit": "kg", "reps": 1})
    assert m == 120 / 2.2046
    assert m == pytest.approx(54.4316, abs=0.001)           # the live-DB value


def test_deadlift_lbs_era_stays_lbs_typed(db):
    # Before the 2025-12-26 switch Deadlift was lbs-native: a kg input there
    # converts like any lbs-native exercise (metric = the kg figure).
    m = _staged_metric("Deadlift", "2025-11-01",
                       {"weight": 60, "unit": "kg", "reps": 5})
    assert m == pytest.approx(60.0)


# ══════════════════════════════════════════════════════════════════════════════
# Defect 4 — write-success claims are gated on an actual write/stage/execute
# ══════════════════════════════════════════════════════════════════════════════

_FALSE_SUCCESS = "Your Barbell Row set was successfully logged to your database."


def test_unbacked_success_claim_is_suppressed(monkeypatch):
    agent = FakeAgent(answer_text=_FALSE_SUCCESS, staging_reached_confirm=False,
                      db_write_effect=False, staged_this_turn=False)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("Yes thats correct"))

    assert answer == MSG_NO_WRITE_OCCURRED


def test_live_failure_turn_shape_end_to_end(monkeypatch):
    # The exact live shape: "Yes thats correct" classifies operational, the
    # agent free-texts success from stale history, nothing wrote/staged.
    agent = FakeAgent(answer_text=_FALSE_SUCCESS, staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    coord._client = _client_returning(_VALID_OPERATIONAL)

    result = asyncio.run(coord.route("Yes thats correct"))

    assert result["route"] == "operational"
    assert result["answer"] == MSG_NO_WRITE_OCCURRED
    assert "successfully" not in result["answer"].lower()   # claim never ships


def test_success_claim_ships_when_write_actually_happened(monkeypatch):
    agent = FakeAgent(answer_text=_FALSE_SUCCESS, staging_reached_confirm=False,
                      db_write_effect=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("log bodyweight 80kg"))

    assert answer == _FALSE_SUCCESS                     # backed claim untouched


def test_success_claim_ships_on_staged_turn(monkeypatch):
    agent = FakeAgent(answer_text=_FALSE_SUCCESS, staging_reached_confirm=False,
                      staged_this_turn=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("log bench 100x5"))

    assert answer == _FALSE_SUCCESS


def test_success_claim_ships_when_execute_attempted(monkeypatch):
    agent = FakeAgent(answer_text=_FALSE_SUCCESS, staging_reached_confirm=True)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("confirm"))

    assert answer == _FALSE_SUCCESS


def test_clarification_question_with_all_false_flags_not_suppressed(monkeypatch):
    # Pins the over-suppression hazard: the date-ask turn shares the identical
    # all-False structural state and must ship untouched.
    q = "What date was this workout? (format: YYYY-MM-DD or say today/yesterday)"
    agent = FakeAgent(answer_text=q, staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational(_ORIG, fallback_write=True))

    assert q in answer                                  # question survives
    assert MSG_NO_WRITE_OCCURRED not in answer


def test_plain_operational_answer_not_suppressed(monkeypatch):
    agent = FakeAgent(answer_text="Your current goal is 100 kg by September.",
                      staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)

    answer = asyncio.run(coord._run_operational("what is my goal?"))

    assert answer == "Your current goal is 100 kg by September."
