"""
/log deterministic write-boundary command (stage 1 of the verification arc).

(a) a "/log <workout>" turn is a TRUSTED write boundary: the prefix is stripped,
    the turn routes operational with no classify call, and the result carries
    log_boundary=True.
(b) an un-prefixed write still works via the regex fallback (graceful
    degradation) — and only then the answer carries the one-line /log nudge.
(c) a /log message with a trailing analytical tail: the workout head alone
    reaches the agent, the answer carries the trailing-note, and the tail is
    NEVER routed (no analytical call, no decomposition).
(d) single-turn follow-up carry: a /log turn ending pending a logging
    clarification arms the carry; the immediately-next turn joins the /log flow
    (consume-and-clear); an unrelated next message clears WITHOUT consuming;
    a staging-complete turn never arms it.
(e) read_staged_workout_slot returns the RAW staged slot as JSON and is absent
    from list_tools() (unexpose-not-delete, caller-driven only).

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
    _LOG_FALLBACK_NUDGE,
    _LOG_TRAILING_NOTE,
    _log_carry_unrelated,
    _split_log_tail,
)
import mcp_servers.combined_server as cs                 # noqa: E402


# ── Fixtures / stubs ─────────────────────────────────────────────────────────

class FakeAgent:
    """AgentSession stand-in: records the question _run_operational forwards and
    returns a scripted answer() dict incl. the staging_reached_confirm signal."""

    def __init__(self, answer_text="Staged.", staging_reached_confirm=True):
        self.questions: list[str] = []
        self.answer_text = answer_text
        self.staging_reached_confirm = staging_reached_confirm

    async def answer(self, question):
        self.questions.append(question)
        return {
            "question": question,
            "answer": self.answer_text,
            "tool_calls_made": 1,
            "error": None,
            "staging_reached_confirm": self.staging_reached_confirm,
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


_VALID_ANALYTICAL = (
    '{"route":"analytical","exercise_names":null,"muscle_groups":null,'
    '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}'
)


def _spy_reads(coord, monkeypatch):
    """Spy classify + analytical only — _run_operational stays REAL so the
    boundary threading, appends, and carry set-site are what's under test."""
    seen = {"analytical": 0, "classify": 0}

    async def an(q, p, resume=None):
        seen["analytical"] += 1
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    _orig = coord._classify

    async def classify(q):
        seen["classify"] += 1
        return await _orig(q)
    monkeypatch.setattr(coord, "_classify", classify)
    return seen


# ══════════════════════════════════════════════════════════════════════════════
# (a) /log prefix — trusted boundary, prefix stripped, no classify
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw,expected", [
    ("/log bench 100x5, 3 sets",        "bench 100x5, 3 sets"),
    ("  /LOG bench 100x5, 3 sets",      "bench 100x5, 3 sets"),   # case + whitespace
    ("/log: bench 100x5, 3 sets",       "bench 100x5, 3 sets"),   # tolerated separator
])
def test_log_prefix_sets_boundary_and_strips(monkeypatch, raw, expected):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    seen = _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route(raw))

    assert result["log_boundary"] is True
    assert result["route"] == "operational"
    assert agent.questions == [expected]          # prefix stripped, workout intact
    assert seen["classify"] == 0                  # trusted boundary — no inference
    assert seen["analytical"] == 0


def test_log_prefix_not_fired_mid_message(monkeypatch):
    # "/log" is a boundary only at the START of the turn.
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route("what does /log do?"))

    assert result["log_boundary"] is False
    assert agent.questions == []                  # never treated as a write
    assert seen["classify"] == 1                  # routed normally


# ══════════════════════════════════════════════════════════════════════════════
# (b) fallback — regex write-intent still works, answer carries the nudge
# ══════════════════════════════════════════════════════════════════════════════

def test_fallback_write_carries_log_nudge(monkeypatch):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    seen = _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route("log my bench 100x5"))

    assert result["log_boundary"] is False
    assert result["route"] == "operational"       # regex fallback still routes the write
    assert agent.questions == ["log my bench 100x5"]
    assert result["answer"].endswith(_LOG_FALLBACK_NUDGE)
    # Stage 3: regex writes spend one classify call to discover chunks; the
    # distrust override still lands this pure write on the operational lane.
    assert seen["classify"] == 1


def test_log_boundary_answer_never_carries_nudge(monkeypatch):
    # Negative: the nudge is fallback-only — a /log turn must not suggest /log.
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route("/log bench 100x5, 3 sets"))

    assert _LOG_FALLBACK_NUDGE not in result["answer"]


# ══════════════════════════════════════════════════════════════════════════════
# (c) trailing analytical tail — note appended, tail never routed
# ══════════════════════════════════════════════════════════════════════════════

def test_trailing_analytical_tail_noted_not_routed(monkeypatch):
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    seen = _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route(
        "/log bench 100x5, 3 sets. Also, how is my back progressing"))

    # Only the workout head reaches the agent — the tail is peeled, not decomposed.
    assert agent.questions == ["bench 100x5, 3 sets."]
    assert _LOG_TRAILING_NOTE in result["answer"]
    assert seen["analytical"] == 0                # tail NEVER routed
    assert seen["classify"] == 0


def test_log_without_tail_carries_no_note(monkeypatch):
    # Negative direction: a pure workout /log must not grow a spurious note.
    agent = FakeAgent()
    coord = _make_coord(monkeypatch, agent)
    _spy_reads(coord, monkeypatch)

    result = asyncio.run(coord.route("/log bench 100x5, 3 sets"))

    assert _LOG_TRAILING_NOTE not in result["answer"]


@pytest.mark.parametrize("text,head,split", [
    # question-mark tail
    ("bench 100x5, 3 sets. How is my back progressing?",
     "bench 100x5, 3 sets.", True),
    # connector + interrogative lead, NO question mark (the _WRITE_QUESTION_RE gap)
    ("bench 100x5, 3 sets. Also, how is my back progressing",
     "bench 100x5, 3 sets.", True),
    # no tail — distinct stays distinct
    ("bench 100x5, 3 sets", "bench 100x5, 3 sets", False),
    # tail carrying a set/rep/weight quantity is WORKOUT content, never peeled
    ("squat 3 sets of 5. also did 100x5 on bench",
     "squat 3 sets of 5. also did 100x5 on bench", False),
    # everything would peel (head empty) → don't split at all
    ("how is my back progressing?", "how is my back progressing?", False),
])
def test_split_log_tail_both_directions(text, head, split):
    got_head, got_split = _split_log_tail(text)
    assert (got_head, got_split) == (head, split)


# ══════════════════════════════════════════════════════════════════════════════
# (d) single-turn follow-up carry
# ══════════════════════════════════════════════════════════════════════════════

def test_pending_clarification_arms_carry_next_turn_consumes(monkeypatch):
    agent = FakeAgent(staging_reached_confirm=False)   # agent asked a clarification
    coord = _make_coord(monkeypatch, agent)
    _spy_reads(coord, monkeypatch)

    asyncio.run(coord.route("/log bench press"))
    assert coord._pending_log_carry is True            # armed: staged-incomplete

    agent.staging_reached_confirm = True               # clarification answered → completes
    result = asyncio.run(coord.route("yesterday, 3 sets of 12 at 100 lbs"))

    assert result["log_boundary"] is True              # joined the /log flow, no prefix
    assert agent.questions[-1] == "yesterday, 3 sets of 12 at 100 lbs"
    assert coord._pending_log_carry is False           # consumed and cleared


def test_unrelated_next_message_clears_without_consuming(monkeypatch):
    agent = FakeAgent(staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy_reads(coord, monkeypatch)

    asyncio.run(coord.route("/log bench press"))
    assert coord._pending_log_carry is True
    n_agent_calls = len(agent.questions)

    result = asyncio.run(coord.route("how is my squat progressing?"))

    assert coord._pending_log_carry is False           # cleared…
    assert len(agent.questions) == n_agent_calls       # …WITHOUT consuming
    assert result["log_boundary"] is False
    assert result["route"] == "analytical"             # routed normally
    assert seen["classify"] == 1


def test_staging_complete_never_arms_carry(monkeypatch):
    agent = FakeAgent(staging_reached_confirm=True)    # batch reached the confirm gate
    coord = _make_coord(monkeypatch, agent)
    _spy_reads(coord, monkeypatch)

    asyncio.run(coord.route("/log bench 100x5, 3 sets"))

    assert coord._pending_log_carry is False


def test_carry_turn_can_rearm_on_second_clarification(monkeypatch):
    # A carry turn that ITSELF ends pending another clarification re-arms for
    # exactly one more turn (multi-step dialogs); each turn re-evaluates.
    agent = FakeAgent(staging_reached_confirm=False)
    coord = _make_coord(monkeypatch, agent)
    _spy_reads(coord, monkeypatch)

    asyncio.run(coord.route("/log bench press"))
    asyncio.run(coord.route("yesterday"))              # consumed, still incomplete
    assert coord._pending_log_carry is True


@pytest.mark.parametrize("msg,unrelated", [
    ("yesterday",                        False),   # date clarification
    ("3 sets of 12",                     False),   # sets/reps clarification
    ("the dumbbell one",                 False),   # name disambiguation
    ("how is my squat progressing?",     True),    # a question — not an answer
    ("thanks",                           True),    # filler — abandonment
])
def test_log_carry_unrelated_rule(msg, unrelated):
    assert _log_carry_unrelated(msg) is unrelated


# ══════════════════════════════════════════════════════════════════════════════
# agent signal — staging_reached_confirm
# ══════════════════════════════════════════════════════════════════════════════

def _scripted_session(monkeypatch, scripts):
    """AgentSession with _run_collect replaced by a script: each entry is
    (text, [tool names]) per iteration. __init__ only — no MCP, no Gemini."""
    from src.agent import AgentSession
    s = AgentSession(os.environ["FITNOTES_DB_PATH"])
    it = iter(scripts)

    def fake_collect(contents, config):
        text, tool_names = next(it)
        fc_parts = [
            SimpleNamespace(function_call=SimpleNamespace(name=n, args={}))
            for n in tool_names
        ]
        return text, fc_parts, None
    monkeypatch.setattr(s, "_run_collect", fake_collect)
    return s


def test_execute_attempt_blocked_for_confirm_sets_signal(monkeypatch):
    s = _scripted_session(monkeypatch, [("Staging done.", ["execute_staged_workout"])])

    async def deny(tool_name, arguments):     # web-style gate: block, await /confirm
        return False
    s.confirmation_handler = deny

    result = asyncio.run(s.answer("log bench 100x5"))
    assert result["staging_reached_confirm"] is True   # attempt counts, even blocked


def test_clarification_turn_leaves_signal_false(monkeypatch):
    s = _scripted_session(monkeypatch, [("What date was that workout?", [])])

    result = asyncio.run(s.answer("/log bench press"))
    assert result["staging_reached_confirm"] is False


# ══════════════════════════════════════════════════════════════════════════════
# (e) read_staged_workout_slot — raw slot JSON, unexposed from list_tools
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
        CREATE TABLE Comment (
            _id INTEGER PRIMARY KEY,
            date DATE,
            owner_type_id INTEGER,
            owner_id INTEGER,
            comment TEXT
        );
        INSERT INTO exercise (_id, name, category_id) VALUES (1, 'Test Press', 5);
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


def test_raw_slot_tool_returns_staged_slot_json(db):
    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Test Press", "date": "2026-06-01",
        "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5}],
    }))
    assert "error" not in staged, staged

    slot = json.loads(cs._read_staged_workout_slot_sync())
    # The RAW slot — the exact payload execute will write — not the preview.
    assert slot["staged_workouts"] == cs._staged_writes["workout"]
    assert len(slot["staged_workouts"]) == 1
    assert slot["staged_workouts"][0]["exercise_id"] == 1
    assert "preview" not in slot


def test_raw_slot_tool_empty_slot_is_empty_list(db):
    slot = json.loads(cs._read_staged_workout_slot_sync())
    assert slot == {"staged_workouts": []}


def test_raw_slot_tool_dispatchable_but_unexposed(db):
    # Dispatchable via the caller-driven call_tool path…
    out = asyncio.run(cs.call_tool("read_staged_workout_slot", {}))
    assert json.loads(out[0].text) == {"staged_workouts": []}
    # …but absent from list_tools (unexpose-not-delete, like execute_staged_workout).
    exposed = {t.name for t in asyncio.run(cs.list_tools())}
    assert "read_staged_workout_slot" not in exposed
