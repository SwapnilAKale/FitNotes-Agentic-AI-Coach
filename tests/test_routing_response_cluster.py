"""
Routing / server-response cluster (#2, #3+#8, #5).

#2  — /chat surfaces the Coordinator's graceful ANSWER, never the raw error /
      internal invariant-ID string; falls back to a generic message only when
      there is genuinely no answer. _reinitialize_session LOGS a close() failure
      instead of silently swallowing it.
#3  — the write-intent pre-guard must NOT fire on coaching QUESTIONS about
      writing (they'd strand in the impoverished operational agent).
#8  — it MUST fire on bare imperative writes ("log my bench 100x5") that the old
      noun-list regex missed.
#5a — bare fillers short-circuit with a canned reply: no classify, no package,
      no pipeline.
#5b — an UNPARSEABLE classify returns a cheap rephrase, not a full analytical
      pipeline on garbage input.

No Gemini (classifier client stubbed), no real subprocess.
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
from src.coordinator import Coordinator, _is_write_intent, _filler_reply  # noqa: E402


# ── Fixtures / stubs ─────────────────────────────────────────────────────────

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


_VALID_ANALYTICAL = (
    '{"route":"analytical","exercise_names":null,"muscle_groups":null,'
    '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}'
)


def _spy(coord, monkeypatch):
    seen = {"analytical": 0, "operational": 0, "classify": 0}

    async def an(q, p, resume=None):
        seen["analytical"] += 1
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):               # **kw: /log boundary flags (ignored here)
        seen["operational"] += 1
        return "OP"
    monkeypatch.setattr(coord, "_run_operational", op)

    _orig = coord._classify

    async def classify(q):
        seen["classify"] += 1
        return await _orig(q)
    monkeypatch.setattr(coord, "_classify", classify)
    return seen


# ══════════════════════════════════════════════════════════════════════════════
# #3 + #8 — write-intent guard, both directions
# ══════════════════════════════════════════════════════════════════════════════

# Coaching QUESTIONS about writing — must NOT fire the guard (→ classifier).
COACHING_QUESTIONS = [
    "should I add weight to my squat",
    "should I add more weight to my squat",
    "can I add a set",
    "is it ok to remove a set",
    "when should I update my goal",
    "do you think I should add a set",
    "would it help to add weight?",
    "how many sets should I add to my chest day",
]

# Imperative WRITES — must fire the guard (→ operational, gated by confirmation).
IMPERATIVE_WRITES = [
    "log my bench 100x5",
    "record squat 80kg x5",
    "add 3 sets of deadlift",
    "log bench press 3x5",
    "log bench press 3x5 100lbs",
    "set a goal for 150 lbs on Lat Pulldown",
    "delete my deadlift goal",
    "update my last set — it was 12 reps not 10",
    "I did chest and triceps today",
    "log today's workout: bench 100x5",
]


@pytest.mark.parametrize("q", COACHING_QUESTIONS)
def test_coaching_questions_do_not_fire_write_guard(q):
    assert _is_write_intent(q) is False


@pytest.mark.parametrize("q", IMPERATIVE_WRITES)
def test_imperative_writes_fire_write_guard(q):
    assert _is_write_intent(q) is True


def test_ambiguous_precedence_question_wins():
    # A message that is BOTH a question AND carries write-shorthand: the question
    # phrasing wins (the implemented precedence) → guard does NOT fire. Rationale:
    # a missed write is still caught by the classifier and blocked by the
    # confirmation gate (recoverable), whereas a false-positive strands a
    # coaching question (no recourse).
    assert _is_write_intent("should I log bench 100x5?") is False
    assert _is_write_intent("can I add 3 sets of squats?") is False


@pytest.mark.parametrize("q", COACHING_QUESTIONS)
def test_coaching_questions_route_analytical_not_operational(coord, monkeypatch, q):
    # End-to-end: the pre-guard stays silent, the (stubbed) classifier routes the
    # read analytical — NOT force-routed operational.
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "analytical"
    assert seen["operational"] == 0 and seen["analytical"] == 1
    assert seen["classify"] == 1            # the classifier, not the guard, decided


@pytest.mark.parametrize("q", IMPERATIVE_WRITES)
def test_imperative_writes_route_operational_without_classify(coord, monkeypatch, q):
    # The pre-guard fires BEFORE the classifier → operational, no classify call.
    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "operational"
    assert seen["operational"] == 1 and seen["analytical"] == 0
    assert seen["classify"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# #5a — filler short-circuit
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q", ["ok", "thanks", "hi", "cool", "okay", "thank you",
                               "yes", "no", "got it", "hello", "  Thanks!  ", "K"])
def test_fillers_short_circuit_no_pipeline(coord, monkeypatch, q):
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "filler"
    assert result["answer"]                       # a non-empty canned reply
    # The crux: NO classify, NO analytical/operational pipeline ran.
    assert seen == {"analytical": 0, "operational": 0, "classify": 0}


def test_thanks_gets_welcome_reply():
    assert "welcome" in (_filler_reply("thanks") or "").lower()
    assert _filler_reply("hi") == coordinator_mod._FILLER_GENERIC_REPLY


def test_polite_prefix_with_real_question_is_not_filler(coord, monkeypatch):
    # "thanks, now how's my squat" carries a real question → NOT filler; routes
    # normally (classifier runs).
    assert _filler_reply("thanks, now how's my squat") is None
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route("thanks, now how's my squat"))
    assert result["route"] == "analytical"
    assert seen["classify"] == 1 and seen["analytical"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# #5b — parse-failure cheap default
# ══════════════════════════════════════════════════════════════════════════════

def test_unparseable_classify_returns_rephrase_no_package(coord, monkeypatch):
    coord._client = _client_returning("this is not json at all")

    built = {"n": 0}
    def _builder(*a, **k):
        built["n"] += 1
        return {}
    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package", _builder)

    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route("asdkfjasldkfj qweoiruzxcv"))
    assert result["route"] == "unparseable"
    assert "rephrase" in result["answer"].lower()
    # No analytical pipeline, no package build on garbage input.
    assert seen["analytical"] == 0 and seen["operational"] == 0
    assert built["n"] == 0


def test_parsed_but_uncertain_still_analytical(coord, monkeypatch):
    # Step-C behavior preserved: a PARSED classification (even a plain default)
    # routes analytical — only the UNPARSEABLE case is short-circuited by #5b.
    coord._client = _client_returning(_VALID_ANALYTICAL)
    seen = _spy(coord, monkeypatch)
    result = asyncio.run(coord.route("how is my back training trending"))
    assert result["route"] == "analytical"
    assert seen["analytical"] == 1


def test_classify_marks_parse_failure_flag(coord):
    coord._client = _client_returning("garbage")
    params = asyncio.run(coord._classify("whatever"))
    assert params["route"] == "analytical"       # still defaults analytical…
    assert params.get("_parse_failed") is True   # …but flagged unparseable


# ══════════════════════════════════════════════════════════════════════════════
# #2 — server surfaces the graceful answer, never the raw error / invariant IDs
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def srv(monkeypatch):
    import server as server_mod
    monkeypatch.setattr(server_mod, "agent_ready", True)
    monkeypatch.setattr(server_mod, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(server_mod, "session", None)
    server_mod._state.update({"pending_confirmation": False,
                              "allow_execute": False, "staging_preview": ""})
    return server_mod


def _body(resp):
    return json.loads(resp.body.decode())


def test_chat_surfaces_answer_when_error_and_answer_present(srv, monkeypatch):
    # Integrity-failure shape: error carries an internal invariant ID, but the
    # coordinator already built a friendly answer. The answer must win; the ID
    # must NOT leak.
    async def route(msg):
        return {
            "answer": "I cannot answer this right now — please try again shortly.",
            "route": "analytical",
            "flagged_claims": [],
            "error": "B3: muscle_group_summary total mismatch (raw invariant text)",
        }
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(route=route))

    resp = asyncio.run(srv._process_turn("why am I plateauing"))
    body = _body(resp)
    assert body["type"] == "answer"
    assert body["text"].startswith("I cannot answer this right now")
    assert "B3" not in body["text"]
    assert "invariant" not in body["text"].lower()


def test_chat_generic_message_when_error_and_no_answer(srv, monkeypatch):
    async def route(msg):
        return {"answer": "", "route": "analytical", "flagged_claims": [],
                "error": "B3: raw internal invariant string"}
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(route=route))

    resp = asyncio.run(srv._process_turn("why am I plateauing"))
    body = _body(resp)
    assert body["type"] == "error"
    assert "B3" not in body["text"]
    assert body["text"] == "Something went wrong while answering that. Please try again."


def test_chat_normal_answer_unaffected(srv, monkeypatch):
    async def route(msg):
        return {"answer": "Your bench is up 10 lbs.", "route": "analytical",
                "flagged_claims": [], "error": None}
    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(route=route))
    resp = asyncio.run(srv._process_turn("how's my bench"))
    body = _body(resp)
    assert body["type"] == "answer" and body["text"] == "Your bench is up 10 lbs."


# ── preview_source observability: slot-read vs args fallback must be tellable ──
# The args fallback renders workout-shaped text too, so without a source tag a
# live check cannot prove the slot-read fired — a false-pass is possible.

def _confirm_turn(srv, monkeypatch, formatter_result):
    # Fake route arms the confirmation exactly as the log_workout staging branch
    # of _confirmation_handler would (route runs AFTER the turn-start reset).
    async def route(msg):
        srv._state["pending_confirmation"] = True
        srv._state["confirmation_preview"] = "ARGS-BLOB"
        srv._state["pending_execute_kind"] = "workout"
        return {"answer": "", "route": "operational", "flagged_claims": [], "error": None}

    async def call_tool(name, args):
        if name == "format_staged_workout_for_confirmation":
            return json.dumps(formatter_result)
        return json.dumps({"discarded": True})   # turn-start discard_staged_writes

    monkeypatch.setattr(srv, "coordinator", SimpleNamespace(route=route))
    monkeypatch.setattr(srv, "session", SimpleNamespace(call_tool=call_tool))
    return _body(asyncio.run(srv._process_turn("log my workout")))


def test_confirmation_preview_source_slot(srv, monkeypatch):
    body = _confirm_turn(srv, monkeypatch,
                         {"preview": "Staged workout — 2026-07-01\n\nBench"})
    assert body["type"] == "confirmation_required"
    assert body["preview"].startswith("Staged workout")
    assert body["preview_source"] == "slot"


def test_confirmation_preview_source_args_fallback_logs(srv, monkeypatch, capsys):
    # Formatter answers with an error payload (no preview key) → args fallback,
    # tagged as such, AND the previously-silent path now logs to stderr.
    body = _confirm_turn(srv, monkeypatch, {"error": "No staged workout found."})
    assert body["type"] == "confirmation_required"
    assert body["preview"] == "ARGS-BLOB"
    assert body["preview_source"] == "args_fallback"
    assert "staged-workout preview empty/error" in capsys.readouterr().err


def test_reinitialize_session_logs_close_failure(srv, monkeypatch, capsys):
    # A close() that raises must be LOGGED (not silently swallowed), and the
    # reload must still proceed.
    class BoomSession:
        async def close(self):
            raise RuntimeError("close blew up unexpectedly")

    class FakeNewSession:
        def __init__(self, *a, **k):
            self.confirmation_handler = None
        async def initialize(self):
            return None

    monkeypatch.setattr(srv, "session", BoomSession())
    monkeypatch.setattr(srv, "AgentSession", FakeNewSession)
    monkeypatch.setattr(srv, "Coordinator", lambda s: SimpleNamespace())

    asyncio.run(srv._reinitialize_session())
    out = capsys.readouterr().out
    assert "close blew up unexpectedly" in out
    assert "Unexpected error closing old session" in out
    assert srv.agent_ready is True            # reload still proceeded
