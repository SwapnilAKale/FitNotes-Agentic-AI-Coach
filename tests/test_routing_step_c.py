"""
Step C, Parts 2 & 3: flip the read default to ANALYTICAL, then unexpose the four
analysis-read tools from the operational agent.

Part 2 — operational is a positive allowlist (writes via the write-intent regex
pre-guard, research/RAG, + out_of_scope). Every other read/coaching question,
and anything uncertain (parse/exception fail), defaults ANALYTICAL. (Stage 3
moved session-display from operational to analytical — it is no longer an
operational allowlist case.)

Part 3 — get_personal_record / get_weekly_volume / query_workout_data /
run_read_only_sql are removed from the operational agent's exposed tool list
(list_tools); their dispatch handlers + _sync functions remain for analytical
reuse / evals. Kept exposed: get_exercise_sessions, resolve_exercise_name,
read_exercise_comments, get_exercise_history, all write tools, RAG, memory,
quirks.

No Gemini — the classifier LLM client is stubbed; routing decisions and the
exposed tool list are asserted directly. No server.
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
    """A fake genai client whose classify call returns `text` as the model output."""
    resp = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                        finish_reason=None)
    ])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def _client_raising(exc):
    def boom(**kw):
        raise exc
    return SimpleNamespace(models=SimpleNamespace(generate_content=boom))


def _spy_downstream(coord, monkeypatch):
    seen = {"analytical": 0, "operational": 0}

    async def an(q, p, resume=None):
        seen["analytical"] += 1
        return "AN", []
    monkeypatch.setattr(coord, "_run_analytical", an)

    async def op(q, **kw):               # **kw: /log boundary flags (ignored here)
        seen["operational"] += 1
        return "OP"
    monkeypatch.setattr(coord, "_run_operational", op)
    return seen


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — the default flip
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad_output", ["this is not json", "{oops", ""])
def test_classify_parse_failure_defaults_analytical(coord, bad_output):
    # Parse failure (and the exception path) must default ANALYTICAL, not operational.
    coord._client = _client_returning(bad_output)
    params = asyncio.run(coord._classify("my lat pulldown"))
    assert params["route"] == "analytical"


def test_classify_exception_defaults_analytical(coord):
    coord._client = _client_raising(RuntimeError("classify boom"))
    params = asyncio.run(coord._classify("how's my back"))
    assert params["route"] == "analytical"


@pytest.mark.parametrize("q", [
    "my PR on Lat Pulldown",          # single-exercise PR (previously leaked operational)
    "Lat Pulldown",                   # terse
    "how often do I train chest",     # frequency
    "total volume by muscle group",   # volume
    "how's my squat?",                # terse follow-up style
    "am I making progress",           # trend
])
def test_ambiguous_reads_route_analytical_under_flip(coord, monkeypatch, q):
    # A parsed-but-uncertain classifier on a read → analytical end-to-end.
    # (An UNPARSEABLE classify is now the #5b cheap-rephrase path, tested
    # separately — Step C's "parsed → analytical" default is what this asserts.)
    coord._client = _client_returning(
        '{"route":"analytical","exercise_names":null,"muscle_groups":null,'
        '"query_period_days":90,"needs_custom_sql":false,"custom_sql_intent":null}'
    )
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "analytical"
    assert seen["analytical"] == 1 and seen["operational"] == 0


@pytest.mark.parametrize("q", [
    "log today's workout: bench 100x5",
    "set a goal for 150 lbs on Lat Pulldown",
    "delete my deadlift goal",
    "update my last set — it was 12 reps not 10",
    "I did chest and triceps today",
])
def test_writes_still_route_operational(coord, monkeypatch, q):
    # The write-intent regex pre-guard fires BEFORE the classifier → operational.
    # (A write that somehow defaulted analytical just wouldn't write — the
    # confirmation gate is operational-only — so data can't be corrupted.)
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route(q))
    assert result["route"] == "operational"
    assert seen["operational"] == 1 and seen["analytical"] == 0


def test_classifier_operational_decision_dispatches_operational(coord, monkeypatch):
    # When the classifier positively says operational (RAG), op path runs.
    async def fake(question):
        return {"route": "operational", "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None}
    monkeypatch.setattr(coord, "_classify", fake)
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("what does science say about training frequency?"))
    assert result["route"] == "operational"
    assert seen["operational"] == 1 and seen["analytical"] == 0


def test_out_of_scope_unchanged(coord, monkeypatch):
    async def fake(question):
        return {"route": "out_of_scope", "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False, "custom_sql_intent": None}
    monkeypatch.setattr(coord, "_classify", fake)
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("write me a poem"))
    assert result["route"] == "out_of_scope"
    assert seen["operational"] == 0 and seen["analytical"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3.5 · dispatch given a classification
#
# These tests gate the DISPATCH layer only: GIVEN a classification result, does
# the coordinator send the question to the correct lane? They stub _classify (the
# instance method, no Gemini call) and observe which lane runs via _spy_downstream.
# They do NOT prove the live model classifies any question correctly — that is
# classification ACCURACY, covered by stage 4's live two-run check. The seam is
# clean: _classify is already monkeypatch-stubbable; no production testability
# change is needed. (Writes are the one exception — they dispatch via the
# write-intent pre-guard BEFORE _classify; see the pre-guard test below.)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_returning(route):
    """An async _classify stub returning a params dict with the given route."""
    async def fake(question):
        return {"route": route, "exercise_names": None, "muscle_groups": None,
                "query_period_days": 90, "needs_custom_sql": False,
                "custom_sql_intent": None}
    return fake


@pytest.mark.parametrize("question,stub_route,exp_an,exp_op", [
    # Load-bearing display cases (the category row was the live-misrouting one).
    ("show me my last Lat Pulldown session",                "analytical",  1, 0),
    ("how was my back ROM split in the last back session",  "analytical",  1, 0),
    # Non-display analytical read.
    ("am I progressing on squat",                           "analytical",  1, 0),
    # Research / RAG → operational.
    ("what does science say about creatine",                "operational", 0, 1),
])
def test_dispatch_classification_to_lane(coord, monkeypatch, question,
                                         stub_route, exp_an, exp_op):
    """Dispatch given a classification: a question classified `stub_route` is sent
    to the matching lane. This pins DISPATCH, not whether the live model
    classifies these questions correctly (stage 4's two-run check covers that).
    The two display rows lock display→analytical dispatch in place."""
    monkeypatch.setattr(coord, "_classify", _classify_returning(stub_route))
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route(question))
    assert result["route"] == stub_route
    assert seen["analytical"] == exp_an
    assert seen["operational"] == exp_op


def test_dispatch_out_of_scope_zero_spend(coord, monkeypatch):
    """Dispatch given an out_of_scope classification: refuse with ZERO pipeline
    spend (neither lane invoked). Dispatch contract, not model accuracy."""
    monkeypatch.setattr(coord, "_classify", _classify_returning("out_of_scope"))
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("write me a poem about squats"))
    assert result["route"] == "out_of_scope"
    assert seen["analytical"] == 0 and seen["operational"] == 0


def test_dispatch_write_via_preguard_ignores_classification(coord, monkeypatch):
    """Writes dispatch via the write-intent pre-guard BEFORE _classify is even
    consulted. Stub _classify to the 'wrong' lane (analytical) and assert an
    imperative write STILL routes operational — proving the pre-guard owns write
    dispatch. Dispatch contract, not model accuracy."""
    monkeypatch.setattr(coord, "_classify", _classify_returning("analytical"))
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route("log my bench 100x5"))
    assert result["route"] == "operational"
    assert seen["operational"] == 1 and seen["analytical"] == 0


def test_dispatch_follows_route_not_text(coord, monkeypatch):
    """Inverse of the display rows of test_dispatch_classification_to_lane: the
    SAME display question, but classified `operational`, dispatches operational.
    Paired with that test (same text, classified analytical → analytical), this
    permanently proves the coordinator dispatches on the classification RESULT,
    not the question TEXT — so the display→analytical tests are non-tautological.
    Pins dispatch-follows-route; it is NOT a claim about how display SHOULD be
    classified (post-stage-3 the model classifies display analytical)."""
    question = "how was my back ROM split in the last back session"
    monkeypatch.setattr(coord, "_classify", _classify_returning("operational"))
    seen = _spy_downstream(coord, monkeypatch)
    result = asyncio.run(coord.route(question))
    assert result["route"] == "operational"
    assert seen["operational"] == 1 and seen["analytical"] == 0


# ── Prompt / docstring presence for the flip ────────────────────────────────

def test_classify_prompt_default_is_analytical():
    from src.coordinator import _CLASSIFY_SYSTEM
    low = _CLASSIFY_SYSTEM.lower()
    assert 'default to "analytical"' in _CLASSIFY_SYSTEM
    assert "positive allowlist" in low
    # operational now constrained to TWO cases (writes + research/RAG); session
    # display moved to the analytical lane (stage 3 boundary flip).
    assert "research" in low
    assert "two cases" in low
    # session display is now ANALYTICAL (single-line fragment, robust to wrapping)
    assert "session display — single-exercise or" in low
    # the old operational allowlist phrasing for display is gone
    assert "specific-date session display" not in low
    # medical is a read → analytical by default, never out_of_scope
    assert "NEVER out_of_scope" in _CLASSIFY_SYSTEM


def test_module_docstring_rationale_flipped():
    assert "reads default ANALYTICAL" in coordinator_mod.__doc__
    assert "default to operational" not in coordinator_mod.__doc__.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — strip the four analysis-read tools from the operational agent
# ══════════════════════════════════════════════════════════════════════════════

_STRIPPED = {"get_personal_record", "get_weekly_volume",
             "query_workout_data", "run_read_only_sql"}
# Stage 4a: the three read/display tools moved off the operational agent —
# session-display is served by the analytical lane (src/data_agent/session_display.py).
_UNEXPOSED_DISPLAY = {"get_exercise_sessions", "read_exercise_comments",
                      "get_exercise_history"}
# resolve_exercise_name STAYS exposed — writes need a pre-resolved exact name.
_KEPT_READS = {"resolve_exercise_name"}
_KEPT_OTHER = {"log_workout", "update_workout_set",
               "delete_workout_set", "set_goal", "search_fitness_knowledge",
               "remember_fact", "recall_memories"}
# Fix 5: workout execute+verify moved off the agent — the server/CLI drives
# execute_staged_workout via session.call_tool on explicit confirm, and verify
# happens inside the execute transaction. Unexposed, handlers kept.
_UNEXPOSED_EXECUTE = {"execute_staged_workout", "verify_workout_logged"}


def _exposed_tool_names():
    from mcp_servers.combined_server import list_tools
    return {t.name for t in asyncio.run(list_tools())}


def test_four_analysis_read_tools_unexposed():
    names = _exposed_tool_names()
    assert _STRIPPED.isdisjoint(names), f"still exposed: {_STRIPPED & names}"


def test_display_read_tools_unexposed_stage4a():
    # Stage 4a: the three read/display tools are no longer exposed to the
    # operational agent; resolve_exercise_name remains (writes depend on it).
    names = _exposed_tool_names()
    assert _UNEXPOSED_DISPLAY.isdisjoint(names), (
        f"still exposed to operational: {_UNEXPOSED_DISPLAY & names}")
    assert "resolve_exercise_name" in names


def test_display_handlers_still_exist_unexposed_not_deleted():
    # Unexpose, don't delete: the _sync handlers stay for analytical reuse /
    # evals / tests (e.g. tests/test_session_display.py imports
    # _get_exercise_sessions_sync as the equality oracle).
    import mcp_servers.combined_server as srv
    for fn in ("_get_exercise_sessions_sync", "_get_exercise_history_sync",
               "_read_exercise_comments_sync"):
        assert hasattr(srv, fn), f"{fn} must remain (unexpose, not delete)"


def test_kept_tools_still_exposed():
    names = _exposed_tool_names()
    missing = (_KEPT_READS | _KEPT_OTHER) - names
    assert not missing, f"unexpectedly removed: {missing}"


def test_workout_execute_verify_unexposed_fix5():
    names = _exposed_tool_names()
    assert _UNEXPOSED_EXECUTE.isdisjoint(names), (
        f"still exposed to operational: {_UNEXPOSED_EXECUTE & names}")
    # Sibling staged flows keep their agent-driven execute tools.
    assert "execute_staged_goal" in names


def test_unused_reuse_functions_removed():
    # Post-#7 cleanup: of the four Step-C-unexposed read functions, the "kept for
    # reuse/evals" justification only ever held for get_weekly_volume's _sync
    # (genuinely reused by tests/test_operational_volume.py). The other three had
    # no caller — grep-proven — and were removed (handler + async wrapper + the
    # dead call_tool dispatch branch). They must stay gone.
    import mcp_servers.combined_server as srv
    for fn in ("_get_personal_record_sync", "_query_workout_data_sync",
               "_run_read_only_sql"):
        assert not hasattr(srv, fn), f"{fn} should have been removed (dead code)"
    assert hasattr(srv, "_get_weekly_volume_sync")   # genuinely reused — kept


# ── SYSTEM_PROMPT: removed-tool refs gone, write/display flow intact (no strand)

def test_system_prompt_drops_removed_tools():
    from src.agent import SYSTEM_PROMPT
    # Stage 4a prompt cleanup: the prompt must not advertise ANY unexposed read
    # tool — the four Step-C reads AND the three display reads. A prompt that still
    # names a tool the model can't call is the fabrication mechanism.
    for t in (_STRIPPED | _UNEXPOSED_DISPLAY):
        assert t not in SYSTEM_PROMPT, f"SYSTEM_PROMPT still references {t}"


def test_system_prompt_keeps_write_flow_and_resolve():
    from src.agent import SYSTEM_PROMPT
    # write/edit flow still uses the kept tools and reaches the confirmation gate
    assert "WRITE ACTIONS" in SYSTEM_PROMPT
    assert "resolve_exercise_name" in SYSTEM_PROMPT          # kept — writes need it
    assert "update_workout_set" in SYSTEM_PROMPT and "delete_workout_set" in SYSTEM_PROMPT
    # the prompt now states operational does NOT read/analyze workout data, so a
    # misrouted read declines honestly instead of fabricating
    assert "DO NOT READ OR ANALYZE WORKOUT DATA" in SYSTEM_PROMPT
