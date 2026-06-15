"""
Step C, Parts 2 & 3: flip the read default to ANALYTICAL, then unexpose the four
analysis-read tools from the operational agent.

Part 2 — operational is a positive allowlist (writes via the write-intent regex
pre-guard, research/RAG, specific-date session display, + out_of_scope). Every
other read/coaching question, and anything uncertain (parse/exception fail),
defaults ANALYTICAL.

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

    async def op(q):
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
    # A low-confidence/parse-fail classifier on a read → analytical end-to-end.
    coord._client = _client_returning("not parseable")
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
    # When the classifier positively says operational (RAG/display), op path runs.
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


# ── Prompt / docstring presence for the flip ────────────────────────────────

def test_classify_prompt_default_is_analytical():
    from src.coordinator import _CLASSIFY_SYSTEM
    assert 'default to "analytical"' in _CLASSIFY_SYSTEM
    assert "positive allowlist" in _CLASSIFY_SYSTEM.lower()
    # operational still constrained to the three cases; out_of_scope intact
    assert "research" in _CLASSIFY_SYSTEM.lower() and "session display" in _CLASSIFY_SYSTEM.lower()
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
_KEPT_READS = {"get_exercise_sessions", "resolve_exercise_name",
               "read_exercise_comments", "get_exercise_history"}
_KEPT_OTHER = {"log_workout", "execute_staged_workout", "update_workout_set",
               "delete_workout_set", "set_goal", "search_fitness_knowledge",
               "remember_fact", "recall_memories"}


def _exposed_tool_names():
    from mcp_servers.combined_server import list_tools
    return {t.name for t in asyncio.run(list_tools())}


def test_four_analysis_read_tools_unexposed():
    names = _exposed_tool_names()
    assert _STRIPPED.isdisjoint(names), f"still exposed: {_STRIPPED & names}"


def test_kept_tools_still_exposed():
    names = _exposed_tool_names()
    missing = (_KEPT_READS | _KEPT_OTHER) - names
    assert not missing, f"unexpectedly removed: {missing}"


def test_stripped_functions_remain_for_reuse():
    # Unexposed, NOT deleted — the handler functions stay for analytical reuse/evals.
    import mcp_servers.combined_server as srv
    for fn in ("_get_personal_record_sync", "_get_weekly_volume_sync",
               "_query_workout_data_sync", "_run_read_only_sql"):
        assert hasattr(srv, fn), f"{fn} was deleted (should only be unexposed)"


# ── SYSTEM_PROMPT: removed-tool refs gone, write/display flow intact (no strand)

def test_system_prompt_drops_removed_tools():
    from src.agent import SYSTEM_PROMPT
    for t in _STRIPPED:
        assert t not in SYSTEM_PROMPT, f"SYSTEM_PROMPT still references {t}"


def test_system_prompt_keeps_write_and_display_flow():
    from src.agent import SYSTEM_PROMPT
    # write/edit flow still uses the kept tools and reaches the confirmation gate
    assert "WRITE ACTIONS" in SYSTEM_PROMPT
    assert "get_exercise_sessions" in SYSTEM_PROMPT
    assert "resolve_exercise_name" in SYSTEM_PROMPT
    assert "update_workout_set" in SYSTEM_PROMPT and "delete_workout_set" in SYSTEM_PROMPT
    # session-display rule intact
    assert "SESSION DISPLAY RULE" in SYSTEM_PROMPT
