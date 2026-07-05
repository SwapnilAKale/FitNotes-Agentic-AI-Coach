"""
Graph-checkpointer state audit — the package-exclusion hard constraint.

The LangGraph SqliteSaver persists graph state at every superstep boundary
(that is what puts a checkpoint immediately before every LLM node). The
analytical package (~366KB) must NEVER appear in that persisted state: it
rides in the non-persisted RunCache context and rebuilds free from params.
These tests run the real analytical subgraph end-to-end with the LLM stages
stubbed and then read the raw persisted checkpoints back out of the saver:

  * params and the VERBATIM draft are present (they are what resume needs),
  * no package marker ever appears in any persisted checkpoint,
  * every persisted state stays orders of magnitude below package size,
  * a checkpoint exists before each LLM stage node (draft/ground/coverage).
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src import analysis_agent as analysis_mod
from src import coordinator as coordinator_mod
from src.coordinator import Coordinator

PKG_MARKER = "PKG_MARKER_MUST_NEVER_PERSIST"
DRAFT_TEXT = "THE-VERBATIM-DRAFT-TEXT for the audit."
FINAL_TEXT = "THE-GROUNDED-ANSWER for the audit."


@pytest.fixture()
def coord(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)

    async def fake_classify(question):
        return {"route": "analytical", "exercise_names": None,
                "muscle_groups": None, "query_period_days": 90,
                "needs_custom_sql": False, "custom_sql_intent": None}
    monkeypatch.setattr(c, "_classify", fake_classify)

    # A "package" big enough that any leak into persisted state is obvious.
    def fake_pkg(**kwargs):
        return {"marker": PKG_MARKER, "blob": "x" * 400_000, "scope": "broad"}
    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package", fake_pkg)

    async def fake_analyze(pkg, question, research, memories, context, custom_query=None):
        assert pkg.get("marker") == PKG_MARKER   # nodes see the real package
        return DRAFT_TEXT
    monkeypatch.setattr(analysis_mod, "analyze", fake_analyze)

    async def fake_ground(draft, gctx):
        return FINAL_TEXT, []
    monkeypatch.setattr(analysis_mod, "ground_check", fake_ground)

    async def fake_coverage(question, answer):
        return answer, True
    monkeypatch.setattr(c, "_coverage_check", fake_coverage)
    return c


def _run_and_collect_checkpoints(coord, monkeypatch):
    """Run the analytical facade with thread pruning disabled, then return
    every checkpoint tuple persisted for the run's analytical thread."""
    from src.graph import persistence as persistence_mod

    kept: list = []
    monkeypatch.setattr(persistence_mod, "cleanup_turn",
                        lambda turn_id: kept.append(turn_id))

    answer, flagged = asyncio.run(
        coord._run_analytical("audit question", {"query_period_days": 90}))
    assert answer == FINAL_TEXT and flagged == []
    assert len(kept) == 1

    saver = persistence_mod.get_saver()
    thread = f"turn-{kept[0]}:analytical"
    tuples = list(saver.list({"configurable": {"thread_id": thread}}))
    assert tuples, "the analytical run must have persisted checkpoints"
    return tuples


def test_package_never_persisted_but_params_and_draft_are(coord, monkeypatch):
    tuples = _run_and_collect_checkpoints(coord, monkeypatch)

    serialized_states = [
        json.dumps(t.checkpoint.get("channel_values", {}), default=str)
        for t in tuples
    ]

    # The package (or anything derived from it) never enters persisted state.
    for s in serialized_states:
        assert PKG_MARKER not in s
        assert "xxxxxxxxxx" not in s          # the 400KB blob
        # Persisted state stays far below package size (params + small strings).
        assert len(s) < 50_000

    # What resume needs IS persisted: params and the verbatim draft text.
    joined = "\n".join(serialized_states)
    assert "query_period_days" in joined
    assert DRAFT_TEXT in joined               # draft persisted verbatim


def test_checkpoint_exists_before_each_llm_stage(coord, monkeypatch):
    tuples = _run_and_collect_checkpoints(coord, monkeypatch)

    # SqliteSaver writes one checkpoint per superstep: the state AFTER each
    # node is durable BEFORE the next node (the next LLM call) runs. The
    # linear chain must therefore leave one checkpoint per completed node.
    sources = [t.metadata.get("source") for t in tuples]
    assert sources.count("loop") >= 6, (
        "expected a superstep checkpoint per analytical node "
        f"(resolve_scope/build_package/draft/ground/coverage/display), got {sources}")

    # The sqlite artifact lives where GRAPH_CHECKPOINT_PATH points (the
    # autouse conftest fixture isolates it per test).
    assert os.path.exists(os.environ["GRAPH_CHECKPOINT_PATH"])
