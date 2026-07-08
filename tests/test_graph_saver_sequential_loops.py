"""
AsyncCompatSqliteSaver must survive sequential asyncio.run() turns.

The CLI runs ONE asyncio.run per user turn, so every event loop dies between
turns. The graph checkpointer must therefore be loop-agnostic: a plain
sqlite3 connection (check_same_thread=False) driven through asyncio.to_thread
wrappers — never an aiosqlite connection or cached loop reference, which
would bind the saver to the first turn's dead loop.

This test compiles the real operational graph with the AsyncCompat saver and
invokes it in TWO separate asyncio.run() calls on the SAME thread id,
asserting the second loop can both read the first loop's persisted
checkpoint and run the graph again without loop-binding or thread errors.
"""

import asyncio
import os

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.agent import AgentSession
from src.graph import persistence as persistence_mod
from src.graph.operational import get_operational_graph
from src.graph.state import GraphRunContext, RunCache


def _scripted_session(text):
    """AgentSession via __init__ only — no MCP, no Gemini; one no-tool step."""
    s = AgentSession("data/unused.fitnotes")
    s._run_collect = lambda contents, config: (text, [], None)
    return s


def test_saver_survives_sequential_asyncio_run_on_same_thread():
    graph = get_operational_graph()
    saver = persistence_mod.get_saver()
    config = {"configurable": {"thread_id": "seq-loop-audit"},
              "recursion_limit": persistence_mod.RECURSION_LIMIT}

    async def turn(text):
        return await graph.ainvoke(
            {"question": "hello"},
            config,
            context=GraphRunContext(session=_scripted_session(text), cache=RunCache()),
        )

    # Turn 1: its own event loop, checkpoints written through the saver.
    r1 = asyncio.run(turn("first answer"))
    assert r1["result"]["answer"] == "first answer"

    # Turn 2: a brand-new event loop. First prove the saver's ASYNC read path
    # works in this second loop (an aiosqlite/loop-bound saver dies here),
    # then run the same thread again end-to-end.
    async def second_turn():
        tup = await saver.aget_tuple(config)
        assert tup is not None, "second loop must read the first loop's checkpoint"
        return await turn("second answer")

    r2 = asyncio.run(second_turn())
    assert r2["result"]["answer"] == "second answer"
