"""
SqliteSaver checkpointer for the orchestration graphs.

Two-layer persistence (deliberate, documented deviation from full
replacement): the JSON checkpoint slot (src/checkpoint.py) keeps ALL domain
resume semantics — continue-intent, discard-confirm, 48h expiry, verbatim
draft, staged-slot restore — because they cross process restarts and the MCP
subprocess boundary. The SqliteSaver supplies the STRUCTURAL invariant: every
LLM call is its own graph node, so a checkpoint exists at the superstep
boundary immediately before each LLM call (proactive, vs. the hand-built
reactive-on-429 saves, which also remain, verbatim, inside the node bodies).

Thread naming: one fresh turn id per Coordinator.route() call —
``turn-{id}`` (parent), ``turn-{id}:analytical`` / ``turn-{id}:operational``
(subgraphs, invoked from facades on their own threads). Threads are pruned
best-effort on clean completion; aborted-run threads are harmless leftovers
because user-facing resume flows through the JSON slot with a new turn id.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _make_async_compat_saver_class():
    """Sync SqliteSaver + executor-based async methods.

    langgraph-checkpoint-sqlite's sync saver deliberately raises
    NotImplementedError from its a* methods; AsyncSqliteSaver needs an
    aiosqlite connection whose lifetime is bound to one event loop — but the
    CLI/tests run one asyncio.run() per turn, so the saver must be
    loop-agnostic. asyncio.to_thread over the sync methods (with
    check_same_thread=False) gives exactly the executor-wrapper behavior the
    graphs need. (Flagged deviation: the design assumed built-in wrappers.)
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    class AsyncCompatSqliteSaver(SqliteSaver):
        async def aget_tuple(self, config):
            return await asyncio.to_thread(self.get_tuple, config)

        async def alist(self, config, *, filter=None, before=None, limit=None):
            items = await asyncio.to_thread(
                lambda: list(self.list(config, filter=filter, before=before, limit=limit))
            )
            for item in items:
                yield item

        async def aput(self, config, checkpoint, metadata, new_versions):
            return await asyncio.to_thread(
                self.put, config, checkpoint, metadata, new_versions)

        async def aput_writes(self, config, writes, task_id, task_path=""):
            return await asyncio.to_thread(
                self.put_writes, config, writes, task_id, task_path)

        async def adelete_thread(self, thread_id):
            return await asyncio.to_thread(self.delete_thread, thread_id)

    return AsyncCompatSqliteSaver

# Worst case operational run: 12 iterations x (agent_step + exec_tools) plus
# prepare/finalize supersteps ~ 28. LangGraph's default recursion_limit of 25
# would raise GraphRecursionError BEFORE the hand-built 12-iteration guard
# fires; 60 is headroom only — the iteration counter in state is the guard.
RECURSION_LIMIT = 60

_saver = None
_saver_path: str | None = None


def _default_path() -> str:
    return os.environ.get(
        "GRAPH_CHECKPOINT_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "graph_checkpoints.sqlite"),
    )


def get_saver():
    """Module-level singleton SqliteSaver, rebuilt if GRAPH_CHECKPOINT_PATH
    changes (tests point it at tmp). Falls back to InMemorySaver if the
    sqlite file cannot be opened (e.g. cloud-sync lock): the structural
    checkpoint layer degrades gracefully rather than breaking answers —
    domain resume never depended on it.
    """
    global _saver, _saver_path
    path = _default_path()
    if _saver is not None and path == _saver_path:
        return _saver
    if _saver is not None:
        # Release the previous sqlite handle so a test's tmp dir can be
        # removed on Windows.
        try:
            conn = getattr(_saver, "conn", None)
            if conn is not None:
                conn.close()
        except Exception:
            pass
    try:
        saver_cls = _make_async_compat_saver_class()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        _saver = saver_cls(conn)
    except Exception as e:
        logger.warning(
            "[graph] SqliteSaver unavailable at %s (%s) — falling back to "
            "InMemorySaver (per-process structural checkpoints only)", path, e)
        from langgraph.checkpoint.memory import InMemorySaver
        _saver = InMemorySaver()
    _saver_path = path
    return _saver


def new_turn_id() -> str:
    return uuid.uuid4().hex


def turn_config(turn_id: str, ns: str = "") -> dict:
    thread = f"turn-{turn_id}" + (f":{ns}" if ns else "")
    return {
        "configurable": {"thread_id": thread},
        "recursion_limit": RECURSION_LIMIT,
    }


def cleanup_turn(turn_id: str) -> None:
    """Best-effort single-slot hygiene: drop the turn's threads after a clean
    completion. Never raises — checkpoint pruning must not mask an answer.
    """
    saver = get_saver()
    for ns in ("", "analytical", "operational"):
        thread = f"turn-{turn_id}" + (f":{ns}" if ns else "")
        try:
            saver.delete_thread(thread)
        except Exception:
            pass
