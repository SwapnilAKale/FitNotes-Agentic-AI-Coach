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

import logging
import os
import sqlite3
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

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
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        _saver = SqliteSaver(conn)
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
