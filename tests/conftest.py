"""Shared test hygiene for the graph checkpointer and the WAL.

Every test gets an isolated GRAPH_CHECKPOINT_PATH so the structural
SqliteSaver checkpoints never land in the repo's data/ directory and never
leak between test modules, and an isolated WAL path so no test can journal
into the real data/agent_writes.json (986 junk records accumulated before
this existed). Purely additive — existing tests manage their own seams
(fake Gemini clients, CHECKPOINT_PATH for the JSON slot, per-test WAL_PATH
patches) exactly as before.
"""

import pytest

from src import wal


@pytest.fixture(autouse=True)
def _isolated_graph_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_CHECKPOINT_PATH", str(tmp_path / "graph_checkpoints.sqlite"))
    yield


@pytest.fixture(autouse=True)
def _isolated_wal(tmp_path, monkeypatch):
    wal_path = str(tmp_path / "agent_writes.json")
    # In-process writes (combined_server helpers called directly by tests).
    monkeypatch.setattr(wal, "WAL_PATH", wal_path)
    # Subprocesses spawned by a test (live MCP servers) inherit the env.
    monkeypatch.setenv("AGENT_WRITES_PATH", wal_path)
    yield
