"""Shared test hygiene for the graph checkpointer.

Every test gets an isolated GRAPH_CHECKPOINT_PATH so the structural
SqliteSaver checkpoints never land in the repo's data/ directory and never
leak between test modules. Purely additive — existing tests manage their own
seams (fake Gemini clients, CHECKPOINT_PATH for the JSON slot) exactly as
before.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_graph_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_CHECKPOINT_PATH", str(tmp_path / "graph_checkpoints.sqlite"))
    yield
