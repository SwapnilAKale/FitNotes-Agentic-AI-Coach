"""Shared test hygiene for the graph checkpointer, the WAL, and the DB.

Every test gets an isolated GRAPH_CHECKPOINT_PATH so the structural
SqliteSaver checkpoints never land in the repo's data/ directory and never
leak between test modules, an isolated WAL path so no test can journal
into the real data/agent_writes.json (986 junk records accumulated before
this existed), and an isolated COPY of the real FitNotes DB so no test can
write rows into data/FitNotes_Backup.fitnotes (77 junk training_log rows
accumulated before this existed). The DB copy keeps every read-realism
value pin passing while diverting stray writes onto the throwaway.
Purely additive — existing tests manage their own seams (fake Gemini
clients, CHECKPOINT_PATH for the JSON slot, per-test WAL_PATH/DB_PATH
patches) exactly as before.
"""

import os
import shutil
import sys

import pytest

from src import wal

_REAL_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "FitNotes_Backup.fitnotes")


@pytest.fixture(autouse=True)
def _isolated_graph_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_CHECKPOINT_PATH", str(tmp_path / "graph_checkpoints.sqlite"))
    yield


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_copy = str(tmp_path / "FitNotes_Backup.fitnotes")
    shutil.copyfile(_REAL_DB, db_copy)
    # Call-time env readers and MCP subprocesses (agent.py forwards
    # FITNOTES_DB_PATH explicitly into the child).
    monkeypatch.setenv("FITNOTES_DB_PATH", db_copy)
    # combined_server reads DB_PATH once at import — patch lazily so tests
    # that never touch it don't pay the import; modules imported mid-test
    # pick up the env var instead.
    cs_mod = sys.modules.get("mcp_servers.combined_server")
    if cs_mod is not None:
        monkeypatch.setattr(cs_mod, "DB_PATH", db_copy)
    yield


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    # src/settings.py reads SETTINGS_PATH at call time — env alone isolates.
    monkeypatch.setenv("SETTINGS_PATH", str(tmp_path / "settings.json"))
    yield


@pytest.fixture(autouse=True)
def _isolated_checkpoint(tmp_path, monkeypatch):
    # src/checkpoint.py reads CHECKPOINT_PATH at call time — env alone isolates, so
    # no test can pollute the real data/checkpoint.json JSON slot (a fake staged
    # "log bench 100x5" write leaked into it once before this existed).
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))
    yield


@pytest.fixture(autouse=True)
def _isolated_wal(tmp_path, monkeypatch):
    wal_path = str(tmp_path / "agent_writes.json")
    # In-process writes (combined_server helpers called directly by tests).
    monkeypatch.setattr(wal, "WAL_PATH", wal_path)
    # Subprocesses spawned by a test (live MCP servers) inherit the env.
    monkeypatch.setenv("AGENT_WRITES_PATH", wal_path)
    yield
