"""
WAL test-isolation guard: no test run may ever write the real
data/agent_writes.json again (986 junk records accumulated before this
existed). Three layers under test:

  1. conftest's autouse _isolated_wal fixture (WAL_PATH + AGENT_WRITES_PATH
     point at tmp_path for every test),
  2. wal._effective_path()'s belt-and-suspenders pytest redirect for code
     paths nobody isolated,
  3. the previously-offending in-process route
     (cs._log_workout_sync -> cs._execute_staged_workout_sync -> _wal_append)
     now lands in the isolated WAL while the default path stays untouched.
"""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mcp_servers.combined_server as cs       # noqa: E402
from src import wal                            # noqa: E402


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── _effective_path() decision table ──────────────────────────────────────────

def test_effective_path_redirects_when_pytest_and_unisolated(monkeypatch):
    # Simulate the pre-fix danger state: default WAL_PATH, no env override,
    # running under pytest (PYTEST_CURRENT_TEST is genuinely set right now).
    monkeypatch.setattr(wal, "WAL_PATH", wal._DEFAULT_WAL_PATH)
    monkeypatch.delenv("AGENT_WRITES_PATH", raising=False)
    assert "PYTEST_CURRENT_TEST" in os.environ
    path = wal._effective_path()
    assert path != wal._DEFAULT_WAL_PATH
    assert os.path.dirname(path) == tempfile.gettempdir()
    assert "pytest" in os.path.basename(path)


def test_effective_path_honors_explicit_override(monkeypatch, tmp_path):
    override = str(tmp_path / "my_wal.json")
    monkeypatch.setattr(wal, "WAL_PATH", override)
    monkeypatch.setenv("AGENT_WRITES_PATH", override)
    assert wal._effective_path() == override


def test_effective_path_default_outside_pytest(monkeypatch):
    monkeypatch.setattr(wal, "WAL_PATH", wal._DEFAULT_WAL_PATH)
    monkeypatch.delenv("AGENT_WRITES_PATH", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert wal._effective_path() == wal._DEFAULT_WAL_PATH


def test_effective_path_patched_walpath_wins_even_without_env(monkeypatch, tmp_path):
    # A test that patches only wal.WAL_PATH (the pre-existing per-test
    # pattern in test_write_path_comments_cardio.py) keeps full control —
    # the pytest redirect must not hijack it.
    patched = str(tmp_path / "patched_wal.json")
    monkeypatch.setattr(wal, "WAL_PATH", patched)
    monkeypatch.delenv("AGENT_WRITES_PATH", raising=False)
    assert wal._effective_path() == patched


# ── append_write never touches the default path under pytest ─────────────────

def test_append_under_pytest_leaves_default_path_untouched(monkeypatch, tmp_path):
    # Danger state again, but with a sentinel "real" WAL sitting at the
    # default relative path in a scratch CWD.
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    sentinel = os.path.join("data", "agent_writes.json")
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write("[]")
    before = _sha(sentinel)

    monkeypatch.setattr(wal, "WAL_PATH", wal._DEFAULT_WAL_PATH)
    monkeypatch.delenv("AGENT_WRITES_PATH", raising=False)
    rec_id = wal.append_write("execute_staged_workout", {"exercise_id": 1})

    assert _sha(sentinel) == before          # negative: real path untouched
    redirect = wal._effective_path()
    with open(redirect, encoding="utf-8") as f:
        ids = [r["id"] for r in json.load(f)]
    assert rec_id in ids                     # redirected, not dropped


# ── The previously-offending route lands in the isolated WAL ──────────────────

def test_stage_execute_journals_to_isolated_wal(monkeypatch, tmp_path):
    # conftest's _isolated_wal has already pointed wal.WAL_PATH at tmp_path.
    assert wal.WAL_PATH != wal._DEFAULT_WAL_PATH
    db = str(tmp_path / "w.fitnotes")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY, exercise_id INTEGER, date DATE,
            metric_weight REAL, reps INTEGER,
            unit INTEGER NOT NULL DEFAULT 0, is_personal_record INTEGER,
            is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0,
            duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE Comment (_id INTEGER PRIMARY KEY, date DATE,
            owner_type_id INTEGER, owner_id INTEGER, comment TEXT);
        INSERT INTO exercise VALUES (1, 'Test Press', 5);
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(cs, "DB_PATH", db)
    cs._staged_writes.clear()

    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Test Press", "date": "2026-06-01",
        "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5}],
    }))
    assert "error" not in staged, staged
    done = json.loads(cs._execute_staged_workout_sync())
    assert done.get("success"), done

    # Journaled (guard redirects, never drops — replay integrity intact) ...
    records = wal.get_records()
    assert len(records) == 1
    assert records[0]["tool"] == "execute_staged_workout"
    assert records[0]["status"] == "pending"
    # ... into the isolated file, which is under this test's tmp_path.
    assert os.path.dirname(wal.WAL_PATH) == str(tmp_path)
    assert os.path.exists(wal.WAL_PATH)


# ── Post-purge shape ───────────────────────────────────────────────────────────

def test_empty_wal_replay_is_noop(tmp_path):
    wal._save([])
    assert wal.get_records() == []
    ghost_db = str(tmp_path / "never_created.fitnotes")
    summary = wal.replay_writes(ghost_db)
    assert summary == {"replayed": 0, "conflicts": 0, "errors": []}
    assert not os.path.exists(ghost_db)      # early return never opens the DB
