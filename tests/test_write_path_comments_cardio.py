"""
Write-path Fixes 1+2: log_workout now persists per-set comments and cardio
distance/duration (previously silently dropped). Covers the stage→execute live
write AND the WAL replay, proving they produce identical rows via the shared
src.db writer helpers. No server, no Gemini — a temp SQLite DB mirroring the real
NOT-NULL-DEFAULT-0 columns.
"""

import json
import os
import sqlite3
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mcp_servers.combined_server as cs       # noqa: E402
from src import wal                            # noqa: E402

CARDIO_CAT = 8


def _make_db(path):
    """Temp DB with exercise / training_log / Comment, mirroring the real schema's
    NOT NULL DEFAULT 0 metric columns so a NULL regression would fail loudly."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY,
            exercise_id INTEGER,
            date DATE,
            metric_weight REAL,
            reps INTEGER,
            unit INTEGER NOT NULL DEFAULT 0,
            is_personal_record INTEGER,
            is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0,
            duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE Comment (
            _id INTEGER PRIMARY KEY,
            date DATE,
            owner_type_id INTEGER,
            owner_id INTEGER,
            comment TEXT
        );
        INSERT INTO exercise (_id, name, category_id) VALUES
            (1, 'Test Press', 5),     -- strength (Back)
            (2, 'Treadmill', 8);      -- cardio
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = str(tmp_path / "w.fitnotes")
    _make_db(p)
    monkeypatch.setattr(cs, "DB_PATH", p)
    cs._staged_writes.clear()
    return p


def _rows(db_path, table="training_log"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out = [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY _id")]
    conn.close()
    return out


def _stage_execute(args):
    staged = json.loads(cs._log_workout_sync(args))
    assert "error" not in staged, staged
    done = json.loads(cs._execute_staged_workout_sync())
    assert done.get("success"), done
    return staged, done


# ── Execute (live write) ──────────────────────────────────────────────────────

def test_strength_set_with_comment_writes_row_and_comment(db):
    _stage_execute({
        "exercise_name": "Test Press", "date": "2026-06-01",
        "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5, "comment": "felt strong"}],
    })
    tl = _rows(db)
    assert len(tl) == 1
    co = _rows(db, "Comment")
    assert len(co) == 1
    assert co[0]["owner_type_id"] == 1
    assert co[0]["owner_id"] == tl[0]["_id"]      # bound to THIS set
    assert co[0]["comment"] == "felt strong"
    assert co[0]["date"] == "2026-06-01"


def test_cardio_distance_duration_writes_unit3_no_pr(db):
    _stage_execute({
        "exercise_name": "Treadmill", "date": "2026-06-02",
        "sets": [{"distance": 2.0, "duration_seconds": 1080}],
    })
    r = _rows(db)[0]
    assert r["metric_weight"] == 0 and r["reps"] == 0
    assert r["distance"] == 2.0 and r["duration_seconds"] == 1080
    assert r["unit"] == 3
    assert r["is_personal_record"] == 0


def test_cardio_duration_only_writes_unit2_distance_zero(db):
    _stage_execute({
        "exercise_name": "Treadmill", "date": "2026-06-03",
        "sets": [{"duration_seconds": 600}],
    })
    r = _rows(db)[0]
    assert r["duration_seconds"] == 600
    assert r["unit"] == 2
    assert r["distance"] == 0          # NOT NULL default honored as 0


def test_strength_unit_stays_zero_regression(db):
    # The vestigial constant must remain 0, and cardio columns default to 0.
    _stage_execute({
        "exercise_name": "Test Press", "date": "2026-06-04",
        "sets": [{"weight": 80.0, "unit": "kg", "reps": 8}],
    })
    r = _rows(db)[0]
    assert r["unit"] == 0
    assert r["distance"] == 0 and r["duration_seconds"] == 0
    assert _rows(db, "Comment") == []          # no comment → no Comment row


def test_multi_set_comment_lands_on_correct_owner(db):
    # 3 sets, only the MIDDLE one has a comment → it must bind to set #2's _id,
    # not the first or last (off-by-one against lastrowid).
    _stage_execute({
        "exercise_name": "Test Press", "date": "2026-06-05",
        "sets": [
            {"weight": 100.0, "unit": "lbs", "reps": 5},
            {"weight": 110.0, "unit": "lbs", "reps": 3, "comment": "grindy"},
            {"weight": 120.0, "unit": "lbs", "reps": 1},
        ],
    })
    tl = _rows(db)
    co = _rows(db, "Comment")
    assert len(tl) == 3 and len(co) == 1
    middle = tl[1]
    assert co[0]["owner_id"] == middle["_id"]
    assert middle["reps"] == 3              # confirm it's truly the 110×3 set


# ── Negative assertion — lbs/kg param never reaches the unit column ────────────

def test_unit_param_never_leaks_into_metric_code(db):
    for u in ("lbs", "kg"):
        cs._staged_writes.clear()
        cs._log_workout_sync({
            "exercise_name": "Test Press", "date": "2026-06-06",
            "sets": [{"weight": 100.0, "unit": u, "reps": 5}],
        })
        staged = cs._staged_writes["workout"]["sets"][0]
        assert staged["unit"] in (0, 2, 3)     # metric code only
        assert staged["unit"] == 0             # strength → 0, never the string
    # cardio staged unit is also a metric code
    cs._staged_writes.clear()
    cs._log_workout_sync({
        "exercise_name": "Treadmill", "date": "2026-06-06",
        "sets": [{"distance": 1.0, "duration_seconds": 600},
                 {"duration_seconds": 300}],
    })
    units = [s["unit"] for s in cs._staged_writes["workout"]["sets"]]
    assert units == [3, 2]


# ── WAL replay parity + dedup ─────────────────────────────────────────────────

def _replay(db_path, params):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        res = wal._replay_workout(conn, params)
        conn.commit()
        return res
    finally:
        conn.close()


def test_wal_replay_reproduces_comment_and_cardio(db):
    # A staged cardio workout with a comment, replayed into a fresh DB, must yield
    # the same training_log row + Comment the live execute path would.
    params = {"exercise_id": 2, "date": "2026-06-07", "sets": [
        {"metric_weight": 0, "reps": 0, "unit": 3, "distance": 5.0,
         "duration_seconds": 1500, "is_personal_record": 0, "comment": "easy 5k"},
    ]}
    _replay(db, params)
    r = _rows(db)[0]
    assert r["unit"] == 3 and r["distance"] == 5.0 and r["duration_seconds"] == 1500
    assert r["metric_weight"] == 0 and r["reps"] == 0
    co = _rows(db, "Comment")
    assert len(co) == 1 and co[0]["owner_id"] == r["_id"] and co[0]["comment"] == "easy 5k"


def test_wal_dedup_distinct_cardio_not_collapsed(db):
    # Two cardio entries on the same date/exercise differing ONLY in
    # distance/duration must BOTH replay — the dedup predicate is cardio-aware.
    params = {"exercise_id": 2, "date": "2026-06-08", "sets": [
        {"metric_weight": 0, "reps": 0, "unit": 3, "distance": 2.0, "duration_seconds": 1080},
        {"metric_weight": 0, "reps": 0, "unit": 3, "distance": 3.0, "duration_seconds": 1500},
    ]}
    res = _replay(db, params)
    assert "2 set(s) inserted" in res
    assert len(_rows(db)) == 2


def test_wal_dedup_identical_cardio_collapsed(db):
    # The SAME cardio session replayed twice → one row (idempotent replay).
    params = {"exercise_id": 2, "date": "2026-06-09", "sets": [
        {"metric_weight": 0, "reps": 0, "unit": 2, "distance": 0, "duration_seconds": 600},
    ]}
    _replay(db, params)
    assert len(_rows(db)) == 1
    with pytest.raises(wal._ReplayConflict):
        _replay(db, params)                    # second time: all sets already present
    assert len(_rows(db)) == 1                  # still one row
