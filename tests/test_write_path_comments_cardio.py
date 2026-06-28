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
        staged = cs._staged_writes["workout"][0]["sets"][0]   # slot is a list of workouts (Fix 3)
        assert staged["unit"] in (0, 2, 3)     # metric code only
        assert staged["unit"] == 0             # strength → 0, never the string
    # cardio staged unit is also a metric code
    cs._staged_writes.clear()
    cs._log_workout_sync({
        "exercise_name": "Treadmill", "date": "2026-06-06",
        "sets": [{"distance": 1.0, "duration_seconds": 600},
                 {"duration_seconds": 300}],
    })
    units = [s["unit"] for s in cs._staged_writes["workout"][0]["sets"]]   # list of workouts (Fix 3)
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


# ── Fix 3: batch staging (slot is a LIST of workouts) ─────────────────────────

def _add_exercise(db_path, _id, name, cat):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO exercise (_id, name, category_id) VALUES (?,?,?)",
                 (_id, name, cat))
    conn.commit()
    conn.close()


def test_batch_three_exercises_all_written(db):
    # 3 log_workout calls before ONE execute → all 3 exercises' sets land (not just
    # the last). Pre-Fix-3 the slot overwrote and only the last survived.
    _add_exercise(db, 3, "Squat", 6)
    _add_exercise(db, 4, "Curl", 3)
    for name, lbs in [("Test Press", 100.0), ("Squat", 200.0), ("Curl", 50.0)]:
        staged = json.loads(cs._log_workout_sync({
            "exercise_name": name, "date": "2026-06-10",
            "sets": [{"weight": lbs, "unit": "lbs", "reps": 5},
                     {"weight": lbs, "unit": "lbs", "reps": 5}]}))
        assert "error" not in staged, staged
    done = json.loads(cs._execute_staged_workout_sync())
    assert done["success"] and done["sets_written"] == 6 and done["exercises_written"] == 3
    by_ex = {}
    for r in _rows(db):
        by_ex[r["exercise_id"]] = by_ex.get(r["exercise_id"], 0) + 1
    assert by_ex == {1: 2, 3: 2, 4: 2}          # every exercise's 2 sets present
    assert cs._staged_writes.get("workout") is None   # slot cleared once


def test_batch_mixed_strength_cardio_comments(db):
    # One batch: strength+comment, distance cardio, duration-only cardio → all land.
    _add_exercise(db, 3, "Cycling", 8)
    for args in [
        {"exercise_name": "Test Press", "date": "2026-06-11",
         "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5, "comment": "solid"}]},
        {"exercise_name": "Treadmill", "date": "2026-06-11",
         "sets": [{"distance": 3.0, "duration_seconds": 1200, "comment": "tempo"}]},
        {"exercise_name": "Cycling", "date": "2026-06-11",
         "sets": [{"duration_seconds": 1800}]},
    ]:
        assert "error" not in json.loads(cs._log_workout_sync(args))
    done = json.loads(cs._execute_staged_workout_sync())
    assert done["success"] and done["exercises_written"] == 3
    tl = _rows(db)
    strength = next(r for r in tl if r["exercise_id"] == 1)
    dist     = next(r for r in tl if r["exercise_id"] == 2)
    dur_only = next(r for r in tl if r["exercise_id"] == 3)
    assert strength["reps"] == 5 and strength["unit"] == 0
    assert dist["unit"] == 3 and dist["distance"] == 3.0 and dist["duration_seconds"] == 1200
    assert dur_only["unit"] == 2 and dur_only["distance"] == 0 and dur_only["duration_seconds"] == 1800
    comments = {c["owner_id"]: c["comment"] for c in _rows(db, "Comment")}
    assert comments[strength["_id"]] == "solid" and comments[dist["_id"]] == "tempo"
    assert dur_only["_id"] not in comments       # no comment on the duration-only set


def test_batch_wal_one_record_per_workout(db, monkeypatch, tmp_path):
    # THE TRAP guard: a 3-exercise batch must journal 3 per-workout WAL records,
    # each params a single workout dict — NOT one record holding the list.
    monkeypatch.setattr(wal, "WAL_PATH", str(tmp_path / "wal.json"))
    _add_exercise(db, 3, "Squat", 6)
    for name in ("Test Press", "Treadmill", "Squat"):
        args = ({"exercise_name": name, "date": "2026-06-12",
                 "sets": [{"distance": 2.0, "duration_seconds": 900}]}
                if name == "Treadmill" else
                {"exercise_name": name, "date": "2026-06-12",
                 "sets": [{"weight": 80.0, "unit": "lbs", "reps": 5}]})
        cs._log_workout_sync(args)
    assert json.loads(cs._execute_staged_workout_sync())["success"]

    records = wal.get_records()
    assert len(records) == 3                      # one PER workout, not one list record
    for rec in records:
        assert rec["tool"] == "execute_staged_workout"
        p = rec["params"]
        assert isinstance(p, dict) and not isinstance(p, list)   # single workout, never a list
        assert "exercise_id" in p and isinstance(p["sets"], list)


def test_batch_single_exercise_still_works(db):
    # Regression: a list-of-one behaves exactly like the old single-workout case.
    assert "error" not in json.loads(cs._log_workout_sync({
        "exercise_name": "Test Press", "date": "2026-06-13",
        "sets": [{"weight": 120.0, "unit": "lbs", "reps": 3}]}))
    done = json.loads(cs._execute_staged_workout_sync())
    assert done["success"] and done["sets_written"] == 1 and done["exercises_written"] == 1
    assert len(_rows(db)) == 1


def _make_db_with_check(path):
    """Same schema as _make_db but training_log carries CHECK(reps >= 0) — a
    constraint SQLite genuinely ENFORCES (raises IntegrityError), unlike a bad-typed
    value that type affinity would silently accept."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY, exercise_id INTEGER, date DATE,
            metric_weight REAL, reps INTEGER CHECK (reps >= 0),
            unit INTEGER NOT NULL DEFAULT 0, is_personal_record INTEGER,
            is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0, duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE Comment (_id INTEGER PRIMARY KEY, date DATE, owner_type_id INTEGER,
                              owner_id INTEGER, comment TEXT);
        INSERT INTO exercise (_id, name, category_id) VALUES (1,'Test Press',5), (2,'Squat',6);
        """
    )
    conn.commit()
    conn.close()


def test_batch_cross_exercise_rollback(tmp_path, monkeypatch):
    # WHOLE batch is one transaction: exercise 1 valid, a LATER exercise fails →
    # exercise 1's sets roll back too (0 rows). Failure mode = CHECK(reps>=0), which
    # SQLite truly raises on (reps=-1), not a silently-accepted bad value.
    p = str(tmp_path / "chk.fitnotes")
    _make_db_with_check(p)
    monkeypatch.setattr(cs, "DB_PATH", p)

    # Non-vacuity guard: a control all-valid batch DOES write on this schema.
    cs._staged_writes.clear()
    cs._log_workout_sync({"exercise_name": "Test Press", "date": "2026-06-14",
                          "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5}]})
    cs._log_workout_sync({"exercise_name": "Squat", "date": "2026-06-14",
                          "sets": [{"weight": 200.0, "unit": "lbs", "reps": 3}]})
    assert json.loads(cs._execute_staged_workout_sync())["success"]
    assert len(_rows(p)) == 2                     # inserts genuinely happen absent a failure

    # Now exercise 1 valid + a LATER exercise with reps=-1 (violates CHECK).
    cs._staged_writes.clear()
    cs._log_workout_sync({"exercise_name": "Test Press", "date": "2026-06-15",
                          "sets": [{"weight": 100.0, "unit": "lbs", "reps": 5}]})
    cs._log_workout_sync({"exercise_name": "Squat", "date": "2026-06-15",
                          "sets": [{"weight": 200.0, "unit": "lbs", "reps": -1}]})
    out = json.loads(cs._execute_staged_workout_sync())
    assert "error" in out                         # the CHECK violation surfaced
    assert [r for r in _rows(p) if r["date"] == "2026-06-15"] == []   # whole batch rolled back


def test_batch_preview_accumulates_and_resets(monkeypatch):
    # Preview shows the WHOLE batch (not just the last exercise); sibling ops do not
    # accumulate; the per-turn reset (server.py:526 sets staging_preview="") starts clean.
    import asyncio
    import server as srv

    srv._state["staging_preview"] = ""           # mimic the start-of-turn reset (:526)
    for name in ("Test Press", "Squat", "Curl"):
        asyncio.run(srv._confirmation_handler(
            "log_workout", {"exercise_name": name, "date": "2026-06-16", "sets": []}))
    preview = srv._state["staging_preview"]
    assert all(n in preview for n in ("Test Press", "Squat", "Curl"))   # all 3 present

    # A sibling single-item op overwrites (does NOT accumulate onto the batch).
    asyncio.run(srv._confirmation_handler(
        "set_goal", {"exercise_name": "Bench", "target_weight": 225}))
    assert "Test Press" not in srv._state["staging_preview"]
    assert "Bench" in srv._state["staging_preview"]

    # Per-turn reset → a fresh stage starts clean (no stale bleed from the prior batch).
    srv._state["staging_preview"] = ""           # what server.py:526 does each /chat turn
    asyncio.run(srv._confirmation_handler(
        "log_workout", {"exercise_name": "Deadlift", "date": "2026-06-17", "sets": []}))
    assert "Squat" not in srv._state["staging_preview"]
    assert "Deadlift" in srv._state["staging_preview"]
