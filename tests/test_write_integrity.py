"""
Does a write check its own work?

WHY THIS EXISTS. Asked whether the agent's "successfully deleted" can be trusted,
the answer turned out not to be about the agent at all: five of the six write
operations never verify anything. `success: true` means only "the SQL raised no
exception". A DELETE matching zero rows raises nothing.

Proven below: stage a delete (staging DOES verify the row and resolve its `_id`),
remove the row out-of-band, execute — and the tool reports success while nothing
was deleted. The claim gate added in 76449f8 derives `db_write_effect` from that
same flag, so the guard against false claims rests on a fact that, for five of six
operations, does not mean what it says.

The one operation that does verify, `execute_staged_workout`, checks only the rows
it inserted. A write landing on the wrong date or a neighbouring row passes,
because the check re-reads only the scope it already knows about.

EVERY TEST RUNS ON A COPY of the database in tmp_path — never the real file.

THIS FILE IS A SAFETY NET, NOT A SPEC. It was written to pass against UNMODIFIED
code. Rows the arc deliberately changes sit in PRE-CHANGE blocks carrying their
current value; a row that moves without such a label is a regression.
"""

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import mcp_servers.combined_server as cs                       # noqa: E402

_REAL_DB = _ROOT / "data" / "FitNotes_Backup.fitnotes"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A COPY of the real database. Real schema — including the AUTOINCREMENT on
    training_log that makes sqlite_sequence churn on every insert, which any
    whole-file integrity check has to tolerate."""
    if not _REAL_DB.exists():
        pytest.skip("real database not present")
    p = str(tmp_path / "copy.fitnotes")
    shutil.copy2(_REAL_DB, p)
    monkeypatch.setattr(cs, "DB_PATH", p)
    cs._staged_writes.clear()
    return p


def _q(db, sql, args=()):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args)]
    con.close(); return out


def _newest_set(db):
    return _q(db, "SELECT * FROM training_log ORDER BY _id DESC LIMIT 1")[0]


def _name_of(db, exercise_id):
    return _q(db, "SELECT name FROM exercise WHERE _id=?", (exercise_id,))[0]["name"]


def _stage_delete(db, row):
    """Stage a delete for a real row. Staging is the half that already works."""
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": _name_of(db, row["exercise_id"]),
        "date": row["date"],
        "weight": round(row["metric_weight"] * 2.2046, 1),
        "reps": row["reps"],
        "unit": "lbs",
    }))
    assert out.get("staged") is True, out
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The half that already works — staging validates
# ══════════════════════════════════════════════════════════════════════════════

def test_staging_a_delete_rejects_a_row_that_does_not_exist(db):
    out = json.loads(cs._delete_workout_set_sync({
        "exercise_name": "Flat Barbell Bench Press",
        "date": "1999-01-01", "weight": 999.0, "reps": 99, "unit": "lbs",
    }))
    assert "error" in out and "No set found" in out["error"]


def test_staging_a_delete_resolves_the_row_id(db):
    row = _newest_set(db)
    _stage_delete(db, row)
    assert cs._staged_writes["delete_set"]["set_id"] == row["_id"]


def test_a_correct_delete_removes_exactly_that_row(db):
    row = _newest_set(db)
    before = len(_q(db, "SELECT _id FROM training_log"))
    _stage_delete(db, row)
    out = json.loads(cs._execute_staged_set_delete_sync())
    assert out["success"] is True
    after = _q(db, "SELECT _id FROM training_log")
    assert len(after) == before - 1
    assert row["_id"] not in {r["_id"] for r in after}


# ══════════════════════════════════════════════════════════════════════════════
# What the whole-file check must tolerate
# ══════════════════════════════════════════════════════════════════════════════

def test_training_log_uses_autoincrement_so_sqlite_sequence_churns(db):
    """Any whole-file integrity check MUST exclude sqlite_sequence, or every
    single insert would be rejected as unexpected damage."""
    sql = _q(db, "SELECT sql FROM sqlite_master WHERE name='training_log'")[0]["sql"]
    assert "AUTOINCREMENT" in sql.upper()

    before = _q(db, "SELECT * FROM sqlite_sequence WHERE name='training_log'")
    con = sqlite3.connect(db)
    con.execute("""INSERT INTO training_log
        (exercise_id, date, metric_weight, reps, unit, is_personal_record,
         is_complete, distance, duration_seconds)
        VALUES (1,'2026-01-01',1.0,1,0,0,1,0,0)""")
    con.commit(); con.close()
    after = _q(db, "SELECT * FROM sqlite_sequence WHERE name='training_log'")
    assert before != after


def test_every_write_reachable_table_keys_on_id(db):
    """The focus-map design assumes a uniform primary key across these tables."""
    for t in ("training_log", "Goal", "BodyWeight", "Comment", "WorkoutComment"):
        pk = [r["name"] for r in _q(db, f"PRAGMA table_info([{t}])") if r["pk"]]
        assert pk == ["_id"], (t, pk)


# ══════════════════════════════════════════════════════════════════════════════
# CHANGED — a write now commits only if the database moved exactly as intended
#
# WAS: a DELETE matching zero rows raises nothing and nothing inspected rowcount,
# so the tool reported "Set deleted successfully" having deleted nothing. The
# claim gate derived db_write_effect from that flag, so the guard against false
# claims rested on it.
# ══════════════════════════════════════════════════════════════════════════════

def test_no_op_delete_is_rejected_and_rolled_back(db):
    row = _newest_set(db)
    _stage_delete(db, row)

    con = sqlite3.connect(db)                       # the row vanishes underneath
    con.execute("DELETE FROM training_log WHERE _id=?", (row["_id"],))
    con.commit(); con.close()
    after_sabotage = _q(db, "SELECT * FROM training_log")

    out = json.loads(cs._execute_staged_set_delete_sync())
    assert out["success"] is False
    assert out["integrity_rejected"] is True
    assert "wasn't saved" in out["message"]
    # rolled back: the database is exactly as the sabotage left it
    assert _q(db, "SELECT * FROM training_log") == after_sabotage
    # the slot survives, so a retry is possible
    assert "delete_set" in cs._staged_writes


def test_no_op_update_is_rejected_and_rolled_back(db):
    row = _newest_set(db)
    staged = json.loads(cs._update_workout_set_sync({
        "exercise_name": _name_of(db, row["exercise_id"]),
        "date": row["date"],
        "old_weight": round(row["metric_weight"] * 2.2046, 1),
        "old_reps": row["reps"],
        "new_weight": round(row["metric_weight"] * 2.2046, 1) + 5,
        "new_reps": row["reps"],
        "unit": "lbs",
    }))
    assert staged.get("staged") is True, staged

    con = sqlite3.connect(db)
    con.execute("DELETE FROM training_log WHERE _id=?", (row["_id"],))
    con.commit(); con.close()

    out = json.loads(cs._execute_staged_set_update_sync())
    assert out["success"] is False and out["integrity_rejected"] is True


def test_a_correct_delete_is_still_accepted(db):
    """The other direction. A guard that rejects everything is not a guard."""
    row = _newest_set(db)
    _stage_delete(db, row)
    out = json.loads(cs._execute_staged_set_delete_sync())
    assert out["success"] is True and out["verified"] is True
    assert not _q(db, "SELECT * FROM training_log WHERE _id=?", (row["_id"],))


def test_a_correct_workout_write_is_still_accepted(db):
    before = len(_q(db, "SELECT _id FROM training_log"))
    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Flat Barbell Bench Press",
        "date": "2026-08-08",
        "sets": [{"weight": 101, "reps": 7, "unit": "lbs"}],
    }))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_workout_sync())
    assert out["success"] is True and out["verified"] is True
    assert len(_q(db, "SELECT _id FROM training_log")) == before + 1


def test_an_insert_is_not_rejected_by_sqlite_sequence_churn(db):
    """training_log is AUTOINCREMENT, so sqlite_sequence moves on every insert.
    If the whole-file check did not exclude it, every log would be rejected."""
    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Flat Barbell Bench Press",
        "date": "2026-08-08",
        "sets": [{"weight": 101, "reps": 7, "unit": "lbs"}],
    }))
    assert staged.get("staged") is True
    assert json.loads(cs._execute_staged_workout_sync())["success"] is True


def test_bodyweight_is_guarded_too(db):
    """CHANGED: it is no longer the odd one out. It used to write straight
    through with no stage/execute pair — which is what left it with no update,
    no delete, a duplicate row whenever body fat followed a weight, and a JSON
    confirm panel. Now it stages like every other write; the guard still applies
    to the execute, which is the part this test is about."""
    before = len(_q(db, "SELECT _id FROM BodyWeight"))
    staged = json.loads(cs._log_bodyweight_sync({"body_weight": 180.0, "unit": "lbs"}))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_bodyweight_sync())
    assert out["success"] is True and out["verified"] is True
    assert len(_q(db, "SELECT _id FROM BodyWeight")) == before + 1


def test_the_bodyweight_update_path_is_guarded(db):
    """The upsert branch writes different SQL, so it needs its own check —
    an UPDATE that matched no rows used to be how a write reported success
    having done nothing."""
    cs._log_bodyweight_sync({"body_weight": 180.0, "unit": "lbs"})
    cs._execute_staged_bodyweight_sync()
    before = len(_q(db, "SELECT _id FROM BodyWeight"))
    cs._log_bodyweight_sync({"body_weight": 182.0, "unit": "lbs"})
    out = json.loads(cs._execute_staged_bodyweight_sync())
    assert out["success"] is True and out["verified"] is True
    assert len(_q(db, "SELECT _id FROM BodyWeight")) == before   # updated, not added


def test_no_write_path_is_left_unguarded():
    """Counted, not spot-checked. Every function that opens a write connection
    must go through the guard — one unguarded path reproduces the whole class of
    bug this arc exists to close."""
    src = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    # The guard itself is the only place allowed to open a write connection.
    opens = [l for l in src.splitlines()
             if "get_write_connection(DB_PATH)" in l and not l.strip().startswith("#")]
    assert len(opens) == 1, f"unguarded write path(s): {opens}"


# Where each _execute_*_sync body ends, for the source-level assertions below.
_ASYNC_BOUNDARY = "\n\nasync def"


@pytest.mark.parametrize("fn", [
    "_execute_staged_goal_sync",
    "_execute_staged_goal_update_sync",
    "_execute_staged_goal_delete_sync",
    "_execute_staged_set_update_sync",
    "_execute_staged_set_delete_sync",
    "_execute_staged_workout_sync",
])
def test_every_execute_now_reports_verified(fn):
    src = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    body = src[src.index(f"def {fn}"):]
    body = body[:body.index(_ASYNC_BOUNDARY)]
    assert '"verified": True' in body, fn
    assert "_guarded_write" in body, fn


# ── CHANGED by ruling 1 ───────────────────────────────────────────────────────
# is_personal_record is FitNotes' column and means "was a PR when performed".
# The update used to recompute it as "is this the best right now" and write that
# in, so editing an old set stamped it from today's data. Delete already left it
# alone; update now does too. This REMOVED a query and a column.

def test_update_no_longer_rewrites_the_pr_flag():
    src = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    body = src[src.index("def _execute_staged_set_update_sync"):]
    body = body[:body.index(_ASYNC_BOUNDARY)]
    # Comments are stripped first — the function explains at length WHY it no
    # longer writes the column, and that prose must not satisfy the assertion.
    code = "\n".join(l.split("#")[0] for l in body.splitlines())
    assert "MAX(metric_weight)" not in code
    assert "is_personal_record" not in code


def test_update_preserves_the_existing_pr_flag(db):
    """Behavioural, not just textual: whatever the column held before an edit is
    what it holds after."""
    row = _q(db, "SELECT * FROM training_log WHERE is_personal_record=1 "
                 "ORDER BY _id DESC LIMIT 1")[0]
    staged = json.loads(cs._update_workout_set_sync({
        "exercise_name": _name_of(db, row["exercise_id"]),
        "date": row["date"],
        "old_weight": round(row["metric_weight"] * 2.2046, 1),
        "old_reps": row["reps"],
        "new_weight": round(row["metric_weight"] * 2.2046, 1),
        "new_reps": row["reps"] + 1,
        "unit": "lbs",
    }))
    assert staged.get("staged") is True, staged
    assert json.loads(cs._execute_staged_set_update_sync())["success"] is True
    after = _q(db, "SELECT * FROM training_log WHERE _id=?", (row["_id"],))[0]
    assert after["reps"] == row["reps"] + 1                  # the edit landed
    assert after["is_personal_record"] == row["is_personal_record"]   # flag untouched
