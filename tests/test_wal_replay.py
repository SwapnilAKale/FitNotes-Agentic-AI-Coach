"""
Do the agent's writes survive a backup upload, and only the agent's writes?

An upload replaces the whole .fitnotes file, so every confirmed write is
journalled and replayed onto the new database. That replay is the most dangerous
code in the system: it runs write SQL against a database it has never seen,
using ids that meant something else where they were recorded.

WHAT THIS FILE EXISTS TO STOP, all of it found live:
  - a journalled id reaching a row the APP owns (the collision);
  - an insert applying while its matching delete conflicts, leaving a phantom
    row in real training data (row 15771, which survived eleven days);
  - a write path that can be journalled but never replayed, so it is dropped
    silently on every upload (comments, for the life of the feature);
  - a workout filed under the wrong exercise because the id was resolved
    instead of the name.

Split out of test_app_data_lock.py, which had grown to cover both the live
write RULE and the replay that re-applies it.

EVERY TEST RUNS ON COPIES in tmp_path — never the real database.
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
import src.settings as settings                                # noqa: E402
import src.wal as wal                                          # noqa: E402

_REAL_DB = _ROOT / "data" / "FitNotes_Backup.fitnotes"

WATERMARK_KEY = "app_data_watermark"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A copy of the real database plus isolated settings and WAL files."""
    if not _REAL_DB.exists():
        pytest.skip("real database not present")
    db = str(tmp_path / "copy.fitnotes")
    shutil.copy2(_REAL_DB, db)
    monkeypatch.setattr(cs, "DB_PATH", db)
    monkeypatch.setenv("SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(wal, "WAL_PATH", str(tmp_path / "wal.json"))
    cs._staged_writes.clear()
    return db


def _q(db, sql, args=()):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args)]
    con.close(); return out


def _newest(db):
    return _q(db, "SELECT * FROM training_log ORDER BY _id DESC LIMIT 1")[0]


def _name_of(db, exercise_id):
    return _q(db, "SELECT name FROM exercise WHERE _id=?", (exercise_id,))[0]["name"]


def _set_watermark(db):
    """Mark everything currently in the database as app-owned, the way /upload
    will once the new database has been written and before replay runs."""
    marks = {t: (_q(db, f"SELECT COALESCE(MAX(_id), 0) m FROM [{t}]")[0]["m"])
             for t in ("training_log", "Goal")}
    settings.set_setting(WATERMARK_KEY, marks)
    return marks


def _log_a_set(db, date="2026-08-08", weight=101, reps=7):
    """Agent-created row: staged, executed, id lands above the watermark."""
    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Flat Barbell Bench Press", "date": date,
        "sets": [{"weight": weight, "reps": reps, "unit": "lbs"}]}))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_workout_sync())
    assert out["success"] is True, out
    return _newest(db)


# ══════════════════════════════════════════════════════════════════════════════
# The collision — the reason this arc exists
# ══════════════════════════════════════════════════════════════════════════════

def _two_sequence_databases(tmp_path):
    """Two databases whose id sequences advanced independently.

    local : the agent's copy — app rows 1-5, then the agent's own row at id 6
    fresh : a later export — the APP's own sixth workout, also at id 6

    Same id, different rows. That is the whole bug.
    """
    schema = """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY AUTOINCREMENT, exercise_id INTEGER, date DATE,
            metric_weight REAL, reps INTEGER, unit INTEGER NOT NULL DEFAULT 0,
            is_personal_record INTEGER, is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0, duration_seconds INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE Comment (_id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE,
            owner_type_id INTEGER, owner_id INTEGER, comment TEXT);
        CREATE TABLE Goal (_id INTEGER PRIMARY KEY AUTOINCREMENT, type_id INTEGER,
            exercise_id INTEGER, metric_weight REAL, reps INTEGER, unit INTEGER,
            title TEXT, target_date TEXT, sort_order INTEGER, distance REAL,
            duration_seconds INTEGER, start_date TEXT);
        CREATE TABLE BodyWeight (_id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE,
            body_weight_metric REAL, body_fat REAL NOT NULL DEFAULT 0,
            comments TEXT);
        INSERT INTO exercise (_id, name, category_id) VALUES (1, 'Bench', 5);
        -- A SECOND exercise, so "resolve by name, not by the journalled id" has
        -- a wrong id available to be caught filing a workout under. With one
        -- exercise the stale-id defect is untestable by construction.
        INSERT INTO exercise (_id, name, category_id) VALUES (2, 'Squat', 5);
    """
    paths = {}
    for which, sixth in (("local", "AGENT"), ("fresh", "APP")):
        p = str(tmp_path / f"{which}.fitnotes")
        con = sqlite3.connect(p)
        con.executescript(schema)
        for i in range(1, 6):                       # the five shared app rows
            con.execute("INSERT INTO training_log (exercise_id, date, metric_weight,"
                        " reps, unit, is_personal_record, is_complete, distance,"
                        " duration_seconds) VALUES (1, ?, ?, 5, 0, 0, 1, 0, 0)",
                        (f"2026-07-0{i}", 40.0 + i))
        # the sixth row differs between the two databases, but shares id 6
        con.execute("INSERT INTO training_log (exercise_id, date, metric_weight,"
                    " reps, unit, is_personal_record, is_complete, distance,"
                    " duration_seconds) VALUES (1, ?, ?, 9, 0, 0, 1, 0, 0)",
                    ("2026-08-01" if sixth == "AGENT" else "2026-08-15", 99.0))
        con.commit(); con.close()
        paths[which] = p
    return paths


def test_replay_must_not_delete_the_apps_row_that_shares_an_id(env, tmp_path):
    """THE COLLISION. The agent deleted ITS row (id 6 in the local copy). A fresh
    export arrives where id 6 is the APP's own workout. Replaying that delete must
    not touch it."""
    dbs = _two_sequence_databases(tmp_path)
    app_row = _q(dbs["fresh"], "SELECT * FROM training_log WHERE _id=6")[0]
    assert app_row["date"] == "2026-08-15"          # the app's workout, not the agent's

    wal.append_write("execute_staged_set_delete", {
        "set_id": 6,                                 # the id in the LOCAL copy
        "exercise_name": "Bench", "date": "2026-08-01",
        "weight": 218.3, "reps": 9, "unit": "lbs",
        "stored_weight": 99.0, "exercise_id": 1,
    })

    wal.replay_writes(dbs["fresh"])

    survivor = _q(dbs["fresh"], "SELECT * FROM training_log WHERE _id=6")
    assert survivor, "replay deleted the app's workout"
    assert survivor[0]["date"] == "2026-08-15", "replay overwrote the app's workout"


def test_replay_remaps_its_own_insert_then_delete(env, tmp_path):
    """The legitimate case must still work: the agent logged a set and later
    deleted it, so replay inserts then deletes ITS OWN row — under a new id."""
    dbs = _two_sequence_databases(tmp_path)
    # Push the fresh database's sequence on, so the replayed insert CANNOT land
    # on the stale id the delete names. Without this the test passes by
    # coincidence — id 7 would be both the replayed row and the delete target,
    # and a broken remap would look correct.
    con = sqlite3.connect(dbs["fresh"])
    con.execute("INSERT INTO training_log (exercise_id, date, metric_weight, reps,"
                " unit, is_personal_record, is_complete, distance, duration_seconds)"
                " VALUES (1, '2026-08-16', 50.0, 5, 0, 0, 1, 0, 0)")
    con.commit(); con.close()
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))

    # row_ids is what the real journal now records: the ids these rows held in
    # the database they were written to. It is the translation replay needs.
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "Bench", "date": "2026-08-20",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}],
        "row_ids": [7]})
    wal.append_write("execute_staged_set_delete", {
        "set_id": 7,                                  # id it held in the local copy
        "exercise_name": "Bench", "date": "2026-08-20",
        "weight": 132.3, "reps": 4, "unit": "lbs",
        "stored_weight": 60.0, "exercise_id": 1})

    wal.replay_writes(dbs["fresh"])

    rows = _q(dbs["fresh"], "SELECT * FROM training_log")
    assert len(rows) == before, "insert-then-delete should net to nothing"
    assert not [r for r in rows if r["date"] == "2026-08-20"]


# ══════════════════════════════════════════════════════════════════════════════
# Row 4 — replay and the live tool must agree about the PR flag
# ══════════════════════════════════════════════════════════════════════════════

def test_replay_does_not_rewrite_the_pr_flag():
    """The live tool stopped recomputing is_personal_record; replay must too, or
    the same edit means different things depending on which path applied it."""
    src = (_ROOT / "src" / "wal.py").read_text(encoding="utf-8")
    body = src[src.index("def _replay_set_update"):]
    body = body[:body.index("\n\ndef ")]
    code = "\n".join(l.split("#")[0] for l in body.splitlines())
    assert "MAX(metric_weight)" not in code
    assert "is_personal_record" not in code


# ══════════════════════════════════════════════════════════════════════════════
# Row 6 — a SKIPPED duplicate must still map its ids
#
# How a phantom row reached the user's real training data and stayed for eleven
# days. TWO journal records described one workout: the first carried no row_ids
# (journalled before they existed) and inserted the row; the second carried
# row_ids and was skipped as a duplicate — so its old id entered no map, and the
# update and the delete that named it both conflicted. The insert applied and
# nothing could undo it: a 101 lb bench set, flagged as a personal record, in
# the middle of a shoulder day.
# ══════════════════════════════════════════════════════════════════════════════

def test_a_duplicate_insert_still_maps_its_ids_so_the_delete_lands(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))

    # Record A — the shape journalled before row_ids existed. It inserts.
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "Bench", "date": "2026-08-20",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}]})
    # Record B — the SAME workout, journalled again, this time carrying the id
    # the row held locally. Replay skips it as a duplicate.
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "Bench", "date": "2026-08-20",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}],
        "row_ids": [7]})
    # The delete names the id only record B could translate.
    wal.append_write("execute_staged_set_delete", {
        "set_id": 7, "exercise_name": "Bench", "date": "2026-08-20",
        "weight": 132.3, "reps": 4, "unit": "lbs",
        "stored_weight": 60.0, "exercise_id": 1})

    wal.replay_writes(dbs["fresh"])

    rows = _q(dbs["fresh"], "SELECT * FROM training_log WHERE date='2026-08-20'")
    assert rows == [], (
        "the duplicate's row_ids were not mapped, so the delete could not find "
        "its target and the insert was left behind")
    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before


def test_a_genuinely_new_insert_is_unaffected(env, tmp_path):
    """Other direction — mapping on the skip path must not stop real inserts."""
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "Bench", "date": "2026-08-21",
        "sets": [{"metric_weight": 61.0, "reps": 3, "unit": 0}],
        "row_ids": [9]})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before + 1


def test_a_delete_for_a_row_this_replay_never_saw_still_conflicts(env, tmp_path):
    """The guard the skip-mapping must NOT weaken: an id this run neither
    inserted nor skipped stays unreachable, or the app's own row gets deleted."""
    dbs = _two_sequence_databases(tmp_path)
    survivor = _q(dbs["fresh"], "SELECT * FROM training_log WHERE _id=6")
    assert survivor, "fixture precondition: the app owns row 6"
    wal.append_write("execute_staged_set_delete", {
        "set_id": 6, "exercise_name": "Bench", "date": "2026-08-15",
        "weight": 132.3, "reps": 4, "unit": "lbs",
        "stored_weight": 60.0, "exercise_id": 1})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM training_log WHERE _id=6"), \
        "replay deleted a row it never created"


# ══════════════════════════════════════════════════════════════════════════════
# Row 8 — replay was missing HALF the write paths
#
# Phase 8 proved the replay toggle works. Auditing the rest of that path before
# extending it found four more defects, all the same omission: the app-data-lock
# arc hardened the training_log SET paths and left every sibling on the old
# pattern.
#   - goals: create replayed, update/delete named the OLD id → orphan (live)
#   - comments: no handler at all → silently dropped (live, 4 records)
#   - bodyweight: never journalled → gone without even a conflict record
#   - exercises: resolved by STALE id with no working fallback → misfiled
# ══════════════════════════════════════════════════════════════════════════════

def _goal_wal(goal_id, lbs=150.0, target="2026-12-31"):
    return {"exercise_id": 1, "exercise_name": "Bench", "goal_id": goal_id,
            "metric_weight": lbs / 2.2046, "reps": 1, "title": "t",
            "target_date": target, "start_date": "2026-09-01"}


# -- goals: the mapping sets already had --------------------------------------

def test_a_goal_created_and_deleted_replays_to_nothing(env, tmp_path):
    """THE live failure. The create replayed as a new id, the delete named the
    old one, missed, and a goal the user had deleted came back to life."""
    dbs = _two_sequence_databases(tmp_path)
    wal.append_write("execute_staged_goal", _goal_wal(7))
    wal.append_write("execute_staged_goal_delete",
                     {"goal_id": 7, "exercise_name": "Bench",
                      "target_date": "2026-12-31"})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM Goal") == [], \
        "the goal came back to life - delete could not find the remapped row"


def test_a_goal_update_lands_on_the_remapped_row(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    wal.append_write("execute_staged_goal", _goal_wal(7))
    wal.append_write("execute_staged_goal_update",
                     {"goal_id": 7, "new_metric_weight": 155 / 2.2046,
                      "new_reps": 1, "new_target_date": "2026-12-31"})
    wal.replay_writes(dbs["fresh"])
    rows = _q(dbs["fresh"], "SELECT * FROM Goal")
    assert len(rows) == 1
    assert round(rows[0]["metric_weight"] * 2.2046) == 155


def test_replay_must_not_delete_a_goal_it_never_created(env, tmp_path):
    """THE DESTRUCTIVE DIRECTION, asserted rather than argued. Without mapping,
    DELETE FROM Goal WHERE _id = 3 hits whatever IS _id 3 - the user's own."""
    dbs = _two_sequence_databases(tmp_path)
    con = sqlite3.connect(dbs["fresh"])
    con.execute("INSERT INTO Goal (type_id, exercise_id, metric_weight, reps,"
                " unit, title, target_date, sort_order, distance,"
                " duration_seconds, start_date)"
                " VALUES (1, 1, 90.0, 5, 0, 'MY OWN GOAL', '2027-01-01', 0, 0, 0, '2026-01-01')")
    con.commit()
    planted = con.execute("SELECT _id FROM Goal").fetchone()[0]
    con.close()

    wal.append_write("execute_staged_goal_delete",
                     {"goal_id": planted, "exercise_name": "Bench",
                      "target_date": "2026-12-31"})
    wal.replay_writes(dbs["fresh"])

    survivor = _q(dbs["fresh"], "SELECT title FROM Goal WHERE _id=?", (planted,))
    assert survivor and survivor[0]["title"] == "MY OWN GOAL", \
        "replay destroyed a goal the user owned"


# -- comments: a handler at last, resolved by CONTENT --------------------------

def test_a_comment_on_an_app_owned_set_still_lands(env, tmp_path):
    """THE CARVE-OUT. A note may sit on a set the app owns - one replay never
    created - so this must NOT go through _mapped_id."""
    dbs = _two_sequence_databases(tmp_path)
    row = _q(dbs["fresh"], "SELECT * FROM training_log ORDER BY _id LIMIT 1")[0]
    wal.append_write("execute_staged_set_comment", {
        "set_id": 999,                       # a stale id that maps to nothing
        "exercise_id": row["exercise_id"], "exercise_name": "Bench",
        "date": row["date"], "reps": row["reps"],
        "stored_weight": row["metric_weight"], "comment": "elbows flared"})
    wal.replay_writes(dbs["fresh"])
    notes = _q(dbs["fresh"], "SELECT comment FROM Comment WHERE owner_id=?",
               (row["_id"],))
    assert notes and notes[0]["comment"] == "elbows flared"


def test_a_comment_refuses_to_guess_between_identical_sets(env, tmp_path):
    """Two identical sets on a day - a note must not be dropped onto whichever
    comes first."""
    dbs = _two_sequence_databases(tmp_path)
    row = _q(dbs["fresh"], "SELECT * FROM training_log ORDER BY _id LIMIT 1")[0]
    con = sqlite3.connect(dbs["fresh"])
    con.execute("INSERT INTO training_log (exercise_id, date, metric_weight, reps,"
                " unit, is_personal_record, is_complete, distance, duration_seconds)"
                " VALUES (?, ?, ?, ?, 0, 0, 1, 0, 0)",
                (row["exercise_id"], row["date"], row["metric_weight"], row["reps"]))
    con.commit()
    con.close()

    wal.append_write("execute_staged_set_comment", {
        "set_id": 999, "exercise_id": row["exercise_id"], "exercise_name": "Bench",
        "date": row["date"], "reps": row["reps"],
        "stored_weight": row["metric_weight"], "comment": "which one"})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM Comment WHERE comment=?",
              ("which one",)) == []


def test_an_empty_comment_clears_the_note(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    row = _q(dbs["fresh"], "SELECT * FROM training_log ORDER BY _id LIMIT 1")[0]
    con = sqlite3.connect(dbs["fresh"])
    con.execute("INSERT INTO Comment (date, owner_type_id, owner_id, comment) "
                "VALUES (?, 1, ?, 'old note')", (row["date"], row["_id"]))
    con.commit()
    con.close()

    wal.append_write("execute_staged_set_comment", {
        "set_id": 999, "exercise_id": row["exercise_id"], "exercise_name": "Bench",
        "date": row["date"], "reps": row["reps"],
        "stored_weight": row["metric_weight"], "comment": ""})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM Comment WHERE owner_id=?",
              (row["_id"],)) == []


# -- bodyweight: journalled, and replayed once ---------------------------------

def test_bodyweight_is_journalled_on_write(env):
    # CHANGED: log_bodyweight STAGES now rather than writing straight through,
    # so the journal entry comes from the execute. Being the only logging
    # operation without a staged/execute pair is what cost it an update, a
    # delete, a duplicate row per body-fat entry, and a JSON confirm panel.
    _set_watermark(env)
    staged = json.loads(cs._log_bodyweight_sync({"body_weight": 180, "unit": "lbs"}))
    assert staged.get("staged") is True, staged
    out = json.loads(cs._execute_staged_bodyweight_sync())
    assert out["success"] is True, out
    recs = json.load(open(wal.WAL_PATH))
    assert any(r.get("tool") == "execute_staged_bodyweight" for r in recs), \
        "body weight was written but never journalled - it dies on the next upload"


def test_bodyweight_replays_once_and_does_not_double(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    wal.append_write("log_bodyweight", {
        "date": "2026-09-11", "body_weight_metric": 81.647, "body_fat": 21.95})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM BodyWeight")) == 1
    # a second identical record must not add a duplicate weigh-in
    wal.append_write("log_bodyweight", {
        "date": "2026-09-11", "body_weight_metric": 81.647, "body_fat": 21.95})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM BodyWeight")) == 1


# -- exercise resolution: name beats id ----------------------------------------

def test_a_recorded_name_beats_a_stale_exercise_id(env, tmp_path):
    """The defect that MISFILES instead of losing. The journalled id belongs to
    the old database; after an export renumbers, it can name another lift."""
    dbs = _two_sequence_databases(tmp_path)
    con = sqlite3.connect(dbs["fresh"])
    con.row_factory = sqlite3.Row
    names = {r["_id"]: r["name"] for r in con.execute("SELECT _id, name FROM exercise")}
    con.close()
    right_id = [i for i, n in names.items() if n == "Bench"][0]
    wrong_id = [i for i in names if i != right_id][0]

    wal.append_write("execute_staged_workout", {
        "exercise_id": wrong_id, "exercise_name": "Bench", "date": "2026-08-22",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}], "row_ids": [77]})
    wal.replay_writes(dbs["fresh"])
    rows = _q(dbs["fresh"],
              "SELECT exercise_id FROM training_log WHERE date='2026-08-22'")
    assert rows and rows[0]["exercise_id"] == right_id, \
        "replay filed the workout under the stale id instead of the named exercise"


def test_an_unknown_name_conflicts_rather_than_falling_back_to_the_id(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "No Such Exercise", "date": "2026-08-23",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}], "row_ids": [78]})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before


def test_records_without_a_name_still_replay_by_id(env, tmp_path):
    """Every record journalled before this change carries no name. They must
    keep working - the id path is their only route."""
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "date": "2026-08-24",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}], "row_ids": [79]})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before + 1


# -- the coverage guard that would have caught comments ------------------------

def test_every_journalled_tool_has_a_replay_route():
    """Comments were journalled for the whole life of the feature and had
    nowhere to go - every record died with 'no replay handler'. A write path
    that can be journalled but not replayed loses data silently."""
    import re
    src = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    journalled = set(re.findall(r'_wal_append\(\s*"([a-z_]+)"', src))
    assert journalled, "no _wal_append call sites found - the scan is broken"
    missing = journalled - set(wal._TOOL_ALIASES)
    assert not missing, f"journalled but not replayable: {sorted(missing)}"
    unknown = set(wal._TOOL_ALIASES.values()) - set(wal._HANDLERS)
    assert not unknown, f"alias points at no handler: {sorted(unknown)}"


# -- Row 5: replayed writes get the blast-radius check -------------------------
#
# src/wal.py referenced the integrity guard ZERO times. Live writes are checked
# for blast radius - the layer that caught a delete matching no rows in Phase 7 -
# and replayed writes were not, despite running the same SQL against a database
# that has changed underneath them.

def test_a_replayed_write_that_changes_too_much_is_rolled_back(env, tmp_path, monkeypatch):
    """A handler whose SQL touches more rows than its record declares must not
    commit. Simulated by a handler that deletes the whole table."""
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))
    assert before > 1, "fixture precondition: more than one row to over-delete"

    def _greedy(conn, params, id_map=None):
        conn.execute("DELETE FROM training_log")      # declared 1, removes all
        return "pretended to delete one row"

    monkeypatch.setitem(wal._HANDLERS, "set_delete", _greedy)
    wal.append_write("execute_staged_set_delete", {
        "set_id": 7, "exercise_name": "Bench", "date": "2026-08-20",
        "weight": 132.3, "reps": 4, "unit": "lbs", "exercise_id": 1})
    wal.replay_writes(dbs["fresh"])

    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before, \
        "an over-wide replayed delete committed - the table was emptied"
    rec = [r for r in json.load(open(wal.WAL_PATH))
           if r.get("tool") == "execute_staged_set_delete"][-1]
    assert rec["status"] == "conflict"
    assert "blast radius" in rec.get("error", "")


def test_an_honest_replayed_write_still_commits(env, tmp_path):
    """The other direction - the guard must not reject correct replays. Covered
    in substance by every test above, asserted here directly."""
    dbs = _two_sequence_databases(tmp_path)
    before = len(_q(dbs["fresh"], "SELECT _id FROM training_log"))
    wal.append_write("execute_staged_workout", {
        "exercise_id": 1, "exercise_name": "Bench", "date": "2026-08-25",
        "sets": [{"metric_weight": 60.0, "reps": 4, "unit": 0}], "row_ids": [88]})
    wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM training_log")) == before + 1
    rec = [r for r in json.load(open(wal.WAL_PATH))
           if r.get("tool") == "execute_staged_workout"][-1]
    assert rec["status"] == "replayed"


# -- replay: the same mapping goals and sets have ------------------------------

def test_a_bodyweight_entry_replays_onto_the_new_database(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    wal.append_write("execute_staged_bodyweight", {
        "row_id": 3, "date": "2026-09-11",
        "body_weight_metric": 81.647, "body_fat": 21.95})
    wal.replay_writes(dbs["fresh"])
    rows = _q(dbs["fresh"], "SELECT * FROM BodyWeight")
    assert len(rows) == 1 and rows[0]["body_fat"] == 21.95


def test_a_replayed_log_then_delete_nets_to_nothing(env, tmp_path):
    """The pair that orphaned twice for goals, pre-empted here."""
    dbs = _two_sequence_databases(tmp_path)
    wal.append_write("execute_staged_bodyweight", {
        "row_id": 3, "date": "2026-09-11",
        "body_weight_metric": 81.647, "body_fat": 0})
    wal.append_write("execute_staged_bodyweight_delete", {
        "row_id": 3, "date": "2026-09-11", "body_weight_metric": 81.647})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM BodyWeight") == []


def test_replay_must_not_delete_a_weigh_in_it_never_created(env, tmp_path):
    """THE row-15771 shape, in a third table. Found in training_log, found again
    in Goal - written with the mapping from the start here."""
    dbs = _two_sequence_databases(tmp_path)
    con = sqlite3.connect(dbs["fresh"])
    con.execute("INSERT INTO BodyWeight (date, body_weight_metric, body_fat) "
                "VALUES ('2026-01-01', 90.0, 15.0)")
    con.commit()
    planted = con.execute("SELECT _id FROM BodyWeight").fetchone()[0]
    con.close()

    wal.append_write("execute_staged_bodyweight_delete", {
        "row_id": planted, "date": "2026-01-01", "body_weight_metric": 90.0})
    wal.replay_writes(dbs["fresh"])
    assert _q(dbs["fresh"], "SELECT _id FROM BodyWeight WHERE _id=?", (planted,)), \
        "replay deleted a weigh-in the user owned"


def test_replaying_the_same_weigh_in_twice_does_not_duplicate_it(env, tmp_path):
    dbs = _two_sequence_databases(tmp_path)
    for _ in range(2):
        wal.append_write("execute_staged_bodyweight", {
            "row_id": 3, "date": "2026-09-11",
            "body_weight_metric": 81.647, "body_fat": 21.95})
        wal.replay_writes(dbs["fresh"])
    assert len(_q(dbs["fresh"], "SELECT _id FROM BodyWeight")) == 1
