#!/usr/bin/env python3
"""
scripts/test_wal_goals.py — prove the WAL goal-insert duplicate guard.

Scenario under test: a goal INSERT replayed onto a backup that ALREADY
contains that goal. Correct behaviour is (a) detect the duplicate and record
a conflict — never (b) silently insert a second row.

Cases:
  1. Backup already contains the goal -> replay must CONFLICT, count stays 1.
  2. Fresh goal not in the backup     -> replay must INSERT, count becomes 2.
  3. Identical record appended again  -> replay must CONFLICT, count stays 2.

Run from the repo root: python scripts/test_wal_goals.py
"""

import os
import shutil
import sqlite3
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src import wal

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        _failures += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def goal_count(db_path: str, exercise_id: int) -> int:
    conn = sqlite3.connect(f"file:{db_path.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM Goal WHERE exercise_id = ?", (exercise_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"FAIL  real DB not found at {DB_PATH}")
        return 1

    fd, tmp_db = tempfile.mkstemp(suffix=".fitnotes")
    os.close(fd)
    shutil.copyfile(DB_PATH, tmp_db)
    print(f"Temp DB: {tmp_db}")

    test_ids: list = []
    try:
        # Resolve a real exercise id for the goal rows
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        exercise_id = conn.execute(
            "SELECT _id FROM exercise WHERE name = 'Lat Pulldown'").fetchone()["_id"]

        # Simulate "the backup already contains this goal": insert the goal
        # directly into the temp DB with the exact SQL the live tool uses.
        goal_a = {
            "exercise_id":   exercise_id,
            "metric_weight": 200.0 / 2.2046,   # 200 lbs target
            "reps":          5,
            "title":         "WAL test goal A",
            "target_date":   "2099-12-31",
            "start_date":    "2099-01-01",
        }
        conn.execute(
            """INSERT INTO Goal
               (type_id, exercise_id, metric_weight, reps, unit, title, target_date,
                sort_order, distance, duration_seconds, start_date)
               VALUES (1, ?, ?, ?, 0, ?, ?, 0, 0, 0, ?)""",
            (goal_a["exercise_id"], goal_a["metric_weight"], goal_a["reps"],
             goal_a["title"], goal_a["target_date"], goal_a["start_date"]),
        )
        conn.commit()
        conn.close()
        base_count = goal_count(tmp_db, exercise_id)
        check("setup: goal pre-exists in 'backup'", base_count == 1,
              f"count={base_count}")

        # Case 1 — replay the same goal onto the backup that already has it
        test_ids.append(wal.append_write("set_goal", dict(goal_a)))
        r1 = wal.replay_writes(tmp_db)
        check("case 1: duplicate goal -> conflicts == 1", r1["conflicts"] == 1,
              f"got {r1['conflicts']}")
        check("case 1: duplicate goal -> replayed == 0", r1["replayed"] == 0,
              f"got {r1['replayed']}")
        check("case 1: NO second row inserted",
              goal_count(tmp_db, exercise_id) == 1,
              f"count={goal_count(tmp_db, exercise_id)}")
        rec = {r["id"]: r for r in wal.get_records()}[test_ids[0]]
        check("case 1: record marked status=conflict with error",
              rec["status"] == "conflict" and "already exists" in rec.get("error", ""),
              f"status={rec['status']!r}, error={rec.get('error', '')!r}")

        # Case 2 — a goal the backup does NOT contain replays cleanly
        goal_b = dict(goal_a, target_date="2098-06-30", title="WAL test goal B")
        test_ids.append(wal.append_write("set_goal", goal_b))
        r2 = wal.replay_writes(tmp_db)
        check("case 2: fresh goal -> replayed == 1", r2["replayed"] == 1,
              f"got {r2['replayed']}, errors={r2['errors']}")
        check("case 2: row inserted", goal_count(tmp_db, exercise_id) == 2,
              f"count={goal_count(tmp_db, exercise_id)}")

        # Case 3 — appending the identical record again conflicts
        test_ids.append(wal.append_write("set_goal", dict(goal_b)))
        r3 = wal.replay_writes(tmp_db)
        check("case 3: re-appended identical goal -> conflicts == 1",
              r3["conflicts"] == 1, f"got {r3['conflicts']}")
        check("case 3: count unchanged", goal_count(tmp_db, exercise_id) == 2,
              f"count={goal_count(tmp_db, exercise_id)}")

        # Sanity: real DB untouched
        real = goal_count(DB_PATH, exercise_id)
        check("real DB untouched (no test goals)", real == 0, f"count={real}")

    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass
        with wal._lock:
            wal._save([r for r in wal._load() if r.get("id") not in test_ids])
        leftover = [r for r in wal.get_records() if r.get("id") in test_ids]
        check("test records removed from agent_writes.json", not leftover)

    print()
    if _failures:
        print(f"FAIL — {_failures} assertion(s) failed")
        return 1
    print("PASS — goal duplicate guard verified: duplicates conflict, never silently insert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
