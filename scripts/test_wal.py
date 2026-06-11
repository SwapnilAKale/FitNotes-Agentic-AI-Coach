#!/usr/bin/env python3
"""
scripts/test_wal.py — standalone WAL verification. No Gemini, no agent.

1. Copies the real DB to a temp file.
2. Appends two log_workout records to the WAL.
3. replay_writes(temp_db) -> expects replayed==2, conflicts==0,
   and the rows physically present in the temp DB.
4. replay_writes again -> expects replayed==0 (idempotent).
5. Removes the test records from agent_writes.json and the temp DB.

Run from the repo root: python scripts/test_wal.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile

# Run from the repo root so wal.py's relative WAL_PATH resolves
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


def count_test_rows(db_path: str) -> int:
    conn = sqlite3.connect(f"file:{db_path.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        return conn.execute(
            """SELECT COUNT(*) FROM training_log tl
               JOIN exercise e ON e._id = tl.exercise_id
               WHERE e.name = 'Lat Pulldown' AND tl.date IN ('2099-01-01', '2099-01-02')"""
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"FAIL  real DB not found at {DB_PATH}")
        return 1

    # 1. Copy the real DB to a temp file
    fd, tmp_db = tempfile.mkstemp(suffix=".fitnotes")
    os.close(fd)
    shutil.copyfile(DB_PATH, tmp_db)
    print(f"Temp DB: {tmp_db}")

    test_ids: list = []
    try:
        # 2. Append two log_workout records (distinct date/weight so neither
        #    is a duplicate of the other at replay time; dates in 2099 so
        #    they can never collide with real training data)
        test_ids.append(wal.append_write("log_workout", {
            "exercise_name": "Lat Pulldown",
            "date": "2099-01-01",
            "sets": [
                {"weight": 100.0, "unit": "lbs", "reps": 10},
                {"weight": 110.0, "unit": "lbs", "reps": 8},
            ],
        }))
        test_ids.append(wal.append_write("log_workout", {
            "exercise_name": "Lat Pulldown",
            "date": "2099-01-02",
            "sets": [
                {"weight": 120.0, "unit": "lbs", "reps": 6},
            ],
        }))
        check("append_write returned two ids", len(test_ids) == 2 and all(test_ids))

        records = {r["id"]: r for r in wal.get_records()}
        check("both records present with status=pending",
              all(records.get(i, {}).get("status") == "pending" for i in test_ids))

        # 3. First replay — both records apply
        result = wal.replay_writes(tmp_db)
        check("first replay: replayed == 2", result["replayed"] == 2,
              f"got {result['replayed']}")
        check("first replay: conflicts == 0", result["conflicts"] == 0,
              f"got {result['conflicts']}, errors={result['errors']}")

        rows = count_test_rows(tmp_db)
        check("3 test sets physically present in temp DB", rows == 3,
              f"found {rows}")

        records = {r["id"]: r for r in wal.get_records()}
        check("both records now status=replayed",
              all(records.get(i, {}).get("status") == "replayed" for i in test_ids))

        # 4. Second replay — idempotent, already-replayed records skipped
        result2 = wal.replay_writes(tmp_db)
        check("second replay: replayed == 0 (idempotent)", result2["replayed"] == 0,
              f"got {result2['replayed']}")
        check("second replay: conflicts == 0", result2["conflicts"] == 0,
              f"got {result2['conflicts']}")

        rows2 = count_test_rows(tmp_db)
        check("no duplicate rows after second replay", rows2 == 3,
              f"found {rows2}")

        # Sanity: the real DB was never touched
        real_rows = count_test_rows(DB_PATH)
        check("real DB untouched (no 2099 test rows)", real_rows == 0,
              f"found {real_rows}")

    finally:
        # 5. Cleanup — temp DB and the test records from agent_writes.json
        try:
            os.unlink(tmp_db)
        except OSError:
            pass
        with wal._lock:
            remaining = [r for r in wal._load() if r.get("id") not in test_ids]
            wal._save(remaining)
        leftover = [r for r in wal.get_records() if r.get("id") in test_ids]
        check("test records removed from agent_writes.json", not leftover)

    print()
    if _failures:
        print(f"FAIL — {_failures} assertion(s) failed")
        return 1
    print("PASS — all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
