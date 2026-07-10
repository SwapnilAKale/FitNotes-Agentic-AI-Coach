"""
DB test-isolation guard: no test may ever write rows into the real
data/FitNotes_Backup.fitnotes again (77 junk training_log rows accumulated
before this existed). conftest's autouse _isolated_db fixture copies the
real DB to tmp_path and points both FITNOTES_DB_PATH and cs.DB_PATH at the
copy, so reads keep their realism and stray writes land on the throwaway.
"""

import hashlib
import json
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mcp_servers.combined_server as cs       # noqa: E402

_REAL_DB = os.path.join(_ROOT, "data", "FitNotes_Backup.fitnotes")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_db_paths_point_into_tmp(tmp_path):
    assert os.environ["FITNOTES_DB_PATH"] == str(tmp_path / "FitNotes_Backup.fitnotes")
    assert cs.DB_PATH == str(tmp_path / "FitNotes_Backup.fitnotes")
    assert os.path.abspath(cs.DB_PATH) != os.path.abspath(_REAL_DB)


def test_copy_carries_real_data_for_realism_reads():
    conn = sqlite3.connect(f"file:{cs.DB_PATH.replace(os.sep, '/')}?mode=ro", uri=True)
    n = conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
    conn.close()
    assert n > 7000          # a real copy, not a synthetic/empty schema


def test_unpatched_write_lands_on_copy_not_real_db():
    # The exact pre-fix failure mode: a test drives stage->execute WITHOUT
    # any per-test DB patch. The write must land on the conftest copy and
    # the real file must stay byte-identical. (_staged_writes is module-global
    # and can hold batches leaked from earlier tests — clear it so exactly
    # one workout executes; the DB path stays unpatched, which is the point.)
    cs._staged_writes.clear()
    real_before = _sha(_REAL_DB)
    conn = sqlite3.connect(cs.DB_PATH)
    copy_before = conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
    conn.close()

    staged = json.loads(cs._log_workout_sync({
        "exercise_name": "Barbell Row", "date": "2026-03-31",
        "sets": [{"weight": 60.0, "unit": "lbs", "reps": 8}],
    }))
    assert "error" not in staged, staged
    done = json.loads(cs._execute_staged_workout_sync())
    assert done.get("success"), done

    conn = sqlite3.connect(cs.DB_PATH)
    copy_after = conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
    conn.close()
    assert copy_after == copy_before + 1     # write landed on the copy
    assert _sha(_REAL_DB) == real_before     # negative: real DB untouched
