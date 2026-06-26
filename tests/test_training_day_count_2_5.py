"""
Bug 2.5 — the all-time training-day COUNT must INCLUDE categories Time/Place/Neck
(ids 10/11/12), because you log time-of-day/location/neck ON real training days, so
those dates ARE training days. Every volume/strength/PR/pattern aggregate keeps the
EXCLUDED (308) scope; only `all_time_summary.total_training_days` adopts the inclusive
(317) scope. This is a count-only scope split.

  • Two real-DB checks use RECOMPUTE-AND-RELATE (independent raw SQL, no shared logic).
  • One SYNTHETIC TYPE-C check pins the split by construction (no DB): the inclusive
    count flows to total_training_days while sets/volume/streaks see only the strength
    dates.
"""

import os
import sqlite3
import sys
from datetime import date

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import collect  # noqa: E402
from src.data_agent.process import _compute_alltime_summary  # noqa: E402


def _ro_conn() -> sqlite3.Connection:
    """Independent read-only connection — ground truth shares no logic with the code."""
    db = os.environ["FITNOTES_DB_PATH"]
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# 1 — real DB: the count INCLUDES Time/Place/Neck (all-category distinct dates)
# ══════════════════════════════════════════════════════════════════════════════

def test_training_day_count_includes_time_place_neck():
    """total_training_days == COUNT(DISTINCT date) over ALL categories, and is strictly
    greater than the non-excluded count — proving the 10/11/12 days are now counted
    (not merely equal by coincidence)."""
    ats = collect(query_period_days=None)["all_time_summary"]

    conn = _ro_conn()
    try:
        all_days = conn.execute(
            "SELECT COUNT(DISTINCT tl.date) "
            "FROM training_log tl JOIN exercise e ON tl.exercise_id = e._id"
        ).fetchone()[0]
        excl_days = conn.execute(
            "SELECT COUNT(DISTINCT tl.date) "
            "FROM training_log tl JOIN exercise e ON tl.exercise_id = e._id "
            "WHERE e.category_id NOT IN (10, 11, 12)"
        ).fetchone()[0]
    finally:
        conn.close()

    assert ats["total_training_days"] == all_days          # inclusive (317) basis
    assert all_days > excl_days                             # the 9 Time/Place/Neck days
    assert ats["total_training_days"] != excl_days          # NOT the excluded (308) count


# ══════════════════════════════════════════════════════════════════════════════
# 2 — real DB: volume/strength aggregate STILL uses the excluded (308) scope
# ══════════════════════════════════════════════════════════════════════════════

def test_volume_strength_scope_unchanged_excludes_time_place_neck():
    """total_sets (a strength/volume aggregate, = len(alltime_rows)) still counts only
    non-excluded categories — the count-only fix must not widen the volume scope."""
    ats = collect(query_period_days=None)["all_time_summary"]

    conn = _ro_conn()
    try:
        excl_sets = conn.execute(
            "SELECT COUNT(*) "
            "FROM training_log tl JOIN exercise e ON tl.exercise_id = e._id "
            "WHERE e.category_id NOT IN (10, 11, 12)"
        ).fetchone()[0]
    finally:
        conn.close()

    assert ats["total_sets"] == excl_sets                  # excluded (308) scope intact


# ══════════════════════════════════════════════════════════════════════════════
# 3 — SYNTHETIC (no DB): the scope split, known by construction
# ══════════════════════════════════════════════════════════════════════════════

def test_scope_split_count_inclusive_volume_and_streaks_exclusive():
    """Two strength dates (the excluded scope) + an inclusive count of 4 (as if two extra
    Time-only / Place-only days were logged). The count surfaces all 4 while sets, volume,
    boundaries, and streaks reflect ONLY the 2 strength dates."""
    all_dates    = ["2024-06-04", "2024-06-05"]            # 2 consecutive strength days
    alltime_rows = [
        {"metric_weight": 50.0, "reps": 5, "exercise_name": "Bench", "date": "2024-06-04"},
        {"metric_weight": 60.0, "reps": 5, "exercise_name": "Bench", "date": "2024-06-05"},
    ]

    ats = _compute_alltime_summary(
        all_dates, alltime_rows, date(2024, 6, 10),
        ctx=None, pr_event_count=0, total_training_day_count=4,
    )

    # Attendance count = the inclusive number (Time/Place/Neck counted)
    assert ats["total_training_days"] == 4
    # Volume/strength scope = only the 2 strength dates
    assert ats["total_sets"] == 2
    # Boundaries + streaks derive from the 2 strength dates, NOT the inclusive count
    assert ats["first_training_date"] == "2024-06-04"
    assert ats["last_training_date"]  == "2024-06-05"
    assert ats["longest_streak_days"] == 2                 # 2 consecutive strength days
