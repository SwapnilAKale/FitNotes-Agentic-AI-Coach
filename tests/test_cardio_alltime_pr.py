"""
Live-check fix — Part 3: the cardio PR is ALL-TIME (mirrors strength's pr/pr_period).

The cardio `pr` and `pr_cardio_locked` now run over the all-time session list (threaded
from process_data, where alltime_cache lives); `pr_period` keeps the period stat. The
LOCKED invariant — a PR always carries its comment — holds even when the all-time PR's
date falls OUTSIDE the query period.

No Gemini, no server. Deterministic real-DB targets (sqlite3-verified).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import prepare_analysis_package                # noqa: E402


def _cardio_ex(pkg, name):
    return next(e for e in pkg["exercises"] if e.get("name") == name)


def test_alltime_pr_carries_out_of_period_comment():
    """LOAD-BEARING (locked invariant). Treadmill's all-time max-distance session is
    2.4km/2025-01-14 with a comment; pin a period that contains ONLY the later
    2025-06-26 (0.6km) session. The all-time pr must be the 2.4km/2025-01-14 session
    AND carry its comment (sourced via the all-time enrichment), while pr_period is the
    in-window 0.6km session."""
    pkg = prepare_analysis_package(
        end_date_str="2025-06-26", query_period_days=30, exercise_names=["Treadmill"])
    ex = _cardio_ex(pkg, "Treadmill")
    assert ex["is_cardio"] is True

    pr = ex["pr"]
    assert pr["date"] == "2025-01-14"
    assert pr["distance_km"] == 2.4
    assert pr.get("comment") and pr["comment"].startswith("Reached 2km")
    assert "weight" not in pr                       # cardio PR never carries weight

    pr_period = ex["pr_period"]
    assert pr_period["date"] == "2025-06-26"        # period stat ≠ all-time
    assert pr_period["distance_km"] != pr["distance_km"]


def test_alltime_locked_pr_walking_ten_min_target():
    """pr_cardio_locked (duration ≥ 600s → max distance), all-time = 1.0km / 2025-02-28
    for Walking (clean ≥10-min target; the dirty <600s 5km row is excluded by the lock)."""
    pkg = prepare_analysis_package(
        query_period_days=None, exercise_names=["Walking"],
        cardio_lock={"field": "duration", "value": 600})
    ex = _cardio_ex(pkg, "Walking")
    locked = ex["pr_cardio_locked"]
    assert locked is not None
    assert locked["date"] == "2025-02-28"
    assert locked["distance_km"] == 1.0

# NOTE: the former test_duration_only_alltime_pr_falls_back_to_max_duration was
# live-DB-pinned (Cycling max duration drifted 720s→1800s as sessions were logged)
# and could not be soundly recompute-and-related: Cycling is duration-only, and the
# no-lock cardio-PR rule is max DISTANCE (0 for every Cycling row), so a "correct"
# recompute pins to nothing. Deleted; the duration-only fallback branch is owned by
# the synthetic-by-construction test_CARDIO_PR_duration_only_fallback_and_none
# (tests/test_data_agent_golden.py).
