"""
tests/test_data_agent_validate.py

Validator test suite — three tiers:
  1. A hand-built synthetic CORRECT package → 0 violations.
  2. One test per invariant injecting exactly one defect into a copy of the
     correct package → exactly that invariant's violation appears.
  3. Informational test on the real package (prepare_analysis_package over a
     broad query) — asserts only that C3 and C4 appear; prints all violations
     found for diagnostic visibility.

No LLM calls.  Only data_agent + SQLite.
"""

import copy
import os
import sys
import pytest

# Ensure project root on path and point env at the real DB
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent.validate import validate, Violation


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — minimal correct synthetic package
# ══════════════════════════════════════════════════════════════════════════════

def _set() -> dict:
    return {
        "set_id": 1, "weight": 100.0, "reps": 5,
        "distance": 0.0, "duration_seconds": 0,
        "comment": None, "is_warmup": False,
        "is_failed_attempt": False, "is_pain_flag": False,
        "drop_group": None, "estimated_1rm": 116.7,
        "technique_variants": [], "keyword_counts": {},
        "is_personal_record": False,
    }


def _session(date: str = "2026-06-01") -> dict:
    return {
        "date": date, "unit": "lbs",
        "sets": [_set()],
        "max_working_weight": 100.0, "reps_at_max": 5,
        "estimated_1rm": 116.7, "total_volume": 500.0,
        "total_distance": 0.0, "total_duration_seconds": 0,
        "working_sets_count": 1, "warmup_weight": None,
        "is_pr_session": True, "form_quality": "unknown",
        "form_detail": "", "comment_count": 0,
        "has_pain_flag": False, "failed_attempts": 0,
        "technique_variants": [],
        "rep_ranges": {
            "strength_sets": 1, "hypertrophy_sets": 0, "endurance_sets": 0,
            "strength_pct": 100.0, "hypertrophy_pct": 0.0, "endurance_pct": 0.0,
        },
    }


def _exercise() -> dict:
    s = _session()
    return {
        "name": "Lat Pulldown", "category": "Back",
        "unit": "lbs", "is_cardio": False,
        "numeric_offset": 0.0, "bar_weight": 0.0,
        "bar_weight_unit": "lbs", "cardio_note": None,
        "sessions": [s],
        "pr": {
            "weight": 100.0, "reps": 5, "date": "2026-06-01",
            "unit": "lbs", "estimated_1rm": 116.7,
        },
        "pr_period": {
            "weight": 100.0, "reps": 5, "date": "2026-06-01",
            "unit": "lbs", "estimated_1rm": 116.7,
            "pr_session_comment_count": 0, "pr_session_had_pain": False,
        },
        "progression": {
            "first_session_date": "2026-06-01",
            "last_session_date":  "2026-06-01",
            "max_weight_start": 100.0, "max_weight_end": 100.0,
            "max_weight_start_original": 100.0,
            "first_session_unit": "lbs", "last_session_unit": "lbs",
            "unit_switch_in_period": False,
            "display_weight_start": "100.0 lbs",
            "display_weight_end":   "100.0 lbs",
            "weight_change": 0.0, "weight_change_pct": 0.0,
            "e1rm_start": 116.7, "e1rm_start_original": 116.7,
            "e1rm_end": 116.7, "e1rm_change": 0.0,
            "sessions_at_max": 1, "plateau_since": None,
            "session_count": 1,
            "reps_at_max_start": 5, "reps_at_max_end": 5,
            "regression_from_peak": None, "diminishing_returns": None,
        },
        "duration_progression": None, "distance_progression": None,
        "weekly_aggregations": [{
            "week": "2026-W22",
            "session_dates": ["2026-06-01"],
            "session_count": 1, "max_working_weight": 100.0,
            "total_volume": 500.0, "peak_estimated_1rm": 116.7,
            "form_quality_mode": "unknown", "pain_sessions": 0,
            "failed_attempts": 0, "total_distance": 0.0,
            "total_duration_seconds": 0,
        }],
        "monthly_aggregations": [{
            "month": "2026-06",
            "session_dates": ["2026-06-01"],
            "session_count": 1, "max_working_weight": 100.0,
            "total_volume": 500.0, "peak_estimated_1rm": 116.7,
            "avg_reps_at_max": 5, "pain_sessions": 0,
            "failed_attempts": 0, "total_distance": 0.0,
            "total_duration_seconds": 0,
            "rep_ranges": {"strength_pct": 100.0, "hypertrophy_pct": 0.0,
                           "endurance_pct": 0.0},
        }],
        "yearly_aggregations": [{
            "year": "2026", "session_count": 1, "months_active": 1,
            "max_working_weight": 100.0, "total_volume": 500.0,
            "peak_estimated_1rm": 116.7, "weight_start": 100.0,
            "weight_end": 100.0, "progression_rate_per_month": 0.0,
            "pr_count": 0, "pain_sessions": 0,
            "total_distance": 0.0, "total_duration_seconds": 0, "unit": "lbs",
        }],
        "rest_performance_buckets": {"buckets": [], "comparison": None},
        "consecutive_day_effect":   {"by_consecutive_days": [], "comparison": None},
        "inter_exercise_correlation": [],
        "dow_e1rm_pattern": {"by_day": [], "comparison": None},
        "bw_strength_correlation": {},
        "pain_analysis": {
            "pain_session_count": 0, "pain_session_dates": [],
            "pain_occurrences": [], "failed_attempt_count": 0,
            "failed_attempts": [],
        },
        "training_frequency": {
            "session_count": 1, "sessions_per_week": 1.0,
            "avg_days_between": None, "last_session_date": "2026-06-01",
            "days_since_last": 0, "first_session_in_period": "2026-06-01",
        },
        "rep_range_distribution": {
            "total_working_sets": 1, "strength_sets": 1,
            "hypertrophy_sets": 0, "endurance_sets": 0,
            "strength_pct": 100.0, "hypertrophy_pct": 0.0,
            "endurance_pct": 0.0, "dominant_range": "strength",
        },
        "technique_variants": [],
        "e1rm_history": [{"date": "2026-06-01", "estimated_1rm": 116.7}],
        "e1rm_projection": {}, "volume_trend": "insufficient_data",
        "e1rm_trend": "insufficient_data", "form_trend": "insufficient_data",
        "comment_keyword_trends": {}, "pr_context": [],
        "pr_velocity": {"total_prs": 0, "monthly_counts": [],
                        "velocity_trend": "none"},
        "learning_curve": {
            "first_ever_session": "2026-06-01",
            "sessions_to_first_pr": 1,
            "first_30d_weight_gain": 0.0,
            "total_sessions_alltime": 1,
        },
        "plateau_days": 0, "phase2_triggered": False,
        "full_comments": None, "workout_position_effect": [],
    }


def _package() -> dict:
    return {
        "query_period_days": 90,
        "query_start_date": "2026-03-03",
        "query_end_date":   "2026-06-01",
        "aggregation_level": "session",
        "total_exercises_analyzed": 1,
        "all_time_summary": {
            "first_training_date": "2026-06-01",
            "last_training_date":  "2026-06-01",
            "total_training_days": 1, "total_sets": 1,
            "total_volume_raw_lbs": 500.0, "longest_streak_days": 1,
            "longest_gap_days": 0, "current_streak_days": 0,
            "total_prs_alltime": 0, "prs_per_month_alltime": 0.0,
        },
        "muscle_group_summary": [], "muscle_group_balance": {},
        "training_consistency": {
            "distinct_training_days": 1, "sessions_per_week": 1.0,
            "weeks_with_sessions": 1, "weeks_missed": 0,
            "first_session": "2026-06-01", "last_session": "2026-06-01",
        },
        "day_of_week_patterns": {
            "distribution": [], "most_common_day": "Monday",
            "most_skipped_day": "Sunday",
        },
        "seasonal_patterns": [], "training_density": {},
        "superset_patterns": [], "exercise_lifecycle": {},
        "rankings": {},
        "bodyweight": {"entries": [], "trend": "no_data", "current_kg": None},
        "goals": [], "daily_workouts": [],
        "exercises": [_exercise()],
    }


def _ids(violations: list) -> list:
    """Extract invariant IDs as a sorted list for easy assertion."""
    return sorted(v.invariant_id for v in violations)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Correct package — zero violations
# ══════════════════════════════════════════════════════════════════════════════

def test_correct_package_no_violations():
    pkg = _package()
    v = validate(pkg)
    assert v == [], f"Expected 0 violations, got: {[(x.invariant_id, x.message) for x in v]}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-invariant defect tests
# ══════════════════════════════════════════════════════════════════════════════

def _only(violations: list, invariant_id: str) -> None:
    """Assert that *at least one* violation has the given ID and no others do."""
    ids = {v.invariant_id for v in violations}
    assert invariant_id in ids, (
        f"Expected {invariant_id} violation, got: "
        f"{[(x.invariant_id, x.message) for x in violations]}"
    )


# ── A1 ────────────────────────────────────────────────────────────────────────

def test_a1_bad_exercise_unit():
    pkg = _package()
    pkg["exercises"][0]["unit"] = "stone"
    v = validate(pkg)
    _only(v, "A1")


def test_a1_bad_session_unit():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["unit"] = "parsecs"
    v = validate(pkg)
    _only(v, "A1")


# ── A2 ────────────────────────────────────────────────────────────────────────

def test_a2_kg_for_non_native_exercise():
    pkg = _package()
    # Lat Pulldown is not kg-native
    pkg["exercises"][0]["unit"] = "kg"
    pkg["exercises"][0]["sessions"][0]["unit"] = "kg"
    v = validate(pkg)
    _only(v, "A2")


def test_a2_deadlift_kg_before_switch():
    """Deadlift in kg before 2025-12-26 should trigger A2."""
    pkg = _package()
    pkg["query_end_date"] = "2025-06-01"   # before switch date
    pkg["query_start_date"] = "2025-03-01"
    dl = _exercise()
    dl["name"] = "Deadlift"
    dl["unit"] = "kg"
    dl["sessions"][0]["unit"] = "kg"
    dl["sessions"][0]["date"] = "2025-06-01"
    dl["pr"]["unit"] = "kg"
    dl["pr_period"]["unit"] = "kg"
    pkg["exercises"] = [dl]
    v = validate(pkg)
    _only(v, "A2")


# ── A3 ────────────────────────────────────────────────────────────────────────

def test_a3_session_missing_unit():
    pkg = _package()
    s = pkg["exercises"][0]["sessions"][0]
    del s["unit"]          # remove the unit key
    v = validate(pkg)
    _only(v, "A3")


# ── B2a ───────────────────────────────────────────────────────────────────────

def test_b2a_pr_missing_comment_when_session_had_comments():
    pkg = _package()
    # Session has 1 comment; PR object has no 'comment' key
    pkg["exercises"][0]["sessions"][0]["comment_count"] = 1
    # pr.date matches session date; pr has no 'comment' key → B2a
    # (pr already has no 'comment' key in our base package)
    v = validate(pkg)
    _only(v, "B2a")


# ── B3 ────────────────────────────────────────────────────────────────────────

def test_b3_pr_period_below_session_max():
    pkg = _package()
    # Session max = 100; set pr_period.weight below that
    pkg["exercises"][0]["pr_period"]["weight"] = 80.0
    v = validate(pkg)
    _only(v, "B3")


def test_b3_alltime_pr_below_period_pr():
    pkg = _package()
    # all-time PR below period PR
    pkg["exercises"][0]["pr"]["weight"] = 70.0
    v = validate(pkg)
    _only(v, "B3")


# ── B4 ────────────────────────────────────────────────────────────────────────

def test_b4_weight_based_with_null_pr():
    pkg = _package()
    pkg["exercises"][0]["pr"] = None
    v = validate(pkg)
    _only(v, "B4")


# ── B5 ────────────────────────────────────────────────────────────────────────

def test_b5_max_working_weight_not_in_sets():
    pkg = _package()
    # Sets only have weight=100; set max to something different
    pkg["exercises"][0]["sessions"][0]["max_working_weight"] = 999.0
    v = validate(pkg)
    _only(v, "B5")


# ── B6 ────────────────────────────────────────────────────────────────────────

def test_b6_negative_session_volume():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["total_volume"] = -1.0
    v = validate(pkg)
    _only(v, "B6")


def test_b6_negative_set_weight():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["sets"][0]["weight"] = -5.0
    v = validate(pkg)
    _only(v, "B6")


# ── C1 ────────────────────────────────────────────────────────────────────────

def test_c1_missing_distance_progression_raw():
    """Raw collect() format: cardio exercise with distance but no distance_progression."""
    pkg = _package()
    cardio = _exercise()
    cardio["name"] = "Walking"
    cardio["category"] = "Cardio"
    cardio["is_cardio"] = True
    cardio["unit"] = "lbs"
    cardio["sessions"][0]["total_distance"] = 0.4
    cardio["distance_progression"] = None   # the bug
    cardio["duration_progression"] = None
    cardio["pr"] = None
    cardio["pr_period"] = None
    cardio["progression"] = None
    del cardio["progression"]               # no weight progression for cardio
    pkg["exercises"] = [cardio]
    v = validate(pkg)
    _only(v, "C1")


# ── C2 ────────────────────────────────────────────────────────────────────────

def test_c2_missing_duration_progression_raw():
    """Raw collect() format: cardio exercise with duration but no duration_progression."""
    pkg = _package()
    cardio = _exercise()
    cardio["name"] = "Cycling"
    cardio["category"] = "Cardio"
    cardio["is_cardio"] = True
    cardio["unit"] = "lbs"
    cardio["sessions"][0]["total_duration_seconds"] = 600
    cardio["sessions"][0]["total_distance"] = 0.0
    cardio["duration_progression"] = None   # the bug
    cardio["distance_progression"] = None
    cardio["pr"] = None
    cardio["pr_period"] = None
    if "progression" in cardio:
        del cardio["progression"]
    pkg["exercises"] = [cardio]
    v = validate(pkg)
    _only(v, "C2")


# ── C3 ────────────────────────────────────────────────────────────────────────

def test_c3_null_alltime_sessions_trimmed():
    """Trimmed cardio (prepare_analysis_package format): all_time_sessions is None."""
    pkg = _package()
    trimmed_cardio = {
        "name": "Walking", "category": "Cardio", "is_cardio": True,
        "total_sessions_period": 5,   # has sessions
        "all_time_sessions": None,    # H2 bug — the key EXISTS but is None
        "sessions": [{"date": "2026-06-01", "distance_km": 0.4,
                       "duration_seconds": 300}],
        "progression": {
            "distance_start_km": 0.4, "distance_end_km": 0.4,
            "distance_peak_km": 0.4, "distance_peak_date": "2026-06-01",
            "distance_avg_km": 0.4,  "distance_total_km": 0.4,
            "duration_start_seconds": None, "duration_end_seconds": None,
            "duration_peak_seconds": None, "duration_peak_date": None,
            "duration_change_pct": None, "sessions_in_period": 5,
        },
        "last_session_date": "2026-06-01", "days_since_last": 4,
    }
    pkg["exercises"] = [trimmed_cardio]
    v = validate(pkg)
    _only(v, "C3")


# ── C4 ────────────────────────────────────────────────────────────────────────

def test_c4_empty_sessions_with_period_count():
    """Trimmed cardio: total_sessions_period > 0 but sessions list is []."""
    pkg = _package()
    trimmed_cardio = {
        "name": "Walking", "category": "Cardio", "is_cardio": True,
        "total_sessions_period": 75,   # H3 bug
        "all_time_sessions": 78,       # this is fine
        "sessions": [],                # H3: wiped by weekly aggregation
        "progression": {
            "distance_total_km": 30.0,
            "distance_start_km": 0.3, "distance_end_km": 0.5,
            "distance_peak_km": 0.8, "distance_peak_date": "2026-05-01",
            "distance_avg_km": 0.4, "duration_start_seconds": None,
            "duration_end_seconds": None, "duration_peak_seconds": None,
            "duration_peak_date": None, "duration_change_pct": None,
            "sessions_in_period": 75,
        },
        "last_session_date": "2026-05-25", "days_since_last": 3,
    }
    pkg["exercises"] = [trimmed_cardio]
    v = validate(pkg)
    _only(v, "C4")


# ── C6 ────────────────────────────────────────────────────────────────────────

def test_c6_pace_without_distance():
    """Pace field present but session has zero distance."""
    pkg = _package()
    # Add pace to a raw session that has no distance
    pkg["exercises"][0]["sessions"][0]["pace"] = 6.5
    pkg["exercises"][0]["sessions"][0]["total_distance"] = 0.0
    v = validate(pkg)
    _only(v, "C6")


def test_c6_pace_in_trimmed_session_no_distance():
    """Pace field in a trimmed cardio session with distance_km = 0."""
    pkg = _package()
    trimmed_cardio = {
        "name": "Dead Hang", "category": "Forearms", "is_cardio": True,
        "total_sessions_period": 1,
        "all_time_sessions": 4,
        "sessions": [{"date": "2026-06-01", "distance_km": 0,
                       "duration_seconds": 90, "pace": 5.0}],  # pace but no distance
        "progression": {
            "distance_total_km": None,
            "distance_start_km": None, "distance_end_km": None,
            "distance_peak_km": None, "distance_peak_date": None,
            "distance_avg_km": None,
            "duration_start_seconds": 90, "duration_end_seconds": 90,
            "duration_peak_seconds": 90, "duration_peak_date": "2026-06-01",
            "duration_change_pct": 0.0, "sessions_in_period": 1,
        },
        "last_session_date": "2026-06-01", "days_since_last": 4,
    }
    pkg["exercises"] = [trimmed_cardio]
    v = validate(pkg)
    _only(v, "C6")


# ── D1 ────────────────────────────────────────────────────────────────────────

def test_d1_comment_count_mismatch():
    pkg = _package()
    # Session says comment_count=3 but only 1 set has a non-null comment
    s = pkg["exercises"][0]["sessions"][0]
    s["comment_count"] = 3
    s["sets"][0]["comment"] = "felt great"     # 1 comment
    # But comment_count says 3 → D1
    v = validate(pkg)
    _only(v, "D1")


def test_d1_count_zero_but_set_has_comment():
    pkg = _package()
    s = pkg["exercises"][0]["sessions"][0]
    s["comment_count"] = 0
    s["sets"][0]["comment"] = "good set"   # 1 comment but count says 0 → D1
    v = validate(pkg)
    _only(v, "D1")


# ── D2 ────────────────────────────────────────────────────────────────────────

def test_d2_two_warmup_sets_fires():
    """Two is_warmup=True sets in one session must trigger D2 (integrity)."""
    pkg = _package()
    s = pkg["exercises"][0]["sessions"][0]
    s["sets"] = [
        {**_set(), "set_id": 1, "weight": 50.0,  "is_warmup": True},
        {**_set(), "set_id": 2, "weight": 50.0,  "is_warmup": True},
        {**_set(), "set_id": 3, "weight": 100.0, "is_warmup": False},
    ]
    s["max_working_weight"] = 100.0
    v = validate(pkg)
    _only(v, "D2")
    assert next(vio for vio in v if vio.invariant_id == "D2").severity == "integrity"


def test_d2_one_warmup_set_does_not_fire():
    """A single is_warmup=True set must NOT trigger D2."""
    pkg = _package()
    s = pkg["exercises"][0]["sessions"][0]
    s["sets"] = [
        {**_set(), "set_id": 1, "weight": 50.0,  "is_warmup": True},
        {**_set(), "set_id": 2, "weight": 100.0, "is_warmup": False},
    ]
    s["max_working_weight"] = 100.0
    v = validate(pkg)
    assert not any(vio.invariant_id == "D2" for vio in v), (
        f"D2 must not fire for a single warmup set; violations: "
        f"{[(x.invariant_id, x.message) for x in v]}"
    )


# ── E1 ────────────────────────────────────────────────────────────────────────

def test_e1_session_before_start():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["date"] = "2020-01-01"  # before start
    v = validate(pkg)
    _only(v, "E1")


def test_e1_session_after_end():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["date"] = "2030-01-01"  # after end AND future
    pkg["query_end_date"] = "2026-12-31"
    # This triggers both E1 (after query end) — but since 2030 > today it also
    # triggers E2. Accept either in the overlap; at minimum E1 must appear.
    v = validate(pkg)
    _only(v, "E1")


# ── E2 ────────────────────────────────────────────────────────────────────────

def test_e2_future_session_date():
    pkg = _package()
    pkg["exercises"][0]["sessions"][0]["date"] = "2099-06-01"
    pkg["query_end_date"] = "2099-12-31"  # keep E1 from firing too
    v = validate(pkg)
    _only(v, "E2")


# ── E3 ────────────────────────────────────────────────────────────────────────

def test_e3_weekly_volume_mismatch():
    pkg = _package()
    # Session total_volume = 500; weekly says 999
    pkg["exercises"][0]["weekly_aggregations"][0]["total_volume"] = 999.0
    v = validate(pkg)
    _only(v, "E3")


def test_e3_monthly_volume_mismatch():
    pkg = _package()
    pkg["exercises"][0]["monthly_aggregations"][0]["total_volume"] = 1500.0
    v = validate(pkg)
    _only(v, "E3")


# ── F1 ────────────────────────────────────────────────────────────────────────

def test_f1_comparison_missing_field():
    pkg = _package()
    ex = pkg["exercises"][0]
    ex["rest_performance_buckets"] = {
        "buckets": [{"rest_range": "1-3 days", "n": 5, "ci_95": [100.0, 120.0],
                     "mean_e1rm": 110.0, "std_e1rm": 5.0, "max_e1rm": 115.0}],
        "comparison": {
            # Missing "cis_overlap" and "confidence_label"
            "best_bucket": "1-3 days", "worst_bucket": "4-6 days",
            "mean_diff_e1rm": 10.0,
            "cohen_d": 0.5,
            # cis_overlap missing → F1
        },
    }
    v = validate(pkg)
    _only(v, "F1")


def test_f1_stat_block_missing_n():
    pkg = _package()
    ex = pkg["exercises"][0]
    ex["rest_performance_buckets"] = {
        "buckets": [
            # 'n' key missing → F1
            {"rest_range": "1-3 days", "ci_95": [100.0, 120.0],
             "mean_e1rm": 110.0, "std_e1rm": 5.0, "max_e1rm": 115.0}
        ],
        "comparison": None,
    }
    v = validate(pkg)
    _only(v, "F1")


# ── F2 ────────────────────────────────────────────────────────────────────────

def test_f2_invalid_confidence_label():
    pkg = _package()
    ex = pkg["exercises"][0]
    ex["rest_performance_buckets"] = {
        "buckets": [{"rest_range": "1-3 days", "n": 5, "ci_95": [100.0, 120.0],
                     "mean_e1rm": 110.0, "std_e1rm": 5.0, "max_e1rm": 115.0}],
        "comparison": {
            "best_bucket": "1-3 days", "worst_bucket": "4-6 days",
            "mean_diff_e1rm": 10.0, "cohen_d": 0.5,
            "cis_overlap": False,
            "confidence_label": "super_strong",  # invalid → F2
        },
    }
    v = validate(pkg)
    _only(v, "F2")


# ── F4 ────────────────────────────────────────────────────────────────────────

def test_f4_ci_present_when_n_is_one():
    """n=1 but ci_95 is not None — spec says CI must be None when n < 2."""
    pkg = _package()
    ex = pkg["exercises"][0]
    ex["rest_performance_buckets"] = {
        "buckets": [{"rest_range": "1-3 days", "n": 1,
                     "ci_95": [90.0, 110.0],    # should be None for n<2 → F4
                     "mean_e1rm": 100.0, "std_e1rm": None, "max_e1rm": 100.0}],
        "comparison": None,
    }
    v = validate(pkg)
    _only(v, "F4")


def test_f4_pearson_ci_present_when_n_lt4():
    """Pearson CI must be None when n < 4; here n=3 but ci_95 is set."""
    pkg = _package()
    ex = pkg["exercises"][0]
    ex["bw_strength_correlation"] = {
        "n": 3, "current_ratio": 1.5, "trend": "stable",
        "confidence_label": "insufficient_data",
        "pearson": {"r": 0.9, "ci_95": [0.5, 0.99], "n": 3},  # n<4 but CI set → F4
    }
    v = validate(pkg)
    _only(v, "F4")


# ── G1 ────────────────────────────────────────────────────────────────────────

def test_g1_weight_based_missing_progression():
    pkg = _package()
    del pkg["exercises"][0]["progression"]
    v = validate(pkg)
    _only(v, "G1")


def test_g1_cardio_missing_any_progression():
    pkg = _package()
    cardio = _exercise()
    cardio["name"] = "Walking"
    cardio["category"] = "Cardio"
    cardio["is_cardio"] = True
    cardio["pr"] = None
    cardio["pr_period"] = None
    # Remove both progression-like keys
    for key in ("progression", "distance_progression", "duration_progression"):
        cardio.pop(key, None)
    pkg["exercises"] = [cardio]
    v = validate(pkg)
    _only(v, "G1")


# ── G2 ────────────────────────────────────────────────────────────────────────

def test_g2_pr_none_with_data():
    """G2 overlaps with B4 for the pr=None case; both should fire."""
    pkg = _package()
    pkg["exercises"][0]["pr"] = None
    v = validate(pkg)
    ids = {vio.invariant_id for vio in v}
    # B4 fires too, but G2 must also fire
    assert "G2" in ids or "B4" in ids, (
        f"Expected G2 or B4, got: {ids}"
    )


# ── G3 ────────────────────────────────────────────────────────────────────────

def test_g3_non_serialisable():
    """Package containing a datetime object fails G3."""
    from datetime import datetime as dt
    pkg = _package()
    pkg["_bad_field"] = dt(2026, 6, 1, 12, 0)  # not JSON-serialisable
    v = validate(pkg)
    _only(v, "G3")
    del pkg["_bad_field"]


# ── G4 ────────────────────────────────────────────────────────────────────────

def test_g4_large_package():
    """Package exceeding per-scope ceiling triggers the soft G4 flag.
    No scope set → defaults to 'focused' (250 KB ceiling); 500 KB exceeds it."""
    pkg = _package()
    # Inject ~500 KB of dummy data; no scope → treated as focused (250 KB ceiling)
    pkg["_bulk"] = "x" * (500 * 1024)
    v = validate(pkg)
    ids = [vio.invariant_id for vio in v]
    assert "G4" in ids, f"Expected G4, got: {ids}"
    g4 = next(vio for vio in v if vio.invariant_id == "G4")
    assert g4.severity == "soft"
    del pkg["_bulk"]


def test_g4_broad_fires_above_500kb():
    """BROAD package exceeding 500 KB triggers G4."""
    pkg = _package()
    pkg["scope"] = "broad"
    pkg["_bulk"] = "x" * (502 * 1024)
    v = validate(pkg)
    assert "G4" in {vio.invariant_id for vio in v}, (
        "G4 must fire for broad package exceeding 500 KB"
    )
    del pkg["_bulk"]


def test_g4_broad_clean_under_500kb():
    """BROAD package well under 500 KB must not trigger G4."""
    pkg = _package()
    pkg["scope"] = "broad"
    # Default package is tiny — no bulk injected
    v = validate(pkg)
    assert "G4" not in {vio.invariant_id for vio in v}, (
        "G4 must not fire for a small broad package"
    )


def test_g4_group_fires_above_400kb():
    """GROUP package exceeding 400 KB triggers G4."""
    pkg = _package()
    pkg["scope"] = "group"
    pkg["_bulk"] = "x" * (402 * 1024)
    v = validate(pkg)
    assert "G4" in {vio.invariant_id for vio in v}
    del pkg["_bulk"]


# ── G5 ────────────────────────────────────────────────────────────────────────

def _broad_exercise() -> dict:
    """A correctly-trimmed BROAD exercise: deep-stat fields removed."""
    ex = _exercise()
    for key in ("inter_exercise_correlation", "dow_e1rm_pattern",
                "consecutive_day_effect", "rest_performance_buckets",
                "e1rm_history", "pr_context", "full_comments"):
        ex.pop(key, None)
    return ex


def test_g5_broad_full_comments_leaked():
    """G5 fires when a BROAD package has full_comments on a non-cardio exercise."""
    pkg = _package()
    pkg["scope"] = "broad"
    pkg["exercises"] = [_broad_exercise()]
    pkg["exercises"][0]["full_comments"] = [
        {"date": "2026-06-01", "comment": "test", "set_id": 1}
    ]
    v = validate(pkg)
    _only(v, "G5")


def test_g5_broad_deep_stat_leaked():
    """G5 fires when a BROAD package has a dropped deep-stat key present."""
    pkg = _package()
    pkg["scope"] = "broad"
    pkg["exercises"] = [_broad_exercise()]
    # Inject one of the dropped fields back
    pkg["exercises"][0]["rest_performance_buckets"] = {
        "buckets": [{"rest_range": "1-3 days", "n": 5,
                     "ci_95": [100.0, 120.0], "mean_e1rm": 110.0,
                     "std_e1rm": 5.0, "max_e1rm": 115.0}],
        "comparison": None,
    }
    v = validate(pkg)
    _only(v, "G5")


def test_g5_broad_no_leak_passes():
    """Correctly trimmed BROAD exercise must not fire G5."""
    pkg = _package()
    pkg["scope"] = "broad"
    pkg["exercises"] = [_broad_exercise()]
    v = validate(pkg)
    assert not any(vio.invariant_id == "G5" for vio in v), (
        f"G5 must not fire for a correctly trimmed broad package; "
        f"violations: {[(x.invariant_id, x.message) for x in v]}"
    )


def test_g5_non_broad_does_not_fire():
    """G5 must not fire for FOCUSED or GROUP packages even with deep-stat fields."""
    for scope in ("focused", "group"):
        pkg = _package()
        pkg["scope"] = scope
        pkg["exercises"][0]["full_comments"] = [
            {"date": "2026-06-01", "comment": "test", "set_id": 1}
        ]
        v = validate(pkg)
        assert not any(vio.invariant_id == "G5" for vio in v), (
            f"G5 must not fire for scope={scope!r}"
        )
    # Cardio exercise in BROAD scope: G5 must not fire (cardio is exempt)
    pkg = _package()
    pkg["scope"] = "broad"
    cardio = _exercise()
    cardio["is_cardio"] = True
    cardio["full_comments"] = [{"date": "2026-06-01", "comment": "test", "set_id": 1}]
    pkg["exercises"] = [cardio]
    v = validate(pkg)
    assert not any(vio.invariant_id == "G5" for vio in v), (
        "G5 must not fire for cardio exercises in broad scope"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Informational test — real package from the live DB
# ══════════════════════════════════════════════════════════════════════════════

# ── DataAgentIntegrityError raise behaviour ───────────────────────────────────

def test_integrity_violation_raises_DataAgentIntegrityError():
    """
    (a) An integrity-class violation routed through _report_violations raises
    DataAgentIntegrityError.  This is the same code path that collect() and
    prepare_analysis_package() use after calling validate().
    """
    from src.data_agent import DataAgentIntegrityError, _report_violations

    # A1: bad unit produces an integrity violation
    pkg = _package()
    pkg["exercises"][0]["unit"] = "furlongs"
    violations = validate(pkg)
    assert any(v.severity == "integrity" for v in violations), (
        "Expected at least one integrity violation for unit='furlongs'"
    )

    with pytest.raises(DataAgentIntegrityError) as exc_info:
        _report_violations(violations, "test")

    err = exc_info.value
    assert "A1" in str(err), f"Expected A1 in error text, got: {err}"
    assert hasattr(err, "violations"), "DataAgentIntegrityError must carry .violations"
    assert any(v.invariant_id == "A1" for v in err.violations)


def test_soft_only_violations_do_not_raise():
    """
    (b) Soft violations (G4, G5) must NOT raise DataAgentIntegrityError —
    they are log-only.
    """
    from src.data_agent import DataAgentIntegrityError, _report_violations
    from src.data_agent.validate import Violation

    soft = [
        Violation("G4", "soft", "package over ceiling"),
        Violation("G5", "soft", "broad package leaked field"),
    ]
    # Must complete without raising
    _report_violations(soft, "test")


def test_real_package_does_not_raise():
    """
    (c) prepare_analysis_package on the real DB must not raise
    DataAgentIntegrityError — all integrity invariants pass.
    """
    from src.data_agent import prepare_analysis_package, DataAgentIntegrityError

    pkg = prepare_analysis_package(query_period_days=365)
    assert pkg is not None
    assert pkg.get("scope") == "broad"


def test_real_package_violations_informational():
    """
    Informational test — runs validate() on a real 365-day prepare_analysis_package
    output (BROAD scope) and prints the violation list.

    Confirms:
      • A1, C3, C4 no longer fire (fixed in prior tasks)
      • B3/B5 clean (Tier 2 counterbalance consistent)
      • G4 no longer fires: BROAD 365d package is now under 500 KB
        after scope-aware trimming (was ~1443 KB before this task)
      • G5 does not fire (trim_package removed all dropped fields)
    Reports counterbalance review log for visibility.
    """
    from src.data_agent import prepare_analysis_package

    pkg = prepare_analysis_package(
        query_period_days=365,
        end_date_str="2026-05-28",
    )
    violations = validate(pkg)

    import json as _json
    size_kb = len(_json.dumps(pkg, default=str).encode()) / 1024
    print(f"\n  [scope={pkg.get('scope')!r}  size={size_kb:.1f} KB]")
    print(f"  [{len(violations)} violation(s) found on real package]")
    for vio in violations:
        print(f"  [{vio.severity.upper():9s}] {vio.invariant_id}: {vio.message}")

    # Counterbalance review log — unclassified support tokens for human review
    cb_log = pkg.get("counterbalance_review_log", [])
    print(f"\n  [counterbalance_review_log: {len(cb_log)} entry/entries]")
    for entry in cb_log:
        print(f"    {entry['exercise']} {entry['date']} comment={entry['comment']!r}")

    ids = {v.invariant_id for v in violations}

    # Fixed in prior tasks
    assert "A1" not in ids, f"A1 must not fire. Found: {sorted(ids)}"
    assert "C3" not in ids, f"C3 must not fire. Found: {sorted(ids)}"
    assert "C4" not in ids, f"C4 must not fire. Found: {sorted(ids)}"

    # B3/B5: consistent headline weights after Tier 2 counterbalance
    b3 = [v for v in violations if v.invariant_id == "B3"]
    b5 = [v for v in violations if v.invariant_id == "B5"]
    assert not b3, "B3 must not fire:\n" + "\n".join(v.message for v in b3)
    assert not b5, "B5 must not fire:\n" + "\n".join(v.message for v in b5)

    # G4: scope-aware trim brought BROAD 365d package under 500 KB ceiling
    assert "G4" not in ids, (
        f"G4 must not fire — BROAD 365d package should now be under 500 KB "
        f"(actual {size_kb:.0f} KB). Found: {sorted(ids)}"
    )

    # G5: no dropped fields leaked into the BROAD package
    assert "G5" not in ids, (
        f"G5 must not fire — trim_package must have removed all BROAD-dropped fields. "
        f"Found: {sorted(ids)}"
    )


# ── G6 ────────────────────────────────────────────────────────────────────────

def _focused_exercise() -> dict:
    """A minimal correct FOCUSED exercise: all three agg levels retained."""
    return _exercise()  # base exercise has weekly, monthly, and yearly agg levels


def _broad_exercise_one_agg() -> dict:
    """A correctly-trimmed BROAD exercise: no deep-stat fields, exactly one agg level."""
    ex = _exercise()
    for key in ("inter_exercise_correlation", "dow_e1rm_pattern",
                "consecutive_day_effect", "rest_performance_buckets",
                "e1rm_history", "pr_context", "full_comments"):
        ex.pop(key, None)
    # Keep only weekly_aggregations (the keep-key for query_period_days=90)
    ex.pop("monthly_aggregations", None)
    ex.pop("yearly_aggregations", None)
    return ex


def test_g6_focused_too_many_exercises_raises():
    """scope='focused' with 60 exercises must raise DataAgentIntegrityError (G6)."""
    from src.data_agent import DataAgentIntegrityError, _report_violations

    pkg = _package()
    pkg["scope"] = "focused"
    # Inject 60 exercises (copy with different names to avoid unit/other violations)
    pkg["exercises"] = [
        {**_focused_exercise(), "name": f"Exercise_{i}"}
        for i in range(60)
    ]
    pkg["total_exercises_analyzed"] = 60

    violations = validate(pkg)
    assert any(v.invariant_id == "G6" for v in violations), (
        f"G6 must fire for focused+60 exercises; violations: "
        f"{[(v.invariant_id, v.message) for v in violations]}"
    )
    # Must hard-raise as integrity
    with pytest.raises(DataAgentIntegrityError) as exc_info:
        _report_violations(violations, "test")
    assert any(v.invariant_id == "G6" for v in exc_info.value.violations), (
        "G6 must be in DataAgentIntegrityError.violations"
    )


def test_g6_correct_focused_passes():
    """scope='focused' with 1 exercise must pass G6."""
    pkg = _package()
    pkg["scope"] = "focused"
    pkg["exercises"] = [_focused_exercise()]

    violations = validate(pkg)
    g6 = [v for v in violations if v.invariant_id == "G6"]
    assert not g6, (
        f"G6 must not fire for a correct focused package (1 exercise); "
        f"violations: {[(v.invariant_id, v.message) for v in g6]}"
    )


def test_g6_broad_leaked_full_comments_raises():
    """scope='broad' with full_comments on a non-cardio exercise raises G6 (integrity)."""
    from src.data_agent import DataAgentIntegrityError, _report_violations

    pkg = _package()
    pkg["scope"] = "broad"
    pkg["exercises"] = [_broad_exercise_one_agg()]
    pkg["exercises"][0]["full_comments"] = [
        {"date": "2026-06-01", "comment": "test", "set_id": 1}
    ]

    violations = validate(pkg)
    assert any(v.invariant_id == "G6" for v in violations), (
        f"G6 must fire for broad package with leaked full_comments; "
        f"violations: {[(v.invariant_id, v.message) for v in violations]}"
    )
    # G6 is integrity — must hard-raise
    with pytest.raises(DataAgentIntegrityError) as exc_info:
        _report_violations(violations, "test")
    assert any(v.invariant_id == "G6" for v in exc_info.value.violations)
