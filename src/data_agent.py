"""
src/data_agent.py
Data Agent — Complete Implementation

Pure Python, zero LLM calls. Single source of truth for all workout data.

Micro to macro: set-level detail through all-time lifetime analytics.
Every gap covered: drop sets, e1RM, pain flags, technique variants, rep ranges,
form trends, learning curves, PR velocity, rest-performance correlation,
consecutive-day fatigue, workout position effects, inter-exercise correlation,
day-of-week patterns, seasonal patterns, superset detection, exercise lifecycle,
abandoned exercises, substitution detection, muscle balance, rankings,
bodyweight-strength correlation, goal projections, training density, and more.

CRITICAL — Comment join rule:
  LEFT JOIN on Comment. comment = None = unremarkable set, valid data.
  Approximate comment matching is NEVER used.
"""

import re
import sqlite3
import json
import os
from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH           = os.environ.get("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
USER_CONTEXT_PATH = os.environ.get("USER_CONTEXT_PATH", "data/user_context.json")

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORY_NAMES = {
    1: "Shoulders", 2: "Triceps",  3: "Biceps",
    4: "Chest",     5: "Back",     6: "Legs",
    7: "Abs",       8: "Cardio",   9: "Forearms",
}
EXCLUDED_CATEGORY_IDS = (10, 11, 12)
# SQL fragment derived from constant above — used in all NOT IN clauses.
# Update EXCLUDED_CATEGORY_IDS to change which categories are excluded;
# the SQL fragment updates automatically.
_EXCL_SQL = f"({', '.join(str(c) for c in EXCLUDED_CATEGORY_IDS)})"
PUSH_CATEGORIES = {"Shoulders", "Triceps", "Chest"}
PULL_CATEGORIES = {"Back", "Biceps"}

WARMUP_THRESHOLD        = 0.60
DEADLIFT_KG_SWITCH_DATE = date(2025, 12, 26)
PLATEAU_TRIGGER_DAYS    = 28
IMPROVEMENT_TRIGGER_PCT = 20.0
SESSION_MAX_DAYS        = 90
WEEKLY_MAX_DAYS         = 365
ABANDONED_DAYS          = 60
DORMANT_DAYS            = 30

PAIN_KEYWORDS = [
    "pain", "painful", "hurt", "hurts", "hurting", "ache", "aching",
    "injury", "injured", "sore", "soreness", "strain", "strained",
    "cramping", "cramp", "snap", "snapped",
    "wrist pain", "shoulder pain", "elbow pain", "knee pain", "back pain",
]

TECHNIQUE_KEYWORDS = {
    "thumbless":   ["thumbless grip", "no thumb", "thumbless", "no thumb grip"],
    "normal_grip": ["normal grip"],
    "wide_grip":   ["wide grip", "broad grip"],
    "narrow_grip": ["narrow grip"],
    "thumbs_up":   ["thumbs up grip", "thumbs up"],
    "single_hand": ["single hand", "each hand", "one hand", "one arm"],
    "both_hands":  ["both hands", "both hand"],
    "overhand":    ["overhand grip", "overhand"],
    "underhand":   ["underhand grip", "underhand", "supinated"],
    "neutral":     ["neutral grip", "hammer grip"],
}

COMMENT_TREND_KEYWORDS = {
    "partial_reps": ["partial", "partials", "barely", "half"],
    "full_rom":     ["below the neck", "chest", "full", "complete", "fully",
                     "touching", "all the way"],
    "form_break":   ["swinging", "using back", "momentum", "using legs",
                     "cheating", "all back"],
    "controlled":   ["controlled", "strict", "slow"],
    "failure":      ["couldn't", "couldnt", "failed"],
    "pain_effort":  ["pain", "hurt", "ache", "strain"],
}


# ── User context helpers ───────────────────────────────────────────────────────

def _load_user_context() -> dict:
    with open(USER_CONTEXT_PATH, "r") as f:
        return json.load(f)


def _get_numeric_offset(ctx: dict, exercise_name: str) -> float:
    for q in ctx.get("exercise_quirks", []):
        if q.get("exercise_name") == exercise_name:
            return float(q.get("numeric_offset", 0))
    return 0.0


def _is_reps_zero_normal(ctx: dict, exercise_name: str) -> bool:
    """
    Returns True if reps=0 is the normal logging convention for this exercise
    (e.g. Farmers Walk, Dead Hang) — meaning it should NOT be flagged as a
    failed attempt. For time/distance-based exercises (Walking, Cycling,
    Treadmill, Dead Hang), duration_seconds > 0 already handles this at
    the set level. This flag handles weight-based exercises like Farmers Walk
    where duration=0 but reps=0 is still intentional.
    """
    for q in ctx.get("exercise_quirks", []):
        if q.get("exercise_name") == exercise_name:
            return bool(q.get("reps_zero_is_normal", False))
    return False


def _is_kg_native(ctx: dict, exercise_name: str,
                  session_date_str: Optional[str] = None) -> bool:
    kg_native = ctx.get("unit_overrides", {}).get("exercises_in_kg", [])
    if exercise_name not in kg_native:
        return False
    if exercise_name == "Deadlift" and session_date_str:
        return (datetime.strptime(session_date_str, "%Y-%m-%d").date()
                >= DEADLIFT_KG_SWITCH_DATE)
    return True


def _get_bar_weight_lbs(ctx: dict, exercise_name: str,
                        session_date_str: str) -> float:
    bwni  = ctx.get("bar_weights_not_included", {})
    smith = bwni.get("smith_machine", {})
    if "Smith Machine" in exercise_name or exercise_name in smith.get("exercises", []):
        return float(smith.get("bar_weight_lbs", 44.09))
    bar_history = bwni.get("exercise_bar_history", {})
    if exercise_name not in bar_history:
        return 0.0
    entry = bar_history[exercise_name]
    if isinstance(entry, list):
        sess_date = datetime.strptime(session_date_str, "%Y-%m-%d").date()
        for dr in entry:
            from_d = (date.min if dr.get("from") == "start"
                      else datetime.strptime(dr["from"], "%Y-%m-%d").date())
            to_d   = (date.max if dr.get("to") == "present"
                      else datetime.strptime(dr["to"], "%Y-%m-%d").date())
            if from_d <= sess_date <= to_d:
                return float(dr["bar_lbs"])
        return 0.0
    if isinstance(entry, dict) and "bar_lbs" in entry:
        return float(entry["bar_lbs"])
    return 0.0


# ── Math helpers ───────────────────────────────────────────────────────────────

def _recover_typed_weight(metric_weight: float, offset: float) -> float:
    return round(metric_weight * 2.2046 + offset, 1)


def _epley_1rm(weight: float, reps: int) -> float:
    if reps <= 0: return 0.0
    if reps == 1: return round(weight, 1)
    return round(weight * (1 + reps / 30), 1)


def _is_pain_comment(comment: Optional[str]) -> bool:
    if not comment: return False
    c = comment.lower()
    return any(kw in c for kw in PAIN_KEYWORDS)


def _detect_technique_variants(comment: Optional[str]) -> list:
    if not comment: return []
    c = comment.lower()
    return [name for name, kws in TECHNIQUE_KEYWORDS.items() if any(kw in c for kw in kws)]


def _count_keyword_categories(comment: Optional[str]) -> dict:
    if not comment:
        return {k: 0 for k in COMMENT_TREND_KEYWORDS}
    c = comment.lower()
    return {cat: sum(1 for kw in kws if kw in c)
            for cat, kws in COMMENT_TREND_KEYWORDS.items()}


# ── Drop set detection ─────────────────────────────────────────────────────────

_DROP_PATTERN     = re.compile(r'\b(\d+)(st|nd|rd|th)\s+set\b', re.IGNORECASE)
_BACK_TO_BACK_PAT = re.compile(r'\bback\s+to\s+back\b',          re.IGNORECASE)


def _detect_drop_group(comment: Optional[str]) -> Optional[int]:
    if not comment: return None
    m = _DROP_PATTERN.search(comment)
    if m: return int(m.group(1))
    if _BACK_TO_BACK_PAT.search(comment): return 1
    return None


# ── Warmup and form ────────────────────────────────────────────────────────────

def _detect_warmup_flags(sets: list) -> None:
    if not sets: return
    session_max = max(s["weight"] for s in sets)
    threshold   = session_max * WARMUP_THRESHOLD
    for i, s in enumerate(sets):
        s["is_warmup"] = (i == 0 and s["weight"] < threshold)


def _derive_form_quality(working_sets: list) -> tuple:
    comments = [s["comment"] for s in working_sets if s.get("comment")]
    if not comments: return "unknown", ""
    combined = " ".join(comments).lower()
    has_good    = any(kw in combined for kw in ["below the neck", "chest", "full",
                      "complete", "all the way", "fully", "touching", "squeezed",
                      "pushed to the top"])
    has_reduced = any(kw in combined for kw in ["neck up", "above the neck", "half",
                      "not touching", "not fully", "almost"])
    has_poor    = any(kw in combined for kw in ["partial", "partials", "barely",
                      "disgracefully"])
    if   has_poor and has_good:        quality = "mixed"
    elif has_poor:                     quality = "partial"
    elif has_good and not has_reduced: quality = "good"
    elif has_reduced:                  quality = "mixed"
    else:                              quality = "mixed"
    return quality, "; ".join(c for c in comments[:3] if c)


# ── Trend helper ───────────────────────────────────────────────────────────────

def _iso_week_key(date_str: str) -> str:
    """
    Return ISO 8601 week key for a date string: 'YYYY-WNN'.
    Handles year-boundary weeks correctly — a session on 2025-12-29
    (Monday of ISO week 1 of 2026) returns '2026-W01', not '2025-W53'.
    Python's strftime('%Y-%W') gets this wrong across year boundaries.
    """
    iso = datetime.strptime(date_str, "%Y-%m-%d").date().isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _trend(values: list, up_pct: float = 0.10, down_pct: float = 0.10) -> str:
    if len(values) < 4: return "insufficient_data"
    mid  = len(values) // 2
    avg1 = sum(values[:mid]) / mid
    avg2 = sum(values[mid:]) / (len(values) - mid)
    if   avg2 > avg1 * (1 + up_pct):   return "increasing"
    elif avg2 < avg1 * (1 - down_pct):  return "decreasing"
    return "stable"


# ── DB queries ─────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all_sets_in_period(conn: sqlite3.Connection,
                               start_date: str, end_date: str) -> list:
    """
    Single bulk query: ALL sets for ALL exercises in period.
    LEFT JOIN Comment — comment = None = unremarkable set, valid data.
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            tl._id               AS set_id,
            tl.date,
            tl.metric_weight,
            tl.reps,
            tl.distance,
            tl.duration_seconds,
            tl.is_personal_record,
            e.name               AS exercise_name,
            e.category_id,
            c.comment
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        LEFT JOIN Comment c ON c.owner_id = tl._id
        WHERE tl.date >= ? AND tl.date <= ?
          AND e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC, tl._id ASC
    """, (start_date, end_date))
    return [dict(row) for row in cur.fetchall()]


def _fetch_exercise_lifecycle(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT e.name AS exercise_name, e.category_id,
               MIN(tl.date) AS first_date, MAX(tl.date) AS last_date,
               COUNT(*) AS total_sets,
               COUNT(DISTINCT tl.date) AS total_sessions
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
        GROUP BY e._id, e.name, e.category_id
        ORDER BY e.category_id, e.name
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_pr_history(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT tl.date, tl.metric_weight, tl.reps, e.name AS exercise_name
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE tl.is_personal_record = 1
          AND e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_all_bodyweight(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("""
        SELECT date, body_weight_metric, body_fat, comments
        FROM BodyWeight
        ORDER BY date ASC
    """)
    return [{"date": row["date"],
             "weight": row["body_weight_metric"],
             "body_fat": row["body_fat"],
             "comments": row["comments"]}
            for row in cur.fetchall()]


def _fetch_goals(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            g._id          AS goal_id,
            e.name         AS exercise_name,
            g.metric_weight,
            g.reps,
            g.target_date,
            g.start_date,
            g.title        AS notes,
            g.type_id,
            g.unit         AS unit_flag
        FROM Goal g
        JOIN exercise e ON g.exercise_id = e._id
        ORDER BY g.target_date ASC
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_all_training_dates(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT tl.date
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC
    """)
    return [row["date"] for row in cur.fetchall()]


def _fetch_full_comments_for_exercise(conn: sqlite3.Connection,
                                       exercise_name: str) -> list:
    cur = conn.cursor()
    cur.execute("""
        SELECT tl._id AS set_id, tl.date, tl.metric_weight, tl.reps,
               tl.distance, tl.duration_seconds, c.comment
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        LEFT JOIN Comment c ON c.owner_id = tl._id
        WHERE e.name = ? AND c.comment IS NOT NULL
        ORDER BY tl.date ASC, tl._id ASC
    """, (exercise_name,))
    return [dict(row) for row in cur.fetchall()]


# ── Session building ───────────────────────────────────────────────────────────

def _build_sessions_from_rows(rows: list, ctx: dict,
                               exercise_name: str) -> list:
    offset = _get_numeric_offset(ctx, exercise_name)
    by_date: dict = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)

    sessions = []
    for session_date in sorted(by_date.keys()):
        date_rows  = by_date[session_date]
        bar_weight_lbs    = _get_bar_weight_lbs(ctx, exercise_name, session_date)
        _session_is_kg    = _is_kg_native(ctx, exercise_name, session_date)
        bar_weight        = bar_weight_lbs / 2.2046 if _session_is_kg else bar_weight_lbs

        reps_zero_normal = _is_reps_zero_normal(ctx, exercise_name)
        sets = []
        for r in date_rows:
            weight     = _recover_typed_weight(r["metric_weight"], offset)
            reps       = r["reps"]
            comment    = r["comment"]
            distance   = r.get("distance", 0) or 0
            duration_s = r.get("duration_seconds", 0) or 0
            # reps=0 is a failed attempt only if: no duration, no distance,
            # and the exercise doesn't use 0-reps as its normal convention
            is_failed  = (reps == 0
                          and duration_s == 0
                          and distance == 0
                          and not reps_zero_normal)
            sets.append({
                "set_id":             r["set_id"],
                "weight":             weight,
                "reps":               reps,
                "distance":           round(distance, 3),
                "duration_seconds":   int(duration_s),
                "comment":            comment,
                "drop_group":         _detect_drop_group(comment),
                "estimated_1rm":      _epley_1rm(weight, reps),
                "is_failed_attempt":  is_failed,
                "is_pain_flag":       _is_pain_comment(comment),
                "technique_variants": _detect_technique_variants(comment),
                "keyword_counts":     _count_keyword_categories(comment),
                "is_personal_record": bool(r["is_personal_record"]),
                "is_warmup":          False,
            })

        _detect_warmup_flags(sets)
        working_sets = [s for s in sets if not s["is_warmup"]] or sets
        warmup_sets  = [s for s in sets if s["is_warmup"]]

        max_working_weight = max(s["weight"] for s in working_sets)
        reps_at_max        = max((s["reps"] for s in working_sets
                                   if s["weight"] == max_working_weight), default=0)
        session_e1rm       = max((s["estimated_1rm"] for s in working_sets
                                   if s["reps"] > 0), default=0.0)
        total_volume       = sum((s["weight"] + bar_weight) * s["reps"] for s in sets)
        total_distance_m   = sum(s["distance"]         for s in sets)
        total_duration_s   = sum(s["duration_seconds"] for s in sets)

        strength_sets    = sum(1 for s in working_sets if 1 <= s["reps"] <= 5)
        hypertrophy_sets = sum(1 for s in working_sets if 6 <= s["reps"] <= 12)
        endurance_sets   = sum(1 for s in working_sets if s["reps"] >= 13)
        total_ws         = len(working_sets)

        form_quality, form_detail = _derive_form_quality(working_sets)
        unit = "kg" if _is_kg_native(ctx, exercise_name, session_date) else "lbs"

        sessions.append({
            "date":               session_date,
            "unit":               unit,
            "sets":               sets,
            "max_working_weight": max_working_weight,
            "reps_at_max":        reps_at_max,
            "estimated_1rm":      session_e1rm,
            "total_volume":       round(total_volume, 1),
            "total_distance":     round(total_distance_m, 3),
            "total_duration_seconds": total_duration_s,
            "working_sets_count": len(working_sets),
            "warmup_weight":      warmup_sets[0]["weight"] if warmup_sets else None,
            "is_pr_session":      any(s["is_personal_record"] for s in sets),
            "form_quality":       form_quality,
            "form_detail":        form_detail,
            "comment_count":      sum(1 for s in sets if s["comment"]),
            "has_pain_flag":      any(s["is_pain_flag"] for s in sets),
            "failed_attempts":    sum(1 for s in sets if s["is_failed_attempt"]),
            "technique_variants": list({v for s in sets for v in s["technique_variants"]}),
            "rep_ranges": {
                "strength_sets":   strength_sets,
                "hypertrophy_sets":hypertrophy_sets,
                "endurance_sets":  endurance_sets,
                "strength_pct":    round(strength_sets    / total_ws * 100, 1) if total_ws else 0,
                "hypertrophy_pct": round(hypertrophy_sets / total_ws * 100, 1) if total_ws else 0,
                "endurance_pct":   round(endurance_sets   / total_ws * 100, 1) if total_ws else 0,
            },
        })
    return sessions


# ── Aggregations ───────────────────────────────────────────────────────────────

def _get_aggregation_level(period_days: Optional[int]) -> str:
    if period_days is None or period_days > WEEKLY_MAX_DAYS: return "monthly"
    elif period_days > SESSION_MAX_DAYS:                     return "weekly"
    return "session"


def _aggregate_weekly(sessions: list) -> list:
    by_week: dict = defaultdict(lambda: {
        "dates": [], "max_weight": 0.0, "total_volume": 0.0,
        "session_count": 0, "e1rm_values": [], "form_qualities": [],
        "pain_count": 0, "failed_count": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        week = _iso_week_key(s["date"])
        bw   = by_week[week]
        bw["dates"].append(s["date"])
        bw["session_count"]  += 1
        bw["total_volume"]   += s["total_volume"]
        bw["form_qualities"].append(s["form_quality"])
        bw["pain_count"]     += 1 if s["has_pain_flag"] else 0
        bw["failed_count"]   += s["failed_attempts"]
        bw["total_distance"]         += s.get("total_distance", 0) or 0
        bw["total_duration_seconds"] += s.get("total_duration_seconds", 0) or 0
        if s["max_working_weight"] > bw["max_weight"]: bw["max_weight"] = s["max_working_weight"]
        if s["estimated_1rm"] > 0: bw["e1rm_values"].append(s["estimated_1rm"])
    result = []
    for week, d in sorted(by_week.items()):
        e1rm = d["e1rm_values"]; q = d["form_qualities"]
        result.append({
            "week":               week,
            "session_dates":      d["dates"],
            "session_count":      d["session_count"],
            "max_working_weight": d["max_weight"],
            "total_volume":       round(d["total_volume"], 1),
            "peak_estimated_1rm": round(max(e1rm), 1) if e1rm else 0.0,
            "form_quality_mode":  max(set(q), key=q.count) if q else "unknown",
            "pain_sessions":      d["pain_count"],
            "failed_attempts":    d["failed_count"],
            "total_distance":         round(d["total_distance"], 3),
            "total_duration_seconds": d["total_duration_seconds"],
        })
    return result


def _aggregate_monthly(sessions: list) -> list:
    by_month: dict = defaultdict(lambda: {
        "dates": [], "max_weight": 0.0, "total_volume": 0.0,
        "session_count": 0, "e1rm_values": [], "reps_at_max_list": [],
        "pain_count": 0, "failed_count": 0,
        "strength_sets": 0, "hypertrophy_sets": 0, "endurance_sets": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        month = s["date"][:7]; bm = by_month[month]
        bm["dates"].append(s["date"])
        bm["session_count"]    += 1
        bm["total_volume"]     += s["total_volume"]
        bm["reps_at_max_list"].append(s["reps_at_max"])
        bm["pain_count"]       += 1 if s["has_pain_flag"] else 0
        bm["failed_count"]     += s["failed_attempts"]
        bm["total_distance"]         += s.get("total_distance", 0) or 0
        bm["total_duration_seconds"] += s.get("total_duration_seconds", 0) or 0
        bm["strength_sets"]    += s["rep_ranges"]["strength_sets"]
        bm["hypertrophy_sets"] += s["rep_ranges"]["hypertrophy_sets"]
        bm["endurance_sets"]   += s["rep_ranges"]["endurance_sets"]
        if s["max_working_weight"] > bm["max_weight"]: bm["max_weight"] = s["max_working_weight"]
        if s["estimated_1rm"] > 0: bm["e1rm_values"].append(s["estimated_1rm"])
    result = []
    for month, d in sorted(by_month.items()):
        e1rm = d["e1rm_values"]; reps = d["reps_at_max_list"]
        total = d["strength_sets"] + d["hypertrophy_sets"] + d["endurance_sets"]
        result.append({
            "month":              month,
            "session_dates":      d["dates"],
            "session_count":      d["session_count"],
            "max_working_weight": d["max_weight"],
            "total_volume":       round(d["total_volume"], 1),
            "peak_estimated_1rm": round(max(e1rm), 1) if e1rm else 0.0,
            "avg_reps_at_max":    round(sum(reps)/len(reps), 1) if reps else 0,
            "pain_sessions":      d["pain_count"],
            "failed_attempts":    d["failed_count"],
            "total_distance":         round(d["total_distance"], 3),
            "total_duration_seconds": d["total_duration_seconds"],
            "rep_ranges": {
                "strength_pct":    round(d["strength_sets"]    / total * 100, 1) if total else 0,
                "hypertrophy_pct": round(d["hypertrophy_sets"] / total * 100, 1) if total else 0,
                "endurance_pct":   round(d["endurance_sets"]   / total * 100, 1) if total else 0,
            },
        })
    return result


def _aggregate_yearly(sessions: list, unit: str) -> list:
    by_year: dict = defaultdict(lambda: {
        "months": set(), "max_weight": 0.0, "total_volume": 0.0,
        "session_count": 0, "e1rm_values": [],
        "first_weight": None, "last_weight": None,
        "pain_count": 0, "pr_count": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        year = s["date"][:4]; by = by_year[year]
        by["months"].add(s["date"][:7])
        by["session_count"]  += 1
        by["total_volume"]   += s["total_volume"]
        by["pain_count"]     += 1 if s["has_pain_flag"] else 0
        by["pr_count"]       += 1 if s["is_pr_session"] else 0
        if s["max_working_weight"] > by["max_weight"]: by["max_weight"] = s["max_working_weight"]
        if s["estimated_1rm"] > 0: by["e1rm_values"].append(s["estimated_1rm"])
        if by["first_weight"] is None: by["first_weight"] = s["max_working_weight"]
        by["last_weight"] = s["max_working_weight"]
        by["total_distance"]         += s.get("total_distance", 0) or 0
        by["total_duration_seconds"] += s.get("total_duration_seconds", 0) or 0
    result = []
    for year, d in sorted(by_year.items()):
        e1rm = d["e1rm_values"]
        first_w = d["first_weight"] or 0.0; last_w = d["last_weight"] or 0.0
        months_active = len(d["months"])
        prog_rate = round((last_w - first_w) / months_active, 2) if months_active > 0 else 0.0
        result.append({
            "year":                       year,
            "session_count":              d["session_count"],
            "months_active":              months_active,
            "max_working_weight":         d["max_weight"],
            "total_volume":               round(d["total_volume"], 1),
            "peak_estimated_1rm":         round(max(e1rm), 1) if e1rm else 0.0,
            "weight_start":               first_w,
            "weight_end":                 last_w,
            "progression_rate_per_month": prog_rate,
            "pr_count":                   d["pr_count"],
            "pain_sessions":              d["pain_count"],
            "total_distance":             round(d["total_distance"], 3),
            "total_duration_seconds":     d["total_duration_seconds"],
            "unit":                       unit,
        })
    return result


# ── Progression, PR, regression ───────────────────────────────────────────────

def _compute_progression(sessions: list) -> dict:
    if not sessions: return {}
    first = sessions[0]; last = sessions[-1]
    max_weight_start_raw  = first["max_working_weight"]
    max_weight_end        = last["max_working_weight"]
    first_unit            = first.get("unit", "lbs")
    last_unit             = last.get("unit",  "lbs")
    unit_switch_in_period = first_unit != last_unit

    # Normalize start weight to the same unit as the end weight for valid comparison
    if unit_switch_in_period:
        if first_unit == "lbs" and last_unit == "kg":
            max_weight_start = round(max_weight_start_raw / 2.2046, 2)
        elif first_unit == "kg" and last_unit == "lbs":
            max_weight_start = round(max_weight_start_raw * 2.2046, 2)
        else:
            max_weight_start = max_weight_start_raw
    else:
        max_weight_start = max_weight_start_raw

    e1rm_start_raw = first["estimated_1rm"]
    e1rm_end       = last["estimated_1rm"]

    if unit_switch_in_period:
        if first_unit == "lbs" and last_unit == "kg":
            e1rm_start = round(e1rm_start_raw / 2.2046, 1)
        elif first_unit == "kg" and last_unit == "lbs":
            e1rm_start = round(e1rm_start_raw * 2.2046, 1)
        else:
            e1rm_start = e1rm_start_raw
    else:
        e1rm_start = e1rm_start_raw

    weight_change     = round(max_weight_end - max_weight_start, 1)
    weight_change_pct = round((weight_change / max_weight_start * 100)
                               if max_weight_start > 0 else 0.0, 1)

    current_max = max_weight_end; plateau_since = last["date"]
    for s in reversed(sessions):
        if s["max_working_weight"] >= current_max: plateau_since = s["date"]
        else: break
    sessions_at_max = sum(1 for s in sessions if s["max_working_weight"] >= current_max)

    peak_weight = max(s["max_working_weight"] for s in sessions)
    peak_date   = next(s["date"] for s in reversed(sessions)
                       if s["max_working_weight"] == peak_weight)
    regression_from_peak = None
    if max_weight_end < peak_weight:
        regression_from_peak = {
            "peak_weight":       peak_weight,
            "peak_date":         peak_date,
            "current_weight":    max_weight_end,
            "regression_amount": round(peak_weight - max_weight_end, 1),
            "regression_pct":    round((peak_weight - max_weight_end) / peak_weight * 100, 1),
        }

    dim_returns = None
    if len(sessions) >= 6:
        third = len(sessions) // 3
        def rate(seg):
            if len(seg) < 2: return 0.0
            delta = seg[-1]["max_working_weight"] - seg[0]["max_working_weight"]
            days  = max((datetime.strptime(seg[-1]["date"], "%Y-%m-%d").date() -
                         datetime.strptime(seg[0]["date"],  "%Y-%m-%d").date()).days, 1)
            return round(delta / days * 30, 2)
        r1, r2, r3 = rate(sessions[:third]), rate(sessions[third:2*third]), rate(sessions[2*third:])
        pattern = ("diminishing" if r1 > 0 and r3 < r1 * 0.5 else
                   "accelerating" if r3 > r1 * 1.2 else
                   "linear" if r1 != 0 and abs(r3 - r1) < abs(r1) * 0.2 else "irregular")
        dim_returns = {"early_rate_per_month": r1, "mid_rate_per_month": r2,
                       "recent_rate_per_month": r3, "pattern": pattern}

    return {
        "first_session_date":   first["date"],
        "last_session_date":    last["date"],
        "max_weight_start":          max_weight_start,
        "max_weight_start_original": max_weight_start_raw,
        "first_session_unit":        first_unit,
        "last_session_unit":         last_unit,
        "unit_switch_in_period":     unit_switch_in_period,
        "max_weight_end":       max_weight_end,
        "display_weight_start": f"{max_weight_start_raw} {first_unit}",
        "display_weight_end":   f"{max_weight_end} {last_unit}",
        "weight_change":        weight_change,
        "weight_change_pct":    weight_change_pct,
        "e1rm_start":           e1rm_start,
        "e1rm_start_original":  e1rm_start_raw,
        "e1rm_end":             e1rm_end,
        "e1rm_change":          round(e1rm_end - e1rm_start, 1),
        "sessions_at_max":      sessions_at_max,
        "plateau_since":        plateau_since if sessions_at_max > 1 else None,
        "session_count":        len(sessions),
        "reps_at_max_start":    first["reps_at_max"],
        "reps_at_max_end":      last["reps_at_max"],
        "regression_from_peak": regression_from_peak,
        "diminishing_returns":  dim_returns,
    }


def _compute_pr(sessions: list, unit: str) -> Optional[dict]:
    if not sessions: return None
    best_weight = max(s["max_working_weight"] for s in sessions)
    best_reps   = max(s["reps_at_max"] for s in sessions
                      if s["max_working_weight"] == best_weight)
    best_e1rm   = max(s["estimated_1rm"] for s in sessions)
    pr_date     = next(s["date"] for s in reversed(sessions)
                       if s["max_working_weight"] == best_weight
                       and s["reps_at_max"] == best_reps)
    pr_session  = next(s for s in reversed(sessions) if s["date"] == pr_date)
    return {
        "weight":                    best_weight,
        "reps":                      best_reps,
        "date":                      pr_date,
        "estimated_1rm":             best_e1rm,
        "unit":                      unit,
        "pr_session_comment_count":  pr_session["comment_count"],
        "pr_session_had_pain":       pr_session["has_pain_flag"],
    }


def _compute_duration_progression(sessions: list) -> Optional[dict]:
    """
    Progression based on duration_seconds for time-based exercises (e.g. Dead Hang).
    Called only for exercises where weight is always 0.
    """
    dur_sessions = [(s["date"], s.get("total_duration_seconds", 0))
                    for s in sessions if s.get("total_duration_seconds", 0) > 0]
    if not dur_sessions:
        return None
    if len(dur_sessions) < 2:
        return {
            "session_count":          1,
            "duration_start_seconds": dur_sessions[0][1],
            "note":                   "only_one_session",
        }
    first_dur = dur_sessions[0][1]
    last_dur  = dur_sessions[-1][1]
    peak_dur  = max(d for _, d in dur_sessions)
    peak_date = next(dt for dt, d in reversed(dur_sessions) if d == peak_dur)
    dur_change     = last_dur - first_dur
    dur_change_pct = round((dur_change / first_dur * 100) if first_dur > 0 else 0.0, 1)
    current = last_dur; plateau_since = dur_sessions[-1][0]
    for dt, dur in reversed(dur_sessions):
        if dur >= current: plateau_since = dt
        else: break
    return {
        "first_session_date":      dur_sessions[0][0],
        "last_session_date":       dur_sessions[-1][0],
        "duration_start_seconds":  first_dur,
        "duration_end_seconds":    last_dur,
        "duration_peak_seconds":   peak_dur,
        "duration_peak_date":      peak_date,
        "duration_change_seconds": dur_change,
        "duration_change_pct":     dur_change_pct,
        "plateau_since":           plateau_since,
        "session_count":           len(dur_sessions),
    }


def _compute_distance_progression(sessions: list) -> Optional[dict]:
    """
    Progression based on total_distance (km) for distance-based exercises
    (e.g. Walking, Treadmill). Called only for exercises where weight is always 0.
    """
    dist_sessions = [(s["date"], round(s.get("total_distance", 0), 3))
                     for s in sessions if s.get("total_distance", 0) > 0]
    if len(dist_sessions) < 2:
        return None
    first_dist = dist_sessions[0][1]
    last_dist  = dist_sessions[-1][1]
    peak_dist  = max(d for _, d in dist_sessions)
    peak_date  = next(dt for dt, d in reversed(dist_sessions) if d == peak_dist)
    avg_dist   = round(sum(d for _, d in dist_sessions) / len(dist_sessions), 3)
    total_dist = round(sum(d for _, d in dist_sessions), 3)
    return {
        "first_session_date": dist_sessions[0][0],
        "last_session_date":  dist_sessions[-1][0],
        "distance_start_km":  first_dist,
        "distance_end_km":    last_dist,
        "distance_peak_km":   peak_dist,
        "distance_peak_date": peak_date,
        "avg_distance_km":    avg_dist,
        "total_distance_km":  total_dist,
        "session_count":      len(dist_sessions),
    }


def _evaluate_phase2(progression: dict, end_date: date) -> tuple:
    plateau_days = 0
    if progression.get("plateau_since"):
        plateau_dt   = datetime.strptime(progression["plateau_since"], "%Y-%m-%d").date()
        plateau_days = (end_date - plateau_dt).days
    triggered = (plateau_days > PLATEAU_TRIGGER_DAYS or
                 progression.get("weight_change_pct", 0) > IMPROVEMENT_TRIGGER_PCT)
    return triggered, plateau_days


# ── Per-exercise analytics ─────────────────────────────────────────────────────

def _compute_form_trend(sessions: list) -> str:
    qmap = {"good": 2, "mixed": 1, "partial": 0, "unknown": None}
    scores = [qmap[s["form_quality"]] for s in sessions if qmap.get(s["form_quality"]) is not None]
    return _trend(scores, up_pct=0.15, down_pct=0.15)


def _compute_comment_keyword_trends(sessions: list) -> dict:
    if len(sessions) < 4: return {}
    mid = len(sessions) // 2
    result = {}
    for cat in COMMENT_TREND_KEYWORDS:
        first_half  = sum(st["keyword_counts"].get(cat, 0)
                          for s in sessions[:mid] for st in s["sets"])
        second_half = sum(st["keyword_counts"].get(cat, 0)
                          for s in sessions[mid:] for st in s["sets"])
        total = first_half + second_half
        trend = ("none" if total == 0 else
                 "increasing" if second_half > first_half * 1.2 else
                 "decreasing" if second_half < first_half * 0.8 else "stable")
        result[cat] = {"first_half_count": first_half,
                       "second_half_count": second_half, "trend": trend}
    return result


def _compute_training_frequency(sessions: list, start_date: str, end_date: str) -> dict:
    if not sessions: return {}
    dates = [s["date"] for s in sessions]
    first_dt = datetime.strptime(dates[0],  "%Y-%m-%d").date()
    last_dt  = datetime.strptime(dates[-1], "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date,  "%Y-%m-%d").date()
    avg_between = round((last_dt - first_dt).days / (len(dates) - 1), 1) if len(dates) > 1 else None
    period_weeks = max((end_dt - datetime.strptime(start_date, "%Y-%m-%d").date()).days / 7, 1)
    return {
        "session_count":           len(dates),
        "sessions_per_week":       round(len(dates) / period_weeks, 2),
        "avg_days_between":        avg_between,
        "last_session_date":       dates[-1],
        "days_since_last":         (end_dt - last_dt).days,
        "first_session_in_period": dates[0],
    }


def _compute_rest_performance_buckets(sessions: list) -> list:
    if len(sessions) < 3: return []
    data = []
    for i in range(1, len(sessions)):
        prev_dt = datetime.strptime(sessions[i-1]["date"], "%Y-%m-%d").date()
        curr_dt = datetime.strptime(sessions[i]["date"],   "%Y-%m-%d").date()
        rest    = (curr_dt - prev_dt).days
        if sessions[i]["estimated_1rm"] > 0:
            data.append((rest, sessions[i]["estimated_1rm"]))
    if not data: return []
    buckets = {"1-3 days": [], "4-6 days": [], "7-13 days": [], "14+ days": []}
    for rest, e1rm in data:
        if   rest <= 3:   buckets["1-3 days"].append(e1rm)
        elif rest <= 6:   buckets["4-6 days"].append(e1rm)
        elif rest <= 13:  buckets["7-13 days"].append(e1rm)
        else:             buckets["14+ days"].append(e1rm)
    return [{"rest_range": b, "session_count": len(e1rms),
             "avg_e1rm": round(sum(e1rms)/len(e1rms), 1),
             "max_e1rm": round(max(e1rms), 1)}
            for b, e1rms in buckets.items() if e1rms]


def _compute_consecutive_day_effect(sessions: list, all_dates: list) -> list:
    if not sessions or not all_dates: return []
    date_set = set(all_dates)
    def consec_before(d_str):
        dt = datetime.strptime(d_str, "%Y-%m-%d").date(); count = 0
        prev = dt - timedelta(days=1)
        while prev.strftime("%Y-%m-%d") in date_set:
            count += 1; prev -= timedelta(days=1)
        return count
    by_c: dict = defaultdict(list)
    for s in sessions:
        c = consec_before(s["date"])
        if s["estimated_1rm"] > 0: by_c[c].append(s["estimated_1rm"])
    return [{"consecutive_days_before": c, "session_count": len(e1rms),
             "avg_e1rm": round(sum(e1rms)/len(e1rms), 1)}
            for c, e1rms in sorted(by_c.items())]


def _compute_rep_range_distribution(sessions: list) -> dict:
    s = h = e = 0
    for sess in sessions:
        s += sess["rep_ranges"]["strength_sets"]
        h += sess["rep_ranges"]["hypertrophy_sets"]
        e += sess["rep_ranges"]["endurance_sets"]
    total = s + h + e
    dominant = ("strength" if s >= h and s >= e else
                "hypertrophy" if h >= e else "endurance") if total else "unknown"
    return {
        "total_working_sets": total,
        "strength_sets": s, "hypertrophy_sets": h, "endurance_sets": e,
        "strength_pct":    round(s/total*100, 1) if total else 0,
        "hypertrophy_pct": round(h/total*100, 1) if total else 0,
        "endurance_pct":   round(e/total*100, 1) if total else 0,
        "dominant_range":  dominant,
    }


def _compute_technique_variants(sessions: list, unit: str) -> list:
    by_v: dict = defaultdict(lambda: {"e1rm_values": [], "session_count": 0})
    for s in sessions:
        for v in s["technique_variants"]:
            by_v[v]["session_count"] += 1
            if s["estimated_1rm"] > 0: by_v[v]["e1rm_values"].append(s["estimated_1rm"])
    return [{"variant": v, "session_count": d["session_count"],
             "avg_e1rm": round(sum(d["e1rm_values"])/len(d["e1rm_values"]), 1) if d["e1rm_values"] else 0.0,
             "max_e1rm": round(max(d["e1rm_values"]), 1) if d["e1rm_values"] else 0.0, "unit": unit}
            for v, d in sorted(by_v.items())]


def _compute_pain_analysis(sessions: list) -> dict:
    pain_sessions = [s for s in sessions if s["has_pain_flag"]]
    pain_occurrences = [{"date": s["date"], "set_id": st["set_id"], "comment": st["comment"]}
                        for s in sessions for st in s["sets"] if st["is_pain_flag"]]
    failed_sets = [{"date": s["date"], "weight": st["weight"], "comment": st["comment"]}
                   for s in sessions for st in s["sets"] if st["is_failed_attempt"]]
    return {
        "pain_session_count":    len(pain_sessions),
        "pain_session_dates":    [s["date"] for s in pain_sessions],
        "pain_occurrences":      pain_occurrences,
        "failed_attempt_count":  sum(s["failed_attempts"] for s in sessions),
        "failed_attempts":       failed_sets,
    }


def _compute_pr_velocity(pr_history: list, exercise_name: str) -> dict:
    exercise_prs = [r for r in pr_history if r["exercise_name"] == exercise_name]
    if not exercise_prs: return {"total_prs": 0, "monthly_counts": [], "velocity_trend": "none"}
    by_month: dict = defaultdict(int)
    for pr in exercise_prs: by_month[pr["date"][:7]] += 1
    monthly = [{"month": m, "pr_count": c} for m, c in sorted(by_month.items())]
    return {
        "total_prs":      len(exercise_prs),
        "monthly_counts": monthly,
        "velocity_trend": _trend([m["pr_count"] for m in monthly], up_pct=0.20, down_pct=0.20),
    }


def _compute_learning_curve(all_time_sessions: list) -> dict:
    if not all_time_sessions: return {}
    first_pr_session = next((i+1 for i, s in enumerate(all_time_sessions)
                              if s["is_pr_session"]), None)
    first_date = datetime.strptime(all_time_sessions[0]["date"], "%Y-%m-%d").date()
    early = [s for s in all_time_sessions
             if (datetime.strptime(s["date"], "%Y-%m-%d").date() - first_date).days <= 30]
    first_30d_gain = round(early[-1]["max_working_weight"] - early[0]["max_working_weight"], 1) \
                     if len(early) >= 2 else 0.0
    return {
        "first_ever_session":     all_time_sessions[0]["date"],
        "sessions_to_first_pr":   first_pr_session,
        "first_30d_weight_gain":  first_30d_gain,
        "total_sessions_alltime": len(all_time_sessions),
    }


def _compute_e1rm_projection(sessions: list) -> dict:
    if len(sessions) < 4: return {}
    recent = sessions[-min(8, len(sessions)):]
    e1rms  = [s["estimated_1rm"] for s in recent if s["estimated_1rm"] > 0]
    if len(e1rms) < 2: return {}
    days_span  = max((datetime.strptime(recent[-1]["date"], "%Y-%m-%d").date() -
                      datetime.strptime(recent[0]["date"],  "%Y-%m-%d").date()).days, 1)
    daily_rate = (e1rms[-1] - e1rms[0]) / days_span
    current    = e1rms[-1]
    confidence = ("high" if _trend(e1rms, 0.05, 0.05) in ("increasing", "decreasing")
                  else "medium" if len(e1rms) >= 4 else "low")
    return {
        "current_e1rm":          round(current, 1),
        "projected_30d":         round(current + daily_rate * 30, 1),
        "projected_60d":         round(current + daily_rate * 60, 1),
        "projected_90d":         round(current + daily_rate * 90, 1),
        "daily_rate":            round(daily_rate, 3),
        "projection_confidence": confidence,
    }


def _compute_pr_context(sessions: list, all_dates: list, bw_entries: list) -> list:
    pr_sessions   = [s for s in sessions if s["is_pr_session"]]
    if not pr_sessions: return []
    date_set      = set(all_dates)
    bw_by_date    = {e["date"]: e["weight"] for e in bw_entries}
    bw_dates_sort = sorted(bw_by_date.keys())
    session_dates = [s["date"] for s in sessions]

    def nearest_bw(d_str):
        if not bw_dates_sort: return None
        nearest = min(bw_dates_sort, key=lambda d: abs(
            (datetime.strptime(d, "%Y-%m-%d").date() -
             datetime.strptime(d_str, "%Y-%m-%d").date()).days))
        gap = abs((datetime.strptime(nearest, "%Y-%m-%d").date() -
                   datetime.strptime(d_str,   "%Y-%m-%d").date()).days)
        return bw_by_date[nearest] if gap <= 14 else None

    result = []
    for s in pr_sessions:
        idx  = session_dates.index(s["date"])
        rest = None
        if idx > 0:
            prev = datetime.strptime(session_dates[idx-1], "%Y-%m-%d").date()
            curr = datetime.strptime(s["date"],             "%Y-%m-%d").date()
            rest = (curr - prev).days
        dt = datetime.strptime(s["date"], "%Y-%m-%d").date()
        consec = 0; prev_d = dt - timedelta(days=1)
        while prev_d.strftime("%Y-%m-%d") in date_set:
            consec += 1; prev_d -= timedelta(days=1)
        result.append({
            "date":                             s["date"],
            "max_weight":                       s["max_working_weight"],
            "estimated_1rm":                    s["estimated_1rm"],
            "days_since_last_exercise_session": rest,
            "consecutive_training_days_before": consec,
            "bodyweight_kg":                    nearest_bw(s["date"]),
            "comment_count":                    s["comment_count"],
            "had_pain":                         s["has_pain_flag"],
        })
    return result


def _compute_goal_projection(sessions: list, goal: dict, unit: str) -> dict:
    if not sessions or not goal: return {}
    target_weight = goal["target_weight"]
    target_e1rm   = _epley_1rm(target_weight, goal.get("target_reps", 1))
    current_e1rm  = sessions[-1]["estimated_1rm"] if sessions else 0
    current_max   = sessions[-1]["max_working_weight"] if sessions else 0
    recent = sessions[-min(12, len(sessions)):]
    monthly_rate  = 0.0
    if len(recent) >= 2:
        days_span = max((datetime.strptime(recent[-1]["date"], "%Y-%m-%d").date() -
                         datetime.strptime(recent[0]["date"],  "%Y-%m-%d").date()).days, 1)
        monthly_rate = (recent[-1]["estimated_1rm"] - recent[0]["estimated_1rm"]) / days_span * 30
    e1rm_gap = target_e1rm - current_e1rm
    projected_date = None; months_needed = None; is_on_track = False
    if monthly_rate > 0 and e1rm_gap > 0:
        months_needed  = round(e1rm_gap / monthly_rate, 1)
        proj_dt        = date.today() + timedelta(days=int(months_needed * 30))
        projected_date = proj_dt.strftime("%Y-%m-%d")
        if goal.get("target_date"):
            is_on_track = proj_dt <= datetime.strptime(goal["target_date"], "%Y-%m-%d").date()
    elif e1rm_gap <= 0:
        is_on_track = True; projected_date = "already_achieved"; months_needed = 0
    return {
        "target_weight": target_weight, "target_reps": goal.get("target_reps", 1),
        "target_date": goal.get("target_date"), "target_e1rm": round(target_e1rm, 1),
        "current_max_weight": current_max, "current_e1rm": round(current_e1rm, 1),
        "monthly_e1rm_rate": round(monthly_rate, 2),
        "months_needed": months_needed,
        "projected_achievement_date": projected_date,
        "is_on_track": is_on_track, "unit": unit,
    }


# ── Daily workout view ─────────────────────────────────────────────────────────

def _build_daily_workouts(all_rows: list, ctx: dict) -> list:
    by_date: dict = defaultdict(list)
    for row in all_rows: by_date[row["date"]].append(row)
    daily = []
    for training_date in sorted(by_date.keys()):
        rows = by_date[training_date]
        by_ex: dict = defaultdict(list)
        for row in rows: by_ex[row["exercise_name"]].append(row)
        exercise_order = sorted(by_ex.keys(),
                                key=lambda ex: min(r["set_id"] for r in by_ex[ex]))
        exercises_done = []; total_vol = 0.0; total_sets = 0
        for pos, ex_name in enumerate(exercise_order, start=1):
            ex_rows    = by_ex[ex_name]
            offset     = _get_numeric_offset(ctx, ex_name)
            bar_weight_lbs = _get_bar_weight_lbs(ctx, ex_name, training_date)
            bar_weight     = bar_weight_lbs / 2.2046 if _is_kg_native(ctx, ex_name, training_date) else bar_weight_lbs
            category   = CATEGORY_NAMES.get(ex_rows[0]["category_id"],
                                             f"Cat_{ex_rows[0]['category_id']}")
            ex_sets = [{"weight":            _recover_typed_weight(r["metric_weight"], offset),
                        "reps":              r["reps"],
                        "distance":          round(r.get("distance", 0) or 0, 3),
                        "duration_seconds":  int(r.get("duration_seconds", 0) or 0),
                        "is_warmup":         False} for r in ex_rows]
            _detect_warmup_flags(ex_sets)
            working = [s for s in ex_sets if not s["is_warmup"]] or ex_sets
            max_w   = max(s["weight"] for s in working)
            vol     = sum((s["weight"] + bar_weight) * s["reps"] for s in ex_sets)
            e1rm    = max((_epley_1rm(s["weight"], s["reps"]) for s in working
                           if s["reps"] > 0), default=0.0)
            total_vol  += vol; total_sets += len(working)
            exercises_done.append({
                "position": pos, "exercise_name": ex_name, "category": category,
                "working_sets": len(working), "max_weight": max_w,
                "volume": round(vol, 1), "estimated_1rm": round(e1rm, 1),
                "total_distance":         round(sum(s.get("distance", 0) for s in ex_sets), 3),
                "total_duration_seconds": sum(s.get("duration_seconds", 0) for s in ex_sets),
            })
        daily.append({
            "date":              training_date,
            "day_of_week":       datetime.strptime(training_date, "%Y-%m-%d").strftime("%A"),
            "exercises_count":   len(exercise_order),
            "total_sets":        total_sets,
            "total_volume":      round(total_vol, 1),
            "exercises":         exercises_done,
            "categories_trained":list({e["category"] for e in exercises_done}),
        })
    return daily


def _compute_exercise_workout_position(sessions: list, daily_workouts: list,
                                        exercise_name: str) -> list:
    pos_lookup = {day["date"]: ex["position"]
                  for day in daily_workouts for ex in day["exercises"]
                  if ex["exercise_name"] == exercise_name}
    by_pos: dict = defaultdict(list)
    for s in sessions:
        pos = pos_lookup.get(s["date"])
        if pos and s["estimated_1rm"] > 0:
            by_pos[pos if pos <= 4 else "5+"].append(s["estimated_1rm"])
    return [{"workout_position": p, "session_count": len(e1rms),
             "avg_e1rm": round(sum(e1rms)/len(e1rms), 1)}
            for p, e1rms in sorted(by_pos.items(),
                                    key=lambda x: int(x[0]) if str(x[0]).isdigit() else 99)]


def _detect_supersets(all_rows: list) -> list:
    by_date: dict = defaultdict(list)
    for row in all_rows: by_date[row["date"]].append(row)
    supersets = []
    for training_date, rows in by_date.items():
        by_ex: dict = defaultdict(list)
        for r in rows: by_ex[r["exercise_name"]].append(r["set_id"])
        names = list(by_ex.keys())
        for i, ex_a in enumerate(names):
            for ex_b in names[i+1:]:
                ids_a = sorted(by_ex[ex_a]); ids_b = sorted(by_ex[ex_b])
                all_ids    = sorted([(sid, "A") for sid in ids_a] +
                                    [(sid, "B") for sid in ids_b])
                transitions = sum(1 for j in range(1, len(all_ids))
                                  if all_ids[j][1] != all_ids[j-1][1])
                total = len(ids_a) + len(ids_b)
                score = transitions / max(total - 1, 1)
                if score >= 0.5 and total >= 4:
                    row_a = next(r for r in rows if r["exercise_name"] == ex_a)
                    row_b = next(r for r in rows if r["exercise_name"] == ex_b)
                    if row_a["category_id"] != row_b["category_id"]:
                        supersets.append({
                            "date": training_date, "exercise_a": ex_a,
                            "exercise_b": ex_b, "interleave_score": round(score, 2),
                        })
    return supersets


def _compute_inter_exercise_correlation(sessions: list, exercise_name: str,
                                         daily_workouts: list) -> list:
    before_map = {
        day["date"]: [ex["exercise_name"] for ex in day["exercises"]
                      if ex["position"] < next(
                          (e["position"] for e in day["exercises"]
                           if e["exercise_name"] == exercise_name), 999)]
        for day in daily_workouts
        if any(e["exercise_name"] == exercise_name for e in day["exercises"])
    }
    if not before_map: return []
    all_others = set(ex for exs in before_map.values() for ex in exs)
    result = []
    for other in all_others:
        preceded     = [s["estimated_1rm"] for s in sessions
                        if s["date"] in before_map
                        and other in before_map[s["date"]]
                        and s["estimated_1rm"] > 0]
        not_preceded = [s["estimated_1rm"] for s in sessions
                        if (s["date"] not in before_map
                            or other not in before_map.get(s["date"], []))
                        and s["estimated_1rm"] > 0]
        if len(preceded) >= 2 and len(not_preceded) >= 2:
            avg_p = round(sum(preceded)/len(preceded), 1)
            avg_n = round(sum(not_preceded)/len(not_preceded), 1)
            diff  = round((avg_p - avg_n) / avg_n * 100, 1) if avg_n else 0
            result.append({
                "preceding_exercise":      other,
                "sessions_preceded_by":    len(preceded),
                "avg_e1rm_when_preceded":  avg_p,
                "sessions_not_preceded":   len(not_preceded),
                "avg_e1rm_when_not":       avg_n,
                "e1rm_diff_pct":           diff,
                "effect": ("negative" if diff < -5 else "positive" if diff > 5 else "neutral"),
            })
    return sorted(result, key=lambda x: abs(x["e1rm_diff_pct"]), reverse=True)


# ── Global analytics ───────────────────────────────────────────────────────────

def _compute_day_of_week_patterns(all_dates: list, start_date: str, end_date: str) -> dict:
    period = [d for d in all_dates if start_date <= d <= end_date]
    dow_count: dict = defaultdict(int)
    for d in period: dow_count[datetime.strptime(d, "%Y-%m-%d").strftime("%A")] += 1
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return {
        "distribution":    [{"day": d, "count": dow_count.get(d, 0)} for d in days],
        "most_common_day": max(dow_count, key=dow_count.get) if dow_count else None,
        "most_skipped_day":min(days, key=lambda d: dow_count.get(d, 0)),
    }


def _compute_exercise_dow_e1rm(sessions: list) -> list:
    by_dow: dict = defaultdict(list)
    for s in sessions:
        dow = datetime.strptime(s["date"], "%Y-%m-%d").strftime("%A")
        if s["estimated_1rm"] > 0: by_dow[dow].append(s["estimated_1rm"])
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return [{"day": d, "session_count": len(by_dow[d]),
             "avg_e1rm": round(sum(by_dow[d])/len(by_dow[d]), 1)}
            for d in days if by_dow.get(d)]


def _compute_seasonal_patterns(all_dates: list) -> list:
    by_cal: dict = defaultdict(lambda: {"years": set(), "day_count": 0})
    for d in all_dates:
        by_cal[d[5:7]]["years"].add(d[:4]); by_cal[d[5:7]]["day_count"] += 1
    month_names = {"01":"January","02":"February","03":"March","04":"April",
                   "05":"May","06":"June","07":"July","08":"August",
                   "09":"September","10":"October","11":"November","12":"December"}
    return [{"month_num": m, "month_name": month_names.get(m, m),
             "avg_training_days": round(d["day_count"]/len(d["years"]), 1) if d["years"] else 0,
             "total_training_days": d["day_count"], "years_with_data": len(d["years"])}
            for m, d in sorted(by_cal.items())]


def _compute_alltime_summary(all_dates: list, conn, pr_history: list) -> dict:
    if not all_dates: return {}
    date_objs = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in all_dates)
    max_streak = cur_streak = 1
    for i in range(1, len(date_objs)):
        if (date_objs[i] - date_objs[i-1]).days == 1:
            cur_streak += 1; max_streak = max(max_streak, cur_streak)
        else: cur_streak = 1
    today = date.today(); cur_now = 0; dates_set = set(all_dates)
    check = today
    while check.strftime("%Y-%m-%d") in dates_set: cur_now += 1; check -= timedelta(days=1)
    max_gap = max((date_objs[i] - date_objs[i-1]).days for i in range(1, len(date_objs))) if len(date_objs) > 1 else 0
    first_dt    = date_objs[0]; last_dt = date_objs[-1]
    months_total = max((last_dt - first_dt).days / 30, 1)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*) AS cnt FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
    """)
    total_sets = cur.fetchone()["cnt"]
    cur.execute(f"""
        SELECT SUM(tl.metric_weight * 2.2046 * tl.reps) AS vol
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
    """)
    row = cur.fetchone()
    total_volume_raw = round(row["vol"] or 0, 0)
    return {
        "first_training_date":   all_dates[0],
        "last_training_date":    all_dates[-1],
        "total_training_days":   len(set(all_dates)),
        "total_sets":            total_sets,
        "total_volume_raw_lbs":  total_volume_raw,
        "longest_streak_days":   max_streak,
        "longest_gap_days":      max_gap,
        "current_streak_days":   cur_now,
        "total_prs_alltime":     len(pr_history),
        "prs_per_month_alltime": round(len(pr_history) / months_total, 2),
    }


def _compute_exercise_lifecycle(lifecycle_rows: list, end_date: str) -> dict:
    today = datetime.strptime(end_date, "%Y-%m-%d").date()
    lifecycle = []
    for row in lifecycle_rows:
        last_dt       = datetime.strptime(row["last_date"], "%Y-%m-%d").date()
        days_inactive = (today - last_dt).days
        status = ("abandoned" if days_inactive >= ABANDONED_DAYS else
                  "dormant"   if days_inactive >= DORMANT_DAYS   else "active")
        lifecycle.append({
            "exercise_name":   row["exercise_name"],
            "category":        CATEGORY_NAMES.get(row["category_id"], "Other"),
            "category_id":     row["category_id"],
            "first_date":      row["first_date"],
            "last_date":       row["last_date"],
            "total_sessions":  row["total_sessions"],
            "total_sets":      row["total_sets"],
            "days_since_last": days_inactive,
            "status":          status,
        })
    abandoned = [e for e in lifecycle if e["status"] == "abandoned"]
    active    = [e for e in lifecycle if e["status"] == "active"]
    substitutions = []
    for ab in abandoned:
        ab_last = datetime.strptime(ab["last_date"], "%Y-%m-%d").date()
        for ac in active:
            if ac["category_id"] == ab["category_id"] and ac["exercise_name"] != ab["exercise_name"]:
                ac_first = datetime.strptime(ac["first_date"], "%Y-%m-%d").date()
                overlap  = (ac_first - ab_last).days
                if -30 <= overlap <= 60:
                    substitutions.append({
                        "stopped_exercise": ab["exercise_name"],
                        "started_exercise": ac["exercise_name"],
                        "category":         ab["category"],
                        "stopped_date":     ab["last_date"],
                        "started_date":     ac["first_date"],
                        "overlap_days":     overlap,
                    })
    return {
        "all_exercises":  lifecycle,
        "active":         active,
        "dormant":        [e for e in lifecycle if e["status"] == "dormant"],
        "abandoned":      abandoned,
        "substitutions":  substitutions,
    }


def _compute_muscle_group_balance(mg_summary: list) -> dict:
    vol_by_group = {mg["muscle_group"]: mg["total_volume"] for mg in mg_summary}
    push_vol = sum(vol_by_group.get(g, 0) for g in PUSH_CATEGORIES)
    pull_vol = sum(vol_by_group.get(g, 0) for g in PULL_CATEGORIES)
    total    = sum(vol_by_group.values())
    return {
        "push_volume":     round(push_vol, 1),
        "pull_volume":     round(pull_vol, 1),
        "push_pull_ratio": round(push_vol/pull_vol, 2) if pull_vol > 0 else None,
        "dominant_type":   ("push" if push_vol > pull_vol else
                            "pull" if pull_vol > push_vol else "balanced"),
        "total_volume":    round(total, 1),
        "distribution":    sorted([{"muscle_group": mg, "volume": vol,
                                    "pct_of_total": round(vol/total*100, 1) if total else 0}
                                   for mg, vol in vol_by_group.items()],
                                  key=lambda x: x["volume"], reverse=True),
    }


def _compute_training_consistency(all_dates: list, start_date: str, end_date: str) -> dict:
    period = [d for d in all_dates if start_date <= d <= end_date]
    if not period: return {}
    start_dt     = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt       = datetime.strptime(end_date,   "%Y-%m-%d").date()
    period_weeks = max((end_dt - start_dt).days / 7, 1)
    distinct     = len(set(period))
    weeks_with = set(_iso_week_key(d) for d in period)
    all_weeks  = set()
    cur        = start_dt
    while cur <= end_dt:
        iso = cur.isocalendar()
        all_weeks.add(f"{iso[0]:04d}-W{iso[1]:02d}")
        cur += timedelta(days=7)
    return {
        "distinct_training_days": distinct,
        "sessions_per_week":      round(distinct / period_weeks, 2),
        "weeks_with_sessions":    len(weeks_with),
        "weeks_missed":           max(0, len(all_weeks) - len(weeks_with)),
        "first_session":          period[0],
        "last_session":           period[-1],
    }


def _compute_training_density(daily_workouts: list) -> dict:
    if not daily_workouts: return {}
    ex_c  = [d["exercises_count"] for d in daily_workouts]
    set_c = [d["total_sets"]      for d in daily_workouts]
    vol_c = [d["total_volume"]    for d in daily_workouts]
    return {
        "avg_exercises_per_session": round(sum(ex_c)/len(ex_c),   1),
        "avg_sets_per_session":      round(sum(set_c)/len(set_c), 1),
        "avg_volume_per_session":    round(sum(vol_c)/len(vol_c), 1),
        "max_exercises_session":     max(ex_c),
        "min_exercises_session":     min(ex_c),
        "exercises_count_trend":     _trend(ex_c),
        "volume_trend":              _trend(vol_c),
    }


def _process_bodyweight(entries: list) -> dict:
    if not entries: return {"entries": [], "trend": "no_data", "current_kg": None}
    weights = [e["weight"] for e in entries]
    return {"entries": entries, "current_kg": round(weights[-1], 2),
            "trend": _trend(weights, 0.01, 0.01),
            "first_date": entries[0]["date"], "last_date": entries[-1]["date"]}


def _compute_bw_strength_correlation(sessions: list, bw_entries: list) -> dict:
    if not sessions or not bw_entries: return {}
    bw_by_date = {e["date"]: e["weight"] for e in bw_entries}
    bw_sorted  = sorted(bw_by_date.keys())
    ratios = []
    for s in sessions:
        if s["estimated_1rm"] <= 0: continue
        nearest = min(bw_sorted, key=lambda d: abs(
            (datetime.strptime(d, "%Y-%m-%d").date() -
             datetime.strptime(s["date"], "%Y-%m-%d").date()).days), default=None)
        if nearest is None: continue
        gap = abs((datetime.strptime(nearest, "%Y-%m-%d").date() -
                   datetime.strptime(s["date"], "%Y-%m-%d").date()).days)
        if gap <= 14:
            bw = bw_by_date[nearest]
            exercise_unit = s.get("unit", "lbs")
            bw_in_exercise_unit = bw * 2.2046 if exercise_unit == "lbs" else bw
            ratios.append({
                "date":                        s["date"],
                "e1rm":                        s["estimated_1rm"],
                "bodyweight_kg":               round(bw, 2),
                "bodyweight_in_exercise_unit": round(bw_in_exercise_unit, 2),
                "exercise_unit":               exercise_unit,
                "e1rm_to_bw":                  round(s["estimated_1rm"] / bw_in_exercise_unit, 3)
                                               if bw_in_exercise_unit > 0 else None,
            })
    if len(ratios) < 2: return {"ratios": ratios, "trend": "insufficient_data"}
    ratio_vals = [r["e1rm_to_bw"] for r in ratios if r["e1rm_to_bw"]]
    return {"ratios": ratios, "current_ratio": ratios[-1]["e1rm_to_bw"] if ratios else None,
            "trend": _trend(ratio_vals, 0.02, 0.02)}


def _process_goals(raw_goals: list, ctx: dict) -> list:
    return [{
        "exercise_name": g["exercise_name"],
        "target_weight": _recover_typed_weight(g["metric_weight"],
                                               _get_numeric_offset(ctx, g["exercise_name"])),
        "target_reps":   g["reps"],
        "target_date":   g["target_date"],
        "unit":          "kg" if _is_kg_native(ctx, g["exercise_name"]) else "lbs",
        "notes":         g["notes"],
    } for g in raw_goals]


def _compute_muscle_group_summary(exercise_results: list) -> list:
    by_g: dict = defaultdict(lambda: {
        "exercise_count": 0, "total_sets": 0, "total_volume": 0.0,
        "weekly_volumes": defaultdict(float),
        "strength_sets": 0, "hypertrophy_sets": 0, "endurance_sets": 0, "pain_sessions": 0,
    })
    for ex in exercise_results:
        g = ex["category"]; by_g[g]["exercise_count"] += 1
        for s in ex.get("sessions", []):
            by_g[g]["total_sets"]      += s["working_sets_count"]
            by_g[g]["total_volume"]    += s["total_volume"]
            by_g[g]["pain_sessions"]   += 1 if s["has_pain_flag"] else 0
            by_g[g]["strength_sets"]   += s["rep_ranges"]["strength_sets"]
            by_g[g]["hypertrophy_sets"]+= s["rep_ranges"]["hypertrophy_sets"]
            by_g[g]["endurance_sets"]  += s["rep_ranges"]["endurance_sets"]
            by_g[g]["weekly_volumes"][_iso_week_key(s["date"])] += s["total_volume"]
    summary = []
    for group, d in sorted(by_g.items()):
        weekly = sorted([{"week": w, "volume": round(v, 1)} for w, v in d["weekly_volumes"].items()], key=lambda x: x["week"])
        total  = d["strength_sets"] + d["hypertrophy_sets"] + d["endurance_sets"]
        summary.append({
            "muscle_group":   group,
            "exercise_count": d["exercise_count"],
            "total_sets":     d["total_sets"],
            "total_volume":   round(d["total_volume"], 1),
            "weekly_volumes": weekly,
            "trend":          _trend([w["volume"] for w in weekly]),
            "pain_sessions":  d["pain_sessions"],
            "rep_ranges": {
                "strength_pct":    round(d["strength_sets"]    / total * 100, 1) if total else 0,
                "hypertrophy_pct": round(d["hypertrophy_sets"] / total * 100, 1) if total else 0,
                "endurance_pct":   round(d["endurance_sets"]   / total * 100, 1) if total else 0,
            },
        })
    return summary


def _compute_rankings(exercise_results: list) -> dict:
    def safe(ex, *keys):
        obj = ex
        for k in keys:
            if not isinstance(obj, dict): return None
            obj = obj.get(k)
        return obj
    non_strength_names = {"Cardio"}
    def rank(key_fn, reverse=True):
        scored = [(ex["name"], key_fn(ex)) for ex in exercise_results
                  if key_fn(ex) is not None
                  and ex.get("category") not in non_strength_names]
        return [{"exercise": n, "value": v}
                for n, v in sorted(scored, key=lambda x: x[1], reverse=reverse)]
    def rank_weight_based(key_fn, reverse=True):
        """Rankings that only make sense for exercises with weight progression."""
        scored = [(ex["name"], key_fn(ex)) for ex in exercise_results
                  if key_fn(ex) is not None
                  and ex.get("category") not in non_strength_names
                  and (ex.get("pr") or {}).get("weight", 0) > 0]
        return [{"exercise": n, "value": v}
                for n, v in sorted(scored, key=lambda x: x[1], reverse=reverse)]
    return {
        "fastest_improving":    rank_weight_based(lambda ex: safe(ex, "progression", "weight_change_pct")),
        "most_stagnant":        rank_weight_based(lambda ex: ex.get("plateau_days")),
        "highest_volume":       rank_weight_based(lambda ex: sum(s["total_volume"] for s in ex.get("sessions", []))),
        "most_frequent":        rank(lambda ex: safe(ex, "training_frequency", "session_count")),
        "best_e1rm":            rank_weight_based(lambda ex: safe(ex, "pr", "estimated_1rm")),
        "most_pain_sessions":   rank(lambda ex: safe(ex, "pain_analysis", "pain_session_count")),
        "most_failed_attempts": rank(lambda ex: safe(ex, "pain_analysis", "failed_attempt_count")),
        "most_commented":       rank(lambda ex: sum(s["comment_count"] for s in ex.get("sessions", []))),
        "most_regressed":       rank(lambda ex: safe(ex, "progression", "regression_from_peak", "regression_pct")),
        "best_pr_velocity":     rank(lambda ex: safe(ex, "pr_velocity", "total_prs")),
    }


# ── Dynamic SQL fallback ───────────────────────────────────────────────────────

def sanitize_sql(sql: str) -> str:
    """
    Replace Unicode curly quotes with straight quotes before execution.
    LLM-generated SQL frequently contains curly quotes which cause OperationalError.
    """
    return (sql
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("‚", "'").replace("‛", "'"))


def query(sql: str) -> dict:
    """
    Fallback dynamic SQL query for questions the pre-built Data Agent functions
    do not cover.

    Use only when collect() data genuinely cannot answer the question.
    The pre-built path (collect) is always preferred — it returns clean,
    unit-converted, offset-applied, bar-weight-included values.

    This function returns RAW DB values. The caller is responsible for
    understanding the conversion rules documented in the WARNING below.

    Args:
        sql: A SELECT statement. Any other statement type is rejected.

    Returns:
        {
            "rows":    list of dicts (column -> raw value),
            "columns": list of column names,
            "row_count": int,
            "warning": str   <-- always present, always read this
        }

    WARNING — Raw values returned, no automatic conversions applied:
        metric_weight : stored as kg (FitNotes always divides typed value by 2.2046).
                        To recover typed value: metric_weight * 2.2046
        numeric_offset: NOT applied. Machine Wrist Extension and similar exercises
                        have an offset in user_context.json that this query does not add.
        bar_weight    : NOT included. Barbell and Smith Machine exercises log plates
                        only. Bar weight must be added separately for true load.
        unit label    : NOT determined. KG-native exercises (Deadlift, Seated Machine
                        Curl (Kg), Machine Wrist Extension, Hand Gripper) report in kg;
                        all others in lbs. The label is not attached to raw rows.
        is_personal_record: raw integer (0 or 1), not boolean.

    The Analysis Agent must apply these conversions or explicitly note in its
    answer that weights shown are raw logged values before conversion.
    """
    sql = sanitize_sql(sql.strip())

    # Reject any non-SELECT statement (WITH...SELECT CTEs are allowed)
    first_word = sql.split()[0].upper() if sql.split() else ""
    is_cte     = (
        first_word == "WITH"
        and "SELECT" in sql.upper()
        and not any(kw in sql.upper()
                    for kw in (" INSERT ", " UPDATE ", " DELETE ",
                                " DROP ",   " CREATE ", " ALTER "))
    )
    if first_word != "SELECT" and not is_cte:
        return {
            "rows":      [],
            "columns":   [],
            "row_count": 0,
            "warning":   (
                "REJECTED: only SELECT statements are permitted. "
                f"Received statement type: {first_word}"
            ),
        }

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows    = [dict(row) for row in cur.fetchall()]
        columns = [description[0] for description in cur.description] if cur.description else []

        # Auto-convert metric_weight to typed_weight so the Analysis Agent
        # never sees raw stored kg values directly.
        has_metric_weight = bool(rows) and "metric_weight" in rows[0]
        if has_metric_weight:
            for row in rows:
                if row.get("metric_weight") is not None:
                    row["typed_weight"] = round(row["metric_weight"] * 2.2046, 1)
                else:
                    row["typed_weight"] = None
            columns = columns + ["typed_weight"]

        return {
            "rows":      rows,
            "columns":   columns,
            "row_count": len(rows),
            "warning":   (
                "PARTIAL CONVERSION APPLIED: typed_weight = metric_weight * 2.2046 "
                "(recovers original typed value). Still missing: numeric_offset (e.g. "
                "Machine Wrist Extension +5), bar_weight (barbell/Smith Machine exercises "
                "log plates only), and unit label (kg-native exercises: Deadlift from "
                "2025-12-26, Seated Machine Curl (Kg), Machine Wrist Extension, Hand Gripper "
                "— all others lbs). Use typed_weight for display, not metric_weight."
            ),
        }
    except Exception as e:
        return {
            "rows":      [],
            "columns":   [],
            "row_count": 0,
            "warning":   f"QUERY ERROR: {str(e)}",
        }
    finally:
        conn.close()


# ── Main entry point ───────────────────────────────────────────────────────────

def collect(
    query_period_days: Optional[int]  = 90,
    end_date_str:      Optional[str]  = None,
    start_date_str:    Optional[str]  = None,
    muscle_groups:     Optional[list] = None,
    exercise_names:    Optional[list] = None,
    aggregation_level: Optional[str]  = None,
    include_phase2:    bool            = False,
) -> dict:
    """
    Complete workout data collection. Single source of truth at any time scale.

    Args:
        query_period_days : Days to look back. None = all-time.
        end_date_str      : End date YYYY-MM-DD. Defaults to today.
        start_date_str    : Direct start date override (takes precedence).
        muscle_groups     : Filter to specific muscle groups e.g. ["Back", "Biceps"].
        exercise_names    : Filter to specific exercises.
        aggregation_level : "session" | "weekly" | "monthly". Auto if None.
        include_phase2    : Fetch full comment history for triggered exercises.
    """
    ctx = _load_user_context()
    end_date = (datetime.strptime(end_date_str, "%Y-%m-%d").date()
                if end_date_str else date.today())
    end_str  = end_date.strftime("%Y-%m-%d")

    conn = _get_connection()
    try:
        all_training_dates = _fetch_all_training_dates(conn)

        if start_date_str:
            start_str = start_date_str
        elif query_period_days is None:
            start_str = all_training_dates[0] if all_training_dates else end_str
        else:
            start_str = (end_date - timedelta(days=query_period_days)).strftime("%Y-%m-%d")

        agg_level = aggregation_level or _get_aggregation_level(query_period_days)

        # ── Bulk fetches ───────────────────────────────────────────────────────
        all_period_rows   = _fetch_all_sets_in_period(conn, start_str, end_str)
        all_bw_entries    = _fetch_all_bodyweight(conn)
        period_bw_entries = [e for e in all_bw_entries if start_str <= e["date"] <= end_str]
        raw_goals         = _fetch_goals(conn)
        lifecycle_rows    = _fetch_exercise_lifecycle(conn)
        pr_history        = _fetch_pr_history(conn)

        # ── Filter rows ────────────────────────────────────────────────────────
        filtered_rows = all_period_rows
        if muscle_groups:
            cat_map     = {v: k for k, v in CATEGORY_NAMES.items()}
            allowed_ids = {cat_map[g] for g in muscle_groups if g in cat_map}
            filtered_rows = [r for r in filtered_rows if r["category_id"] in allowed_ids]
        if exercise_names:
            filtered_rows = [r for r in filtered_rows if r["exercise_name"] in exercise_names]

        # ── Daily workout view and supersets ──────────────────────────────────
        daily_workouts    = _build_daily_workouts(filtered_rows, ctx)
        superset_patterns = _detect_supersets(filtered_rows)

        # ── Per-exercise ───────────────────────────────────────────────────────
        by_exercise: dict = defaultdict(list)
        for row in filtered_rows: by_exercise[row["exercise_name"]].append(row)

        exercise_results = []
        alltime_all_rows = _fetch_all_sets_in_period(conn, "2000-01-01", end_str)
        alltime_cache: dict = defaultdict(list)
        for _r in alltime_all_rows:
            alltime_cache[_r["exercise_name"]].append(_r)

        for ex_name, ex_rows in sorted(by_exercise.items()):
            if not ex_rows: continue
            cat_id   = ex_rows[0]["category_id"]
            category = CATEGORY_NAMES.get(cat_id, f"Category_{cat_id}")
            is_cardio = category == "Cardio"
            cardio_note = (
                "Cardio exercise. weight=0 and reps=0 on all entries — these carry no "
                "meaningful information. Performance data is in the comment field AND/OR "
                "the distance (km) and duration_seconds fields — check both. "
                "Cycling: duration_seconds + comments (effort structure, intervals). "
                "Treadmill: distance + duration_seconds + comments (speed levels, incidents). "
                "Walking: no text comments but real distance and duration_seconds data. "
                "Pre-gym walk (~0.4 km, ~300 seconds) done before strength training — "
                "useful for correlating walk duration/distance with same-day performance."
            ) if is_cardio else None
            offset   = _get_numeric_offset(ctx, ex_name)
            bar_wt   = _get_bar_weight_lbs(ctx, ex_name, end_str)
            unit     = "kg" if _is_kg_native(ctx, ex_name, end_str) else "lbs"
            bar_wt_native = round(bar_wt / 2.2046, 2) if unit == "kg" else bar_wt

            sessions = _build_sessions_from_rows(ex_rows, ctx, ex_name)
            if not sessions: continue

            pr = _compute_pr(sessions, unit)
            is_weight_based_ex = (pr is not None and pr.get("weight", 0) > 0)

            if is_weight_based_ex:
                progression      = _compute_progression(sessions)
                phase2_triggered, plateau_days = _evaluate_phase2(progression, end_date)
            else:
                progression      = None
                plateau_days     = 0
                # Non-weight exercises: trigger Phase 2 if any sessions have comments
                # (since all meaningful data for these exercises is in comments)
                phase2_triggered = any(s["comment_count"] > 0 for s in sessions)

            duration_progression = (_compute_duration_progression(sessions)
                                    if not is_weight_based_ex else None)
            distance_progression = (_compute_distance_progression(sessions)
                                    if not is_weight_based_ex else None)

            full_comments = None
            if include_phase2 and phase2_triggered:
                full_comments = _fetch_full_comments_for_exercise(conn, ex_name)

            # All-time sessions for learning curve
            alltime_sessions = (_build_sessions_from_rows(alltime_cache[ex_name], ctx, ex_name)
                                 if alltime_cache.get(ex_name) else sessions)

            exercise_results.append({
                "name":          ex_name, "category": category,
                "unit":          unit, "numeric_offset": offset,
                "bar_weight":    bar_wt_native, "bar_weight_unit": unit,
                "is_cardio":     is_cardio, "cardio_note": cardio_note,

                # Zoom levels
                "sessions":              sessions,
                "weekly_aggregations":   _aggregate_weekly(sessions),
                "monthly_aggregations":  _aggregate_monthly(sessions),
                "yearly_aggregations":   _aggregate_yearly(sessions, unit),

                # Progression and PR
                "progression":           progression,
                "duration_progression":  duration_progression,
                "distance_progression":  distance_progression,
                "pr":                    pr,
                "pr_context":            _compute_pr_context(sessions, all_training_dates, all_bw_entries),
                "pr_velocity":           _compute_pr_velocity(pr_history, ex_name),

                # Trends
                "volume_trend":          _trend([s["total_volume"]  for s in sessions]),
                "e1rm_trend":            _trend([s["estimated_1rm"] for s in sessions
                                                  if s["estimated_1rm"] > 0], 0.05, 0.05),
                "form_trend":            _compute_form_trend(sessions),
                "comment_keyword_trends":_compute_comment_keyword_trends(sessions),

                # e1RM
                "e1rm_history":          [{"date": s["date"], "estimated_1rm": s["estimated_1rm"]}
                                           for s in sessions if s["estimated_1rm"] > 0],
                "e1rm_projection":       _compute_e1rm_projection(sessions),

                # Analysis
                "rep_range_distribution":  _compute_rep_range_distribution(sessions),
                "technique_variants":      _compute_technique_variants(sessions, unit),
                "pain_analysis":           _compute_pain_analysis(sessions),
                "training_frequency":      _compute_training_frequency(sessions, start_str, end_str),
                "rest_performance_buckets":_compute_rest_performance_buckets(sessions),
                "consecutive_day_effect":  _compute_consecutive_day_effect(sessions, all_training_dates),
                "workout_position_effect": _compute_exercise_workout_position(sessions, daily_workouts, ex_name),
                "inter_exercise_correlation": _compute_inter_exercise_correlation(sessions, ex_name, daily_workouts),
                "dow_e1rm_pattern":        _compute_exercise_dow_e1rm(sessions),
                "bw_strength_correlation": _compute_bw_strength_correlation(sessions, period_bw_entries),
                "learning_curve":          _compute_learning_curve(alltime_sessions),

                # Phase 2
                "plateau_days":    plateau_days,
                "phase2_triggered":phase2_triggered,
                "full_comments":   full_comments,
            })

        # ── Global ─────────────────────────────────────────────────────────────
        mg_summary = _compute_muscle_group_summary(exercise_results)
        rankings   = _compute_rankings(exercise_results)

        if agg_level != "session":
            for ex in exercise_results:
                ex["sessions"] = []

        goals        = _process_goals(raw_goals, ctx)
        goal_projs   = []
        for g in goals:
            ex   = next((e for e in exercise_results if e["name"] == g["exercise_name"]), None)
            sess = (ex.get("sessions") if ex and ex.get("sessions") else
                    [{"max_working_weight": m["max_working_weight"],
                      "estimated_1rm": m["peak_estimated_1rm"],
                      "date": m["month"] + "-15"}
                     for m in (ex["monthly_aggregations"] if ex else [])]) if ex else []
            goal_projs.append({**g, "projection": _compute_goal_projection(sess, g, g["unit"]) if sess else None})

        return {
            "query_period_days":        query_period_days,
            "query_start_date":         start_str,
            "query_end_date":           end_str,
            "aggregation_level":        agg_level,
            "total_exercises_analyzed": len(exercise_results),
            "all_time_summary":         _compute_alltime_summary(all_training_dates, conn, pr_history),
            "muscle_group_summary":     mg_summary,
            "muscle_group_balance":     _compute_muscle_group_balance(mg_summary),
            "training_consistency":     _compute_training_consistency(all_training_dates, start_str, end_str),
            "day_of_week_patterns":     _compute_day_of_week_patterns(all_training_dates, start_str, end_str),
            "seasonal_patterns":        _compute_seasonal_patterns(all_training_dates),
            "daily_workouts":           daily_workouts,
            "training_density":         _compute_training_density(daily_workouts),
            "superset_patterns":        superset_patterns,
            "exercise_lifecycle":       _compute_exercise_lifecycle(lifecycle_rows, end_str),
            "rankings":                 rankings,
            "bodyweight":               _process_bodyweight(period_bw_entries),
            "goals":                    goal_projs,
            "exercises":                exercise_results,
        }
    finally:
        conn.close()


# ── Verification ───────────────────────────────────────────────────────────────

def _print_summary(data: dict) -> None:
    print(f"\n  Period     : {data['query_start_date']} -> {data['query_end_date']}")
    print(f"  Aggregation: {data['aggregation_level']}")
    print(f"  Exercises  : {data['total_exercises_analyzed']}")
    cons = data["training_consistency"]
    if cons:
        print(f"  Consistency: {cons['distinct_training_days']} days  "
              f"{cons['sessions_per_week']}/week  {cons['weeks_missed']} weeks missed")
    bw = data["bodyweight"]
    if bw.get("current_kg"):
        print(f"  Bodyweight : {bw['current_kg']} kg  trend={bw['trend']}")
    ats = data.get("all_time_summary", {})
    if ats:
        print(f"  ALL-TIME   first={ats['first_training_date']}  "
              f"days={ats['total_training_days']}  sets={ats['total_sets']}  "
              f"streak={ats['longest_streak_days']}d  gap={ats['longest_gap_days']}d  "
              f"PRs={ats['total_prs_alltime']}")
    if data["goals"]:
        print(f"\n  GOALS")
        for g in data["goals"]:
            proj = g.get("projection") or {}
            print(f"    {g['exercise_name']}: {g['target_weight']} {g['unit']} "
                  f"x{g['target_reps']} by {g['target_date']}  "
                  f"on_track={proj.get('is_on_track')}  "
                  f"projected={proj.get('projected_achievement_date')}")
    print(f"\n  MUSCLE GROUP SUMMARY")
    print(f"  {'-'*68}")
    for mg in data["muscle_group_summary"]:
        rr = mg["rep_ranges"]
        print(f"  {mg['muscle_group']:12s}  ex={mg['exercise_count']:2d}  "
              f"sets={mg['total_sets']:4d}  vol={mg['total_volume']:10.0f}  "
              f"trend={mg['trend']:20s}  "
              f"S={rr['strength_pct']}% H={rr['hypertrophy_pct']}% E={rr['endurance_pct']}%")
    bal = data["muscle_group_balance"]
    if bal:
        print(f"\n  PUSH/PULL  push={bal['push_volume']:.0f}  "
              f"pull={bal['pull_volume']:.0f}  ratio={bal['push_pull_ratio']}  "
              f"dominant={bal['dominant_type']}")
    dow = data["day_of_week_patterns"]
    if dow:
        counts = {d["day"]: d["count"] for d in dow["distribution"]}
        print(f"  DAY OF WEEK  most={dow['most_common_day']}  skipped={dow['most_skipped_day']}")
        print("    " + "  ".join(f"{d[:3]}={counts.get(d,0)}" for d in
              ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
    ranks = data.get("rankings", {})
    if ranks.get("fastest_improving"):
        print(f"\n  FASTEST IMPROVING (top 5)")
        for r in ranks["fastest_improving"][:5]:
            print(f"    {r['exercise']:40s}  {r['value']:+.1f}%")
    if ranks.get("most_stagnant"):
        print(f"\n  MOST STAGNANT (top 5)")
        for r in [x for x in ranks["most_stagnant"] if x["value"] and x["value"] > 0][:5]:
            print(f"    {r['exercise']:40s}  {r['value']} days")
    lc = data.get("exercise_lifecycle", {})
    if lc.get("abandoned"):
        print(f"\n  ABANDONED EXERCISES ({len(lc['abandoned'])})")
        for e in lc["abandoned"][:5]:
            print(f"    {e['exercise_name']:40s}  last={e['last_date']}  ({e['days_since_last']}d ago)")
    if lc.get("substitutions"):
        print(f"\n  SUBSTITUTIONS DETECTED")
        for s in lc["substitutions"][:3]:
            print(f"    {s['stopped_exercise']} -> {s['started_exercise']}  ({s['category']})")


def _print_exercise(data: dict, exercise_name: str) -> None:
    ex = next((e for e in data["exercises"] if e["name"] == exercise_name), None)
    if ex is None:
        print(f"\n[NOT FOUND] '{exercise_name}'"); return
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  {ex['name']}  [{ex['category']}]  unit={ex['unit']}  "
          f"offset={ex['numeric_offset']}  bar={ex['bar_weight']} {ex['bar_weight_unit']}")
    print(sep)
    prog = ex["progression"]
    if prog:
        print(f"\n  PROGRESSION")
        print(f"    {prog['first_session_date']} -> {prog['last_session_date']}  "
              f"{prog['display_weight_start']} -> {prog['display_weight_end']}  "
              f"({prog['weight_change_pct']:+.1f}%)  e1RM: {prog['e1rm_start']} -> {prog['e1rm_end']}")
        if prog.get("plateau_since"):
            print(f"    Plateau since {prog['plateau_since']} ({ex['plateau_days']} days)")
        if prog.get("regression_from_peak"):
            r = prog["regression_from_peak"]
            print(f"    REGRESSION: peak {r['peak_weight']} on {r['peak_date']} -> "
                  f"now {r['current_weight']} ({r['regression_pct']:.1f}% drop)")
        if prog.get("diminishing_returns"):
            dr = prog["diminishing_returns"]
            print(f"    Returns: {dr['pattern']}  "
                  f"{dr['early_rate_per_month']:+.2f} -> {dr['recent_rate_per_month']:+.2f} /mo")
    if ex["pr"]:
        pr = ex["pr"]
        print(f"\n  PR  {pr['weight']} {ex['unit']} x {pr['reps']}  "
              f"({pr['date']})  e1RM={pr['estimated_1rm']}")
    freq = ex["training_frequency"]
    if freq:
        gap_str = f"{freq['avg_days_between']}d" if freq.get('avg_days_between') is not None else "N/A"
        print(f"\n  FREQUENCY  {freq['sessions_per_week']}/week  "
              f"gap={gap_str}  since_last={freq['days_since_last']}d")
    rr = ex["rep_range_distribution"]
    print(f"  REP RANGES  S={rr['strength_pct']}%  H={rr['hypertrophy_pct']}%  "
          f"E={rr['endurance_pct']}%  dominant={rr['dominant_range']}")
    proj = ex.get("e1rm_projection", {})
    if proj:
        print(f"  e1RM PROJ   30d={proj['projected_30d']}  60d={proj['projected_60d']}  "
              f"90d={proj['projected_90d']}  conf={proj['projection_confidence']}")
    pa = ex["pain_analysis"]
    print(f"  PAIN        {pa['pain_session_count']} sessions  "
          f"{pa['failed_attempt_count']} failed attempts")
    if ex.get("technique_variants"):
        print(f"  TECHNIQUES  " +
              "  ".join(f"{tv['variant']}(n={tv['session_count']},e1RM={tv['avg_e1rm']})"
                        for tv in ex["technique_variants"]))
    if ex.get("rest_performance_buckets"):
        print(f"  REST EFFECT " +
              "  ".join(f"{b['rest_range']}:{b['avg_e1rm']}"
                        for b in ex["rest_performance_buckets"]))
    if ex.get("inter_exercise_correlation"):
        print(f"  INTER-EX (top 3)")
        for c in ex["inter_exercise_correlation"][:3]:
            print(f"    {c['preceding_exercise']:35s}  "
                  f"diff={c['e1rm_diff_pct']:+.1f}%  {c['effect']}")
    lc = ex.get("learning_curve", {})
    if lc:
        print(f"  LEARNING    first={lc['first_ever_session']}  "
              f"sessions_to_PR={lc['sessions_to_first_pr']}  "
              f"total_alltime={lc['total_sessions_alltime']}")
    dur_prog = ex.get("duration_progression")
    if dur_prog and dur_prog.get("session_count", 0) >= 2:
        print(f"\n  DURATION PROGRESSION")
        print(f"    {dur_prog['first_session_date']} -> {dur_prog['last_session_date']}  "
              f"{dur_prog['duration_start_seconds']}s -> {dur_prog['duration_end_seconds']}s  "
              f"({dur_prog['duration_change_pct']:+.1f}%)  "
              f"peak={dur_prog['duration_peak_seconds']}s on {dur_prog['duration_peak_date']}")
    dist_prog = ex.get("distance_progression")
    if dist_prog:
        print(f"\n  DISTANCE PROGRESSION")
        print(f"    {dist_prog['first_session_date']} -> {dist_prog['last_session_date']}  "
              f"avg={dist_prog['avg_distance_km']}km/session  "
              f"peak={dist_prog['distance_peak_km']}km on {dist_prog['distance_peak_date']}  "
              f"total={dist_prog['total_distance_km']}km")
    agg = data["aggregation_level"]
    if agg == "session" and ex["sessions"]:
        print(f"\n  LAST 3 SESSIONS")
        for s in ex["sessions"][-3:]:
            dist_s = f"  total_dist={s['total_distance']}km" if s.get('total_distance', 0) > 0 else ""
            dur_s  = f"  total_dur={s['total_duration_seconds']}s" if s.get('total_duration_seconds', 0) > 0 else ""
            print(f"\n    {s['date']}  max={s['max_working_weight']} {s['unit']}  "
                  f"e1RM={s['estimated_1rm']}  vol={s['total_volume']}  "
                  f"sets={s['working_sets_count']}  form={s['form_quality']}"
                  f"{dist_s}{dur_s}")
            for i, st in enumerate(s["sets"]):
                tags = ("(W)" if st["is_warmup"] else "") + \
                       (f"[D{st['drop_group']}]" if st["drop_group"] else "") + \
                       ("[PAIN]" if st["is_pain_flag"] else "") + \
                       ("[FAIL]" if st["is_failed_attempt"] else "")
                cmt  = f"  [{st['comment']}]" if st["comment"] else ""
                dist_tag = f"  dist={st['distance']}km" if st.get('distance', 0) > 0 else ""
                dur_tag  = f"  dur={st['duration_seconds']}s" if st.get('duration_seconds', 0) > 0 else ""
                print(f"      Set {i+1}{tags}: "
                      f"{st['weight']} {s['unit']} x {st['reps']}  "
                      f"e1RM={st['estimated_1rm']}"
                      f"{dist_tag}{dur_tag}{cmt}")
    elif agg == "monthly" and ex["monthly_aggregations"]:
        print(f"\n  MONTHLY")
        for m in ex["monthly_aggregations"]:
            dist_str = f"  dist={m['total_distance']}km" if m.get('total_distance', 0) > 0 else ""
            dur_str  = f"  dur={m['total_duration_seconds']}s" if m.get('total_duration_seconds', 0) > 0 else ""
            print(f"    {m['month']}  max={m['max_working_weight']} {ex['unit']}  "
                  f"e1RM={m['peak_estimated_1rm']}  vol={m['total_volume']}  "
                  f"sessions={m['session_count']}{dist_str}{dur_str}")
    elif agg == "weekly" and ex["weekly_aggregations"]:
        print(f"\n  WEEKLY")
        for w in ex["weekly_aggregations"]:
            dist_str = f"  dist={w['total_distance']}km" if w.get('total_distance', 0) > 0 else ""
            dur_str  = f"  dur={w['total_duration_seconds']}s" if w.get('total_duration_seconds', 0) > 0 else ""
            print(f"    {w['week']}  max={w['max_working_weight']} {ex['unit']}  "
                  f"e1RM={w['peak_estimated_1rm']}  vol={w['total_volume']}  "
                  f"sessions={w['session_count']}{dist_str}{dur_str}")


if __name__ == "__main__":
    import sys

    def _looks_like_date(s: str) -> bool:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    args = sys.argv[1:]
    period   = args[0] if len(args) > 0 else "90"
    end_arg  = args[1] if len(args) > 1 and _looks_like_date(args[1]) else None
    offset   = 1 if end_arg else 0
    ex1      = args[1 + offset] if len(args) > 1 + offset and not _looks_like_date(args[1 + offset]) else "Lat Pulldown"
    ex2      = args[2 + offset] if len(args) > 2 + offset else "Machine Wrist Extension"
    period_val = None if period.lower() == "all" else int(period)
    print(f"\n{'='*72}")
    print(f"  Data Agent — Complete  |  period={period}")
    print(f"{'='*72}")
    data = collect(query_period_days=period_val, end_date_str=end_arg)
    _print_summary(data)
    _print_exercise(data, ex1)
    _print_exercise(data, ex2)
