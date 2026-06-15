"""
src/data_agent/validate.py
Post-condition validator — inspects a finished package and returns violations.

validate(package) -> list[Violation]

Runs ALL checks, returns the full list.  Does NOT raise and does NOT alter the
package.  The caller (facade) decides whether to log or raise.

Severity:
  "integrity"  — A, B, C, D, G class; wrong numbers must never reach the agent
  "soft"       — G4 size warning, D3 unclassified-comment log

Invariants implemented here (checkable from the package alone):
  A1  unit in {kg, lbs}
  A2  unit==kg only for known-kg-native exercises
  A3  no weight without a unit label
  B2a all-time PR carries a comment field when its PR session had a comment
  B3  pr.weight >= pr_period.weight >= every in-period session max_working_weight
  B4  pr is None only for non-weight exercises
  B5  max_working_weight matches an actual set weight (when sets are present)
  B6  no negative weight / reps / distance / duration
  C1  cardio distance_progression non-null when any session has distance > 0
  C2  cardio duration_progression non-null when any session has duration > 0
  C3  all_time_sessions non-null when the exercise has ≥1 session
  C4  cardio sessions non-empty when total_sessions_period > 0
  C6  pace field only on sessions that have distance > 0
  D1  comment_count == count of sets with a non-null comment (sets present only)
  D5  (integrity) each set's comment matches the Comment row bound to its own
      training_log._id (set_db_id) — verified against the DB; catches misattribution
  D2  no exercise session has more than one warmup set
  E1  every session date within [query_start_date, query_end_date]
  E2  no session dated in the future
  E3  weekly/monthly volume totals reconcile to member-session sums (≤ 0.5 diff)
  F1  correlational comparison blocks carry required statistical fields
  F2  confidence_label is one of the four allowed values
  F4  CI is None when n < 2; Pearson CI is None when n < 4
  G1  required keys present per exercise type (weight-based / cardio)
  G2  no None where a value is required given data exists
  G3  package serialises to JSON
  G4  (soft) flag filtered packages over ~400 KB
  G5  (soft) BROAD packages must not contain dropped deep-stat fields
  G6  scope consistency (integrity): focused→≤3 exercises;
      broad→no leaked deep-stat fields + exactly one agg level per exercise
"""

import json
import logging
import os
import sqlite3
from datetime import date
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

# Same DB the fetch layer reads — used ONLY by the D5 comment-binding check to
# verify each comment against its own Comment row (independent ground truth).
_DB_PATH = os.environ.get("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")

# Kg-native rule — single source of truth in src/units.py (was a local copy).
from src.units import KG_NATIVE_EXERCISES as _KG_NATIVE, DEADLIFT_KG_SWITCH as _DEADLIFT_KG_SWITCH

# ── Allowed confidence labels (spec F2) ───────────────────────────────────────
_VALID_CONFIDENCE = frozenset({"insufficient_data", "weak", "moderate", "strong"})

# ── G4 per-scope size ceilings (KB) ───────────────────────────────────────────
_G4_THRESHOLDS: dict = {"focused": 250, "group": 400, "broad": 500}

# ── G5 fields that must be absent from every non-cardio BROAD exercise ─────────
_BROAD_ABSENT_FIELDS: frozenset = frozenset({
    "full_comments",
    "inter_exercise_correlation",
    "dow_e1rm_pattern",
    "consecutive_day_effect",
    "rest_performance_buckets",
    "e1rm_history",
    "pr_context",
})

# ── Required fields for correlational comparison blocks (spec F1) ─────────────
_COMPARISON_REQUIRED = ("cohen_d", "cis_overlap", "confidence_label")
_STAT_BLOCK_REQUIRED  = ("n", "ci_95")


# ── Violation record ──────────────────────────────────────────────────────────

class Violation(NamedTuple):
    invariant_id: str
    severity: str    # "integrity" | "soft"
    message: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_kg(weight: float, unit: str) -> float:
    return weight / 2.2046 if unit == "lbs" else weight


def _is_trimmed_cardio(ex: dict) -> bool:
    """True when the exercise is a restructured cardio block from trim_package."""
    return bool(ex.get("is_cardio")) and "all_time_sessions" in ex


def _viol(inv_id: str, msg: str) -> Violation:
    return Violation(inv_id, "integrity", msg)


def _soft(inv_id: str, msg: str) -> Violation:
    return Violation(inv_id, "soft", msg)


# ── Per-exercise checks ───────────────────────────────────────────────────────

def _check_units(ex: dict, v: list) -> None:
    """A1, A2, A3."""
    name = ex.get("name", "?")
    unit = ex.get("unit") or ""
    end_str = ex.get("_end_str", "")  # injected by caller

    # A1 — cardio exercises measure km and seconds, not weight; skip the kg/lbs check.
    # The trimmed cardio rebuild drops the unit field entirely, so an absent unit
    # for a cardio exercise is correct behaviour, not a violation.
    if ex.get("is_cardio"):
        return

    # A1 — exercise-level unit
    if unit not in ("kg", "lbs"):
        v.append(_viol("A1", f"{name}: unit={unit!r} is not 'kg' or 'lbs'"))

    # A2 — kg only for known-kg-native
    if unit == "kg":
        if name not in _KG_NATIVE:
            v.append(_viol("A2",
                f"{name}: unit=kg but '{name}' is not a kg-native exercise"))
        elif name == "Deadlift" and end_str and end_str < _DEADLIFT_KG_SWITCH:
            v.append(_viol("A2",
                f"Deadlift: unit=kg but query_end_date {end_str} "
                f"is before the kg-switch date {_DEADLIFT_KG_SWITCH}"))

    # A1/A2/A3 — per-session
    for s in ex.get("sessions", []):
        su = s.get("unit") or ""
        sd = s.get("date", "?")
        if su not in ("kg", "lbs"):
            v.append(_viol("A1", f"{name} session {sd}: unit={su!r} is not 'kg' or 'lbs'"))
        if su == "kg" and name not in _KG_NATIVE:
            v.append(_viol("A2",
                f"{name} session {sd}: unit=kg but '{name}' is not a kg-native exercise"))
        if su == "kg" and name == "Deadlift" and sd and sd < _DEADLIFT_KG_SWITCH:
            v.append(_viol("A2",
                f"Deadlift session {sd}: unit=kg but session is before switch date"))
        # A3 — weight present but no unit
        if s.get("max_working_weight") is not None and not su:
            v.append(_viol("A3",
                f"{name} session {sd}: max_working_weight present but session has no unit"))

    # A3 — PR objects carry units
    for label, pr_obj in [("pr", ex.get("pr")), ("pr_period", ex.get("pr_period"))]:
        if pr_obj and pr_obj.get("weight") is not None and not pr_obj.get("unit"):
            v.append(_viol("A3", f"{name}: {label}.weight present but {label} has no unit"))


def _check_pr_and_progression(ex: dict, v: list, end_str: str) -> None:
    """B2a, B3, B4, B5, B6."""
    name = ex.get("name", "?")
    is_cardio = ex.get("is_cardio", False)
    sessions = ex.get("sessions", [])
    pr  = ex.get("pr")
    prp = ex.get("pr_period")

    # B2a — all-time PR carries comment when its session had a comment
    if pr and sessions:
        pr_date = pr.get("date")
        pr_sess = next((s for s in sessions if s.get("date") == pr_date), None)
        if pr_sess and pr_sess.get("comment_count", 0) > 0 and "comment" not in pr:
            v.append(_viol("B2a",
                f"{name}: all-time PR (date={pr_date}) session had "
                f"{pr_sess['comment_count']} comment(s) but pr has no 'comment' field"))

    # B3 — pr.weight >= pr_period.weight >= every session max
    if pr and prp and sessions:
        pr_kg  = _to_kg(pr.get("weight") or 0,  pr.get("unit")  or "lbs")
        prp_kg = _to_kg(prp.get("weight") or 0, prp.get("unit") or "lbs")
        if pr_kg < prp_kg - 0.01:
            v.append(_viol("B3",
                f"{name}: pr.weight ({pr.get('weight')} {pr.get('unit')}) < "
                f"pr_period.weight ({prp.get('weight')} {prp.get('unit')})"))
        for s in sessions:
            smw = s.get("max_working_weight") or 0
            su  = s.get("unit") or "lbs"
            if _to_kg(smw, su) > prp_kg + 0.01:
                v.append(_viol("B3",
                    f"{name} session {s.get('date')}: max_working_weight "
                    f"{smw} {su} exceeds pr_period.weight "
                    f"{prp.get('weight')} {prp.get('unit')}"))

    # B4 — pr not None for weight-based exercises with real weights
    if not is_cardio and sessions and pr is None:
        max_mww = max((s.get("max_working_weight") or 0 for s in sessions), default=0)
        if max_mww > 0:
            v.append(_viol("B4",
                f"{name}: weight-based exercise (max={max_mww}) but pr is None"))

    # B5 — max_working_weight equals an actual set's headline weight (when sets present)
    # headline_weight = plates + bar (bar-inclusive); falls back to weight for no-bar sets.
    for s in sessions:
        raw_sets = s.get("sets")
        if not raw_sets:
            continue  # stripped in prepare_analysis_package — can't check
        mww = s.get("max_working_weight") or 0
        working = [st.get("headline_weight", st["weight"])
                   for st in raw_sets if not st.get("is_warmup", False)]
        all_w   = working or [st.get("headline_weight", st["weight"]) for st in raw_sets]
        if all_w and not any(abs(w - mww) < 0.01 for w in all_w):
            v.append(_viol("B5",
                f"{name} session {s.get('date')}: max_working_weight={mww} "
                f"not in set headline weights {sorted(set(all_w))}"))

    # B6 — no negative values
    for s in sessions:
        for field, val in (
            ("max_working_weight", s.get("max_working_weight")),
            ("total_volume",       s.get("total_volume")),
            ("total_distance",     s.get("total_distance")),
            ("total_duration_seconds", s.get("total_duration_seconds")),
        ):
            if isinstance(val, (int, float)) and val < 0:
                v.append(_viol("B6",
                    f"{name} session {s.get('date')}: {field}={val} is negative"))
        for st in s.get("sets", []):
            for sf, sv in (
                ("weight",           st.get("weight")),
                ("reps",             st.get("reps")),
                ("distance",         st.get("distance")),
                ("duration_seconds", st.get("duration_seconds")),
            ):
                if isinstance(sv, (int, float)) and sv < 0:
                    v.append(_viol("B6",
                        f"{name} session {s.get('date')} set "
                        f"{st.get('set_id', '?')}: {sf}={sv} is negative"))


def _check_cardio_raw(ex: dict, v: list) -> None:
    """C1, C2 for the raw collect() exercise format."""
    name = ex.get("name", "?")
    sessions = ex.get("sessions", [])
    if not sessions:
        return

    has_distance = any((s.get("total_distance") or 0) > 0 for s in sessions)
    has_duration = any((s.get("total_duration_seconds") or 0) > 0 for s in sessions)

    if has_distance and ex.get("distance_progression") is None:
        v.append(_viol("C1",
            f"{name}: sessions with total_distance>0 exist but distance_progression is None"))
    if has_duration and ex.get("duration_progression") is None:
        v.append(_viol("C2",
            f"{name}: sessions with total_duration_seconds>0 exist "
            "but duration_progression is None"))


def _check_cardio_trimmed(ex: dict, v: list) -> None:
    """C1, C2, C3, C4, C6 for the trimmed prepare_analysis_package() cardio format."""
    name = ex.get("name", "?")
    sessions  = ex.get("sessions", [])
    prog      = ex.get("progression") or {}
    total_p   = ex.get("total_sessions_period") or 0
    alltime_s = ex.get("all_time_sessions")

    # C1 — distance progression non-null when any session has distance > 0
    has_dist = any((s.get("distance_km") or 0) > 0 for s in sessions)
    if has_dist and prog.get("distance_total_km") is None:
        v.append(_viol("C1",
            f"{name}: sessions with distance_km>0 "
            "but progression.distance_total_km is None"))

    # C2 — duration progression non-null when any session has duration > 0
    has_dur = any((s.get("duration_seconds") or 0) > 0 for s in sessions)
    if has_dur and prog.get("duration_start_seconds") is None:
        v.append(_viol("C2",
            f"{name}: sessions with duration_seconds>0 "
            "but progression.duration_start_seconds is None"))

    # C3 — all_time_sessions non-null whenever the exercise has ≥1 session
    if total_p > 0 and alltime_s is None:
        v.append(_viol("C3",
            f"{name}: total_sessions_period={total_p} but all_time_sessions is None "
            "(likely the total_alltime_sessions / total_sessions_alltime key typo)"))

    # C4 — sessions list non-empty when total_sessions_period > 0
    if total_p > 0 and len(sessions) == 0:
        v.append(_viol("C4",
            f"{name}: total_sessions_period={total_p} but sessions list is []"))

    # C6 — pace field only on sessions with distance > 0
    for s in sessions:
        if "pace" in s and not (s.get("distance_km") or 0) > 0:
            v.append(_viol("C6",
                f"{name} session {s.get('date')}: has 'pace' field "
                "but distance_km is 0 or absent"))


def _check_comments(ex: dict, v: list) -> None:
    """D1, D2."""
    name = ex.get("name", "?")
    for s in ex.get("sessions", []):
        raw_sets = s.get("sets")
        if not raw_sets:
            continue
        # D1 — comment_count matches sets with non-null comment
        stated = s.get("comment_count", 0)
        actual = sum(1 for st in raw_sets if st.get("comment") is not None)
        if stated != actual:
            v.append(_viol("D1",
                f"{name} session {s.get('date')}: comment_count={stated} "
                f"but {actual} set(s) have a non-null comment"))
        # D2 — at most one warmup set per session
        warmup_count = sum(1 for st in raw_sets if st.get("is_warmup", False))
        if warmup_count > 1:
            v.append(_viol("D2",
                f"{name} session {s.get('date')}: {warmup_count} sets have "
                "is_warmup=True (at most 1 allowed per session)"))


def _check_c6_raw(ex: dict, v: list) -> None:
    """C6 for raw sessions (pace field must not appear on zero-distance sessions)."""
    name = ex.get("name", "?")
    for s in ex.get("sessions", []):
        if "pace" in s and not (s.get("total_distance") or 0) > 0:
            v.append(_viol("C6",
                f"{name} session {s.get('date')}: has 'pace' field "
                "but total_distance is 0"))


def _check_temporal(ex: dict, v: list, start_str: str, end_str: str,
                    today_str: str) -> None:
    """E1, E2."""
    name = ex.get("name", "?")
    for s in ex.get("sessions", []):
        sd = s.get("date") or ""
        if not sd:
            continue
        if start_str and sd < start_str:
            v.append(_viol("E1",
                f"{name} session {sd}: date is before query_start_date {start_str}"))
        if end_str and sd > end_str:
            v.append(_viol("E1",
                f"{name} session {sd}: date is after query_end_date {end_str}"))
        if sd > today_str:
            v.append(_viol("E2",
                f"{name} session {sd}: date is in the future (today={today_str})"))


def _check_aggregation_consistency(ex: dict, v: list) -> None:
    """
    E3 — weekly/monthly per-unit volume buckets reconcile to member-session
    sums of the same typed-unit frame (lbs sessions sum to total_volume_lbs,
    kg sessions to total_volume_kg — frames are never added together).
    """
    name = ex.get("name", "?")
    sessions = ex.get("sessions", [])
    if not sessions:
        return  # sessions stripped — can't check

    # date -> (unit, volume); sessions are single-date, single-unit
    by_date = {s["date"]: (s.get("unit") or "lbs", s.get("total_volume") or 0)
               for s in sessions}

    def _expected(dates: list, unit: str) -> float:
        return round(sum(vol for d in dates
                         for u, vol in [by_date.get(d, ("lbs", 0))]
                         if u == unit), 1)

    for label, key, aggs in (
        ("week",  "week",  ex.get("weekly_aggregations", [])),
        ("month", "month", ex.get("monthly_aggregations", [])),
    ):
        for a in aggs:
            for unit, vol_key in (("lbs", "total_volume_lbs"),
                                  ("kg",  "total_volume_kg")):
                expected = _expected(a.get("session_dates", []), unit)
                actual   = round(a.get(vol_key) or 0, 1)
                if abs(expected - actual) > 0.5:
                    v.append(_viol("E3",
                        f"{name} {label} {a.get(key)}: {vol_key}={actual} "
                        f"but sum of member-session volumes ({unit})={expected} "
                        f"(diff={abs(expected-actual):.1f})"))


def _check_stat_block(name: str, context: str, block: dict, v: list) -> None:
    """F1 (required fields in a per-condition stat block), F4 (CI vs n)."""
    for field in _STAT_BLOCK_REQUIRED:
        if field not in block:
            v.append(_viol("F1",
                f"{name} {context}: missing required field '{field}'"))
    n     = block.get("n")
    ci    = block.get("ci_95")
    if isinstance(n, int):
        if n < 2 and ci is not None:
            v.append(_viol("F4",
                f"{name} {context}: n={n} (<2) but ci_95 is not None"))
        if n >= 2 and ci is None:
            v.append(_viol("F4",
                f"{name} {context}: n={n} (≥2) but ci_95 is None"))


def _check_comparison_block(name: str, context: str, block: dict, v: list) -> None:
    """F1 (required fields in a comparison block), F2 (valid confidence label)."""
    for field in _COMPARISON_REQUIRED:
        if field not in block:
            v.append(_viol("F1",
                f"{name} {context}: comparison missing required field '{field}'"))
    label = block.get("confidence_label")
    if label is not None and label not in _VALID_CONFIDENCE:
        v.append(_viol("F2",
            f"{name} {context}: confidence_label={label!r} "
            f"not in {sorted(_VALID_CONFIDENCE)}"))


def _check_inter_ex_entry(name: str, entry: dict, v: list) -> None:
    """F1, F2, F4 for a single inter_exercise_correlation entry."""
    ctx = f"inter_exercise_correlation[{entry.get('preceding_exercise','?')}]"
    for field in ("n_preceded", "n_not_preceded", "cohen_d", "cis_overlap", "confidence_label"):
        if field not in entry:
            v.append(_viol("F1", f"{name} {ctx}: missing required field '{field}'"))
    label = entry.get("confidence_label")
    if label is not None and label not in _VALID_CONFIDENCE:
        v.append(_viol("F2",
            f"{name} {ctx}: confidence_label={label!r} "
            f"not in {sorted(_VALID_CONFIDENCE)}"))
    # F4 per-condition CI checks (if present)
    for ci_key, n_key in (("ci_95_when_preceded", "n_preceded"),
                           ("ci_95_when_not", "n_not_preceded")):
        n  = entry.get(n_key)
        ci = entry.get(ci_key)
        if isinstance(n, int):
            if n < 2 and ci is not None:
                v.append(_viol("F4",
                    f"{name} {ctx}: {n_key}={n} (<2) but {ci_key} is not None"))
            if n >= 2 and ci is None:
                v.append(_viol("F4",
                    f"{name} {ctx}: {n_key}={n} (≥2) but {ci_key} is None"))


def _check_bw_correlation(name: str, bwc: dict, v: list) -> None:
    """F1, F2, F4 for bw_strength_correlation."""
    if not bwc:
        return
    # F1 — required top-level keys
    for field in ("n", "confidence_label"):
        if field not in bwc:
            v.append(_viol("F1",
                f"{name} bw_strength_correlation: missing required field '{field}'"))
    # F2
    label = bwc.get("confidence_label")
    if label is not None and label not in _VALID_CONFIDENCE:
        v.append(_viol("F2",
            f"{name} bw_strength_correlation: confidence_label={label!r} "
            f"not in {sorted(_VALID_CONFIDENCE)}"))
    # F1 — pearson block
    pearson = bwc.get("pearson") or {}
    if pearson:
        for field in ("r", "ci_95", "n"):
            if field not in pearson:
                v.append(_viol("F1",
                    f"{name} bw_strength_correlation.pearson: "
                    f"missing required field '{field}'"))
        # F4 — Pearson CI None when n < 4
        pn  = pearson.get("n")
        pci = pearson.get("ci_95")
        if isinstance(pn, int):
            if pn < 4 and pci is not None:
                v.append(_viol("F4",
                    f"{name} bw_strength_correlation.pearson: "
                    f"n={pn} (<4) but ci_95 is not None"))
            if pn >= 4 and pci is None:
                v.append(_viol("F4",
                    f"{name} bw_strength_correlation.pearson: "
                    f"n={pn} (≥4) but ci_95 is None"))


def _check_statistical(ex: dict, v: list) -> None:
    """F1, F2, F4 across all correlational outputs for one exercise."""
    name = ex.get("name", "?")

    for key in ("rest_performance_buckets", "consecutive_day_effect"):
        block = ex.get(key) or {}
        for row in block.get("buckets", []) + block.get("by_consecutive_days", []):
            _check_stat_block(name, key, row, v)
        comp = block.get("comparison")
        if comp:
            _check_comparison_block(name, f"{key}.comparison", comp, v)

    for entry in ex.get("inter_exercise_correlation", []):
        _check_inter_ex_entry(name, entry, v)

    dep = ex.get("dow_e1rm_pattern") or {}
    for row in dep.get("by_day", []):
        _check_stat_block(name, "dow_e1rm_pattern", row, v)
    comp = dep.get("comparison")
    if comp:
        _check_comparison_block(name, "dow_e1rm_pattern.comparison", comp, v)

    _check_bw_correlation(name, ex.get("bw_strength_correlation") or {}, v)


def _check_structural(ex: dict, v: list) -> None:
    """G1, G2."""
    name = ex.get("name", "?")
    is_cardio = ex.get("is_cardio", False)
    sessions  = ex.get("sessions", [])

    # G1 — required keys per type
    if is_cardio:
        if "progression" not in ex and "distance_progression" not in ex:
            v.append(_viol("G1",
                f"{name}: cardio exercise has neither 'progression' "
                "nor 'distance_progression' block"))
    else:
        if sessions and "progression" not in ex:
            v.append(_viol("G1",
                f"{name}: weight-based exercise with sessions "
                "but missing 'progression' key"))

    # G2 — no required None
    if not is_cardio and sessions:
        max_mww = max((s.get("max_working_weight") or 0 for s in sessions), default=0)
        if max_mww > 0 and ex.get("pr") is None:
            v.append(_viol("G2",
                f"{name}: sessions have max_working_weight={max_mww} "
                "but 'pr' (all-time) is None"))
        prog = ex.get("progression")
        if prog is not None:
            for key in ("first_session_date", "last_session_date",
                        "max_weight_start", "max_weight_end"):
                if prog.get(key) is None:
                    v.append(_viol("G2",
                        f"{name}: progression.{key} is None despite data existing"))


# ── Package-level checks ──────────────────────────────────────────────────────

def _ro_connection() -> Optional[sqlite3.Connection]:
    """Best-effort read-only connection for ground-truth checks; None on failure."""
    try:
        norm = _DB_PATH.replace("\\", "/")
        conn = sqlite3.connect(f"file:{norm}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _check_comment_binding(package: dict, v: list) -> None:
    """
    D5 (integrity) — every set carrying a non-null comment must carry the
    Comment row bound to its OWN training_log._id (set_db_id), verified
    independently against the Comment table (Comment.owner_id), not against
    the package's own derivation. This is exactly the misattribution class —
    a comment shown against the wrong set — so it raises, never ships.

    Rules:
      • comment is None        → correct, skipped (≈46% of sets, expected).
      • comment + no set_db_id → not cross-checkable here (no binding); skipped.
      • DB unavailable         → check skipped (pure synthetic fixtures).
    """
    bound: list = []
    for ex in package.get("exercises", []):
        name = ex.get("name", "?")
        for s in ex.get("sessions", []) or []:
            for st in s.get("sets", []) or []:
                if st.get("comment") is None:
                    continue
                sid = st.get("set_db_id")
                if sid is None:
                    continue
                bound.append((name, sid, st["comment"]))
    if not bound:
        return   # no commented, id-bound sets (e.g. trimmed package) → no DB hit

    conn = _ro_connection()
    if conn is None:
        return
    try:
        # 1:1 in current data; if a future schema allows several, the
        # deterministic first by Comment._id wins (matches the fetch contract).
        truth: dict = {}
        for row in conn.execute("SELECT owner_id, comment FROM Comment ORDER BY _id ASC"):
            oid = row["owner_id"]
            if oid not in truth:
                truth[oid] = row["comment"]
    except Exception:
        return
    finally:
        conn.close()

    for name, sid, comment in bound:
        if truth.get(sid) != comment:
            v.append(_viol("D5",
                f"{name} set training_log._id={sid}: carried comment does not "
                f"match the Comment row bound to it (owner_id={sid}) — "
                f"comment misattributed to the wrong set"))


def _check_g3(package: dict, v: list) -> None:
    """G3 — package must be JSON-serialisable."""
    try:
        json.dumps(package)
    except (TypeError, ValueError) as exc:
        v.append(_viol("G3", f"Package is not JSON-serialisable: {exc}"))


def _check_g4(package: dict, v: list) -> None:
    """G4 (soft) — per-scope size ceiling: BROAD 500 KB, GROUP 400 KB, FOCUSED 250 KB."""
    try:
        has_scope = "scope" in package
        scope     = package.get("scope", "focused")
        ceiling   = _G4_THRESHOLDS.get(scope, 250)
        size_kb   = len(json.dumps(package, default=str).encode()) / 1024
        if size_kb > ceiling:
            n_ex = package.get("total_exercises_analyzed", "?")
            if has_scope:
                v.append(_soft("G4",
                    f"Package size is {size_kb:.0f} KB "
                    f"(>{ceiling} KB ceiling for scope={scope!r}); "
                    f"exercises={n_ex}. Consider narrowing the query scope."))
            else:
                # No scope key yet — this is the pre-trim intermediate package
                # inside collect(). Naming a scope it doesn't have made this
                # line read as if an untrimmed package shipped to the LLM.
                v.append(_soft("G4",
                    f"pre-trim package is {size_kb:.0f} KB "
                    f"({n_ex} exercises) — informational, "
                    f"trimming runs next"))
    except Exception:
        pass


def _check_g5(package: dict, v: list) -> None:
    """G5 (soft) — BROAD packages must not contain dropped deep-stat fields."""
    if package.get("scope") != "broad":
        return
    for ex in package.get("exercises", []):
        if ex.get("is_cardio"):
            continue
        name = ex.get("name", "?")
        for field in _BROAD_ABSENT_FIELDS:
            if field in ex:
                v.append(_soft("G5",
                    f"{name}: BROAD package has '{field}' present "
                    "(trim_package should have removed it)"))


def _check_g6(package: dict, v: list) -> None:
    """
    G6 (integrity) — scope consistency.

    focused : exercises in package must be <= 3.
              A focused-labeled 60-exercise package is a scope-derivation bug;
              it must never reach the Analysis Agent.

    broad   : (a) no non-cardio exercise may carry full_comments or any
                  dropped deep-stat key (hard version of the soft G5 check);
              (b) each non-cardio exercise must have exactly one of the three
                  aggregation-level arrays present (trim_package enforces this).
    """
    scope = package.get("scope")
    exercises = [ex for ex in package.get("exercises", []) if not ex.get("is_cardio")]

    if scope == "focused":
        n = len(package.get("exercises", []))
        if n > 3:
            v.append(_viol("G6",
                f"scope='focused' but package contains {n} exercises "
                f"(must be <= 3); scope was derived from classifier intent, "
                f"not effective filter results"))
        return

    if scope == "broad":
        # (a) Leaked deep-stat fields — integrity version of G5
        for ex in exercises:
            name = ex.get("name", "?")
            for field in _BROAD_ABSENT_FIELDS:
                if field in ex:
                    v.append(_viol("G6",
                        f"{name}: scope='broad' package has '{field}' present "
                        f"(must be absent after trim)"))

        # (b) Exactly one aggregation level per exercise
        _AGG_KEYS = ("weekly_aggregations", "monthly_aggregations", "yearly_aggregations")
        for ex in exercises:
            present = [k for k in _AGG_KEYS if k in ex]
            if len(present) != 1:
                name = ex.get("name", "?")
                v.append(_viol("G6",
                    f"{name}: scope='broad' package has {len(present)} aggregation "
                    f"level(s) present ({present}); expected exactly 1"))


# ── Public entry point ────────────────────────────────────────────────────────

def validate(package: dict) -> list:
    """
    Check all post-condition invariants on a finished package.

    Returns a list of Violation named-tuples.  Never raises.
    Empty list means every checked invariant passed.
    """
    violations: list = []

    start_str  = package.get("query_start_date") or ""
    end_str    = package.get("query_end_date")   or ""
    today_str  = date.today().strftime("%Y-%m-%d")

    for ex in package.get("exercises", []):
        # Thread end_str for A2 Deadlift date-check (side-channel — not stored in ex)
        # Make a shallow proxy so _check_units can read it without mutating ex.
        ex_view = dict(ex)
        ex_view["_end_str"] = end_str

        _check_units(ex_view, violations)
        _check_pr_and_progression(ex, violations, end_str)

        is_cardio = ex.get("is_cardio", False)
        if is_cardio and _is_trimmed_cardio(ex):
            _check_cardio_trimmed(ex, violations)
        elif is_cardio:
            _check_cardio_raw(ex, violations)

        _check_c6_raw(ex, violations)          # C6 for raw sessions
        _check_comments(ex, violations)        # D1
        _check_temporal(ex, violations, start_str, end_str, today_str)  # E1, E2
        _check_aggregation_consistency(ex, violations)  # E3
        _check_statistical(ex, violations)     # F1, F2, F4
        _check_structural(ex, violations)      # G1, G2

    _check_comment_binding(package, violations)  # D5 (DB-verified comment binding)
    _check_g3(package, violations)             # G3
    _check_g4(package, violations)             # G4
    _check_g5(package, violations)             # G5
    _check_g6(package, violations)             # G6

    return violations
