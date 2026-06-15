"""
src/data_agent/process.py
Pure processing stage — no sqlite, no open(), no os.environ, no date.today().

Public surface:
  process_data(bundle, ctx, start_str, end_str, today,
               query_period_days, muscle_groups, exercise_names,
               agg_level, include_phase2)  -> package dict
  trim_package(package)                   -> package dict (in-place trim)
  _get_aggregation_level(period_days)     -> str   (used by the facade)
"""

import re
import json
import math
import logging
from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import Optional

# Kg-native switch date — single source of truth in src/units.py.
# (_is_kg_native stays config-driven, reading user_context.exercises_in_kg;
# only the switch DATE was a duplicated constant.)
from src.units import DEADLIFT_KG_SWITCH_DATE

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORY_NAMES = {
    1: "Shoulders", 2: "Triceps",  3: "Biceps",
    4: "Chest",     5: "Back",     6: "Legs",
    7: "Abs",       8: "Cardio",   9: "Forearms",
}
PUSH_CATEGORIES = {"Shoulders", "Triceps", "Chest"}
PULL_CATEGORIES = {"Back", "Biceps"}

WARMUP_GAP_RATIO  = 1.5   # gap(s1→s2) must exceed this × max(working_steps) to flag
WARMUP_FLAT_RATIO = 0.60  # equal-working-set case: s1 ≤ this fraction of s2
WARMUP_MIN_REPS   = 12    # minimum reps on the first set for weight-based warmup
HEAVY_FRACTION    = 0.50  # 0-weight opener: working max must be ≥ this × alltime max
PLATEAU_TRIGGER_DAYS    = 28
IMPROVEMENT_TRIGGER_PCT = 20.0

# ── Plateau / regression / "current ability" detection ────────────────────────
# Every threshold is structural (a count or tolerance applied to the exercise's
# OWN sessions), never a per-user or per-exercise hardcode.
NEW_BEST_WEIGHT_TOL_KG          = 0.05  # kg tolerance for "same weight" across unit-switch rounding
CURRENT_ABILITY_SESSIONS        = 3     # "current" = best (weight→reps) over the exercise's last N sessions; one back-off/high-rep day cannot redefine it
PLATEAU_MIN_SESSIONS_SINCE_BEST = 5     # need ≥ this many of THIS exercise's sessions with no new best before a plateau — normal gaps between PRs during a rising run must not count
PLATEAU_SLOPE_WINDOW            = 6     # sessions used for the e1RM trend-direction gate; a rising recent e1RM slope blocks a plateau regardless of the count
MIN_SESSIONS_FOR_TREND          = 4     # below this, refuse to opine on plateau/regression (thin data → report sample size only)
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

# ── Tier 1 pre-pass: unit-token detection and warmup gate ─────────────────────
_STANDALONE_UNIT_RE     = re.compile(r'^\s*(?:kg|kgs|pounds|lbs)\s*$', re.IGNORECASE)
_UNIT_KG_TOKENS         = frozenset({"kg", "kgs"})
_UNIT_LBS_TOKENS        = frozenset({"pounds", "lbs"})

# ── Tier 2 pre-pass: Smith counterbalance detection ───────────────────────────
# Noun-form only: word boundary after "supports?" prevents "supported" from matching.
_CB_ONE_RE       = re.compile(r'\b(one|single|1)\s+supports?\b',  re.IGNORECASE)
_CB_TWO_RE       = re.compile(r'\b(two|2|double)\s+supports?\b',  re.IGNORECASE)
_CB_NO_RE        = re.compile(r'\bno\s+supports?\b',               re.IGNORECASE)
_SUPPORT_NOUN_RE = re.compile(r'\bsupports?\b',                    re.IGNORECASE)

SMITH_BAR_KG           = 20.0
COUNTERBALANCE_KG_EACH = 10.0

# ROM terms for the warmup negative gate — ≥2 distinct hits = genuine form hierarchy
_ROM_GATE_TERMS = [
    "below the neck", "neck up", "neck ups", "above the neck",
    "partials", "partial reps", "touching chest",
    "not fully", "not touching", "full range", "full rom",
    "below parallel",
]
# Pain attribution vocabulary for the warmup gate
_WARMUP_GATE_PAIN_VOCAB = ["pain", "hurt", "ache", "elbow"]

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


def _to_kg(weight: float, unit: str) -> float:
    """
    Normalize a display weight to kg for cross-unit comparison.
    Used when comparing weights that may span a unit switch (e.g. Deadlift
    logged in lbs before 2025-12-26, kg after).
    """
    return weight / 2.2046 if unit == "lbs" else weight


def _detect_unit_comment_token(comment: Optional[str]) -> Optional[str]:
    """
    Returns 'kg' or 'lbs' if the comment is *essentially* a standalone unit
    token (whole comment, not inside a numeric phrase like '50 pounds').
    Returns None if no clean match.
    """
    if not comment:
        return None
    if not _STANDALONE_UNIT_RE.match(comment):
        return None
    tok = comment.strip().lower()
    if tok in _UNIT_KG_TOKENS:
        return "kg"
    if tok in _UNIT_LBS_TOKENS:
        return "lbs"
    return None


def _is_smith_machine(ctx: dict, exercise_name: str) -> bool:
    """True if this exercise uses the Smith machine bar (20 kg)."""
    smith = ctx.get("bar_weights_not_included", {}).get("smith_machine", {})
    return ("Smith Machine" in exercise_name
            or exercise_name in smith.get("exercises", []))


def _detect_counterbalance(comment: Optional[str]) -> Optional[str]:
    """
    Tier 2 — classify noun-form 'support(s)' tokens in a comment.

    Returns 'one_support', 'two_supports', 'no_support',
    'unclassified_support', or None (no support token found).

    The word-boundary after 'supports?' in every pattern guarantees that
    past-tense 'supported' never matches any branch.
    """
    if not comment:
        return None
    if _CB_ONE_RE.search(comment):
        return "one_support"
    if _CB_TWO_RE.search(comment):
        return "two_supports"
    if _CB_NO_RE.search(comment):
        return "no_support"
    if _SUPPORT_NOUN_RE.search(comment):
        return "unclassified_support"
    return None


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

def _detect_warmup_flags(sets: list,
                          exercise_name: str = "",
                          ctx: dict = None,
                          weight_eligible: bool = True,
                          exercise_alltime_max: float = 0.0) -> None:
    """
    Tier 1 pre-pass — mark AT MOST ONE warmup set per session, always the
    first set (lowest set_id).

    Requires ≥ 3 sets total (≥ 2 working sets after the potential warmup).
    Sessions with fewer than 3 sets never receive a warmup flag from the
    weight-based path; an explicit comment still applies regardless.

    weight_eligible: False when this exercise was NOT the first in its category
    on that day (category-first-exercise gate). Explicit "warmup" always wins.

    exercise_alltime_max: the exercise's all-time max headline weight (plates +
    bar), used only for the 0-weight-opener gate.

    The first set is the warmup if:
      • it carries an explicit 'warmup' / 'warm up' comment (no gate), OR
      • weight_eligible is True AND it has ≥ WARMUP_MIN_REPS reps, s1 < s2, AND:
          – s1 == 0 (bodyweight opener): working-set max ≥ HEAVY_FRACTION × alltime
            max — the session is heavy relative to history (genuine warm-up before
            max-range work). If alltime max is unknown (0), no flag.
          – s1 > 0 (weighted opener): existing gap/flat-ratio rule:
              all working sets equal (max_step == 0): s1 ≤ WARMUP_FLAT_RATIO × s2
              working sets vary (max_step > 0): gap(s1→s2) > WARMUP_GAP_RATIO × max_step

    Negative gate: ≥ 2 ROM terms in the first set's comment AND no pain
    attribution → genuine form hierarchy → nothing flagged.
    Pain attribution overrides the gate (warmup still allowed).

    ctx is kept as a parameter for call-site compatibility but is unused.
    """
    if not sets:
        return

    for s in sets:
        s["is_warmup"] = False

    ordered = sorted(sets, key=lambda s: s.get("set_id", 0))
    first   = ordered[0]
    rest    = ordered[1:]

    comment = (first.get("comment") or "").lower()

    has_explicit_warmup = "warmup" in comment or "warm up" in comment

    # ROM negative gate — ≥2 distinct terms + no pain = genuine form hierarchy
    rom_hits      = sum(1 for t in _ROM_GATE_TERMS if t in comment)
    has_pain_gate = any(kw in comment for kw in _WARMUP_GATE_PAIN_VOCAB)
    if rom_hits >= 2 and not has_pain_gate:
        return

    # Explicit comment always wins regardless of any other gate
    if has_explicit_warmup:
        first["is_warmup"] = True
        return

    # Blocked when not first-in-category, < 3 sets, or < 12 reps on the opener
    if not weight_eligible or len(rest) < 2 or first.get("reps", 0) < WARMUP_MIN_REPS:
        return

    s1 = first.get("weight", 0)
    s2 = rest[0].get("weight", 0)
    if s1 >= s2:
        return

    # ── 0-weight (bodyweight / empty-bar) opener ──────────────────────────────
    # History-relative heaviness check: the opener is a genuine warmup only when
    # the working sets reach >= HEAVY_FRACTION of the exercise's all-time best
    # headline weight. exercise_alltime_max is BAR-INCLUSIVE, so working_max must
    # be too — use the set's own headline_weight (plates + eff_bar, already
    # computed by the session builder), not plates, or the bar understates the
    # left side and genuine empty-bar warmups are missed. (s1 stays plates: an
    # "empty bar" opener is 0 PLATES.) headline_weight is absent only on the
    # daily-workouts set dicts, where it harmlessly falls back to plates.
    if s1 == 0:
        if exercise_alltime_max > 0:
            working_max = max(s.get("headline_weight", s.get("weight", 0))
                              for s in rest)
            if working_max >= HEAVY_FRACTION * exercise_alltime_max:
                first["is_warmup"] = True
        return   # 0-openers never fall through to gap/flat-ratio logic

    # ── Non-zero weighted opener: gap/flat-ratio rule ─────────────────────────
    working_weights = [s.get("weight", 0) for s in rest]
    working_steps   = [abs(working_weights[i + 1] - working_weights[i])
                       for i in range(len(working_weights) - 1)]
    max_step = max(working_steps)

    if max_step == 0:
        # All working sets equal: opener must be well below s2
        qualifies = s1 <= WARMUP_FLAT_RATIO * s2
    else:
        # Working sets vary: gap from opener to first working set must break the ramp
        qualifies = (s2 - s1) > WARMUP_GAP_RATIO * max_step

    if qualifies:
        first["is_warmup"] = True


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


def _safe_compute(fn, *args, default=None, label="", **kwargs):
    """
    Call fn(*args, **kwargs), returning default on any exception.
    Ensures one failing computation never crashes the entire analysis.
    A partial result missing one metric is far more useful than no result.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("[data_agent] %s failed: %s", label or fn.__name__, e)
        return default


# ── Thin-data gating — statistical helpers ────────────────────────────────────
#
# Principle: Python reports evidence strength (facts about the data).
# The Analysis Agent decides the bar (a judgment about domain knowledge).
#
# Every correlational output carries: n per condition, effect size, CI, overlap.
# Do NOT collapse these into a single percentage — that loses the three
# independent dimensions the Analysis Agent needs to reason correctly.
#
# CI uses the t-distribution without scipy. n=2 → df=1 → t=12.7 → huge margin.
# The CI blows up automatically on thin data. This is correct behavior:
# the math self-flags thin data without requiring a chosen cutoff.
# ─────────────────────────────────────────────────────────────────────────────

# 95% CI t-quantiles (two-tailed, alpha=0.025 per tail)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,  5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000,
}

def _t95(df: int) -> float:
    """95% CI t-quantile. Linearly interpolates between table entries."""
    if df <= 0:    return float("inf")
    if df in _T95: return _T95[df]
    if df >= 60:   return 1.960
    keys = sorted(_T95)
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    frac = (df - lo) / (hi - lo)
    return _T95[lo] + frac * (_T95[hi] - _T95[lo])


def _ci_stats(values: list) -> dict:
    """
    n, mean, std, and 95% CI for a list of values.
    n=1 → CI=None (can't estimate spread from one point).
    n≥2 → t-distribution CI. Blows up for small n — correct behavior.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci_95": None}
    if n == 1:
        return {"n": 1, "mean": round(values[0], 1), "std": None, "ci_95": None}
    mean   = sum(values) / n
    var    = sum((x - mean) ** 2 for x in values) / (n - 1)
    std    = var ** 0.5
    margin = _t95(n - 1) * std / (n ** 0.5)
    return {
        "n":     n,
        "mean":  round(mean, 1),
        "std":   round(std, 1),
        "ci_95": [round(mean - margin, 1), round(mean + margin, 1)],
    }


def _cohen_d(a: list, b: list) -> Optional[float]:
    """
    Cohen's d between two groups. None if either group < 2 values.
    |d| < 0.2 small, 0.2–0.5 medium, > 0.5 large.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a);  mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = (((len(a) - 1) * va + (len(b) - 1) * vb) /
              (len(a) + len(b) - 2)) ** 0.5
    return round((ma - mb) / pooled, 2) if pooled > 0 else 0.0


def _cis_overlap(ci_a: Optional[list], ci_b: Optional[list]) -> bool:
    """True if two CIs overlap. Conservative: returns True if either is None."""
    if not ci_a or not ci_b:
        return True
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def _effect_label(d: Optional[float], n_min: int) -> str:
    """
    Convenience label from Cohen's d and minimum group size.
    Always travels with the underlying numbers — label alone is never reported.
    """
    if d is None or n_min < 2:
        return "insufficient_data"
    ad = abs(d)
    if ad < 0.2:   return "weak"
    if ad < 0.5:   return "moderate"
    return "strong"


def _pearson_r_with_ci(xs: list, ys: list) -> dict:
    """
    Pearson r with 95% CI via Fisher z-transform.
    Requires n ≥ 4 for a meaningful CI (denominator is sqrt(n-3)).
    Returns {"r": ..., "ci_95": [...], "n": n}.
    """
    n = len(xs)
    if n < 3:
        return {"r": None, "ci_95": None, "n": n}
    mx = sum(xs) / n;  my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy  = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return {"r": 0.0, "ci_95": None, "n": n}
    r = max(-1.0, min(1.0, num / (dx * dy)))
    if n < 4 or abs(r) >= 1.0:
        return {"r": round(r, 3), "ci_95": None, "n": n}
    z    = 0.5 * math.log((1 + r) / (1 - r))
    se   = 1.0 / (n - 3) ** 0.5
    r_lo = math.tanh(z - 1.960 * se)
    r_hi = math.tanh(z + 1.960 * se)
    return {"r": round(r, 3), "ci_95": [round(r_lo, 3), round(r_hi, 3)], "n": n}


# ── Session building ───────────────────────────────────────────────────────────

def _build_sessions_from_rows(rows: list, ctx: dict,
                               exercise_name: str,
                               units_review_log: Optional[list] = None,
                               warmup_eligible: Optional[frozenset] = None,
                               exercise_alltime_max: float = 0.0,
                               counterbalance_review_log: Optional[list] = None) -> list:
    offset    = _get_numeric_offset(ctx, exercise_name)
    _is_smith = _is_smith_machine(ctx, exercise_name)
    by_date: dict = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)

    sessions = []
    for session_date in sorted(by_date.keys()):
        date_rows  = by_date[session_date]
        bar_weight_lbs    = _get_bar_weight_lbs(ctx, exercise_name, session_date)
        _session_is_kg    = _is_kg_native(ctx, exercise_name, session_date)
        bar_weight        = bar_weight_lbs / 2.2046 if _session_is_kg else bar_weight_lbs
        curated_unit      = "kg" if _session_is_kg else "lbs"

        reps_zero_normal = _is_reps_zero_normal(ctx, exercise_name)
        sets = []
        for r in date_rows:
            comment    = r["comment"]

            # ── Tier 1 unit prepass ────────────────────────────────────────────
            # Detect a standalone unit token in the comment.  A token that
            # DISAGREES with the curated unit is applied to this set only and
            # logged loud; one that agrees is a no-op.
            comment_unit = _detect_unit_comment_token(comment)
            if comment_unit is not None and comment_unit != curated_unit:
                if units_review_log is not None:
                    units_review_log.append({
                        "exercise": exercise_name,
                        "date":     session_date,
                        "comment":  comment,
                    })
                # Apply this-set override: recover weight in the comment's unit.
                # comment_unit=="kg" in a lbs exercise: metric_weight is the kg
                # value the user typed → use raw metric (no × 2.2046).
                # comment_unit=="lbs" in a kg exercise: recover in lbs.
                if comment_unit == "kg":
                    weight = round(r["metric_weight"], 1)
                else:
                    weight = _recover_typed_weight(r["metric_weight"], offset)
            else:
                weight = _recover_typed_weight(r["metric_weight"], offset)

            # ── Tier 2 counterbalance pre-pass ────────────────────────────────
            _cb = _detect_counterbalance(comment)
            if _cb == "unclassified_support":
                if counterbalance_review_log is not None:
                    counterbalance_review_log.append({
                        "exercise": exercise_name,
                        "date":     session_date,
                        "comment":  comment,
                    })
                _cb = None
            elif _cb in ("one_support", "two_supports") and not _is_smith:
                logger.warning(
                    "[data_agent] counterbalance declaration on non-Smith "
                    "exercise %s %s comment=%r — not applied",
                    exercise_name, session_date, comment,
                )
                _cb = None

            # Effective bar for this set (reduced for Smith counterbalance sets)
            if _is_smith and _cb in ("one_support", "two_supports"):
                _n_supports = 1 if _cb == "one_support" else 2
                if _session_is_kg:
                    eff_bar = max(0.0,
                                  bar_weight - _n_supports * COUNTERBALANCE_KG_EACH)
                else:
                    eff_bar = max(0.0,
                                  bar_weight - _n_supports * COUNTERBALANCE_KG_EACH * 2.2046)
            else:
                eff_bar = bar_weight

            reps       = r["reps"]
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
                # set_db_id == training_log._id: the row the comment is bound to
                # by the fetch LEFT JOIN (Comment.owner_id = tl._id). The comment
                # is carried on its own set by id — never re-found by weight/reps.
                "set_db_id":          r["set_id"],
                "weight":             weight,
                "reps":               reps,
                "distance":           round(distance, 3),
                "duration_seconds":   int(duration_s),
                "comment":            comment,
                "drop_group":         _detect_drop_group(comment),
                "headline_weight":    round(weight + eff_bar, 2),
                "estimated_1rm":      _epley_1rm(weight + eff_bar, reps),
                "is_failed_attempt":  is_failed,
                "is_pain_flag":       _is_pain_comment(comment),
                "technique_variants": _detect_technique_variants(comment),
                "keyword_counts":     _count_keyword_categories(comment),
                "is_personal_record": bool(r["is_personal_record"]),
                "is_warmup":          False,
            })

        eligible = warmup_eligible is None or (exercise_name, session_date) in warmup_eligible
        _detect_warmup_flags(sets, exercise_name=exercise_name, ctx=ctx,
                             weight_eligible=eligible,
                             exercise_alltime_max=exercise_alltime_max)
        working_sets = [s for s in sets if not s["is_warmup"]] or sets
        warmup_sets  = [s for s in sets if s["is_warmup"]]

        max_working_weight = round(
            max(s["headline_weight"] for s in working_sets), 2
        )
        reps_at_max        = max(
            (s["reps"] for s in working_sets
             if abs(s["headline_weight"] - max_working_weight) < 0.01),
            default=0,
        )
        session_e1rm       = max((s["estimated_1rm"] for s in working_sets
                                   if s["reps"] > 0), default=0.0)
        total_volume       = sum(s["headline_weight"] * s["reps"] for s in sets)
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
            "warmup_weight_plates_only": True,
            "is_pr_session":      any(s["is_personal_record"] for s in sets),
            "form_quality":       form_quality,
            "form_detail":        form_detail,
            "comment_count":      sum(1 for s in sets if s["comment"]),
            "has_pain_flag":      any(s["is_pain_flag"] for s in sets),
            "failed_attempts":    sum(1 for s in sets if s["is_failed_attempt"]),
            "technique_variants": sorted({v for s in sets for v in s["technique_variants"]}),
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
        "dates": [], "max_weight": 0.0,
        "total_volume_lbs": 0.0, "total_volume_kg": 0.0,
        "session_count": 0, "e1rm_values": [], "form_qualities": [],
        "pain_count": 0, "failed_count": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        week = _iso_week_key(s["date"])
        bw   = by_week[week]
        bw["dates"].append(s["date"])
        bw["session_count"]  += 1
        # Per-unit volume buckets: a week may span a unit switch (Deadlift
        # lbs->kg on 2025-12-26) — never add a kg session onto an lbs sum.
        bw["total_volume_kg" if s.get("unit") == "kg" else "total_volume_lbs"] \
            += s["total_volume"]
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
            "total_volume_lbs":   round(d["total_volume_lbs"], 1),
            "total_volume_kg":    round(d["total_volume_kg"], 1),
            "peak_estimated_1rm": round(max(e1rm), 1) if e1rm else 0.0,
            # sorted() tie-break: deterministic across processes (set iteration
            # order is hash-randomized; equal-count modes used to flip randomly)
            "form_quality_mode":  max(sorted(set(q)), key=q.count) if q else "unknown",
            "pain_sessions":      d["pain_count"],
            "failed_attempts":    d["failed_count"],
            "total_distance":         round(d["total_distance"], 3),
            "total_duration_seconds": d["total_duration_seconds"],
        })
    return result


def _aggregate_monthly(sessions: list) -> list:
    by_month: dict = defaultdict(lambda: {
        "dates": [], "max_weight": 0.0,
        "total_volume_lbs": 0.0, "total_volume_kg": 0.0,
        "session_count": 0, "e1rm_values": [], "reps_at_max_list": [],
        "pain_count": 0, "failed_count": 0,
        "strength_sets": 0, "hypertrophy_sets": 0, "endurance_sets": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        month = s["date"][:7]; bm = by_month[month]
        bm["dates"].append(s["date"])
        bm["session_count"]    += 1
        bm["total_volume_kg" if s.get("unit") == "kg" else "total_volume_lbs"] \
            += s["total_volume"]
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
            "total_volume_lbs":   round(d["total_volume_lbs"], 1),
            "total_volume_kg":    round(d["total_volume_kg"], 1),
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
        "months": set(), "max_weight": 0.0,
        "total_volume_lbs": 0.0, "total_volume_kg": 0.0,
        "session_count": 0, "e1rm_values": [],
        "first_weight": None, "last_weight": None,
        "pain_count": 0, "pr_count": 0,
        "total_distance": 0.0, "total_duration_seconds": 0,
    })
    for s in sessions:
        year = s["date"][:4]; by = by_year[year]
        by["months"].add(s["date"][:7])
        by["session_count"]  += 1
        by["total_volume_kg" if s.get("unit") == "kg" else "total_volume_lbs"] \
            += s["total_volume"]
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
            "total_volume_lbs":           round(d["total_volume_lbs"], 1),
            "total_volume_kg":            round(d["total_volume_kg"], 1),
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

def _session_wr_kg(s: dict, default_unit: str) -> tuple:
    """
    (kg-normalized working-max weight, reps-at-that-weight) for a session —
    the pair used for new-best / current-ability comparisons. Weight is
    kg-normalized so a mid-history unit switch (Deadlift lbs→kg) compares
    correctly; reps are the reps achieved AT that working max in the SAME
    session (never mixed across sets).
    """
    return (_to_kg(s["max_working_weight"], s.get("unit", default_unit)),
            s.get("reps_at_max", 0) or 0)


def _is_new_best(w_kg: float, r: int, best_w_kg, best_r) -> bool:
    """
    PR rule (weight → reps), matching _compute_alltime_pr so plateau-logic and
    PR-logic never contradict. A session is a new best iff its top working
    weight is heavier, OR the same weight with more reps. e1RM is deliberately
    NOT used here — Epley inflates light high-rep sets and would crown a warmup.
    """
    if best_w_kg is None:
        return True
    if w_kg > best_w_kg + NEW_BEST_WEIGHT_TOL_KG:
        return True
    if abs(w_kg - best_w_kg) <= NEW_BEST_WEIGHT_TOL_KG and r > best_r:
        return True
    return False


def _last_new_best_index(sessions: list, default_unit: str) -> int:
    """Index of the most recent session that set a new best by the PR rule."""
    best_w = best_r = None
    last_idx = 0
    for i, s in enumerate(sessions):
        w, r = _session_wr_kg(s, default_unit)
        if _is_new_best(w, r, best_w, best_r):
            best_w, best_r = w, r
            last_idx = i
    return last_idx


def _recent_best_session(sessions: list, default_unit: str, k: int) -> dict:
    """
    The session holding the best (weight → reps) over the exercise's last k
    sessions — the robust 'current ability'. A lone back-off / high-rep day
    cannot lower it below the established working max.
    """
    window = sessions[-k:] if k else sessions
    return max(window, key=lambda s: _session_wr_kg(s, default_unit))


def _compute_progression(sessions: list) -> dict:
    if not sessions: return {}
    first = sessions[0]; last = sessions[-1]
    first_unit            = first.get("unit", "lbs")
    last_unit             = last.get("unit",  "lbs")
    unit_switch_in_period = first_unit != last_unit
    # max_weight_start / end / weight_change / e1rm are computed AFTER current
    # ability below: the progression "end" is the robust recent best (never the
    # possibly back-off last session), and "start" is re-anchored into the current
    # unit frame when the window spans a unit switch (no boundary-spanning %).

    # Plateau / peak / regression / current-ability are kg-normalized per
    # session unit. Raw comparison is a latent cross-unit bug for exercises
    # with a mid-history unit switch (Deadlift): a 150 lbs session would
    # compare as "above" a 70 kg session. For single-unit exercises the kg
    # conversion is the same monotonic transform on every value.
    def _mww_kg(s: dict) -> float:
        return _to_kg(s["max_working_weight"], s.get("unit", last_unit))

    def _to_last_unit(w_kg: float) -> float:
        return round(w_kg if last_unit == "kg" else w_kg * 2.2046, 1)

    n = len(sessions)
    trend_assessable = n >= MIN_SESSIONS_FOR_TREND

    # ── Peak (best headline weight, kg-normalized) ─────────────────────────────
    peak_kg      = max(_mww_kg(s) for s in sessions)
    peak_session = next(s for s in reversed(sessions)
                        if _mww_kg(s) >= peak_kg - 1e-9)
    peak_date    = peak_session["date"]
    peak_weight  = (peak_session["max_working_weight"]
                    if peak_session.get("unit", last_unit) == last_unit
                    else _to_last_unit(peak_kg))

    # ── "Current ability" = robust recent best, NOT the single last session ────
    # Best (weight → reps) over the exercise's last CURRENT_ABILITY_SESSIONS, so
    # a lone back-off / high-rep day cannot lower current below the working max.
    cur_session    = _recent_best_session(sessions, last_unit, CURRENT_ABILITY_SESSIONS)
    current_kg     = _mww_kg(cur_session)
    current_weight = (cur_session["max_working_weight"]
                      if cur_session.get("unit", last_unit) == last_unit
                      else _to_last_unit(current_kg))
    current_reps   = cur_session.get("reps_at_max", 0)
    current_basis  = f"best of last {min(CURRENT_ABILITY_SESSIONS, n)} sessions"

    # ── Start anchor (Fix 2: same-unit baseline across a unit switch) ──────────
    # Normally the window's first session. When the window spans a unit switch
    # (e.g. Deadlift lbs→kg on 2025-12-26), anchor "start" on the first session
    # in the END (current) unit frame so the % reflects same-unit progress and
    # never spans the boundary. True window bounds stay in first/last_session_date.
    progression_note = None
    if unit_switch_in_period:
        frame_sessions = [s for s in sessions
                          if s.get("unit", last_unit) == last_unit]
        start_session  = frame_sessions[0] if frame_sessions else first
        progression_note = (
            f"window spans a {first_unit}->{last_unit} unit switch; weight change "
            f"is computed within the {last_unit} era only "
            f"(from {start_session['date']})"
        )
    else:
        frame_sessions = sessions
        start_session  = first
    max_weight_start = start_session["max_working_weight"]
    e1rm_start       = start_session["estimated_1rm"]
    start_unit       = start_session.get("unit", first_unit)

    # ── End anchor = CURRENT ABILITY, never the back-off last session (Fix 1) ──
    # max_weight_end / weight_change / e1rm_end track the robust recent best, so
    # a lighter last session can't read as the end of progression (the −7.7% /
    # "60→70" artifacts came from anchoring on sessions[-1]).
    max_weight_end = current_weight
    e1rm_end       = cur_session["estimated_1rm"]
    weight_change  = round(max_weight_end - max_weight_start, 1)
    if max_weight_start > 0 and len(frame_sessions) >= 2:
        weight_change_pct = round(weight_change / max_weight_start * 100, 1)
    else:
        # Too few same-unit sessions to anchor a meaningful percentage.
        weight_change_pct = None
        if unit_switch_in_period and progression_note:
            progression_note += " — too few same-unit sessions for a percentage"

    # ── Latest session vs current ability (Fix 3: back-off label) ──────────────
    # The most recent session, labeled distinctly from current ability so a
    # lighter/back-off day is not narrated as a decline. is_backoff is True when
    # the latest session is below current ability by the new-best weight→reps rule.
    latest_session_date   = last["date"]
    latest_session_weight = last["max_working_weight"]
    latest_session_reps   = last.get("reps_at_max", 0)
    _last_kg = _mww_kg(last)
    latest_session_is_backoff = (
        _last_kg < current_kg - NEW_BEST_WEIGHT_TOL_KG
        or (abs(_last_kg - current_kg) <= NEW_BEST_WEIGHT_TOL_KG
            and latest_session_reps < current_reps)
    )

    # ── New-best tracking (literal weight → reps, never e1RM) ──────────────────
    last_best_idx       = _last_new_best_index(sessions, last_unit)
    last_new_best_date  = sessions[last_best_idx]["date"]
    sessions_since_best = (n - 1) - last_best_idx
    plateau_span_days   = (datetime.strptime(last["date"], "%Y-%m-%d").date() -
                           datetime.strptime(last_new_best_date, "%Y-%m-%d").date()).days

    # ── e1RM trend-direction gate (e1RM used ONLY here) ────────────────────────
    # A rising recent e1RM slope blocks a plateau even past the session count —
    # same-weight rep gains (130×5 → 130×9) push e1RM up and must not read flat.
    recent_e1rms = [s["estimated_1rm"] for s in sessions[-PLATEAU_SLOPE_WINDOW:]
                    if s["estimated_1rm"] > 0]
    e1rm_rising  = _trend(recent_e1rms, 0.02, 0.02) == "increasing"

    # ── Plateau: ALL of (a) enough sessions since the last new best,
    #    (b) recent e1RM not rising, (c) enough sessions to judge ───────────────
    is_plateau = (trend_assessable
                  and sessions_since_best >= PLATEAU_MIN_SESSIONS_SINCE_BEST
                  and not e1rm_rising)
    plateau_since = last_new_best_date if is_plateau else None

    if not trend_assessable:
        plateau_note = f"not enough sessions to assess trend (only {n} logged)"
    elif is_plateau:
        plateau_note = (f"no new best in the last {sessions_since_best} "
                        f"sessions (~{plateau_span_days} days)")
    else:
        plateau_note = (f"progressing — last new best {last_new_best_date} "
                        f"({sessions_since_best} session(s) ago)")

    # sessions_at_max: informational count sitting at the peak weight.
    sessions_at_max = sum(1 for s in sessions if _mww_kg(s) >= peak_kg - 1e-9)

    # ── Regression: peak vs the robust recent best, NOT the last session ───────
    # A single lighter / back-off day must not register as a regression; only a
    # sustained drop in the recent-best window counts. Thin data: don't opine.
    regression_from_peak = None
    if trend_assessable and current_kg < peak_kg - 0.005:
        regression_from_peak = {
            "peak_weight":       peak_weight,
            "peak_date":         peak_date,
            "current_weight":    current_weight,
            "current_basis":     current_basis,
            "regression_amount": round(peak_weight - current_weight, 1),
            "regression_pct":    round((peak_weight - current_weight) / peak_weight * 100, 1),
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
        "max_weight_start_original": max_weight_start,
        "start_session_date":        start_session["date"],
        "first_session_unit":        first_unit,
        "last_session_unit":         last_unit,
        "unit_switch_in_period":     unit_switch_in_period,
        "progression_note":          progression_note,
        # End fields track CURRENT ABILITY (recent best), not the last session
        "max_weight_end":       max_weight_end,
        "display_weight_start": f"{max_weight_start} {start_unit}",
        "display_weight_end":   f"{max_weight_end} {last_unit}",
        "weight_change":        weight_change,
        "weight_change_pct":    weight_change_pct,
        "e1rm_start":           e1rm_start,
        "e1rm_start_original":  e1rm_start,
        "e1rm_end":             e1rm_end,
        "e1rm_change":          round(e1rm_end - e1rm_start, 1),
        "sessions_at_max":      sessions_at_max,
        "session_count":        len(sessions),
        "reps_at_max_start":    start_session["reps_at_max"],
        "reps_at_max_end":      current_reps,
        # ── Current ability (robust recent best, not the last session) ────────
        "current_weight":       current_weight,
        "current_reps":         current_reps,
        "current_basis":        current_basis,
        # ── Most recent session (labeled distinct from current ability) ───────
        "latest_session_date":       latest_session_date,
        "latest_session_weight":     latest_session_weight,
        "latest_session_reps":       latest_session_reps,
        "latest_session_is_backoff": latest_session_is_backoff,
        # ── Plateau (rep-aware new-best rule, cadence-scaled) ─────────────────
        "is_plateau":           is_plateau,
        "plateau_since":        plateau_since,
        "last_new_best_date":   last_new_best_date,
        "sessions_since_best":  sessions_since_best,
        "plateau_span_days":    plateau_span_days if is_plateau else None,
        "plateau_note":         plateau_note,
        "trend_assessable":     trend_assessable,
        "e1rm_recent_rising":   e1rm_rising,
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
    # B2a: comment from the working set with the highest plates weight
    _raw = pr_session.get("sets", [])
    _working_c = [st for st in _raw if not st.get("is_warmup", False)] or _raw
    _pr_set = max(_working_c, key=lambda st: st["weight"], default=None)
    return {
        "weight":                    best_weight,
        "reps":                      best_reps,
        "date":                      pr_date,
        "comment":                   _pr_set.get("comment") if _pr_set else None,
        "estimated_1rm":             best_e1rm,
        "unit":                      unit,
        "pr_session_comment_count":  pr_session["comment_count"],
        "pr_session_had_pain":       pr_session["has_pain_flag"],
    }


def _compute_alltime_pr(alltime_sessions: list, unit: str) -> Optional[dict]:
    """
    Compute all-time PR from the full alltime_sessions history (B2 — not the app flag).

    Uses kg-normalized comparison to handle exercises that changed units
    mid-history (e.g. Deadlift: lbs before 2025-12-26, kg after).
    Returns the bar-inclusive headline weight, reps, date, comment (B2a), and e1rm.
    """
    if not alltime_sessions:
        return None

    def _mww_kg(s: dict) -> float:
        return _to_kg(s["max_working_weight"], s.get("unit", unit))

    # Find best by kg-normalized headline weight, then max reps, then most recent
    best_kg      = max(_mww_kg(s) for s in alltime_sessions)
    at_best      = [s for s in alltime_sessions if abs(_mww_kg(s) - best_kg) < 0.05]
    best_reps    = max(s["reps_at_max"] for s in at_best)
    at_best_reps = [s for s in at_best if s["reps_at_max"] == best_reps]
    pr_session   = sorted(at_best_reps, key=lambda s: s["date"])[-1]

    # B2a: comment from the working set with the highest plates weight
    _raw = pr_session.get("sets", [])
    _working_c = [st for st in _raw if not st.get("is_warmup", False)] or _raw
    _pr_set = max(_working_c, key=lambda st: st["weight"], default=None)
    pr_comment = _pr_set.get("comment") if _pr_set else None

    # All-time best e1RM — may come from a higher-rep session at moderate weight
    best_e1rm = max(
        (s["estimated_1rm"] for s in alltime_sessions if s["estimated_1rm"] > 0),
        default=0.0,
    )

    return {
        "weight":        pr_session["max_working_weight"],
        "reps":          pr_session["reps_at_max"],
        "date":          pr_session["date"],
        "comment":       pr_comment,
        "estimated_1rm": best_e1rm,
        "unit":          pr_session.get("unit", unit),
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
    if not dist_sessions:
        return None
    if len(dist_sessions) < 2:
        # Single session: return a stub with all trim_package-readable keys populated
        # so progression.distance_total_km is non-null (C1) even for one-session windows.
        return {
            "session_count":      1,
            "first_session_date": dist_sessions[0][0],
            "last_session_date":  dist_sessions[0][0],
            "distance_start_km":  dist_sessions[0][1],
            "distance_end_km":    dist_sessions[0][1],
            "distance_peak_km":   dist_sessions[0][1],
            "distance_peak_date": dist_sessions[0][0],
            "avg_distance_km":    dist_sessions[0][1],
            "total_distance_km":  dist_sessions[0][1],
            "note":               "only_one_session",
        }
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


def _evaluate_phase2(progression: dict, end_date) -> tuple:
    # Plateau is now a rep-aware, cadence-scaled judgment made in
    # _compute_progression (≥ N sessions with no new best AND recent e1RM not
    # rising). Trust that decision instead of re-deriving from a raw calendar
    # delta that ignored cadence and rep progress. plateau_days is the span
    # since the last new best.
    plateau_days = 0
    if progression.get("is_plateau"):
        plateau_days = progression.get("plateau_span_days") or 0
        if not plateau_days and progression.get("plateau_since"):
            plateau_dt   = datetime.strptime(progression["plateau_since"], "%Y-%m-%d").date()
            plateau_days = (end_date - plateau_dt).days
    # A declared plateau OR a large swing pulls the full comment history —
    # comments are where the reason lives (injury, deload, technique rebuild).
    # abs(): a >20% REGRESSION warrants it as much as a >20% improvement.
    triggered = (
        bool(progression.get("is_plateau")) or
        abs(progression.get("weight_change_pct") or 0) > IMPROVEMENT_TRIGGER_PCT
    )
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


def _compute_rest_performance_buckets(sessions: list) -> dict:
    """
    Group sessions by days of rest since the previous session, compare e1RM.

    Each bucket: n, mean, std, 95% CI.
    CI blows up for small n — wide CI signals thin data to the Analysis Agent.
    comparison.confidence_label always travels with the raw statistical numbers.
    """
    if len(sessions) < 3:
        return {"buckets": [], "comparison": None}
    data = []
    for i in range(1, len(sessions)):
        prev = datetime.strptime(sessions[i-1]["date"], "%Y-%m-%d").date()
        curr = datetime.strptime(sessions[i  ]["date"], "%Y-%m-%d").date()
        rest = (curr - prev).days
        if sessions[i]["estimated_1rm"] > 0:
            data.append((rest, sessions[i]["estimated_1rm"]))
    if not data:
        return {"buckets": [], "comparison": None}

    raw: dict = {"1-3 days": [], "4-6 days": [], "7-13 days": [], "14+ days": []}
    for rest, e1rm in data:
        if   rest <= 3:  raw["1-3 days"].append(e1rm)
        elif rest <= 6:  raw["4-6 days"].append(e1rm)
        elif rest <= 13: raw["7-13 days"].append(e1rm)
        else:            raw["14+ days"].append(e1rm)

    buckets = []
    for name, vals in raw.items():
        if not vals:
            continue
        s = _ci_stats(vals)
        buckets.append({
            "rest_range": name,
            "n":          s["n"],
            "mean_e1rm":  s["mean"],
            "std_e1rm":   s["std"],
            "ci_95":      s["ci_95"],
            "max_e1rm":   round(max(vals), 1),
        })

    comparison = None
    if len(buckets) >= 2:
        best  = max(buckets, key=lambda b: b["mean_e1rm"] or 0)
        worst = min(buckets, key=lambda b: b["mean_e1rm"] or 0)
        if best["rest_range"] != worst["rest_range"]:
            d = _cohen_d(raw[best["rest_range"]], raw[worst["rest_range"]])
            comparison = {
                "best_bucket":      best["rest_range"],
                "worst_bucket":     worst["rest_range"],
                "mean_diff_e1rm":   round((best["mean_e1rm"] or 0) - (worst["mean_e1rm"] or 0), 1),
                "cohen_d":          d,
                "cis_overlap":      _cis_overlap(best["ci_95"], worst["ci_95"]),
                "confidence_label": _effect_label(d, min(best["n"], worst["n"])),
            }

    return {"buckets": buckets, "comparison": comparison}


def _compute_consecutive_day_effect(sessions: list, all_dates: list) -> dict:
    """
    Group sessions by consecutive training days before them, compare e1RM.
    Each condition: n, mean, std, 95% CI. comparison carries effect size.
    """
    if not sessions or not all_dates:
        return {"by_consecutive_days": [], "comparison": None}
    date_set = set(all_dates)
    def consec_before(d_str):
        dt = datetime.strptime(d_str, "%Y-%m-%d").date()
        count = 0
        prev  = dt - timedelta(days=1)
        while prev.strftime("%Y-%m-%d") in date_set:
            count += 1; prev -= timedelta(days=1)
        return count
    by_c: dict = defaultdict(list)
    for s in sessions:
        c = consec_before(s["date"])
        if s["estimated_1rm"] > 0:
            by_c[c].append(s["estimated_1rm"])

    rows = []
    for c, vals in sorted(by_c.items()):
        st = _ci_stats(vals)
        rows.append({
            "consecutive_days_before": c,
            "n":          st["n"],
            "mean_e1rm":  st["mean"],
            "std_e1rm":   st["std"],
            "ci_95":      st["ci_95"],
        })

    comparison = None
    if len(rows) >= 2:
        best  = max(rows, key=lambda r: r["mean_e1rm"] or 0)
        worst = min(rows, key=lambda r: r["mean_e1rm"] or 0)
        if best["consecutive_days_before"] != worst["consecutive_days_before"]:
            d = _cohen_d(
                by_c[best["consecutive_days_before"]],
                by_c[worst["consecutive_days_before"]])
            comparison = {
                "best_condition":   best["consecutive_days_before"],
                "worst_condition":  worst["consecutive_days_before"],
                "mean_diff_e1rm":   round((best["mean_e1rm"] or 0) - (worst["mean_e1rm"] or 0), 1),
                "cohen_d":          d,
                "cis_overlap":      _cis_overlap(best["ci_95"], worst["ci_95"]),
                "confidence_label": _effect_label(d, min(best["n"], worst["n"])),
            }

    return {"by_consecutive_days": rows, "comparison": comparison}


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
    failed_sets = [{"date": s["date"], "weight": st["weight"],
                    "weight_plates_only": True, "comment": st["comment"]}
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


def _compute_goal_projection(sessions: list, goal: dict, unit: str, today,
                             ctx: dict = None) -> dict:
    """today is passed in explicitly — process.py never calls date.today()."""
    if not sessions or not goal: return {}
    target_weight = goal["target_weight"]
    # B1a bar-inclusive frame: current_e1rm / current_max come from the last
    # session as headline values (plates + bar). target_weight is a typed plate
    # number, so add the exercise's bar (same value + frame the session builder
    # uses for the last session's date) before computing target_e1rm — otherwise
    # the gap is understated and is_on_track / months_needed run optimistic.
    bar_weight = 0.0
    exercise_name = goal.get("exercise_name", "")
    if ctx is not None and exercise_name:
        _last_date = sessions[-1].get("date")
        if _last_date:
            _bar_lbs = _get_bar_weight_lbs(ctx, exercise_name, _last_date)
            bar_weight = (_bar_lbs / 2.2046
                          if _is_kg_native(ctx, exercise_name, _last_date)
                          else _bar_lbs)
    target_e1rm   = _epley_1rm(target_weight + bar_weight, goal.get("target_reps", 1))
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
        proj_dt        = today + timedelta(days=int(months_needed * 30))
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

def _build_daily_workouts(all_rows: list, ctx: dict,
                           warmup_eligible: Optional[frozenset] = None,
                           ex_alltime_max_map: Optional[dict] = None) -> list:
    by_date: dict = defaultdict(list)
    for row in all_rows: by_date[row["date"]].append(row)
    daily = []
    for training_date in sorted(by_date.keys()):
        rows = by_date[training_date]
        by_ex: dict = defaultdict(list)
        for row in rows: by_ex[row["exercise_name"]].append(row)
        exercise_order = sorted(by_ex.keys(),
                                key=lambda ex: min(r["set_id"] for r in by_ex[ex]))
        exercises_done = []; total_vol_lbs = 0.0; total_vol_kg = 0.0; total_sets = 0
        for pos, ex_name in enumerate(exercise_order, start=1):
            ex_rows    = by_ex[ex_name]
            offset     = _get_numeric_offset(ctx, ex_name)
            bar_weight_lbs = _get_bar_weight_lbs(ctx, ex_name, training_date)
            _ex_is_kg      = _is_kg_native(ctx, ex_name, training_date)
            bar_weight     = bar_weight_lbs / 2.2046 if _ex_is_kg else bar_weight_lbs
            category   = CATEGORY_NAMES.get(ex_rows[0]["category_id"],
                                             f"Cat_{ex_rows[0]['category_id']}")
            ex_sets = [{"set_id":           r["set_id"],
                        "set_db_id":         r["set_id"],   # training_log._id; comment bound by id
                        "weight":            _recover_typed_weight(r["metric_weight"], offset),
                        "comment":           r.get("comment"),
                        "reps":              r["reps"],
                        "distance":          round(r.get("distance", 0) or 0, 3),
                        "duration_seconds":  int(r.get("duration_seconds", 0) or 0),
                        "is_warmup":         False} for r in ex_rows]
            eligible = warmup_eligible is None or (ex_name, training_date) in warmup_eligible
            amax     = ex_alltime_max_map.get(ex_name, 0.0) if ex_alltime_max_map else 0.0
            _detect_warmup_flags(ex_sets, exercise_name=ex_name, ctx=ctx,
                                 weight_eligible=eligible,
                                 exercise_alltime_max=amax)
            working = [s for s in ex_sets if not s["is_warmup"]] or ex_sets
            # Bar-inclusive (B1a): max_weight and estimated_1rm are headline
            # (plates + bar), matching the per-session values. bar_weight is in
            # the typed/kg-native frame already (same value the .volume line uses).
            max_w   = max(s["weight"] + bar_weight for s in working)
            vol     = sum((s["weight"] + bar_weight) * s["reps"] for s in ex_sets)
            e1rm    = max((_epley_1rm(s["weight"] + bar_weight, s["reps"]) for s in working
                           if s["reps"] > 0), default=0.0)
            if _ex_is_kg:
                total_vol_kg += vol
            else:
                total_vol_lbs += vol
            total_sets += len(working)
            exercises_done.append({
                "position": pos, "exercise_name": ex_name, "category": category,
                "unit": "kg" if _ex_is_kg else "lbs",
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
            "total_volume_lbs":  round(total_vol_lbs, 1),
            "total_volume_kg":   round(total_vol_kg, 1),
            "exercises":         exercises_done,
            "categories_trained":sorted({e["category"] for e in exercises_done}),
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
    """
    For each exercise that preceded this one in the same session, compare
    e1RM when preceded vs not preceded.

    Each entry: n per condition, mean e1rm, CI, Cohen's d, CI overlap, label.
    Sorted by abs(mean_diff_e1rm) descending — largest effects first.
    Minimum 2 sessions per condition required to include an entry.
    """
    before_map = {
        day["date"]: [ex["exercise_name"] for ex in day["exercises"]
                      if ex["position"] < next(
                          (e["position"] for e in day["exercises"]
                           if e["exercise_name"] == exercise_name), 999)]
        for day in daily_workouts
        if any(e["exercise_name"] == exercise_name for e in day["exercises"])
    }
    if not before_map:
        return []
    all_others = set(ex for exs in before_map.values() for ex in exs)
    result = []
    for other in all_others:
        preceded = [s["estimated_1rm"] for s in sessions
                    if s["date"] in before_map
                    and other in before_map[s["date"]]
                    and s["estimated_1rm"] > 0]
        not_preceded = [s["estimated_1rm"] for s in sessions
                        if (s["date"] not in before_map
                            or other not in before_map.get(s["date"], []))
                        and s["estimated_1rm"] > 0]
        if len(preceded) < 2 or len(not_preceded) < 2:
            continue
        sp   = _ci_stats(preceded)
        snp  = _ci_stats(not_preceded)
        d    = _cohen_d(preceded, not_preceded)
        mean_diff = round((sp["mean"] or 0) - (snp["mean"] or 0), 1)
        effect = ("negative" if mean_diff < -5 else
                  "positive" if mean_diff > 5  else "neutral")
        result.append({
            "preceding_exercise":       other,
            "n_preceded":               sp["n"],
            "n_not_preceded":           snp["n"],
            "mean_e1rm_when_preceded":  sp["mean"],
            "ci_95_when_preceded":      sp["ci_95"],
            "mean_e1rm_when_not":       snp["mean"],
            "ci_95_when_not":           snp["ci_95"],
            "mean_diff_e1rm":           mean_diff,
            "cohen_d":                  d,
            "cis_overlap":              _cis_overlap(sp["ci_95"], snp["ci_95"]),
            "confidence_label":         _effect_label(d, min(sp["n"], snp["n"])),
            "effect":                   effect,
        })
    # Secondary key makes tie order deterministic (result is built from set
    # iteration, which is hash-randomized across processes)
    return sorted(result,
                  key=lambda x: (-abs(x["mean_diff_e1rm"]), x["preceding_exercise"]))


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


def _compute_exercise_dow_e1rm(sessions: list) -> dict:
    """
    Group sessions by day of week, compare e1RM.
    Each day: n, mean, std, 95% CI.
    comparison carries best/worst day, Cohen's d, CI overlap, label.
    """
    by_dow: dict = defaultdict(list)
    for s in sessions:
        dow = datetime.strptime(s["date"], "%Y-%m-%d").strftime("%A")
        if s["estimated_1rm"] > 0:
            by_dow[dow].append(s["estimated_1rm"])

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    rows = []
    for d in days:
        vals = by_dow.get(d, [])
        if not vals:
            continue
        st = _ci_stats(vals)
        rows.append({
            "day":       d,
            "n":         st["n"],
            "mean_e1rm": st["mean"],
            "std_e1rm":  st["std"],
            "ci_95":     st["ci_95"],
        })

    comparison = None
    if len(rows) >= 2:
        best  = max(rows, key=lambda r: r["mean_e1rm"] or 0)
        worst = min(rows, key=lambda r: r["mean_e1rm"] or 0)
        if best["day"] != worst["day"]:
            d = _cohen_d(by_dow[best["day"]], by_dow[worst["day"]])
            comparison = {
                "best_day":         best["day"],
                "worst_day":        worst["day"],
                "mean_diff_e1rm":   round((best["mean_e1rm"] or 0) - (worst["mean_e1rm"] or 0), 1),
                "cohen_d":          d,
                "cis_overlap":      _cis_overlap(best["ci_95"], worst["ci_95"]),
                "confidence_label": _effect_label(d, min(best["n"], worst["n"])),
            }

    return {"by_day": rows, "comparison": comparison}


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


def _compute_alltime_summary(all_dates: list, alltime_rows: list,
                              pr_history: list, today, ctx: dict = None) -> dict:
    """
    All-time summary stats.

    today       — passed in explicitly (facade supplies date.today()).
    alltime_rows — all raw set rows; total_sets and total_volume are
                   computed in Python to avoid a second DB round-trip.
    ctx         — user context; used to bucket typed volume per unit frame.
    """
    if not all_dates: return {}
    date_objs = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in all_dates)
    max_streak = cur_streak = 1
    for i in range(1, len(date_objs)):
        if (date_objs[i] - date_objs[i-1]).days == 1:
            cur_streak += 1; max_streak = max(max_streak, cur_streak)
        else: cur_streak = 1
    cur_now = 0; dates_set = set(all_dates)
    check = today
    while check.strftime("%Y-%m-%d") in dates_set: cur_now += 1; check -= timedelta(days=1)
    max_gap = max((date_objs[i] - date_objs[i-1]).days for i in range(1, len(date_objs))) if len(date_objs) > 1 else 0
    first_dt = date_objs[0]; last_dt = date_objs[-1]
    months_total = max((last_dt - first_dt).days / 30, 1)
    # Derived from alltime_rows — same filter as the original SQL (excluded categories
    # already removed at fetch time, so no extra WHERE clause needed here).
    total_sets = len(alltime_rows)
    # Typed volume per unit frame: metric_weight × 2.2046 recovers the number
    # the user TYPED; for kg-typed exercises that number is kg, for lbs-typed
    # it is lbs. Bucket by frame — never add the two.
    vol_lbs = vol_kg = 0.0
    for r in alltime_rows:
        typed = r["metric_weight"] * 2.2046 * r["reps"]
        if ctx is not None and _is_kg_native(ctx, r["exercise_name"], r["date"]):
            vol_kg += typed
        else:
            vol_lbs += typed
    return {
        "first_training_date":   all_dates[0],
        "last_training_date":    all_dates[-1],
        "total_training_days":   len(set(all_dates)),
        "total_sets":            total_sets,
        "total_volume_raw_typed_lbs": round(vol_lbs, 0),
        "total_volume_raw_typed_kg":  round(vol_kg, 0),
        "total_volume_raw_note": (
            "sum of typed plate weight × reps per typed-unit frame; excludes "
            "bar weight and numeric offsets. _lbs is the lbs-typed frame and "
            "_kg the kg-typed frame — separate frames, never add them. "
            "Plates-only and NOT bar-inclusive."
        ),
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
    # Per typed-unit frame (lbs / kg): ratios and percentages are computed
    # within one frame only — lbs and kg contributions are never added.
    vol_by_group = {mg["muscle_group"]: (mg["total_volume_lbs"], mg["total_volume_kg"])
                    for mg in mg_summary}
    push_lbs = sum(vol_by_group.get(g, (0, 0))[0] for g in PUSH_CATEGORIES)
    pull_lbs = sum(vol_by_group.get(g, (0, 0))[0] for g in PULL_CATEGORIES)
    push_kg  = sum(vol_by_group.get(g, (0, 0))[1] for g in PUSH_CATEGORIES)
    pull_kg  = sum(vol_by_group.get(g, (0, 0))[1] for g in PULL_CATEGORIES)
    total_lbs = sum(v[0] for v in vol_by_group.values())
    total_kg  = sum(v[1] for v in vol_by_group.values())

    def _dominant(push, pull):
        if push == 0 and pull == 0: return None
        return ("push" if push > pull else
                "pull" if pull > push else "balanced")

    return {
        "push_volume_lbs":     round(push_lbs, 1),
        "pull_volume_lbs":     round(pull_lbs, 1),
        "push_volume_kg":      round(push_kg, 1),
        "pull_volume_kg":      round(pull_kg, 1),
        "push_pull_ratio_lbs": round(push_lbs/pull_lbs, 2) if pull_lbs > 0 else None,
        "push_pull_ratio_kg":  round(push_kg/pull_kg, 2)   if pull_kg  > 0 else None,
        "dominant_type_lbs":   _dominant(push_lbs, pull_lbs),
        "dominant_type_kg":    _dominant(push_kg, pull_kg),
        "total_volume_lbs":    round(total_lbs, 1),
        "total_volume_kg":     round(total_kg, 1),
        "note": ("volumes are reported per typed-unit frame; _lbs and _kg are "
                 "separate frames and must never be added together"),
        "distribution":    sorted([{"muscle_group": mg,
                                    "volume_lbs": round(v[0], 1),
                                    "volume_kg":  round(v[1], 1),
                                    "pct_of_lbs_total": round(v[0]/total_lbs*100, 1) if total_lbs else 0,
                                    "pct_of_kg_total":  round(v[1]/total_kg*100, 1)  if total_kg  else 0}
                                   for mg, v in vol_by_group.items()],
                                  key=lambda x: (x["volume_lbs"], x["volume_kg"]),
                                  reverse=True),
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
    ex_c    = [d["exercises_count"]  for d in daily_workouts]
    set_c   = [d["total_sets"]       for d in daily_workouts]
    vol_lbs = [d["total_volume_lbs"] for d in daily_workouts]
    vol_kg  = [d["total_volume_kg"]  for d in daily_workouts]
    return {
        "avg_exercises_per_session":     round(sum(ex_c)/len(ex_c),   1),
        "avg_sets_per_session":          round(sum(set_c)/len(set_c), 1),
        # Per typed-unit frame — day totals never mix lbs and kg contributions
        "avg_volume_per_session_lbs":    round(sum(vol_lbs)/len(vol_lbs), 1),
        "avg_volume_per_session_kg":     round(sum(vol_kg)/len(vol_kg), 1),
        "max_exercises_session":         max(ex_c),
        "min_exercises_session":         min(ex_c),
        "exercises_count_trend":         _trend(ex_c),
        "volume_trend_lbs":              _trend(vol_lbs) if any(vol_lbs) else "insufficient_data",
        "volume_trend_kg":               _trend(vol_kg)  if any(vol_kg)  else "insufficient_data",
    }


def _process_bodyweight(entries: list) -> dict:
    if not entries: return {"entries": [], "trend": "no_data", "current_kg": None}
    weights = [e["weight"] for e in entries]
    return {"entries": entries, "current_kg": round(weights[-1], 2),
            "trend": _trend(weights, 0.01, 0.01),
            "first_date": entries[0]["date"], "last_date": entries[-1]["date"]}


def _compute_bw_strength_correlation(sessions: list, bw_entries: list) -> dict:
    """
    Correlate bodyweight with e1RM over time.
    Adds Pearson r with 95% CI (Fisher z-transform) and confidence label.
    n < 3: no correlation computed. n < 4: r reported but CI is None.
    """
    if not sessions or not bw_entries:
        return {}
    bw_by_date = {e["date"]: e["weight"] for e in bw_entries}
    bw_sorted  = sorted(bw_by_date.keys())
    ratios = []
    for s in sessions:
        if s["estimated_1rm"] <= 0:
            continue
        nearest = min(bw_sorted, key=lambda d: abs(
            (datetime.strptime(d,            "%Y-%m-%d").date() -
             datetime.strptime(s["date"],    "%Y-%m-%d").date()).days), default=None)
        if nearest is None:
            continue
        gap = abs((datetime.strptime(nearest,   "%Y-%m-%d").date() -
                   datetime.strptime(s["date"], "%Y-%m-%d").date()).days)
        if gap <= 14:
            bw = bw_by_date[nearest]
            ratios.append({
                "date":          s["date"],
                "e1rm":          s["estimated_1rm"],
                "bodyweight_kg": round(bw, 2),
                "e1rm_to_bw":    round(s["estimated_1rm"] / bw, 3) if bw > 0 else None,
            })
    n = len(ratios)
    if n < 2:
        return {"ratios": ratios, "n": n, "trend": "insufficient_data",
                "pearson": {"r": None, "ci_95": None, "n": n},
                "confidence_label": "insufficient_data"}
    ratio_vals = [r["e1rm_to_bw"] for r in ratios if r["e1rm_to_bw"]]
    bw_vals    = [r["bodyweight_kg"] for r in ratios]
    e1rm_vals  = [r["e1rm"]          for r in ratios]
    pearson    = _pearson_r_with_ci(bw_vals, e1rm_vals)
    r_val      = pearson["r"]
    if r_val is None or n < 4:
        conf_label = "insufficient_data"
    elif abs(r_val) < 0.3:
        conf_label = "weak"
    elif abs(r_val) < 0.6:
        conf_label = "moderate"
    else:
        conf_label = "strong"
    return {
        "ratios":           ratios,
        "n":                n,
        "current_ratio":    ratios[-1]["e1rm_to_bw"] if ratios else None,
        "trend":            _trend(ratio_vals, 0.02, 0.02),
        "pearson":          pearson,
        "confidence_label": conf_label,
    }


def _process_goals(raw_goals: list, ctx: dict) -> list:
    return [{
        "exercise_name": g["exercise_name"],
        "target_weight": _recover_typed_weight(g["metric_weight"],
                                               _get_numeric_offset(ctx, g["exercise_name"])),
        "target_weight_plates_only": True,
        "target_reps":   g["reps"],
        "target_date":   g["target_date"],
        "unit":          "kg" if _is_kg_native(ctx, g["exercise_name"]) else "lbs",
        "notes":         g["notes"],
    } for g in raw_goals]


def _compute_muscle_group_summary(exercise_results: list) -> list:
    # Volumes are kept per typed-unit frame (lbs / kg) — a group mixing
    # lbs-typed exercises with kg-typed sessions (e.g. Back with post-switch
    # Deadlift) must never add raw kg numbers onto an lbs sum.
    by_g: dict = defaultdict(lambda: {
        "exercise_count": 0, "total_sets": 0,
        "total_volume_lbs": 0.0, "total_volume_kg": 0.0,
        "weekly_volumes": defaultdict(lambda: [0.0, 0.0]),   # [lbs, kg]
        "strength_sets": 0, "hypertrophy_sets": 0, "endurance_sets": 0, "pain_sessions": 0,
    })
    for ex in exercise_results:
        g = ex["category"]; by_g[g]["exercise_count"] += 1
        for s in ex.get("sessions", []):
            _is_kg = s.get("unit") == "kg"
            by_g[g]["total_sets"]      += s["working_sets_count"]
            by_g[g]["total_volume_kg" if _is_kg else "total_volume_lbs"] \
                += s["total_volume"]
            by_g[g]["pain_sessions"]   += 1 if s["has_pain_flag"] else 0
            by_g[g]["strength_sets"]   += s["rep_ranges"]["strength_sets"]
            by_g[g]["hypertrophy_sets"]+= s["rep_ranges"]["hypertrophy_sets"]
            by_g[g]["endurance_sets"]  += s["rep_ranges"]["endurance_sets"]
            by_g[g]["weekly_volumes"][_iso_week_key(s["date"])][1 if _is_kg else 0] \
                += s["total_volume"]
    summary = []
    for group, d in sorted(by_g.items()):
        weekly = sorted([{"week": w, "volume_lbs": round(v[0], 1),
                          "volume_kg": round(v[1], 1)}
                         for w, v in d["weekly_volumes"].items()],
                        key=lambda x: x["week"])
        lbs_series = [w["volume_lbs"] for w in weekly]
        kg_series  = [w["volume_kg"]  for w in weekly]
        total  = d["strength_sets"] + d["hypertrophy_sets"] + d["endurance_sets"]
        summary.append({
            "muscle_group":   group,
            "exercise_count": d["exercise_count"],
            "total_sets":     d["total_sets"],
            "total_volume_lbs": round(d["total_volume_lbs"], 1),
            "total_volume_kg":  round(d["total_volume_kg"], 1),
            "weekly_volumes": weekly,
            "trend_lbs":      _trend(lbs_series) if any(lbs_series) else "insufficient_data",
            "trend_kg":       _trend(kg_series)  if any(kg_series)  else "insufficient_data",
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
    def rank_volume_per_unit():
        """
        Per-unit volume ranking. Reports volume_lbs / volume_kg per exercise
        (typed frames, never added together). Ordering uses a kg-equivalent
        key INTERNALLY only — the blended number is never emitted.
        Reads period_volume_* (computed before any session wipe) so long
        windows rank correctly too.
        """
        scored = [(ex["name"],
                   ex.get("period_volume_lbs") or 0.0,
                   ex.get("period_volume_kg") or 0.0)
                  for ex in exercise_results
                  if ex.get("category") not in non_strength_names
                  and (ex.get("pr") or {}).get("weight", 0) > 0]
        return [{"exercise": n, "volume_lbs": lbs, "volume_kg": kg}
                for n, lbs, kg in sorted(scored,
                                         key=lambda x: x[1] / 2.2046 + x[2],
                                         reverse=True)]
    return {
        "fastest_improving":    rank_weight_based(lambda ex: safe(ex, "progression", "weight_change_pct")),
        "most_stagnant":        rank_weight_based(lambda ex: ex.get("plateau_days")),
        "highest_volume":       rank_volume_per_unit(),
        "most_frequent":        rank(lambda ex: safe(ex, "training_frequency", "session_count")),
        "best_e1rm":            rank_weight_based(lambda ex: safe(ex, "pr", "estimated_1rm")),
        "most_pain_sessions":   rank(lambda ex: safe(ex, "pain_analysis", "pain_session_count")),
        "most_failed_attempts": rank(lambda ex: safe(ex, "pain_analysis", "failed_attempt_count")),
        "most_commented":       rank(lambda ex: sum(s["comment_count"] for s in ex.get("sessions", []))),
        "most_regressed":       rank(lambda ex: safe(ex, "progression", "regression_from_peak", "regression_pct")),
        "best_pr_velocity":     rank(lambda ex: safe(ex, "pr_velocity", "total_prs")),
    }


# ── Main processing entry point ────────────────────────────────────────────────

def process_data(
    bundle: dict,
    ctx: dict,
    start_str: str,
    end_str: str,
    today,              # datetime.date — the "now" anchor (facade passes date.today())
    query_period_days,  # Optional[int]
    muscle_groups,      # Optional[list]
    exercise_names,     # Optional[list]
    agg_level: str,
    include_phase2: bool,
) -> dict:
    """
    Pure processing stage.  No DB, no file I/O, no clock.

    bundle keys: alltime_rows, bodyweight, goals, lifecycle, pr_history,
                 training_dates
    today      : the facade's date.today() — used for current-streak and
                 goal-projection calculations to preserve original behaviour.
    """
    all_training_dates = bundle["training_dates"]
    all_bw_entries     = bundle["bodyweight"]
    raw_goals          = bundle["goals"]
    lifecycle_rows     = bundle["lifecycle"]
    pr_history         = bundle["pr_history"]
    alltime_rows       = bundle["alltime_rows"]

    # Derive period rows from the already-fetched alltime set
    all_period_rows   = [r for r in alltime_rows if start_str <= r["date"] <= end_str]
    period_bw_entries = [e for e in all_bw_entries if start_str <= e["date"] <= end_str]

    # ── Filter rows ────────────────────────────────────────────────────────────
    filtered_rows = all_period_rows
    if muscle_groups:
        cat_map     = {v: k for k, v in CATEGORY_NAMES.items()}
        allowed_ids = {cat_map[g] for g in muscle_groups if g in cat_map}
        filtered_rows = [r for r in filtered_rows if r["category_id"] in allowed_ids]
    if exercise_names:
        lower_names   = {n.lower() for n in exercise_names}
        filtered_rows = [r for r in filtered_rows if r["exercise_name"].lower() in lower_names]

    # ── Warmup eligibility and per-exercise alltime maxima ────────────────────
    # Both are derived from alltime_rows in a single pass so they are available
    # before _build_daily_workouts and _build_sessions_from_rows are called.
    #
    # warmup_eligible: (exercise_name, date) pairs where the exercise was the
    #   first performed in its category on that day (min set_id).  Only these
    #   pairs may receive a weight-based warmup flag; explicit comments win always.
    # _ex_alltime_max: per-exercise max headline weight (plates + bar) across all
    #   time — used by the 0-weight opener gate in _detect_warmup_flags.
    _first_in_cat: dict = {}   # (date, category_id) → (min_set_id, exercise_name)
    _ex_alltime_max: dict = {}  # exercise_name → float
    for r in alltime_rows:
        key = (r["date"], r["category_id"])
        sid = r["set_id"]
        if key not in _first_in_cat or sid < _first_in_cat[key][0]:
            _first_in_cat[key] = (sid, r["exercise_name"])
        _ek  = r["exercise_name"]
        _off = _get_numeric_offset(ctx, _ek)
        _bar_lbs = _get_bar_weight_lbs(ctx, _ek, r["date"])
        _bar     = _bar_lbs / 2.2046 if _is_kg_native(ctx, _ek, r["date"]) else _bar_lbs
        _hw      = _recover_typed_weight(r["metric_weight"], _off) + _bar
        if _hw > _ex_alltime_max.get(_ek, 0.0):
            _ex_alltime_max[_ek] = _hw
    warmup_eligible: frozenset = frozenset(
        (ex_name, date)
        for (date, _cat), (_sid, ex_name) in _first_in_cat.items()
    )

    # ── Daily workout view and supersets ──────────────────────────────────────
    daily_workouts    = _build_daily_workouts(filtered_rows, ctx,
                                              warmup_eligible=warmup_eligible,
                                              ex_alltime_max_map=_ex_alltime_max)
    superset_patterns = _detect_supersets(filtered_rows)

    # ── Per-exercise ───────────────────────────────────────────────────────────
    by_exercise: dict = defaultdict(list)
    for row in filtered_rows: by_exercise[row["exercise_name"]].append(row)

    # Build alltime_cache from alltime_rows (replaces second DB fetch)
    alltime_cache: dict = defaultdict(list)
    for r in alltime_rows:
        alltime_cache[r["exercise_name"]].append(r)

    exercise_results = []
    end_date_obj = datetime.strptime(end_str, "%Y-%m-%d").date()

    # Tier 1 unit pre-pass: accumulate disagreements across all exercises/sessions.
    # Both period and alltime sessions are scanned; we deduplicate at the end.
    _raw_units_log: list = []
    # Tier 2 counterbalance: unclassified support tokens for human review.
    _raw_counterbalance_log: list = []

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

        ex_alltime_max = _ex_alltime_max.get(ex_name, 0.0)
        sessions = _build_sessions_from_rows(ex_rows, ctx, ex_name,
                                              units_review_log=_raw_units_log,
                                              warmup_eligible=warmup_eligible,
                                              exercise_alltime_max=ex_alltime_max,
                                              counterbalance_review_log=_raw_counterbalance_log)
        if not sessions: continue

        pr = _safe_compute(_compute_pr, sessions, unit, default=None, label="pr_period")
        is_weight_based_ex = (pr is not None and pr.get("weight", 0) > 0)

        if is_weight_based_ex:
            progression = _safe_compute(
                _compute_progression, sessions, default={}, label="progression")
            p2_result   = _safe_compute(
                _evaluate_phase2, progression, end_date_obj,
                default=(False, 0), label="evaluate_phase2")
            phase2_triggered, plateau_days = p2_result
        else:
            progression      = None
            plateau_days     = 0
            phase2_triggered = any(s["comment_count"] > 0 for s in sessions)

        duration_progression = (_compute_duration_progression(sessions)
                                if not is_weight_based_ex else None)
        distance_progression = (_compute_distance_progression(sessions)
                                if not is_weight_based_ex else None)

        # full_comments: derived from alltime_rows (no extra DB query needed)
        full_comments = None
        if include_phase2 and phase2_triggered:
            try:
                ex_alltime = alltime_cache.get(ex_name, [])
                full_comments = [
                    {k: v for k, v in r.items()}
                    for r in sorted(
                        (r for r in ex_alltime if r.get("comment") is not None),
                        key=lambda r: (r["date"], r["set_id"])
                    )
                ]
            except Exception as e:
                logger.warning("[data_agent] full_comments fetch failed for %s: %s",
                               ex_name, e)
                full_comments = None

        # All-time sessions for learning curve
        # Pass the log so disagreements outside the query period are also captured.
        alltime_sessions = (_build_sessions_from_rows(alltime_cache[ex_name], ctx, ex_name,
                                                       units_review_log=_raw_units_log,
                                                       warmup_eligible=warmup_eligible,
                                                       exercise_alltime_max=ex_alltime_max,
                                                       counterbalance_review_log=_raw_counterbalance_log)
                             if alltime_cache.get(ex_name) else sessions)

        exercise_results.append({
            "name":           ex_name,
            "category":       category,
            "unit":           unit,
            "numeric_offset": offset,
            "bar_weight":     bar_wt_native,
            "bar_weight_unit": unit,
            "is_cardio":      is_cardio,
            "cardio_note":    cardio_note,

            # Zoom levels
            "sessions":              sessions,
            "weekly_aggregations":   _safe_compute(_aggregate_weekly,   sessions,       default=[], label="weekly_agg"),
            "monthly_aggregations":  _safe_compute(_aggregate_monthly,  sessions,       default=[], label="monthly_agg"),
            "yearly_aggregations":   _safe_compute(_aggregate_yearly,   sessions, unit, default=[], label="yearly_agg"),

            # Progression and PR
            "progression":           progression,
            "duration_progression":  duration_progression,
            "distance_progression":  distance_progression,
            "pr":                    _safe_compute(_compute_alltime_pr,  alltime_sessions, unit,                    default=None, label="pr_alltime"),
            "pr_period":             pr,
            "pr_context":            _safe_compute(_compute_pr_context,  sessions, all_training_dates, all_bw_entries, default=[],   label="pr_context"),
            "pr_velocity":           _safe_compute(_compute_pr_velocity, pr_history, ex_name,                        default={"total_prs": 0, "monthly_counts": [], "velocity_trend": "none"}, label="pr_velocity"),

            # Period volume per typed-unit frame (computed from sessions BEFORE
            # any session wipe; lbs and kg are separate frames, never added)
            "period_volume_lbs":     round(sum(s["total_volume"] for s in sessions
                                               if s.get("unit") != "kg"), 1),
            "period_volume_kg":      round(sum(s["total_volume"] for s in sessions
                                               if s.get("unit") == "kg"), 1),

            # Trends (volume trend per unit frame — a mixed lbs->kg series
            # would otherwise read as a spurious "decreasing")
            "volume_trend_lbs":      _safe_compute(_trend, [s["total_volume"]  for s in sessions if s.get("unit") != "kg"], default="insufficient_data", label="volume_trend_lbs"),
            "volume_trend_kg":       _safe_compute(_trend, [s["total_volume"]  for s in sessions if s.get("unit") == "kg"], default="insufficient_data", label="volume_trend_kg"),
            "e1rm_trend":            _safe_compute(_trend, [s["estimated_1rm"] for s in sessions if s["estimated_1rm"] > 0], 0.05, 0.05, default="insufficient_data", label="e1rm_trend"),
            "form_trend":            _safe_compute(_compute_form_trend,             sessions,       default="insufficient_data", label="form_trend"),
            "comment_keyword_trends":_safe_compute(_compute_comment_keyword_trends, sessions,       default={},                  label="comment_keyword_trends"),

            # e1RM
            "e1rm_history":          [{"date": s["date"], "estimated_1rm": s["estimated_1rm"]}
                                       for s in sessions if s["estimated_1rm"] > 0],
            "e1rm_projection":       _safe_compute(_compute_e1rm_projection, sessions, default={}, label="e1rm_projection"),

            # Analysis
            "rep_range_distribution":   _safe_compute(_compute_rep_range_distribution,     sessions,                              default={},  label="rep_range_dist"),
            "technique_variants":       _safe_compute(_compute_technique_variants,          sessions, unit,                        default=[],  label="technique_variants"),
            "pain_analysis":            _safe_compute(_compute_pain_analysis,               sessions,                              default={"pain_session_count": 0, "pain_session_dates": [], "pain_occurrences": [], "failed_attempt_count": 0, "failed_attempts": []}, label="pain_analysis"),
            "training_frequency":       _safe_compute(_compute_training_frequency,          sessions, start_str, end_str,          default={},  label="training_frequency"),
            "rest_performance_buckets": _safe_compute(_compute_rest_performance_buckets,    sessions,                              default={"buckets": [], "comparison": None},            label="rest_perf_buckets"),
            "consecutive_day_effect":   _safe_compute(_compute_consecutive_day_effect,      sessions, all_training_dates,          default={"by_consecutive_days": [], "comparison": None}, label="consec_day_effect"),
            "workout_position_effect":  _safe_compute(_compute_exercise_workout_position,   sessions, daily_workouts, ex_name,     default=[],  label="workout_pos_effect"),
            "inter_exercise_correlation": _safe_compute(_compute_inter_exercise_correlation, sessions, ex_name, daily_workouts,    default=[],  label="inter_ex_corr"),
            "dow_e1rm_pattern":         _safe_compute(_compute_exercise_dow_e1rm,           sessions,                              default={"by_day": [], "comparison": None},              label="dow_e1rm"),
            "bw_strength_correlation":  _safe_compute(_compute_bw_strength_correlation,     sessions, period_bw_entries,           default={},  label="bw_strength_corr"),
            "learning_curve":           _safe_compute(_compute_learning_curve,              alltime_sessions,                      default={},  label="learning_curve"),

            # Phase 2
            "plateau_days":     plateau_days,
            "phase2_triggered": phase2_triggered,
            "full_comments":    full_comments,
        })

    # ── Deduplicate units review log (period + alltime may overlap) ────────────
    _seen_log: set = set()
    units_review_log: list = []
    for entry in _raw_units_log:
        key = (entry["exercise"], entry["date"], entry["comment"])
        if key not in _seen_log:
            _seen_log.add(key)
            units_review_log.append(entry)
    if units_review_log:
        logger.warning(
            "[data_agent] units_review_log: %d disagreement(s) between comment "
            "unit tokens and curated units:\n%s",
            len(units_review_log),
            "\n".join(f"  {e['exercise']} {e['date']} comment={e['comment']!r}"
                      for e in units_review_log),
        )

    # ── Deduplicate counterbalance review log ──────────────────────────────────
    _seen_cb: set = set()
    counterbalance_review_log: list = []
    for entry in _raw_counterbalance_log:
        key = (entry["exercise"], entry["date"], entry["comment"])
        if key not in _seen_cb:
            _seen_cb.add(key)
            counterbalance_review_log.append(entry)
    if counterbalance_review_log:
        logger.warning(
            "[data_agent] counterbalance_review_log: %d unclassified "
            "support token(s):\n%s",
            len(counterbalance_review_log),
            "\n".join(f"  {e['exercise']} {e['date']} comment={e['comment']!r}"
                      for e in counterbalance_review_log),
        )

    # ── Global ─────────────────────────────────────────────────────────────────
    mg_summary = _compute_muscle_group_summary(exercise_results)

    if agg_level != "session":
        for ex in exercise_results:
            # C4: cardio sessions must never be wiped — the per-session
            # distance/duration list IS the cardio data for any window length.
            if not ex.get("is_cardio"):
                ex["sessions"] = []

    goals      = _process_goals(raw_goals, ctx)
    goal_projs = []
    for g in goals:
        ex   = next((e for e in exercise_results if e["name"] == g["exercise_name"]), None)
        sess = (ex.get("sessions") if ex and ex.get("sessions") else
                [{"max_working_weight": m["max_working_weight"],
                  "estimated_1rm": m["peak_estimated_1rm"],
                  "date": m["month"] + "-15"}
                 for m in (ex["monthly_aggregations"] if ex else [])]) if ex else []
        goal_projs.append({
            **g,
            "projection": _compute_goal_projection(sess, g, g["unit"], today, ctx) if sess else None
        })

    return {
        "query_period_days":        query_period_days,
        "query_start_date":         start_str,
        "query_end_date":           end_str,
        "aggregation_level":        agg_level,
        "total_exercises_analyzed": len(exercise_results),
        "units_review_log":         units_review_log,
        "counterbalance_review_log": counterbalance_review_log,
        "all_time_summary":         _safe_compute(_compute_alltime_summary,    all_training_dates, alltime_rows, pr_history, today, ctx, default={},  label="alltime_summary"),
        "muscle_group_summary":     mg_summary,
        "muscle_group_balance":     _safe_compute(_compute_muscle_group_balance, mg_summary,                               default={},  label="mg_balance"),
        "training_consistency":     _safe_compute(_compute_training_consistency, all_training_dates, start_str, end_str,   default={},  label="training_consistency"),
        "day_of_week_patterns":     _safe_compute(_compute_day_of_week_patterns, all_training_dates, start_str, end_str,   default={},  label="dow_patterns"),
        "seasonal_patterns":        _safe_compute(_compute_seasonal_patterns,    all_training_dates,                       default=[],  label="seasonal_patterns"),
        "daily_workouts":           daily_workouts,
        "training_density":         _safe_compute(_compute_training_density,     daily_workouts,                           default={},  label="training_density"),
        "superset_patterns":        superset_patterns,
        "exercise_lifecycle":       _safe_compute(_compute_exercise_lifecycle,   lifecycle_rows, end_str,                  default={},  label="exercise_lifecycle"),
        "rankings":                 _safe_compute(_compute_rankings,             exercise_results,                         default={},  label="rankings"),
        "bodyweight":               _safe_compute(_process_bodyweight,           period_bw_entries,                        default={"entries": [], "trend": "no_data", "current_kg": None}, label="bodyweight"),
        "goals":                    goal_projs,
        "exercises":                exercise_results,
    }


# ── Trimming for prepare_analysis_package ─────────────────────────────────────

_BROAD_DROPPED_FIELDS: frozenset = frozenset({
    "full_comments",
    "inter_exercise_correlation",
    "dow_e1rm_pattern",
    "consecutive_day_effect",
    "rest_performance_buckets",
    "e1rm_history",
    "pr_context",
})

_ALL_AGG_KEYS: tuple = (
    "weekly_aggregations",
    "monthly_aggregations",
    "yearly_aggregations",
)


def _agg_keep_key(query_period_days: Optional[int]) -> str:
    """Single aggregation array to retain for GROUP / BROAD packages."""
    if query_period_days is None or query_period_days > 730:
        return "yearly_aggregations"
    if query_period_days >= 180:
        return "monthly_aggregations"
    return "weekly_aggregations"


def _cap_full_comments(full_comments: list, recent_limit: int = 30) -> list:
    """
    GROUP scope: keep the <recent_limit> most recent entries plus all
    pain-flagged entries, deduped and in original chronological order.
    """
    if len(full_comments) <= recent_limit:
        return full_comments
    pain_indices   = frozenset(
        i for i, c in enumerate(full_comments)
        if _is_pain_comment(c.get("comment"))
    )
    recent_start   = len(full_comments) - recent_limit
    recent_indices = frozenset(range(recent_start, len(full_comments)))
    return [full_comments[i] for i in sorted(pain_indices | recent_indices)]


def trim_package(package: dict, scope: str = "focused") -> dict:
    """
    In-place trim of the full package for the Analysis Agent.
    Strips raw set arrays and bulk enumerations; keeps all analytics.
    Mirrors the post-collect() body of the original prepare_analysis_package.

    scope -- "focused" | "group" | "broad"
      focused : full detail unchanged -- all stat blocks, all agg levels.
      group   : full_comments capped (30 most recent + all pain-flagged);
                exactly one aggregation level retained.
      broad   : full_comments removed; deep-stat blocks removed;
                exactly one aggregation level retained.
    """
    package["scope"] = scope
    # ── C5: Pre-capture cardio session comments BEFORE sets are stripped ───────
    # ex.clear() in the cardio rebuild would destroy this information otherwise.
    # We read set-level comments + pain flags here, then embed them per session.
    _cardio_sess_info: dict = {}   # {ex_name: {date: {"comment": str|None, "has_pain": bool}}}
    for ex in package.get("exercises", []):
        if not ex.get("is_cardio"):
            continue
        per_sess: dict = {}
        for s in ex.get("sessions", []):
            raw_sets = s.get("sets") or []
            texts = [st["comment"] for st in raw_sets if st.get("comment")]
            has_pain = any(st.get("is_pain_flag", False) for st in raw_sets)
            per_sess[s["date"]] = {
                "comment":  "\n".join(texts) if texts else None,
                "has_pain": has_pain,
            }
        _cardio_sess_info[ex.get("name", "")] = per_sess

    for ex in package.get("exercises", []):

        # Strip per-set arrays from sessions
        for session in ex.get("sessions", []):
            session.pop("sets", None)

        if ex.get("is_cardio"):
            dp  = ex.get("distance_progression") or {}
            dup = ex.get("duration_progression") or {}
            tf  = ex.get("training_frequency") or {}
            raw_sessions = ex.get("sessions", [])
            ex_name = ex.get("name", "")
            sess_info = _cardio_sess_info.get(ex_name, {})

            # ── C6: Pre-compute pace from raw sessions (distance>0 only) ──────
            # pace_min_per_km = (duration_seconds / 60) / distance_km
            # Cycling and Dead Hang (distance always 0) get NO pace fields.
            pace_rows = [
                (round((s.get("total_duration_seconds") or 0) / 60
                       / (s.get("total_distance") or 0), 2),)
                for s in raw_sessions
                if (s.get("total_distance") or 0) > 0
                and (s.get("total_duration_seconds") or 0) > 0
            ]
            # unpack: list of (pace,) tuples
            pace_values = [p[0] for p in pace_rows]
            has_pace = bool(pace_values)

            def _build_session_dict(s: dict) -> dict:
                d_km   = s.get("total_distance") or s.get("distance_km") or 0
                dur_s  = s.get("total_duration_seconds") or s.get("duration_seconds") or 0
                sd = s.get("date", "")
                out: dict = {
                    "date":             sd,
                    "distance_km":      d_km,
                    "duration_seconds": dur_s,
                }
                # C5: carry pre-captured comment + pain flag
                si = sess_info.get(sd, {})
                if si.get("comment") is not None:
                    out["comment"]  = si["comment"]
                if si.get("has_pain"):
                    out["has_pain"] = True
                # C6: pace only when distance > 0
                if d_km > 0 and dur_s > 0:
                    out["pace_min_per_km"] = round((dur_s / 60) / d_km, 2)
                return out

            prog: dict = {
                "distance_start_km":       dp.get("distance_start_km"),
                "distance_end_km":         dp.get("distance_end_km"),
                "distance_peak_km":        dp.get("distance_peak_km"),
                "distance_peak_date":      dp.get("distance_peak_date"),
                "distance_avg_km":         dp.get("avg_distance_km"),
                "distance_total_km":       dp.get("total_distance_km"),
                "duration_start_seconds":  dup.get("duration_start_seconds"),
                "duration_end_seconds":    dup.get("duration_end_seconds"),
                "duration_peak_seconds":   dup.get("duration_peak_seconds"),
                "duration_peak_date":      dup.get("duration_peak_date"),
                "duration_change_pct":     dup.get("duration_change_pct"),
                "sessions_in_period":      dp.get("session_count") or dup.get("session_count"),
            }
            # C6: pace aggregates in the progression block (distance exercises only)
            if has_pace:
                prog["pace_start_min_per_km"] = pace_values[0]
                prog["pace_end_min_per_km"]   = pace_values[-1]
                prog["pace_best_min_per_km"]  = min(pace_values)   # lower = faster

            cardio_ex = {
                "name":     ex_name,
                "category": ex.get("category"),
                "is_cardio": True,
                "total_sessions_period": dp.get("session_count") or dup.get("session_count"),
                # C3: fixed key — _compute_learning_curve returns "total_sessions_alltime"
                # (the old code read "total_alltime_sessions", which was always None).
                "all_time_sessions": ex.get("learning_curve", {}).get("total_sessions_alltime"),
                "sessions": [_build_session_dict(s) for s in raw_sessions],
                "progression": prog,
                "last_session_date":  tf.get("last_session_date"),
                "days_since_last":    tf.get("days_since_last"),
                # C5: keep full_comments so Phase-2 comment text survives the rebuild
                "full_comments":      ex.get("full_comments"),
            }

            ex.clear()
            ex.update(cardio_ex)

            exc_name = cardio_ex.get("name", "")
            lifecycle = package.get("exercise_lifecycle", {})
            for section in lifecycle.values():
                if isinstance(section, list):
                    section[:] = [
                        e for e in section
                        if e.get("exercise_name") != exc_name
                        and e.get("name") != exc_name
                    ]
                elif isinstance(section, dict):
                    section.pop(exc_name, None)
            continue

        # inter_exercise_correlation: top-5 trim for FOCUSED/GROUP; dropped for BROAD
        if scope != "broad" and ex.get("inter_exercise_correlation"):
            ex["inter_exercise_correlation"] = \
                ex["inter_exercise_correlation"][:5]

        # bw_strength_correlation: strip raw ratios in all scopes
        bw = ex.get("bw_strength_correlation")
        if isinstance(bw, dict) and "ratios" in bw:
            bw.pop("ratios", None)

        # e1rm_history: trim to 20 most recent for FOCUSED/GROUP; dropped for BROAD
        if scope != "broad" and ex.get("e1rm_history"):
            ex["e1rm_history"] = ex["e1rm_history"][-20:]

        # full_comments: scope-based cap/remove
        if scope == "group" and ex.get("full_comments"):
            ex["full_comments"] = _cap_full_comments(ex["full_comments"])
        elif scope == "focused" and ex.get("full_comments"):
            ex["full_comments"] = ex["full_comments"][-150:]
        # (scope == "broad": full_comments dropped below with other deep-stat fields)

        # One-aggregation-level rule: GROUP and BROAD keep only one zoom level
        if scope in ("group", "broad"):
            _keep = _agg_keep_key(package.get("query_period_days"))
            for _agg in _ALL_AGG_KEYS:
                if _agg != _keep:
                    ex.pop(_agg, None)

        # BROAD: drop deep-stat blocks + full_comments
        if scope == "broad":
            for _field in _BROAD_DROPPED_FIELDS:
                ex.pop(_field, None)

    # Strip exercise_lifecycle full list — keep active/dormant/abandoned/
    # substitutions only
    lifecycle = package.get("exercise_lifecycle")
    if isinstance(lifecycle, dict):
        lifecycle.pop("all_exercises", None)

    # Strip daily_workouts — workout-order detail not needed for analysis
    package.pop("daily_workouts", None)

    try:
        size_kb = len(json.dumps(package).encode()) / 1024
        logger.debug("[prepare_analysis_package] package size: %.1f KB", size_kb)
    except Exception:
        pass

    return package
