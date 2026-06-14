"""
Golden-case test suite for data_agent.py — spec section "Golden cases".
Tests against the PUBLIC API only (collect, prepare_analysis_package, query).
All expected values pinned to the 2026-05-28 backup snapshot.
Re-pin against the current DB before trusting these if the DB changes.

xfail tests document known bugs / unimplemented features.
strict=True means an unexpectedly-passing xfail is an error (signals a fix landed).
"""

import os
import sys
import pytest

# Ensure project root is on sys.path so `from src.data_agent import ...` works
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Point at the real DB and user context before importing the module
os.environ.setdefault("FITNOTES_DB_PATH",   "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH",  "data/user_context.json")

from src.data_agent import collect, prepare_analysis_package, query  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────

def _ex(data: dict, name: str) -> dict:
    return next(e for e in data["exercises"] if e["name"] == name)


def _sess(ex: dict, date: str) -> dict:
    return next(s for s in ex["sessions"] if s["date"] == date)


# ══════════════════════════════════════════════════════════════════════════════
# G-PR1 · Lat Pulldown all-time PR — 130.0 lbs × 9, 2026-05-18
# spec invariants: B1, B2, B2a, B5
# ══════════════════════════════════════════════════════════════════════════════

def test_G_PR1_lat_pulldown_alltime_pr():
    """
    Lat Pulldown period PR (full-history scan) must equal 130.0 lbs × 9 on 2026-05-18
    and the PR set must carry the comment 'First 3 below the neck …'.
    Pinned to 2026-05-28 snapshot.
    """
    # pr_period uses the session scan (full history), not is_personal_record.
    # With query_period_days=None (all-time) the period and all-time histories agree.
    data = collect(query_period_days=None, exercise_names=["Lat Pulldown"])
    ex = _ex(data, "Lat Pulldown")

    pr = ex["pr_period"]
    assert pr is not None
    assert pr["weight"] == pytest.approx(130.0)
    assert pr["reps"] == 9
    assert pr["date"] == "2026-05-18"
    assert pr["unit"] == "lbs"

    # B2a: the PR set carries a comment — verify via session-level data
    day = collect(
        start_date_str="2026-05-18", end_date_str="2026-05-18",
        exercise_names=["Lat Pulldown"],
    )
    ex_day = _ex(day, "Lat Pulldown")
    sess = ex_day["sessions"][0]
    pr_set = next(
        s for s in sess["sets"]
        if pytest.approx(s["weight"]) == 130.0 and s["reps"] == 9
    )
    assert pr_set["comment"] is not None
    # Pinned comment text (DB stores \n as separator, spec shows /):
    assert "First 3 below the neck" in pr_set["comment"]


# ══════════════════════════════════════════════════════════════════════════════
# G-PR2 · Deadlift all-time PR — 85.0 kg × 5, 2026-04-20  (65 kg plates + 20 kg bar)
# spec invariants: B1a, B2, B2a, B7
# XFAIL B1a: code currently returns 65 kg (plates only, bar excluded)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_PR2_deadlift_alltime_pr_includes_bar():
    """
    Deadlift post-2025-12-26 all-time PR: 85.0 kg × 5 on 2026-04-20
    (65 kg plates + 20 kg Olympic bar).  Comment 'Saw my back curl…' present.
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(query_period_days=None, exercise_names=["Deadlift"])
    ex = _ex(data, "Deadlift")

    pr = ex["pr"]           # all-time PR (flag-based; flag matches correct set here)
    assert pr is not None
    assert pr["unit"] == "kg"
    assert pr["date"] == "2026-04-20"
    assert pr["reps"] == 5
    # B1a: 65 kg plates + 20 kg bar = 85 kg headline weight
    assert pr["weight"] == pytest.approx(85.0), (
        f"Expected 85.0 kg (incl. bar), got {pr['weight']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-PR3 · Lat Pulldown PR from full history vs app flag
# spec invariants: B2, B8
# XFAIL B2/B8: _compute_alltime_pr uses is_personal_record rows (flag-based path);
#              the all-time PR object must also carry the set comment (B2a) which
#              the flag-based path never fetches.
# ══════════════════════════════════════════════════════════════════════════════

def test_G_PR3_pr_from_full_history_not_flag():
    """
    Lat Pulldown all-time PR: 130.0 lbs × 9 on 2026-05-18.
    The flag also marks 115×12 (2026-03-16) and 60×15 (2024-07-20).
    The full-history result (130×9) must not be overridden by those older marks,
    and the all-time PR object must carry the set comment (B2a).
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(query_period_days=None, exercise_names=["Lat Pulldown"])
    ex = _ex(data, "Lat Pulldown")

    pr = ex["pr"]   # all-time PR (currently flag-based)
    assert pr is not None
    # Weight and reps are correct even with flag path (flag marks 130×9)
    assert pr["weight"] == pytest.approx(130.0)
    assert pr["reps"] == 9
    # B2a: the all-time PR object must carry the comment from its set.
    # This requires the full-history path (with comment JOIN), not the flag path.
    assert "comment" in pr, "B2a: all-time PR must carry comment from its set"
    assert pr["comment"] is not None
    assert "First 3 below the neck" in pr["comment"]


# ══════════════════════════════════════════════════════════════════════════════
# G-MWE · Machine Wrist Extension — stored 9.072 → 25.0 kg (offset+5), no bar
# spec invariants: A2, B1
# ══════════════════════════════════════════════════════════════════════════════

def test_G_MWE_machine_wrist_extension_offset_unit():
    """
    MWE: stored metric_weight 9.072 → displayed 25.0 kg
    (formula: 9.072 × 2.2046 + 5 = 25.0).  Unit must be 'kg'; bar_weight = 0.
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(query_period_days=None, exercise_names=["Machine Wrist Extension"])
    ex = _ex(data, "Machine Wrist Extension")

    # A2: kg-native exercise
    assert ex["unit"] == "kg"
    # B1: no bar for this machine exercise
    assert ex["bar_weight"] == pytest.approx(0.0)
    # B1: offset=5 applied — all-time max typed weight = 25.0 kg
    pr = ex["pr_period"]
    assert pr is not None
    assert pr["weight"] == pytest.approx(25.0)
    assert pr["unit"] == "kg"


# ══════════════════════════════════════════════════════════════════════════════
# G-WALK · Walking all-time sessions — 78
# spec invariant: C3
# ══════════════════════════════════════════════════════════════════════════════

def test_G_WALK_walking_alltime_sessions():
    """
    Walking all-time session count: 79 distinct training days.
    Pinned to 2026-06-11 snapshot.
    """
    data = collect(query_period_days=None, exercise_names=["Walking"])
    ex = _ex(data, "Walking")

    # C3: total_sessions_alltime must be non-null and correct
    lc = ex.get("learning_curve", {})
    assert lc.get("total_sessions_alltime") == 79


# ══════════════════════════════════════════════════════════════════════════════
# G-TREAD · Treadmill 365-day window — sessions non-empty, 0.6 km, comment
# spec invariants: C1, C4, C5
# Fixed in this task: C4 (cardio sessions now skip the agg-level wipe),
#   C3 (total_sessions_alltime key typo corrected), C5 (comment carried through).
# ══════════════════════════════════════════════════════════════════════════════

def test_G_TREAD_treadmill_365d_sessions_distance_comment():
    """
    Treadmill within 365 days ending 2026-05-28:
      • sessions list is non-empty  (C4/H3)
      • 2025-06-26 session has distance_km ≈ 0.6  (C1/H3)
      • that session carries the comment 'kidney started paining'  (C5/H1)
    Pinned to 2026-05-28 snapshot.
    """
    pkg = prepare_analysis_package(
        query_period_days=365,
        end_date_str="2026-05-28",
        exercise_names=["Treadmill"],
    )
    treadmill = next(
        (e for e in pkg.get("exercises", []) if e.get("name") == "Treadmill"),
        None,
    )
    assert treadmill is not None

    # C4 / H3: sessions must survive the weekly-aggregation rebuild
    sessions = treadmill.get("sessions", [])
    assert len(sessions) > 0, "H3: sessions wiped to [] by weekly aggregation for cardio"

    # C1 / H3: the 0.6 km session must be present with its distance
    kidney_sess = next(
        (s for s in sessions if s.get("date") == "2025-06-26"), None
    )
    assert kidney_sess is not None, "H3: 2025-06-26 session missing after cardio rebuild"
    assert kidney_sess.get("distance_km") == pytest.approx(0.6), \
        "H3: distance_km null or wrong for 2025-06-26 session"

    # C5 / H1: comment must not be stripped by ex.clear() rebuild
    assert "kidney started paining" in (kidney_sess.get("comment") or ""), \
        "H1: comment dropped by ex.clear() + cardio_ex rebuild"


# ══════════════════════════════════════════════════════════════════════════════
# G-CARDIO0 · Cycling / Dead Hang — duration progression, no distance, no pace
# spec invariants: C2, C6
# ══════════════════════════════════════════════════════════════════════════════

def test_G_CARDIO0_cycling_dead_hang_duration_no_pace():
    """
    Cycling and Dead Hang:
      • duration_progression present and has ≥ 2 sessions  (C2)
      • distance_progression is None (distance always 0)   (C6)
      • no 'pace' field anywhere                            (C6)
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(query_period_days=None, exercise_names=["Cycling", "Dead Hang"])

    for name in ("Cycling", "Dead Hang"):
        ex = _ex(data, name)
        # C2: duration progression present
        assert ex["duration_progression"] is not None, \
            f"{name}: duration_progression is None"
        assert ex["duration_progression"].get("session_count", 0) >= 2, \
            f"{name}: duration_progression has < 2 sessions"
        # C6: no distance progression (distance always 0)
        assert ex["distance_progression"] is None, \
            f"{name}: distance_progression should be None"
        # C6: no pace field (pace would only exist for exercises with real distance)
        assert "pace" not in ex, \
            f"{name}: 'pace' field must not exist for zero-distance exercise"


# ══════════════════════════════════════════════════════════════════════════════
# G-ALLTIME · All-time summary — first 2024-06-04, last 2026-05-25, 300 days
# spec invariants: E1, E4
# ══════════════════════════════════════════════════════════════════════════════

def test_G_ALLTIME_alltime_summary():
    """
    All-time training summary: first session 2024-06-04, last 2026-06-13,
    306 distinct training days.
    Re-pinned to 2026-06-13 snapshot (DB extended past the prior 2026-06-11 pin;
    all_time_summary is unrelated to the plateau-logic change).
    """
    data = collect(query_period_days=None)
    ats = data["all_time_summary"]

    # E1: boundary dates
    assert ats["first_training_date"] == "2024-06-04"
    assert ats["last_training_date"]  == "2026-06-13"
    # E4: distinct training day count
    assert ats["total_training_days"] == 306


# ══════════════════════════════════════════════════════════════════════════════
# G-BAR · Barbell Curl — bar weight per era included in headline weight
# spec invariant: B1a
# XFAIL B1a: max_working_weight currently contains plates only;
#            bar IS already included in total_volume but NOT in max_working_weight
# ══════════════════════════════════════════════════════════════════════════════

def test_G_BAR_barbell_curl_bar_in_headline_all_eras():
    """
    Barbell Curl max_working_weight must equal plates + bar for each bar era:
      Era 1 (≤2024-09-23):      bar = 22.05 lbs  →  20 + 22.05 = 42.05 lbs
      Era 2 (2024-09-24–2025-10-30): bar = 27.56 lbs → 10 + 27.56 = 37.56 lbs
      Era 3 (≥2025-10-31):      bar = 33.07 lbs  →  30 + 33.07 = 63.07 lbs
    Pinned to 2026-05-28 snapshot.
    """
    # Era 1 — 2024-09-12: max plate 20.0 lbs, bar 22.05 lbs
    d1 = collect(start_date_str="2024-09-12", end_date_str="2024-09-12",
                 exercise_names=["Barbell Curl"])
    e1 = _ex(d1, "Barbell Curl")
    s1 = _sess(e1, "2024-09-12")
    assert s1["max_working_weight"] == pytest.approx(20.0 + 22.05), (
        f"B1a era-1: expected 42.05 lbs, got {s1['max_working_weight']}"
    )

    # Era 2 — 2025-09-02: max plate 10.0 lbs, bar 27.56 lbs
    d2 = collect(start_date_str="2025-09-02", end_date_str="2025-09-02",
                 exercise_names=["Barbell Curl"])
    e2 = _ex(d2, "Barbell Curl")
    s2 = _sess(e2, "2025-09-02")
    assert s2["max_working_weight"] == pytest.approx(10.0 + 27.56), (
        f"B1a era-2: expected 37.56 lbs, got {s2['max_working_weight']}"
    )

    # Era 3 — 2026-04-30: max plate 30.0 lbs, bar 33.07 lbs
    d3 = collect(start_date_str="2026-04-30", end_date_str="2026-04-30",
                 exercise_names=["Barbell Curl"])
    e3 = _ex(d3, "Barbell Curl")
    s3 = _sess(e3, "2026-04-30")
    assert s3["max_working_weight"] == pytest.approx(30.0 + 33.07), (
        f"B1a era-3: expected 63.07 lbs, got {s3['max_working_weight']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-WARMUP · Flat Dumbbell Bench Press — Tier 1 warmup pre-pass
# spec invariant: D3 (Tier 1)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_WARMUP_flat_db_bench_comment_seeded_profile():
    """
    Flat Dumbbell Bench Press:
      2025-02-10 — sets 30×12 / 35×8 / 40×3; first set has comment 'Done first
        as a warmup' → explicit warmup comment → is_warmup=True.
        NOTE: On 2025-02-10, Incline DB Bench (set_id 6612) preceded Flat DB
        Bench (set_id 6614) in cat=4, so Flat DB Bench is NOT category-eligible.
        The explicit comment overrides the category-first-exercise gate.
      2025-02-17 — sets 30×12 / 35×11 / 40×5; no comment on any set.
        Flat DB Bench IS first in cat=4 on this day (set_id 6433 is the minimum).
        → smooth ramp (+5 at each step): gap(30→35)=5 is NOT > 1.5×max_step(5)=7.5
        → gap rule does NOT fire → is_warmup=False (not a warmup, just a ramp).
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2025-02-10", end_date_str="2025-02-17",
        exercise_names=["Flat Dumbbell Bench Press"],
    )
    ex = _ex(data, "Flat Dumbbell Bench Press")

    # 2025-02-10: explicit warmup comment → must be flagged
    sess_0210 = _sess(ex, "2025-02-10")
    warmup_0210 = next(
        s for s in sess_0210["sets"] if pytest.approx(s["weight"]) == 30.0
    )
    assert warmup_0210["is_warmup"], \
        "Explicit 'Done first as a warmup' comment must flag the 30-lb set"

    # 2025-02-17: smooth 30→35→40 ramp — gap rule must NOT fire
    sess_0217 = _sess(ex, "2025-02-17")
    first_0217 = next(
        s for s in sess_0217["sets"] if pytest.approx(s["weight"]) == 30.0
    )
    assert not first_0217["is_warmup"], (
        "D3: 30/35/40 ramp on 2025-02-17 is a progression, not a warmup — "
        "gap(5) must not exceed 1.5 × max_step(5)=7.5"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-WARMUP-RAMP · Dumbbell Squats 2024-06-04 — smooth ramp must not be flagged
# spec invariant: D3 (negative case)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_WARMUP_RAMP_dumbbell_squats_smooth_ramp_not_flagged():
    """
    Dumbbell Squats 2024-06-04: sets 10×15 / 15×15 / 20×15 — equal step ramp.
    gap(10→15)=5, max_step(15→20)=5 → 5 is NOT > 1.5×5=7.5 → NO warmup.
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2024-06-04", end_date_str="2024-06-04",
        exercise_names=["Dumbbell Squats"],
    )
    ex = _ex(data, "Dumbbell Squats")
    sess = _sess(ex, "2024-06-04")

    assert not any(s["is_warmup"] for s in sess["sets"]), (
        "D3 negative: a smooth 10/15/20 ramp must not produce any warmup flag"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-ZERO-POS · 0-weight opener IS a warmup when working sets are near history max
# spec invariant: D3 (Tier 1 — 0-weight opener positive)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_ZERO_POS_sumo_squats_0_opener_heavy_session_is_warmup():
    """
    Sumo Squats 2026-04-27: sets 0×12 / 45×12 / 70×12 / 100×12.
    All-time max = 100 lbs.  Working max = 100 lbs.
    100 >= HEAVY_FRACTION(0.5) × 100 = 50 → first set IS a warmup.
    Exercise is first in its Legs category on that day (category gate: eligible).
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2026-04-27", end_date_str="2026-04-27",
        exercise_names=["Sumo Squats"],
    )
    ex   = _ex(data, "Sumo Squats")
    sess = _sess(ex, "2026-04-27")
    opener = next(s for s in sess["sets"] if pytest.approx(s["weight"]) == 0.0)
    assert opener["is_warmup"], (
        "D3: bodyweight (0-lb) opener before heavy Sumo Squats must be a warmup "
        "when working max (100) >= 0.5 × alltime max (100)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-ZERO-NEG · 0-weight opener is NOT a warmup in a light/learning-era session
# spec invariant: D3 (Tier 1 — 0-weight opener negative)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_ZERO_NEG_deadlift_0_opener_light_session_not_warmup():
    """
    Deadlift 2026-01-06: sets 0x16 / 10x12 / 30x12 (kg, post-2025-12-26 switch).
    Query end = 2026-03-31 so alltime_max = 80 kg headline (60 plates + 20 bar,
    from the 2026-03-30 session). Working plates max = 30 kg.
    30 < HEAVY_FRACTION(0.5) x 80 = 40 -> opener is NOT a warmup (light session).
    Exercise is first in its Back category on that day (category gate: eligible).
    Pinned to 2026-05-28 snapshot.
    """
    # end_date_str=2026-03-31 (84 days, agg_level=session) gives alltime_rows
    # through 2026-03-30 (Deadlift 60 plates+20 bar=80 kg), so alltime_max=80 kg.
    # Working plates max on 2026-01-06 = 30 kg; 30 < 0.5*80=40 -> NOT warmup.
    data = collect(
        start_date_str="2026-01-06", end_date_str="2026-03-31",
        exercise_names=["Deadlift"],
    )
    ex   = _ex(data, "Deadlift")
    sess = _sess(ex, "2026-01-06")
    opener = next(s for s in sess["sets"] if pytest.approx(s["weight"]) == 0.0)
    assert not opener["is_warmup"], (
        "D3: 0-kg opener before light Deadlift sets (plates 30 < 0.5*alltime_max_80=40) "
        "must NOT be a warmup — bodyweight here is a light learning-era working set"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-CATGATE · Category-first-exercise gate — later exercises in same category
#             must never receive a weight-based warmup flag
# spec invariant: D3 (negative case)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# G-CB1 · Barbell Calf Raise 2024-09-21 "One support" → effective bar 10 kg
# spec invariant: Tier 2 counterbalance
# ══════════════════════════════════════════════════════════════════════════════

def test_G_CB1_barbell_calf_raise_one_support_effective_bar():
    """
    Barbell Calf Raise 2024-09-21: set with comment 'One support'.
    Smith bar = 20 kg (44.09 lbs).  One support → effective bar = 10 kg ≈ 22.046 lbs.
    Plates = 0 on bar-only sets.  headline_weight must be ≈ 22.046 lbs
    (not 0.0 — bar missing, and not 44.09 — counterbalance not applied).
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2024-09-21", end_date_str="2024-09-21",
        exercise_names=["Barbell Calf Raise"],
    )
    ex   = _ex(data, "Barbell Calf Raise")
    sess = _sess(ex, "2024-09-21")

    one_support_sets = [
        s for s in sess["sets"]
        if s.get("comment") and "one support" in s["comment"].lower()
    ]
    assert one_support_sets, "G-CB1: no 'One support' set found on 2024-09-21"

    for s in one_support_sets:
        assert s["headline_weight"] != pytest.approx(0.0, abs=1.0), (
            "G-CB1: headline_weight is 0 — Smith bar not being applied at all"
        )
        assert s["headline_weight"] != pytest.approx(44.09, rel=0.01), (
            "G-CB1: headline_weight equals full bar — counterbalance not applied"
        )
        # Effective bar = 10 kg = 10 × 2.2046 ≈ 22.046 lbs; plates = 0
        assert s["headline_weight"] == pytest.approx(10 * 2.2046, rel=0.01), (
            f"G-CB1: expected ≈ {10 * 2.2046:.3f} lbs, got {s['headline_weight']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# G-CB2 · Barbell Calf Raise 2025-02-23 — per-set proof, same session
# spec invariant: Tier 2 counterbalance
# ══════════════════════════════════════════════════════════════════════════════

def test_G_CB2_barbell_calf_raise_per_set_same_session():
    """
    Barbell Calf Raise 2025-02-23: two sets, same session.
      • 'One support' set → headline ≈ 22.046 lbs (effective bar 10 kg)
      • 'No support but did not raise fully' set → headline ≈ 44.09 lbs (full bar)
    Session max_working_weight must equal the no-support set's headline (≈ 44.09 lbs).
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2025-02-23", end_date_str="2025-02-23",
        exercise_names=["Barbell Calf Raise"],
    )
    ex   = _ex(data, "Barbell Calf Raise")
    sess = _sess(ex, "2025-02-23")

    one_sets = [
        s for s in sess["sets"]
        if s.get("comment") and "one support" in s["comment"].lower()
    ]
    no_sets = [
        s for s in sess["sets"]
        if s.get("comment") and "no support" in s["comment"].lower()
    ]
    assert one_sets, "G-CB2: no 'One support' set found on 2025-02-23"
    assert no_sets,  "G-CB2: no 'No support' set found on 2025-02-23"

    for s in one_sets:
        assert s["headline_weight"] == pytest.approx(10 * 2.2046, rel=0.01), (
            f"G-CB2: 'One support' headline expected ≈ {10 * 2.2046:.3f} lbs, "
            f"got {s['headline_weight']}"
        )
    for s in no_sets:
        assert s["headline_weight"] == pytest.approx(44.09, rel=0.01), (
            f"G-CB2: 'No support' headline expected ≈ 44.09 lbs (full bar), "
            f"got {s['headline_weight']}"
        )

    # Session max_working_weight must come from the no-support (full-bar) set
    assert sess["max_working_weight"] == pytest.approx(44.09, rel=0.01), (
        f"G-CB2: session max_working_weight must equal no-support headline "
        f"(≈ 44.09), got {sess['max_working_weight']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-CB3 · Smith Machine Press "Last one supported" — NOT a counterbalance
# spec invariant: Tier 2 counterbalance negative (past tense = spotter assist)
# ══════════════════════════════════════════════════════════════════════════════

def test_G_CB3_smith_press_last_one_supported_not_counterbalance():
    """
    Smith Machine Press 2026-02-27: comment 'Last one supported' (past tense).
    'supported' must never match the counterbalance regex — it is a spotter-assist
    rep note, not a counterbalance declaration.
    Full Smith bar (44.09 lbs) must appear in the headline weight.
    bar_in_headline = headline_weight − plates must equal ≈ 44.09 lbs.
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2026-02-27", end_date_str="2026-02-27",
        exercise_names=["Smith Machine Press"],
    )
    ex   = _ex(data, "Smith Machine Press")
    sess = _sess(ex, "2026-02-27")

    supported_sets = [
        s for s in sess["sets"]
        if s.get("comment") and "last one supported" in s["comment"].lower()
    ]
    assert supported_sets, (
        "G-CB3: no 'Last one supported' set found on 2026-02-27 "
        "for Smith Machine Press — check the DB snapshot"
    )

    for s in supported_sets:
        bar_in_headline = s["headline_weight"] - s["weight"]
        assert bar_in_headline == pytest.approx(44.09, rel=0.01), (
            f"G-CB3: 'Last one supported' (past tense) should use full bar (44.09 lbs) "
            f"but bar_in_headline = {bar_in_headline:.3f}. "
            "Counterbalance must NOT apply to past-tense 'supported'."
        )


# ══════════════════════════════════════════════════════════════════════════════
# G-SCOPE-BROAD · 365d no-filter → scope="broad"; under 500 KB; PR comment intact
# ══════════════════════════════════════════════════════════════════════════════

_BROAD_DROPPED = frozenset({
    "full_comments", "inter_exercise_correlation", "dow_e1rm_pattern",
    "consecutive_day_effect", "rest_performance_buckets",
    "e1rm_history", "pr_context",
})
_AGG_KEYS = ("weekly_aggregations", "monthly_aggregations", "yearly_aggregations")


def test_G_SCOPE_BROAD_size_and_fields():
    """
    prepare_analysis_package(365d, no filter) must:
      • scope == 'broad'
      • no dropped deep-stat keys on any non-cardio exercise
      • exactly monthly_aggregations present (365d >= 180d)
      • serialized size < 500 KB
      • Lat Pulldown pr.weight 130.0 × 9 with comment (carried on pr object,
        not via full_comments — must survive the drop)
    Pinned to 2026-05-28 snapshot.
    """
    import json as _json
    pkg = prepare_analysis_package(query_period_days=365)

    assert pkg.get("scope") == "broad", f"Expected scope='broad', got {pkg.get('scope')!r}"

    for ex in pkg.get("exercises", []):
        if ex.get("is_cardio"):
            continue
        name = ex.get("name", "?")
        for field in _BROAD_DROPPED:
            assert field not in ex, (
                f"G-SCOPE-BROAD: {name} has leaked field {field!r} in broad package"
            )
        present_agg = [k for k in _AGG_KEYS if k in ex]
        assert len(present_agg) == 1, (
            f"G-SCOPE-BROAD: {name} has {len(present_agg)} agg levels {present_agg}; "
            "expected exactly 1"
        )
        assert present_agg[0] == "monthly_aggregations", (
            f"G-SCOPE-BROAD: {name} expected monthly for 365d, got {present_agg[0]!r}"
        )

    size_kb = len(_json.dumps(pkg, default=str).encode()) / 1024
    assert size_kb < 500, (
        f"G-SCOPE-BROAD: package is {size_kb:.1f} KB, expected < 500 KB"
    )

    # PR comment must survive on the pr object (not via full_comments)
    lat_ex = next(e for e in pkg["exercises"] if e["name"] == "Lat Pulldown")
    pr = lat_ex.get("pr")
    assert pr is not None
    assert pr["weight"] == pytest.approx(130.0)
    assert pr["reps"] == 9
    assert "comment" in pr and pr["comment"] is not None, (
        "G-SCOPE-BROAD: Lat Pulldown pr.comment missing — PR comment must live on pr "
        "object, not depend on full_comments"
    )
    assert "First 3 below the neck" in pr["comment"]


# ══════════════════════════════════════════════════════════════════════════════
# G-SCOPE-FOCUSED · single-exercise 90d → scope="focused"; all blocks present
# ══════════════════════════════════════════════════════════════════════════════

def test_G_SCOPE_FOCUSED_full_detail():
    """
    prepare_analysis_package(90d, exercise_names=['Lat Pulldown']) must:
      • scope == 'focused'
      • e1rm_history key present
      • inter_exercise_correlation key present
      • all three aggregation level keys present
      • sessions non-empty (agg_level=session for 90d)
    Pinned to 2026-05-28 snapshot.
    """
    pkg = prepare_analysis_package(
        query_period_days=90,
        exercise_names=["Lat Pulldown"],
    )

    assert pkg.get("scope") == "focused", (
        f"Expected scope='focused', got {pkg.get('scope')!r}"
    )

    lat_ex = next(e for e in pkg["exercises"] if e["name"] == "Lat Pulldown")

    assert "e1rm_history" in lat_ex, (
        "G-SCOPE-FOCUSED: e1rm_history must be present in focused package"
    )
    assert "inter_exercise_correlation" in lat_ex, (
        "G-SCOPE-FOCUSED: inter_exercise_correlation must be present in focused package"
    )

    # All three aggregation levels retained in FOCUSED
    present_agg = [k for k in _AGG_KEYS if k in lat_ex]
    assert len(present_agg) == 3, (
        f"G-SCOPE-FOCUSED: expected all 3 agg levels, got {present_agg}"
    )

    assert lat_ex.get("sessions"), (
        "G-SCOPE-FOCUSED: sessions must be non-empty for 90d focused package "
        "(agg_level=session keeps session arrays)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# G-SCOPE-GROUP · muscle-group 365d → scope="group"; cap + one agg level
# ══════════════════════════════════════════════════════════════════════════════

def test_G_SCOPE_GROUP_cap_and_one_agg():
    """
    prepare_analysis_package(365d, muscle_groups=['Back']) must:
      • scope == 'group'
      • full_comments per exercise <= 30 + pain-flagged count in that list
      • exactly monthly_aggregations present (365d >= 180d)
    Pinned to 2026-05-28 snapshot.
    """
    from src.data_agent.process import _is_pain_comment as _ipc

    pkg = prepare_analysis_package(
        query_period_days=365,
        muscle_groups=["Back"],
    )

    assert pkg.get("scope") == "group", (
        f"Expected scope='group', got {pkg.get('scope')!r}"
    )

    for ex in pkg.get("exercises", []):
        if ex.get("is_cardio"):
            continue
        name = ex.get("name", "?")

        # One aggregation level only
        present_agg = [k for k in _AGG_KEYS if k in ex]
        assert len(present_agg) == 1, (
            f"G-SCOPE-GROUP: {name} has {len(present_agg)} agg levels {present_agg}"
        )
        assert present_agg[0] == "monthly_aggregations", (
            f"G-SCOPE-GROUP: {name} expected monthly for 365d, got {present_agg[0]!r}"
        )

        # full_comments cap: len <= 30 + pain_count in the capped list
        fc = ex.get("full_comments") or []
        if not fc:
            continue
        pain_count = sum(1 for c in fc if _ipc(c.get("comment")))
        assert len(fc) <= 30 + pain_count, (
            f"G-SCOPE-GROUP: {name} full_comments len={len(fc)} "
            f"> 30 + pain={pain_count} = {30 + pain_count}"
        )


def test_G_CATGATE_later_category_exercises_no_warmup():
    """
    2024-06-04 exercise order by set_id:
      cat=1 (Shoulders): Lateral Dumbbell Raise (1-3), Overhead Press (4-6),
                         Cable Face Pull (7-9)
      cat=6 (Legs):      Dumbbell Squats (10-12), Leg Press (13-15)

    Cable Face Pull and Leg Press are NOT first in their respective categories
    on this day → weight-based warmup gate blocked → no warmup regardless of
    weights.  (Both sessions happen to be flat-working-set patterns that would
    otherwise qualify under WARMUP_FLAT_RATIO.)
    Pinned to 2026-05-28 snapshot.
    """
    data = collect(
        start_date_str="2024-06-04", end_date_str="2024-06-04",
        exercise_names=["Leg Press", "Cable Face Pull"],
    )
    for ex_name in ("Leg Press", "Cable Face Pull"):
        ex   = _ex(data, ex_name)
        sess = _sess(ex, "2024-06-04")
        assert not any(s["is_warmup"] for s in sess["sets"]), (
            f"D3 negative: {ex_name} is not first-in-category on 2024-06-04 "
            "— category gate must block weight-based warmup detection"
        )


# ══════════════════════════════════════════════════════════════════════════════
# G-SCOPE-FALLBACK · nonexistent exercise_names → broad fallback + loud log
# spec: scope derived from effective package contents, not classifier intent
# ══════════════════════════════════════════════════════════════════════════════

def test_G_SCOPE_FALLBACK_nonexistent_exercise():
    """
    prepare_analysis_package with exercise_names=["Nonexistent Exercise XYZ"]
    must:
      • return scope == 'broad'  (filter matched nothing → broad fallback)
      • return a trimmed package (< 500 KB)
      • record the unresolved name in package["unresolved_exercise_names"]
      • emit a WARNING log containing the unresolved name
    """
    import json as _json2
    import logging

    log_records: list = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record.getMessage())

    da_log = logging.getLogger("src.data_agent")
    handler = _Capture()
    da_log.addHandler(handler)
    try:
        pkg = prepare_analysis_package(
            exercise_names=["Nonexistent Exercise XYZ"],
            query_period_days=90,
            include_phase2=False,
        )
    finally:
        da_log.removeHandler(handler)

    assert pkg.get("scope") == "broad", (
        f"G-SCOPE-FALLBACK: expected scope='broad', got {pkg.get('scope')!r}"
    )

    size_kb = len(_json2.dumps(pkg, default=str).encode()) / 1024
    assert size_kb < 500, (
        f"G-SCOPE-FALLBACK: package is {size_kb:.1f} KB, expected < 500 KB "
        f"(broad trim must run when filter matches nothing)"
    )

    unresolved = pkg.get("unresolved_exercise_names")
    assert unresolved, (
        "G-SCOPE-FALLBACK: 'unresolved_exercise_names' key must be present and non-empty"
    )
    assert "Nonexistent Exercise XYZ" in unresolved, (
        f"G-SCOPE-FALLBACK: expected 'Nonexistent Exercise XYZ' in unresolved list, "
        f"got {unresolved!r}"
    )

    loud_logged = any(
        "Nonexistent Exercise XYZ" in msg or "resolved to 0" in msg
        for msg in log_records
    )
    assert loud_logged, (
        f"G-SCOPE-FALLBACK: expected a WARNING log containing the unresolved name; "
        f"captured log messages: {log_records[:10]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plateau / regression / current-ability — rep-aware, cadence-scaled detection
# Locks the false-89-day-plateau / false-7.7%-regression bug out for good.
# (process.py _compute_progression / _evaluate_phase2)
# ══════════════════════════════════════════════════════════════════════════════

_LAT_KW = dict(query_period_days=400, exercise_names=["Lat Pulldown"],
               aggregation_level="session", include_phase2=True)


def _synth_session(date: str, w: float, reps: int) -> dict:
    """Minimal session dict accepted by _compute_progression (Epley e1RM)."""
    e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
    return {"date": date, "unit": "lbs", "max_working_weight": float(w),
            "reps_at_max": reps, "estimated_1rm": e}


# ── G-PLATEAU-FALSE · Lat Pulldown rising run must NOT read as a plateau ───────

def test_G_PLATEAU_FALSE_lat_pulldown():
    data = collect(**_LAT_KW)
    ex = _ex(data, "Lat Pulldown")
    p = ex["progression"]
    # The 12-session 130 run (incl. 130x9) is progression, not a plateau.
    assert p["is_plateau"] is False, p.get("plateau_note")
    assert p["plateau_since"] is None
    assert p["last_new_best_date"] == "2026-05-18"      # the 130x9 rep-gain session
    assert p["sessions_since_best"] < 5                  # below the plateau threshold
    # "current ability" reflects the established 130, NOT today's 120 back-off day
    assert p["current_weight"] == 130.0
    assert ex["plateau_days"] == 0
    assert ex["phase2_triggered"] is False


# ── G-NO-REGRESSION · today's 120 back-off day is not a regression ────────────

def test_G_NO_REGRESSION_lat_pulldown_backoff():
    data = collect(**_LAT_KW)
    p = _ex(data, "Lat Pulldown")["progression"]
    assert p["regression_from_peak"] is None             # was a false 7.7% drop
    assert p["max_weight_end"] == 120.0                  # literal last session (factual)
    assert p["current_weight"] == 130.0                  # but current ability holds at 130


# ── G-REP-PROGRESS · same-weight rep gain IS a new best ───────────────────────

def test_G_REP_PROGRESS_same_weight_more_reps():
    from src.data_agent.process import _is_new_best, _last_new_best_index
    assert _is_new_best(130.0, 9, 130.0, 5) is True      # 130x5 -> 130x9
    assert _is_new_best(130.0, 7, 130.0, 9) is False     # fewer reps, same weight
    assert _is_new_best(130.0, 9, 130.0, 9) is False     # equal is not a new best
    assert _is_new_best(135.0, 1, 130.0, 9) is True      # heavier always wins
    # Real history: the last new best is the 130x9 rep-gain session
    sess = _ex(collect(**_LAT_KW), "Lat Pulldown")["sessions"]
    idx = _last_new_best_index(sess, "lbs")
    assert sess[idx]["date"] == "2026-05-18"
    assert sess[idx]["max_working_weight"] == 130.0 and sess[idx]["reps_at_max"] == 9


# ── G-WARMUP-NOT-BEST · a light high-rep set is NOT a new best (e1RM ≠ basis) ──

def test_G_WARMUP_NOT_BEST_light_highrep():
    from src.data_agent.process import _is_new_best, _epley_1rm
    # Epley would WRONGLY crown the lighter high-rep set...
    assert _epley_1rm(20, 15) > _epley_1rm(25, 3)        # 30.0 > 27.5
    # ...but the weight->reps PR rule does not.
    assert _is_new_best(20.0, 15, 25.0, 3) is False
    # Real exercise: PR + current ability are the heavy 25x3, never a light set.
    ex = _ex(collect(query_period_days=None, aggregation_level="session"),
             "dumbbell skull crusher")
    assert ex["pr"]["weight"] == 25.0 and ex["pr"]["reps"] == 3
    assert ex["progression"]["current_weight"] == 25.0


# ── G-REAL-PLATEAU · genuine flat run (no new best, e1RM not rising) plateaus ──

def test_G_REAL_PLATEAU_synthetic():
    from src.data_agent.process import _compute_progression, _evaluate_phase2
    from datetime import datetime
    sessions = [
        _synth_session("2026-01-05", 100, 5),   # the only new best
        _synth_session("2026-01-12", 100, 5),
        _synth_session("2026-01-19",  95, 6),
        _synth_session("2026-01-26", 100, 4),
        _synth_session("2026-02-02", 100, 5),
        _synth_session("2026-02-09",  95, 5),
        _synth_session("2026-02-16", 100, 5),
    ]
    p = _compute_progression(sessions)
    assert p["is_plateau"] is True
    assert p["last_new_best_date"] == "2026-01-05"
    assert p["sessions_since_best"] == 6
    assert p["plateau_span_days"] and p["plateau_span_days"] > 0
    assert "no new best" in p["plateau_note"]
    trig, days = _evaluate_phase2(p, datetime.strptime("2026-02-16", "%Y-%m-%d").date())
    assert trig is True and days > 0


# ── G-THIN-DATA · too few sessions → refuse to opine ──────────────────────────

def test_G_THIN_DATA_too_few_sessions():
    from src.data_agent.process import _compute_progression
    sessions = [
        _synth_session("2026-01-05", 100, 5),
        _synth_session("2026-01-12", 100, 5),
        _synth_session("2026-01-19", 100, 5),
    ]  # 3 sessions < MIN_SESSIONS_FOR_TREND (4)
    p = _compute_progression(sessions)
    assert p["trend_assessable"] is False
    assert p["is_plateau"] is False
    assert p["regression_from_peak"] is None
    assert "not enough sessions" in p["plateau_note"] and "only 3" in p["plateau_note"]


# ══════════════════════════════════════════════════════════════════════════════
# Comment binding — each comment sits on its OWN set, bound by training_log._id
# (Comment.owner_id). Locks out the weight/reps re-correlation misattribution.
# ══════════════════════════════════════════════════════════════════════════════

_LATPD = dict(query_period_days=400, exercise_names=["Lat Pulldown"],
              aggregation_level="session", include_phase2=True)


def _set_by_weight(ex: dict, date: str, weight: float) -> dict:
    s = _sess(ex, date)
    return next(st for st in s["sets"] if abs(st["weight"] - weight) < 0.5)


# ── G-COMMENT-BIND · 2026-05-18: 130x9 and 115x12 hold their OWN comments ─────

def test_G_COMMENT_BIND_lat_pulldown_0518():
    ex = _ex(collect(**_LATPD), "Lat Pulldown")
    top = _set_by_weight(ex, "2026-05-18", 130.0)   # the 130x9 PR set
    mid = _set_by_weight(ex, "2026-05-18", 115.0)
    assert top["set_db_id"] == 14371
    assert top["comment"] == "First 3 below the neck\nNext 3 neck ups\nLast 2 partials"
    assert mid["set_db_id"] == 14370
    assert mid["comment"] == "First 8 below the neck\nLast 4 neck ups"
    # not swapped / cross-matched
    assert top["comment"] != mid["comment"]


# ── G-COMMENT-NULL · a real no-comment set carries comment=None ───────────────

def test_G_COMMENT_NULL_lat_pulldown_0518_warmup():
    ex = _ex(collect(**_LATPD), "Lat Pulldown")
    warm = _set_by_weight(ex, "2026-05-18", 70.0)
    assert warm["set_db_id"] == 14368
    assert warm["comment"] is None        # no neighbour's comment leaked onto it


# ── G-COMMENT-MARCH16 · different sets, the prior bug quoted the 115 as the top ─

def test_G_COMMENT_MARCH16_lat_pulldown():
    ex = _ex(collect(**_LATPD), "Lat Pulldown")
    mid = _set_by_weight(ex, "2026-03-16", 115.0)
    top = _set_by_weight(ex, "2026-03-16", 130.0)
    assert mid["set_db_id"] == 13417
    assert mid["comment"] == "First 8 below the neck\nNext 2 neck ups\nLast 2 partials"
    assert top["set_db_id"] == 13418
    assert top["comment"] == "3rd set\nFirst 2 neck ups\nLast 3 partials"
    assert mid["comment"] != top["comment"]
