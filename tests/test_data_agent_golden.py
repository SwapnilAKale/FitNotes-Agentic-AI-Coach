"""
Golden-case test suite for data_agent.py — spec section "Golden cases".
Tests against the PUBLIC API only (collect, prepare_analysis_package, query).

SOUNDNESS MODEL (two kinds of golden, no live-literal pins):
  • Fixed-window historical goldens read settled past sessions (pinned
    start/end dates) — stable under append-only data growth.
  • Live-aggregate goldens (all-time counts/volume) use RECOMPUTE-AND-RELATE:
    the expected value is recomputed in-test via independent raw SQL (sharing no
    logic with the code under test) and asserted equal to the production output,
    so both grow together and only a code regression diverges.
  • Locked logic (PR rules, warmup, plateau) is pinned with SYNTHETIC TYPE-C
    inputs whose correct answer is known by construction (no DB).
There are no xfail decorators in this file; every assertion is a live check.
"""

import os
import sqlite3
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

def _ro_conn() -> sqlite3.Connection:
    """Independent read-only connection for recompute-and-relate ground truth.
    Deliberately bypasses src.data_agent — the expected value must be derived by
    a path that shares no logic with the code under test."""
    db = os.environ["FITNOTES_DB_PATH"]
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ex(data: dict, name: str) -> dict:
    return next(e for e in data["exercises"] if e["name"] == name)


def _sess(ex: dict, date: str) -> dict:
    return next(s for s in ex["sessions"] if s["date"] == date)


# ══════════════════════════════════════════════════════════════════════════════
# PR record goldens (G-PR1/PR2/PR3) were REMOVED: they pinned an absolute real-DB
# record (Lat Pulldown 130×9, Deadlift 85 kg) that legitimately changes the moment
# a heavier set is logged — unsound as a regression gate. The locked PR RULES are
# now pinned permanently on synthetic input in the "PR LOGIC" section below.
# ══════════════════════════════════════════════════════════════════════════════


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

    # A2: kg-native exercise (config/label logic — does not drift with new data)
    assert ex["unit"] == "kg"
    # B1: no bar for this machine exercise
    assert ex["bar_weight"] == pytest.approx(0.0)
    # PR is labeled in kg (the absolute-max value is a record pin — removed; the
    # offset+kg PR LOGIC is covered synthetically in the PR LOGIC section).
    pr = ex["pr_period"]
    assert pr is not None
    assert pr["unit"] == "kg"


# ══════════════════════════════════════════════════════════════════════════════
# PR LOGIC · permanent synthetic TYPE-C tests of the locked PR rules.
# Constructed input only (no DB) — the correct answer is known by construction, so
# these never drift with new data. They replace the deleted real-DB record pins
# (G-PR1/PR2/PR3 and the record halves of G-MWE/G-PROG_CROSSFRAME/G-REP_PROGRESS/
# G-WARMUP_NOT_BEST). Each test pins ONE locked rule of _compute_pr /
# _build_sessions_from_rows. DO NOT delete after passing — these are the gate.
# ══════════════════════════════════════════════════════════════════════════════

def _pr_session(date: str, w: float, reps: int, comment=None, unit: str = "lbs") -> dict:
    """A minimal session dict accepted by _compute_pr, answer known by construction."""
    e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
    return {
        "date": date, "unit": unit,
        "max_working_weight": float(w), "reps_at_max": reps, "estimated_1rm": e,
        "sets": [{"weight": float(w), "reps": reps, "comment": comment, "is_warmup": False}],
        "comment_count": 1 if comment else 0, "has_pain_flag": False,
    }


def _pr_row(set_id: int, date: str, metric_weight: float, reps: int,
            comment=None, is_pr: int = 0) -> dict:
    """A raw training_log-shaped row accepted by _build_sessions_from_rows."""
    return {"set_id": set_id, "date": date, "metric_weight": metric_weight,
            "reps": reps, "comment": comment, "is_personal_record": is_pr}


def test_PR_LOGIC_picks_highest_weight():
    """Rule: PR is the highest working weight."""
    from src.data_agent.process import _compute_pr
    pr = _compute_pr([_pr_session("2026-01-01", 100, 5),
                      _pr_session("2026-01-08", 120, 5),
                      _pr_session("2026-01-15", 110, 5)], "lbs")
    assert pr["weight"] == pytest.approx(120.0)


def test_PR_LOGIC_weight_tie_more_reps_wins():
    """Rule: on a weight tie, the most reps wins."""
    from src.data_agent.process import _compute_pr
    pr = _compute_pr([_pr_session("2026-01-01", 120, 5),
                      _pr_session("2026-01-08", 120, 8)], "lbs")
    assert pr["weight"] == pytest.approx(120.0)
    assert pr["reps"] == 8


def test_PR_LOGIC_weight_reps_tie_most_recent_date():
    """Rule: on a weight+reps tie, the most recent date wins."""
    from src.data_agent.process import _compute_pr
    pr = _compute_pr([_pr_session("2026-01-01", 120, 5),
                      _pr_session("2026-02-01", 120, 5)], "lbs")  # ascending order
    assert pr["date"] == "2026-02-01"


def test_PR_LOGIC_pr_carries_comment():
    """Rule: the PR object carries the comment of its top working set."""
    from src.data_agent.process import _compute_pr
    pr = _compute_pr([_pr_session("2026-01-01", 100, 5, comment="warmup-ish"),
                      _pr_session("2026-01-08", 120, 5, comment="all-time best")], "lbs")
    assert pr["comment"] == "all-time best"


def test_PR_LOGIC_ignores_is_personal_record_flag():
    """Rule (locked): the in-app is_personal_record flag is untrusted. A flag set on
    a NON-max set must not steer the PR — it is recomputed from set history."""
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr
    # One session: light set carries the flag, heavier set does NOT.
    rows = [_pr_row(1, "2026-01-01", 100 / 2.2046, 12, comment="flagged light", is_pr=1),
            _pr_row(2, "2026-01-01", 140 / 2.2046, 5,  comment="true max",      is_pr=0)]
    sessions = _build_sessions_from_rows(rows, ctx={}, exercise_name="Fake Cable Ex")
    pr = _compute_pr(sessions, sessions[0]["unit"])
    assert pr["weight"] == pytest.approx(140.0, abs=0.1)   # true max, not the flagged 100
    assert pr["comment"] == "true max"


def test_PR_LOGIC_barbell_includes_bar():
    """Rule: a barbell exercise's PR is bar-inclusive (plates + bar)."""
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr
    ctx = {"bar_weights_not_included": {"exercise_bar_history": {"Fake BB": {"bar_lbs": 44.09}}}}
    rows = [_pr_row(1, "2026-01-01", 60 / 2.2046, 5)]      # 60 lbs plates
    sessions = _build_sessions_from_rows(rows, ctx=ctx, exercise_name="Fake BB")
    pr = _compute_pr(sessions, sessions[0]["unit"])
    assert pr["weight"] == pytest.approx(60.0 + 44.09, abs=0.1)   # plates + bar


def test_PR_LOGIC_kg_native_in_kg():
    """Rule: a kg-native exercise's PR is labeled and valued in kg."""
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr
    ctx = {"unit_overrides": {"exercises_in_kg": ["Fake KG Ex"]}}
    rows = [_pr_row(1, "2026-01-01", 9.072, 5)]            # typed 20.0 kg (9.072*2.2046)
    sessions = _build_sessions_from_rows(rows, ctx=ctx, exercise_name="Fake KG Ex")
    pr = _compute_pr(sessions, sessions[0]["unit"])
    assert pr["unit"] == "kg"
    assert pr["weight"] == pytest.approx(round(9.072 * 2.2046, 1), abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# PR RECOMPUTE stage 6a — rep-floor strength PR + cardio PR (computation layer)
# Locked rule B2: recomputed from set history, never the is_personal_record flag.
# Synthetic TYPE-C (known-by-construction) + one end-to-end build. No LLM.
# ══════════════════════════════════════════════════════════════════════════════

def test_PR_REPFLOOR_setlevel_within_session():
    """A '5-rep PR' is the heaviest single SET done for >=5 reps — even when it is a
    LIGHTER set inside a heavier session (140x3 + 120x8 → 5-rep PR is 120, not 140).
    A session-aggregate filter on reps_at_max (=3 here) would wrongly return None."""
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr
    rows = [_pr_row(1, "2026-01-01", 140 / 2.2046, 3, comment="heavy triple", is_pr=1),
            _pr_row(2, "2026-01-01", 120 / 2.2046, 8, comment="rep PR",       is_pr=0)]
    sessions = _build_sessions_from_rows(rows, ctx={}, exercise_name="Fake Cable Ex")
    unit = sessions[0]["unit"]
    # overall PR (no floor) is the heavy 140x3
    assert _compute_pr(sessions, unit)["weight"] == pytest.approx(140.0, abs=0.1)
    # 5-rep PR is the lighter, higher-rep set
    pr5 = _compute_pr(sessions, unit, reps_floor=5)
    assert pr5["weight"] == pytest.approx(120.0, abs=0.1)
    assert pr5["reps"] == 8
    assert pr5["comment"] == "rep PR"
    assert pr5["reps_floor"] == 5


def test_PR_REPFLOOR_crosssession_and_none():
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr, _compute_alltime_pr
    rows = [_pr_row(1, "2026-01-01", 100 / 2.2046, 10, comment="ten"),
            _pr_row(2, "2026-01-08", 140 / 2.2046, 3,  comment="triple"),
            _pr_row(3, "2026-01-15", 120 / 2.2046, 6,  comment="six")]
    sessions = _build_sessions_from_rows(rows, ctx={}, exercise_name="Fake Ex")
    unit = sessions[0]["unit"]
    # reps_floor=5 → heaviest set with reps>=5 is 120x6 (the 140x3 is excluded)
    assert _compute_pr(sessions, unit, reps_floor=5)["weight"] == pytest.approx(120.0, abs=0.1)
    assert _compute_alltime_pr(sessions, unit, reps_floor=5)["weight"] == pytest.approx(120.0, abs=0.1)
    # reps_floor higher than any set → None
    assert _compute_pr(sessions, unit, reps_floor=20) is None
    assert _compute_alltime_pr(sessions, unit, reps_floor=20) is None


def test_PR_REPFLOOR_does_not_filter_warmups():
    """DELIBERATE (6a patch): the rep-floor PR does NOT exclude is_warmup sets — unlike
    the default max-weight PR. A wrongly-flagged warmup must not drop a real working-set
    PR, and at realistic floors warmups never win anyway. Here a heavy is_warmup set is
    the heaviest at the floor → it WINS. Do not re-add a warmup filter."""
    from src.data_agent.process import _rep_floor_pr
    sessions = [{
        "date": "2026-01-01", "unit": "lbs",
        "sets": [
            {"headline_weight": 200.0, "reps": 20, "is_warmup": True,
             "comment": "flagged warmup", "estimated_1rm": 0.0},
            {"headline_weight": 100.0, "reps": 20, "is_warmup": False,
             "comment": "working", "estimated_1rm": 0.0},
        ],
    }]
    pr = _rep_floor_pr(sessions, "lbs", reps_floor=20, kg_normalize=False)
    assert pr["weight"] == 200.0                 # the warmup-flagged set is NOT excluded
    assert pr["comment"] == "flagged warmup"


def test_PR_REPFLOOR_none_is_unchanged():
    """reps_floor=None (default) must be byte-identical to the no-arg single max."""
    from src.data_agent.process import _build_sessions_from_rows, _compute_pr, _compute_alltime_pr
    rows = [_pr_row(1, "2026-01-01", 100 / 2.2046, 5),
            _pr_row(2, "2026-01-08", 120 / 2.2046, 5)]
    sessions = _build_sessions_from_rows(rows, ctx={}, exercise_name="Fake Ex")
    unit = sessions[0]["unit"]
    assert _compute_pr(sessions, unit) == _compute_pr(sessions, unit, reps_floor=None)
    assert _compute_alltime_pr(sessions, unit) == _compute_alltime_pr(sessions, unit, reps_floor=None)


# ── Cardio PR (distance/duration only — never weight/reps) ───────────────────────

def _cs(date, dist, dur, comment=None):
    """A built cardio session dict, the shape _compute_cardio_pr consumes."""
    d = {"date": date, "distance_km": dist, "duration_seconds": dur}
    if comment is not None:
        d["comment"] = comment
    return d


def test_CARDIO_PR_default_max_distance():
    from src.data_agent.process import _compute_cardio_pr
    s = [_cs("2026-01-01", 3.0, 1800, "easy"),
         _cs("2026-01-08", 5.0, 3600, "long run"),
         _cs("2026-01-15", 2.0, 1200)]
    pr = _compute_cardio_pr(s)
    assert pr["distance_km"] == 5.0
    assert pr["duration_seconds"] == 3600
    assert pr["date"] == "2026-01-08"
    assert pr["comment"] == "long run"
    assert "weight" not in pr            # validator-critical: cardio PR has no weight


def test_CARDIO_PR_lock_distance_min_duration():
    from src.data_agent.process import _compute_cardio_pr
    s = [_cs("2026-01-08", 5.0, 3600, "slow 5k"),
         _cs("2026-02-01", 5.0, 3000, "fast 5k"),
         _cs("2026-02-08", 3.0, 1500, "short")]
    pr = _compute_cardio_pr(s, lock="distance", lock_value=5)
    assert pr["duration_seconds"] == 3000    # fastest among distance>=5
    assert pr["distance_km"] == 5.0
    assert pr["comment"] == "fast 5k"


def test_CARDIO_PR_lock_duration_max_distance():
    from src.data_agent.process import _compute_cardio_pr
    s = [_cs("2026-01-08", 5.0, 3600, "long"),
         _cs("2026-02-01", 8.0, 4000, "longest"),
         _cs("2026-02-08", 2.0, 300, "tiny")]
    pr = _compute_cardio_pr(s, lock="duration", lock_value=600)
    assert pr["distance_km"] == 8.0          # farthest among duration>=600
    assert pr["comment"] == "longest"


def test_CARDIO_PR_duration_only_fallback_and_none():
    from src.data_agent.process import _compute_cardio_pr
    cyc = [_cs("2025-01-01", 0, 600, "spin"),
           _cs("2025-02-01", 0, 1200, "long spin")]
    # lock=None: no distance → fall back to max duration
    pr = _compute_cardio_pr(cyc)
    assert pr["duration_seconds"] == 1200
    assert pr["comment"] == "long spin"
    assert "weight" not in pr
    # a distance lock on duration-only data → None
    assert _compute_cardio_pr(cyc, lock="distance", lock_value=1) is None
    # empty → None
    assert _compute_cardio_pr([]) is None


def test_CARDIO_PR_endtoend_walking_carries_pr():
    """Built package: the Walking cardio block now carries a pr (distance/date, no
    weight key) and the build passes validation (no DataAgentIntegrityError)."""
    pkg = prepare_analysis_package(query_period_days=None, exercise_names=["Walking"])
    ex = next(e for e in pkg["exercises"] if e.get("name") == "Walking")
    assert ex.get("is_cardio") is True
    pr = ex.get("pr")
    assert isinstance(pr, dict)
    assert "distance_km" in pr and "date" in pr
    assert "weight" not in pr


# ══════════════════════════════════════════════════════════════════════════════
# G-WALK · Walking all-time sessions
# spec invariant: C3
# ══════════════════════════════════════════════════════════════════════════════

def test_G_WALK_walking_alltime_sessions():
    """
    Walking all-time session count.
    RECOMPUTE-AND-RELATE: ground truth is an independent COUNT(DISTINCT date) for
    Walking, not a live literal — grows with new walks, diverges only on a code bug.
    """
    data = collect(query_period_days=None, exercise_names=["Walking"])
    ex = _ex(data, "Walking")

    conn = _ro_conn()
    try:
        indep = conn.execute(
            "SELECT COUNT(DISTINCT tl.date) c FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id = e._id WHERE e.name = 'Walking'"
        ).fetchone()["c"]
    finally:
        conn.close()

    # C3: total_sessions_alltime must be non-null and equal the independent count
    lc = ex.get("learning_curve", {})
    assert lc.get("total_sessions_alltime") == indep


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
    All-time training summary boundary dates + distinct training-day count.
    RECOMPUTE-AND-RELATE: ground truth is an independent raw-SQL query, NOT a live
    literal — so new data moves both sides together and only a code regression diverges.

    SCOPE SPLIT (Bug 2.5): boundaries (first/last_training_date) are the EXCLUDED strength
    scope (cats 10/11/12 dropped, 308-basis), but total_training_days is the ATTENDANCE
    count over ALL categories (Time/Place/Neck INCLUDED, 317-basis) — a day you logged
    time/location/neck is still a training day, so it is counted even though the volume
    math excludes those categories.
    """
    data = collect(query_period_days=None)
    ats = data["all_time_summary"]

    conn = _ro_conn()
    try:
        # Excluded strength scope — drives the boundary dates.
        r = conn.execute(
            "SELECT MIN(tl.date) mn, MAX(tl.date) mx "
            "FROM training_log tl JOIN exercise e ON tl.exercise_id = e._id "
            "WHERE e.category_id NOT IN (10, 11, 12)"
        ).fetchone()
        # All-category attendance scope — drives the training-day count.
        all_days = conn.execute(
            "SELECT COUNT(DISTINCT tl.date) "
            "FROM training_log tl JOIN exercise e ON tl.exercise_id = e._id"
        ).fetchone()[0]
    finally:
        conn.close()

    # E1: boundary dates == independent MIN/MAX over the EXCLUDED (strength) scope
    assert ats["first_training_date"] == r["mn"]
    assert ats["last_training_date"]  == r["mx"]
    # E4: training-day count == independent COUNT(DISTINCT date) over ALL categories
    assert ats["total_training_days"] == all_days


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
    Deadlift 2025-12-26 (all-time): 0-plate (empty 20kg bar) x16 opener, working
    sets up to 32 kg HEADLINE (12 plates + 20 bar). All-time max headline = 85 kg.
    32 < HEAVY_FRACTION(0.5) x 85 = 42.5 -> opener is NOT a warmup (genuinely
    light session). This is a true negative in the BAR-INCLUSIVE frame (not the
    old plates-vs-headline artifact): even counting the bar, the working sets stay
    below half the all-time best. Exercise is first in its Back category that day.
    Pinned to current DB.
    """
    data = collect(query_period_days=None, exercise_names=["Deadlift"],
                   aggregation_level="session")
    ex   = _ex(data, "Deadlift")
    sess = _sess(ex, "2025-12-26")
    opener = next(s for s in sess["sets"] if pytest.approx(s["weight"]) == 0.0)
    work_head_max = max(st["headline_weight"] for st in sess["sets"]
                        if st is not opener)
    assert work_head_max == 32.0                  # headline-frame working max
    assert not opener["is_warmup"], (
        "D3: 0-kg opener before light Deadlift sets (headline 32 < 0.5*alltime_85=42.5) "
        "must NOT be a warmup — even with the bar counted, the session is light"
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
    Structure-only and snapshot-INDEPENDENT (no live-DB value pins). PR value/comment
    logic — incl. that pr.comment lives on the pr object and never depends on
    full_comments — is covered by the synthetic PR LOGIC section above
    (test_PR_LOGIC_picks_highest_weight / _weight_tie_more_reps_wins / _pr_carries_comment).
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

    # NOTE: the former Lat Pulldown pr.weight==130×9 (+comment) assertions were
    # DELETED — they were a live-DB pin that drifted (true all-time PR is now
    # 145×5 2026-06-10 as heavier sets were logged after the 2026-05-28 snapshot;
    # code correct, golden stale). PR value + the pr.comment-not-via-full_comments
    # invariant are covered by the synthetic PR LOGIC section above.


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

def _synth_session(date: str, w: float, reps: int) -> dict:
    """Minimal session dict accepted by _compute_progression (Epley e1RM)."""
    e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
    return {"date": date, "unit": "lbs", "max_working_weight": float(w),
            "reps_at_max": reps, "estimated_1rm": e}


# ── G-REP-PROGRESS · same-weight rep gain IS a new best ───────────────────────

def test_G_REP_PROGRESS_same_weight_more_reps():
    # Pure PR-rule logic (the real-history half was a one-PR-away record pin —
    # removed; _last_new_best_index on synthetic input is covered in PR LOGIC).
    from src.data_agent.process import _is_new_best
    assert _is_new_best(130.0, 9, 130.0, 5) is True      # 130x5 -> 130x9
    assert _is_new_best(130.0, 7, 130.0, 9) is False     # fewer reps, same weight
    assert _is_new_best(130.0, 9, 130.0, 9) is False     # equal is not a new best
    assert _is_new_best(135.0, 1, 130.0, 9) is True      # heavier always wins


# ── G-WARMUP-NOT-BEST · a light high-rep set is NOT a new best (e1RM ≠ basis) ──

def test_G_WARMUP_NOT_BEST_light_highrep():
    # Pure logic (the real-exercise half was a one-PR-away record pin — removed).
    from src.data_agent.process import _is_new_best, _epley_1rm
    # Epley would WRONGLY crown the lighter high-rep set...
    assert _epley_1rm(20, 15) > _epley_1rm(25, 3)        # 30.0 > 27.5
    # ...but the weight->reps PR rule does not.
    assert _is_new_best(20.0, 15, 25.0, 3) is False


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


# ══════════════════════════════════════════════════════════════════════════════
# Progression end-anchor + cross-frame % guard + back-off label
# (process.py _compute_progression — end = current ability, not the last session)
# ══════════════════════════════════════════════════════════════════════════════

# test_G_PROG_CROSSFRAME_deadlift_alltime — DELETED (unsound live-DB golden:
# weight_change_pct drifted to 337.5% as 2026-06-27 kg Deadlift rows grew the
# live data). Replaced by the synthetic by-construction test_G_PROG_CROSSFRAME_synthetic.


def test_G_NO_BACKOFF_last_session_is_new_best():
    # A clean rising run ending on its best: no false back-off flag; end == latest.
    from src.data_agent.process import _compute_progression

    def S(d, w, reps):
        e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
        return {"date": d, "unit": "lbs", "max_working_weight": float(w),
                "reps_at_max": reps, "estimated_1rm": e}

    sessions = [S("2026-01-05", 100, 5), S("2026-01-12", 105, 5),
                S("2026-01-19", 110, 5), S("2026-01-26", 115, 5)]   # last is best
    p = _compute_progression(sessions)
    assert p["latest_session_is_backoff"] is False
    assert p["current_weight"] == 115.0
    assert p["latest_session_weight"] == 115.0
    assert p["max_weight_end"] == 115.0                  # end == latest == current


# ── G-BACKOFF-TRUE · rising run ending on a deliberate back-off (synthetic) ───
# Complement to test_G_NO_BACKOFF: the most recent session is LOWER load than the
# established working max, so latest_session_is_backoff is True BY CONSTRUCTION
# (no today-relative window). Asserts the full back-off-branch union the four
# deleted live goldens used to cover: end-anchor on current ability, the back-off
# label fields, % that tracks current ability (not the back-off day), and no
# false regression / plateau.

def test_G_BACKOFF_TRUE_synthetic():
    from src.data_agent.process import _compute_progression, _evaluate_phase2
    from datetime import datetime
    sessions = [
        _synth_session("2026-01-05", 100, 5),   # start
        _synth_session("2026-01-12", 110, 5),
        _synth_session("2026-01-19", 120, 5),
        _synth_session("2026-01-26", 130, 5),    # the PR / last new best
        _synth_session("2026-02-02", 120, 6),    # deliberate back-off (last session)
    ]
    p = _compute_progression(sessions)
    # Latest session is the back-off, labeled distinct from current ability.
    assert p["latest_session_is_backoff"] is True
    assert p["latest_session_weight"] == 120.0
    assert p["latest_session_reps"] == 6
    # Current ability holds the established 130, NOT the 120 back-off day.
    assert p["current_weight"] == 130.0
    # End anchor = current ability (130), never the back-off last session (120).
    assert p["max_weight_end"] == 130.0
    # % is start->current ability (100->130 = +30%), NOT start->back-off (->120 = +20%).
    assert p["max_weight_start"] == 100.0
    assert p["weight_change"] == 30.0
    assert p["weight_change_pct"] == 30.0
    # A single lighter day is neither a regression nor a plateau.
    assert p["regression_from_peak"] is None
    assert p["is_plateau"] is False
    assert p["plateau_since"] is None
    assert p["last_new_best_date"] == "2026-01-26"       # the PR, not the back-off
    assert p["sessions_since_best"] == 1                 # < PLATEAU_MIN_SESSIONS_SINCE_BEST (5)
    # Phase 2 fires on the genuine +30% improvement, not on a plateau.
    trig, days = _evaluate_phase2(p, datetime.strptime("2026-02-02", "%Y-%m-%d").date())
    assert trig is True
    assert days == 0                                     # not a plateau -> no span


# ── G-PROG-CROSSFRAME (synthetic) · unit-switch window re-anchors the % ────────
# Synthetic by-construction replacement for the deleted live-DB
# test_G_PROG_CROSSFRAME_deadlift_alltime (it drifted to 337.5% as 2026-06-27 kg
# rows grew the live Deadlift). A window spanning a lbs->kg switch must re-anchor
# "start" into the END (kg) frame so the % is same-frame, never boundary-spanning.
# Validity: kg-era start=90, current best=100 -> +11.1% is computed BY HAND from
# the placed values, sharing no logic with _compute_progression's re-anchoring.

def test_G_PROG_CROSSFRAME_synthetic():
    from src.data_agent.process import _compute_progression

    def S(date, w, reps, unit):
        e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
        return {"date": date, "unit": unit, "max_working_weight": float(w),
                "reps_at_max": reps, "estimated_1rm": e}

    sessions = [
        S("2026-01-05", 200, 5, "lbs"),   # pre-switch (lbs era)
        S("2026-01-12", 210, 5, "lbs"),
        S("2026-01-19",  90, 5, "kg"),    # post-switch start (kg era)
        S("2026-01-26", 100, 5, "kg"),    # kg-era best == current ability
    ]
    p = _compute_progression(sessions)
    assert p["unit_switch_in_period"] is True
    assert p["progression_note"] and "unit switch" in p["progression_note"]
    # start re-anchored to the first kg-era session (90), NOT the lbs 200/210.
    assert p["max_weight_start"] == 90.0
    # % is kg-era 90 -> 100 = +11.1%, never a frame-spanning value (185.4 / 337.5).
    assert p["weight_change_pct"] == 11.1


# ══════════════════════════════════════════════════════════════════════════════
# Warmup 0-opener gate — bar-inclusive frame (process.py _detect_warmup_flags)
# The empty-bar opener's heaviness check now compares headline-vs-headline, so a
# genuine empty-bar warmup before heavy working sets is no longer missed.
# ══════════════════════════════════════════════════════════════════════════════

# test_G_WARMUP_EMPTYBAR_deadlift_2026_01_17 — DELETED (unsound live-DB golden:
# 2026-06-27 Deadlift rows raised the all-time max so the 2026-01-17 opener is no
# longer flagged). The 0-plate-opener-flag branch is covered by construction in
# test_G_WARMUP_NO_FALSE_light_session_and_preconditions (heavy case) and the
# bar-inclusive headline math by the bar tests / session_display port-equality.


def test_G_WARMUP_NO_FALSE_light_session_and_preconditions():
    # The fix must NOT create false warmups: the heaviness threshold still gates,
    # and every precondition still holds.
    from src.data_agent.process import _detect_warmup_flags

    def S(sid, plate, head, reps):
        return {"set_id": sid, "weight": plate, "headline_weight": head,
                "reps": reps, "comment": None, "is_warmup": False}

    # 0-plate opener, 12 reps, 3 sets, eligible — but working sets LIGHT
    # (headline 5 < 0.5 * alltime 100 = 50) -> NOT a warmup.
    light = [S(1, 0.0, 0.0, 12), S(2, 5.0, 5.0, 8), S(3, 5.0, 5.0, 8)]
    _detect_warmup_flags(light, exercise_name="X", weight_eligible=True,
                         exercise_alltime_max=100.0)
    assert light[0]["is_warmup"] is False

    # same opener, HEAVY working sets (headline 60 >= 50) -> flagged (gate works)
    heavy = [S(1, 0.0, 0.0, 12), S(2, 60.0, 60.0, 8), S(3, 60.0, 60.0, 8)]
    _detect_warmup_flags(heavy, exercise_name="X", weight_eligible=True,
                         exercise_alltime_max=100.0)
    assert heavy[0]["is_warmup"] is True

    # precondition: not first-in-category -> never flagged, even heavy
    not_elig = [S(1, 0.0, 0.0, 12), S(2, 60.0, 60.0, 8), S(3, 60.0, 60.0, 8)]
    _detect_warmup_flags(not_elig, exercise_name="X", weight_eligible=False,
                         exercise_alltime_max=100.0)
    assert not_elig[0]["is_warmup"] is False

    # precondition: opener < 12 reps -> not flagged
    low_reps = [S(1, 0.0, 0.0, 8), S(2, 60.0, 60.0, 8), S(3, 60.0, 60.0, 8)]
    _detect_warmup_flags(low_reps, exercise_name="X", weight_eligible=True,
                         exercise_alltime_max=100.0)
    assert low_reps[0]["is_warmup"] is False

    # precondition: < 3 sets (only 1 working set) -> not flagged
    two = [S(1, 0.0, 0.0, 12), S(2, 60.0, 60.0, 8)]
    _detect_warmup_flags(two, exercise_name="X", weight_eligible=True,
                         exercise_alltime_max=100.0)
    assert two[0]["is_warmup"] is False


# ── G-WARMUP-AGGREGATE (synthetic) · flag excludes from count, not from volume ─
# Synthetic by-construction replacement for the deleted live-DB
# test_G_WARMUP_COUNT_UNCHANGED_FIELDS_deadlift_2026_01_17 (it drifted to
# working_sets_count==3 as 2026-06-27 Deadlift rows raised the all-time max).
# Builds raw rows through _build_sessions_from_rows (the seam that both flags the
# warmup AND computes the session aggregates). Validity: 3 sets, opener flagged ->
# 2 working; the two 8-rep working sets are hypertrophy; the opener's 15 reps
# (endurance) is excluded -> endurance_sets==0; volume sums ALL sets. Every
# expected value is counted/summed BY HAND, not by re-running the aggregator.

def test_G_WARMUP_AGGREGATE_synthetic():
    from src.data_agent.process import _build_sessions_from_rows

    mw60 = 60.0 / 2.2046   # no bar/offset -> headline == round(mw*2.2046,1) == 60.0
    rows = [
        {"set_id": 1, "date": "2026-01-10", "metric_weight": 0.0,  "reps": 15,
         "comment": None, "distance": 0, "duration_seconds": 0},   # 0-plate opener
        {"set_id": 2, "date": "2026-01-10", "metric_weight": mw60, "reps": 8,
         "comment": None, "distance": 0, "duration_seconds": 0},
        {"set_id": 3, "date": "2026-01-10", "metric_weight": mw60, "reps": 8,
         "comment": None, "distance": 0, "duration_seconds": 0},
    ]
    # The 0-opener gate compares a kg-NORMALIZED working_max against a kg
    # exercise_alltime_max. SynLift is lbs, so working 60 lbs -> ~27.2 kg; pass a
    # kg all-time max of 40.0 so 27.2 >= 0.5*40 = 20 -> the opener flags by construction.
    sessions = _build_sessions_from_rows(
        rows, ctx={}, exercise_name="SynLift",
        warmup_eligible=frozenset({("SynLift", "2026-01-10")}),
        exercise_alltime_max=40.0)
    s = sessions[0]
    assert s["sets"][0]["is_warmup"] is True
    assert s["working_sets_count"] == 2                  # opener excluded from working
    assert s["rep_ranges"]["hypertrophy_sets"] == 2      # the two 8-rep working sets
    assert s["rep_ranges"]["endurance_sets"] == 0        # opener's 15 reps excluded
    assert s["rep_ranges"]["strength_sets"] == 0
    assert s["max_working_weight"] == 60.0               # from a working set, not the opener
    assert s["total_volume"] == 960.0                    # 0*15 + 60*8 + 60*8 (all sets, flag-agnostic)


# test_G_WARMUP_COUNT_UNCHANGED_FIELDS_deadlift_2026_01_17 — DELETED (unsound
# live-DB golden: working_sets_count drifted 2->3 as 2026-06-27 rows raised the
# all-time max). The warmup-flag -> session-aggregate effect (working_sets_count /
# rep_ranges exclusion; max/volume flag-agnostic) is covered by construction in
# test_G_WARMUP_AGGREGATE_synthetic.


# ── G-WARMUP-ALLTIME-MAX-CROSSFRAME (synthetic) · normalize-then-max pre-pass ──
# Closes the gap the deleted (masked) test_masked_deadlift_unchanged left open: the
# pre-pass (process.py:2556-2569, inline in process_data) computes an exercise's
# all-time max by kg-NORMALIZING each mixed-frame row BEFORE maxing. The masked test
# couldn't catch a bug here because its user's max was kg-era (normalize == no-op).
#
# This fixture makes the kg-normalized max DIFFER from the raw-numeric max: an lbs-era
# set typed 200 (raw 200, but only ~90.7 kg) vs a kg-era set typed 100 (raw 100, and
# 100 kg). Correct normalize-then-max = 100 kg; a bypass would take raw max 200.
# "Deadlift" is used because _is_kg_native date-gates ONLY Deadlift (the real signal).
#
# Assertion is by GATE EFFECT (the normalized max is a local, not emitted): a target
# kg session's 0-plate opener flags as warmup iff working_max(70 kg) >= 0.5*max. With
# the correct max 100 -> 70>=50 -> flagged (working_sets_count==2). Under a bypass
# (max 200) -> 70<100 -> NOT flagged (count==3). 70 sits in [50,100) BY CONSTRUCTION,
# so a broken/absent normalization changes the result -> the test is sound (verified
# by a temporary bypass-and-revert demonstration during development).

def test_G_WARMUP_ALLTIME_MAX_CROSSFRAME_synthetic():
    from datetime import date as _date
    from src.data_agent.process import process_data

    ctx = {"unit_overrides": {"exercises_in_kg": ["Deadlift"]}}   # no bar -> bar 0

    def R(set_id, d, typed, reps):   # metric_weight is stored as typed/2.2046
        return {"set_id": set_id, "date": d, "metric_weight": typed / 2.2046,
                "reps": reps, "distance": 0, "duration_seconds": 0,
                "exercise_name": "Deadlift", "category_id": 5, "comment": None}

    alltime_rows = [
        R(1, "2025-12-01", 200, 5),   # lbs era (< 2025-12-26): raw 200, ~90.7 kg
        R(2, "2026-01-05", 100, 5),   # kg era (>= cutoff): raw 100, 100 kg  -> the true max
        # target kg session: 0-plate opener (12 reps) + two 70 kg working sets
        R(3, "2026-01-12",   0, 12),
        R(4, "2026-01-12",  70, 8),
        R(5, "2026-01-12",  70, 8),
    ]
    bundle = {
        "alltime_rows": alltime_rows, "bodyweight": [], "goals": [], "lifecycle": [],
        "training_dates": ["2025-12-01", "2026-01-05", "2026-01-12"],
        "total_training_day_count": 3,
    }
    data = process_data(bundle, ctx, "2000-01-01", "2026-01-31",
                        _date(2026, 1, 31), None, None, ["Deadlift"], "session", False)

    s = _sess(_ex(data, "Deadlift"), "2026-01-12")
    # By construction the gate threshold differs by which max the pre-pass produced:
    assert 0.5 * 100 <= 70 < 0.5 * 200            # 50 <= 70 < 100 (the gate gap)
    assert s["sets"][0]["is_warmup"] is True       # flagged -> pre-pass used kg max 100, not raw 200
    assert s["working_sets_count"] == 2            # opener excluded; would be 3 under a raw-max bypass
