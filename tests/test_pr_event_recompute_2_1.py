"""
Bug 2.1 — sub-stage 1: recompute the PR-event count + all flag-rooted PR numbers from
full set history, replacing the untrusted is_personal_record flag.

The in-app flag encodes only "was-true-at-the-time" and massively under-counts (all-time
it saw 155 flagged sets vs 557 true PR events) because it misses "more reps at the same
top weight" beats. _pr_event_dates is the single source for total_prs_alltime,
is_pr_session, and pr_velocity — a running-max walk over working sets on the SAME basis
as _compute_alltime_pr (bar-inclusive headline weight, kg-normalized), cardio excluded.

PRIMARY tests are synthetic / known-by-construction — the GATE, never delete.
COMPLEMENT is real-DB, basis-aware, NOT the gate.

No Gemini, no server.
"""

import os
import sqlite3
import sys
from datetime import date

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import prepare_analysis_package                       # noqa: E402
from src.data_agent.process import (                                      # noqa: E402
    _pr_event_dates, _compute_pr_velocity, _compute_alltime_summary,
    _compute_alltime_pr, _build_sessions_from_rows,
)


# ── Synthetic builders (known by construction) ───────────────────────────────────

def _set(headline: float, reps: int, is_warmup: bool = False) -> dict:
    # weight == headline (no bar/offset) so _compute_alltime_pr can run on these dicts.
    return {"headline_weight": float(headline), "weight": float(headline),
            "reps": reps, "is_warmup": is_warmup}


def _sess(date: str, sets: list, unit: str = "lbs") -> dict:
    """Minimal alltime-session dict accepted by _pr_event_dates. max_working_weight /
    reps_at_max are filled so _compute_alltime_pr can run on the SAME dicts."""
    working = [st for st in sets if not st["is_warmup"]] or sets
    mww = round(max(st["headline_weight"] for st in working), 2)
    rar = max((st["reps"] for st in working
               if abs(st["headline_weight"] - mww) < 0.01), default=0)
    e1rm = max((round(st["headline_weight"] * (1 + st["reps"] / 30), 1)
                for st in working if st["reps"] > 0), default=0.0)
    return {"date": date, "unit": unit, "sets": sets,
            "max_working_weight": mww, "reps_at_max": rar, "estimated_1rm": e1rm}


def _row(set_id: int, date: str, headline_lbs: float, reps: int, is_pr: int) -> dict:
    """A raw training_log-shaped row (lbs, no bar/offset → headline == typed lbs)."""
    return {"set_id": set_id, "date": date,
            "metric_weight": headline_lbs / 2.2046, "reps": reps,
            "comment": None, "is_personal_record": is_pr}


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY 1 — flag count ≠ true PR-event count (incl. same-weight-more-reps beat)
# ══════════════════════════════════════════════════════════════════════════════

def test_pr_event_count_diverges_from_flag_incl_same_weight_more_reps():
    """One exercise where the flag (1 flagged set, on a NON-max set) ≠ the true PR-event
    count (3). The unflagged beats include a 110→110 SAME-WEIGHT-MORE-REPS event, which
    the flag-walk cannot see. The flag is independent of every recomputed number."""
    # 100x5 (event: first best) → 110x5 (event: new max, UNFLAGGED)
    # → 110x8 (event: same top weight, MORE reps, UNFLAGGED) → 90x12 (NOT an event;
    # the ONLY flagged set, on a non-max set).
    rows = [
        _row(1, "2026-01-01", 100, 5,  is_pr=0),
        _row(2, "2026-02-01", 110, 5,  is_pr=0),
        _row(3, "2026-03-01", 110, 8,  is_pr=0),
        _row(4, "2026-04-01", 90, 12,  is_pr=1),   # flagged, NON-max, NOT a PR event
    ]
    sessions = _build_sessions_from_rows(rows, ctx={}, exercise_name="Fake Cable Ex")

    dates = _pr_event_dates(sessions, sessions[0]["unit"])
    assert dates == ["2026-01-01", "2026-02-01", "2026-03-01"]   # exactly the 3 beats
    assert "2026-03-01" in dates                                 # same-weight-more-reps
    assert "2026-04-01" not in dates                             # flagged ≠ event

    flag_count = sum(r["is_personal_record"] for r in rows)
    assert flag_count == 1                                       # flag under-counts
    assert len(dates) == 3                                       # true count

    # total_prs_alltime is the recomputed count, NOT the flag count
    summary = _compute_alltime_summary(
        ["2026-01-01", "2026-04-01"], rows, today=date(2026, 4, 1),
        ctx=None, pr_event_count=len(dates))
    assert summary["total_prs_alltime"] == 3
    assert summary["total_prs_alltime"] != flag_count

    # pr_velocity recomputed from the same event dates
    assert _compute_pr_velocity(dates)["total_prs"] == 3

    # is_pr_session post-pass semantics: True iff the session date is a PR-event date —
    # independent of the flag (04-01 is flagged but False; the rest unflagged but True).
    date_set = set(dates)
    expected = {"2026-01-01": True, "2026-02-01": True,
                "2026-03-01": True, "2026-04-01": False}
    assert {s["date"]: (s["date"] in date_set) for s in sessions} == expected

    # The PR OBJECT (untouched) is still correct: heaviest weight, most reps at it.
    pr = _compute_alltime_pr(sessions, sessions[0]["unit"])
    assert pr["weight"] == 110.0 and pr["reps"] == 8


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY 2 — kg-normalized basis across a mid-history lbs→kg unit switch
# ══════════════════════════════════════════════════════════════════════════════

def test_pr_event_dates_kg_normalized_across_unit_switch():
    """A mid-history lbs→kg switch must not create phantom/missed events. 220.46 lbs and
    100 kg are the SAME load; only the third session (same load, more reps) is a beat. A
    raw (un-normalized) walk would read 100 kg as a huge drop from 220 and miscount."""
    sessions = [
        _sess("2026-01-01", [_set(220.46, 5)], unit="lbs"),   # 100.0 kg → event 1
        _sess("2026-02-01", [_set(100.0, 5)],  unit="kg"),    # 100.0 kg, same reps → no
        _sess("2026-03-01", [_set(100.0, 7)],  unit="kg"),    # 100.0 kg, more reps → ev2
    ]
    dates = _pr_event_dates(sessions, "kg")
    assert dates == ["2026-01-01", "2026-03-01"]

    # Same normalized basis as the PR object: best load == 100 kg, 7 reps.
    pr = _compute_alltime_pr(sessions, "kg")
    assert pr["reps"] == 7
    assert abs(pr["weight"] - 100.0) < 0.05


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY 3 — cardio (category Cardio) contributes ZERO PR events; bodyweight does NOT
# ══════════════════════════════════════════════════════════════════════════════

def test_cardio_excluded_from_pr_events_by_category_not_weight_zero():
    """A CARDIO exercise (excluded via is_cardio=True — category, not a weight=0 proxy)
    owns a distance/duration PR object and adds NOTHING to the strength PR-event count —
    even with rising reps that would otherwise be a more-reps PR event."""
    sessions = [
        _sess("2026-01-01", [_set(0.0, 5)]),
        _sess("2026-02-01", [_set(0.0, 8)]),   # a reps beat — but cardio is excluded
    ]
    assert _pr_event_dates(sessions, is_cardio=True) == []
    assert _compute_pr_velocity(_pr_event_dates(sessions, is_cardio=True))["total_prs"] == 0


def test_bodyweight_zero_weight_noncardio_counts_reps_progression():
    """BUG-FIX REGRESSION: a bodyweight exercise (non-cardio, weight=0 throughout) has
    legitimate reps-progression that the running-max rule must count — more reps at the
    constant 0 top weight is a PR event. The old weight=0 proxy wrongly dropped it to
    zero; with the cardio-specific guard it now counts. 0×5 (baseline) → 0×8 (more reps:
    PR event) → 0×6 (fewer reps: not an event)."""
    sessions = [
        _sess("2026-01-01", [_set(0.0, 5)]),
        _sess("2026-02-01", [_set(0.0, 8)]),
        _sess("2026-03-01", [_set(0.0, 6)]),
    ]
    dates = _pr_event_dates(sessions, is_cardio=False)
    assert dates == ["2026-01-01", "2026-02-01"]            # baseline + more-reps beat
    assert _compute_pr_velocity(dates)["total_prs"] == 2    # NON-zero (the bug fixed)


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEMENT — real DB: recompute >> flag (the flag under-counts), basis-consistent
# ══════════════════════════════════════════════════════════════════════════════

def test_realdb_recompute_far_exceeds_flag_count():
    """Real-DB sanity (NOT the gate): total_prs_alltime is the recomputed count and is
    far higher than the is_personal_record flag count (the flag misses same-weight-more-
    reps beats). No brittle exact pin — the working-set/kg basis may shift it slightly."""
    pkg = prepare_analysis_package(query_period_days=None)
    recompute = pkg["all_time_summary"]["total_prs_alltime"]

    conn = sqlite3.connect(os.environ["FITNOTES_DB_PATH"])
    flag_count = conn.execute(
        "SELECT COUNT(*) FROM training_log WHERE is_personal_record = 1").fetchone()[0]
    conn.close()

    assert recompute > flag_count                       # flag under-counts
    assert recompute > 2 * flag_count                   # massively (diagnosis: 155 vs 557)


def test_realdb_total_equals_per_exercise_event_sum():
    """Internal identity: in an UNFILTERED package every exercise is in scope, so
    total_prs_alltime == the sum of per-exercise pr_velocity.total_prs (both are the
    _pr_event_dates count). Pins that no surface drifted from the single source."""
    pkg = prepare_analysis_package(query_period_days=None)
    recompute = pkg["all_time_summary"]["total_prs_alltime"]
    # Cardio exercises are rebuilt by trim_package and carry no pr_velocity; they
    # contribute 0 PR events anyway (weight=0 → excluded), so default to 0.
    per_ex_sum = sum(e.get("pr_velocity", {}).get("total_prs", 0)
                     for e in pkg["exercises"])
    assert recompute == per_ex_sum
