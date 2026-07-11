"""
Issue 3 Defect A (structural) — pain occurrences carry body-part locations so a
"this specific pain N times" claim can be symptom-aware. The live bug: the model
called a behind-the-knee cramp "five times" while one of the five Hamstring Curls
Machine pain comments was actually "right hip pain" (a distinct symptom).

Pure synthetic tests for the mapping + grouping; one live-DB property check that the
hip pain is tagged distinctly from the knee cramps (not conflated).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent.process import _pain_locations, _compute_pain_analysis   # noqa: E402
from src.data_agent import prepare_analysis_package                          # noqa: E402


# ── _pain_locations (pure) — the real live comments ───────────────────────────

def test_pain_locations_real_comments():
    assert _pain_locations("Last 3 halfs because of right hip pain") == ["hip"]
    assert _pain_locations("Last one cramp at the spot behind my left knee") == ["knee"]
    assert _pain_locations("Started to cramp a little at 12 so stopped") == []
    assert set(_pain_locations(
        "cramps at part between hamstrings and calfs, behind the knee")) == {
        "knee", "hamstring", "calf"}
    assert _pain_locations("Stopped because same pain but during the set") == []


def test_pain_locations_none_and_back_narrowing():
    assert _pain_locations(None) == []
    assert _pain_locations("") == []
    # bare "back" (support / going back) must NOT tag as back pain
    assert _pain_locations("leaning back on the rod for support") == []
    assert _pain_locations("very low back pain at the end") == ["back"]


# ── _compute_pain_analysis grouping (synthetic, deterministic) ────────────────

def _sess(date, comment):
    return {"date": date, "has_pain_flag": True, "failed_attempts": 0,
            "sets": [{"set_id": 1, "comment": comment,
                      "is_pain_flag": True, "is_failed_attempt": False, "weight": 50}]}


def test_compute_pain_analysis_per_location_grouping():
    sessions = [
        _sess("2024-12-19", "right hip pain"),
        _sess("2025-07-18", "cramp at the spot behind my left knee"),
        _sess("2025-12-03", "cramps between hamstrings and calfs, behind the knee"),
        _sess("2026-06-15", "same pain during the set"),
    ]
    pa = _compute_pain_analysis(sessions)
    assert pa["pain_session_count"] == 4
    # per-occurrence locations
    locs = {o["date"]: set(o["locations"]) for o in pa["pain_occurrences"]}
    assert locs["2024-12-19"] == {"hip"}
    assert locs["2025-07-18"] == {"knee"}
    assert locs["2025-12-03"] == {"knee", "hamstring", "calf"}
    assert locs["2026-06-15"] == set()                     # untagged → no symptom
    # grouped summary — knee count is 2, NOT lumped with the hip session
    pbl = pa["pain_by_location"]
    assert pbl["knee"] == ["2025-07-18", "2025-12-03"]
    assert pbl["hip"] == ["2024-12-19"]
    assert "2024-12-19" not in pbl["knee"]                 # hip never folded into knee
    assert "2026-06-15" not in [d for ds in pbl.values() for d in ds]  # untagged in none


# ── Live-DB property check — hip stays distinct from knee in full_comments ─────

def test_live_full_comments_hip_distinct_from_knee():
    # All-time HCM triggers phase2 → full_comments carries the historical pain
    # comments, each pain entry tagged. The hip-pain comment must be tagged 'hip',
    # never merged into the knee-cramp set (the conflation the fix targets).
    pkg = prepare_analysis_package(query_period_days=None,
                                   exercise_names=["Hamstring Curls Machine"],
                                   include_phase2=True)
    hc = next(e for e in pkg["exercises"] if e["name"] == "Hamstring Curls Machine")
    tagged = [c for c in (hc.get("full_comments") or []) if "pain_locations" in c]
    assert tagged, "expected pain-tagged all-time comments for HCM"
    hip_dates  = {c["date"] for c in tagged if "hip"  in c["pain_locations"]}
    knee_dates = {c["date"] for c in tagged if "knee" in c["pain_locations"]}
    assert hip_dates and knee_dates                        # both symptoms present
    assert hip_dates.isdisjoint(knee_dates)                # never the same occurrence
