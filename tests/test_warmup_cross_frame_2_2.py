"""
Gap #2 fix — the warmup 0-opener heaviness gate (working_max >= 0.50 * exercise_alltime_max)
must compare in ONE unit frame. Pre-fix, exercise_alltime_max was maxed from raw display numbers
(lbs for pre-switch rows, kg for post-switch) and working_max was raw too, so across a mid-history
unit switch (Deadlift lbs->kg on 2025-12-26) the 0.5x compare straddled frames and mis-gated. The
fix kg-normalizes BOTH sides (pre-pass normalizes per row before maxing; the gate normalizes
working_max via the session's frame).

  • Load-bearing: same physical situation flips the flag between the raw (buggy) and kg-normalized
    (fixed) compare — the normalization is the cause.
  • Single-frame: normalization is a no-op (both sides divided by the same constant → same ratio).
  • The warmup flag's effect on session aggregates (working_sets_count / rep_ranges) is covered
    by construction in test_data_agent_golden.py::test_G_WARMUP_AGGREGATE_synthetic.
"""

import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent.process import _detect_warmup_flags, HEAVY_FRACTION  # noqa: E402

_KG = "Deadlift"  # date-gated kg-native (kg from 2025-12-26); the only mid-switch exercise
_KG_CTX = {"unit_overrides": {"exercises_in_kg": ["Deadlift"]}}
_KG_DATE = "2026-01-01"  # >= 2025-12-26 → kg frame


def _opener_session(working_headline_kg_or_lbs: float):
    """A 3-set session: 0-plate opener (12 reps) + two heavy working sets. headline_weight is the
    bar-inclusive value in the session's display frame; weight is plates (opener = 0 plates)."""
    return [
        {"set_id": 1, "weight": 0.0,  "headline_weight": 20.0,
         "reps": 12, "comment": None, "is_warmup": False},   # empty-bar opener
        {"set_id": 2, "weight": working_headline_kg_or_lbs - 20.0,
         "headline_weight": working_headline_kg_or_lbs, "reps": 8, "comment": None, "is_warmup": False},
        {"set_id": 3, "weight": working_headline_kg_or_lbs - 20.0,
         "headline_weight": working_headline_kg_or_lbs, "reps": 8, "comment": None, "is_warmup": False},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — LOAD-BEARING: the cross-frame mis-gate flips with the fix
#     working_max = 70 kg ; all-time physical max = 250 lbs (= 113.4 kg)
# ══════════════════════════════════════════════════════════════════════════════

def test_unit_switch_0opener_flag_flips_with_normalization():
    raw_lbs_max = 250.0                 # what the OLD pre-pass emitted (raw lbs number)
    kg_max      = round(250.0 / 2.2046, 1)   # what the NEW pre-pass emits (kg) ≈ 113.4
    working_kg  = 70.0

    # numeric statement of the bug vs the fix
    assert working_kg < HEAVY_FRACTION * raw_lbs_max     # 70 < 125  → raw compare FAILS
    assert working_kg >= HEAVY_FRACTION * kg_max         # 70 >= 56.7 → kg compare PASSES

    # BUG path (legacy raw compare): no session_date, raw lbs all-time max → NOT flagged
    buggy = _opener_session(working_kg)
    _detect_warmup_flags(buggy, exercise_name=_KG, ctx=_KG_CTX,
                         weight_eligible=True, exercise_alltime_max=raw_lbs_max)
    assert buggy[0]["is_warmup"] is False                # mis-gate reproduced

    # FIXED path: kg all-time max + session_date (kg frame) → correctly flagged
    fixed = _opener_session(working_kg)
    _detect_warmup_flags(fixed, exercise_name=_KG, ctx=_KG_CTX,
                         weight_eligible=True, exercise_alltime_max=kg_max,
                         session_date=_KG_DATE)
    assert fixed[0]["is_warmup"] is True                 # normalization fixes it

    # the flag flipped — the normalization is the cause
    assert buggy[0]["is_warmup"] != fixed[0]["is_warmup"]


# ══════════════════════════════════════════════════════════════════════════════
# 2 — SINGLE-FRAME REGRESSION: normalization is a no-op (same ratio either way)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kg_native", [False, True])
@pytest.mark.parametrize("working_headline,expected", [(60.0, True), (25.0, False)])
def test_single_frame_normalization_is_noop(kg_native, working_headline, expected):
    """An lbs-only and a kg-only exercise each yield the SAME flag with the frame (normalized)
    as without it (legacy raw), because both sides divide by the same constant."""
    ex   = "KgEx" if kg_native else "LbsEx"
    ctx  = {"unit_overrides": {"exercises_in_kg": ["KgEx"]}} if kg_native else {}
    # raw display max = 100 in the exercise's own frame; kg-equivalent for the normalized call
    raw_max = 100.0
    kg_max  = 100.0 if kg_native else round(100.0 / 2.2046, 1)

    legacy = _opener_session(working_headline)
    _detect_warmup_flags(legacy, exercise_name=ex, ctx=ctx,
                         weight_eligible=True, exercise_alltime_max=raw_max)   # no session_date

    normalized = _opener_session(working_headline)
    _detect_warmup_flags(normalized, exercise_name=ex, ctx=ctx,
                         weight_eligible=True, exercise_alltime_max=kg_max,
                         session_date=_KG_DATE)

    assert legacy[0]["is_warmup"] is expected
    assert normalized[0]["is_warmup"] is expected          # identical to the raw compare


# test_masked_deadlift_unchanged — DELETED (unsound live-DB golden: working_sets_count
# drifted 2->3 as 2026-06-27 Deadlift rows raised the all-time max). It was a MASKED
# no-op anyway — this user is kg-era, so cross-frame normalization changes nothing for
# him and it could never catch the cross-frame bug it nominally guarded. The genuine
# cross-frame GATE is covered by construction in test 1 above
# (test_unit_switch_0opener_flag_flips_with_normalization); the single-effective-frame
# no-op by test 2 (test_single_frame_normalization_is_noop). NOTE (residual gap, not
# introduced by this deletion): the pre-pass that COMPUTES the kg-normalized all-time
# max from mixed-frame rows (process.py inline in process_data) has no sound test — a
# synthetic-DB collect() test on hand-built mixed-frame rows is a recommended follow-up.
