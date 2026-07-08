"""
Deload-block vs regression classification (process.py _compute_progression /
_detect_deload_block).

Root bug: current_weight = max over the last CURRENT_ABILITY_SESSIONS=3
sessions collapsed onto a sustained deload block (three 50/55/60 sessions after
nine stable at 130–145), producing a false "began at 130, currently 60,
regressed 58.6% from peak" narrative. A deload is a SHARP step-down from a
STABLE level, bounded in length; a regression is a decline trend. The
classification is deterministic at the tool level (training_state /
deload_block package fields) — the LLM never infers it.

Bias (stated for THIS classifier): false positive = calling a genuine
regression a "deload" (hides real decline — the WORSE error); false negative =
calling a genuine deload a regression (false alarm — the old bug class). All
detection conditions must hold simultaneously, so ambiguity always falls
through to the regression-VISIBLE path.

All inputs synthetic and known by construction — never pinned to the live DB.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_agent.process import (          # noqa: E402
    _compute_progression,
    _detect_deload_block,
    DELOAD_MAX_SESSIONS,
)


def S(date: str, w: float, reps: int = 5, unit: str = "lbs") -> dict:
    """Minimal session dict accepted by _compute_progression (Epley e1RM)."""
    e = round(w * (1 + reps / 30), 1) if reps > 1 else float(w)
    return {"date": date, "unit": unit, "max_working_weight": float(w),
            "reps_at_max": reps, "estimated_1rm": e}


def _dates(n, start_day=1):
    return [f"2026-03-{start_day + i:02d}" for i in range(n)]


def _sessions(maxes, reps=5):
    return [S(d, w, reps) for d, w in zip(_dates(len(maxes)), maxes)]


# The live Lat Pulldown shape (DB-verified session maxes, chronological).
_LIVE_STABLE = [130, 130, 145, 145, 120, 145, 145, 145, 145]
_LIVE_DELOAD = _LIVE_STABLE + [60, 60, 60]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Sustained deload block — the live failure shape
# ══════════════════════════════════════════════════════════════════════════════

def test_sustained_deload_block_classified_and_current_holds_established_level():
    p = _compute_progression(_sessions(_LIVE_DELOAD))

    assert p["training_state"] == "deload"
    assert p["deload_block"] is not None
    assert p["deload_block"]["sessions"] == 3
    assert p["deload_block"]["start_date"] == _dates(len(_LIVE_DELOAD))[-3]
    assert p["deload_block"]["block_max_weight"] == 60.0
    assert p["deload_block"]["established_weight"] == 145.0

    # Current ability = the established 145 level, NOT the deload max.
    assert p["current_weight"] == 145.0
    assert p["max_weight_end"] == 145.0
    assert "established working level" in p["current_basis"]

    # The false-regression narrative is gone in every derived field.
    assert p["weight_change_pct"] > 0                    # 130 -> 145 = +11.5
    assert p["weight_change_pct"] == 11.5
    assert p["regression_from_peak"] is None
    # The deload day IS labeled a back-off now (was wrongly False).
    assert p["latest_session_is_backoff"] is True
    # An intentional deload is not a plateau.
    assert p["is_plateau"] is False
    assert "deload" in p["plateau_note"]


def test_sustained_deload_never_reports_the_deload_max_as_current():
    p = _compute_progression(_sessions(_LIVE_DELOAD))
    # Negative direction: 60 must not reach any "current/end" sink.
    assert p["current_weight"] != 60.0
    assert p["max_weight_end"] != 60.0
    assert p["weight_change_pct"] != -53.8
    # But the latest-session fields still truthfully show the deload day.
    assert p["latest_session_weight"] == 60.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Isolated single deload day — existing behavior preserved (pin)
# ══════════════════════════════════════════════════════════════════════════════

def test_isolated_single_deload_day_pins_existing_behavior():
    p = _compute_progression(_sessions([130, 145, 145, 145, 60]))

    assert p["current_weight"] == 145.0                  # unchanged from today
    assert p["max_weight_end"] == 145.0
    assert p["regression_from_peak"] is None
    assert p["latest_session_is_backoff"] is True
    assert p["is_plateau"] is False
    # The single categorical back-off day now carries the explicit label.
    assert p["training_state"] == "deload"
    assert p["deload_block"]["sessions"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. Genuine gradual regression — never masked as a deload
# ══════════════════════════════════════════════════════════════════════════════

def test_gradual_regression_not_masked_as_deload():
    p = _compute_progression(_sessions([145, 130, 120, 110, 100]))

    assert p["training_state"] == "regression"
    assert p["deload_block"] is None
    assert p["regression_from_peak"] is not None
    assert p["regression_from_peak"]["peak_weight"] == 145.0
    assert p["regression_from_peak"]["current_weight"] == 120.0   # best of last 3
    assert "established working level" not in p["current_basis"]


def test_detector_rejects_gradual_decline_directly():
    assert _detect_deload_block([145.0, 130.0, 120.0, 110.0, 100.0]) is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stable working level — no change from today
# ══════════════════════════════════════════════════════════════════════════════

def test_stable_working_level_unchanged():
    p = _compute_progression(_sessions(_LIVE_STABLE))

    assert p["training_state"] == "working"
    assert p["deload_block"] is None
    assert p["current_weight"] == 145.0
    assert p["regression_from_peak"] is None
    assert p["current_basis"] == "best of last 3 sessions"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Boundary — deload block then return to the working level
# ══════════════════════════════════════════════════════════════════════════════

def test_deload_then_return_recognized_not_new_pr_not_regression():
    maxes = [145, 145, 145, 145, 60, 60, 60, 145]
    p = _compute_progression(_sessions(maxes))

    assert p["training_state"] == "working"              # back at the level
    assert p["deload_block"] is None
    assert p["current_weight"] == 145.0
    assert p["regression_from_peak"] is None
    # The return to 145 is NOT a new best (equal is not a new best) — the
    # new-best date stays the original first 145.
    assert p["last_new_best_date"] == _dates(len(maxes))[0]


# ══════════════════════════════════════════════════════════════════════════════
# 6. Bias cap — an over-long "deload" is surfaced as regression, never masked
# ══════════════════════════════════════════════════════════════════════════════

def test_overlong_low_block_exceeding_cap_is_regression_not_deload():
    maxes = [145, 145, 145, 145, 145] + [60] * (DELOAD_MAX_SESSIONS + 1)
    p = _compute_progression(_sessions(maxes))

    assert p["training_state"] == "regression"           # decline is VISIBLE
    assert p["deload_block"] is None
    assert p["regression_from_peak"] is not None
    assert p["regression_from_peak"]["current_weight"] == 60.0


# ══════════════════════════════════════════════════════════════════════════════
# Supporting edges
# ══════════════════════════════════════════════════════════════════════════════

def test_thin_data_reports_insufficient_data():
    p = _compute_progression(_sessions([100, 100, 100]))
    assert p["trend_assessable"] is False
    assert p["training_state"] == "insufficient_data"
    assert p["deload_block"] is None


def test_moderate_backoff_day_is_not_categorical_deload():
    # 120 after 130 (−7.7%) is a normal lighter day, not a categorical
    # step-down — the existing k-window handles it and no deload is declared.
    p = _compute_progression(_sessions([100, 110, 120, 130, 120]))
    assert p["training_state"] == "working"
    assert p["deload_block"] is None
    assert p["current_weight"] == 130.0


def test_unstable_prior_level_blocks_deload_classification():
    # The three pre-block sessions are erratic (145, 90, 145): no stable
    # established level exists, so the low tail is NOT called a deload.
    p = _compute_progression(_sessions([145, 145, 145, 90, 145, 60, 60, 60]))
    assert p["training_state"] == "regression"           # surfaced, not masked
    assert p["deload_block"] is None
