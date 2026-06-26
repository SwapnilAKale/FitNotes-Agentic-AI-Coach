"""
Live-check fix — Part 4: permissive (READ-path) resolver rank-and-pick.

permissive=True auto-resolves a clear-margin difflib winner among LIKE-tier candidates
("walk" → Walking). permissive=False (default; WRITE path) keeps strict disambiguation —
a silent wrong write is unrecoverable (the locked auto-pick-removal rule). Genuine
ambiguity (close ratios) still disambiguates even when permissive.

No Gemini, no server. Runs against the pinned project DB.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.shared.resolver import resolve_exercise_name              # noqa: E402

_DB = os.environ["FITNOTES_DB_PATH"]


def test_permissive_walk_now_asks():
    """Bug 2.4(ii): FLIPPED from the old name-margin auto-pick (walk → Walking).
    'walk' matches TWO real-data exercises (Walking 79 sets + Farmers Walk 58 sets),
    so the read path now ASKS rather than guessing on a fragile name margin."""
    out = resolve_exercise_name("walk", _DB, permissive=True)
    assert out["match"] is None
    assert "Walking" in out["candidates"] and "Farmers Walk" in out["candidates"]


def test_strict_default_walk_still_disambiguates():
    """LOCKED write-path guarantee: default (permissive=False) NEVER auto-picks —
    'walk' still returns multiple candidates for the user to choose."""
    out = resolve_exercise_name("walk", _DB)              # default strict
    assert out["match"] is None
    assert "Walking" in out["candidates"] and "Farmers Walk" in out["candidates"]


def test_permissive_calf_raise_picks_real_target_not_sparse():
    """LOAD-BEARING: old behavior auto-picked 'Calf Raises' (1 set) by name. Now only
    'Barbell Calf Raise' (27 sets) clears the real-data floor, so it is the sole real
    target and is auto-picked — never the sparse 1-set 'Calf Raises'."""
    out = resolve_exercise_name("calf raise", _DB, permissive=True)
    assert out["match"] == "Barbell Calf Raise"
    assert out["match"] != "Calf Raises"
    assert out["candidates"] == []


def test_permissive_dumbbell_bench_press_asks():
    """Flat/Incline/Decline Dumbbell Bench Press are all heavily logged → 3 real-data
    candidates → ASK, never guess one."""
    out = resolve_exercise_name("dumbbell bench press", _DB, permissive=True)
    assert out["match"] is None
    assert len(out["candidates"]) >= 2
    assert all("Dumbbell Bench Press" in c for c in out["candidates"])


def test_permissive_squat_asks():
    """Multiple real-data squat variants (Sumo/Dumbbell/Smith Machine Squats) → ASK."""
    out = resolve_exercise_name("squat", _DB, permissive=True)
    assert out["match"] is None
    assert len(out["candidates"]) >= 2


def test_permissive_ask_list_is_data_first():
    """Nicety: the returned ask-list leads with the data-richest target. 'walk' →
    Walking (79 sets) before Farmers Walk (58 sets)."""
    out = resolve_exercise_name("walk", _DB, permissive=True)
    assert out["match"] is None
    assert out["candidates"][0] == "Walking"


def test_permissive_all_unlogged_keeps_name_only_fallback():
    """0 real-data candidates → name-only fallback unchanged. 'leg curl' →
    [Lying/Seated Leg Curl Machine] are both 0-set with near-equal ratios
    (gap < margin) → still disambiguate (no data-based auto-pick)."""
    out = resolve_exercise_name("leg curl", _DB, permissive=True)
    assert out["match"] is None
    assert len(out["candidates"]) >= 2
    assert all("Leg Curl" in c for c in out["candidates"])


def test_strict_write_path_applies_no_data_filter():
    """WRITE path untouched: 'calf raise' with permissive=False does NOT apply the
    real-data filter — it returns the full candidate list for strict disambiguation,
    including the sparse/zero-set variants (a first-time-logged exercise has 0 sets)."""
    out = resolve_exercise_name("calf raise", _DB)        # default strict
    assert out["match"] is None
    assert "Barbell Calf Raise" in out["candidates"]
    assert len(out["candidates"]) >= 2


def test_permissive_single_candidate_returned_directly():
    """Single-candidate resolution is unchanged: 'hammer curl' matches only
    'Dumbbell Hammer Curl' → returned directly, no ambiguity."""
    out = resolve_exercise_name("hammer curl", _DB, permissive=True)
    assert out["match"] == "Dumbbell Hammer Curl"
    assert out["candidates"] == []
