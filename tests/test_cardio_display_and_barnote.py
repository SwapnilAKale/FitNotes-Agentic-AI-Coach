"""
Live-check fixes — Part 1 (cardio display format) + Part 2 (bar-note leak).

Part 1: cardio sessions render distance/duration/pace, not "0.0 lbs × 0 reps".
Part 2: the internal bar_weight_note is NOT in the flat display_sets list (it would
        otherwise be forced verbatim into the user answer by the DISPLAY SETS CHECK),
        while the structured dict KEY is retained.

No Gemini, no server. Runs against the pinned project DB.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent.session_display import (                       # noqa: E402
    get_exercise_sessions, build_display_sets,
)


# ── Part 2: bar-note leak ────────────────────────────────────────────────────

def test_barnote_not_in_flat_display_sets():
    """Barbell Row is a bar exercise — its flat display_sets must NOT contain the
    internal BAR-INCLUSIVE note (the verbatim DISPLAY SETS CHECK would leak it)."""
    flat = build_display_sets("exercise", "Barbell Row")
    assert flat, "expected some display lines for Barbell Row"
    for line in flat:
        assert "BAR-INCLUSIVE" not in line
        assert "do not add the bar" not in line


def test_barnote_dict_key_retained():
    """The structured return keeps the bar_weight_note KEY (consumers/tests rely on
    it) — only the flat-list append was removed."""
    out = get_exercise_sessions("Barbell Row", mode="recent")
    assert "bar_weight_note" in out
    assert "BAR-INCLUSIVE" in out["bar_weight_note"]


# ── Part 1: cardio display format ────────────────────────────────────────────

def test_walking_renders_distance_duration_not_strength():
    """Walking (distance + duration) renders 'X km in Ys (pace min/km)', never the
    strength '0.0 lbs × 0 reps' template."""
    out = get_exercise_sessions("Walking", mode="recent")
    lines = [ln for s in out["sessions"] for ln in s["display_sets"]]
    assert lines, "expected Walking display lines"
    assert any("km in" in ln and ln.rstrip().endswith(")") for ln in lines) or \
           any("km in" in ln for ln in lines)
    for ln in lines:
        assert "lbs" not in ln and "reps" not in ln
        assert "km in" in ln and "s" in ln


def test_cycling_duration_only_renders_seconds_only():
    """Cycling is duration-only (distance == 0): render '{seconds}s', no 'km'."""
    out = get_exercise_sessions("Cycling", mode="recent")
    lines = [ln for s in out["sessions"] for ln in s["display_sets"]]
    assert lines, "expected Cycling display lines"
    for ln in lines:
        assert "km" not in ln
        assert "lbs" not in ln and "reps" not in ln
        assert "s" in ln   # "{duration}s"


def test_strength_display_unchanged():
    """A strength exercise still renders the weight × reps template (no cardio leak)."""
    out = get_exercise_sessions("Lat Pulldown", mode="recent")
    lines = [ln for s in out["sessions"] for ln in s["display_sets"]]
    assert lines
    assert all("km in" not in ln for ln in lines)
    assert any("reps" in ln for ln in lines)
