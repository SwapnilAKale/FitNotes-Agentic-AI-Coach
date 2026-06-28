"""
Approach (b): session-display as a NORMAL package field (`display_sets`).

Covers the three new units, no Gemini / no server:
  - the deterministic scope-XOR (_display_scope, pure)
  - the ≤1 prior-session trigger (pure predicates + DB-backed both modes)
  - the DISPLAY SETS CHECK (containment + reframe-once + raw-assembly fallback)

DB-backed assertions run against the pinned project DB (same as the golden tests).
"""

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import _display_scope, prepare_analysis_package    # noqa: E402
from src.data_agent.session_display import (                          # noqa: E402
    _exercise_prior_needed, _category_prior_needed, build_display_sets,
)
from src.analysis_agent import (                                      # noqa: E402
    display_sets_missing, raw_display_assembly, enforce_display_fidelity,
    append_missing_display,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Scope-XOR (pure) — display_sets present iff EXACTLY ONE narrow scope holds
# ══════════════════════════════════════════════════════════════════════════════

def test_scope_single_exercise():
    assert _display_scope(["Bench Press"], None, []) == [("exercise", "Bench Press")]


def test_scope_single_category():
    assert _display_scope(None, ["Back"], []) == [("category", "Back")]


def test_scope_exercise_plus_group_both_blocks():
    # Regression for the XOR-drop bug: a single-exercise question that ALSO carries
    # an inferred parent-group tag now yields BOTH a display target — exercise AND
    # category — instead of None. (Was test_scope_both_filters_none.)
    assert _display_scope(["Bench Press"], ["Back"], []) == [
        ("exercise", "Bench Press"), ("category", "Back")]


def test_scope_neither_filter_empty():
    assert _display_scope(None, None, []) == []
    assert _display_scope([], [], []) == []


def test_scope_two_exercises_both_blocks():
    # Two resolved exercises → one display target each. (Was test_scope_two_exercises_none.)
    assert _display_scope(["Bench Press", "Deadlift"], None, []) == [
        ("exercise", "Bench Press"), ("exercise", "Deadlift")]


def test_scope_single_name_but_unresolved_empty():
    # The one requested name did not resolve → no resolved exercise → no targets.
    assert _display_scope(["Nonexistent Lift"], None, ["Nonexistent Lift"]) == []


# ══════════════════════════════════════════════════════════════════════════════
# 2. ≤1 trigger — pure predicates
# ══════════════════════════════════════════════════════════════════════════════

def test_exercise_prior_needed_sets_only():
    # Single-exercise = SETS-ONLY: prior iff most-recent date had ≤1 set.
    assert _exercise_prior_needed([{"total_sets": 1}, {"total_sets": 3}]) is True
    assert _exercise_prior_needed([{"total_sets": 3}, {"total_sets": 3}]) is False
    assert _exercise_prior_needed([{"total_sets": 1}]) is False          # no prior date
    assert _exercise_prior_needed([]) is False


def test_category_prior_needed_exercises_or_sets():
    # Category: prior iff ≤1 exercise OR ≤1 set total on the most-recent date.
    assert _category_prior_needed([{"total_sets": 3}]) is True            # 1 exercise
    assert _category_prior_needed([{"total_sets": 1},
                                   {"total_sets": 0}]) is True            # 1 set total
    assert _category_prior_needed([{"total_sets": 3},
                                   {"total_sets": 2}]) is False           # rich date
    assert _category_prior_needed([]) is False


# ══════════════════════════════════════════════════════════════════════════════
# 2b. ≤1 trigger — DB-backed, both modes
# ══════════════════════════════════════════════════════════════════════════════

# test_build_single_exercise_no_prior_when_rich and
# test_build_single_category_prior_included_when_thin — DELETED (unsound live-DB
# goldens: most-recent date drifted to 2026-06-27). The <=1 trigger branches are
# covered by the pure predicates test_exercise_prior_needed_sets_only /
# test_category_prior_needed_exercises_or_sets above; the flatten header format,
# most-recent resolution, and prior-append integration are covered by construction
# in tests/test_session_display_synthetic.py.


def test_build_unknown_scope_empty():
    assert build_display_sets("exercise", "No Such Exercise") == []
    assert build_display_sets("category", "NotAMuscleGroup") == []


# ══════════════════════════════════════════════════════════════════════════════
# 2c. End-to-end: prepare_analysis_package attaches display_sets on a narrow scope
#     (and validate() tolerates the new top-level list field), absent when broad.
# ══════════════════════════════════════════════════════════════════════════════

# test_package_has_display_sets_for_single_exercise — DELETED (unsound live-DB
# golden: display_sets[0] date drifted to 2026-06-27). That display_sets is
# attached for a narrow scope is covered (not date-pinned) by
# test_package_display_sets_for_exercise_plus_group below; most-recent + flatten
# format are covered by construction in tests/test_session_display_synthetic.py.


def test_package_no_display_sets_for_broad_scope():
    pkg = prepare_analysis_package(query_period_days=90)
    assert "display_sets" not in pkg


def test_package_display_sets_for_exercise_plus_group():
    # End-to-end regression for the XOR-drop bug: "show me my last Lat Pulldown
    # session" sometimes also carries muscle_groups=["Back"] from the classifier.
    # Under the old XOR this attached NOTHING; now both blocks are built and
    # flattened — an exercise header AND a category header must both be present.
    pkg = prepare_analysis_package(query_period_days=None,
                                   exercise_names=["Lat Pulldown"],
                                   muscle_groups=["Back"])
    ds = pkg.get("display_sets")
    assert isinstance(ds, list) and ds
    assert any(" — Lat Pulldown (" in s for s in ds)     # the exercise block
    assert any(s.endswith("— Back:") for s in ds)        # the category block


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISPLAY SETS CHECK — containment + reframe-once + raw-assembly fallback
# ══════════════════════════════════════════════════════════════════════════════

_PKG = {"display_sets": ["Set 1: 100.0 lbs × 10 reps",
                         "Set 2: 110.0 lbs × 8 reps"]}


def test_display_sets_missing_all_present():
    answer = ("Here is your last session.\n"
              "Set 1: 100.0 lbs × 10 reps\nSet 2: 110.0 lbs × 8 reps\nNice work.")
    assert display_sets_missing(answer, _PKG) == []


def test_display_sets_missing_one_paraphrased():
    answer = "Set 1: 100.0 lbs × 10 reps, then it dropped to 110 for 8."
    missing = display_sets_missing(answer, _PKG)
    assert missing == ["Set 2: 110.0 lbs × 8 reps"]


def test_raw_display_assembly_exact():
    assert raw_display_assembly(_PKG) == (
        "Set 1: 100.0 lbs × 10 reps\nSet 2: 110.0 lbs × 8 reps")
    assert raw_display_assembly({}) == ""


def test_enforce_clean_answer_no_reframe():
    answer = "Set 1: 100.0 lbs × 10 reps and Set 2: 110.0 lbs × 8 reps."
    calls = {"n": 0}

    async def reframe():
        calls["n"] += 1
        return "unused"

    out = asyncio.run(enforce_display_fidelity(answer, _PKG, reframe))
    assert out == answer
    assert calls["n"] == 0                          # clean → reframe never called


def test_enforce_reframe_fixes():
    bad = "I think you did roughly 100 and 110 lbs."

    async def reframe():
        return ("Your last session:\nSet 1: 100.0 lbs × 10 reps\n"
                "Set 2: 110.0 lbs × 8 reps")

    out = asyncio.run(enforce_display_fidelity(bad, _PKG, reframe))
    assert display_sets_missing(out, _PKG) == []     # reframed output is verbatim


def test_enforce_falls_back_appends_not_replaces():
    # FLIPPED from the old destructive expectation (out == raw_display_assembly):
    # that assertion encoded the bug — the analysis was REPLACED by the raw block.
    # Post-fix the fallback APPENDS: the original prose survives AND the verbatim
    # sets are present.
    bad = "I think you did roughly 100 and 110 lbs."

    async def reframe():
        return "Still paraphrasing, no verbatim lines here."

    out = asyncio.run(enforce_display_fidelity(bad, _PKG, reframe))
    assert bad in out                                # original answer NOT discarded
    assert display_sets_missing(out, _PKG) == []     # verbatim sets now present
    assert out != raw_display_assembly(_PKG)         # not a bare replace


def test_enforce_load_bearing_analysis_survives():
    # THE load-bearing case: an analytical answer (real prose) that does NOT contain
    # the verbatim display strings, reframe also non-verbatim. Pre-fix this returned
    # ONLY the raw block (analysis destroyed). Post-fix the analysis survives and the
    # exact sets are appended underneath.
    analysis = ("Your Lat Pulldown is in a back-off phase — load is down from peak "
                "but rep quality held, so this looks intentional rather than a stall.")

    async def reframe():
        return "Still just a summary, no exact set lines."

    out = asyncio.run(enforce_display_fidelity(analysis, _PKG, reframe))
    assert analysis in out                           # the real analysis is preserved
    assert display_sets_missing(out, _PKG) == []     # exact sets appended


def test_enforce_no_double_append_when_present():
    # A display answer already containing every verbatim string passes containment
    # and is returned UNCHANGED — the fix must not append a duplicate block.
    answer = ("Here is your last session.\n"
              "Set 1: 100.0 lbs × 10 reps\nSet 2: 110.0 lbs × 8 reps\nNice work.")
    calls = {"n": 0}

    async def reframe():
        calls["n"] += 1
        return "unused"

    out = asyncio.run(enforce_display_fidelity(answer, _PKG, reframe))
    assert out == answer                             # unchanged, no appended duplicate
    assert calls["n"] == 0                           # clean → reframe never called
    assert out.count("Set 1: 100.0 lbs × 10 reps") == 1   # not duplicated


def test_append_missing_display_helper():
    # Non-empty answer → block appended after it (analysis kept).
    out = append_missing_display("My analysis.", _PKG)
    assert out.startswith("My analysis.")
    assert display_sets_missing(out, _PKG) == []
    # Empty package block → answer returned unchanged.
    assert append_missing_display("My analysis.", {}) == "My analysis."
    # Empty / blank answer → block alone (no leading blank line).
    assert append_missing_display("", _PKG) == raw_display_assembly(_PKG)
    assert append_missing_display("   ", _PKG) == raw_display_assembly(_PKG)


def test_enforce_no_display_sets_is_noop():
    async def reframe():
        raise AssertionError("reframe must not be called when no display_sets")

    out = asyncio.run(enforce_display_fidelity("anything", {}, reframe))
    assert out == "anything"
