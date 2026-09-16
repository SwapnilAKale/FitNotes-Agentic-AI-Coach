"""
Deterministic answer guards for the live-verification defects of 2026-08-06.

The live check got every number right and still shipped:
  B1  "leg training frequency is low at 5.3 sessions per week"   (5.3 SETS)
  B2  "25 sets for Chest, 28 for Back, and 14 for Shoulders"     (invented)
  B3  "Hamstring Curls Machine", "T Bar Barbell Row"             (near-miss names)
  B4  a plan quoting 1.8 sets a week with no target
  B5  "90-day recovery trends"
Prompt rules for B1, B3 and B4 existed and did not hold. These guards make them
invariants. Pure functions, a synthetic package, no live data, no network.
"""

import ast
import re
from pathlib import Path

import pytest

from src import citations as cite

_ROOT = Path(__file__).resolve().parent.parent

PKG = {
    "query_period_days": 90,
    "training_frequency": {"sessions_per_week": 3.2},
    "muscle_ontology_summary": {
        "weeks_in_window": 13.0,
        "muscles": [
            {"muscle": "Legs", "primary_sets": 69, "secondary_sets": 20, "limiting_sets": 0,
             "primary_sets_per_week": 5.3, "secondary_sets_per_week": 1.5},
            {"muscle": "Hamstrings", "primary_sets": 23, "secondary_sets": 0, "limiting_sets": 0,
             "primary_sets_per_week": 1.8, "secondary_sets_per_week": 0.0},
            {"muscle": "Mid Traps", "primary_sets": 60, "secondary_sets": 3, "limiting_sets": 0,
             "primary_sets_per_week": 4.6, "secondary_sets_per_week": 0.2},
        ],
    },
    "muscle_group_summary": [
        {"muscle_group": "Chest", "total_sets": 120},
        {"muscle_group": "Back", "total_sets": 185},
        {"muscle_group": "Shoulders", "total_sets": 122},
    ],
    "exercises": [{"name": "Hamstring Curl Machine"}, {"name": "T-Bar Barbell Row"},
                  {"name": "Incline Smith Machine Press"}, {"name": "Smith Machine Press"},
                  {"name": "Deadlift"}],
    "suggestable_exercises": ["Cable Crunch"],
}
ONT = {"exercises": {1: {"canonical_name": "Lateral Dumbbell Raise"}}}


# ── B1 · sets are never sessions ──────────────────────────────────────────────

def test_sets_rate_called_sessions_is_rewritten():
    out, flags = cite.sets_as_sessions_guard(
        "Your current leg training frequency is low at 5.3 sessions per week.", PKG)
    assert out == "Your current leg training volume is low at 5.3 sets per week."
    assert flags and flags[0]["kind"] == "sets_as_sessions"


def test_the_real_session_rate_is_left_alone():
    """The hard case: the session rate happens to equal a muscle's set rate."""
    pkg = dict(PKG, training_frequency={"sessions_per_week": 4.6})   # = Mid Traps
    text = "You train 4.6 sessions per week on average."
    assert cite.sets_as_sessions_guard(text, pkg) == (text, [])


def test_a_number_matching_no_set_rate_is_left_alone():
    text = "Aim for 4 sessions a week."
    assert cite.sets_as_sessions_guard(text, PKG) == (text, [])


# ── B5 · a window label is not physiology ─────────────────────────────────────

def test_window_label_on_recovery_is_rewritten():
    out, flags = cite.window_label_guard(
        "Looking at your 90-day recovery trends, the hamstrings lag.", PKG)
    assert out == "Looking at your recovery trends over the last 90 days, the hamstrings lag."
    assert flags


def test_window_label_at_sentence_start_is_capitalised():
    out, _ = cite.window_label_guard("90-day fatigue data is flat.", PKG)
    assert out == "Fatigue data over the last 90 days is flat."


@pytest.mark.parametrize("text", ["Your 90-day total is 185 sets.",
                                  "Within the 90-day window, legs got 69 sets.",
                                  "Your 30-day recovery trends look fine."])
def test_window_label_on_a_count_or_another_window_is_left_alone(text):
    assert cite.window_label_guard(text, PKG) == (text, [])


# ── B3 · exercise names come from the store, exactly ──────────────────────────

@pytest.mark.parametrize("wrong, right", [
    ("Hamstring Curls Machine", "Hamstring Curl Machine"),
    ("T Bar Barbell Row", "T-Bar Barbell Row"),
    ("lateral dumbbell raises", "Lateral Dumbbell Raise"),     # from the graph
    ("cable crunches", "Cable Crunch"),                       # from suggestable
])
def test_near_miss_names_are_rewritten(wrong, right):
    out, flags = cite.exercise_name_guard(f"Add {wrong} on Tuesday.", PKG, ONT)
    assert out == f"Add {right} on Tuesday."
    assert flags[0]["original"] == wrong


def test_exact_names_are_left_alone():
    text = "Keep Hamstring Curl Machine and T-Bar Barbell Row."
    assert cite.exercise_name_guard(text, PKG, ONT) == (text, [])


def test_single_word_names_are_never_touched():
    text = "Your deadlifts are progressing."
    assert cite.exercise_name_guard(text, PKG, ONT) == (text, [])


def test_the_longest_name_wins_its_span():
    text = "Swap in Incline Smith Machine Presses."
    out, flags = cite.exercise_name_guard(text, PKG, ONT)
    assert out == "Swap in Incline Smith Machine Press."
    assert [f["corrected"] for f in flags] == ["Incline Smith Machine Press"]


# ── B2 · a claimed set count must exist ───────────────────────────────────────

LIVE_B2 = "You currently perform 25 sets for Chest, 28 for Back, and 14 for Shoulders."


def test_invented_set_counts_are_violations():
    violations, flags = cite.set_count_guard(LIVE_B2, PKG)
    assert [f["name"] for f in flags] == ["Chest", "Back", "Shoulders"]
    assert "Chest: said 25; your data has 120 sets over the window (9.2 a week)" in violations


@pytest.mark.parametrize("text", [
    "Over the last 90 days you did 120 sets for Chest.",
    "You are currently getting 9.2 sets a week for Chest.",
    "You're averaging 9 sets per week for Chest right now.",        # rounded per-week
    "Mid Traps: 60 sets over the last 90 days.",
])
def test_real_set_counts_pass(text):
    assert cite.set_count_guard(text, PKG) == ([], [])


def test_a_prescription_is_not_a_claim():
    """It even says 'currently' — the plan cue is what marks it a prescription."""
    text = "You are currently below range, so aim for 12 sets for Chest."
    assert cite.set_count_guard(text, PKG) == ([], [])


def test_counts_inside_a_plan_are_not_claims():
    plan = ("Here is your week.\n### Day 1\nYou perform 4 sets for Chest.\n"
            "### Day 2\nYou perform 6 sets for Back.\n")
    assert cite.set_count_guard(plan, PKG) == ([], [])


# ── B4 · a sets-per-week figure in a plan comes with its target ───────────────

PLAN = ("Your hamstrings get 1.8 sets per week over the last 90 days, so this week "
        "brings them up.\n### Day 1\nHamstring Curl Machine 4x10\n### Day 2\nRest\n")


def test_a_plan_quoting_sets_per_week_gets_the_target():
    out, flags = cite.target_guard(PLAN, PKG)
    assert "1.8 sets per week (the usual target is 10–20 sets a week per muscle)" in out
    assert out.count("the usual target") == 1 and flags


def test_a_stated_range_is_left_alone():
    text = PLAN.replace("brings them up", "brings them toward 10-20 sets")
    assert cite.target_guard(text, PKG) == (text, [])


def test_a_plain_answer_without_a_plan_or_improvement_is_left_alone():
    text = "Your hamstrings get 1.8 sets per week over the last 90 days."
    assert cite.target_guard(text, PKG) == (text, [])


# ── Idempotence: a rewrite must not re-trigger on its own output ──────────────

@pytest.mark.parametrize("guard, text", [
    (lambda t: cite.sets_as_sessions_guard(t, PKG), "Legs sit at 5.3 sessions per week."),
    (lambda t: cite.window_label_guard(t, PKG), "Your 90-day recovery trends are fine."),
    (lambda t: cite.exercise_name_guard(t, PKG, ONT), "Add T Bar Barbell Row."),
    (lambda t: cite.target_guard(t, PKG), PLAN),
])
def test_rewrite_guards_are_idempotent(guard, text):
    once, first = guard(text)
    assert first, "the fixture must actually trigger the guard"
    twice, second = guard(once)
    assert twice == once and second == []


# ── Wiring ────────────────────────────────────────────────────────────────────

def _coordinator_src():
    return (_ROOT / "src" / "coordinator.py").read_text(encoding="utf-8")


def test_coordinator_calls_every_guard():
    src = _coordinator_src()
    for call in ("_cite.exercise_name_guard", "_cite.sets_as_sessions_guard",
                 "_cite.window_label_guard", "_cite.set_count_guard", "_cite.target_guard"):
        assert call in src, f"coordinator never calls {call}"


def test_rewrite_guards_run_before_the_plan_guard():
    src = _coordinator_src()
    assert (src.index("answer, rewrite_flags = _run_rewrite_guards(answer, pkg, _guard_ont)")
            < src.index("_cite.plan_guard(answer, _ont)"))
    # A re-prompted answer is guarded too.
    assert src.count("_run_rewrite_guards(retried, pkg,") == 2


def test_every_guard_is_isolated():
    src = _coordinator_src()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_run_rewrite_guards")
    assert any(isinstance(n, ast.Try) for n in ast.walk(fn))
    assert "set-count guard skipped" in src and "target guard skipped" in src
