"""
A plan keeps every other muscle group at maintenance — unless the user asks for
"only X" (F6, live re-check 2026-09-16; rule set by the user 2026-09-17).

WHY THIS EXISTS. "plan me a week that brings up my hamstrings…" came back with
Arms at 6 sets a week against the user's current 19.9, and Back at 12 against
18.0. Nothing looked. Nobody trains only the muscle they want bigger, so holding
the rest at maintenance is the DEFAULT: every plan is checked, and only an
explicit "arms only / nothing else / drop the rest" switches the check off.

THE PACKAGE DECIDES; THE REGEX ONLY LOCATES. Current figures are the package's
group-level primary_sets_per_week; the plan's figure is the sets it prescribes,
counted through the graph exactly as the package counts (once per group per
exercise, primary role only).
"""

import csv
import os

import pytest

from src import citations as cite
from src import ontology as ont_mod

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Detectors, on the real graph's muscle names (read only) ──────────────────

@pytest.fixture(scope="module")
def real_ontology():
    old = os.environ.get("ONTOLOGY_DIR")
    os.environ["ONTOLOGY_DIR"] = os.path.join(_ROOT, "ontology")
    ont_mod.clear_cache()
    try:
        yield ont_mod.load_ontology(force=True)
    finally:
        if old is None:
            os.environ.pop("ONTOLOGY_DIR", None)
        else:
            os.environ["ONTOLOGY_DIR"] = old
        ont_mod.clear_cache()


@pytest.mark.parametrize("question", [
    "give me a chest-only plan",
    "plan a week with only arms",
    "just legs, nothing else",
    "build a split for back and drop everything else",
    "a 4 day split that trains only my back",
    "plan me an arms only week",
    "give me a week of just biceps and triceps",
    "make a plan for shoulders, skip the rest",
    "plan a week that is purely glutes",
    "I want a week with no other muscles, just hamstrings",
    "plan me a week for legs only",
])
def test_an_explicit_only_request_opts_out(question, real_ontology):
    assert cite.asks_to_drop_the_rest(question, real_ontology)


@pytest.mark.parametrize("question", [
    "plan me a week that brings up my hamstrings, keep everything else steady",
    "build me a week of training that pushes and focuses my arms without dropping anything else",
    "build me a week of training that pushes my calves and rear delts without dropping anything else",
    "give me a 4 day split focused on chest",
    "Make me a one week plan that helps me reach this goal and also, give reasons for what the plan",
    "show my last week, then plan next week",
    "can you design a new push pull legs routine for me?",
    "write me a three-day program",
    "plan me a week, I only have 4 days",
    "build me a week, I just want bigger arms",
    "plan a week where I only train 3 times",
    "plan a week, my legs only recover slowly",
    "give me a split that brings up my back, not just my arms",
    "plan my week, not only chest",
])
def test_a_focus_is_not_an_opt_out(question, real_ontology):
    """A focus keeps the rest at maintenance. "only" about days, times or
    recovery — or negated — is not a request to drop muscle groups."""
    assert not cite.asks_to_drop_the_rest(question, real_ontology)


@pytest.mark.parametrize("question, muscle, expected", [
    ("plan me a week with less back work", "Back", True),
    ("plan a week and deload my shoulders", "Shoulders", True),
    ("give me a split that cuts back on chest", "Chest", True),
    ("give me a week with fewer sets for arms", "Arms", True),
    ("plan me a week that brings up my hamstrings", "Hamstrings", False),
    ("plan a week, bring my back up", "Back", False),
    ("plan me a week, my back hurts less now", "Back", False),
])
def test_a_group_the_user_asks_to_reduce_is_recognised(question, muscle, expected):
    assert cite.asks_to_reduce(question, muscle) is expected


# ── Counting, on a synthetic graph ───────────────────────────────────────────

#   Back            Arms              Legs
#     ├ Lats          ├ Biceps          └ Glutes
#     └ Rhomboids     └ Triceps
_MUSCLES = [(1, "Back", "", "large"), (2, "Lats", 1, "large"), (3, "Rhomboids", 1, "medium"),
            (4, "Arms", "", ""), (5, "Biceps", 4, "medium"), (6, "Triceps", 4, "medium"),
            (7, "Legs", "", "large"), (8, "Glutes", 7, "large")]
_EXERCISES = [(1, "Barbell Row", "barbell", "horizontal pull"),
              (2, "Barbell Curl", "barbell", "elbow flexion"),
              (3, "Skull Crusher", "dumbbell", "elbow extension"),
              (4, "Hip Thrust", "barbell", "hip extension")]
_EDGES = [(1, 2, "primary", "test"), (1, 3, "primary", "test"),   # two primaries, one group
          (2, 5, "primary", "test"), (3, 6, "primary", "test"), (4, 8, "primary", "test")]


@pytest.fixture
def ont(tmp_path, monkeypatch):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], _MUSCLES)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"], _EXERCISES)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], _EDGES)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], [(n, i) for i, n, *_ in _EXERCISES])
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    assert o["errors"] == []
    yield o
    ont_mod.clear_cache()


def _pkg(**per_week):
    return {"muscle_ontology_summary": {"weeks_in_window": 13.0, "muscles": [
        {"muscle": m, "primary_sets_per_week": v} for m, v in per_week.items()]}}


PKG = _pkg(Back=8.0, Arms=10.0, Legs=1.5)
Q = "build me a week of training"


def _week(*days):
    return "".join(f"* **{d}:**\n" + "".join(f" * {line}\n" for line in lines) for d, lines in days)


def test_a_plan_that_keeps_every_group_is_fine(ont):
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl (5 sets)"]),
                 ("Wednesday", ["Barbell Row (4 sets)", "Skull Crusher (5 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, Q) == []


def test_a_group_cut_below_maintenance_is_reported(ont):
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl (2 sets)"]),
                 ("Wednesday", ["Barbell Row (4 sets)", "Skull Crusher (2 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, Q) == [
        {"group": "Arms", "current": 10.0, "planned": 4}]


def test_a_small_dip_within_twenty_percent_is_not_a_cut(ont):
    """One planned week against a 13-week average wobbles: 9 against 10 is fine."""
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl (5 sets)"]),
                 ("Wednesday", ["Barbell Row (4 sets)", "Skull Crusher (4 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, Q) == []


def test_an_explicit_only_request_is_not_checked(ont):
    plan = _week(("Monday", ["Barbell Curl (2 sets)"]), ("Wednesday", ["Skull Crusher (2 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, "build me an arms only week") == []


def test_a_group_the_user_asks_to_reduce_is_not_reported(ont):
    plan = _week(("Monday", ["Barbell Row (2 sets)", "Barbell Curl (5 sets)"]),
                 ("Wednesday", ["Skull Crusher (5 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, "plan me a week with less back work") == []


def test_an_exercise_with_two_muscles_in_one_group_counts_once(ont):
    """Barbell Row trains Lats AND Rhomboids, both in Back. The package counts
    its sets once for Back, so the plan must too — 7, not 14."""
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl (5 sets)"]),
                 ("Wednesday", ["Barbell Row (3 sets)", "Skull Crusher (5 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, _pkg(Back=12.0, Arms=10.0), Q) == [
        {"group": "Back", "current": 12.0, "planned": 7}]


def test_several_exercises_in_one_table_cell_are_each_counted(ont):
    plan = ("| Day | Work |\n|:--- |:--- |\n"
            "| 1 | Barbell Row (4), Barbell Curl (5) |\n"
            "| 3 | Barbell Row (4), Skull Crusher (5) |\n")
    assert cite.plan_volume_shortfalls(plan, ont, PKG, Q) == []


def test_an_exercise_without_a_set_count_leaves_its_group_unjudged(ont):
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl"]),
                 ("Wednesday", ["Barbell Row (4 sets)", "Skull Crusher (2 sets)"]))
    assert cite.plan_volume_shortfalls(plan, ont, PKG, Q) == []


def test_a_group_trained_under_two_sets_a_week_is_not_judged(ont):
    """Legs at 1.5 a week, absent from the plan: not a maintenance group."""
    plan = _week(("Monday", ["Barbell Row (4 sets)", "Barbell Curl (5 sets)"]),
                 ("Wednesday", ["Barbell Row (4 sets)", "Skull Crusher (5 sets)"]))
    assert all(s["group"] != "Legs" for s in cite.plan_volume_shortfalls(plan, ont, PKG, Q))


def test_a_single_session_is_not_a_week(ont):
    assert cite.plan_volume_shortfalls("Barbell Curl (2 sets)\nSkull Crusher (2 sets)",
                                       ont, PKG, "plan my next arm session") == []


def test_the_requirement_states_the_real_figure_not_the_draft():
    reqs = cite.volume_requirements([{"group": "Back", "current": 18.0, "planned": 12}])
    assert reqs == ["Back: the user currently does 18 primary sets a week — "
                    "the plan must keep it at least there."]


# ── The live answer, on the real graph ───────────────────────────────────────

def test_live_prompt_1_cut_arms_and_back(real_ontology):
    from test_plan_guard import _LIVE_PROMPT_1
    pkg = _pkg(Arms=19.9, Back=18.0, Shoulders=11.8, Chest=11.4, Legs=6.0, Core=0.0)
    question = "plan me a week that brings up my hamstrings, keep everything else steady"
    assert cite.plan_volume_shortfalls(_LIVE_PROMPT_1, real_ontology, pkg, question) == [
        {"group": "Arms", "current": 19.9, "planned": 6},
        {"group": "Back", "current": 18.0, "planned": 12}]
