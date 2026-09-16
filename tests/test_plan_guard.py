"""
plan_guard — a muscle must not get DIRECT work on two consecutive days.

WHY THIS EXISTS. Asked for a one-week arm plan, the coach produced:

    | 4 | Shoulders/Arms | DB Skull Crusher (4 sets), Barbell Curl (4 sets) |
    | 5 | Back/Biceps    | Seated Machine Curl (4 sets)                     |

Direct biceps work on back-to-back days. Every fact needed to catch that was
already in the ontology — and nothing looked at it. The graph INFORMED the answer
but never CHECKED it, which is exactly the defect limiting_claim_guard exists for.

THE GRAPH DECIDES; THE REGEX ONLY LOCATES. Day headings and exercise mentions are
found by pattern; whether two days clash comes from edges_by_exercise. Exercise
names are a closed vocabulary read from the store, never an open text match.

Runs on a synthetic store so both directions are exercised explicitly.
"""

import csv

import pytest

from src import ontology as ont_mod
from src.citations import plan_guard

#   Back            Arms                Legs
#     └ Lats          ├ Biceps            └ Glutes
#                     └ Triceps
_MUSCLES = [
    (1, "Back",    "", "large"), (2, "Lats",    1, "large"),
    (3, "Arms",    "", ""),      (4, "Biceps",  3, "medium"),
    (5, "Triceps", 3, "medium"), (6, "Legs",    "", "large"),
    (7, "Glutes",  6, "large"),
    # A sub-head, so parent/child overlap is exercisable.
    (8, "Triceps Long Head", 5, "medium"),
    # A sibling of Biceps, for the substring-bleed case.
    (9, "Brachialis", 3, "small"),
]
_EXERCISES = [
    (1, "Barbell Curl",   "barbell", "elbow flexion"),
    (2, "Machine Curl",   "machine", "elbow flexion"),
    (3, "Skull Crusher",  "dumbbell", "elbow extension"),
    (4, "Hip Thrust",     "barbell", "hip extension"),
    (5, "Chin Up",        "bodyweight", "vertical pull"),
    (6, "Close Grip Bench", "barbell", "horizontal push"),
    # Name CONTAINS "Machine Curl" — the substring-bleed case.
    (7, "Preacher Machine Curl", "machine", "elbow flexion"),
]
_EDGES = [
    (1, 4, "primary",   "test"),   # Barbell Curl  -> Biceps
    (2, 4, "primary",   "test"),   # Machine Curl  -> Biceps
    (3, 8, "primary",   "test"),   # Skull Crusher -> Triceps Long Head (CHILD)
    (4, 7, "primary",   "test"),   # Hip Thrust    -> Glutes
    (5, 2, "primary",   "test"),   # Chin Up       -> Lats
    (5, 4, "secondary", "test"),   # Chin Up       -> Biceps (ASSISTING only)
    (6, 5, "primary",   "test"),   # Close Grip Bench     -> Triceps (PARENT)
    (7, 9, "primary",   "test"),   # Preacher Machine Curl -> Brachialis ONLY
]
_ALIASES = [(n, i) for i, n, *_ in _EXERCISES]


@pytest.fixture
def ont(tmp_path, monkeypatch):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], _MUSCLES)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"],
       _EXERCISES)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], _EDGES)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], _ALIASES)
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    assert o["errors"] == [], o["errors"]
    yield o
    ont_mod.clear_cache()


def _plan(*rows):
    head = "| Day | Work |\n|:--- |:--- |\n"
    return head + "\n".join(f"| {d} | {w} |" for d, w in rows)


# ── The failure ──────────────────────────────────────────────────────────────

def test_flags_the_same_muscle_on_consecutive_days(ont):
    """The live shape: two different lifts, same primary muscle, adjacent days."""
    v, flags, _c = plan_guard(_plan((4, "Skull Crusher, Barbell Curl"),
                                (5, "Machine Curl")), ont)
    assert len(v) == 1
    assert "Biceps" in v[0] and "day 4" in v[0] and "day 5" in v[0]
    assert flags[0]["muscle"] == "Biceps"
    assert flags[0]["day_a"] == 4 and flags[0]["day_b"] == 5


def test_flags_the_same_exercise_repeated_on_adjacent_days(ont):
    v, _f, _c = plan_guard(_plan((1, "Barbell Curl"), (2, "Barbell Curl")), ont)
    assert len(v) == 1 and "Biceps" in v[0]


# ── The negatives that matter ────────────────────────────────────────────────

def test_allows_the_same_muscle_with_a_day_between(ont):
    """A guard that rejects correct plans is worse than none."""
    v, flags, _c = plan_guard(_plan((1, "Barbell Curl"), (2, "Hip Thrust"),
                                (3, "Machine Curl")), ont)
    assert v == [] and flags == []


def test_secondary_overlap_is_not_a_clash(ont):
    """Chin Up only ASSISTS the biceps. A back day beside a biceps day is a
    normal, correct split and must pass untouched."""
    v, _f, _c = plan_guard(_plan((1, "Chin Up"), (2, "Barbell Curl")), ont)
    assert v == []


def test_different_muscles_on_adjacent_days_are_fine(ont):
    v, _f, _c = plan_guard(_plan((1, "Skull Crusher"), (2, "Hip Thrust")), ont)
    assert v == []


def test_non_adjacent_day_numbers_are_not_compared(ont):
    """Rows numbered 1 and 3 are not consecutive days."""
    v, _f, _c = plan_guard(_plan((1, "Barbell Curl"), (3, "Machine Curl")), ont)
    assert v == []


# ── Inertness ────────────────────────────────────────────────────────────────

def test_inert_without_a_plan(ont):
    for text in ("You did 82 sets of curls in the last 90 days.",
                 "Your Barbell Curl has stalled for 6 sessions.",
                 ""):
        assert plan_guard(text, ont) == ([], [], [])


def test_inert_without_an_ontology():
    plan = _plan((1, "Barbell Curl"), (2, "Machine Curl"))
    for bad in ({}, None, {"muscles": {}}):
        assert plan_guard(plan, bad) == ([], [], [])


def test_unknown_exercise_names_are_ignored(ont):
    """Closed vocabulary: a lift the store has never heard of contributes
    nothing rather than guessing at its muscles."""
    v, _f, _c = plan_guard(_plan((1, "Zercher Wobble"), (2, "Gironda Dip")), ont)
    assert v == []


def test_flags_carry_the_detail_needed_to_repair(ont):
    """The re-prompt names the exact clash, so the flag must carry it."""
    _v, flags, _c = plan_guard(_plan((2, "Barbell Curl"), (3, "Machine Curl")), ont)
    f = flags[0]
    assert f["kind"] == "plan_consecutive_days"
    assert f["exercise_a"] == "Barbell Curl" and f["exercise_b"] == "Machine Curl"
    assert "direct work needs a day between" in f["reason"]


# ── A NUMBERED LIST IS NOT A PLAN ────────────────────────────────────────────
#
# Found while trying to make test_inert_without_a_plan fail. The first version
# treated ANY line-leading digit as a day, so a ranked list of the user's most
# used lifts read as a multi-day plan and flagged a clash — in an answer with no
# plan in it. That would have fired a pointless re-prompt and appended a nonsense
# "this plan still has..." note to a straight answer.
#
# A day now needs the WORD day, or a table row under a declared Day column.

def test_a_ranked_list_is_not_a_plan(ont):
    text = ("Your most-used arm lifts:\n"
            "1. Machine Curl - 416 sets\n"
            "2. Barbell Curl - 113 sets")
    assert plan_guard(text, ont) == ([], [], [])


def test_numbered_advice_is_not_a_plan(ont):
    text = ("Two changes to make:\n"
            "1. Move Barbell Curl earlier in the session.\n"
            "2. Keep Machine Curl at the end.")
    assert plan_guard(text, ont) == ([], [], [])


def test_the_word_day_makes_it_a_plan(ont):
    """Both heading styles the model actually produces."""
    for text in ("### Day 1\nBarbell Curl 4 sets\n### Day 2\nMachine Curl 4 sets",
                 "**Day 1** Barbell Curl\n**Day 2** Machine Curl"):
        v, _f, _c = plan_guard(text, ont)
        assert len(v) == 1, text


def test_a_day_column_makes_numbered_rows_days(ont):
    """The live failure's shape: only the HEADER says Day; the rows are bare
    numbers. That must still be read as a plan."""
    text = ("| Day | Focus | Work |\n|:--- |:--- |:--- |\n"
            "| 4 | Arms | Skull Crusher, Barbell Curl |\n"
            "| 5 | Back | Machine Curl |")
    v, _f, _c = plan_guard(text, ont)
    assert len(v) == 1 and "Biceps" in v[0]


def test_a_table_without_a_day_column_is_not_a_plan(ont):
    text = ("| Rank | Exercise |\n|:--- |:--- |\n"
            "| 1 | Barbell Curl |\n| 2 | Machine Curl |")
    assert plan_guard(text, ont) == ([], [], [])


# ── Weekday names, ancestry, and secondary interference ──────────────────────
#
# All three plans in the live check used Mon/Tue/Thu/Fri, so the guard scored
# ZERO on three plans that every one of them violated. It would also have missed
# the arms clash regardless: Friday's Skull Crusher hits Triceps Long Head,
# Saturday's Close Grip Bench hits Triceps — parent and child, compared as
# strings.

def _wk(*rows):
    head = "| Day | Work |\n|:--- |:--- |\n"
    return head + "\n".join(f"| {d} | {w} |" for d, w in rows)


def test_weekday_names_are_days(ont):
    v, _f, _c = plan_guard(_wk(("Mon", "Barbell Curl"), ("Tue", "Machine Curl")), ont)
    assert len(v) == 1 and "Biceps" in v[0]


def test_weekday_gaps_are_respected(ont):
    """THE NEGATIVE. Tue and Thu are not consecutive — Wednesday sits between."""
    v, _f, _c = plan_guard(_wk(("Tue", "Barbell Curl"), ("Thu", "Machine Curl")), ont)
    assert v == []


def test_long_and_short_weekday_names_both_work(ont):
    v, _f, _c = plan_guard(_wk(("Monday", "Barbell Curl"),
                               ("Tuesday", "Machine Curl")), ont)
    assert len(v) == 1


def test_a_parent_and_child_muscle_clash(ont):
    """Skull Crusher -> Triceps Long Head, Close Grip Bench -> Triceps. Different
    names, overlapping tissue: the child sits inside the parent."""
    v, _f, _c = plan_guard(_wk(("Fri", "Skull Crusher"),
                               ("Sat", "Close Grip Bench")), ont)
    assert len(v) == 1 and "Triceps" in v[0]


def test_sibling_muscles_do_not_clash(ont):
    """THE NEGATIVE THAT MATTERS. Biceps and Triceps share the Arms ancestor but
    are entirely different muscles. Matching on a COMMON ancestor instead of an
    ancestor-of relationship would flag every arm split ever written."""
    v, _f, _c = plan_guard(_wk(("Mon", "Barbell Curl"),
                               ("Tue", "Skull Crusher")), ont)
    assert v == []


def test_secondary_then_primary_is_a_concern_not_a_violation(ont):
    """The arms-day-after-chest-and-back case. Chin Up works the biceps as a
    SECONDARY muscle; the next day targets them directly. Reported, not blocked —
    push/pull/legs on consecutive days is legitimate."""
    v, flags, c = plan_guard(_wk(("Mon", "Chin Up"), ("Tue", "Barbell Curl")), ont)
    assert v == []                       # not a violation
    assert len(c) == 1 and "secondary" in c[0]
    assert flags[0]["severity"] == "concern"
    assert flags[0]["kind"] == "plan_secondary_interference"


def test_a_violation_outranks_a_concern_for_the_same_pair(ont):
    """Direct-on-direct must not ALSO be reported as a soft concern."""
    v, flags, c = plan_guard(_wk(("Mon", "Barbell Curl"),
                                 ("Tue", "Machine Curl")), ont)
    assert len(v) == 1
    assert all(f["severity"] == "violation" for f in flags)


def test_longest_exercise_name_wins_its_span(ont):
    """Plain substring matching bled: live, "Incline Smith Machine Press"
    contained "Smith Machine Press", so an incline day (Upper Chest) also
    counted as the flat lift (Mid Chest) and two days sharing no muscle at all
    were reported as a clash.

    Here "Preacher Machine Curl" contains "Machine Curl". It must map to
    Brachialis ONLY — never also to Biceps — so the next day's Barbell Curl is
    not a clash."""
    v, _f, _c = plan_guard(_wk(("Mon", "Preacher Machine Curl"),
                               ("Tue", "Barbell Curl")), ont)
    assert v == []
