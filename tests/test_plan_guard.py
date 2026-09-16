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
# The user logs Hip Thrust under their own spelling; every other alias is the
# graph name. A note about a clash must use the name the user knows.
_ALIASES = [(("Lying Hip Thrusts" if n == "Hip Thrust" else n), i) for i, n, *_ in _EXERCISES]


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
    assert "Biceps" in v[0] and "Day 4" in v[0] and "Day 5" in v[0]
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


# ══════════════════════════════════════════════════════════════════════════════
# A DAY ENDS WHERE THE PLAN'S LAYOUT ENDS (live re-check, 2026-09-16)
# ══════════════════════════════════════════════════════════════════════════════
#
# A day used to run to the next day label, so the LAST day ran to the end of the
# answer. The prompt requires "EXPLAIN THE STRUCTURE" after every plan, and that
# explanation names every day's exercises — so a plan whose last day was arm work
# was reported as hamstrings on "day 5 and day 6", re-prompted, and shipped with
# a false "Note: this plan still has…".

_WEEK = ("### Your week\n\n"
         "* **Monday: Pull**\n * Chin Up (4 sets)\n"
         "* **Tuesday: Arms**\n * Barbell Curl (4 sets)\n"
         "* **Wednesday: Legs**\n * Hip Thrust (4 sets)\n")
_EXPLANATION = ("\n### Structure\n"
                "* **Spacing:** Barbell Curl sits on Tuesday so Machine Curl never "
                "follows it; Hip Thrust closes the week.\n")


def test_the_explanation_after_a_plan_is_not_part_of_the_last_day(ont):
    """Wednesday is legs only. The explanation names Barbell Curl and Machine
    Curl — read as Wednesday, that was a Tuesday/Wednesday biceps clash."""
    v, _f, _c = plan_guard(_WEEK + _EXPLANATION, ont)
    assert v == []


def test_a_real_clash_on_the_last_day_is_still_caught(ont):
    plan = _WEEK.replace(" * Hip Thrust (4 sets)\n", " * Machine Curl (4 sets)\n")
    v, _f, _c = plan_guard(plan + _EXPLANATION, ont)
    assert len(v) == 1 and "Biceps" in v[0]


def test_a_table_row_is_its_own_day(ont):
    """The live table shape: Day 7 is rest, and the reasons below name Day 6's lift."""
    text = ("| Session | Focus | Exercises |\n|:--- |:--- |:--- |\n"
            "| Day 1 | Legs | Hip Thrust (4 sets) |\n"
            "| Day 6 | Arms | Barbell Curl (4 sets) |\n"
            "| Day 7 | Rest | Complete rest |\n\n"
            "### Reasons for This Plan\n- Barbell Curl on Day 6 keeps arm work late.")
    v, _f, _c = plan_guard(text, ont)
    assert v == []


def test_a_table_row_ends_at_its_line_even_with_prose_right_below(ont):
    """No blank line between the table and the text under it: the row still ends
    at its own line, so the prose naming Day 6's lift is not read as Day 7."""
    text = ("| Day | Work |\n|:--- |:--- |\n"
            "| 6 | Barbell Curl (4 sets) |\n"
            "| 7 | Rest |\n"
            "Barbell Curl sits on Day 6 so the week ends on rest.")
    v, _f, _c = plan_guard(text, ont)
    assert v == []


def test_heading_days_keep_content_after_a_blank_line(ont):
    """A heading day's exercises often sit after a blank line. Cutting the day at
    the blank line would silently miss this real clash."""
    text = ("### Day 1\n\nBarbell Curl 4 sets\n\nGo heavy.\n\n"
            "### Day 2\n\nMachine Curl 4 sets\n\n## Why this works\nHip Thrust later.")
    v, _f, _c = plan_guard(text, ont)
    assert len(v) == 1 and "Biceps" in v[0]


_RECAP = ("Last week you logged:\n\n"
          "* **Monday (2026-09-07):** Barbell Curl (4 sets)\n"
          "* **Tuesday (2026-09-08):** Machine Curl (4 sets)\n")


def test_dated_days_are_logged_training_not_a_plan(ont):
    """What the user DID is not a plan to judge. A recap of their own sessions
    was reported as a biceps clash."""
    assert plan_guard(_RECAP, ont) == ([], [], [])


def test_a_recap_before_a_plan_leaves_only_the_plan_judged(ont):
    text = _RECAP + ("\nNext week:\n\n* **Monday:** Hip Thrust (4 sets)\n"
                     "* **Tuesday:** Chin Up (4 sets)\n")
    v, _f, _c = plan_guard(text, ont)
    assert v == []


def test_a_weekday_line_inside_the_explanation_does_not_join_the_plan(ont):
    """"Thursday" opens a line in the prose after the plan. It is not a day that
    follows Wednesday."""
    text = (_WEEK + "\nThis keeps the week balanced.\n\n"
            "Thursday stays free; do Hip Thrust again only if recovered.\n")
    v, _f, _c = plan_guard(text, ont)
    assert v == []


# ── The note names days and exercises the way the user reads them ────────────

def test_the_clash_names_the_plans_own_days_and_the_users_spelling(ont):
    text = ("* **Friday:** Hip Thrust (4 sets)\n"
            "* **Saturday/Sunday:** Hip Thrust (3 sets)\n")
    v, flags, _c = plan_guard(text, ont)
    assert v == ["Lying Hip Thrusts on Friday and Lying Hip Thrusts on "
                 "Saturday/Sunday both train Glutes directly"]
    assert (flags[0]["label_a"], flags[0]["label_b"]) == ("Friday", "Saturday/Sunday")
    assert (flags[0]["day_a"], flags[0]["day_b"]) == (5, 6)


# ── The live answer itself, against the real store ───────────────────────────

_LIVE_PROMPT_1 = """To increase your hamstring volume to the target of 10 sets per week, this plan splits that direct work across two separate days. I have sequenced the exercises to prevent any muscle group from receiving direct primary training on consecutive days, while addressing your concerns regarding secondary muscle involvement.

### Weekly Training Plan

This 5-day split spaces your hamstring work and ensures that your pressing movements do not immediately stress your front delts before their primary training day.

* **Monday: Back Primary**
 * Lat Pulldown (4 sets)
 * T-Bar Barbell Row (4 sets)
 * Wide Grip Cable Row (4 sets)
* **Tuesday: Hamstring Focus (Direct Work)**
 * Hamstring Curls Machine (5 sets)
* **Wednesday: Chest Primary**
 * Flat Dumbbell Bench Press (4 sets)
 * Incline Dumbbell Bench Press (4 sets)
 * Seated Machine Fly (4 sets)
* **Thursday: Shoulder Primary (Front/Side/Rear)**
 * Seated Dumbbell Press (4 sets)
 * Lateral Dumbbell Raise (4 sets)
 * Rear Delt Machine Fly (4 sets)
* **Friday: Hamstring Focus (Direct Work) & Leg Maintenance**
 * Hamstring Curls Machine (5 sets)
 * Sumo Squats (4 sets)
 * Leg Extension Machine (4 sets)
* **Saturday/Sunday: Active Recovery/Arms**
 * *Optional:* Barbell Curl (3 sets) and dumbbell skull crusher (3 sets)

### Structure and Sequencing
* **Hamstring Dose:** You are moving from **2.0** to **10 sets** per week. By placing these sessions on Tuesday and Friday, you provide 48 hours of recovery between direct bouts.
* **Pressing Sequencing:** In the previous design, the overlap between Incline Bench Press (secondary front delt work) and Seated DB Press (direct front delt work) was problematic. By placing your chest primary work on Wednesday and your shoulder primary work on Thursday, you are technically working the front delts on back-to-back days. However, because Incline Dumbbell Bench Press uses the front delts only as a secondary muscle, the stress is significantly lower than the primary volume on Thursday. If your front delts feel fatigued, you may swap Monday and Wednesday to create a full day of rest between these pressing movements.
* **Maintenance:** Your back volume remains aligned with your current **18.0**, and chest volume stays consistent with your **11.4**.
* **Leg Maintenance:** Sumo Squats and Leg Extensions are kept on your second hamstring day to ensure your quads continue to receive maintenance volume without interfering with your fresh hamstring performance on Tuesday.

This plan uses only exercises you have already logged. Monitor your performance on the Hamstring Curls Machine; as your volume increases, focus on full-range execution to avoid the increasing reliance on partial repetitions seen in your recent training data."""


def test_live_prompt_1_has_no_false_clash(monkeypatch):
    """The exact answer from the 2026-09-16 re-check, on the real store (read
    only). It reported four clashes on "day 5 and day 6"; its real hamstring days
    are Tuesday and Friday."""
    import os
    monkeypatch.setenv("ONTOLOGY_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ontology"))
    ont_mod.clear_cache()
    try:
        v, _f, _c = plan_guard(_LIVE_PROMPT_1, ont_mod.load_ontology(force=True))
    finally:
        ont_mod.clear_cache()
    assert v == []


# ══════════════════════════════════════════════════════════════════════════════
# THE GUARD JUDGES ONLY PLANS THE USER ASKED FOR
# ══════════════════════════════════════════════════════════════════════════════
#
# It ran on every analysis answer, so a recap of the user's own week was reported
# as a scheduling error, and a split the user asked for on purpose (the same
# muscle every day) was overridden. The user's QUESTION decides: a plan is judged
# only when they asked the coach to build one, and not when they asked for
# back-to-back training. Real wordings from earlier live checks.

def is_plan_request(question):
    from src.citations import is_plan_request as detector
    return detector(question)


def asks_for_consecutive(question):
    from src.citations import asks_for_consecutive as detector
    return detector(question)


@pytest.mark.parametrize("question", [
    "plan me a week that brings up my hamstrings, keep everything else steady",
    "build me a week of training that pushes and focuses my arms without dropping anything else",
    "build me a week of training that pushes my calves and rear delts without dropping anything else",
    "give me a 4 day split focused on chest",
    "Make me a one week plan that helps me reach this goal and also, give reasons for what the plan",
    "plan my next triceps session",
    "show my last week, then plan next week",
    "can you design a new push pull legs routine for me?",
    "write me a three-day program",
])
def test_a_plan_request_is_recognised(question):
    assert is_plan_request(question)


@pytest.mark.parametrize("question", [
    "how was my back ROM split in the last back session",
    "Should I take a deload week?",
    "how many sets per week am I doing for my mid traps?",
    "How many sets per week is optimal for triceps?",
    "how has my Seated Machine Fly gone over the past 3 weeks?",
    "what does the research say about training a muscle twice a week?",
    "what split am I running?",
    "give me a summary of my week",
    "show me my last week",
    "give me a breakdown of last week's sessions",
    "is my plan working?",
    "how did my last session go?",
    "make a note that my shoulder hurt this week",
    "log 3 sets of curls for today",
])
def test_other_questions_are_not_plan_requests(question):
    assert not is_plan_request(question)


@pytest.mark.parametrize("question, expected", [
    ("build me a week that trains arms every day", True),
    ("give me a split with biceps daily", True),
    ("plan back-to-back leg days for me", True),
    ("make me a week that hits chest two days in a row", True),
    ("plan me a week that brings up my hamstrings, keep everything else steady", False),
    ("give me a 4 day split focused on chest", False),
    ("how many days a week do I train?", False),
])
def test_back_to_back_requests_are_recognised(question, expected):
    assert asks_for_consecutive(question) is expected


# ── Wiring: the coordinator asks the question before judging the answer ──────

_CLASH = "* **Monday:** Barbell Curl (4 sets)\n* **Tuesday:** Machine Curl (4 sets)\n"


def _run_guard_stage(monkeypatch, question, answer):
    """Drive the real _stage_display_fidelity with a fake cache; record whether
    the plan guard's re-prompt ran."""
    import asyncio
    from types import SimpleNamespace
    from src import analysis_agent
    from src.coordinator import Coordinator

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")    # client construction only; no call is made
    calls = []

    async def fake_analyze(*args, **kwargs):
        calls.append(args)
        return "* **Monday:** Barbell Curl (4 sets)\n* **Wednesday:** Machine Curl (4 sets)\n"

    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)
    pkg = {"exercises": [], "query_period_days": 90}

    async def ensure_package(_coordinator, _state):
        return pkg

    cache = SimpleNamespace(ensure_package=ensure_package, research=None, memories=None,
                            conversation_context=None, custom_query=None)
    state = {"question": question, "scoped_question": question, "answer": answer,
             "params": {"query_period_days": 90}}
    out = asyncio.run(Coordinator(agent_session=None)._stage_display_fidelity(state, cache))
    return out["answer"], calls


def test_a_recap_is_never_judged_as_a_plan(ont, monkeypatch):
    answer, calls = _run_guard_stage(monkeypatch, "what did I train last week?", _CLASH)
    assert calls == [] and answer == _CLASH


def test_a_requested_plan_is_judged(ont, monkeypatch):
    _answer, calls = _run_guard_stage(monkeypatch, "build me a week of arm training", _CLASH)
    assert len(calls) == 1


def test_a_requested_back_to_back_plan_is_not_overridden(ont, monkeypatch):
    answer, calls = _run_guard_stage(
        monkeypatch, "build me a week that trains arms every day", _CLASH)
    assert calls == [] and answer == _CLASH
