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
    # The REAL home of the session rate. This fixture once said
    # `training_frequency` — a key that exists only per exercise — so the guard
    # read a field no real package has and its protection was silently off
    # (live re-check, 2026-09-16). test_fixture_fields_exist_in_a_real_package
    # now holds every key here to a package built from the DB.
    "training_consistency": {"sessions_per_week": 3.2},
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
    # The user's OWN spellings, as the DB holds them. This fixture once listed the
    # graph names here ("Hamstring Curl Machine", "T-Bar Barbell Row"), so the
    # tests asserted that the user's real names were "wrong" — the B3 misdiagnosis.
    "exercises": [{"name": "Hamstring Curls Machine"}, {"name": "T Bar Barbell Row"},
                  {"name": "Incline Smith Machine Press"}, {"name": "Smith Machine Press"},
                  # A real per-exercise SESSION rate, deliberately equal to the
                  # Hamstrings SET rate — the collision 24 of 47 real exercises have.
                  {"name": "Deadlift", "training_frequency": {"sessions_per_week": 1.8}}],
    "suggestable_exercises": ["Cable Crunch"],
}
def _ont(entries):
    """An ontology shaped as load_ontology builds it: canonical names by id,
    aliases keyed lowercase, alias_names keeping the user's spelling."""
    o = {"exercises": {}, "aliases": {}, "alias_names": {}}
    for eid, (canonical, user_spelling) in enumerate(entries, start=1):
        o["exercises"][eid] = {"canonical_name": canonical}
        if user_spelling:
            o["aliases"][user_spelling.lower()] = eid
            o["alias_names"][user_spelling.lower()] = user_spelling
    return o


ONT = _ont([
    ("Hamstring Curl Machine",       "Hamstring Curls Machine"),
    ("T-Bar Barbell Row",            "T Bar Barbell Row"),
    ("Reverse Cable Curl",           "Reverse Cable Curls"),
    ("Barbell Curl",                 "Barbell Curl"),
    ("Reverse EZ-Bar Curl",          "Reverse Zig Zag Barbell Curls"),
    ("Push Up",                      "Pushups"),
    ("Dumbbell Skull Crusher",       "dumbbell skull crusher"),   # differs only in capitals
    ("Incline Smith Machine Press",  "Incline Smith Machine Press"),
    ("Smith Machine Bench Press",    "Smith Machine Press"),
    ("Lateral Dumbbell Raise",       None),                        # graph only, never logged
    ("Seated Machine Curl",          "Seated Machine Curl (Kg)"),  # brackets in the user's name
    ("Running (Outdoor)",            None),                        # brackets in the graph's name
])


# ── B1 · sets are never sessions ──────────────────────────────────────────────

def test_sets_rate_called_sessions_is_rewritten():
    out, flags = cite.sets_as_sessions_guard(
        "Your current leg training frequency is low at 5.3 sessions per week.", PKG)
    assert out == "Your current leg training volume is low at 5.3 sets per week."
    assert flags and flags[0]["kind"] == "sets_as_sessions"


def test_the_real_session_rate_is_left_alone():
    """The hard case: the overall session rate equals the NAMED muscle's set rate."""
    pkg = dict(PKG, training_consistency={"sessions_per_week": 4.6})   # = Mid Traps
    text = "You train 4.6 sessions per week on average, mid traps included."
    assert cite.sets_as_sessions_guard(text, pkg) == (text, [])


# A NUMBER ALONE IS NOT EVIDENCE. A real package holds 49 muscle set rates spread
# from 0.2 to 20.2 and a real session rate for every exercise; 24 of 47 exercises
# collide with some muscle's set rate. Deciding from the number turned the true
# "Lat Pulldown about 1.2 sessions per week" into "1.2 sets" (1.2 was the Wrist
# Extensors' rate). Each test below isolates one of the four conditions.

def test_a_set_rate_with_no_muscle_named_is_left_alone():
    text = "Overall you are at 5.3 sessions per week."          # 5.3 = Legs, unnamed
    assert cite.sets_as_sessions_guard(text, PKG) == (text, [])


def test_another_muscles_set_rate_is_left_alone():
    text = "Your hamstring frequency is 5.3 sessions per week."  # 5.3 is Legs', not Hamstrings'
    assert cite.sets_as_sessions_guard(text, PKG) == (text, [])


def test_a_named_exercises_real_session_rate_is_left_alone():
    text = "Deadlift hits your hamstrings 1.8 sessions per week."  # Deadlift's own rate
    assert cite.sets_as_sessions_guard(text, PKG) == (text, [])


def test_the_named_muscles_own_rate_is_still_rewritten():
    """The positive for the three negatives above: same sentence shapes, all four
    conditions met."""
    out, flags = cite.sets_as_sessions_guard(
        "Your hamstring frequency is 1.8 sessions per week.", PKG)
    assert out == "Your hamstring volume is 1.8 sets per week."
    assert flags


# ── Fixtures must have the shape of a REAL package ────────────────────────────

def _key_paths(obj, prefix=""):
    """Every dict key path in obj; a list contributes its first element's keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            yield path
            yield from _key_paths(v, path)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        yield from _key_paths(obj[0], prefix + "[]")


@pytest.fixture(scope="module")
def real_package():
    from src.data_agent import prepare_analysis_package
    return prepare_analysis_package(query_period_days=90)


def test_fixture_fields_exist_in_a_real_package(real_package):
    """A synthetic fixture may simplify VALUES, never invent FIELDS. Every guard
    here was green against a package shape the data agent never produces."""
    real = set(_key_paths(real_package))
    invented = sorted(p for p in _key_paths(PKG) if p not in real)
    assert invented == [], f"fixture keys no real package has: {invented}"


def test_session_rate_is_protected_on_a_real_package(real_package):
    """The hard case on the real shape: the user's genuine session rate equals
    the named muscle's sets-per-week figure. It must not be renamed to sets."""
    row = real_package["muscle_ontology_summary"]["muscles"][0]
    rate = row["primary_sets_per_week"]
    pkg = dict(real_package, training_consistency=dict(
        real_package["training_consistency"], sessions_per_week=rate))
    text = f"You train {rate} sessions per week on average, {row['muscle']} included."
    assert cite.sets_as_sessions_guard(text, pkg) == (text, [])


def test_real_per_exercise_session_rates_are_left_alone(real_package):
    """Every exercise's real session rate, stated in the sentence shape that
    tripped the old guard: the exercise AND a muscle whose set rate collides."""
    rows = cite._muscle_rows(real_package)
    cases = []
    for ex in real_package["exercises"]:
        rate = (ex.get("training_frequency") or {}).get("sessions_per_week")
        muscle = next((r["muscle"] for r in rows
                       if rate is not None and any(cite._close(rate, r.get(k)) for k in
                                                   ("primary_sets_per_week",
                                                    "secondary_sets_per_week"))), None)
        if muscle:
            cases.append(f"You do {ex['name']} about {rate} sessions per week for your {muscle}.")
    assert cases, "no colliding rate in the real package — this test would prove nothing"
    changed = [t for t in cases if cite.sets_as_sessions_guard(t, real_package)[0] != t]
    assert changed == [], f"{len(changed)}/{len(cases)} real session rates renamed to sets: {changed[:3]}"


# ── Every field a guard reads must exist in a real package ────────────────────

class _Recording(dict):
    """A package that records every key looked up and not found."""
    def __init__(self, data, path, misses):
        super().__init__({k: _record(v, f"{path}.{k}" if path else k, misses)
                          for k, v in data.items()})
        self._path, self._misses = path, misses

    def _note(self, key):
        if key not in self:
            self._misses.add(f"{self._path}.{key}" if self._path else key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)


def _record(value, path, misses):
    if isinstance(value, dict):
        return _Recording(value, path, misses)
    if isinstance(value, list):
        return [_record(v, path + "[]", misses) for v in value]
    return value


# Allowed misses — each a real per-kind difference, never a field nobody produces:
#   a cardio exercise has no progression session date; its date is the top-level
#     `last_session_date`, read next;
#   a cardio exercise has no `training_frequency` (only session counts), so it has
#     no session rate for sets_as_sessions_guard to protect.
_ALLOWED_MISSES = {"exercises[].progression.latest_session_date",
                   "exercises[].training_frequency"}

_PATH_SENTENCES = [
    "Your most recent session was on 2020-01-01.",
    "Your most recent Sumo Squats session was on 2020-01-01.",
    "Your leg training frequency is 6.0 sessions per week.",
    "Looking at your 90-day recovery trends.",
    "Add Barbell Curls and cable crunches.",
    "You currently perform 25 sets for Chest, 28 for Back.",
    "Plan: bring up Hamstrings from 2.0 sets per week.\nMonday: Lat Pulldown\nTuesday: Barbell Curl",
    "Your Lat Pulldowns do train your biceps.",
]


@pytest.mark.parametrize("scope", [{}, {"exercise_names": ["Sumo Squats"]},
                                   {"exercise_names": ["Walking", "Cycling"]}],
                         ids=["broad", "one-exercise", "cardio"])
def test_guards_read_only_fields_a_real_package_has(scope):
    """STRUCTURAL. Two guards read `training_frequency` at the top level of the
    package — a key that exists only per exercise — so one's protection and the
    other's correction were silently off, and every synthetic test stayed green.
    This runs each guard over a REAL package and fails on any field it asks for
    that the data agent never produces."""
    from src.data_agent import prepare_analysis_package
    from src.ontology import load_ontology
    misses: set = set()
    pkg = _record(prepare_analysis_package(query_period_days=90, **scope), "", misses)
    ontology = load_ontology()
    for text in _PATH_SENTENCES:
        cite.recency_guard(text, pkg)
        cite.limiting_claim_guard(text, pkg)
        cite.sets_as_sessions_guard(text, pkg)
        cite.window_label_guard(text, pkg)
        cite.exercise_name_guard(text, pkg, ontology)
        cite.set_count_guard(text, pkg)
        cite.target_guard(text, pkg)
    assert sorted(misses - _ALLOWED_MISSES) == []


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
    ("Hamstring Curl Machines", "Hamstring Curls Machine"),     # → the USER's spelling
    ("Pushup", "Pushups"),                                       # → the USER's spelling
    ("Barbell Curls", "Barbell Curl"),                           # the live correction
    ("lateral dumbbell raises", "Lateral Dumbbell Raise"),       # graph only
    ("cable crunches", "Cable Crunch"),                          # from suggestable
    ("Dumbbell Skull Crushers", "Dumbbell Skull Crusher"),       # user's differs only in capitals
])
def test_near_miss_names_are_rewritten(wrong, right):
    out, flags = cite.exercise_name_guard(f"Add {wrong} on Tuesday.", PKG, ONT)
    assert out == f"Add {right} on Tuesday."
    assert flags[0]["original"] == wrong


def test_exact_names_are_left_alone():
    """The user's spellings AND the graph's names, both written exactly."""
    text = ("Keep Hamstring Curls Machine and T Bar Barbell Row; "
            "Hamstring Curl Machine and T-Bar Barbell Row are the same lifts.")
    assert cite.exercise_name_guard(text, PKG, ONT) == (text, [])


def test_your_spelling_outside_the_package_is_left_alone():
    """THE REGRESSION (live re-check, 2026-09-16). A hamstring plan's package did
    not hold Reverse Cable Curls or T Bar Barbell Row, so the guard "corrected"
    the user's own logged names to the graph's. The package is not the only
    record of how the user spells things."""
    pkg = dict(PKG, exercises=[e for e in PKG["exercises"] if e["name"] != "T Bar Barbell Row"])
    text = "Add Reverse Cable Curls after T Bar Barbell Row."
    assert cite.exercise_name_guard(text, pkg, ONT) == (text, [])


@pytest.mark.parametrize("wrong, right", [
    ("Seated Machine Curls (Kg)", "Seated Machine Curl (Kg)"),
    ("Running (Outdoors)", "Running (Outdoor)"),
])
def test_names_with_brackets_are_corrected_once(wrong, right):
    """The matcher used to stop at a bracket: it matched only "Seated Machine
    Curls", corrected that to the user's "Seated Machine Curl (Kg)", and the
    "(Kg)" already in the text followed — "Seated Machine Curl (Kg) (Kg)"."""
    out, _ = cite.exercise_name_guard(f"Add {wrong} today.", PKG, ONT)
    assert out == f"Add {right} today."


def test_names_with_brackets_written_exactly_are_left_alone():
    text = "Add Seated Machine Curl (Kg) and Running (Outdoor) today."
    assert cite.exercise_name_guard(text, PKG, ONT) == (text, [])


def test_a_longer_name_of_yours_is_not_partly_rewritten():
    """"Reverse Zig Zag Barbell Curls" contains "Barbell Curls". Matched alone,
    that became "Reverse Zig Zag Barbell Curl"."""
    text = "Add Reverse Zig Zag Barbell Curls."
    assert cite.exercise_name_guard(text, PKG, ONT) == (text, [])


# ── On the REAL store and DB ──────────────────────────────────────────────────

def _db_names(logged_only=False):
    import os
    import sqlite3
    con = sqlite3.connect(f"file:{os.environ['FITNOTES_DB_PATH']}?mode=ro", uri=True)
    try:
        sql = ("SELECT DISTINCT e.name FROM training_log t JOIN exercise e ON e._id = t.exercise_id"
               if logged_only else "SELECT name FROM exercise")
        return sorted({r[0] for r in con.execute(sql)})
    finally:
        con.close()


@pytest.fixture(scope="module")
def one_exercise_package():
    """Like the live hamstring plan: a package holding almost none of the user's lifts."""
    from src.data_agent import prepare_analysis_package
    return prepare_analysis_package(query_period_days=90, exercise_names=["Sumo Squats"])


@pytest.fixture(scope="module")
def real_ontology():
    from src.ontology import load_ontology
    return load_ontology()


def test_no_real_exercise_name_is_changed(one_exercise_package, real_ontology):
    """Every exercise name in the user's DB (logged or not) and every alias
    spelling, written exactly, comes back exactly — whatever the package holds."""
    names = set(_db_names()) | set(real_ontology["alias_names"].values())
    changed = {}
    for name in sorted(names):
        text = f"Add {name} (3 sets)."
        out, _ = cite.exercise_name_guard(text, one_exercise_package, real_ontology)
        if out != text:
            changed[name] = out
    assert changed == {}, f"{len(changed)} real names rewritten: {list(changed.items())[:5]}"


def test_real_near_misses_come_back_as_your_spelling(one_exercise_package, real_ontology):
    """A plural added or dropped on any word of a logged name is corrected to the
    user's own spelling — or to the graph's capitals when that is the only
    difference. Never to a different exercise, never to the graph's wording."""
    ont = real_ontology
    canonical = {eid: e["canonical_name"] for eid, e in ont["exercises"].items()}
    wrong = []
    checked = 0
    for name, key in sorted((n, n.lower()) for n in ont["alias_names"].values()):
        eid = ont["aliases"][key]
        expected = (canonical[eid] if name.lower() == canonical[eid].lower() else name)
        words = name.split()
        for i, w in enumerate(words):
            if len(w) <= 2 or not w.isalpha():
                continue
            v = w[:-1] if w.lower().endswith("s") and not w.lower().endswith("ss") else w + "s"
            variant = " ".join(words[:i] + [v] + words[i + 1:])
            out, flags = cite.exercise_name_guard(f"Add {variant} (3 sets).", one_exercise_package, ont)
            if not flags:
                continue
            checked += 1
            if out != f"Add {expected} (3 sets).":
                wrong.append((variant, out[4:-10], expected))
    assert checked > 100, f"only {checked} variants were corrected — the sweep proves little"
    assert wrong == [], f"{len(wrong)} near-misses not returned as the user's spelling: {wrong[:5]}"


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


# ── Bold is where key figures live ────────────────────────────────────────────
#
# The prompt tells the model to "**bold** key figures", and the number guards
# matched only plain text: "You currently perform **25** sets for Chest" slipped
# past the set-count guard, "**5.3** sessions" was never corrected, and a plan
# quoting "**1.8** sets" got no target. The guards now read the text without
# the markers and put every edit back where it belongs, bold intact.

@pytest.mark.parametrize("text", [
    "You currently perform **25** sets for Chest.",
    "You currently perform **25 sets** for Chest.",
    "**You currently perform 25 sets for Chest.**",
])
def test_a_bold_invented_set_count_is_caught(text):
    violations, _ = cite.set_count_guard(text, PKG)
    assert violations and "Chest" in violations[0]


@pytest.mark.parametrize("text, expected", [
    ("Your current leg training frequency is low at **5.3** sessions per week.",
     "Your current leg training volume is low at **5.3** sets per week."),
    ("Your current leg training frequency is low at **5.3 sessions** per week.",
     "Your current leg training volume is low at **5.3 sets** per week."),
])
def test_a_bold_set_rate_called_sessions_is_corrected_and_stays_bold(text, expected):
    out, flags = cite.sets_as_sessions_guard(text, PKG)
    assert out == expected and flags


@pytest.mark.parametrize("figure, expected", [
    ("**1.8** sets per week",
     "**1.8** sets per week (the usual target is 10–20 sets a week per muscle)"),
    ("**1.8 sets per week**",
     "**1.8 sets per week** (the usual target is 10–20 sets a week per muscle)"),
])
def test_a_bold_figure_in_a_plan_still_gets_its_target(figure, expected):
    out, flags = cite.target_guard(PLAN.replace("1.8 sets per week", figure), PKG)
    assert expected in out and flags


@pytest.mark.parametrize("text, expected", [
    ("Your **90-day** recovery trends are flat.",
     "Your recovery trends over the last 90 days are flat."),
    ("Your **90-day recovery trends** are flat.",
     "Your **recovery trends over the last 90 days** are flat."),
])
def test_a_bold_window_label_is_caught_without_a_false_capital(text, expected):
    out, flags = cite.window_label_guard(text, PKG)
    assert out == expected and flags


# ── Idempotence: a rewrite must not re-trigger on its own output ──────────────

@pytest.mark.parametrize("guard, text", [
    (lambda t: cite.sets_as_sessions_guard(t, PKG), "Legs sit at 5.3 sessions per week."),
    (lambda t: cite.window_label_guard(t, PKG), "Your 90-day recovery trends are fine."),
    (lambda t: cite.exercise_name_guard(t, PKG, ONT), "Add Hamstring Curl Machines."),
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
    # Both retries (set count, plan) go through the one checked redraft, which
    # runs the rewrite guards and every other first-draft check;
    # tests/test_guard_retries.py proves what it does.
    assert src.count("await self._redraft_checked(") == 2


def test_every_guard_is_isolated():
    src = _coordinator_src()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_run_rewrite_guards")
    assert any(isinstance(n, ast.Try) for n in ast.walk(fn))
    assert "set-count guard skipped" in src and "target guard skipped" in src
