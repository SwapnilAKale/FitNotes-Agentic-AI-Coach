"""
exercise_muscle_map — the package field that answers "does exercise X train
muscle Y?", plus the honesty rules around a named exercise that produced no rows.

WHY THIS FILE EXISTS. A live check asked "do my lat pulldowns train my biceps?"
and got:

    "Note: Lat Pulldown wasn't found in your workout history, so this answer
     covers your overall training instead. Your Lat Pulldown sessions
     contribute to your biceps development as a secondary muscle."

Both halves were false. Lat Pulldown had 366 logged sets, and the graph said it
HOLDS the biceps rather than training them. Three faults stacked:

  1. exercise_names and muscle_groups were ANDed, and FitNotes files Lat Pulldown
     under Back — so "Lat Pulldown AND filed-under-Biceps" was empty BY
     CONSTRUCTION. The cross-category question the ontology exists to answer was
     being killed by the single category the ontology exists to work around.
  2. The package carried per-muscle counts and an `exercise_count` INTEGER — the
     exercise names behind each muscle were dropped, so nothing in the payload
     could answer the question and the agent inferred it from counts belonging to
     other lifts.
  3. The "no rows" fact was hidden from the model and a contradicting note glued
     on afterwards.

Every test here pins one of those.
"""

import csv

import pytest

from src import ontology as ont_mod
from src.coordinator import unresolved_scope_note
from src.data_agent.process import _compute_exercise_muscle_map as build_map

# ── Synthetic store ───────────────────────────────────────────────────────────
# Carries all THREE roles, because the whole point is that they stay distinct.
#
#   Back            Arms
#     └ Lats          ├ Biceps
#                     └ Grip
_MUSCLES = [
    (1, "Back",   "", "large"),
    (2, "Lats",   1,  "large"),
    (3, "Arms",   "", ""),
    (4, "Biceps", 3,  "medium"),
    (5, "Grip",   3,  "small"),
]
_EXERCISES = [
    (1, "Lat Pulldown", "cable",    "vertical pull"),
    (2, "Barbell Curl", "barbell",  "elbow flexion"),
    (3, "Shrug",        "barbell",  "shrug"),
    (4, "Treadmill",    "machine",  "cardio"),
]
_EDGES = [
    (1, 2, "primary",   "test"),   # Lat Pulldown -> Lats
    (1, 4, "limiting",  "test"),   # Lat Pulldown -> Biceps  (HELD, not trained)
    (2, 4, "primary",   "test"),   # Barbell Curl -> Biceps
    (3, 5, "limiting",  "test"),   # Shrug        -> Grip
    (3, 2, "secondary", "test"),   # Shrug        -> Lats
]
_ALIASES = [("Lat Pulldown", 1), ("Barbell Curl", 2), ("Shrug", 3),
            ("Treadmill", 4)]


@pytest.fixture
def ont(tmp_path, monkeypatch):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
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


# A user who trains curls constantly and has not touched a pulldown in months.
LOGGED = {"Barbell Curl": [{"date": "2026-06-01", "working_sets_count": 4}],
          "Lat Pulldown": [{"date": "2025-11-02", "working_sets_count": 5}]}


# ── The role distinction survives into the map ────────────────────────────────

def test_limiting_is_reported_separately_from_secondary(ont):
    """The claim the agent got wrong. Biceps must arrive under `limiting` and
    must NOT appear under secondary, or "trains your biceps" becomes citable."""
    m = build_map(ont, ["Lat Pulldown"], LOGGED)["Lat Pulldown"]
    assert m["limiting"] == ["Biceps"]
    assert m["secondary"] == []
    assert "Biceps" not in m["primary"]
    # And the muscle it genuinely trains is still there.
    assert m["primary"] == ["Lats"]


def test_a_lift_that_trains_the_muscle_is_not_demoted(ont):
    """Both directions: the same muscle, reached by a lift that DOES train it."""
    m = build_map(ont, ["Barbell Curl"], LOGGED)["Barbell Curl"]
    assert m["primary"] == ["Biceps"]
    assert m["limiting"] == []


# ── Window independence — the property that makes a scoping miss survivable ───

def test_map_is_window_independent(ont):
    """"Does X train Y" is a fact about the exercise, not about the last 90
    days. An exercise with NO sessions at all still gets its muscles, which is
    what stops a scoping miss from turning into a guess."""
    m = build_map(ont, ["Lat Pulldown"], {})["Lat Pulldown"]
    assert m["primary"] == ["Lats"] and m["limiting"] == ["Biceps"]
    assert m["in_store"] is True
    assert m["ever_logged"] is False


def test_ever_logged_tracks_history_not_the_window(ont):
    m = build_map(ont, ["Lat Pulldown"], LOGGED)["Lat Pulldown"]
    assert m["ever_logged"] is True


def test_unknown_exercise_is_admitted_never_guessed(ont):
    """in_store=False is a curation gap. Empty role lists mean 'no mapping',
    which the prompt turns into an admission — never invented muscles."""
    m = build_map(ont, ["Zercher Wobble"], LOGGED)["Zercher Wobble"]
    assert m["in_store"] is False
    assert m["primary"] == [] and m["secondary"] == [] and m["limiting"] == []


# ── Bounded payload ───────────────────────────────────────────────────────────

def test_map_is_empty_when_no_exercise_was_named(ont):
    """O(names), never O(graph) — a broad question must not carry the map."""
    assert build_map(ont, None, LOGGED) == {}
    assert build_map(ont, [], LOGGED) == {}


def test_map_covers_only_the_named_exercises(ont):
    m = build_map(ont, ["Shrug"], LOGGED)
    assert set(m) == {"Shrug"}


# ── The scope note tells the truth about which kind of miss it was ────────────

_EMAP_LOGGED = {"Lat Pulldown": {"ever_logged": True, "in_store": True}}
_EMAP_NEVER = {"Barbell Squat": {"ever_logged": False, "in_store": False}}


def test_absent_from_window_is_not_reported_as_absent_from_history():
    """The exact sentence the user was told about a 366-set exercise."""
    note = unresolved_scope_note(["Lat Pulldown"], _EMAP_LOGGED, None, 90)
    assert "haven't done Lat Pulldown in the last 90 days" in note
    assert "training history" not in note


def test_never_performed_says_so_plainly():
    note = unresolved_scope_note(["Barbell Squat"], _EMAP_NEVER, None, 90)
    assert "isn't in your training history" in note
    assert "haven't done" not in note


def test_the_two_misses_produce_different_sentences():
    """The negative that matters: one message for both cases is what created
    the untruth, so the two must not collapse back together."""
    a = unresolved_scope_note(["Lat Pulldown"], _EMAP_LOGGED, None, 90)
    b = unresolved_scope_note(["Barbell Squat"], _EMAP_NEVER, None, 90)
    assert a != b


def test_both_kinds_in_one_message_stay_distinct():
    emap = {**_EMAP_LOGGED, **_EMAP_NEVER}
    note = unresolved_scope_note(["Lat Pulldown", "Barbell Squat"], emap, None, 90)
    assert "haven't done Lat Pulldown in the last 90 days" in note
    assert "Barbell Squat isn't in your training history" in note


def test_no_note_when_everything_resolved():
    assert unresolved_scope_note([], {}, None, 90) == ""
    assert unresolved_scope_note(None, {}, None, 90) == ""


def test_missing_map_entry_does_not_claim_history():
    """Unknown ever_logged must fall to the weaker claim, not assert the user
    trained something. Absence of evidence is not evidence of a session."""
    note = unresolved_scope_note(["Mystery Lift"], {}, None, 90)
    assert "training history" in note


# ── The claim is citable; the FALSE claim is not ──────────────────────────────
#
# Same enforcement pattern as the no-verdict rule: it is not a phrase blocklist,
# it is the SHAPE of the data. A claim with nothing to cite is rejected by the
# grounding stage, so the three roles must stay three separate leaves.

def _resolve(emap, key, field):
    from src.citations import build_index, resolve_tag
    idx = build_index({"exercise_muscle_map": emap})
    return resolve_tag(idx, "exercise_muscle_map", key, field)


def test_each_role_resolves_to_a_scalar(ont):
    from src.citations import OK
    emap = build_map(ont, ["Lat Pulldown"], LOGGED)
    assert _resolve(emap, "Lat Pulldown", "limiting") == (OK, "Biceps")
    assert _resolve(emap, "Lat Pulldown", "primary")  == (OK, "Lats")


def test_there_is_no_blended_leaf_to_cite(ont):
    """The structural gate. If a combined "muscles worked" leaf existed, "lat
    pulldowns work your biceps" would be citable and would pass grounding."""
    from src.citations import NOT_FOUND
    emap = build_map(ont, ["Lat Pulldown"], LOGGED)
    for invented in ("muscles_worked", "muscles", "trained", "all"):
        assert _resolve(emap, "Lat Pulldown", invented)[0] == NOT_FOUND


def test_an_unnamed_exercise_cannot_be_cited(ont):
    """A claim about a lift not in the map must flag, not silently resolve."""
    from src.citations import MATCH_KEY_FLAG
    emap = build_map(ont, ["Lat Pulldown"], LOGGED)
    assert _resolve(emap, "Bench Press", "primary")[0] == MATCH_KEY_FLAG


# ── The model must SEE that it has no rows ────────────────────────────────────

def test_unresolved_names_are_not_hidden_from_the_model():
    """`unresolved_exercise_names` used to be popped off the package before the
    LLM saw it, and a disclaimer prepended to the finished answer instead. The
    model therefore answered about a named exercise with no idea it had no rows
    for it, and filled the gap from counts belonging to other lifts.

    A text guard, honestly: it pins that the pop is gone, not that the model
    behaves. The live check is what proves the behaviour.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "src" / "coordinator.py").read_text(encoding="utf-8")
    assert 'pkg.pop("unresolved_exercise_names"' not in src, \
        "the unresolved-names fact must stay in the package the model sees"
    assert 'pkg.get("unresolved_exercise_names")' in src


def test_the_package_carries_both_fields_to_the_model():
    """End-to-end on the real path the coordinator calls: the trim must not
    drop either field."""
    from src.data_agent import prepare_analysis_package
    pkg = prepare_analysis_package(query_period_days=90,
                                   exercise_names=["Zercher Wobble"])
    assert pkg.get("unresolved_exercise_names") == ["Zercher Wobble"]
    assert "Zercher Wobble" in (pkg.get("exercise_muscle_map") or {})


# ═════════════════════════════════════════════════════════════════════════════
# ROUND 2: a held muscle must never be DESCRIBED as trained
#
# The first round gave the model the graph and it read it correctly — the live
# draft cited `limiting` with one clean tag — then wrote around it:
#
#   "Actually, your Lat Pulldowns do train your biceps; the biceps act as a
#    limiting muscle, meaning they hold and stabilize the load..."
#
# The citation gate proves WHICH FIELD was read, never what the sentence means.
# Hence `trains` / `holds_only` (the wrong answer has no leaf to cite) and
# limiting_claim_guard (the wrong sentence is repaired deterministically).
# ═════════════════════════════════════════════════════════════════════════════

from src.citations import limiting_claim_guard, limiting_truth

_PKG = {"exercise_muscle_map": {"Lat Pulldown": {
    "primary": ["Lats"], "secondary": ["Teres Major"], "limiting": ["Biceps"],
    "trains": ["Lats", "Teres Major"], "holds_only": ["Biceps"]}}}


def test_trains_excludes_a_held_muscle(ont):
    """Both directions. `trains` is the ONE safe merge because it is the merge
    that EXCLUDES limiting — the affirmative claim has no leaf to cite."""
    m = build_map(ont, ["Lat Pulldown"], LOGGED)["Lat Pulldown"]
    assert "Biceps" not in m["trains"]
    assert "Lats" in m["trains"]


def test_holds_only_is_exactly_the_limiting_role(ont):
    for name in ("Lat Pulldown", "Shrug", "Barbell Curl"):
        m = build_map(ont, [name], LOGGED)[name]
        assert m["holds_only"] == m["limiting"]


def test_trains_is_primary_plus_secondary(ont):
    m = build_map(ont, ["Shrug"], LOGGED)["Shrug"]
    assert m["trains"] == sorted(set(m["primary"]) | set(m["secondary"]))


def test_trains_resolves_as_a_scalar_citation(ont):
    from src.citations import OK
    emap = build_map(ont, ["Lat Pulldown"], LOGGED)
    status, value = _resolve(emap, "Lat Pulldown", "trains")
    assert status == OK
    assert "Biceps" not in value


# ── the guard ────────────────────────────────────────────────────────────────

def test_guard_corrects_the_live_sentence():
    """Verbatim from the live run."""
    bad = ("Actually, your Lat Pulldowns do train your biceps; the biceps act "
           "as a limiting muscle, meaning they hold and stabilize the load.")
    out, flags = limiting_claim_guard(bad, _PKG)
    assert out != bad
    assert "does not train your Biceps" in out
    assert len(flags) == 1 and flags[0]["muscle"] == "Biceps"


def test_guard_catches_other_training_verbs():
    for verb in ("works", "builds", "develops", "hits", "targets"):
        bad = f"The Lat Pulldown really {verb} your biceps."
        out, _ = limiting_claim_guard(bad, _PKG)
        assert out != bad, f"{verb!r} slipped through"


def test_guard_leaves_a_true_training_claim_alone():
    """THE NEGATIVE THAT MATTERS. A guard that rewrites correct sentences is
    worse than no guard — Lats really are trained by a pulldown."""
    good = "Your Lat Pulldown trains your Lats hard."
    assert limiting_claim_guard(good, _PKG) == (good, [])


def test_guard_leaves_an_already_negated_claim_alone():
    for good in ("Your Lat Pulldown does not train your biceps.",
                 "Your Lat Pulldown doesn't train your biceps.",
                 "Your Lat Pulldown never trains the biceps."):
        assert limiting_claim_guard(good, _PKG) == (good, [])


def test_guard_does_not_fire_across_clauses():
    """The false positive that would matter: the training verb belongs to Lats,
    and the held muscle sits in a different clause making no training claim.

    Deliberately carries NO negation word — an earlier version of this test said
    "the biceps MERELY hold", which the negation list already caught, so it
    stayed green even with clause scoping removed. It was testing the negation
    list, not the clause split. This wording isolates the clause split.
    """
    good = "Your Lat Pulldown trains your lats, and the biceps hold the load."
    assert limiting_claim_guard(good, _PKG) == (good, [])


def test_guard_is_inert_without_a_map():
    txt = "Your Lat Pulldown trains your biceps."
    for pkg in ({}, None, {"exercise_muscle_map": {}},
                {"exercise_muscle_map": {"Lat Pulldown": {"limiting": []}}}):
        assert limiting_claim_guard(txt, pkg) == (txt, [])


def test_guard_never_touches_an_unrelated_sentence():
    txt = "You did 40 sets of squats last week and your bodyweight is steady."
    assert limiting_claim_guard(txt, _PKG) == (txt, [])


def test_guard_flags_what_it_changed():
    """A silent rewrite is not acceptable — the flag is the audit trail."""
    bad = "Your Lat Pulldown builds your biceps."
    _, flags = limiting_claim_guard(bad, _PKG)
    assert flags and flags[0]["kind"] == "limiting_claim"
    assert flags[0]["exercise"] == "Lat Pulldown"
    assert flags[0]["original"] == bad
    assert "held" in flags[0]["reason"].lower()


def test_guard_repairs_only_the_offending_sentence():
    text = ("You trained hard this month. Your Lat Pulldown trains your biceps. "
            "Keep the volume steady.")
    out, _ = limiting_claim_guard(text, _PKG)
    assert "You trained hard this month." in out
    assert "Keep the volume steady." in out
    assert "does not train your Biceps" in out


def test_limiting_truth_skips_exercises_with_nothing_held():
    truth = limiting_truth({"exercise_muscle_map": {
        "Barbell Curl": {"primary": ["Biceps"], "limiting": [], "trains": ["Biceps"]},
        "Lat Pulldown": _PKG["exercise_muscle_map"]["Lat Pulldown"]}})
    assert set(truth) == {"Lat Pulldown"}


@pytest.mark.parametrize("muscle, primary, expected_tail", [
    ("Grip", ["Lats"], "the grip holds the load while your Lats do the work."),
    ("Biceps", ["Chest"], "the biceps hold the load while your Chest does the work."),
    ("Forearms", ["Lats", "Biceps"], "the forearms hold the load while your Lats and Biceps do the work."),
    ("Core", [], "the core holds the load without being trained by it."),
])
def test_the_repair_sentence_agrees_with_singular_and_plural_names(muscle, primary, expected_tail):
    """"the grip hold the load" / "your Chest do the work": the verbs were fixed
    in the plural, so every singular muscle came out ungrammatical."""
    from src.citations import _held_claim_repair
    assert _held_claim_repair("Lat Pulldown", muscle, primary).endswith(expected_tail)


def test_guard_repair_text_comes_from_the_map():
    """The replacement names the real primary movers, not a canned phrase."""
    out, _ = limiting_claim_guard("Your Lat Pulldown trains your biceps.", _PKG)
    assert "Lats" in out


# ── False positives the LIVE run found and the unit tests had missed ─────────
#
# The guard shipped green and then rewrote a CORRECT, more informative sentence,
# destroying half the answer. Cause: "do the work" at the end of it — the NOUN
# "work" matched the training-verb pattern. The earlier true-claim test used a
# short sentence and never exercised that. An example is one sample of a class.

def test_guard_keeps_a_correct_sentence_that_ends_with_do_the_work():
    """Verbatim from the live log. Correct, and richer than the repair — the
    guard must leave it entirely alone."""
    good = ("Your Lat Pulldowns train your lats and teres major; the biceps act "
            "as a limiting muscle that holds and stabilizes the load while your "
            "lats do the work.")
    assert limiting_claim_guard(good, _PKG) == (good, [])


def test_a_training_word_after_a_determiner_is_a_noun():
    for good in ("Your Lat Pulldown builds the lats while the biceps do the work.",
                 "The biceps are along for the ride during your Lat Pulldown work.",
                 "Your Lat Pulldown hits the lats; the biceps get a little work."):
        assert limiting_claim_guard(good, _PKG) == (good, []), good


def test_a_clause_describing_the_held_role_is_left_alone():
    """Saying the muscle is limiting / holding / stabilising IS the right
    answer. Rewriting it would replace a good sentence with a blunter one."""
    for good in ("Your Lat Pulldown builds the lats; the biceps are a limiting muscle here.",
                 "During the Lat Pulldown the biceps stabilize the load.",
                 "The Lat Pulldown grows your lats while the biceps hold the load."):
        assert limiting_claim_guard(good, _PKG) == (good, []), good


def test_guard_is_idempotent():
    """Its own repair text ends '...do the work'. Without the determiner rule
    the guard re-triggered on its own output."""
    bad = "Your Lat Pulldown trains your biceps."
    once, f1 = limiting_claim_guard(bad, _PKG)
    twice, f2 = limiting_claim_guard(once, _PKG)
    assert once == twice
    assert f1 and not f2


def test_guard_catches_the_passive_voice():
    """Position-based detection needs the passive named explicitly, or
    "your biceps are trained" walks straight past it."""
    bad = "During the Lat Pulldown your biceps are trained as well."
    out, flags = limiting_claim_guard(bad, _PKG)
    assert out != bad and flags


# ── The suggestable pool ─────────────────────────────────────────────────────
#
# Live, the coach proposed "Face Pulls", "Standing Calf Raises" and "Rear Delt
# Flys" — real movements the store already holds as Cable Face Pull, Calf Raise
# and Rear Delt Machine Fly. It invented labels for exercises sitting right
# there, because nothing in the package said what the catalogue contains.

from src.data_agent.process import _compute_suggestable_exercises as suggestable


def test_suggestable_excludes_what_the_user_trains(ont):
    out = suggestable(ont, LOGGED)              # LOGGED: Barbell Curl, Lat Pulldown
    assert "Barbell Curl" not in out
    assert "Lat Pulldown" not in out


def test_suggestable_offers_what_they_have_never_logged(ont):
    out = suggestable(ont, LOGGED)
    assert "Shrug" in out and "Treadmill" in out


def test_suggestable_is_everything_when_nothing_is_logged(ont):
    assert len(suggestable(ont, {})) == len(ont["exercises"])


def test_suggestable_is_empty_without_an_ontology():
    assert suggestable({}, LOGGED) == []
    assert suggestable(None, LOGGED) == []


def test_suggestable_is_names_only(ont):
    """Bounded payload — the pool is a list of strings, never rows."""
    out = suggestable(ont, LOGGED)
    assert out and all(isinstance(n, str) for n in out)
    assert out == sorted(out)


def test_the_package_carries_the_pool():
    from src.data_agent import prepare_analysis_package
    pkg = prepare_analysis_package(query_period_days=90)
    pool = pkg.get("suggestable_exercises")
    assert isinstance(pool, list)
    logged = {e["name"].lower() for e in pkg.get("exercises", [])}
    assert not (logged & {n.lower() for n in pool}), \
        "the pool must not offer something already being trained"
