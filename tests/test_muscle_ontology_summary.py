"""
_compute_muscle_ontology_summary — the per-muscle set counts.

Runs on a synthetic store so every rule is exercised in both directions with an
explicit negative assertion: rollup counts once, primary beats secondary,
unmapped sets reach no muscle, an unreachable muscle is never called untouched.
"""

import csv
import os

import pytest

from src import ontology as ont_mod
from src.data_agent.process import _compute_muscle_ontology_summary as summarise
from src.data_agent.process import resolve_category

# ── Synthetic store ───────────────────────────────────────────────────────────
#
#   Arms                     Chest        Legs
#     └ Triceps                             ├ Quads
#         └ Long Head                       ├ Glutes
#                                           ├ Calves
#                                           └ Ghost   (no edge anywhere -> unreachable)
#
_MUSCLES = [
    (1, "Arms",       "",  ""),
    (2, "Triceps",    1,   "medium"),
    (3, "Long Head",  2,   "medium"),
    (4, "Chest",      "",  "large"),
    (5, "Legs",       "",  "large"),
    (6, "Quads",      5,   "large"),
    (7, "Glutes",     5,   "large"),
    (8, "Calves",     5,   "medium"),
    (9, "Ghost",      5,   "small"),
]
_EXERCISES = [
    (1, "Skull Crusher",     "dumbbell", "elbow extension"),
    (2, "Bench Press",       "barbell",  "horizontal push"),
    (3, "Squat",             "barbell",  "squat"),
    (4, "Walking",           "bodyweight", "cardio"),
    (5, "Calf Raise",        "barbell",  "calf raise"),
    (6, "Overhead Extension", "cable",   "elbow extension"),
]
_EDGES = [
    (1, 3, "primary",   "test"),                      # Skull Crusher -> Long Head
    (2, 4, "primary",   "test"),                      # Bench -> Chest
    (2, 2, "secondary", "test"),                      # Bench -> Triceps (assisting)
    (3, 6, "primary",   "test"),                      # Squat -> Quads
    (3, 7, "secondary", "test"),                      # Squat -> Glutes (assisting)
    (5, 8, "primary",   "test"),                      # Calf Raise -> Calves
    (6, 2, "primary",   "test"),                      # Overhead Ext -> Triceps AND
    (6, 3, "primary",   "test"),                      #                 Long Head
]
_ALIASES = [("Skull Crusher", 1), ("Bench Press", 2), ("Squat", 3),
            ("Walking", 4), ("Calf Raise", 5), ("Overhead Extension", 6)]

WINDOW = ("2026-04-01", "2026-06-30")      # span 90 (91 days inclusive)
PRIOR = ("2025-12-31", "2026-03-31")       # the same span, immediately before


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


def _sessions(*pairs):
    return [{"date": d, "working_sets_count": n} for d, n in pairs]


def _run(ont, by_ex, first_training_date="2025-01-01", pending=None):
    return summarise(by_ex, ont, WINDOW[0], WINDOW[1], first_training_date,
                     frozenset(pending or ()))


def _row(result, muscle):
    return next((r for r in result["muscles"] if r["muscle"] == muscle), None)


# ── Window ────────────────────────────────────────────────────────────────────

def test_window_is_echoed_so_a_coverage_claim_can_name_it(ont):
    r = _run(ont, {"Bench Press": _sessions(("2026-05-01", 3))})
    assert r["window"] == {"start": WINDOW[0], "end": WINDOW[1], "days": 90}
    assert r["prior_window"]["start"] == PRIOR[0]
    assert r["prior_window"]["end"] == PRIOR[1]


def test_prior_window_is_equal_length_immediately_before(ont):
    r = _run(ont, {"Bench Press": _sessions(("2026-05-01", 3),    # in window
                                            ("2026-02-15", 7))})  # in prior
    chest = _row(r, "Chest")
    assert chest["primary_sets"] == 3
    assert chest["prior_primary_sets"] == 7


def test_sets_outside_both_windows_are_ignored(ont):
    r = _run(ont, {"Bench Press": _sessions(("2025-06-01", 99))})
    chest = _row(r, "Chest")
    assert chest["primary_sets"] == 0 and chest["prior_primary_sets"] == 0


def test_prior_window_marked_incomplete_when_it_predates_the_first_session(ont):
    complete = _run(ont, {}, first_training_date="2025-01-01")
    assert complete["prior_window"]["complete"] is True
    # First ever session lands INSIDE the prior window -> the comparison would
    # read a missing history as a drop.
    partial = _run(ont, {}, first_training_date="2026-02-01")
    assert partial["prior_window"]["complete"] is False


# ── Two columns, never blended ────────────────────────────────────────────────

def test_primary_and_secondary_are_separate_columns(ont):
    r = _run(ont, {
        "Skull Crusher": _sessions(("2026-05-01", 4)),   # -> Long Head (primary)
        "Bench Press":   _sessions(("2026-05-02", 6)),   # -> Triceps (secondary)
    })
    tri = _row(r, "Triceps")
    assert tri["primary_sets"] == 4      # rolled up from Long Head
    assert tri["secondary_sets"] == 6


def test_no_field_anywhere_holds_the_blended_total(ont):
    """A fractional/blended number would park mostly-secondary muscles at the
    bottom of every ranking as an artifact of the weighting constant."""
    r = _run(ont, {
        "Skull Crusher": _sessions(("2026-05-01", 4)),
        "Bench Press":   _sessions(("2026-05-02", 6)),
    })
    tri = _row(r, "Triceps")
    blended = tri["primary_sets"] + tri["secondary_sets"]     # 10
    assert blended == 10
    assert blended not in [v for k, v in tri.items()
                           if isinstance(v, int) and k not in ("primary_sets",
                                                               "secondary_sets")]


# ── Rollup ────────────────────────────────────────────────────────────────────

def test_parent_receives_its_descendants_counts(ont):
    r = _run(ont, {"Skull Crusher": _sessions(("2026-05-01", 5))})
    assert _row(r, "Long Head")["primary_sets"] == 5
    assert _row(r, "Triceps")["primary_sets"] == 5
    assert _row(r, "Arms")["primary_sets"] == 5


def test_exercise_mapped_to_parent_AND_child_is_counted_once(ont):
    """Overhead Extension maps to BOTH Triceps and Long Head. If rollup summed
    per edge, Triceps would read 16 instead of 8."""
    r = _run(ont, {"Overhead Extension": _sessions(("2026-05-01", 8))})
    assert _row(r, "Long Head")["primary_sets"] == 8
    assert _row(r, "Triceps")["primary_sets"] == 8      # NOT 16
    assert _row(r, "Arms")["primary_sets"] == 8


def test_rollup_does_not_leak_sideways_into_a_sibling(ont):
    r = _run(ont, {"Skull Crusher": _sessions(("2026-05-01", 5))})
    assert _row(r, "Chest")["primary_sets"] == 0
    assert _row(r, "Quads")["primary_sets"] == 0


def test_primary_wins_when_one_exercise_reaches_a_muscle_both_ways(ont):
    """Squat is primary via Quads and secondary via Glutes; both roll up to Legs.
    Counting it in both columns would double-count the same sets."""
    r = _run(ont, {"Squat": _sessions(("2026-05-01", 10))})
    legs = _row(r, "Legs")
    assert legs["primary_sets"] == 10
    assert legs["secondary_sets"] == 0          # not also 10
    assert _row(r, "Quads")["primary_sets"] == 10
    assert _row(r, "Glutes")["secondary_sets"] == 10
    assert _row(r, "Glutes")["primary_sets"] == 0


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_zero_coverage_lists_a_muscle_with_no_sets_in_the_window(ont):
    r = _run(ont, {"Squat": _sessions(("2026-05-01", 10))})
    assert "Calves" in r["zero_coverage"]
    assert _row(r, "Calves")["last_trained_date"] is None


def test_zero_coverage_excludes_a_muscle_whose_only_sets_are_secondary(ont):
    """Glutes gets nothing but assistance work — that is still stimulus, and
    calling it untouched would be false."""
    r = _run(ont, {"Squat": _sessions(("2026-05-01", 10))})
    assert "Glutes" not in r["zero_coverage"]
    assert _row(r, "Glutes")["secondary_sets"] == 10


def test_coverage_is_window_scoped_not_all_time(ont):
    # Trained in the prior window only -> untouched in THIS window.
    r = _run(ont, {"Calf Raise": _sessions(("2026-02-01", 12))})
    assert "Calves" in r["zero_coverage"]
    assert _row(r, "Calves")["prior_primary_sets"] == 12


def test_unreachable_muscle_is_never_reported_at_all(ont):
    """Ghost has no exercise mapping to it, so it is permanently 0 no matter how
    the user trains. Reporting it as untouched would be a false claim about the
    user rather than a true one about the data."""
    r = _run(ont, {"Squat": _sessions(("2026-05-01", 10))})
    assert _row(r, "Ghost") is None
    assert "Ghost" not in r["zero_coverage"]


def test_last_trained_date_is_the_latest_in_window(ont):
    r = _run(ont, {"Bench Press": _sessions(("2026-04-05", 3),
                                            ("2026-06-20", 3),
                                            ("2026-05-10", 3))})
    assert _row(r, "Chest")["last_trained_date"] == "2026-06-20"


# ── Nothing is invisible ──────────────────────────────────────────────────────

def test_unmapped_exercise_is_named_and_its_sets_reach_no_muscle(ont):
    r = _run(ont, {
        "Bench Press":     _sessions(("2026-05-01", 4)),
        "Mystery Machine": _sessions(("2026-05-02", 25)),   # not in aliases.csv
    })
    assert r["unmapped_exercises"] == ["Mystery Machine"]
    assert r["unmapped_sets"] == 25
    assert "Mystery Machine" not in r["counted_exercises"]
    # Negative: those 25 sets must not have landed anywhere. Bench Press reaches
    # exactly three nodes (Chest primary; Triceps and Arms secondary), so the
    # grand total across the section is 4 x 3 and nothing else.
    assert sum(m["primary_sets"] + m["secondary_sets"] for m in r["muscles"]) == 12
    assert all(m["primary_sets"] != 25 and m["secondary_sets"] != 25
               for m in r["muscles"])


def test_cardio_is_unattributed_by_design_not_a_curation_gap(ont):
    r = _run(ont, {"Walking": _sessions(("2026-05-01", 9))})
    assert r["unattributed_exercises"] == ["Walking"]
    assert r["unattributed_sets"] == 9
    assert r["unmapped_exercises"] == []      # must NOT be reported as a gap
    assert "Walking" not in r["counted_exercises"]


def test_every_in_window_exercise_lands_in_exactly_one_list(ont):
    by_ex = {
        "Bench Press":     _sessions(("2026-05-01", 4)),
        "Walking":         _sessions(("2026-05-02", 9)),
        "Mystery Machine": _sessions(("2026-05-03", 25)),
        "Queued Lift":     _sessions(("2026-05-04", 11)),
    }
    r = _run(ont, by_ex, pending=["Queued Lift"])
    lists = [set(r["counted_exercises"]), set(r["unmapped_exercises"]),
             set(r["pending_review_exercises"]), set(r["unattributed_exercises"])]
    assert set().union(*lists) == set(by_ex)
    for i, a in enumerate(lists):
        for b in lists[i + 1:]:
            assert not (a & b)


# ── Awaiting approval is not the same as never mapped ─────────────────────────

def test_queued_exercise_is_reported_as_pending_not_unmapped(ont):
    """A new lift the user has been asked to approve is 'waiting on you'; an
    unknown one is a gap in the graph. Both excluded, different sentences."""
    r = _run(ont, {"Queued Lift": _sessions(("2026-05-01", 11))},
             pending=["Queued Lift"])
    assert r["pending_review_exercises"] == ["Queued Lift"]
    assert r["pending_review_sets"] == 11
    assert r["unmapped_exercises"] == [] and r["unmapped_sets"] == 0


def test_pending_sets_reach_no_muscle(ont):
    """R10 — nothing is counted before approval."""
    r = _run(ont, {"Bench Press": _sessions(("2026-05-01", 4)),
                   "Queued Lift": _sessions(("2026-05-02", 11))},
             pending=["Queued Lift"])
    assert sum(m["primary_sets"] + m["secondary_sets"] for m in r["muscles"]) == 12
    assert all(m["primary_sets"] != 11 and m["secondary_sets"] != 11
               for m in r["muscles"])


def test_an_unqueued_unknown_stays_unmapped(ont):
    r = _run(ont, {"Mystery Machine": _sessions(("2026-05-01", 5))}, pending=[])
    assert r["unmapped_exercises"] == ["Mystery Machine"]
    assert r["pending_review_exercises"] == []


def test_a_mapped_exercise_is_never_pending_even_if_queued(ont):
    # Belt and braces: once promoted, a stale queue entry must not hide a
    # counted exercise from the numbers.
    r = _run(ont, {"Bench Press": _sessions(("2026-05-01", 4))},
             pending=["Bench Press"])
    assert r["counted_exercises"] == ["Bench Press"]
    assert r["pending_review_exercises"] == []
    assert _row(r, "Chest")["primary_sets"] == 4


# ── R8: an unknown FitNotes category never becomes a fabricated group ─────────

def test_known_category_is_unchanged(ont):
    assert resolve_category(5, "Bench Press", ont) == "Back"     # id 5 == Back
    assert resolve_category(4, "Bench Press", ont) == "Chest"


def test_unknown_category_resolves_through_the_ontology(ont):
    """A user-created category (AUTOINCREMENT ids go past 12) used to surface as
    the literal 'Category_15'. A new type of row is still Back."""
    assert resolve_category(15, "Squat", ont) == "Legs"
    assert resolve_category(15, "Bench Press", ont) == "Chest"
    assert resolve_category(15, "Skull Crusher", ont) == "Arms"   # via Long Head


def test_no_category_string_is_ever_fabricated(ont):
    for name in ("Squat", "Mystery Machine", ""):
        got = resolve_category(15, name, ont)
        assert "Category_" not in got and "Cat_" not in got


def test_unknown_category_and_unmapped_exercise_is_honestly_unknown(ont):
    assert resolve_category(15, "Mystery Machine", ont) == "Uncategorised"


def test_unknown_category_with_no_ontology_degrades_safely(ont):
    assert resolve_category(15, "Squat", None) == "Uncategorised"
    assert resolve_category(15, "Squat", {}) == "Uncategorised"


def test_secondary_only_exercise_does_not_pick_a_group_from_assistance(ont):
    # Only PRIMARY edges may name the group — otherwise a bench press could be
    # filed under Arms because the triceps assist.
    assert resolve_category(15, "Bench Press", ont) == "Chest"


def test_prior_window_only_exercise_is_not_listed_as_in_window(ont):
    # Lists describe the CURRENT window; a lift done only before it is not
    # "counted" now, and must not be reported as an unmapped gap either.
    r = _run(ont, {"Mystery Machine": _sessions(("2026-02-01", 5))})
    assert r["unmapped_exercises"] == []
    assert r["unmapped_sets"] == 0
    assert r["counted_exercises"] == []


# ── Working sets only ─────────────────────────────────────────────────────────

def test_only_working_sets_are_counted(ont):
    # working_sets_count already excludes the warmup set; the summary must read
    # that field and never the raw set list.
    r = _run(ont, {"Bench Press": [
        {"date": "2026-05-01", "working_sets_count": 3,
         "sets": [{}, {}, {}, {}]},          # 4 logged, 1 was a warmup
    ]})
    assert _row(r, "Chest")["primary_sets"] == 3


def test_session_with_zero_working_sets_contributes_nothing(ont):
    r = _run(ont, {"Bench Press": _sessions(("2026-05-01", 0))})
    assert _row(r, "Chest")["primary_sets"] == 0
    assert r["counted_exercises"] == []


# ── Degradation ───────────────────────────────────────────────────────────────

def test_empty_ontology_yields_an_empty_section(ont):
    empty = ont_mod.empty_ontology()
    assert summarise({"Bench Press": _sessions(("2026-05-01", 4))},
                     empty, WINDOW[0], WINDOW[1], "2025-01-01") == {}
    assert summarise({}, {}, WINDOW[0], WINDOW[1], "2025-01-01") == {}


def test_no_sessions_still_produces_a_usable_coverage_answer(ont):
    r = _run(ont, {})
    assert r["window"]["days"] == 90
    # Everything reachable is untouched — a true statement, not an error.
    assert set(r["zero_coverage"]) == {"Arms", "Triceps", "Long Head", "Chest",
                                       "Legs", "Quads", "Glutes", "Calves"}
    assert r["unmapped_sets"] == 0
