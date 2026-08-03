"""
The `limiting` role.

The audit used to ask one question — "does this muscle do enough work here to
change what you train tomorrow?" — which conflates two different claims. Grip on
Smith Machine Shrugs is the case that exposed it: grip genuinely caps how much
you can shrug, but 266 sets of shrugs train the grip for nothing. Keeping the
edge as `secondary` credits false volume; cutting it loses a real scheduling
fact.

The rule is INTERFERENCE DIRECTION:
  secondary  both ways   — delt work before an incline press hurts the press,
                           AND pressing hurts later delt work. It is trained.
  limiting   one way     — grip work before shrugs ruins the shrugs, but shrugs
                           leave the grip fine. It is held, not trained.

`limiting` NEVER counts as training volume and never counts as coverage.
"""

import csv

import pytest

from src import ontology as ont_mod
from src.data_agent.process import _compute_muscle_ontology_summary as summarise

WINDOW = ("2026-04-01", "2026-06-30")

_MUSCLES = [
    (1, "Arms", "", ""),
    (2, "Forearms", 1, "medium"),
    (3, "Grip", 2, "small"),
    (4, "Back", "", "large"),
    (5, "Traps", 4, "medium"),
]
_EXERCISES = [
    (1, "Smith Machine Shrug", "smith", "shrug"),
    (2, "Farmers Walk", "dumbbell", "carry"),
]
_ALIASES = [("Smith Machine Shrug", 1), ("Farmers Walk", 2)]


def _store(tmp_path, edges):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], _MUSCLES)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"],
       _EXERCISES)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], edges)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], _ALIASES)


@pytest.fixture
def ont(tmp_path, monkeypatch):
    # Shrug TARGETS traps and only HOLDS with the grip — the motivating case.
    _store(tmp_path, [(1, 5, "primary", "test"), (1, 3, "limiting", "test")])
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    assert o["errors"] == [], o["errors"]
    yield o
    ont_mod.clear_cache()


def _sessions(*pairs):
    return [{"date": d, "working_sets_count": n} for d, n in pairs]


def _run(ont, by_ex):
    return summarise(by_ex, ont, WINDOW[0], WINDOW[1], "2025-01-01", frozenset())


def _row(r, name):
    return next((x for x in r["muscles"] if x["muscle"] == name), None)


# ── the role exists and loads ─────────────────────────────────────────────────

def test_limiting_is_a_valid_role():
    assert "limiting" in ont_mod._VALID_ROLES
    assert ont_mod.ROLE_PRECEDENCE == ("primary", "secondary", "limiting")


def test_limiting_edge_loads(ont):
    edges = [e for e in ont["edges"] if e["role"] == "limiting"]
    assert len(edges) == 1


@pytest.mark.parametrize("rows", [
    [(1, 5, "primary", "t"), (1, 5, "limiting", "t")],   # primary first
    [(1, 5, "limiting", "t"), (1, 5, "primary", "t")],   # limiting first
])
def test_one_muscle_cannot_carry_two_roles_for_one_lift(rows, tmp_path, monkeypatch):
    """A muscle the lift TARGETS is trained by definition, so "also limiting" is
    incoherent. No special guard is needed: (exercise, muscle) is unique, so the
    duplicate-pair rule already makes the combination unrepresentable — in either
    row order."""
    _store(tmp_path, rows)
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    assert any("duplicate edge" in e for e in o["errors"]), o["errors"]
    assert len([e for e in o["edges"] if e["muscle_id"] == 5]) == 1
    ont_mod.clear_cache()


# ── THE BUG: the silent else ─────────────────────────────────────────────────

def test_limiting_sets_reach_neither_training_column(ont):
    """This is the test that would have caught

        target = primary_ids if role == "primary" else secondary_ids

    which swept every non-primary role into `secondary`, so a limiting edge
    would have been counted as training volume — the exact thing the role
    exists to prevent."""
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 266))})
    grip = _row(r, "Grip")
    assert grip["limiting_sets"] == 266
    assert grip["primary_sets"] == 0
    assert grip["secondary_sets"] == 0


def test_limiting_rolls_up_but_still_is_not_training(ont):
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 266))})
    for name in ("Grip", "Forearms", "Arms"):
        row = _row(r, name)
        assert row["limiting_sets"] == 266, name
        assert row["primary_sets"] == 0 and row["secondary_sets"] == 0, name


def test_the_targeted_muscle_is_unaffected(ont):
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 266))})
    traps = _row(r, "Traps")
    assert traps["primary_sets"] == 266 and traps["limiting_sets"] == 0


def test_no_field_holds_the_three_columns_summed(ont):
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 100))})
    for row in r["muscles"]:
        total = row["primary_sets"] + row["secondary_sets"] + row["limiting_sets"]
        if total == 0:
            continue
        others = [v for k, v in row.items()
                  if isinstance(v, int)
                  and k not in ("primary_sets", "secondary_sets", "limiting_sets")]
        assert total not in others, f"{row['muscle']} exposes a blended total"


def test_prior_window_limiting_is_tracked_separately(ont):
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 10),
                                                    ("2026-02-01", 40))})
    grip = _row(r, "Grip")
    assert grip["limiting_sets"] == 10 and grip["prior_limiting_sets"] == 40


# ── coverage stays honest ─────────────────────────────────────────────────────

def test_a_muscle_that_is_only_held_is_still_untouched(ont):
    """266 sets of shrugs do not train the grip, so coverage must still say the
    grip was untouched. This is the R1 honesty rule meeting the new role."""
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 266))})
    assert "Grip" in r["zero_coverage"]
    assert "Traps" not in r["zero_coverage"]


def test_an_exercise_with_only_limiting_edges_still_counts_as_mapped(ont):
    """It reached the graph — it is not unmapped, and not cardio."""
    r = _run(ont, {"Smith Machine Shrug": _sessions(("2026-05-01", 5))})
    assert "Smith Machine Shrug" in r["counted_exercises"]
    assert r["unmapped_exercises"] == [] and r["unattributed_exercises"] == []


# ── precedence ────────────────────────────────────────────────────────────────

def test_secondary_beats_limiting_on_the_same_muscle(tmp_path, monkeypatch):
    """Farmers Walk holds with the grip; if some other path also TRAINS the
    grip, the trained reading wins — a muscle genuinely worked is not demoted
    because another path merely leans on it."""
    _store(tmp_path, [(2, 3, "limiting", "t"), (2, 2, "secondary", "t"),
                      (2, 5, "primary", "t")])
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    r = summarise({"Farmers Walk": _sessions(("2026-05-01", 58))},
                  o, WINDOW[0], WINDOW[1], "2025-01-01", frozenset())
    forearms = next(x for x in r["muscles"] if x["muscle"] == "Forearms")
    # Forearms is reached as secondary directly AND as limiting via Grip.
    assert forearms["secondary_sets"] == 58
    assert forearms["limiting_sets"] == 0, "secondary must beat limiting"
    ont_mod.clear_cache()


def test_an_unknown_role_is_counted_nowhere(tmp_path, monkeypatch):
    """A role the loader rejects must not silently land in a column."""
    _store(tmp_path, [(1, 5, "primary", "t"), (1, 3, "invented", "t")])
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    o = ont_mod.load_ontology(force=True)
    r = summarise({"Smith Machine Shrug": _sessions(("2026-05-01", 20))},
                  o, WINDOW[0], WINDOW[1], "2025-01-01", frozenset())
    grip = next((x for x in r["muscles"] if x["muscle"] == "Grip"), None)
    if grip:
        assert grip["primary_sets"] == grip["secondary_sets"] == 0
        assert grip["limiting_sets"] == 0
    ont_mod.clear_cache()


# ── the panel must agree with the package ─────────────────────────────────────

def test_the_note_warns_against_blending():
    from src.ontology import load_ontology
    import os
    o = load_ontology(force=True)
    r = summarise({}, o, WINDOW[0], WINDOW[1], "2025-01-01", frozenset())
    assert "limiting_sets" in r["note"]
    assert "never" in r["note"].lower()
