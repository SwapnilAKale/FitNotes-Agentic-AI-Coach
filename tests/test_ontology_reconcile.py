"""
Stage 1 reconciliation: what auto-resolves, what must wait for approval, and the
guarantee that nothing reaches muscles.csv.

The load-bearing negatives here are as important as the positives:
  • a real typo does NOT auto-alias (fuzzy matching is not a tier on purpose)
  • 'Machine Shrug Row' does NOT match 'Machine Shrug' (different word SET)
  • an unapproved row writes nothing
  • muscles.csv is byte-identical across a promote (R7 mechanism #2)
"""

import csv
import hashlib
import os

import pytest

from src import ontology as ont_mod
from src import ontology_reconcile as rec

_MUSCLES = [
    (1, "Back",       "", "large"),
    (2, "Lats",       1,  "large"),
    (3, "Traps",      1,  "medium"),
    (4, "Arms",       "", ""),
    (5, "Triceps",    4,  "medium"),
    (6, "Biceps",     4,  "medium"),
]
_EXERCISES = [
    (1, "Dumbbell Skull Crusher", "dumbbell", "elbow extension"),
    (2, "Machine Shrug",          "machine",  "shrug"),
    (3, "Cable Lat Pull with EZ Bar", "cable", "vertical pull"),
]
_EDGES = [
    (1, 5, "primary", "test"),
    (2, 3, "primary", "test"),
    (3, 2, "primary", "test"),
]
_ALIASES = [("dumbbell skull crusher", 1), ("Machine Shrug", 2),
            ("Cable Lat Pull With Ez Bar", 3)]


@pytest.fixture
def store(tmp_path, monkeypatch):
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


def _digest(tmp_path, name):
    return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


# ── normalize ─────────────────────────────────────────────────────────────────

def test_normalize_ignores_case_space_and_punctuation():
    assert rec.normalize("Cable Lat Pull With Ez-Bar") == rec.normalize(
        "cable  lat_pull   withezbar")


def test_normalize_does_not_fix_spelling():
    # 'dumbell' is missing a b — these must NOT collapse, or a typo would
    # silently resolve to the wrong exercise.
    assert rec.normalize("Dumbell Skullcrusher") != rec.normalize(
        "Dumbbell Skull Crusher")


# ── The deterministic tier ────────────────────────────────────────────────────

def test_spacing_and_punctuation_variant_auto_matches(store):
    m = rec.deterministic_match("dumbbellskullcrusher", store)
    assert m.exercise_id == 1 and m.reason == "normalized-exact"


def test_word_order_variant_auto_matches(store):
    m = rec.deterministic_match("Skull Crusher Dumbbell", store)
    assert m.exercise_id == 1 and m.reason == "word-permutation"


def test_plural_variant_auto_matches(store):
    m = rec.deterministic_match("Machine Shrugs", store)
    assert m.exercise_id == 2 and m.reason == "word-permutation"


def test_matching_works_against_an_existing_alias_not_just_canonical(store):
    # 'Cable Lat Pull With Ez Bar' is only an ALIAS; a later user typing a
    # spacing variant of it should still resolve.
    m = rec.deterministic_match("cable lat pull with ez bar", store)
    assert m.exercise_id == 3


def test_extra_word_does_NOT_match(store):
    """The user's own example: 'Machine Shrug Row' is a different word set, so
    it must fall through to review rather than being folded into the shrug."""
    m = rec.deterministic_match("Machine Shrug Row", store)
    assert m.exercise_id is None and m.reason == "none"


def test_a_real_typo_does_NOT_auto_alias(store):
    m = rec.deterministic_match("Dumbell Skullcrusher", store)
    assert m.exercise_id is None


def test_completely_new_exercise_does_not_match(store):
    assert rec.deterministic_match("Pendlay Row", store).exercise_id is None


def test_empty_name_never_matches(store):
    assert rec.deterministic_match("", store).exercise_id is None
    assert rec.deterministic_match("   ", store).exercise_id is None


# ── detect_new ────────────────────────────────────────────────────────────────

def test_exact_existing_alias_is_already_known(store):
    r = rec.detect_new(["Machine Shrug"], store)
    assert r.already_known == ["Machine Shrug"]
    assert r.auto_aliased == [] and r.pending == []


def test_detect_splits_auto_from_pending(store):
    r = rec.detect_new(["Skull Crusher Dumbbell", "Pendlay Row"], store)
    assert [a["db_exercise_name"] for a in r.auto_aliased] == ["Skull Crusher Dumbbell"]
    assert r.auto_aliased[0]["canonical_name"] == "Dumbbell Skull Crusher"
    assert [p["db_exercise_name"] for p in r.pending] == ["Pendlay Row"]
    assert r.pending[0]["approved"] == ""       # the gate starts closed


def test_pending_is_ordered_by_logged_sets(store):
    r = rec.detect_new(["Rare Thing", "Pendlay Row"], store,
                       set_counts={"Pendlay Row": 300, "Rare Thing": 2})
    assert [p["db_exercise_name"] for p in r.pending] == ["Pendlay Row", "Rare Thing"]


def test_category_is_carried_as_a_hint_only(store):
    r = rec.detect_new(["Pendlay Row"], store, categories={"Pendlay Row": "Category_15"})
    assert r.pending[0]["fitnotes_category"] == "Category_15"
    # A hint must not become a decision.
    assert r.pending[0]["decision"] == "pending"
    assert r.pending[0]["muscles"] == ""


# ── R7: the write guard ───────────────────────────────────────────────────────

def test_writing_muscles_csv_is_refused(store):
    with pytest.raises(ValueError, match="muscle set is closed"):
        rec._assert_writable("muscles.csv")
    with pytest.raises(ValueError):
        rec._assert_writable("something_else.csv")


def test_auto_alias_only_touches_aliases_csv(store, tmp_path):
    before = {n: _digest(tmp_path, n) for n in
              ("muscles.csv", "exercises.csv", "exercise_muscle.csv")}
    r = rec.detect_new(["Skull Crusher Dumbbell"], store)
    assert rec.write_auto_aliases(r.auto_aliased) == 1
    for name, digest in before.items():
        assert _digest(tmp_path, name) == digest, f"{name} was modified"
    assert ont_mod.resolve_db_exercise(
        ont_mod.load_ontology(force=True), "Skull Crusher Dumbbell") == 1


# ── The queue and the approval gate ───────────────────────────────────────────

def test_pending_round_trips(store):
    r = rec.detect_new(["Pendlay Row"], store)
    rec.save_pending(r.pending)
    assert [p["db_exercise_name"] for p in rec.load_pending()] == ["Pendlay Row"]


def test_merge_does_not_reset_an_already_reviewed_row(store):
    r = rec.detect_new(["Pendlay Row"], store)
    reviewed = dict(r.pending[0], decision="new", muscles="Lats:primary",
                    approved="y")
    rec.save_pending([reviewed])
    rec.merge_pending(r.pending)              # same name detected again
    rows = rec.load_pending()
    assert len(rows) == 1
    assert rows[0]["approved"] == "y" and rows[0]["muscles"] == "Lats:primary"


def test_unapproved_row_writes_nothing(store, tmp_path):
    r = rec.detect_new(["Pendlay Row"], store)
    rec.save_pending([dict(r.pending[0], decision="new", muscles="Lats:primary",
                           approved="")])
    before = {n: _digest(tmp_path, n) for n in
              ("muscles.csv", "exercises.csv", "aliases.csv", "exercise_muscle.csv")}
    result = rec.promote(store)
    assert result.promoted == []
    for name, digest in before.items():
        assert _digest(tmp_path, name) == digest
    assert len(rec.load_pending()) == 1        # still queued, not dropped


# ── promote ───────────────────────────────────────────────────────────────────

def _queue(store, **over):
    row = rec.detect_new([over.pop("name", "Pendlay Row")], store).pending[0]
    rec.save_pending([dict(row, **over)])


def test_promote_writes_a_new_exercise_with_its_edges(store, tmp_path):
    _queue(store, decision="new", muscles="Lats:primary|Biceps:secondary",
           evidence="Barbell row variant", sources="https://example.org/pendlay",
           approved="y")
    result = rec.promote(store)
    assert result.skipped == []
    assert result.promoted[0]["as"] == "new"

    o = ont_mod.load_ontology(force=True)
    eid = ont_mod.resolve_db_exercise(o, "Pendlay Row")
    assert eid is not None
    got = {(o["muscles"][e["muscle_id"]]["name"], e["role"])
           for e in o["edges_by_exercise"][eid]}
    assert got == {("Lats", "primary"), ("Biceps", "secondary")}
    # Provenance is carried, not invented.
    assert all("https://example.org/pendlay" in e["source"]
               for e in o["edges_by_exercise"][eid])
    assert rec.load_pending() == []            # dequeued once written


def test_promote_writes_an_alias_without_creating_an_exercise(store, tmp_path):
    ex_before = _digest(tmp_path, "exercises.csv")
    edges_before = _digest(tmp_path, "exercise_muscle.csv")
    _queue(store, name="Machine Shrug Row", decision="alias",
           alias_of="Machine Shrug", sources="https://example.org/x", approved="y")
    result = rec.promote(store)
    assert result.promoted[0]["as"] == "alias"
    assert _digest(tmp_path, "exercises.csv") == ex_before
    assert _digest(tmp_path, "exercise_muscle.csv") == edges_before
    o = ont_mod.load_ontology(force=True)
    assert ont_mod.resolve_db_exercise(o, "Machine Shrug Row") == 2


def test_promote_NEVER_touches_muscles_csv(store, tmp_path):
    """R7 mechanism #2 — the whole point of this arc's write path."""
    before = _digest(tmp_path, "muscles.csv")
    _queue(store, decision="new", muscles="Lats:primary", approved="y")
    rec.promote(store)
    assert _digest(tmp_path, "muscles.csv") == before
    assert len(ont_mod.load_ontology(force=True)["muscles"]) == len(_MUSCLES)


def test_unknown_muscle_skips_the_row_and_creates_nothing(store, tmp_path):
    before = _digest(tmp_path, "muscles.csv")
    _queue(store, decision="new", muscles="Sternocleidomastoid:primary", approved="y")
    result = rec.promote(store)

    assert result.promoted == []
    assert "Sternocleidomastoid" in result.skipped[0][1]
    assert _digest(tmp_path, "muscles.csv") == before
    o = ont_mod.load_ontology(force=True)
    assert "Sternocleidomastoid" not in o["by_muscle_name"]
    assert ont_mod.resolve_db_exercise(o, "Pendlay Row") is None
    assert len(rec.load_pending()) == 1         # kept for a human, not discarded


def test_row_with_no_primary_muscle_is_skipped(store):
    _queue(store, decision="new", muscles="Lats:secondary", approved="y")
    result = rec.promote(store)
    assert result.promoted == []
    assert "no primary" in result.skipped[0][1]


def test_unsure_decision_is_never_promoted(store):
    _queue(store, decision="unsure", muscles="Lats:primary", approved="y")
    result = rec.promote(store)
    assert result.promoted == []
    assert "unsure" in result.skipped[0][1]


def test_alias_to_an_unknown_exercise_is_skipped(store):
    _queue(store, decision="alias", alias_of="Nonexistent Lift", approved="y")
    result = rec.promote(store)
    assert result.promoted == []
    assert "not a known exercise" in result.skipped[0][1]


def test_promoted_exercise_is_immediately_usable(store):
    _queue(store, decision="new", muscles="Lats:primary", approved="y")
    rec.promote(store)
    # The module-level cache must have been invalidated, or the next package
    # build would still report the exercise as unmapped.
    o = ont_mod.load_ontology()
    assert ont_mod.resolve_db_exercise(o, "Pendlay Row") is not None
