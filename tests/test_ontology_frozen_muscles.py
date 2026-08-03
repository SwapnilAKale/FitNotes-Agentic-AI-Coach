"""
R7 — THE MUSCLE SET IS CLOSED.

Every human has the same muscles and the tree already covers them, so no user,
no upload and no model may add, remove or rename one. exercises.csv,
aliases.csv and exercise_muscle.csv grow as new lifts are reconciled;
muscles.csv never does.

This file is mechanism #1 of three: the snapshot below pins every row, so an
accidental or automated edit to muscles.csv fails the suite. If you are changing
the taxonomy ON PURPOSE, update this list in the same commit — that is the point,
the change has to be deliberate and reviewed.

(#2 is ontology.WRITABLE_FILES + the byte-identity test in
tests/test_ontology_reconcile.py; #3 is ontology.resolve_muscle_names, which
rejects an unknown name instead of creating it.)
"""

import csv
import os

import pytest

from src import ontology as ont_mod
from src.ontology import (WRITABLE_FILES, MUSCLES_FILE, closed_muscle_names,
                          resolve_muscle_names, top_level_region)

_STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "ontology")

# (id, name, parent_id, size_class) — the entire closed set.
FROZEN_MUSCLES = [
    (1, 'Chest', None, 'large'),
    (2, 'Upper Chest', 1, 'medium'),
    (3, 'Mid Chest', 1, 'medium'),
    (4, 'Lower Chest', 1, 'medium'),
    (5, 'Back', None, 'large'),
    (6, 'Lats', 5, 'large'),
    (7, 'Traps', 5, 'medium'),
    (8, 'Upper Traps', 7, 'medium'),
    (9, 'Mid Traps', 7, 'small'),
    (10, 'Lower Traps', 7, 'small'),
    (11, 'Rhomboids', 5, 'small'),
    (12, 'Erectors', 5, 'medium'),
    (13, 'Teres Major', 5, 'small'),
    (14, 'Shoulders', None, 'medium'),
    (15, 'Front Delts', 14, 'small'),
    (16, 'Side Delts', 14, 'small'),
    (17, 'Rear Delts', 14, 'small'),
    (18, 'Arms', None, None),
    (19, 'Triceps', 18, 'medium'),
    (20, 'Triceps Long Head', 19, 'medium'),
    (21, 'Triceps Lateral Head', 19, 'small'),
    (23, 'Biceps', 18, 'medium'),
    (26, 'Brachialis', 18, 'small'),
    (27, 'Forearms', 18, 'medium'),
    (28, 'Wrist Flexors', 27, 'small'),
    (29, 'Wrist Extensors', 27, 'small'),
    (30, 'Brachioradialis', 27, 'small'),
    (31, 'Grip', 27, 'small'),
    (32, 'Legs', None, 'large'),
    (33, 'Quads', 32, 'large'),
    (34, 'Hamstrings', 32, 'large'),
    (35, 'Glutes', 32, 'large'),
    (36, 'Calves', 32, 'medium'),
    (37, 'Adductors', 32, 'medium'),
    (38, 'Hip Flexors', 32, 'small'),
    (39, 'Core', None, 'medium'),
    (40, 'Rectus Abdominis', 39, 'medium'),
    (41, 'Obliques', 39, 'small'),
    (42, 'Transverse Abdominis', 39, 'small'),
]


@pytest.fixture(autouse=True)
def _clear_cache():
    ont_mod.clear_cache()
    yield
    ont_mod.clear_cache()


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("ONTOLOGY_DIR", _STORE)
    return ont_mod.load_ontology(force=True)


def _csv_rows():
    with open(os.path.join(_STORE, MUSCLES_FILE), encoding="utf-8-sig",
              newline="") as fh:
        return list(csv.DictReader(fh))


def test_muscle_set_matches_the_frozen_snapshot():
    actual = [
        (int(r["id"]), r["name"],
         int(r["parent_id"]) if r["parent_id"].strip() else None,
         r["size_class"].strip() or None)
        for r in _csv_rows()
    ]
    assert actual == FROZEN_MUSCLES, (
        "muscles.csv changed. The muscle set is CLOSED (R7) — if this edit is "
        "deliberate, update FROZEN_MUSCLES in the same commit; if it is not, "
        "revert it. Automated paths must never reach this file."
    )


def test_muscle_count_is_exactly_39():
    assert len(_csv_rows()) == 39


def test_muscles_csv_is_not_in_the_writable_set():
    """Mechanism #2: the promote path is only ever allowed to append to these."""
    assert MUSCLES_FILE not in WRITABLE_FILES
    assert set(WRITABLE_FILES) == {"exercises.csv", "aliases.csv",
                                   "exercise_muscle.csv"}


# ── resolve_muscle_names rejects, never creates ───────────────────────────────

def test_known_muscle_names_resolve(store):
    ids, unknown = resolve_muscle_names(store, ["Lats", "Mid Traps"])
    assert unknown == ()
    assert [store["muscles"][i]["name"] for i in ids] == ["Lats", "Mid Traps"]


def test_name_matching_is_case_insensitive(store):
    ids, unknown = resolve_muscle_names(store, ["lats", "REAR DELTS"])
    assert unknown == () and len(ids) == 2


def test_unknown_muscle_is_returned_as_unknown_not_created(store):
    before = len(store["muscles"])
    ids, unknown = resolve_muscle_names(store, ["Lats", "Sternocleidomastoid"])
    assert unknown == ("Sternocleidomastoid",)
    assert len(ids) == 1                       # only the real one resolved
    assert len(store["muscles"]) == before     # nothing was added
    assert "Sternocleidomastoid" not in store["by_muscle_name"]


def test_duplicate_names_collapse(store):
    ids, unknown = resolve_muscle_names(store, ["Lats", "lats", "Lats"])
    assert unknown == () and len(ids) == 1


def test_closed_vocabulary_is_the_whole_set(store):
    names = closed_muscle_names(store)
    assert len(names) == 39
    assert names == tuple(sorted(n for _i, n, _p, _s in FROZEN_MUSCLES))


# ── R8 support: an unknown category resolves THROUGH the ontology ─────────────

def test_top_level_region_rolls_a_deep_muscle_to_its_root(store):
    upper_traps = store["by_muscle_name"]["Upper Traps"]
    assert top_level_region(store, upper_traps) == "Back"
    long_head = store["by_muscle_name"]["Triceps Long Head"]
    assert top_level_region(store, long_head) == "Arms"


def test_top_level_region_of_a_root_is_itself(store):
    assert top_level_region(store, store["by_muscle_name"]["Back"]) == "Back"


def test_top_level_region_of_unknown_id_is_none(store):
    assert top_level_region(store, 99999) is None
