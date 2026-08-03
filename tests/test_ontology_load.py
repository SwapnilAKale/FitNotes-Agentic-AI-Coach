"""
Store integrity for the muscle ontology (ontology/*.csv) plus the loader's
never-raises contract.

The integrity half runs against the REAL store on purpose — these files are
hand-curated, so a typo in a parent_id or a muscle_id is exactly the failure
mode worth a test. The degradation half points ONTOLOGY_DIR at a tmp directory.
"""

import csv
import os

import pytest

from src import ontology as ont_mod
from src.ontology import (CARDIO_PATTERN, is_unattributed, load_ontology,
                          resolve_db_exercise, subtree)

_STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "ontology")


@pytest.fixture(autouse=True)
def _clear_ontology_cache():
    # The loader caches per-directory at module level; tests that repoint
    # ONTOLOGY_DIR must not inherit a previous test's store.
    ont_mod.clear_cache()
    yield
    ont_mod.clear_cache()


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("ONTOLOGY_DIR", _STORE)
    return load_ontology(force=True)


# ── The real store loads cleanly ──────────────────────────────────────────────

def test_store_loads_with_no_errors(store):
    assert store["loaded"] is True
    assert store["errors"] == [], "\n".join(store["errors"])
    assert store["muscles"] and store["exercises"] and store["edges"]


def test_every_parent_id_resolves(store):
    for m in store["muscles"].values():
        if m["parent_id"] is not None:
            assert m["parent_id"] in store["muscles"], m


def test_no_parent_cycles(store):
    # A cycle would make `path` collapse to a bare name; walking every chain to a
    # root with a bounded guard is the direct assertion.
    for mid in store["muscles"]:
        seen, node = set(), mid
        while node is not None:
            assert node not in seen, f"cycle through muscle {mid}"
            seen.add(node)
            node = store["muscles"][node]["parent_id"]


def test_muscle_names_are_globally_unique(store):
    # The name is the citation match-key — a duplicate would make a cited number
    # resolve against the wrong muscle.
    names = [m["name"] for m in store["muscles"].values()]
    assert len(names) == len(set(names))


def test_size_class_values_are_valid(store):
    for m in store["muscles"].values():
        assert m["size_class"] in (None, "large", "medium", "small"), m


def test_every_edge_resolves_and_is_sourced(store):
    for e in store["edges"]:
        assert e["exercise_id"] in store["exercises"], e
        assert e["muscle_id"] in store["muscles"], e
        assert e["role"] in ("primary", "secondary"), e
        assert e["source"].strip(), e


def test_no_duplicate_edges(store):
    pairs = [(e["exercise_id"], e["muscle_id"]) for e in store["edges"]]
    assert len(pairs) == len(set(pairs))


def test_every_alias_target_resolves(store):
    for name, eid in store["aliases"].items():
        assert eid in store["exercises"], name


def test_every_non_cardio_exercise_has_a_primary_edge(store):
    for eid, ex in store["exercises"].items():
        if ex["movement_pattern"] == CARDIO_PATTERN:
            continue
        roles = [e["role"] for e in store["edges_by_exercise"].get(eid, [])]
        assert "primary" in roles, f"{ex['canonical_name']} has no primary muscle"


def test_cardio_exercises_carry_no_edges(store):
    # Deliberately unattributed — walking has no set count worth attributing to a
    # muscle. This must stay distinct from "we forgot to map it".
    for eid, ex in store["exercises"].items():
        if ex["movement_pattern"] == CARDIO_PATTERN:
            assert store["edges_by_exercise"].get(eid, []) == []
            assert is_unattributed(store, eid) is True


def test_every_logged_exercise_is_aliased(store):
    """
    The whole point of the bridge: a logged exercise with no alias has its sets
    silently excluded from muscle maths. Reads the user's DB directly.
    """
    import sqlite3
    db = os.environ.get("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
    conn = sqlite3.connect(f"file:{db.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """SELECT DISTINCT e.name FROM training_log tl
                 JOIN exercise e ON e._id = tl.exercise_id
                WHERE e.category_id NOT IN (10, 11, 12)""").fetchall()
    finally:
        conn.close()
    missing = [r[0] for r in rows if resolve_db_exercise(store, r[0]) is None]
    assert not missing, f"logged but unaliased: {missing}"


# ── Rollup and reachability ───────────────────────────────────────────────────

def test_subtree_includes_self_and_descendants(store):
    triceps = store["by_muscle_name"]["Triceps"]
    long_head = store["by_muscle_name"]["Triceps Long Head"]
    desc = subtree(store, triceps)
    assert triceps in desc and long_head in desc
    # ...and does NOT reach sideways into a sibling branch.
    assert store["by_muscle_name"]["Biceps"] not in desc


def test_subtree_of_leaf_is_just_itself(store):
    leaf = store["by_muscle_name"]["Triceps Long Head"]
    assert subtree(store, leaf) == frozenset({leaf})


def test_subtree_of_unknown_id_is_empty(store):
    assert subtree(store, 99999) == frozenset()


def test_every_muscle_is_reachable(store):
    """
    An unreachable muscle can only ever report zero sets, which would be a FALSE
    coverage claim rather than a fact about training. The store must not contain
    one — model the node only when some exercise can fill it.
    """
    unreachable = sorted(store["muscles"][mid]["name"]
                         for mid in store["muscles"]
                         if mid not in store["reachable"])
    assert not unreachable, f"unreachable muscles: {unreachable}"


def test_reachability_rolls_up_not_down(store):
    # Chest is reachable because exercises map beneath it...
    chest = store["by_muscle_name"]["Chest"]
    assert chest in store["reachable"]
    # ...and reachability is computed from edges, so a node with a direct edge is
    # reachable regardless of having no children.
    side_delts = store["by_muscle_name"]["Side Delts"]
    assert side_delts in store["reachable"]
    assert store["children"][side_delts] == []


# ── The two lifts this whole arc exists to fix ────────────────────────────────

def _muscles_for(store, db_name, role):
    eid = resolve_db_exercise(store, db_name)
    return {store["muscles"][e["muscle_id"]]["name"]
            for e in store["edges_by_exercise"].get(eid, []) if e["role"] == role}


def test_smith_machine_shrugs_is_traps_not_back(store):
    primary = _muscles_for(store, "Smith Machine Shrugs", "primary")
    assert primary == {"Upper Traps"}
    # Negative: the 266 sets must NOT land on the pulling muscles of the back.
    assert "Lats" not in primary and "Rhomboids" not in primary


def test_reverse_cable_curls_is_elbow_flexion_not_wrist_work(store):
    primary = _muscles_for(store, "Reverse Cable Curls", "primary")
    assert primary == {"Brachioradialis", "Brachialis"}
    # Negative: the 296 sets must NOT be counted as wrist work.
    assert "Wrist Flexors" not in primary and "Wrist Extensors" not in primary


def test_user_ruling_r2_edges_are_present(store):
    # Flat/incline dumbbell bench -> triceps; incline additionally -> front delts
    # (the shoulder can be the limiting factor); decline is NOT shoulder-loaded
    # the same way. These came from the user directly.
    assert "Triceps" in _muscles_for(store, "Flat Dumbbell Bench Press", "secondary")
    incline = _muscles_for(store, "Incline Dumbbell Bench Press", "secondary")
    assert {"Triceps", "Front Delts"} <= incline
    decline = _muscles_for(store, "Decline Dumbbell Bench Press", "secondary")
    assert "Triceps" in decline
    assert "Front Delts" not in decline
    deadlift = _muscles_for(store, "Deadlift", "primary")
    assert {"Hamstrings", "Glutes", "Erectors"} <= deadlift


# ── Never raises ──────────────────────────────────────────────────────────────

def test_missing_store_degrades_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path / "nope"))
    o = load_ontology(force=True)
    assert o["loaded"] is False
    assert o["muscles"] == {} and o["edges"] == [] and o["aliases"] == {}
    # An empty store must not crash the bridge either — everything is unmapped.
    assert resolve_db_exercise(o, "Deadlift") is None


def test_corrupt_csv_degrades_to_empty(tmp_path, monkeypatch):
    # muscles.csv missing its required columns entirely.
    (tmp_path / "muscles.csv").write_text("nonsense,columns\n1,2\n", encoding="utf-8")
    for name in ("exercises.csv", "exercise_muscle.csv", "aliases.csv"):
        (tmp_path / name).write_text("a,b\n", encoding="utf-8")
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    o = load_ontology(force=True)
    assert o["loaded"] is False
    assert o["muscles"] == {}


def test_bad_rows_are_dropped_and_reported_not_raised(tmp_path, monkeypatch):
    _write_store(
        tmp_path,
        muscles=[(1, "Chest", "", "large"), (2, "Ghost", 999, "large")],
        exercises=[(1, "Bench", "barbell", "horizontal push")],
        edges=[(1, 1, "primary", "test"),
               (1, 999, "primary", "test"),      # unknown muscle
               (1, 1, "secondary", "test"),      # duplicate pair
               (1, 1, "invented", "test"),       # bad role
               (99, 1, "primary", "test")],      # unknown exercise
        aliases=[("Bench Press", 1), ("Nowhere", 42)],
    )
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    o = load_ontology(force=True)

    assert o["loaded"] is True           # partial store still usable
    assert len(o["edges"]) == 1          # only the one good edge survived
    assert o["muscles"][2]["parent_id"] is None   # dangling parent cleared
    assert o["aliases"] == {"bench press": 1}     # bad alias dropped
    assert len(o["errors"]) >= 5


def test_unsourced_edge_is_rejected(tmp_path, monkeypatch):
    # Provenance is mandatory — sub-head attribution is contested, so an edge
    # with no stated origin is not admissible.
    _write_store(
        tmp_path,
        muscles=[(1, "Chest", "", "large")],
        exercises=[(1, "Bench", "barbell", "horizontal push")],
        edges=[(1, 1, "primary", "")],
        aliases=[("Bench", 1)],
    )
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    o = load_ontology(force=True)
    assert o["edges"] == []
    assert any("no source" in e for e in o["errors"])


def test_comment_rows_are_ignored(tmp_path, monkeypatch):
    _write_store(
        tmp_path,
        muscles=[(1, "Chest", "", "large"), ("# REVIEW me", "", "", "")],
        exercises=[(1, "Bench", "barbell", "horizontal push")],
        edges=[(1, 1, "primary", "test")],
        aliases=[("Bench", 1)],
    )
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    o = load_ontology(force=True)
    assert list(o["muscles"]) == [1]
    assert o["errors"] == []


def _write_store(directory, muscles, exercises, edges, aliases):
    def _w(name, header, rows):
        with open(os.path.join(str(directory), name), "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], muscles)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"],
       exercises)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], edges)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], aliases)
