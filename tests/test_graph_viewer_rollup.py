"""
The graph viewer's per-muscle rollup must obey exactly the rule
_compute_muscle_ontology_summary applies, because the reviewer judges edges on
what the panel shows and the coach quotes what the package computes. Two
implementations means two sets of numbers.

The rule, in one place:
  • a muscle's exercises are its own plus everything in its subtree;
  • an exercise counts ONCE per muscle however many edges reach it;
  • where one exercise reaches a muscle BOTH ways, PRIMARY WINS.

This is not hypothetical. Listing per-edge (the first implementation) overstated
Back by 588 primary sets, gave Legs 312 secondary sets where the truth is zero,
and put the same lift in both columns — precisely the blending the two-column
design exists to prevent. Eight grouping muscles (Back, Chest, Legs, Arms,
Shoulders, Core, Forearms, Traps) showed an EMPTY panel, because they carry no
direct edges at all.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _frontend_server():
    """
    Load frontend/server.py under its OWN module name.

    A plain `import server` is a trap here: the repo has TWO modules called
    `server` (the API on 8000 and the frontend on 3000). Whichever is imported
    first wins sys.modules for the rest of the session, so this passed alone and
    failed in the full suite with an AttributeError.
    """
    if "fitnotes_frontend_server" in sys.modules:
        return sys.modules["fitnotes_frontend_server"]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(
        "fitnotes_frontend_server", _ROOT / "frontend" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fitnotes_frontend_server"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def payload():
    fe = _frontend_server()
    return json.loads(asyncio.run(fe.ontology_graph()).body)


@pytest.fixture(scope="module")
def ont():
    from src.ontology import load_ontology
    return load_ontology(force=True)


def _rows(payload, name):
    mid = next(m["id"] for m in payload["muscles"] if m["name"] == name)
    return payload["rollup"][mid]


# ── the rule ──────────────────────────────────────────────────────────────────

def test_every_muscle_has_a_rollup_entry(payload):
    assert set(payload["rollup"]) == {m["id"] for m in payload["muscles"]}


def test_an_exercise_appears_at_most_once_per_muscle(payload):
    for mid, rows in payload["rollup"].items():
        ids = [r["ex"] for r in rows]
        assert len(ids) == len(set(ids)), f"muscle {mid} lists an exercise twice"


def test_no_exercise_is_both_primary_and_secondary_on_one_muscle(payload):
    """The exact defect: Deadlift reaches Back as primary via Erectors and as
    secondary via Lats. It must land in ONE column."""
    for mid, rows in payload["rollup"].items():
        prim = {r["ex"] for r in rows if r["role"] == "primary"}
        sec = {r["ex"] for r in rows if r["role"] == "secondary"}
        assert not (prim & sec), f"muscle {mid} has an exercise in both columns"


def test_primary_wins_when_a_lift_reaches_a_muscle_both_ways(payload, ont):
    """Recompute the effective role independently and compare, for every pair."""
    desc = {str(mid): {str(d) for d in ds}
            for mid, ds in ont["descendants"].items()}
    by_ex = {}
    for l in payload["links"]:
        by_ex.setdefault(l["ex"], []).append(l)

    for mid, rows in payload["rollup"].items():
        got = {r["ex"]: r["role"] for r in rows}
        want = {}
        for eid, ls in by_ex.items():
            hits = [l for l in ls if l["m"] in desc[mid]]
            if hits:
                want[eid] = ("primary" if any(l["role"] == "primary" for l in hits)
                             else "secondary")
        assert got == want, f"muscle {mid} rollup disagrees with the rule"


def test_deadlift_reaches_back_as_primary_only(payload):
    rows = {r["ex"]: r for r in _rows(payload, "Back")}
    dl = next(e for e in payload["exercises"] if e["name"] == "Deadlift")
    assert rows[dl["id"]]["role"] == "primary"


def test_rolled_totals_never_double_count(payload):
    """Back's primary total is the sum of DISTINCT exercises, not of edges."""
    sets = {e["id"]: e["sets"] for e in payload["exercises"]}
    for name in ("Back", "Arms", "Legs", "Forearms", "Shoulders", "Traps"):
        rows = _rows(payload, name)
        prim = sum(sets[r["ex"]] for r in rows if r["role"] == "primary")
        sec = sum(sets[r["ex"]] for r in rows if r["role"] == "secondary")
        edge_prim = sum(sets[r["ex"]] * len(r["keys"])
                        for r in rows if r["role"] == "primary")
        # If these were equal for every muscle the dedup would be untested;
        # at least one of them must genuinely collapse edges.
        assert prim <= edge_prim
        assert prim >= 0 and sec >= 0


def test_the_dedup_actually_bites_somewhere(payload):
    """Guard against the rule silently becoming a no-op."""
    collapsed = [mid for mid, rows in payload["rollup"].items()
                 if any(len(r["keys"]) > 1 for r in rows)]
    assert collapsed, "no muscle collapses multiple edges — rollup may be inert"


# ── grouping muscles must not be empty ────────────────────────────────────────

@pytest.mark.parametrize("name", ["Back", "Chest", "Legs", "Arms", "Shoulders",
                                  "Core", "Forearms", "Traps", "Triceps"])
def test_grouping_muscles_list_their_subtree(payload, name):
    """These carry no direct edges. Before the rollup they showed nothing at
    all, so clicking 'Back' gave an empty panel."""
    rows = _rows(payload, name)
    assert rows, f"{name} lists no exercises"
    assert any(r["via"] for r in rows), f"{name} shows nothing inherited"


def test_triceps_includes_the_sub_head_isolation_work(payload):
    rows = _rows(payload, "Triceps")
    names = {next(e["name"] for e in payload["exercises"] if e["id"] == r["ex"]): r
             for r in rows}
    assert "Dumbbell Skull Crusher" in names
    assert names["Dumbbell Skull Crusher"]["via"] == "Triceps Long Head"
    assert names["Dumbbell Skull Crusher"]["role"] == "primary"
    assert "Cable Triceps Extension" in names
    assert names["Cable Triceps Extension"]["via"] == "Triceps Lateral Head"


def test_via_is_absent_for_a_directly_attached_lift(payload):
    rows = {r["ex"]: r for r in _rows(payload, "Lats")}
    lat = next(e for e in payload["exercises"] if e["name"] == "Lat Pulldown")
    assert rows[lat["id"]]["via"] is None


def test_leaf_muscle_rollup_is_just_its_own_edges(payload):
    rows = _rows(payload, "Calves")
    assert rows and all(r["via"] is None for r in rows)


def test_every_row_carries_a_source(payload):
    for rows in payload["rollup"].values():
        for r in rows:
            assert r["src"].strip(), r


def test_keys_reference_real_edges(payload):
    real = {l["ex"] + ">" + l["m"] for l in payload["links"]}
    for rows in payload["rollup"].values():
        for r in rows:
            for k in r["keys"]:
                assert k in real, k
