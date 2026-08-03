"""
The figure, and its binding to the ontology.

frontend/body/figure.json is a human figure and knows nothing about FitNotes.
These tests check the geometry stands up on its own, and separately that every
ontology node lands on a real volume — because the previous design had muscle
positions hand-typed in a second place, and they drifted 2.2x wider than any
plausible torso. Positions are now volume centroids, so "muscle floating outside
the body" is unrepresentable rather than merely unlikely.
"""

import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
FIGURE = _ROOT / "frontend" / "body" / "figure.json"
GRAPH_HTML = _ROOT / "frontend" / "graph.html"

# 8-head canon, normalised sole=0 crown=1
CANON = {"crown": 1.000, "chin": 0.875, "shoulder": 0.820, "nipples": 0.750,
         "navel": 0.625, "crotch": 0.500, "knee": 0.250, "sole": 0.000}


def _frontend_server():
    """Load frontend/server.py under its own name — the repo has TWO modules
    called `server`, and whichever imports first would otherwise win."""
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
def figure():
    return json.loads(FIGURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload():
    return json.loads(asyncio.run(_frontend_server().ontology_graph()).body)


def _parts(figure, layer=None):
    return [p for p in figure["parts"] if layer is None or p.get("layer") == layer]


# ── the figure stands up on its own ───────────────────────────────────────────

def test_figure_has_a_base_form_and_muscles(figure):
    assert len(_parts(figure, "base")) >= 8
    assert len(_parts(figure, "muscle")) == 30


def test_every_part_is_a_usable_chain(figure):
    for p in figure["parts"]:
        assert len(p["rings"]) >= 2, f"{p['name']} is a single ring, not a chain"
        for r in p["rings"]:
            assert len(r["c"]) == 3
            assert r["rx"] > 0 and r["rz"] > 0, f"{p['name']} has a zero radius"


def test_chains_are_monotonic_along_their_axis(figure):
    """A chain that doubles back self-intersects when lofted. Checked on the
    axis the chain actually travels along, so the foot (which runs forward, not
    up) is judged on z rather than y."""
    for p in figure["parts"]:
        cs = [r["c"] for r in p["rings"]]
        spans = [max(c[i] for c in cs) - min(c[i] for c in cs) for i in range(3)]
        ax = spans.index(max(spans))
        vals = [c[ax] for c in cs]
        assert vals == sorted(vals) or vals == sorted(vals, reverse=True), \
            f"{p['name']} folds back on axis {ax}"


def test_figure_occupies_the_canonical_height(figure):
    ys = [r["c"][1] for p in figure["parts"] for r in p["rings"]]
    assert min(ys) < 0.04, "figure does not reach the ground"
    assert 0.97 < max(ys) <= 1.02, "figure is not one unit tall"


def test_named_landmarks_have_geometry_at_them(figure):
    """Every canonical landmark should have some part crossing it."""
    for name, y in CANON.items():
        if name in ("crown", "sole"):
            continue
        hit = any(min(r["c"][1] for r in p["rings"]) - 0.03 <= y
                  <= max(r["c"][1] for r in p["rings"]) + 0.03
                  for p in figure["parts"])
        assert hit, f"nothing spans the {name} landmark at y={y}"


# The figure must be COMPLETE — a body with a part missing reads as a mannequin
# no matter how good the rest is. Deleting the neck passed every other check in
# this file (the cranium and torso spans overlap that height), so completeness
# needs asserting by name rather than inferring from coverage.
REQUIRED_BASE = ["cranium", "neck", "torso", "upper arm", "forearm", "hand",
                 "thigh", "shin", "foot"]
REQUIRED_MUSCLES = [
    "Front Delts", "Side Delts", "Rear Delts",
    "Upper Chest", "Mid Chest", "Lower Chest",
    "Lats", "Upper Traps", "Mid Traps", "Lower Traps", "Rhomboids",
    "Erectors", "Teres Major",
    "Biceps", "Triceps Long Head", "Triceps Lateral Head", "Brachialis",
    "Brachioradialis", "Wrist Flexors", "Wrist Extensors", "Grip",
    "Rectus Abdominis", "Obliques", "Transverse Abdominis",
    "Quads", "Hamstrings", "Glutes", "Calves", "Adductors", "Hip Flexors",
]


@pytest.mark.parametrize("name", REQUIRED_BASE)
def test_base_part_present(figure, name):
    assert any(p["name"] == name for p in _parts(figure, "base")), \
        f"the figure has no {name}"


@pytest.mark.parametrize("name", REQUIRED_MUSCLES)
def test_muscle_volume_present(figure, name):
    assert any(p["name"] == name for p in _parts(figure, "muscle")), \
        f"the figure has no {name} volume"


def test_midline_parts_sit_on_the_midline(figure):
    for p in figure["parts"]:
        if p.get("mirror"):
            continue
        for r in p["rings"]:
            assert abs(r["c"][0]) < 1e-9, f"{p['name']} is off-centre but not mirrored"


def test_paired_parts_are_mirrored_not_typed_twice(figure):
    """Symmetry is structural. Two hand-typed halves drift."""
    names = [p["name"] for p in figure["parts"]]
    assert len(names) == len(set(names)), "a part is declared twice"
    for p in figure["parts"]:
        if p.get("mirror"):
            assert all(r["c"][0] > 0 for r in p["rings"]), \
                f"{p['name']} is mirrored, so it must be authored on one side"


def test_shoulder_breadth_is_anatomical(figure):
    """Breadth ~= 1/4 of height. The old hand-typed coordinates were 2.2x this,
    which is why the scene read as a cloud rather than a body."""
    widest = 0.0
    for p in figure["parts"]:
        if p["name"] not in ("Side Delts", "upper arm"):
            continue
        for r in p["rings"]:
            if 0.74 <= r["c"][1] <= 0.83:
                widest = max(widest, abs(r["c"][0]) + r["rx"])
    assert 0.21 <= widest * 2 <= 0.29, f"shoulder breadth {widest*2:.3f}"


def test_the_figure_has_depth_not_just_width(figure):
    """A rotationally symmetric figure reads as a column. Chest, glutes and foot
    must sit at different z."""
    torso = next(p for p in figure["parts"] if p["name"] == "torso")
    zs = [r["c"][2] for r in torso["rings"]]
    assert max(zs) - min(zs) > 0.03, "torso has no spinal curve"
    foot = next(p for p in figure["parts"] if p["name"] == "foot")
    fz = [r["c"][2] for r in foot["rings"]]
    assert max(fz) - min(fz) > 0.04, "foot does not project forward"


# ── binding to the ontology ───────────────────────────────────────────────────

def test_every_ontology_node_has_a_body(payload):
    missing = [m["name"] for m in payload["muscles"] if not m["volumes"]]
    assert not missing, f"nodes with no volume: {missing}"


def test_every_volume_is_claimed(payload, figure):
    vols = {p["name"] for p in _parts(figure, "muscle")}
    used = {v for m in payload["muscles"] for v in m["volumes"]}
    assert vols == used, f"unclaimed: {sorted(vols - used)}"


def test_grouping_nodes_are_the_union_of_their_children(payload):
    by_id = {m["id"]: m for m in payload["muscles"]}
    kids = {}
    for m in payload["muscles"]:
        if m["parent"]:
            kids.setdefault(m["parent"], []).append(m["id"])
    for pid, ks in kids.items():
        union = set()
        for k in ks:
            union |= set(by_id[k]["volumes"])
        assert set(by_id[pid]["volumes"]) == union, by_id[pid]["name"]


def test_muscle_positions_lie_inside_the_body(payload, figure):
    """Centroids come from the volumes, so this cannot drift — but assert it,
    because the previous design's hand-typed coordinates put the arms outside
    any plausible torso and nothing caught it."""
    for m in payload["muscles"]:
        x, y, z = m["x"], m["y"], m["z"]
        assert 0.0 <= y <= 1.0, f"{m['name']} is off the body vertically"
        assert abs(x) < 0.20, f"{m['name']} sits {abs(x):.3f} out — outside the figure"
        assert abs(z) < 0.12, f"{m['name']} sits {abs(z):.3f} deep"


def test_depths_are_the_three_expected_levels(payload):
    import collections
    counts = collections.Counter(m["depth"] for m in payload["muscles"])
    assert dict(sorted(counts.items())) == {1: 6, 2: 24, 3: 9}


def test_selection_depth_resolves_at_every_level(payload):
    """A click walks UP to the chosen level; a node already shallower stays put.
    Every node must resolve to exactly one target at each level, or an audit edge
    hanging off a sub-head becomes unreachable."""
    by_id = {m["id"]: m for m in payload["muscles"]}

    def resolve(mid, level):
        cur = by_id[mid]
        while cur["depth"] > level and cur["parent"]:
            cur = by_id[cur["parent"]]
        return cur

    for m in payload["muscles"]:
        for level in (1, 2, 3):
            t = resolve(m["id"], level)
            assert t["depth"] <= max(level, 1)
            if m["depth"] <= level:
                assert t["id"] == m["id"], \
                    f"{m['name']} at level {level} should stay put"

    # concrete: the three sub-head audit targets must be reachable at Head
    for name in ("Triceps Long Head", "Triceps Lateral Head", "Upper Traps"):
        node = next(m for m in payload["muscles"] if m["name"] == name)
        assert resolve(node["id"], 3)["name"] == name
        assert resolve(node["id"], 1)["depth"] == 1     # ...and roll up to a region


def test_lats_at_head_level_stays_lats(payload):
    """Graceful degradation: nothing deeper exists under Lats."""
    by_id = {m["id"]: m for m in payload["muscles"]}
    lats = next(m for m in payload["muscles"] if m["name"] == "Lats")
    cur = lats
    while cur["depth"] > 3 and cur["parent"]:
        cur = by_id[cur["parent"]]
    assert cur["name"] == "Lats"


# ── one source of truth ───────────────────────────────────────────────────────

def test_viewer_does_not_embed_a_copy_of_the_figure(payload):
    html = GRAPH_HTML.read_text(encoding="utf-8")
    assert "figure.json" not in html or "DATA.figure" in html
    # no hand-typed ring data in the viewer
    assert not re.search(r'"rx"\s*:\s*0\.\d+', html), \
        "graph.html contains literal ring data — the figure must come from the server"
    assert "MUSCLE_POS" not in html


def test_server_no_longer_hand_types_muscle_positions():
    src = (_ROOT / "frontend" / "server.py").read_text(encoding="utf-8")
    assert "MUSCLE_POS" not in src, \
        "positions must be volume centroids, not a second hand-maintained table"
    assert "_centroid" in src
