"""
The 3D graph viewer's JavaScript module graph must be CLOSED — every import
reachable from frontend/graph.html has to resolve to a file that is actually
vendored.

This exists because it already broke once. `three.module.js` is a thin
re-export shim that imports `./three.core.js`; only the two obvious files were
vendored, both returned 200 when requested directly, and the page still hung
forever on its loading state because the browser could not resolve the
transitive import. Checking the files you remembered to add proves nothing —
the graph has to be walked.

Pure static analysis over the repo, so it runs in the normal suite with no
server and no network.
"""

import json
import os
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = _ROOT / "frontend"
GRAPH_HTML = FRONTEND / "graph.html"
VENDOR = FRONTEND / "vendor"

# `import x from 'y'` / `export … from 'y'`, and bare side-effect `import 'y'`.
_FROM = re.compile(r"""(?:^|\n)\s*(?:import|export)\b[^;\n]*?\sfrom\s*['"]([^'"]+)['"]""")
_SIDE = re.compile(r"""(?:^|\n)\s*import\s*['"]([^'"]+)['"]""")


def _module_script() -> str:
    html = GRAPH_HTML.read_text(encoding="utf-8")
    m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
    assert m, "graph.html has no <script type=module>"
    return m.group(1)


def _import_map() -> dict:
    html = GRAPH_HTML.read_text(encoding="utf-8")
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert m, "graph.html has no import map — bare specifiers would not resolve"
    return json.loads(m.group(1))["imports"]


def _resolve(spec: str, referrer: Path, imports: dict) -> Path:
    target = imports.get(spec, spec)
    if target.startswith("/"):
        return FRONTEND / target.lstrip("/")
    return (referrer.parent / target).resolve()


def test_graph_html_exists():
    assert GRAPH_HTML.is_file()


def test_import_map_points_at_a_vendored_file():
    for spec, target in _import_map().items():
        path = FRONTEND / target.lstrip("/")
        assert path.is_file(), f"import map sends {spec!r} to missing {target}"


def test_module_graph_is_closed():
    """Walk every import from the entry script. Any unresolved file means the
    page cannot boot — which looks like an infinite loading spinner, not an
    error, so it must be caught here."""
    imports = _import_map()
    entry = set(_FROM.findall(_module_script())) | set(_SIDE.findall(_module_script()))
    assert entry, "entry script imports nothing — did the script block move?"

    seen: set = set()
    missing: list = []
    queue = [(s, GRAPH_HTML) for s in entry]

    while queue:
        spec, referrer = queue.pop()
        path = _resolve(spec, referrer, imports)
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            missing.append(f"{spec!r} (from {referrer.name}) -> {path}")
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for nxt in set(_FROM.findall(body)) | set(_SIDE.findall(body)):
            queue.append((nxt, path))

    assert not missing, "unresolved module import(s):\n  " + "\n  ".join(missing)
    # three.module.js -> three.core.js is the transitive edge that broke before.
    assert (VENDOR / "three.core.js") in seen, \
        "three.core.js was not reached — the shim/core split may have changed"


def test_vendored_three_is_present():
    for name in ("three.module.js", "three.core.js", "OrbitControls.js"):
        assert (VENDOR / name).is_file(), f"frontend/vendor/{name} is missing"


def test_viewer_does_not_reach_the_network():
    """Vendored on purpose: the viewer must work offline and must not silently
    drift to a different Three.js version between sessions."""
    script = _module_script()
    html = GRAPH_HTML.read_text(encoding="utf-8")
    for bad in ("unpkg.com", "cdn.jsdelivr", "cdnjs.", "esm.sh", "skypack"):
        assert bad not in html, f"graph.html reaches out to {bad}"
    for spec in set(_FROM.findall(script)) | set(_SIDE.findall(script)):
        assert not spec.startswith(("http://", "https://", "//")), spec


def test_vendor_route_is_registered_and_scoped():
    src = (FRONTEND / "server.py").read_text(encoding="utf-8")
    assert '@app.get("/vendor/{name}")' in src
    assert '@app.get("/graph")' in src
    # Path traversal out of frontend/vendor must be refused.
    assert ".resolve()" in src and "not found" in src


@pytest.mark.parametrize("route", ["/ontology-graph", "/ontology-review"])
def test_data_routes_registered(route):
    src = (FRONTEND / "server.py").read_text(encoding="utf-8")
    assert route in src


def test_review_endpoint_never_writes_the_graph():
    """
    R7/R10 at the HTTP boundary: a click in the browser can only ever write the
    decision log. Editing exercise_muscle.csv is a separate, deliberate script,
    so the graph change always arrives as a reviewable diff.

    Checks executable statements only — the prose in docstrings and comments
    legitimately names those files while explaining why it does not touch them.
    """
    import ast

    src = (FRONTEND / "server.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
              and n.name == "ontology_review")
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]                      # drop the docstring
    code = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))

    assert "REVIEW_CUTS" in code, "the endpoint should write only the decision log"
    for forbidden in ("muscles.csv", "exercise_muscle.csv", "aliases.csv",
                      "exercises.csv", "EDGES"):
        assert forbidden not in code, \
            f"/ontology-review must not touch {forbidden} — use apply_review_cuts.py"


# ── the third role reaches the UI ─────────────────────────────────────────────
#
# These three are TEXT-LEVEL guards, not behavioural tests: there is no JS
# runner in this project, so they catch a deletion or a rename and nothing more.
# What they defend is a gap that really happened — `limiting` was wired into the
# 3D lines, the colours, the legend and the audit console, but the muscle panel
# built only a primary and a secondary list, so 16 promoted edges would have
# drawn on the body while being absent from the rail beside it.

def test_muscle_panel_renders_limiting():
    js = _module_script()
    assert "r.role === 'limiting'" in js, \
        "muscleHTML builds no limiting list — promoted edges would be invisible in the rail"
    assert "rowsFor(lim, false)" in js, \
        "limiting rows must render with an EMPTY count column, never a set number"


def test_limiting_rows_carry_no_per_lift_set_count():
    """The count column is what would re-introduce the false volume: printing
    266 beside Smith Machine Shrug on Grip claims exactly the training the role
    exists to deny. The aggregate belongs in the summary line, labelled."""
    js = _module_script()
    assert "function rowsFor(rows, showSets = true)" in js
    assert "sets held" in js, "the aggregate must be labelled 'sets held', not 'sets'"


def test_limiting_filter_is_offered():
    js = _module_script()
    m = re.search(r"\[([^\]]*)\]\.map\(f =>", js)
    assert m and "'limiting'" in m.group(1), "no limiting filter button"
    assert "UI.filter === 'limiting'" in js, \
        "the filter button has no matching edgeShown branch — it would draw nothing"


def test_a_finished_audit_does_not_pin_edges():
    """Audit mode ignores the selection by design, and the mode is remembered
    across reloads. With every call answered that left the whole queue drawn
    permanently and unresponsive to clicks."""
    js = _module_script()
    assert "mode === 'audit' && QUEUE.some(l => !decisions[l.key])" in js, \
        "a complete audit queue must fall through to normal selection behaviour"


def test_apply_script_is_the_only_path_that_edits_edges():
    """The counterpart: the graph edit lives in a script that defaults to a dry
    run, and it still never opens muscles.csv."""
    src = (_ROOT / "scripts" / "apply_review_cuts.py").read_text(encoding="utf-8")
    assert "--apply" in src and "DRY RUN" in src
    assert "exercise_muscle.csv" in src
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    body = code[code.index("def main("):] if "def main(" in code else code
    assert '"muscles.csv"' not in body and "'muscles.csv'" not in body
