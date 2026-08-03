# python server.py
# Serves the UI at http://localhost:3000

import asyncio
import csv
import json
import os
import sqlite3
import sys
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

FRONTEND_DIR = Path(__file__).parent
REPO_ROOT = FRONTEND_DIR.parent
DB_PATH = REPO_ROOT / "data" / "FitNotes_Backup.fitnotes"
ONTOLOGY_DIR = REPO_ROOT / "ontology"
REVIEW_CUTS = ONTOLOGY_DIR / "review_cuts.csv"

# The graph viewer reads the ontology through src.ontology.load_ontology — the
# SAME parser the analytical pipeline uses — so the page can never disagree with
# the numbers the agent reports, and there is no second CSV reader to keep in sync.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = FastAPI()


@app.on_event("startup")
async def open_browser():
    if os.environ.get("LAUNCHED_BY_MAIN"):
        return  # Main server handles browser open
    await asyncio.sleep(0.5)
    webbrowser.open("http://localhost:3000")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")


@app.get("/graph")
async def graph_page():
    return FileResponse(str(FRONTEND_DIR / "graph.html"), media_type="text/html")


@app.get("/vendor/{name}")
async def vendor(name: str):
    """Three.js and OrbitControls, vendored so the viewer works offline."""
    path = (FRONTEND_DIR / "vendor" / name).resolve()
    if path.parent != (FRONTEND_DIR / "vendor").resolve() or not path.is_file():
        return JSONResponse(status_code=404, content={"detail": "not found"})
    return FileResponse(str(path), media_type="text/javascript")


# ── The figure ────────────────────────────────────────────────────────────────
# frontend/body/figure.json holds a human figure as chains of rings and knows
# nothing about this project. Muscle positions are NOT hand-typed any more: each
# ontology muscle IS one of the figure's volumes, so its position is that
# volume's centroid. Nothing can float outside the body, because there is no
# separate coordinate to drift.
FIGURE_PATH = FRONTEND_DIR / "body" / "figure.json"


def _figure() -> dict:
    try:
        with open(FIGURE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[Server] figure.json unreadable ({exc}) — body disabled")
        return {"parts": []}


def _centroid(part: dict) -> tuple:
    """Ring-weighted centre of a volume. Weighting by cross-section keeps the
    label on the belly of a muscle rather than halfway up a tapering tail."""
    tx = ty = tz = tw = 0.0
    for r in part["rings"]:
        w = max(r["rx"] * r["rz"], 1e-6)
        tx += r["c"][0] * w
        ty += r["c"][1] * w
        tz += r["c"][2] * w
        tw += w
    return (tx / tw, ty / tw, tz / tw)


def _training_stats() -> tuple:
    """(sets, last-performed) per FitNotes exercise name. Read-only; never raises —
    the viewer must still open against a missing or locked database."""
    try:
        conn = sqlite3.connect(f"file:{str(DB_PATH).replace(os.sep, '/')}?mode=ro", uri=True)
        try:
            sets = {r[0]: r[1] for r in conn.execute(
                "SELECT e.name, COUNT(*) FROM training_log tl "
                "JOIN exercise e ON e._id = tl.exercise_id GROUP BY e.name")}
            last = {r[0]: r[1] for r in conn.execute(
                "SELECT e.name, MAX(tl.date) FROM training_log tl "
                "JOIN exercise e ON e._id = tl.exercise_id GROUP BY e.name")}
            return sets, last
        finally:
            conn.close()
    except Exception:
        return {}, {}


def _risk(link_source: str, muscle_name: str, role: str) -> str:
    """Which edges actually need a human. Ordered most-specific first."""
    if "corrects FitNotes" in link_source:
        return "corrects"
    if "user-ruling" in link_source:
        return "ruling"
    if "contested" in link_source:
        return "contested"
    if muscle_name.endswith("Head"):
        return "subhead"
    if role == "limiting":
        return "limiting"
    return "secondary" if role == "secondary" else "primary"


@app.get("/ontology-graph")
async def ontology_graph():
    from src.ontology import load_ontology, top_level_region

    ont = load_ontology(force=True)
    if not ont["loaded"]:
        return JSONResponse(status_code=503,
                            content={"detail": "ontology store did not load"})

    sets_by_db, last_by_db = _training_stats()
    ex_sets, ex_last, ex_alias = {}, {}, {}
    for db_name, eid in ont["aliases"].items():
        # aliases are keyed lowercase; recover the exact DB spelling for display
        exact = next((n for n in sets_by_db if n.lower() == db_name), db_name)
        ex_alias.setdefault(eid, []).append(exact)
        ex_sets[eid] = ex_sets.get(eid, 0) + sets_by_db.get(exact, 0)
        d = last_by_db.get(exact)
        if d and d > ex_last.get(eid, ""):
            ex_last[eid] = d

    figure = _figure()
    volumes = {p["name"]: p for p in figure.get("parts", [])
               if p.get("layer") == "muscle"}
    kids: dict = {}
    for m in ont["muscles"].values():
        if m["parent_id"]:
            kids.setdefault(m["parent_id"], []).append(m["id"])

    def leaves_under(mid):
        """Leaf descendants — the volumes a node actually occupies. A grouping
        node has no shape of its own; it IS the union of its children, which is
        the same union `_rollup` already uses for the numbers."""
        out, stack = [], [mid]
        while stack:
            c = stack.pop()
            if c in kids:
                stack.extend(kids[c])
            else:
                out.append(c)
        return out

    muscles = []
    for mid, m in ont["muscles"].items():
        own = leaves_under(mid)
        vol_names = [ont["muscles"][l]["name"] for l in own
                     if ont["muscles"][l]["name"] in volumes]
        pts = [_centroid(volumes[v]) for v in vol_names]
        cx = sum(p[0] for p in pts) / len(pts) if pts else 0.0
        cy = sum(p[1] for p in pts) / len(pts) if pts else 0.0
        cz = sum(p[2] for p in pts) / len(pts) if pts else 0.0
        muscles.append({"id": str(mid), "name": m["name"],
                        "parent": str(m["parent_id"]) if m["parent_id"] else None,
                        "size": m["size_class"], "path": ont["path"][mid],
                        "region": top_level_region(ont, mid),
                        "depth": len(ont["ancestors"][mid]),
                        "volumes": vol_names,
                        "x": cx, "y": cy, "z": cz})

    exercises = [{"id": str(eid), "name": e["canonical_name"],
                  "equip": e["equipment"], "pattern": e["movement_pattern"],
                  "sets": ex_sets.get(eid, 0), "last": ex_last.get(eid),
                  "aliases": sorted(ex_alias.get(eid, []))}
                 for eid, e in ont["exercises"].items()]

    links = [{"ex": str(l["exercise_id"]), "m": str(l["muscle_id"]),
              "role": l["role"], "src": l["source"],
              "risk": _risk(l["source"], ont["muscles"][l["muscle_id"]]["name"],
                            l["role"])}
             for l in ont["edges"]]

    return JSONResponse(content={"muscles": muscles, "exercises": exercises,
                                 "links": links, "rollup": _rollup(ont),
                                 "figure": figure, "cuts": _load_cuts()})


def _rollup(ont: dict) -> dict:
    """
    Per muscle: every exercise at or beneath it, with its EFFECTIVE role.

    Computed here, in Python, deliberately — the viewer must not reimplement the
    rollup, because a second implementation is a second set of numbers. The rule
    is exactly the one `_compute_muscle_ontology_summary` applies:

      • a muscle's exercises are its own plus everything in its subtree;
      • an exercise counts ONCE per muscle no matter how many edges reach it;
      • where one exercise reaches the same muscle several ways — Deadlift
        reaches Back as primary via Erectors AND as secondary via Lats —
        the STRONGEST role wins: primary > secondary > limiting.

    Getting this wrong is not cosmetic. Listing per-edge instead of per-exercise
    overstated Back by 588 primary sets and gave Legs 312 secondary sets where
    the true answer is zero, and it put the same lift in both columns — the
    blending the multi-column design exists to prevent.
    """
    out: dict = {}
    edges_by_ex: dict = {}
    for e in ont["edges"]:
        edges_by_ex.setdefault(e["exercise_id"], []).append(e)

    for mid in ont["muscles"]:
        subtree = ont["descendants"].get(mid, frozenset())
        rows = []
        for eid, edges in edges_by_ex.items():
            hits = [e for e in edges if e["muscle_id"] in subtree]
            if not hits:
                continue
            # Strongest role wins. Mirrors _compute_muscle_ontology_summary's
            # `secondary -= primary; limiting -= primary | secondary`, so the
            # panel and the coach's numbers cannot drift apart.
            role = next((r for r in ("primary", "secondary", "limiting")
                         if any(e["role"] == r for e in hits)), "secondary")
            effective = [e for e in hits if e["role"] == role] or hits
            # Name the attachment point only when it is NOT the muscle itself,
            # so the panel can show "via Triceps Long Head".
            direct = any(e["muscle_id"] == mid for e in effective)
            via = (None if direct
                   else ont["muscles"][effective[0]["muscle_id"]]["name"])
            rows.append({
                "ex": str(eid),
                "role": role,
                "via": via,
                "src": effective[0]["source"],
                "keys": [f"{e['exercise_id']}>{e['muscle_id']}" for e in hits],
            })
        out[str(mid)] = rows
    return out


def _load_cuts() -> list:
    if not REVIEW_CUTS.exists():
        return []
    try:
        with open(REVIEW_CUTS, encoding="utf-8-sig", newline="") as fh:
            return [r for r in csv.DictReader(fh) if r.get("exercise_id")]
    except Exception:
        return []


@app.post("/ontology-review")
async def ontology_review(request: Request):
    """
    Persist the reviewer's keep/cut decisions.

    Writes ONLY ontology/review_cuts.csv — a decision log. It does not touch the
    graph: scripts/apply_review_cuts.py performs that edit as a separate,
    deliberate step, so the change to exercise_muscle.csv stays a reviewable
    diff. muscles.csv is unreachable from here by construction (R7).
    """
    body = await request.json()
    rows = body.get("decisions") or []
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REVIEW_CUTS.with_suffix(".csv.tmp")
    cols = ("exercise_id", "muscle_id", "exercise_name", "muscle_name",
            "role", "decision")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    for attempt in range(10):
        try:
            os.replace(tmp, REVIEW_CUTS)
            break
        except PermissionError:
            if attempt == 9:
                raise
            await asyncio.sleep(0.2)
    cuts = sum(1 for r in rows if r.get("decision") == "cut")
    return JSONResponse(content={"status": "saved", "total": len(rows), "cuts": cuts})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".fitnotes"):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Only .fitnotes files are accepted"},
        )
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        DB_PATH.write_bytes(await file.read())
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)},
        )

    async def _notify_reload() -> None:
        def _call():
            urllib.request.urlopen(
                urllib.request.Request("http://localhost:8000/reload-db", method="POST"),
                timeout=3,
            )
        try:
            await asyncio.to_thread(_call)
        except Exception:
            pass

    await _notify_reload()
    return JSONResponse(content={"status": "success"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
