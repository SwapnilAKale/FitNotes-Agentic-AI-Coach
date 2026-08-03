#!/usr/bin/env python3
"""
scripts/apply_review_cuts.py
Fold the reviewer's decisions from the 3D graph viewer into the ontology.

The viewer writes ontology/review_cuts.csv — a DECISION LOG, not the graph. This
script is the separate, deliberate step that edits ontology/exercise_muscle.csv,
so the change to the graph always arrives as a reviewable `git diff` rather than
appearing behind a click in a browser.

    python scripts/apply_review_cuts.py            # dry run — prints, writes nothing
    python scripts/apply_review_cuts.py --apply    # actually edit the graph

Two refusals, because a cut can silently break things the tests would otherwise
catch only much later:

  • an exercise must keep at least one PRIMARY muscle, or its sets stop reaching
    any muscle and quietly vanish from every count;
  • a muscle that loses its last incoming edge becomes UNREACHABLE, which means
    it can no longer be honestly reported as zero-coverage (a permanently-0
    muscle is an artifact of the taxonomy, not a fact about training).

The first blocks the cut. The second warns loudly and needs --allow-unreachable,
because it is occasionally the right call.

muscles.csv is never opened. The muscle set is closed (R7).
"""

import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ontology import load_ontology                      # noqa: E402

ONTOLOGY_DIR = os.environ.get("ONTOLOGY_DIR", os.path.join(_ROOT, "ontology"))
CUTS = os.path.join(ONTOLOGY_DIR, "review_cuts.csv")
EDGES = os.path.join(ONTOLOGY_DIR, "exercise_muscle.csv")
EDGE_COLUMNS = ("exercise_id", "muscle_id", "role", "source")


def load_decisions() -> tuple:
    """(cuts, reclassifications) from the review log.

    'limiting' is a ROLE REWRITE, not a deletion — the edge stays, but the
    muscle stops counting as trained by the lift. Grip really does cap a heavy
    shrug; it just gets no growth from it. Deleting the edge would lose the
    scheduling fact along with the false volume.
    """
    if not os.path.exists(CUTS):
        return [], []
    with open(CUTS, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cuts = [r for r in rows if (r.get("decision") or "").strip().lower() == "cut"]
    recl = [r for r in rows if (r.get("decision") or "").strip().lower() == "limiting"]
    return cuts, recl


def load_edges() -> list:
    with open(EDGES, encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if not str(r.get("exercise_id") or "").lstrip().startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default is a dry run)")
    ap.add_argument("--allow-unreachable", action="store_true",
                    help="permit cuts that leave a muscle with no incoming edge")
    args = ap.parse_args()

    ont = load_ontology(force=True)
    if not ont["loaded"]:
        print("!! ontology store did not load — refusing to touch anything")
        return 1

    cuts, recl = load_decisions()
    if not cuts and not recl:
        print(f"Nothing to apply from {CUTS}.\n"
              f"Open http://localhost:3000/graph and work through the Audit queue first.")
        return 0

    edges = load_edges()
    wanted = {(c["exercise_id"], c["muscle_id"]) for c in cuts}
    to_limit = {(r["exercise_id"], r["muscle_id"]) for r in recl}
    keeping = [e for e in edges if (e["exercise_id"], e["muscle_id"]) not in wanted]
    removing = [e for e in edges if (e["exercise_id"], e["muscle_id"]) in wanted]

    # Reclassify in place. The edge SURVIVES — only its role changes, so the
    # scheduling fact is kept while the false training volume goes away.
    rewritten = []
    for e in keeping:
        if (e["exercise_id"], e["muscle_id"]) in to_limit and e["role"] != "limiting":
            rewritten.append((e["exercise_id"], e["muscle_id"], e["role"]))
            e["role"] = "limiting"

    name_ex = {str(i): e["canonical_name"] for i, e in ont["exercises"].items()}
    name_mu = {str(i): m["name"] for i, m in ont["muscles"].items()}
    nx = lambda i: name_ex.get(str(i), f"exercise {i}")
    nm = lambda i: name_mu.get(str(i), f"muscle {i}")

    print(f"{len(cuts)} cut(s) and {len(recl)} reclassification(s) recorded · "
          f"{len(removing)} edge(s) to delete, {len(rewritten)} to relabel · "
          f"{len(edges)} -> {len(keeping)} edges\n")

    # ── refusal 1: an exercise must keep a primary ────────────────────────────
    # Applies to relabels too: turning an exercise's only primary into
    # 'limiting' would leave its sets reaching no muscle at all.
    blocked = set()
    for eid in ({e["exercise_id"] for e in removing}
                | {r[0] for r in rewritten}):
        after = [e for e in keeping if e["exercise_id"] == eid]
        if not any(e["role"] == "primary" for e in after):
            blocked.add(eid)
            print(f"  BLOCKED  {nx(eid)}: cutting this leaves it with no primary muscle, "
                  f"so its sets would reach no muscle at all")
    if blocked:
        keeping = edges[:]                       # abandon the whole edit
        print("\nNothing applied. Restore at least one primary on the exercise(s) above, "
              "or reject that cut in the viewer.")
        return 1

    # ── refusal 2: a muscle must stay reachable ───────────────────────────────
    def reachable(edge_rows):
        touched = {e["muscle_id"] for e in edge_rows}
        return {str(mid) for mid, desc in ont["descendants"].items()
                if {str(d) for d in desc} & touched}

    lost = reachable(edges) - reachable(keeping)
    if lost:
        print("  WARNING  these muscles lose their last incoming edge and can no longer "
              "be reported as untouched:")
        for mid in sorted(lost, key=nm):
            print(f"             {nm(mid)}")
        if not args.allow_unreachable:
            print("\nNothing applied. Re-run with --allow-unreachable if that is intended.")
            return 1

    for eid, mid, was in rewritten:
        print(f"  limit  {nx(eid)}  ->  {nm(mid)}  ({was} -> limiting; kept, "
              f"no longer counts as training)")
    for e in removing:
        print(f"  cut    {nx(e['exercise_id'])}  ->  {nm(e['muscle_id'])}  ({e['role']})")

    if not args.apply:
        print(f"\nDRY RUN — nothing written. Re-run with --apply to edit "
              f"{os.path.relpath(EDGES, _ROOT)}.")
        return 0

    tmp = EDGES + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(EDGE_COLUMNS))
        w.writeheader()
        for e in keeping:
            w.writerow({c: e.get(c, "") for c in EDGE_COLUMNS})
    os.replace(tmp, EDGES)
    load_ontology(force=True)

    print(f"\nApplied. {len(removing)} edge(s) removed, {len(rewritten)} relabelled "
          f"to 'limiting' in {os.path.relpath(EDGES, _ROOT)}.")
    print("Next: `git diff ontology/` to review, then `pytest` and "
          "`python scripts/draft_ontology_edges.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
