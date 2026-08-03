#!/usr/bin/env python3
"""
scripts/review_pending.py
The approval gate for new exercises (R10). Nothing a model proposes enters the
muscle ontology except through this script, and only for rows you marked
approved=y.

Workflow:

    python scripts/review_pending.py                 # show the queue
    python scripts/review_pending.py --propose       # web-search a draft for each
    <edit ontology/pending_review.csv: set approved to y or n>
    python scripts/review_pending.py --promote       # write the y rows into the graph

--propose is the only step that touches the network. It never writes to the
store; it only fills in the decision/muscles/evidence/sources columns so you
have something to judge. --promote is the only step that writes, and it can
never touch muscles.csv: the muscle set is closed (R7).

Until a row is promoted, that exercise's sets are excluded from every muscle
number and the agent says so.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ontology import load_ontology                      # noqa: E402
from src import ontology_reconcile as rec                   # noqa: E402


def _fmt(row: dict) -> str:
    name = row.get("db_exercise_name", "?")
    decision = (row.get("decision") or "pending").lower()
    approved = (row.get("approved") or "").strip().lower()
    mark = {"y": "[APPROVED]", "n": "[rejected]"}.get(approved, "[   ?   ]")
    sets = row.get("logged_sets") or "0"

    body = []
    if decision == "alias":
        body.append(f"      -> SAME AS: {row.get('alias_of')}")
    elif decision == "new":
        body.append(f"      -> NEW, muscles: {row.get('muscles')}")
    elif decision == "unsure":
        body.append("      -> UNSURE (stays excluded until you decide)")
    else:
        body.append("      -> not proposed yet (run --propose)")
    if row.get("evidence"):
        body.append(f"         {row['evidence']}")
    if row.get("sources"):
        body.append(f"         sources: {row['sources']}")
    cat = row.get("fitnotes_category")
    if cat:
        body.append(f"         app category (weak hint): {cat}")
    return f"  {mark} {sets:>5} sets  {name}\n" + "\n".join(body)


def show(rows: list) -> int:
    if not rows:
        print("Nothing pending — every logged exercise is mapped.")
        return 0
    print(f"{len(rows)} exercise(s) awaiting review "
          f"(their sets are EXCLUDED from muscle counts until promoted)\n")
    for row in rows:
        print(_fmt(row))
        print()
    undecided = sum(1 for r in rows if not (r.get("approved") or "").strip())
    print(f"{undecided} still undecided. Edit ontology/{rec.PENDING_FILE}, set "
          f"approved to y or n, then run --promote.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--propose", action="store_true",
                    help="draft a mapping for each undecided row using web search")
    ap.add_argument("--promote", action="store_true",
                    help="write approved=y rows into the ontology")
    args = ap.parse_args()

    ontology = load_ontology(force=True)
    if not ontology["loaded"]:
        print("!! ontology store did not load — refusing to do anything")
        return 1

    rows = rec.load_pending()

    if args.propose:
        if not rows:
            print("Nothing pending.")
            return 0
        # Imported lazily: this is the only path that needs the LLM SDK and a
        # network connection, and `--promote` / plain listing must work offline.
        from src import ontology_propose as propose
        print(f"Searching for {len(rows)} exercise(s)...\n")
        rows = propose.propose_all(rows, ontology)
        rec.save_pending(rows)
        print("Proposals written to the queue. NOTHING has entered the ontology "
              "yet — review below, then --promote.\n")
        return show(rows)

    if args.promote:
        result = rec.promote(ontology)
        print(result.summary())
        for p in result.promoted:
            if p["as"] == "alias":
                print(f"  + alias  {p['name']!r} -> {p['target']!r}")
            else:
                print(f"  + new    {p['name']!r} -> {', '.join(p['muscles'])}")
        for name, why in result.skipped:
            print(f"  ! skipped {name!r}: {why}")
        if result.skipped:
            print("\nSkipped rows stay in the queue. A muscle outside the closed "
                  "set is never created — fix the row or reject it.")
        return 0

    return show(rows)


if __name__ == "__main__":
    raise SystemExit(main())
