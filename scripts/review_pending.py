#!/usr/bin/env python3
"""
scripts/review_pending.py
The approval gate for new exercises (R10). Nothing a model proposes enters the
muscle ontology except through this script, and only for rows you marked
approved=y.

Workflow:

    python scripts/review_pending.py                 # show the queue
    python scripts/review_pending.py --sweep         # find logged-but-unmapped
    python scripts/review_pending.py --propose       # web-search a draft for each
    <edit ontology/pending_review.csv: set approved to y or n>
    python scripts/review_pending.py --promote       # write the y rows into the graph

--sweep rescans the database for exercises that have LOGGED SETS but no place in
the graph, and queues them. Reconciliation normally does this at upload time;
--sweep is how you catch up without waiting for one, and it is safe to re-run —
a row you have already reviewed is never reset.

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
# The API key lives in .env. server.py loads it at startup; this script is a
# separate entry point and never did — so --propose, the one path here that needs
# the key, could never reach the model when run as documented.
from dotenv import load_dotenv                              # noqa: E402

KEY_VAR = "GEMINI_API_KEY"
SEARCH_KEY_VAR = "TAVILY_API_KEY"     # the search is run by code, not the model


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
        pattern = (row.get("movement_pattern") or "").strip() or "no pattern"
        body.append(f"      -> NEW ({pattern}), muscles: {row.get('muscles')}")
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


_LEGACY_EXCLUDED_CATEGORY_IDS = (10, 11, 12)


def _migrate_legacy_exclusions(db_path: str) -> list:
    """ONE-TIME: move the old category-id exclusion into the dismissal list.

    "Not an exercise" used to mean `category_id IN (10, 11, 12)` — this user's
    ids, hardcoded in nine places. It is now a named list. Without this step the
    switch-over would briefly let those entries into volume, session and count
    queries and every number would move; with it, nothing is reclassified and
    nothing is silently hidden — the names that were ALREADY being excluded are
    simply written down where the whole system can read them.

    This records an existing decision in a better place. It does not make one.
    """
    import sqlite3
    if rec.not_exercise_names():
        return []                       # already migrated
    try:
        conn = sqlite3.connect(
            f"file:{str(db_path).replace(os.sep, '/')}?mode=ro", uri=True)
        try:
            holes = ", ".join("?" * len(_LEGACY_EXCLUDED_CATEGORY_IDS))
            names = [r[0] for r in conn.execute(
                f"""SELECT DISTINCT e.name FROM training_log tl
                      JOIN exercise e ON e._id = tl.exercise_id
                     WHERE e.category_id IN ({holes})""",
                _LEGACY_EXCLUDED_CATEGORY_IDS).fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    if names:
        rec.dismiss_as_not_exercise(
            names, note="migrated from the category-id exclusion")
    return sorted(names)


def sweep(db_path: str) -> int:
    """Queue every logged exercise the graph does not know."""
    migrated = _migrate_legacy_exclusions(db_path)
    if migrated:
        print(f"Migrated {len(migrated)} name(s) into "
              f"ontology/{rec.NOT_EXERCISES_FILE} — these were ALREADY being "
              f"excluded as non-exercises, now recorded by name:")
        for n in migrated:
            print(f"    {n}")
        print("  Delete a line from that file if any of them is a real exercise.\n")

    names = rec.logged_exercise_names(db_path)

    # QUEUE HYGIENE. The old reconciliation queued newly-DEFINED exercises, so
    # the queue can hold rows for things never trained ("Good Morning", 0 sets).
    # Nothing without logged sets needs a mapping, and asking about it is noise.
    # Only UNDECIDED rows are dropped — a row the user has already answered is
    # never discarded.
    trained = {n.lower() for n in names}
    queue = rec.load_pending()
    stale = [q for q in queue
             if (q.get("db_exercise_name") or "").lower() not in trained
             and not (q.get("approved") or "").strip()
             and (q.get("decision") or "pending").strip().lower() == "pending"]
    if stale:
        rec.save_pending([q for q in queue if q not in stale])
        print(f"Dropped {len(stale)} queued row(s) with no logged sets "
              f"(left over from the old defined-exercise rule):")
        for q in stale:
            print(f"    {q.get('db_exercise_name')}")
        print()

    result = rec.detect_new(names, load_ontology(force=True),
                            *_context(db_path))
    if result.auto_aliased:
        rec.write_auto_aliases(result.auto_aliased)
        load_ontology(force=True)
        print(f"Auto-aliased {len(result.auto_aliased)} obvious name variant(s):")
        for a in result.auto_aliased:
            print(f"    {a['db_exercise_name']}  ->  {a['canonical_name']}")
        print()
    if result.pending:
        rec.merge_pending(result.pending)
    print(f"Swept {len(names)} logged exercise(s): {result.summary()}\n")
    return 0


def _context(db_path: str) -> tuple:
    """(set_counts, categories) — ordering and hints only, never the decision."""
    import sqlite3
    try:
        conn = sqlite3.connect(
            f"file:{str(db_path).replace(os.sep, '/')}?mode=ro", uri=True)
        try:
            counts = {r[0]: r[1] for r in conn.execute(
                """SELECT e.name, COUNT(*) FROM training_log tl
                     JOIN exercise e ON e._id = tl.exercise_id
                    GROUP BY e.name""")}
            cats = {r[0]: r[1] for r in conn.execute(
                """SELECT e.name, c.name FROM exercise e
                     JOIN Category c ON c._id = e.category_id""")}
            return counts, cats
        finally:
            conn.close()
    except sqlite3.Error:
        return {}, {}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true",
                    help="queue every logged exercise the graph does not know")
    ap.add_argument("--propose", action="store_true",
                    help="draft a mapping for each undecided row using web search")
    ap.add_argument("--promote", action="store_true",
                    help="write approved=y rows into the ontology")
    args = ap.parse_args()

    ontology = load_ontology(force=True)
    if not ontology["loaded"]:
        print("!! ontology store did not load — refusing to do anything")
        return 1

    if args.sweep:
        db = os.environ.get("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
        sweep(db)
        return show(rec.load_pending())

    rows = rec.load_pending()

    if args.propose:
        if not rows:
            print("Nothing pending.")
            return 0
        # Refuse BEFORE spending anything. Without this, every row failed one by
        # one, was written back, and the run exited 0 as if it had worked.
        missing = [v for v in (KEY_VAR, SEARCH_KEY_VAR) if not os.environ.get(v)]
        if missing:
            print(f"!! {' and '.join(missing)} not set (looked in the environment "
                  f"and .env). Nothing was proposed and the queue is unchanged.")
            return 2
        # Imported lazily: this is the only path that needs the LLM SDK and a
        # network connection, and `--promote` / plain listing must work offline.
        from src import ontology_propose as propose
        print(f"Searching for {len(rows)} exercise(s)...\n")
        rows = propose.propose_all(rows, ontology)
        rec.save_pending(rows)
        failed = [r for r in rows
                  if (r.get("evidence") or "").startswith(propose.NOT_PROPOSED)]
        print("Proposals written to the queue. NOTHING has entered the ontology "
              "yet — review below, then --promote.\n")
        show(rows)
        if failed:
            # Real proposals from rows that succeeded are kept; the failures stay
            # pending and are retried next run. But the run did not fully work,
            # and the exit code must say so.
            print(f"\n!! {len(failed)} of {len(rows)} could NOT be proposed — the "
                  f"model was not reached. First reason: "
                  f"{failed[0]['evidence'][len(propose.NOT_PROPOSED):].strip()}")
            return 1
        return 0

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
