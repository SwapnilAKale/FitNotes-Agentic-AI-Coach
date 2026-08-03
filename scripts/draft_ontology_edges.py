#!/usr/bin/env python3
"""
scripts/draft_ontology_edges.py
Read-only curation aid for the muscle ontology. No LLM, no writes to the DB.

Two modes:

  --draft     Emit ontology/exercises.draft.csv and ontology/aliases.draft.csv
              seeded from the exercises the user has actually LOGGED (Tier A).
              Review them, then drop the ".draft" from the filenames.

  (default)   Coverage report against the CURRENT store — the thing to re-run
              after every curation pass:
                • logged exercises with no alias  (their sets are excluded)
                • aliased exercises with no primary edge
                • muscles with no incoming edge at all
                • the two known-miscategorised lifts, called out by name

Categories 10/11/12 (Time / Place / Neck) are excluded, matching
EXCLUDED_CATEGORY_IDS in src/data_agent/fetch.py — "Morning" and "Society" are
not exercises and must never appear in the ontology.

Run:
    python scripts/draft_ontology_edges.py
    python scripts/draft_ontology_edges.py --draft
"""

import argparse
import csv
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ontology import (load_ontology, resolve_db_exercise,   # noqa: E402
                          is_unattributed)

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
ONTOLOGY_DIR = os.environ.get("ONTOLOGY_DIR", os.path.join(_ROOT, "ontology"))

# Lifts whose single FitNotes category is anatomically wrong. These are the
# concrete proof the arc exists, so the draft flags them rather than letting a
# category-seeded guess be rubber-stamped.
KNOWN_MISCATEGORISED = {
    "Smith Machine Shrugs":  "filed Back — it is Upper Traps",
    "Reverse Cable Curls":   "filed Forearms — it is Brachioradialis / Brachialis (elbow flexion)",
}


def logged_exercises() -> list:
    """[(db_name, category_name, set_count)] for every exercise with logged sets."""
    normalized = DB_PATH.replace("\\", "/")
    conn = sqlite3.connect(f"file:{normalized}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT e.name AS name, c.name AS category, COUNT(*) AS sets
                 FROM training_log tl
                 JOIN exercise e  ON e._id = tl.exercise_id
                 JOIN Category c  ON c._id = e.category_id
                WHERE e.category_id NOT IN (10, 11, 12)
             GROUP BY e.name
             ORDER BY COUNT(*) DESC""").fetchall()
        return [(r["name"], r["category"], r["sets"]) for r in rows]
    finally:
        conn.close()


def _canonical(db_name: str) -> str:
    """Title-cased canonical form. A starting point for review, not an answer —
    real typos in the DB ("Dumbell Wrist Curls") are fixed by hand afterwards."""
    small = {"the", "with", "of", "to", "and"}
    words = db_name.split()
    return " ".join(w.lower() if i and w.lower() in small else
                    (w if any(ch.isupper() for ch in w[1:]) else w.capitalize())
                    for i, w in enumerate(words))


def write_drafts() -> None:
    rows = logged_exercises()
    ex_path    = os.path.join(ONTOLOGY_DIR, "exercises.draft.csv")
    alias_path = os.path.join(ONTOLOGY_DIR, "aliases.draft.csv")
    os.makedirs(ONTOLOGY_DIR, exist_ok=True)

    with open(ex_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "canonical_name", "equipment", "movement_pattern"])
        for i, (name, _cat, _n) in enumerate(rows, start=1):
            w.writerow([i, _canonical(name), "", ""])

    with open(alias_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["db_exercise_name", "exercise_id"])
        for i, (name, _cat, _n) in enumerate(rows, start=1):
            w.writerow([name, i])

    print(f"wrote {ex_path}    ({len(rows)} exercises)")
    print(f"wrote {alias_path} ({len(rows)} aliases)")
    print("\nReview both, then drop the '.draft' from each filename.")
    print("exercise_muscle.csv is authored separately — primaries first, "
          "secondaries only where a muscle is a SIGNIFICANT part of the lift.")


def report() -> int:
    ont = load_ontology(force=True)
    rows = logged_exercises()

    if not ont["loaded"] or not ont["muscles"]:
        print(f"!! ontology store at {ONTOLOGY_DIR} did not load — nothing to report")
        return 1

    print(f"ontology: {len(ont['muscles'])} muscles, {len(ont['exercises'])} exercises, "
          f"{len(ont['edges'])} edges, {len(ont['aliases'])} aliases")
    if ont["errors"]:
        print(f"\n!! {len(ont['errors'])} store problem(s):")
        for e in ont["errors"]:
            print(f"   {e}")

    unmapped, no_primary, unattributed = [], [], []
    mapped_sets = unmapped_sets = unattributed_sets = 0
    for name, category, sets in rows:
        eid = resolve_db_exercise(ont, name)
        if eid is None:
            unmapped.append((name, category, sets))
            unmapped_sets += sets
            continue
        if is_unattributed(ont, eid):
            # Deliberately edge-free (cardio) — not a curation gap.
            unattributed.append((name, category, sets))
            unattributed_sets += sets
            continue
        mapped_sets += sets
        edges = ont["edges_by_exercise"].get(eid, [])
        if not any(e["role"] == "primary" for e in edges):
            no_primary.append((name, category, sets))

    print(f"\nlogged exercises: {len(rows)}   mapped "
          f"{len(rows) - len(unmapped) - len(unattributed)} / "
          f"cardio {len(unattributed)} / unmapped {len(unmapped)}")
    print(f"logged sets:      {mapped_sets + unmapped_sets + unattributed_sets:,}   "
          f"mapped {mapped_sets:,} / cardio {unattributed_sets:,} / "
          f"UNMAPPED {unmapped_sets:,}")

    if unmapped:
        print("\n-- no alias (sets are excluded from all muscle maths) --")
        for name, category, sets in unmapped:
            print(f"   {sets:>5}  [{category:<9}] {name}")
    if no_primary:
        print("\n-- aliased but no PRIMARY edge --")
        for name, category, sets in no_primary:
            print(f"   {sets:>5}  [{category:<9}] {name}")
    if unattributed:
        print("\n-- cardio: no muscle edges BY DESIGN --")
        for name, category, sets in unattributed:
            print(f"   {sets:>5}  [{category:<9}] {name}")

    touched = {e["muscle_id"] for e in ont["edges"]}
    # A parent counts as touched when any descendant is touched (the rollup rule).
    rolled = {mid for mid, desc in ont["descendants"].items() if desc & touched}
    orphans = [m["name"] for mid, m in sorted(ont["muscles"].items())
               if mid not in rolled]
    if orphans:
        print(f"\n-- muscles with no edge anywhere beneath them ({len(orphans)}) --")
        print("   " + ", ".join(orphans))

    print("\n-- known-miscategorised lifts (verify these by hand) --")
    for name, why in KNOWN_MISCATEGORISED.items():
        eid = resolve_db_exercise(ont, name)
        edges = ont["edges_by_exercise"].get(eid, []) if eid else []
        got = ", ".join(f"{ont['muscles'][e['muscle_id']]['name']}({e['role']})"
                        for e in edges) or "NOTHING"
        print(f"   {name}\n       {why}\n       -> {got}")

    return 0 if not (unmapped or no_primary or ont["errors"]) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true",
                    help="emit exercises.draft.csv + aliases.draft.csv and exit")
    args = ap.parse_args()
    if args.draft:
        write_drafts()
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
