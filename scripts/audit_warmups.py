#!/usr/bin/env python3
"""
scripts/audit_warmups.py
Read-only diagnostic — reports warmup-flagged sets across the FULL DB history.

Uses the exact same code path the Data Agent package uses:
  fetch_data()               -> alltime_rows
  _build_sessions_from_rows()  -> per-exercise sessions with is_warmup flags set
                                  by _detect_warmup_flags()

Nothing is reimplemented; if the warmup rule changes in process.py the output
of this script changes automatically.

Run:
    python scripts/audit_warmups.py
"""

import os
import sys
import logging
from collections import defaultdict
from datetime import date

# ── Project root on sys.path ───────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

# Suppress data_agent logger noise so only our output reaches stdout
logging.basicConfig(level=logging.WARNING)
logging.getLogger("src.data_agent").setLevel(logging.CRITICAL)

from src.data_agent.fetch   import fetch_data, load_user_context                         # noqa: E402
from src.data_agent.process import (_build_sessions_from_rows,                           # noqa: E402
                                     _get_numeric_offset, _get_bar_weight_lbs,
                                     _is_kg_native, _recover_typed_weight)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    today_str = date.today().strftime("%Y-%m-%d")

    print(f"audit_warmups.py  —  full-history scan to {today_str}")
    print("Loading DB and user_context ...", end=" ", flush=True)

    ctx          = load_user_context()
    bundle       = fetch_data(end_str=today_str)
    alltime_rows = bundle["alltime_rows"]

    print(f"done  ({len(alltime_rows):,} rows across all exercises)\n")

    # Group raw rows by exercise name (preserving DB order)
    by_exercise: dict = defaultdict(list)
    for row in alltime_rows:
        by_exercise[row["exercise_name"]].append(row)

    # Compute the same warmup eligibility and alltime maxima used by process_data
    _first_in_cat:   dict = {}
    _ex_alltime_max: dict = {}
    for r in alltime_rows:
        key = (r["date"], r["category_id"])
        sid = r["set_id"]
        if key not in _first_in_cat or sid < _first_in_cat[key][0]:
            _first_in_cat[key] = (sid, r["exercise_name"])
        _ek  = r["exercise_name"]
        _off = _get_numeric_offset(ctx, _ek)
        _bar_lbs = _get_bar_weight_lbs(ctx, _ek, r["date"])
        _bar     = _bar_lbs / 2.2046 if _is_kg_native(ctx, _ek, r["date"]) else _bar_lbs
        _hw      = _recover_typed_weight(r["metric_weight"], _off) + _bar
        if _hw > _ex_alltime_max.get(_ek, 0.0):
            _ex_alltime_max[_ek] = _hw
    warmup_eligible: frozenset = frozenset(
        (ex_name, date)
        for (date, _cat), (_sid, ex_name) in _first_in_cat.items()
    )

    # ── Walk every exercise, build sessions, collect warmup sets ───────────────
    warmup_records: list = []  # list of dicts

    for ex_name in sorted(by_exercise):
        rows     = by_exercise[ex_name]
        sessions = _build_sessions_from_rows(rows, ctx, ex_name,
                                             warmup_eligible=warmup_eligible,
                                             exercise_alltime_max=_ex_alltime_max.get(ex_name, 0.0))
        for s in sessions:
            unit = s["unit"]
            for st in s["sets"]:
                if st.get("is_warmup"):
                    warmup_records.append({
                        "exercise": ex_name,
                        "date":     s["date"],
                        "weight":   st["weight"],
                        "unit":     unit,
                        "reps":     st["reps"],
                        "comment":  st.get("comment"),
                    })

    warmup_records.sort(key=lambda r: (r["date"], r["exercise"]))

    if not warmup_records:
        print("No warmup sets found in the full history.")
        return

    # ── 1. Earliest warmup set ever ────────────────────────────────────────────
    earliest = warmup_records[0]
    print("=" * 60)
    print("1.  Earliest warmup set ever recorded")
    print("=" * 60)
    print(f"    Date:     {earliest['date']}")
    print(f"    Exercise: {earliest['exercise']}")
    print(f"    Weight:   {earliest['weight']} {earliest['unit']}")
    print(f"    Reps:     {earliest['reps']}")
    if earliest["comment"]:
        print(f"    Comment:  {earliest['comment']!r}")

    # ── 2. Per-exercise table ──────────────────────────────────────────────────
    by_ex: dict = defaultdict(list)
    for rec in warmup_records:
        by_ex[rec["exercise"]].append(rec)

    table_rows = []
    for ex_name, recs in by_ex.items():
        dates = [r["date"] for r in recs]
        table_rows.append({
            "exercise":     ex_name,
            "count":        len(recs),
            "earliest":     min(dates),
            "latest":       max(dates),
        })
    table_rows.sort(key=lambda r: (-r["count"], r["exercise"]))

    print()
    print("=" * 60)
    print("2.  Exercises with >=1 warmup set  (sorted by count desc)")
    print("=" * 60)
    col_ex = max(len(r["exercise"]) for r in table_rows)
    col_ex = max(col_ex, len("Exercise"))
    hdr = (f"  {'Exercise':<{col_ex}}  {'Count':>5}  "
           f"{'Earliest':>10}  {'Latest':>10}")
    sep = "  " + "-" * (col_ex) + "  " + "-"*5 + "  " + "-"*10 + "  " + "-"*10
    print(hdr)
    print(sep)
    for r in table_rows:
        print(f"  {r['exercise']:<{col_ex}}  {r['count']:>5}  "
              f"{r['earliest']:>10}  {r['latest']:>10}")

    # ── 3. Totals ──────────────────────────────────────────────────────────────
    total_sets      = len(warmup_records)
    total_exercises = len(by_ex)
    # A "session with a warmup" = unique (exercise, date) pair
    sessions_with_warmup = len({(r["exercise"], r["date"]) for r in warmup_records})

    print()
    print("=" * 60)
    print("3.  Totals")
    print("=" * 60)
    print(f"    Total warmup sets logged:            {total_sets:>6,}")
    print(f"    Distinct exercises with a warmup:    {total_exercises:>6,}")
    print(f"    Sessions containing a warmup set:    {sessions_with_warmup:>6,}")


if __name__ == "__main__":
    main()
