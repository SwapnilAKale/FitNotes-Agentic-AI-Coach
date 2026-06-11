#!/usr/bin/env python3
"""
scripts/audit_package_size.py
Read-only diagnostic — reports where the bytes live in the broad analytical
package that the Coordinator builds for a wide-open question.

Builds: prepare_analysis_package(query_period_days=365, no filters)
No pipeline changes; no LLM calls; pure data-layer.

Run:
    python scripts/audit_package_size.py
"""

import os
import sys
import json
from collections import defaultdict

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

# Silence data-agent log noise during package build
import logging
logging.disable(logging.WARNING)

from src.data_agent import prepare_analysis_package

# ── Helpers ───────────────────────────────────────────────────────────────────

def _kb(obj) -> float:
    """Serialized size of obj in KB."""
    return len(json.dumps(obj, default=str).encode()) / 1024


def _bar(pct: float, width: int = 30) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)


def _fmt(kb: float, total_kb: float) -> str:
    pct = kb / total_kb * 100 if total_kb else 0
    return f"{kb:8.1f} KB  {pct:5.1f}%  {_bar(pct)}"


# ── Build package ─────────────────────────────────────────────────────────────
print("Building prepare_analysis_package(query_period_days=365) …", flush=True)

pkg = prepare_analysis_package(query_period_days=365)

logging.disable(logging.NOTSET)

# ── 1. Total size ─────────────────────────────────────────────────────────────
raw_json  = json.dumps(pkg, default=str).encode()
total_kb  = len(raw_json) / 1024

print()
print("-" * 70)
print(f"  PACKAGE SIZE AUDIT  —  365-day window, all exercises, no filters")
print("-" * 70)
print(f"\n  Total serialized size: {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)\n")

# ── 2. Top-level key sizes ────────────────────────────────────────────────────
print("-" * 70)
print("  TOP-LEVEL KEY SIZES")
print("-" * 70)
key_sizes = []
for k, v in pkg.items():
    key_sizes.append((k, _kb(v)))
key_sizes.sort(key=lambda x: x[1], reverse=True)

for k, kb in key_sizes:
    print(f"  {k:<35s} {_fmt(kb, total_kb)}")

# ── 3. Per-exercise breakdown (top 10) ────────────────────────────────────────
exercises = pkg.get("exercises", [])
print()
print("-" * 70)
print(f"  PER-EXERCISE (top 10 of {len(exercises)})")
print("-" * 70)
ex_sizes = []
for ex in exercises:
    ex_sizes.append((ex.get("name", "?"), _kb(ex)))
ex_sizes.sort(key=lambda x: x[1], reverse=True)

for name, kb in ex_sizes[:10]:
    print(f"  {name:<40s} {_fmt(kb, total_kb)}")

if len(ex_sizes) > 10:
    rest_kb = sum(kb for _, kb in ex_sizes[10:])
    print(f"  {'… remaining ' + str(len(ex_sizes) - 10) + ' exercises':<40s} {_fmt(rest_kb, total_kb)}")

# ── 4. Within-exercise field breakdown (aggregated) ───────────────────────────
print()
print("-" * 70)
print("  WITHIN-EXERCISE FIELD BREAKDOWN (aggregated across all exercises)")
print("-" * 70)
field_totals: dict = defaultdict(float)
for ex in exercises:
    for field, val in ex.items():
        field_totals[field] += _kb(val)

field_list = sorted(field_totals.items(), key=lambda x: x[1], reverse=True)
for field, kb in field_list:
    print(f"  {field:<35s} {_fmt(kb, total_kb)}")

# ── 5. Counts for context ─────────────────────────────────────────────────────
print()
print("-" * 70)
print("  COUNTS")
print("-" * 70)

n_exercises = len(exercises)
n_sessions  = sum(len(ex.get("sessions", [])) for ex in exercises)
n_sets      = sum(
    len(s.get("sets", []))
    for ex in exercises
    for s in ex.get("sessions", [])
)
n_comments_included = sum(
    len(ex.get("full_comments") or [])
    for ex in exercises
)
n_ex_with_comments = sum(
    1 for ex in exercises if ex.get("full_comments")
)
n_sets_with_sets = sum(
    1 for ex in exercises
    for s in ex.get("sessions", [])
    if s.get("sets")
)

print(f"  Exercises analyzed          : {n_exercises}")
print(f"  Total sessions              : {n_sessions}")
print(f"  Total sets (w/ set arrays)  : {n_sets}")
print(f"  Exercises with full_comments: {n_ex_with_comments}")
print(f"  Total full_comment entries  : {n_comments_included}")

# Also report sizes for the non-exercise heavy keys
print()
print("-" * 70)
print("  NON-EXERCISE HEAVY KEYS (detail)")
print("-" * 70)
for k in ("daily_workouts", "superset_patterns", "exercise_lifecycle",
          "rankings", "seasonal_patterns", "day_of_week_patterns",
          "training_consistency", "all_time_summary"):
    if k in pkg:
        kb = _kb(pkg[k])
        print(f"  {k:<35s} {_fmt(kb, total_kb)}")

print()
print("-" * 70)
