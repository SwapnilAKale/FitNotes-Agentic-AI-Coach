"""
Operational-path (MCP) cross-unit volume — get_weekly_volume must report volume
per UNIT FRAME (total_volume_lbs / total_volume_kg), never a single blended sum
that adds a kg-native exercise's kilograms onto a category's pounds.

Mirrors the analytical Pass-2 per-unit volume tests, on the chat-agent surface.
No Gemini, no server — calls the sync tool function directly against the real DB.
"""

import json
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from mcp_servers.combined_server import (  # noqa: E402
    _get_weekly_volume_sync, _kg_native_volume_case,
)

_DB = os.environ["FITNOTES_DB_PATH"]


def _all_history_blended() -> dict:
    """The OLD single blended SUM per category (what the bug returned)."""
    conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT c.name AS mg,
                   ROUND(SUM(tl.metric_weight * 2.2046 * tl.reps), 1) AS v
            FROM training_log tl
            JOIN exercise e ON tl.exercise_id = e._id
            JOIN Category c ON e.category_id = c._id
            GROUP BY c.name
        """).fetchall()
    finally:
        conn.close()
    return {r["mg"]: (r["v"] or 0.0) for r in rows}


def _by_group(days: int = 100000) -> dict:
    out = json.loads(_get_weekly_volume_sync(days=days))   # days huge → all history
    return {r["muscle_group"]: r for r in out["volume_by_muscle_group"]}


def test_get_weekly_volume_is_per_unit_not_blended():
    by = _by_group()
    back = by["Back"]
    # New per-unit shape; the old blended single field is gone.
    assert "total_volume" not in back
    assert "total_volume_lbs" in back and "total_volume_kg" in back


def test_kg_native_category_splits_into_both_buckets():
    # Back contains Deadlift (kg-native post-2025-12-26) → both buckets non-zero.
    back = _by_group()["Back"]
    assert back["total_volume_kg"] > 0
    assert back["total_volume_lbs"] > 0


def test_kg_native_free_category_has_zero_kg_bucket():
    # Chest/Legs/Shoulders/Triceps hold no kg-native exercise → kg bucket == 0.
    by = _by_group()
    for cat in ("Chest", "Legs", "Shoulders", "Triceps"):
        assert by[cat]["total_volume_kg"] == 0, (cat, by[cat])
        assert by[cat]["total_volume_lbs"] > 0


def test_every_category_reconciles_lbs_plus_kg_equals_old_blend():
    # No volume created or lost — only split. lbs + kg == old blended sum.
    by = _by_group()
    blended = _all_history_blended()
    for mg, r in by.items():
        expected = blended.get(mg, 0.0)
        got = r["total_volume_lbs"] + r["total_volume_kg"]
        assert abs(got - expected) < 0.2, (mg, got, expected)


def test_back_reconciliation_explicit():
    # The provable Deadlift-blend case: Back splits, sum reconciles.
    back = _by_group()["Back"]
    blended = _all_history_blended()["Back"]
    assert abs((back["total_volume_lbs"] + back["total_volume_kg"]) - blended) < 0.2


def test_kg_native_predicate_reuses_rule():
    lbs_sql, kg_sql = _kg_native_volume_case("tl.metric_weight * 2.2046 * tl.reps")
    assert kg_sql.startswith("SUM(CASE WHEN")
    assert "Deadlift" in kg_sql and "2025-12-26" in kg_sql
    assert "Seated Machine Curl (Kg)" in kg_sql
    assert "NOT (" in lbs_sql                      # lbs bucket is the complement
