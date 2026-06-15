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


def _analytical_mg() -> dict:
    """Analytical bar-inclusive per-category volume (the surface we must match)."""
    from src.data_agent import collect
    pkg = collect(query_period_days=None, aggregation_level="session")
    return {m["muscle_group"]: m for m in pkg["muscle_group_summary"]}


def _deadlift_plates_and_reps() -> tuple:
    """(plates_kg_vol, reps_in_kg_frame, plates_lbs_vol, reps_in_lbs_frame) for Deadlift."""
    from src.data_agent.process import _is_kg_native
    from src.data_agent.fetch import load_user_context
    ctx = load_user_context()
    conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT tl.date, tl.metric_weight, tl.reps FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id = e._id WHERE e.name = 'Deadlift'"
        ).fetchall()
    finally:
        conn.close()
    pk = pl = 0.0
    rk = rl = 0
    for r in rows:
        v = r["metric_weight"] * 2.2046 * r["reps"]
        if _is_kg_native(ctx, "Deadlift", r["date"]):
            pk += v; rk += r["reps"]
        else:
            pl += v; rl += r["reps"]
    return pk, rk, pl, rl


def test_operational_volume_now_bar_inclusive_matches_analytical():
    # THE point: operational per-category volume now equals the analytical
    # bar-inclusive muscle_group_summary. kg side is exact (kg-native exercises
    # have no Smith counterbalance); lbs side is exact except Legs (Smith-squat
    # counterbalance reductions the operational pass doesn't apply — documented).
    op = _by_group()
    an = _analytical_mg()
    for cat in ("Back", "Biceps", "Chest", "Forearms", "Shoulders", "Triceps"):
        assert abs(op[cat]["total_volume_lbs"] - an[cat]["total_volume_lbs"]) < 2.0, cat
        assert abs(op[cat]["total_volume_kg"]  - an[cat]["total_volume_kg"])  < 2.0, cat
    # kg buckets agree for every category (no Smith counterbalance in kg frame)
    for cat in op:
        if cat in an:
            assert abs(op[cat]["total_volume_kg"] - an[cat]["total_volume_kg"]) < 2.0, cat


def test_bar_inclusive_exceeds_old_plates_only():
    # Bar (+offset) was added: a bar/offset category's total is now strictly
    # larger than the old plates-only blend.
    by = _by_group()
    plates = _all_history_blended()
    for cat in ("Back", "Forearms", "Biceps"):
        new_total = by[cat]["total_volume_lbs"] + by[cat]["total_volume_kg"]
        assert new_total > plates[cat] + 1.0, cat


def test_deadlift_bar_lands_in_kg_frame_only():
    # Deadlift's 20kg bar is added in the KG bucket: analytical kg ==
    # plates_kg + 20 * reps_kg; the pre-switch lbs sessions get the 44.09lbs bar
    # in the LBS bucket. No bar leaks across the frame boundary.
    pk, rk, pl, rl = _deadlift_plates_and_reps()
    from src.data_agent import collect
    ex = next(e for e in collect(query_period_days=None, exercise_names=["Deadlift"],
                                 aggregation_level="session")["exercises"]
              if e["name"] == "Deadlift")
    an_kg  = sum(s["total_volume"] for s in ex["sessions"] if s["unit"] == "kg")
    an_lbs = sum(s["total_volume"] for s in ex["sessions"] if s["unit"] == "lbs")
    assert abs(an_kg  - (pk + 20.0   * rk)) < 2.0      # 20 kg bar, kg frame
    assert abs(an_lbs - (pl + 44.09  * rl)) < 2.0      # 44.09 lbs bar, lbs frame


def test_no_bar_no_offset_exercise_unchanged():
    # Lat Pulldown has no bar and no offset → bar-inclusive == plates-only,
    # and it stays entirely in the lbs frame.
    from src.data_agent import collect
    ex = next(e for e in collect(query_period_days=None, exercise_names=["Lat Pulldown"],
                                 aggregation_level="session")["exercises"]
              if e["name"] == "Lat Pulldown")
    an_lbs = sum(s["total_volume"] for s in ex["sessions"] if s["unit"] == "lbs")
    an_kg  = sum(s["total_volume"] for s in ex["sessions"] if s["unit"] == "kg")
    conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    try:
        plates = conn.execute(
            "SELECT ROUND(SUM(tl.metric_weight*2.2046*tl.reps),1) v FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id=e._id WHERE e.name='Lat Pulldown'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert an_kg == 0.0
    assert abs(an_lbs - plates) < 5.0      # equal up to per-set rounding


def test_kg_native_predicate_reuses_rule():
    lbs_sql, kg_sql = _kg_native_volume_case("tl.metric_weight * 2.2046 * tl.reps")
    assert kg_sql.startswith("SUM(CASE WHEN")
    assert "Deadlift" in kg_sql and "2025-12-26" in kg_sql
    assert "Seated Machine Curl (Kg)" in kg_sql
    assert "NOT (" in lbs_sql                      # lbs bucket is the complement
