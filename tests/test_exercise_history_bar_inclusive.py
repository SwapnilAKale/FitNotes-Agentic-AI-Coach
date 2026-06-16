"""
#6 — get_exercise_history (and get_exercise_sessions) report BAR-INCLUSIVE
weights, consistent with the analytical package.

Before this fix both operational "recent sets" reads returned plates-only
(metric_weight * 2.2046 + offset, NO bar), so barbell/Smith exercises read
~bar-weight light — e.g. Barbell Curl showed 30 lbs when the bar-inclusive
headline is 63 lbs. Both now call ONE shared conversion
(_bar_inclusive_weight) that composes the analytical-path primitives
(process._get_bar_weight_lbs / _is_kg_native / _recover_typed_weight /
_get_numeric_offset) — the same source of truth the package and
get_weekly_volume use.

No Gemini, no server. Runs against the pinned project DB (same as the golden
tests).
"""

import json
import os
import sqlite3
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from mcp_servers import combined_server as srv          # noqa: E402
from mcp_servers.combined_server import (                # noqa: E402
    _get_exercise_history_sync, _get_exercise_sessions_sync,
    _bar_inclusive_weight, DB_PATH,
)
from src.data_agent import prepare_analysis_package      # noqa: E402
from src.data_agent.fetch import load_user_context       # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _history(name, days=400):
    return json.loads(_get_exercise_history_sync(name, days))


def _sessions(name, **kw):
    return json.loads(_get_exercise_sessions_sync(
        {"exercise_name": name, "mode": "recent", **kw}))


def _pkg_ex(name):
    p = prepare_analysis_package(query_period_days=420, exercise_names=[name])
    exs = p["exercises"]
    return exs[name] if isinstance(exs, dict) else exs[0]


def _raw_rows(name):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT tl.date, tl.metric_weight, tl.reps FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id = e._id WHERE e.name = ? "
            "ORDER BY tl.date DESC", (name,)).fetchall()
    finally:
        conn.close()


# ── core consistency: history weight == sessions weight (same barbell sets) ──

def test_history_max_equals_sessions_max_barbell():
    """The core consistency check: for the SAME barbell sessions, the per-date
    max weight reported by get_exercise_history equals get_exercise_sessions."""
    h = _history("Barbell Curl")
    se = _sessions("Barbell Curl", limit=20)

    hist_max = {}
    for r in h["rows"]:
        hist_max[r["date"]] = max(hist_max.get(r["date"], 0.0), r["typed_value"])
    sess_max = {x["date"]: x["max_weight"] for x in se["sessions"]}

    common = set(hist_max) & set(sess_max)
    assert common, "no overlapping dates between history and sessions"
    for d in sorted(common):
        assert hist_max[d] == sess_max[d], (d, hist_max[d], sess_max[d])


def test_barbell_is_bar_inclusive_and_matches_package():
    """Barbell Curl on 2026-05-28: plates-only would be 30 lbs; bar-inclusive is
    ~63 (plates 30 + ~33 lbs date-ranged curl bar), matching the package PR."""
    h = _history("Barbell Curl")
    day = [r for r in h["rows"] if r["date"] == "2026-05-28"]
    assert day, "expected Barbell Curl sets on 2026-05-28 in the pinned DB"
    top = max(r["typed_value"] for r in day)

    assert top > 60, f"bar not added — got {top} (plates-only would be ~30)"
    assert all(r["unit"] == "lbs" for r in day)

    ex = _pkg_ex("Barbell Curl")
    assert ex["bar_weight"] > 0
    # matches the analytical package PR to display (1-decimal) precision
    assert round(ex["pr"]["weight"], 1) == top


# ── kg-native: reported in kg, bar applied, history == sessions ──────────────

def test_kg_native_deadlift_kg_bar_inclusive():
    h = _history("Deadlift")
    se = _sessions("Deadlift", limit=20)

    # Deadlift is date-ranged: kg only from 2025-12-26 (DEADLIFT_KG_SWITCH).
    recent = [r for r in h["rows"] if r["date"] >= "2025-12-26"]
    assert recent, "expected post-switch Deadlift rows"
    assert all(r["unit"] == "kg" for r in recent), "post-2025-12-26 Deadlift is kg"

    # history and sessions must AGREE per date — same unit (date-ranged) and the
    # same bar-inclusive max — for every overlapping session date.
    hist_max, hist_unit = {}, {}
    for r in h["rows"]:
        if r["typed_value"] >= hist_max.get(r["date"], -1.0):
            hist_max[r["date"]] = r["typed_value"]
        hist_unit[r["date"]] = r["unit"]
    overlap = [x for x in se["sessions"] if x["date"] in hist_max]
    assert overlap
    for x in overlap:
        assert x["unit"] == hist_unit[x["date"]], x["date"]
        assert x["max_weight"] == hist_max[x["date"]], x["date"]

    ex = _pkg_ex("Deadlift")
    assert ex["unit"] == "kg" and ex["bar_weight"] > 0


# ── non-bar exercise: unchanged (plates only, no bar) ────────────────────────

def test_non_bar_exercise_unchanged():
    """Lat Pulldown has no bar and is not kg-native → weight is plates only
    (metric_weight * 2.2046), unit lbs, no bar added."""
    name = "Lat Pulldown"
    h = _history(name)
    assert h.get("rows"), "expected Lat Pulldown history"
    assert all(r["unit"] == "lbs" for r in h["rows"])

    ctx = load_user_context()
    for r in _raw_rows(name)[:10]:
        w, unit, plates = _bar_inclusive_weight(ctx, name, r["date"], r["metric_weight"])
        # no bar, no offset → headline == plates == round(mw * 2.2046, 1)
        assert unit == "lbs"
        assert w == plates == round(r["metric_weight"] * 2.2046, 1)


# ── offset exercise: offset applied once, not double-counted ─────────────────

def test_offset_applied_once_not_doubled():
    name = "Machine Wrist Extension"      # numeric_offset = 5 (kg-native, no bar)
    ctx = load_user_context()
    offset = next((q.get("numeric_offset", 0) for q in ctx.get("exercise_quirks", [])
                   if q.get("exercise_name") == name), 0)
    assert offset, "test assumes this exercise carries a numeric_offset"

    rows = _raw_rows(name)
    assert rows
    r = rows[0]
    w, unit, plates = _bar_inclusive_weight(ctx, name, r["date"], r["metric_weight"])
    # plates = mw*2.2046 + offset (ONCE); no bar for this exercise → w == plates
    expected_plates = round(r["metric_weight"] * 2.2046 + offset, 1)
    assert plates == expected_plates
    assert w == plates                                    # no bar, not doubled
    # and it lines up with the analytical package PR unit
    assert unit == "kg" and _pkg_ex(name)["unit"] == "kg"


# ── shapes preserved + one source of truth ───────────────────────────────────

def test_output_shapes_preserved():
    h = _history("Barbell Curl")
    assert {"exercise", "days", "note", "rows"} <= set(h)
    assert {"date", "typed_value", "reps", "unit"} <= set(h["rows"][0])

    se = _sessions("Barbell Curl", limit=3)
    assert {"exercise", "mode", "sessions", "count"} <= set(se)
    sess0 = se["sessions"][0]
    assert {"date", "unit", "max_weight", "total_sets", "display_sets"} <= set(sess0)
    # bar exercise carries the (now bar-inclusive) note, no longer "add the bar"
    assert "BAR-INCLUSIVE" in se["bar_weight_note"]


def test_shared_helper_matches_manual_composition():
    """_bar_inclusive_weight is the single conversion; verify it equals the
    analytical primitives composed by hand (no divergent copy)."""
    from src.data_agent.process import (
        _get_bar_weight_lbs, _get_numeric_offset, _is_kg_native, _recover_typed_weight,
    )
    ctx = load_user_context()
    for name in ("Barbell Curl", "Deadlift", "Lat Pulldown", "Machine Wrist Extension"):
        for r in _raw_rows(name)[:5]:
            d, mw = r["date"], r["metric_weight"]
            w, unit, plates = _bar_inclusive_weight(ctx, name, d, mw)
            exp_plates = _recover_typed_weight(mw, _get_numeric_offset(ctx, name))
            is_kg = _is_kg_native(ctx, name, d)
            bar_lbs = _get_bar_weight_lbs(ctx, name, d)
            bar = bar_lbs / 2.2046 if is_kg else bar_lbs
            assert plates == exp_plates
            assert unit == ("kg" if is_kg else "lbs")
            assert w == round(exp_plates + bar, 1)
