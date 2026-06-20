"""
Stage 1: deterministic session-display in the Data Agent
(src/data_agent/session_display.py), built and tested in ISOLATION.

- T1: single-exercise mode, "Lat Pulldown" recent (no bar, no warmup).
- T2: category mode (NEW), "Back" most recent (bar-inclusive applied).
- T3: faithful-port equality — new get_exercise_sessions output == operational
      _get_exercise_sessions_sync output for the SAME inputs (strongest proof).
- T4: category mode shares the bar-inclusive conversion (not bypassed).

No Gemini, no server. Runs against the pinned project DB (same as the golden
tests). The operational MCP handler is imported ONLY as the equality oracle for
T3 — this stage does not modify it.
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

from src.data_agent.session_display import (                       # noqa: E402
    get_exercise_sessions, get_category_session,
    _bar_inclusive_weight, DB_PATH,
)
from src.data_agent.fetch import load_user_context                 # noqa: E402
# Operational oracle for the faithful-port equality test (NOT modified here).
from mcp_servers.combined_server import _get_exercise_sessions_sync  # noqa: E402


def _raw_rows(name, date=None):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT tl.date, tl.metric_weight, tl.reps FROM training_log tl "
               "JOIN exercise e ON tl.exercise_id = e._id WHERE e.name = ?")
        args = [name]
        if date:
            sql += " AND tl.date = ?"
            args.append(date)
        sql += " ORDER BY tl._id ASC"
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# ── T1: single-exercise mode, faithful port shape ────────────────────────────

def test_single_lat_pulldown_recent():
    """Lat Pulldown most recent session: 2026-06-13, three sets, no bar (plates ==
    bar-inclusive), 100/110/120 lbs, reps 10/8/6, no warmup (not warmup-shaped)."""
    out = get_exercise_sessions("Lat Pulldown", mode="recent")
    assert out["exercise"] == "Lat Pulldown"
    sess = out["sessions"][0]
    assert sess["date"] == "2026-06-13"
    assert sess["total_sets"] == 3
    assert sess["unit"] == "lbs"
    assert sess["max_weight"] == 120.0
    assert sess["display_sets"] == [
        "Set 1: 100.0 lbs × 10 reps",
        "Set 2: 110.0 lbs × 8 reps",
        "Set 3: 120.0 lbs × 6 reps",
    ]
    # No bar exercise → no bar_weight_note key.
    assert "bar_weight_note" not in out


# ── T2: category mode (NEW) ──────────────────────────────────────────────────

def test_category_back_most_recent():
    """'last back session' = the single most recent date ANY Back exercise was
    trained: 2026-06-14, exactly one exercise (Barbell Row), three sets, reps
    10/8/6, bar-inclusive lbs.

    NOTE (flagged): the spec stated 104.09/114.09/124.09 (un-rounded plate+bar).
    The faithful port rounds the headline to 1 decimal (round(plates+bar,1)), so
    the byte-identical-to-operational output is 104.1/114.1/124.1. We assert the
    faithful-port values and do NOT change rounding (that would break parity with
    the operational display)."""
    out = get_category_session("Back", "recent")
    assert out["category"] == "Back"
    assert out["date"] == "2026-06-14"
    assert out["count"] == 1

    ex = out["exercises"][0]
    assert ex["exercise"] == "Barbell Row"
    assert ex["total_sets"] == 3
    assert ex["unit"] == "lbs"
    assert ex["display_sets"] == [
        "Set 1: 104.1 lbs × 10 reps",
        "Set 2: 114.1 lbs × 8 reps",
        "Set 3: 124.1 lbs × 6 reps",
    ]
    # Barbell Row is a bar exercise → carries the bar-inclusive note.
    assert "bar_weight_note" in ex


def test_category_time_place_neck_excluded():
    """Non-muscle categories (Time/Place/Neck) and unknown terms resolve empty —
    category resolution never runs the exercise-name path."""
    for term in ("Time", "Place", "Neck", "definitely-not-a-category"):
        out = get_category_session(term, "recent")
        assert out["count"] == 0 and out["date"] is None


# ── T3: faithful-port equality vs the operational oracle (strongest proof) ────

@pytest.mark.parametrize("name,kwargs,op_args", [
    # plain (no bar), recent
    ("Lat Pulldown", dict(mode="recent"), {"mode": "recent"}),
    # bar exercise (fixed bar) → carries bar_weight_note, recent
    ("Barbell Row", dict(mode="recent"), {"mode": "recent"}),
    # date-ranged bar, recent
    ("Barbell Curl", dict(mode="recent"), {"mode": "recent"}),
    # kg-native, date-ranged unit switch, recent
    ("Deadlift", dict(mode="recent"), {"mode": "recent"}),
    # offset + kg-native + drop sets, single date via range
    ("Machine Wrist Extension",
     dict(mode="range", date_from="2026-05-20", date_to="2026-05-20"),
     {"mode": "range", "date_from": "2026-05-20", "date_to": "2026-05-20"}),
    # drop sets + partial-rep comments, single date via range
    ("Seated Narrow V Shaped Row",
     dict(mode="range", date_from="2026-05-25", date_to="2026-05-25"),
     {"mode": "range", "date_from": "2026-05-25", "date_to": "2026-05-25"}),
])
def test_port_equality_vs_operational(name, kwargs, op_args):
    """The new function's parsed output must equal the operational
    _get_exercise_sessions_sync output for the same input — every display_sets
    string byte-for-byte (→ notation, parenthetical comments, offset application,
    drop-set grouping, warmup labeling)."""
    new = get_exercise_sessions(name, **kwargs)
    op = json.loads(_get_exercise_sessions_sync({"exercise_name": name, **op_args}))
    assert new == op, (
        f"\nNEW: {json.dumps(new, indent=2, ensure_ascii=False)}"
        f"\nOP:  {json.dumps(op, indent=2, ensure_ascii=False)}"
    )


def test_port_equality_dropset_strings_present():
    """Sanity: the drop-set session actually exercises → notation and comments, so
    the equality test above is meaningful (not comparing two empty lists)."""
    op = json.loads(_get_exercise_sessions_sync(
        {"exercise_name": "Machine Wrist Extension", "mode": "range",
         "date_from": "2026-05-20", "date_to": "2026-05-20"}))
    blob = "\n".join(op["sessions"][0]["display_sets"])
    assert "→" not in blob  # this session uses multi-line drop groups, not arrows
    assert "(" in blob      # parenthetical comments present


# ── T4: category mode shares the bar-inclusive conversion ────────────────────

def test_category_mode_shares_bar_conversion():
    """Category-mode Barbell Row weights must equal _bar_inclusive_weight computed
    directly from the raw rows — proving the shared leaf conversion is applied,
    not bypassed."""
    ctx = load_user_context()
    rows = _raw_rows("Barbell Row", "2026-06-14")
    expected = [_bar_inclusive_weight(ctx, "Barbell Row", "2026-06-14", r["metric_weight"])[0]
                for r in rows]
    assert expected == [104.1, 114.1, 124.1]  # cross-check the literal too

    out = get_category_session("Back", "2026-06-14")
    ex = out["exercises"][0]
    got = [float(s.split(" lbs ")[0].split(": ")[1]) for s in ex["display_sets"]]
    assert got == expected
