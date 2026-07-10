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


# T1 (test_single_lat_pulldown_recent) and T2 (test_category_back_most_recent) —
# DELETED (unsound live-DB goldens: most-recent date drifted 2026-06-13/14 ->
# 2026-06-27 when stress-test rows landed). Most-recent date resolution
# (single-exercise latest; category MAX-across-group) + flat header format are
# covered by construction in tests/test_session_display_synthetic.py; display-string
# formatting parity is covered by test_port_equality_vs_operational below (T3) and
# the bar conversion by test_category_mode_shares_bar_conversion (T4).


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
    # 2026-03-30 is the only real Barbell Row session (the previous pin's
    # 2026-06-14 rows were test pollution, removed in the Session-13 cleanup).
    ctx = load_user_context()
    rows = _raw_rows("Barbell Row", "2026-03-30")
    expected = [_bar_inclusive_weight(ctx, "Barbell Row", "2026-03-30", r["metric_weight"])[0]
                for r in rows]
    assert expected == [44.1, 44.1, 54.1, 44.1]  # cross-check the literal too

    out = get_category_session("Back", "2026-03-30")
    ex = next(e for e in out["exercises"] if e["exercise"] == "Barbell Row")
    got = [float(s.split(" lbs ")[0].split(": ")[1]) for s in ex["display_sets"]]
    assert got == expected
