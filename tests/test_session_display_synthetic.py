"""
Synthetic by-construction replacements for the deleted live-DB session-display
goldens (test_session_display::test_single_lat_pulldown_recent / _category_back_most_recent
and test_display_sets_approach_b::test_build_single_exercise_no_prior_when_rich /
_build_single_category_prior_included_when_thin / _package_has_display_sets_for_single_exercise).
Those pinned the most-recent date to the live DB, which drifted when 2026-06-27 stress-test rows
landed.

These cover the genuine SOLE-coverage logic — most-recent date resolution (single-exercise
latest; category MAX-across-group), the flat header format, and the <=1 prior-session append —
on a HAND-BUILT temp DB where the correct answer is known by construction (no shared logic with
the SQL ORDER BY date DESC / MAX(date) under test). Display-string formatting parity is covered
separately by test_session_display::test_port_equality_vs_operational (a differential new-vs-
operational test, not date-pinned) and the <=1 trigger predicates by the pure-predicate tests in
test_display_sets_approach_b. A no-bar exercise is used so headline == plate weight (the
bar-inclusive math is not under test here).
"""

import os
import sqlite3
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.data_agent.session_display as sd          # noqa: E402

# Cardio category id (8) is excluded; use ordinary strength categories.
_BACK = 5


def _make_db(path):
    """Temp FitNotes-shaped DB: exercise + training_log + Comment. Metric columns
    mirror the real NOT-NULL-DEFAULT-0 shape. metric_weight is the kg-stored value;
    a no-bar/no-offset exercise recovers headline = round(metric_weight*2.2046, 1)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY, exercise_id INTEGER, date DATE,
            metric_weight REAL, reps INTEGER,
            unit INTEGER NOT NULL DEFAULT 0, is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0, duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE Comment (_id INTEGER PRIMARY KEY, date DATE, owner_type_id INTEGER,
                              owner_id INTEGER, comment TEXT);
        """
    )
    conn.commit()
    conn.close()


def _kg(lbs):
    """metric_weight (kg-stored) that recovers to a clean lbs headline (no bar/offset)."""
    return lbs / 2.2046


def _ins(conn, ex_id, date, lbs, reps):
    conn.execute(
        "INSERT INTO training_log (exercise_id, date, metric_weight, reps) VALUES (?,?,?,?)",
        (ex_id, date, _kg(lbs), reps),
    )


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = str(tmp_path / "disp.fitnotes")
    _make_db(p)
    # session_display reads its module-global DB_PATH for every query.
    monkeypatch.setattr(sd, "DB_PATH", p)
    return p


# ── single-exercise most-recent resolution + flatten format ───────────────────

def test_single_exercise_most_recent_and_flatten(db):
    # Two sessions; the LATER date (2026-03-02) is most-recent BY CONSTRUCTION.
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO exercise (_id, name, category_id) VALUES (1, 'SynRow', ?)", (_BACK,))
    _ins(conn, 1, "2026-02-01", 100.0, 5)                       # older session
    for lbs, reps in [(100.0, 10), (110.0, 8), (120.0, 6)]:     # most-recent, 3 sets
        _ins(conn, 1, "2026-03-02", lbs, reps)
    conn.commit(); conn.close()

    out = sd.get_exercise_sessions("SynRow", mode="recent")
    assert out["sessions"][0]["date"] == "2026-03-02"          # latest, not 2026-02-01
    assert out["sessions"][0]["total_sets"] == 3

    flat = sd.build_display_sets("exercise", "SynRow")
    # Rich most-recent (3 sets) -> <=1 trigger does NOT fire -> one date block.
    assert flat[0] == "2026-03-02 — SynRow (3 sets):"
    assert flat[1:4] == [
        "Set 1: 100.0 lbs × 10 reps",
        "Set 2: 110.0 lbs × 8 reps",
        "Set 3: 120.0 lbs × 6 reps",
    ]
    assert sum(1 for s in flat if " — SynRow (" in s) == 1     # no prior block appended


# ── category most-recent = MAX date across the group ──────────────────────────

def test_category_most_recent_is_max_across_group(db):
    # Two Back exercises trained on different dates; "last Back session" is the
    # MAX date across the group (2026-04-10), with only that day's exercise.
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO exercise (_id, name, category_id) VALUES (1, 'SynRowA', ?)", (_BACK,))
    conn.execute("INSERT INTO exercise (_id, name, category_id) VALUES (2, 'SynRowB', ?)", (_BACK,))
    for lbs, reps in [(100.0, 10), (110.0, 8)]:
        _ins(conn, 1, "2026-04-01", lbs, reps)                 # SynRowA earlier
    for lbs, reps in [(50.0, 12), (60.0, 10), (70.0, 8)]:
        _ins(conn, 2, "2026-04-10", lbs, reps)                 # SynRowB latest
    conn.commit(); conn.close()

    out = sd.get_category_session("Back", "recent")
    assert out["date"] == "2026-04-10"                          # MAX across the group
    assert out["count"] == 1
    assert out["exercises"][0]["exercise"] == "SynRowB"        # only the latest-day exercise
    assert out["exercises"][0]["total_sets"] == 3


# ── <=1 prior-session append (thin most-recent -> prior date included) ─────────

def test_prior_session_appended_when_most_recent_thin(db):
    # Most-recent date has a SINGLE set -> the <=1 SETS-ONLY trigger fires -> the
    # prior date's block is appended. Two date headers BY CONSTRUCTION.
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO exercise (_id, name, category_id) VALUES (1, 'SynThin', ?)", (_BACK,))
    for lbs, reps in [(100.0, 10), (110.0, 8), (120.0, 6)]:
        _ins(conn, 1, "2026-05-01", lbs, reps)                 # prior (rich)
    _ins(conn, 1, "2026-05-08", 130.0, 5)                       # most-recent (thin: 1 set)
    conn.commit(); conn.close()

    flat = sd.build_display_sets("exercise", "SynThin")
    headers = [s for s in flat if " — SynThin (" in s]
    assert len(headers) == 2                                    # most-recent + appended prior
    assert flat[0] == "2026-05-08 — SynThin (1 sets):"         # thin most-recent first
    assert "2026-05-01 — SynThin (3 sets):" in headers         # prior appended
