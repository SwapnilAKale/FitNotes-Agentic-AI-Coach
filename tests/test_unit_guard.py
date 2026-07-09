"""
Issue 1 (final) — native-unit guard at staging.

The agent can NEVER log a weight in an exercise's non-native unit: an
agent-side unit change (per-record or config-level) either poisons every
aggregate with a mixed frame or contradicts the next uploaded FitNotes
backup. `log_workout` refuses the whole call BEFORE anything is staged and
asks for the weight restated in the native unit, including a tool-computed
≈-conversion so the user can simply confirm the converted number.

No server, no Gemini — temp SQLite + direct sync-tool calls, mirroring
tests/test_write_path_fixes.py fixtures.
"""

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import mcp_servers.combined_server as cs       # noqa: E402


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY,
            exercise_id INTEGER,
            date DATE,
            metric_weight REAL,
            reps INTEGER,
            unit INTEGER NOT NULL DEFAULT 0,
            is_personal_record INTEGER,
            is_complete INTEGER NOT NULL DEFAULT 0,
            distance REAL NOT NULL DEFAULT 0,
            duration_seconds INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO exercise (_id, name, category_id) VALUES
            (1, 'Barbell Row', 5),                  -- lbs-native
            (2, 'Seated Machine Curl (Kg)', 2),     -- kg-native (all history)
            (3, 'Deadlift', 6),                     -- kg-native from 2025-12-26
            (4, 'Treadmill', 8);                    -- cardio
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = str(tmp_path / "w.fitnotes")
    _make_db(p)
    monkeypatch.setattr(cs, "DB_PATH", p)
    cs._staged_writes.clear()
    return p


def _log(exercise, date, sets):
    return json.loads(cs._log_workout_sync(
        {"exercise_name": exercise, "date": date, "sets": sets}))


def _staged():
    return cs._staged_writes.get("workout", [])


# ── 1-2. Both mismatch directions refused, nothing staged ─────────────────────

def test_kg_on_lbs_native_refused(db):
    out = _log("Barbell Row", "2026-07-09", [{"weight": 60, "unit": "kg", "reps": 8}])
    assert out.get("error") is True
    assert out.get("needs_clarification") is True
    assert out.get("unit_mismatch") is True
    assert "logged in lbs, not kg" in out["message"]
    assert "132.3" in out["message"]              # 60 kg ≈ 132.3 lbs, tool-computed
    assert "never be changed" in out["message"]
    assert _staged() == []                        # NOTHING staged


def test_lbs_on_kg_native_refused(db):
    out = _log("Seated Machine Curl (Kg)", "2026-07-09",
               [{"weight": 60, "unit": "lbs", "reps": 8}])
    assert out.get("unit_mismatch") is True
    assert "logged in kg, not lbs" in out["message"]
    assert "27.2" in out["message"]               # 60 lbs ≈ 27.2 kg
    assert _staged() == []


# ── 3. Matched / absent units stage normally (byte-identical negative) ────────

@pytest.mark.parametrize("exercise,unit", [
    ("Barbell Row", "lbs"),
    ("Seated Machine Curl (Kg)", "kg"),
])
def test_matched_units_stage_normally(db, exercise, unit):
    out = _log(exercise, "2026-07-09", [{"weight": 60, "unit": unit, "reps": 8}])
    assert out.get("staged") is True, out
    assert _staged()[0]["sets"][0]["metric_weight"] == 60 / 2.2046   # exact arithmetic


def test_absent_unit_stages_normally(db):
    out = _log("Barbell Row", "2026-07-09", [{"weight": 60, "reps": 8}])
    assert out.get("staged") is True, out
    assert _staged()[0]["sets"][0]["metric_weight"] == 60 / 2.2046


# ── 4. Deadlift era boundary (guard is date-aware) ────────────────────────────

def test_deadlift_kg_era_kg_stages(db):
    out = _log("Deadlift", "2026-07-05", [{"weight": 120, "unit": "kg", "reps": 1}])
    assert out.get("staged") is True, out
    assert _staged()[0]["sets"][0]["metric_weight"] == pytest.approx(120 / 2.2046)


def test_deadlift_lbs_era_kg_refused(db):
    out = _log("Deadlift", "2025-11-01", [{"weight": 60, "unit": "kg", "reps": 5}])
    assert out.get("unit_mismatch") is True
    assert "logged in lbs, not kg" in out["message"]
    assert _staged() == []


def test_deadlift_lbs_era_lbs_stages(db):
    out = _log("Deadlift", "2025-11-01", [{"weight": 200, "unit": "lbs", "reps": 5}])
    assert out.get("staged") is True, out


# ── 5-6. Refusal atomicity ────────────────────────────────────────────────────

def test_refusal_preserves_earlier_staged_exercises(db):
    ok = _log("Barbell Row", "2026-07-09", [{"weight": 100, "unit": "lbs", "reps": 5}])
    assert ok.get("staged") is True
    bad = _log("Seated Machine Curl (Kg)", "2026-07-09",
               [{"weight": 60, "unit": "lbs", "reps": 8}])
    assert bad.get("unit_mismatch") is True
    slot = _staged()
    assert len(slot) == 1                          # earlier exercise survives
    assert slot[0]["exercise_id"] == 1             # and it's the Barbell Row batch


def test_one_mismatched_set_refuses_whole_call(db):
    out = _log("Barbell Row", "2026-07-09", [
        {"weight": 100, "unit": "lbs", "reps": 5},
        {"weight": 60, "unit": "kg", "reps": 8},   # the one bad set
        {"weight": 110, "unit": "lbs", "reps": 3},
    ])
    assert out.get("unit_mismatch") is True
    assert _staged() == []                         # whole call atomic — zero sets


# ── 7. Cardio bypasses the guard entirely ─────────────────────────────────────

def test_cardio_bypasses_guard(db):
    out = _log("Treadmill", "2026-07-09",
               [{"distance": 3.0, "duration_seconds": 1200}])
    assert out.get("staged") is True, out
    s = _staged()[0]["sets"][0]
    assert s["unit"] == 3                          # metric-type code intact
