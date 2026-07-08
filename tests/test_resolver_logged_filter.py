"""
Resolver defects 2.3 + 2.4 (src/shared/resolver.py), READ path only.

2.3 — the ≥2-real-data ASK offered the FULL candidate list, 0-set names
included (live: "squat" offered Barbell Squat + Barbell Front Squat, both zero
logged sets). Fix: the ask list is filtered to candidates with
>= _LOGGED_FLOOR logged sets; the 0-real-data name-only fallback keeps the
full list unchanged. Logged-count is a FILTER, never a ranking key.

2.4 — Tier 3's multi-candidate word-match short-circuited before any stronger
name evidence ran ("dumbbell flat bench press" asked instead of resolving).
Fix (user-locked): word-PERMUTATION rescue — exactly one candidate whose word
multiset equals the query's (plural-flip tolerant) is returned as the match.
Categorical, no tuned threshold (difflib >=0.75 is NOT single here: Incline/
Decline Dumbbell Bench Press score 0.766 next to Flat's 0.818).

WRITE path (permissive=False, default) is byte-unchanged — every new branch
lives inside `if permissive:`. Synthetic temp-DB tests + read-only live
spot-checks pinned to structural facts (membership), never to set counts.
"""

import os
import sqlite3
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.shared.resolver import (                    # noqa: E402
    resolve_exercise_name, _LOGGED_FLOOR, _permutation_match,
)

_LIVE_DB = os.environ["FITNOTES_DB_PATH"]


# ── Synthetic fixture ─────────────────────────────────────────────────────────

def _make_db(path, exercises):
    """exercises: list of (name, logged_set_count)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE exercise (_id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER);
        CREATE TABLE training_log (
            _id INTEGER PRIMARY KEY, exercise_id INTEGER, date DATE,
            metric_weight REAL, reps INTEGER
        );
        """
    )
    for i, (name, sets) in enumerate(exercises, start=1):
        conn.execute("INSERT INTO exercise VALUES (?, ?, 1)", (i, name))
        for _ in range(sets):
            conn.execute(
                "INSERT INTO training_log (exercise_id, date, metric_weight, reps) "
                "VALUES (?, '2026-01-01', 45.0, 5)", (i,))
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def squat_db(tmp_path):
    # Floor boundary by construction: 5 == _LOGGED_FLOOR kept, 4 dropped.
    assert _LOGGED_FLOOR == 5
    return _make_db(str(tmp_path / "s.db"), [
        ("Alpha Squat", 10),
        ("Beta Squat", 5),      # == floor → kept
        ("Gamma Squat", 0),     # 0 sets   → dropped from ask
        ("Delta Squat", 4),     # < floor  → dropped from ask
    ])


@pytest.fixture
def bench_db(tmp_path):
    # Tier 2 empty for "dumbbell flat bench press" (no name contains that
    # ordered phrase); Tier 3 multi-candidate; all logged (real-data ask
    # pre-fix). The permutation rescue must pick Flat.
    return _make_db(str(tmp_path / "b.db"), [
        ("Flat Dumbbell Bench Press", 30),
        ("Incline Dumbbell Bench Press", 30),
        ("Decline Dumbbell Bench Press", 30),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 2.3 — ask list filtered to real-data candidates (READ path)
# ══════════════════════════════════════════════════════════════════════════════

def test_ask_list_offers_only_logged_candidates(squat_db):
    out = resolve_exercise_name("squat", squat_db, permissive=True)
    assert out["match"] is None                        # ≥2 real-data → ask
    assert out["candidates"] == ["Alpha Squat", "Beta Squat"]   # data-first
    # Negative direction: no unlogged/sparse name may reach the ask.
    assert "Gamma Squat" not in out["candidates"]      # 0 sets
    assert "Delta Squat" not in out["candidates"]      # 4 < floor


def test_sole_real_data_candidate_auto_picked(tmp_path):
    db = _make_db(str(tmp_path / "sole.db"), [
        ("Alpha Squat", 10), ("Gamma Squat", 0), ("Delta Squat", 4)])
    out = resolve_exercise_name("squat", db, permissive=True)
    assert out["match"] == "Alpha Squat"
    assert out["candidates"] == []


def test_all_zero_data_keeps_name_only_fallback_full_list(tmp_path):
    # 0 real-data candidates → name-only fallback path byte-unchanged: near-equal
    # ratios → still ask, and the FULL list is offered (no data filter applies).
    db = _make_db(str(tmp_path / "zero.db"), [
        ("Lying Leg Curl Machine", 0), ("Seated Leg Curl Machine", 0)])
    out = resolve_exercise_name("leg curl", db, permissive=True)
    assert out["match"] is None
    assert sorted(out["candidates"]) == [
        "Lying Leg Curl Machine", "Seated Leg Curl Machine"]


# ══════════════════════════════════════════════════════════════════════════════
# 2.4 — word-permutation rescue in Tier 3 (READ path)
# ══════════════════════════════════════════════════════════════════════════════

def test_permutation_rescue_resolves_single_reordered_name(bench_db):
    # Pre-fix: Tier-3 multi-candidate ask (3 real-data names). Post-fix: the
    # query is a word permutation of exactly ONE name → resolved, no ask.
    out = resolve_exercise_name("dumbbell flat bench press", bench_db,
                                permissive=True)
    assert out["match"] == "Flat Dumbbell Bench Press"
    assert out["candidates"] == []


def test_permutation_rescue_plural_flip_tolerant(tmp_path):
    db = _make_db(str(tmp_path / "p.db"), [
        ("Sumo Squats", 20), ("Dumbbell Squats", 20)])
    # "squats sumo" → permutation of "Sumo Squats" (exact); also "squat sumo"
    # reaches it via the plural flip variant.
    out = resolve_exercise_name("squat sumo", db, permissive=True)
    assert out["match"] == "Sumo Squats"


def test_permutation_ambiguous_two_matches_still_asks():
    # Direct unit pin: TWO candidates share the query's word multiset → None.
    assert _permutation_match(
        "press bench", ["Bench Press", "Press Bench"]) is None
    assert _permutation_match(
        "press bench", ["Bench Press", "Shoulder Press"]) == "Bench Press"


def test_permutation_rescue_not_on_write_path(bench_db):
    # Strict path byte-unchanged: the same query still multi-candidate asks.
    out = resolve_exercise_name("dumbbell flat bench press", bench_db)
    assert out["match"] is None
    assert len(out["candidates"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# WRITE path byte-unchanged (strict default)
# ══════════════════════════════════════════════════════════════════════════════

def test_write_path_offers_full_list_including_unlogged(squat_db):
    out = resolve_exercise_name("squat", squat_db)     # permissive=False
    assert out["match"] is None
    assert sorted(out["candidates"]) == [
        "Alpha Squat", "Beta Squat", "Delta Squat", "Gamma Squat"]


def test_write_path_zero_set_exercise_still_resolves_exactly(squat_db):
    # First-time logging: an exact name with 0 sets resolves on the write path.
    out = resolve_exercise_name("Gamma Squat", squat_db)
    assert out["match"] == "Gamma Squat"


# ══════════════════════════════════════════════════════════════════════════════
# Live spot-checks (read-only; structural membership, never set counts)
# ══════════════════════════════════════════════════════════════════════════════

def test_live_squat_ask_offers_only_logged_squats():
    # DB-confirmed: Sumo Squats (121), Dumbbell Squats (58), Smith Machine
    # Squats (29) are logged; Barbell Squat / Barbell Front Squat have ZERO
    # sets — the live-failure pair must never be offered on the read path.
    out = resolve_exercise_name("squat", _LIVE_DB, permissive=True)
    assert out["match"] is None
    assert set(out["candidates"]) == {
        "Sumo Squats", "Dumbbell Squats", "Smith Machine Squats"}
    assert "Barbell Squat" not in out["candidates"]
    assert "Barbell Front Squat" not in out["candidates"]


def test_live_dumbbell_flat_bench_press_resolves():
    out = resolve_exercise_name("dumbbell flat bench press", _LIVE_DB,
                                permissive=True)
    assert out["match"] == "Flat Dumbbell Bench Press"
    assert out["candidates"] == []
