"""
Step B routing: muscle-group Category guard + plates-only volume steering.

Fix 1 — a term that names a muscle-group Category (Triceps, Chest, Back, …) is a
MUSCLE GROUP, not an exercise: it routes to muscle_groups (GROUP scope) and
NEVER hits resolve_exercise_name / the disambiguation prompt. A genuine
exercise name ("dumbbell bench press") still disambiguates among its variants.

Fix 2 — the analytical package's plates-only raw volume is demoted under
all_time_summary._raw_volume_crosscheck (was top-level total_volume_raw_typed_*),
and _ANALYSIS_SYSTEM steers the model to the bar-inclusive muscle_group_summary
volume fields.

No Gemini — the classifier LLM is not called; the package builder, resolver, and
downstream analysis are stubbed where needed.
"""

import asyncio
import json
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src import coordinator as coordinator_mod          # noqa: E402
from src.coordinator import Coordinator                 # noqa: E402
from src.data_agent import match_muscle_group, MUSCLE_GROUP_NAMES  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# match_muscle_group — the canonical category matcher (pure)
# ══════════════════════════════════════════════════════════════════════════════

def test_muscle_group_names_are_the_nine_categories():
    assert MUSCLE_GROUP_NAMES == frozenset({
        "Shoulders", "Triceps", "Biceps", "Chest", "Back",
        "Legs", "Abs", "Cardio", "Forearms",
    })
    # Time / Place / Neck (non-muscle, ids 10/11/12) are excluded
    assert "Neck" not in MUSCLE_GROUP_NAMES


@pytest.mark.parametrize("term,expected", [
    ("triceps", "Triceps"), ("Triceps", "Triceps"), ("tricep", "Triceps"),
    ("chest", "Chest"), ("CHEST", "Chest"),
    ("back", "Back"), ("Back", "Back"), ("backs", "Back"),
    ("legs", "Legs"), ("leg", "Legs"),
    ("abs", "Abs"), ("ab", "Abs"),
    ("shoulders", "Shoulders"), ("forearms", "Forearms"),
    ("cardio", "Cardio"), ("biceps", "Biceps"),
    ("  Triceps  ", "Triceps"),
])
def test_match_muscle_group_hits(term, expected):
    assert match_muscle_group(term) == expected


@pytest.mark.parametrize("term", [
    "dumbbell bench press", "bench", "squat", "deadlift",
    "lat pulldown", "neck", "time", "place", "", None,
])
def test_match_muscle_group_misses(term):
    assert match_muscle_group(term) is None


# ══════════════════════════════════════════════════════════════════════════════
# Fix 1 — Coordinator Category guard in _run_analytical
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def coord(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    return Coordinator(agent_session=None)


class _StopAfterPkg(Exception):
    """Raised by the package stub to halt _run_analytical right after the guard."""


def _spy_resolver(monkeypatch, *, match=None, candidates=None):
    """Patch resolve_exercise_name at its source module; record every call."""
    import src.shared.resolver as resolver_mod
    calls: list = []

    def fake(name, db_path, permissive=False):
        calls.append(name)
        return {"match": match, "candidates": candidates or []}

    monkeypatch.setattr(resolver_mod, "resolve_exercise_name", fake)
    return calls


def _capture_pkg(monkeypatch):
    """Patch prepare_analysis_package to capture kwargs then stop the pipeline."""
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        raise _StopAfterPkg()

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package", fake)
    return captured


def _params(exercise_names=None, muscle_groups=None):
    return {
        "route": "analytical",
        "exercise_names": exercise_names,
        "muscle_groups": muscle_groups,
        "query_period_days": 90,
        "needs_custom_sql": False,
        "custom_sql_intent": None,
    }


def test_category_term_in_exercise_names_routes_to_group_no_resolve(coord, monkeypatch):
    # Live trace: classifier mis-slotted "triceps" into exercise_names.
    resolved = _spy_resolver(monkeypatch)
    captured = _capture_pkg(monkeypatch)

    with pytest.raises(_StopAfterPkg):
        asyncio.run(coord._run_analytical(
            "my strength drops when I train triceps after chest",
            _params(exercise_names=["triceps", "chest"]),
        ))

    # Category terms are NEVER resolved / disambiguated
    assert resolved == []
    # They become canonical muscle_groups → GROUP scope; exercise_names cleared
    assert captured["muscle_groups"] == ["Triceps", "Chest"]
    assert captured["exercise_names"] is None


def test_category_term_in_muscle_groups_is_canonicalized(coord, monkeypatch):
    resolved = _spy_resolver(monkeypatch)
    captured = _capture_pkg(monkeypatch)

    with pytest.raises(_StopAfterPkg):
        asyncio.run(coord._run_analytical(
            "how is my back volume",
            _params(muscle_groups=["back"]),   # lowercase from classifier
        ))

    assert resolved == []
    assert captured["muscle_groups"] == ["Back"]   # canonical form for cat_map
    assert captured["exercise_names"] is None


def test_real_exercise_still_disambiguates(coord, monkeypatch):
    # "dumbbell bench press" is NOT a category → resolver runs → 3 variants →
    # disambiguation prompt, and the package is never built.
    resolved = _spy_resolver(monkeypatch, candidates=[
        "Flat Dumbbell Bench Press",
        "Incline Dumbbell Bench Press",
        "Decline Dumbbell Bench Press",
    ])
    pkg_built = {"n": 0}
    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: pkg_built.__setitem__("n", pkg_built["n"] + 1) or {})

    answer, flagged = asyncio.run(coord._run_analytical(
        "how is my dumbbell bench press progressing",
        _params(exercise_names=["dumbbell bench press"]),
    ))

    assert resolved == ["dumbbell bench press"]    # resolver WAS called
    assert "dumbbell bench press" in answer.lower()
    assert "which one did you mean" in answer.lower()
    assert pkg_built["n"] == 0                      # no package built on disambiguation


def test_mixed_terms_split_correctly(coord, monkeypatch):
    # One category + one real exercise: category → group, exercise → resolve.
    resolved = _spy_resolver(monkeypatch, match="Lat Pulldown")
    captured = _capture_pkg(monkeypatch)

    with pytest.raises(_StopAfterPkg):
        asyncio.run(coord._run_analytical(
            "compare my triceps to my lat pulldown",
            _params(exercise_names=["triceps", "lat pulldown"]),
        ))

    assert resolved == ["lat pulldown"]            # only the non-category name
    assert captured["muscle_groups"] == ["Triceps"]
    assert captured["exercise_names"] == ["Lat Pulldown"]


# ══════════════════════════════════════════════════════════════════════════════
# Fix 2 — plates-only volume steering + key demotion
# ══════════════════════════════════════════════════════════════════════════════

def test_analysis_prompt_has_volume_steering():
    from src.analysis_agent import _ANALYSIS_SYSTEM
    assert "VOLUME RULES" in _ANALYSIS_SYSTEM
    # authoritative bar-inclusive fields
    assert "muscle_group_summary.total_volume_lbs" in _ANALYSIS_SYSTEM
    assert "total_volume_kg" in _ANALYSIS_SYSTEM
    # forbid the plates-only footgun
    assert "NEVER quote _raw_volume_crosscheck" in _ANALYSIS_SYSTEM
    # per-unit phrasing rule
    assert "pounds-frame" in _ANALYSIS_SYSTEM and "kilograms-frame" in _ANALYSIS_SYSTEM


def test_raw_volume_demoted_under_crosscheck_value_unchanged():
    from src.data_agent import collect
    ats = collect(query_period_days=None)["all_time_summary"]

    # Old top-level keys are gone
    assert "total_volume_raw_typed_lbs" not in ats
    assert "total_volume_raw_typed_kg" not in ats
    assert "total_volume_raw_note" not in ats

    # Structural: new nested location + note text
    cc = ats["_raw_volume_crosscheck"]
    assert "NOT the user's volume" in cc["note"]

    # RECOMPUTE-AND-RELATE: independently re-sum the typed plates-only volume per
    # unit frame via raw SQL. The kg-native list is read straight from
    # user_context.json (not process.py) and the category exclusion (10/11/12) is
    # reproduced here — so this path shares no logic with the code under test.
    # Production stores round(vol, 0); allow abs=1.0 for that rounding.
    kg_names = json.load(open(os.environ["USER_CONTEXT_PATH"])) \
        ["unit_overrides"]["exercises_in_kg"]
    quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in kg_names)
    kg_pred = (f"(e.name IN ({quoted}) "
               f"AND NOT (e.name = 'Deadlift' AND tl.date < '2025-12-26'))")

    conn = sqlite3.connect(f"file:{os.environ['FITNOTES_DB_PATH']}?mode=ro", uri=True)
    try:
        indep_lbs = conn.execute(
            "SELECT SUM(tl.metric_weight * 2.2046 * tl.reps) v FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id = e._id "
            f"WHERE e.category_id NOT IN (10, 11, 12) AND NOT {kg_pred}"
        ).fetchone()[0]
        indep_kg = conn.execute(
            "SELECT SUM(tl.metric_weight * 2.2046 * tl.reps) v FROM training_log tl "
            "JOIN exercise e ON tl.exercise_id = e._id "
            f"WHERE e.category_id NOT IN (10, 11, 12) AND {kg_pred}"
        ).fetchone()[0]
    finally:
        conn.close()

    assert cc["typed_lbs"] == pytest.approx(indep_lbs, abs=1.0)
    assert cc["typed_kg"] == pytest.approx(indep_kg, abs=1.0)
