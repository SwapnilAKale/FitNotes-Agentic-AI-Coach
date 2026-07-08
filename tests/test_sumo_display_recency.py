"""
Sumo Squats live-answer fixes (two independent defects):

DEFECT A — wrong-field "most recent" predicate: the draft attached the
pain-flagged (most salient) session date to a "most recent session" claim.
Root: the prompt gave the model no recency anchor — it had to infer "most
recent" by scanning dates, and the pain date won. Fix is prompt-side (the
package already carries progression.latest_session_date); these tests pin the
package field's correctness and the prompt anchor's presence. Whether the
DRAFT actually obeys the anchor is live-only-verifiable.

DEFECT B — display triplication: (1)+(2) the documented exercise+parent-
category package duplication, now deduped structurally at flatten time
(build_all_display_sets — category copy kept); (3) the LLM-side re-render,
discouraged by an exactly-once [DISPLAY] instruction (prompt-presence pinned;
the containment DISPLAY SETS CHECK cannot count occurrences — known
limitation, deliberately not "fixed").

No Gemini, no server, no live-DB dependence (dedup tests monkeypatch the
session_display fetchers).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import session_display as sd            # noqa: E402
from src.data_agent.process import _compute_progression     # noqa: E402
from src.analysis_agent import _ANALYSIS_SYSTEM, _fmt_display  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT A — the package exposes an unambiguous most-recent field
# ══════════════════════════════════════════════════════════════════════════════

def _S(date, w, reps=5):
    e = round(w * (1 + reps / 30), 1)
    return {"date": date, "unit": "lbs", "max_working_weight": float(w),
            "reps_at_max": reps, "estimated_1rm": e}


def test_latest_session_date_is_the_true_last_not_the_pain_salient_date():
    # Live shape: the pain-flagged / last-commented session (06-15) is the
    # SECOND-most-recent; the true latest is 06-25. The labeled recency field
    # must be the true last date, never the salient one.
    pain_date, true_last = "2026-06-15", "2026-06-25"
    sessions = [_S("2026-06-01", 100), _S("2026-06-08", 105),
                _S(pain_date, 110), _S(true_last, 110)]
    p = _compute_progression(sessions)
    assert p["latest_session_date"] == true_last
    assert p["latest_session_date"] != pain_date
    assert p["last_session_date"] == true_last


def test_prompt_carries_recency_anchor():
    # The RECENCY rule must name the anchor field and forbid deriving recency
    # from the salient pain/comment fields. Draft obedience is live-only.
    assert "RECENCY" in _ANALYSIS_SYSTEM
    assert "latest_session_date" in _ANALYSIS_SYSTEM
    assert "pain_analysis" in _ANALYSIS_SYSTEM
    assert "full_comments" in _ANALYSIS_SYSTEM


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT B2 — [DISPLAY] exactly-once instruction present
# ══════════════════════════════════════════════════════════════════════════════

def test_display_prompt_instructs_exactly_once():
    block = _fmt_display({"display_sets": ["Set 1: 100.0 lbs × 5 reps"]})
    assert "EXACTLY ONCE" in block
    assert "never repeat a block" in block
    # No display_sets → no block at all (unchanged).
    assert _fmt_display({}) == ""


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT B1 — exercise-inside-category dedup at the flatten (monkeypatched)
# ══════════════════════════════════════════════════════════════════════════════

_EX_LINES = ["Set 1: 185.0 lbs × 8 reps", "Set 2: 195.0 lbs × 6 reps"]


def _fake_exercise_sessions(name, date="2026-06-25"):
    def fake(exercise_name, mode="recent", **kw):
        assert exercise_name == name
        return {"exercise": name, "mode": mode, "count": 1,
                "sessions": [{"date": date, "unit": "lbs", "max_weight": 195.0,
                              "total_sets": 2, "display_sets": list(_EX_LINES)}]}
    return fake


def _fake_category_session(category, date, exercises):
    """exercises: list of (name, display_sets). Rich blocks (no prior trigger)."""
    def fake(cat, target="recent"):
        assert cat == category
        return {"category": category, "date": date, "count": len(exercises),
                "exercises": [{"exercise": n, "unit": "lbs", "max_weight": 100.0,
                               "total_sets": 2, "display_sets": list(lines)}
                              for n, lines in exercises]}
    return fake


def test_exercise_contained_in_category_emitted_once_category_copy_kept(monkeypatch):
    # Sumo Squats + Legs where the Legs block CONTAINS Sumo Squats: the
    # standalone exercise block is skipped; the category copy is the one kept.
    monkeypatch.setattr(sd, "get_exercise_sessions",
                        _fake_exercise_sessions("Sumo Squats"))
    monkeypatch.setattr(sd, "get_category_session", _fake_category_session(
        "Legs", "2026-06-25",
        [("Sumo Squats", _EX_LINES), ("Leg Press", ["Set 1: 300.0 lbs × 10 reps"])]))

    flat = sd.build_all_display_sets(
        [("exercise", "Sumo Squats"), ("category", "Legs")])

    # The Sumo block appears exactly once (inside the category block).
    assert sum(1 for s in flat if s == "Sumo Squats (2 sets):") == 1
    assert sum(1 for s in flat if s == _EX_LINES[0]) == 1
    # No standalone exercise header ("date — name (N sets):" form).
    assert not any("— Sumo Squats (" in s for s in flat)
    # Category block retained with its date header and the sibling exercise.
    assert "2026-06-25 — Legs:" in flat
    assert "Leg Press (2 sets):" in flat


def test_exercise_not_on_category_date_keeps_both_blocks(monkeypatch):
    # Sumo's own latest (06-20) is NOT the Legs block date (06-25, Sumo absent):
    # no duplication exists, so BOTH blocks must be present.
    monkeypatch.setattr(sd, "get_exercise_sessions",
                        _fake_exercise_sessions("Sumo Squats", date="2026-06-20"))
    monkeypatch.setattr(sd, "get_category_session", _fake_category_session(
        "Legs", "2026-06-25",
        [("Leg Press", ["Set 1: 300.0 lbs × 10 reps"]),
         ("Leg Curl", ["Set 1: 90.0 lbs × 12 reps"])]))

    flat = sd.build_all_display_sets(
        [("exercise", "Sumo Squats"), ("category", "Legs")])

    assert "2026-06-20 — Sumo Squats (2 sets):" in flat     # standalone kept
    assert "2026-06-25 — Legs:" in flat                     # category kept
    # Standalone precedes the category block (historical flatten order).
    assert flat.index("2026-06-20 — Sumo Squats (2 sets):") < \
           flat.index("2026-06-25 — Legs:")


def test_exercise_only_scope_unchanged(monkeypatch):
    monkeypatch.setattr(sd, "get_exercise_sessions",
                        _fake_exercise_sessions("Sumo Squats"))
    flat = sd.build_all_display_sets([("exercise", "Sumo Squats")])
    assert flat == sd.build_display_sets("exercise", "Sumo Squats")
    assert flat[0] == "2026-06-25 — Sumo Squats (2 sets):"


def test_category_only_scope_unchanged(monkeypatch):
    monkeypatch.setattr(sd, "get_category_session", _fake_category_session(
        "Legs", "2026-06-25",
        [("Leg Press", ["Set 1: 300.0 lbs × 10 reps"]),
         ("Leg Curl", ["Set 1: 90.0 lbs × 12 reps"])]))
    flat = sd.build_all_display_sets([("category", "Legs")])
    assert flat == sd.build_display_sets("category", "Legs")
    assert flat[0] == "2026-06-25 — Legs:"


def test_exercise_contained_only_in_prior_category_block_still_deduped(monkeypatch):
    # The category's latest date is thin (1 exercise → prior trigger fires);
    # Sumo appears only in the PRIOR block. Contained-names include the prior
    # block, so the standalone is still skipped — Sumo shows exactly once.
    def fake_cat(cat, target="recent"):
        assert cat == "Legs"
        if target == "recent":
            return {"category": "Legs", "date": "2026-06-25", "count": 1,
                    "exercises": [{"exercise": "Leg Press", "unit": "lbs",
                                   "max_weight": 300.0, "total_sets": 3,
                                   "display_sets": ["Set 1: 300.0 lbs × 10 reps"]}]}
        assert target == "2026-06-22"
        return {"category": "Legs", "date": "2026-06-22", "count": 2,
                "exercises": [{"exercise": "Sumo Squats", "unit": "lbs",
                               "max_weight": 195.0, "total_sets": 2,
                               "display_sets": list(_EX_LINES)},
                              {"exercise": "Leg Curl", "unit": "lbs",
                               "max_weight": 90.0, "total_sets": 2,
                               "display_sets": ["Set 1: 90.0 lbs × 12 reps"]}]}

    monkeypatch.setattr(sd, "get_category_session", fake_cat)
    monkeypatch.setattr(sd, "_prior_category_date", lambda c, d: "2026-06-22")
    monkeypatch.setattr(sd, "get_exercise_sessions",
                        _fake_exercise_sessions("Sumo Squats", date="2026-06-22"))

    flat = sd.build_all_display_sets(
        [("exercise", "Sumo Squats"), ("category", "Legs")])

    assert sum(1 for s in flat if s == "Sumo Squats (2 sets):") == 1
    assert not any("— Sumo Squats (" in s for s in flat)    # no standalone
    assert "2026-06-25 — Legs:" in flat                     # latest block
    assert "2026-06-22 — Legs:" in flat                     # prior block
