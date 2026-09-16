"""
Issue 3 — recency wrong-predicate class (structural guarantee).

The "most recent / latest / last session" date is a deterministic fact the
package already computes (progression.latest_session_date). A generative draft
can still bind that predicate to a WRONG date — a salient pain/comment session,
a stale date, or an invented one. src.citations.recency_guard is the pure
deterministic guarantee: any most-recent claim whose date isn't the exercise's
true latest is rewritten in place; every other date mention is untouched.

Live-reproduced context (diagnose_recency.py, Sumo Squats): true latest =
2026-06-25, salient pain date = 2026-06-15. These tests are synthetic and
by-construction (no live-DB golden pinned to a growing DB), with one live-DB
anchor that recency_truth agrees with the package field.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

import pytest                                                    # noqa: E402

from src import citations as C                                  # noqa: E402
from src.data_agent import prepare_analysis_package             # noqa: E402


LATEST  = "2026-06-25"     # the true most-recent Sumo Squats session
SALIENT = "2026-06-15"     # the pain-flagged (last-commented) session — NOT latest


def _pkg_single():
    """One exercise: latest 2026-06-25, an earlier salient session 2026-06-15."""
    return {
        "exercises": [{
            "name": "Sumo Squats",
            "progression": {"latest_session_date": LATEST},
            "sessions": [{"date": SALIENT}, {"date": LATEST}],
            "pain_analysis": {"pain_session_dates": [SALIENT]},
        }],
    }


def _pkg_multi():
    """Two exercises with different latest dates."""
    return {
        "exercises": [
            {"name": "Sumo Squats",
             "progression": {"latest_session_date": LATEST},
             "sessions": [{"date": SALIENT}, {"date": LATEST}]},
            {"name": "Bench Press",
             "progression": {"latest_session_date": "2026-06-20"},
             "sessions": [{"date": "2026-05-01"}, {"date": "2026-06-20"}]},
        ],
    }


def _pkg_out_of_window():
    """The REAL shape of a package with no exercises: the user asked about an
    exercise with nothing logged in the window. training_consistency is computed
    from every training date, so it still carries the user's overall last
    session — which is NOT that exercise's last session."""
    return {"exercises": [], "training_consistency": {"last_session": LATEST}}


def _pkg_cardio():
    """A cardio exercise as a real package holds it: the date is top-level
    `last_session_date`; its progression has no session date. The sessions list
    here deliberately stops earlier, so reading the field and falling back to
    the list give different answers."""
    return {"exercises": [{"name": "Walking", "progression": {},
                           "last_session_date": LATEST,
                           "sessions": [{"date": SALIENT}]}]}


# ══════════════════════════════════════════════════════════════════════════════
# recency_truth
# ══════════════════════════════════════════════════════════════════════════════

def test_truth_per_exercise():
    t = C.recency_truth(_pkg_single())
    assert t == {"per_exercise": {"Sumo Squats": LATEST}}


def test_truth_reads_the_cardio_date_field():
    assert C.recency_truth(_pkg_cardio())["per_exercise"]["Walking"] == LATEST


def test_truth_none_tolerant():
    # No KeyError / no crash on empty, None, or minimal shapes.
    assert C.recency_truth({})   == {"per_exercise": {}}
    assert C.recency_truth(None) == {"per_exercise": {}}
    # Exercise with no progression: falls back to max(session dates).
    t = C.recency_truth({"exercises": [{"name": "X",
                                        "sessions": [{"date": "2026-01-02"},
                                                     {"date": "2026-01-09"}]}]})
    assert t["per_exercise"]["X"] == "2026-01-09"


# ══════════════════════════════════════════════════════════════════════════════
# Corrects — the wrong date is rewritten to the true latest (both X and Z reduce
# to the same prose here; the guard runs on the final stripped answer)
# ══════════════════════════════════════════════════════════════════════════════

def test_corrects_salient_date_bare_predicate():
    ans = f"Your most recent session was on {SALIENT}. Keep it up!"
    out, flags = C.recency_guard(ans, _pkg_single())
    assert LATEST in out and SALIENT not in out
    assert len(flags) == 1
    assert flags[0]["action"]    == "recency_corrected"
    assert flags[0]["original"]  == SALIENT
    assert flags[0]["corrected"] == LATEST
    assert flags[0]["exercise"]  == "Sumo Squats"


def test_corrects_with_exercise_named_in_predicate_multi():
    ans = f"Your most recent Sumo Squats session was on {SALIENT}."
    out, flags = C.recency_guard(ans, _pkg_multi())
    assert out == f"Your most recent Sumo Squats session was on {LATEST}."
    assert flags and flags[0]["exercise"] == "Sumo Squats"


def test_corrects_invented_date_no_salient_gating():
    # The wrong date is in NEITHER the session list NOR the pain set — proves the
    # invariant is "date == latest", not "date is a known salient date".
    ans = "Your latest session was on 2020-01-01, a while back."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert LATEST in out and "2020-01-01" not in out
    assert len(flags) == 1 and flags[0]["original"] == "2020-01-01"


def test_out_of_window_exercise_date_is_left_alone():
    """There is no program-wide correction, and there must not be one. The only
    real package with no exercises is a question about an exercise with nothing
    in the window; its answer quotes THAT exercise's older date, which is right.
    Correcting it to the overall last session would make it wrong (re-check,
    2026-09-16: Close Grip Smith Machine Bench Press 2026-06-13 → 2026-09-10)."""
    ans = ("You haven't done Deadlift in the last 90 days. "
           "Your most recent session was on 2026-03-01.")
    assert C.recency_guard(ans, _pkg_out_of_window()) == (ans, [])


def test_multi_exercise_scopes_to_nearest_named_before():
    ans = ("Bench Press is trending up. Your most recent session was on "
           "2020-01-01, which is worth noting.")
    out, flags = C.recency_guard(ans, _pkg_multi())
    # Scoped to Bench Press (named before), so corrected to ITS latest, not Sumo's.
    assert "2026-06-20" in out and "2020-01-01" not in out
    assert flags[0]["exercise"] == "Bench Press"


# ══════════════════════════════════════════════════════════════════════════════
# Distinct-stays — a correct claim is byte-for-byte untouched
# ══════════════════════════════════════════════════════════════════════════════

def test_correct_date_untouched():
    ans = f"Your most recent session was on {LATEST}. Nice work."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans
    assert flags == []


def test_display_block_header_untouched():
    # The verbatim display header carries the latest date; only the wrong date in
    # the most-recent CLAIM is rewritten, the header stays exactly as-is.
    ans = (f"Your most recent session was on {SALIENT}.\n\n"
           f"{LATEST} — Sumo Squats (4 sets):\nSet 1 (Warmup): 0.0 lbs x 12 reps")
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == (f"Your most recent session was on {LATEST}.\n\n"
                   f"{LATEST} — Sumo Squats (4 sets):\nSet 1 (Warmup): 0.0 lbs x 12 reps")
    assert len(flags) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Negative — the guard must NOT touch these (no over-reach)
# ══════════════════════════════════════════════════════════════════════════════

def test_plain_pain_mention_untouched():
    ans = f"You noted very slight knee pain on {SALIENT}, so monitor your form."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_only_predicate_clause_date_replaced():
    # Same salient date appears twice: once as a false "last session" claim, once
    # as a correct pain mention. Only the claim is rewritten.
    ans = (f"Your last session was on {SALIENT}. Earlier, you reported knee "
           f"pain on {SALIENT}.")
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == (f"Your last session was on {LATEST}. Earlier, you reported "
                   f"knee pain on {SALIENT}.")
    assert len(flags) == 1


def test_qualified_pr_session_untouched():
    ans = f"Your last PR session was on {SALIENT}."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_qualified_heavy_session_untouched():
    ans = f"Your last heavy session was on {SALIENT}."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_unknown_qualifier_not_an_exercise_untouched():
    ans = f"Your most recent leg session was on {SALIENT}."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_previous_session_not_a_recency_keyword():
    ans = f"During your previous session on {SALIENT}, you felt some discomfort."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_plural_sessions_not_matched():
    ans = f"Your most recent sessions were solid, including one on {SALIENT}."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_date_in_next_sentence_not_grabbed():
    ans = f"Your most recent session went well. On {SALIENT} you trained hard."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_multi_exercise_unattributable_left_unchanged():
    # Bare "most recent session", no exercise nameable before it → ambiguous which
    # latest applies → leave unchanged (safe fallback).
    ans = "Here is a quick update. Your most recent session was on 2020-01-01."
    out, flags = C.recency_guard(ans, _pkg_multi())
    assert out == ans and flags == []


def test_empty_and_missing_inputs():
    assert C.recency_guard("", _pkg_single())   == ("", [])
    assert C.recency_guard("   ", _pkg_single()) == ("   ", [])
    ans = f"Your most recent session was on {SALIENT}."
    assert C.recency_guard(ans, {})   == (ans, [])   # no truth → no-op
    assert C.recency_guard(ans, None) == (ans, [])


# ══════════════════════════════════════════════════════════════════════════════
# Human date formats — the live-repro shape. The app writes "June 25, 2026", NOT
# ISO (ISO appears only inside the display block), so an ISO-only matcher never
# fires in production. These lock the real form and the styled correction.
# ══════════════════════════════════════════════════════════════════════════════

def test_corrects_month_first_full_with_comma():
    ans = "Your most recent session was on June 15, 2026. Keep going!"
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == "Your most recent session was on June 25, 2026. Keep going!"
    assert len(flags) == 1
    assert flags[0]["original"]  == "June 15, 2026"
    assert flags[0]["corrected"] == "June 25, 2026"


def test_corrects_month_first_abbrev_no_comma():
    ans = "Your latest session was on Jun 15 2026."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert "Jun 25 2026" in out and "Jun 15 2026" not in out
    assert flags[0]["corrected"] == "Jun 25 2026"     # abbrev + no comma mirrored


def test_corrects_day_first():
    ans = "Your last session was on 15 June 2026, a while ago."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert "25 June 2026" in out and "15 June 2026" not in out
    assert flags[0]["corrected"] == "25 June 2026"


def test_corrects_day_first_with_ordinal():
    ans = "Your most recent session was on 15th June 2026."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert "25 June 2026" in out and "15th June 2026" not in out


def test_month_first_style_mirrored_in_multi():
    ans = "Your most recent Sumo Squats session was on June 15, 2026."
    out, flags = C.recency_guard(ans, _pkg_multi())
    assert out == "Your most recent Sumo Squats session was on June 25, 2026."
    assert flags[0]["exercise"] == "Sumo Squats"


def test_human_date_correct_untouched():
    ans = "Your most recent session was on June 25, 2026. Nice."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_human_date_pain_mention_untouched():
    ans = "You noted knee pain on June 15, 2026, so monitor it."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_human_date_next_sentence_not_grabbed():
    ans = "Your most recent session went well. On June 15, 2026 you had pain."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_numeric_slash_date_left_alone():
    # Ambiguous M/D vs D/M — documented no-op.
    ans = "Your most recent session was on 06/15/2026."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert out == ans and flags == []


def test_non_month_word_not_mistaken_for_date():
    # "session 15 2026" shape must not be read as a date.
    ans = "Your most recent session 15 2026 units of work were logged."
    out, flags = C.recency_guard(ans, _pkg_single())
    assert flags == []


# ══════════════════════════════════════════════════════════════════════════════
# Live-DB anchor — recency_truth agrees with the authoritative package field
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sumo_pkg():
    return prepare_analysis_package(
        query_period_days=90, exercise_names=["Sumo Squats"], include_phase2=True)


def test_live_out_of_window_exercise_date_is_left_alone():
    """The same case on a real package: an exercise last trained before the
    window, picked from the DB rather than hard-coded, so it survives new logs."""
    import datetime
    import sqlite3
    start = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
    con = sqlite3.connect(f"file:{os.environ['FITNOTES_DB_PATH']}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT e.name, MAX(t.date) AS d FROM training_log t "
            "JOIN exercise e ON e._id = t.exercise_id "
            "GROUP BY e.name HAVING d < ? ORDER BY d DESC LIMIT 1", (start,)).fetchone()
    finally:
        con.close()
    if row is None:
        pytest.skip("every logged exercise was trained inside the window")
    name, last = row
    pkg = prepare_analysis_package(query_period_days=90, exercise_names=[name])
    assert pkg["exercises"] == [] and pkg["training_consistency"]["last_session"] != last
    ans = f"You haven't done {name} in the last 90 days. Your most recent session was on {last}."
    assert C.recency_guard(ans, pkg) == (ans, [])


def test_live_truth_matches_package_field(sumo_pkg):
    ex = next(e for e in sumo_pkg["exercises"] if e["name"] == "Sumo Squats")
    field_latest = ex["progression"]["latest_session_date"]
    t = C.recency_truth(sumo_pkg)
    assert t["per_exercise"]["Sumo Squats"] == field_latest
    # And a bad draft against the live package gets corrected to that field.
    bad = f"Your most recent Sumo Squats session was on {SALIENT}."
    out, flags = C.recency_guard(bad, sumo_pkg)
    assert field_latest in out and SALIENT not in out
    assert flags and flags[0]["corrected"] == field_latest
