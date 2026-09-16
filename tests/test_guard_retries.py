"""
The two guard retries in Coordinator._stage_display_fidelity: the plan guard
(a real clash) and the set-count guard (a set count the data doesn't have).

WHY THIS EXISTS (live re-check, 2026-09-16). Both retries put the first draft
into the conversation as the coach's message and the correction as a message
FROM THE USER, so the model answered the user: prompt 1 shipped "while addressing
your concerns regarding secondary muscle involvement" and "In the previous
design, the overlap … was problematic". A concern alone — which the guard itself
calls legitimate — set off that retry. And the retried answer skipped every check
the first draft passed: no grounding, no recency or limiting guard. A retry that
failed was caught as "skipped", so a real clash could ship with no note at all.

A retry is now a FRESH answer: the model never sees its draft, nothing is voiced
as the user, and code supplies the facts as [REQUIREMENTS]. The retry then passes
the same checks as the first draft.

Real Coordinator stage; fake analyze / ground_check that record what they get;
a synthetic ontology in a tmp dir.
"""

import asyncio
import csv
from types import SimpleNamespace

import pytest

from src import analysis_agent
from src import ontology as ont_mod

#   Arms              Legs         Chest
#     ├ Biceps          └ Glutes
#     └ Triceps
_MUSCLES = [(1, "Arms", "", ""), (2, "Biceps", 1, "medium"), (3, "Triceps", 1, "medium"),
            (4, "Legs", "", "large"), (5, "Glutes", 4, "large"), (6, "Chest", "", "large")]
_EXERCISES = [(1, "Barbell Curl", "barbell", "elbow flexion"),
              (2, "Machine Curl", "machine", "elbow flexion"),
              (3, "Hip Thrust", "barbell", "hip extension"),
              (4, "Close Grip Bench", "barbell", "horizontal push"),
              (5, "Skull Crusher", "dumbbell", "elbow extension")]
_EDGES = [(1, 2, "primary", "test"), (2, 2, "primary", "test"), (3, 5, "primary", "test"),
          (4, 6, "primary", "test"), (4, 3, "secondary", "test"), (5, 3, "primary", "test")]
# The user logs Machine Curl as "Machine Curls".
_ALIASES = [("Barbell Curl", 1), ("Machine Curls", 2), ("Hip Thrust", 3),
            ("Close Grip Bench", 4), ("Skull Crusher", 5)]

LATEST = "2026-09-10"
PKG = {
    "query_period_days": 90,
    "exercises": [{"name": "Barbell Curl", "progression": {"latest_session_date": LATEST}}],
    "suggestable_exercises": [],
    "muscle_ontology_summary": {"weeks_in_window": 13.0, "muscles": [
        {"muscle": "Chest", "primary_sets": 120, "secondary_sets": 0, "limiting_sets": 0,
         "primary_sets_per_week": 9.2, "secondary_sets_per_week": 0.0}]},
}

PLAN_Q = "build me a week of arm training"
CLASH = "* **Monday:** Barbell Curl (4 sets)\n* **Tuesday:** Machine Curl (4 sets)\n"
CLEAN = "* **Monday:** Barbell Curl (4 sets)\n* **Wednesday:** Machine Curl (4 sets)\n"
CONCERN_ONLY = "* **Monday:** Close Grip Bench (4 sets)\n* **Tuesday:** Skull Crusher (4 sets)\n"
CLASH_TEXT = "Barbell Curl on Monday and Machine Curls on Tuesday both train Biceps directly"
PLAN_NOTE = "Note: this plan still has"
COUNT_NOTE = "some set counts above do not match your data"


@pytest.fixture(autouse=True)
def ont(tmp_path, monkeypatch):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], _MUSCLES)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"], _EXERCISES)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], _EDGES)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], _ALIASES)
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")     # client construction only
    ont_mod.clear_cache()
    assert ont_mod.load_ontology(force=True)["errors"] == []
    yield
    ont_mod.clear_cache()


def _fakes(monkeypatch, retry, grounding_flags):
    """Fake analyze / ground_check that record what they receive."""
    analyze_calls, grounded = [], []

    async def fake_analyze(package, q, research, memories, conversation_context,
                           custom_query, requirements=None):
        analyze_calls.append({"conversation": conversation_context,
                              "requirements": requirements})
        if isinstance(retry, Exception):
            raise retry
        return retry

    async def fake_ground_check(draft, grounding_context):
        grounded.append(draft)
        return draft, list(grounding_flags or [])

    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)
    monkeypatch.setattr(analysis_agent, "ground_check", fake_ground_check)
    return analyze_calls, grounded


def _cache(pkg):
    async def ensure_package(_coordinator, _state):
        return pkg
    return SimpleNamespace(ensure_package=ensure_package, research=None, memories=None,
                           conversation_context=None, custom_query=None)


def run_stage(monkeypatch, question, answer, retry=None, grounding_flags=None, pkg=None):
    """Run the real stage. `retry` is what the redraft returns, or an Exception
    to raise. Returns (stage output, analyze calls, ground_check drafts)."""
    from src.coordinator import Coordinator

    if pkg is not None:
        analyze_calls, grounded = _fakes(monkeypatch, retry, grounding_flags)
        state = {"question": question, "scoped_question": question, "answer": answer,
                 "params": {"query_period_days": 90}}
        out = asyncio.run(Coordinator(agent_session=None)._stage_display_fidelity(
            state, _cache(pkg)))
        return out, analyze_calls, grounded

    analyze_calls, grounded = [], []

    async def fake_analyze(package, q, research, memories, conversation_context,
                           custom_query, requirements=None):
        analyze_calls.append({"conversation": conversation_context,
                              "requirements": requirements})
        if isinstance(retry, Exception):
            raise retry
        return retry

    async def fake_ground_check(draft, grounding_context):
        grounded.append(draft)
        return draft, list(grounding_flags or [])

    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)
    monkeypatch.setattr(analysis_agent, "ground_check", fake_ground_check)

    async def ensure_package(_coordinator, _state):
        return PKG

    cache = SimpleNamespace(ensure_package=ensure_package, research=None, memories=None,
                            conversation_context=None, custom_query=None)
    state = {"question": question, "scoped_question": question, "answer": answer,
             "params": {"query_period_days": 90}}
    out = asyncio.run(Coordinator(agent_session=None)._stage_display_fidelity(state, cache))
    return out, analyze_calls, grounded


def _no_draft_no_user_voice(conversation):
    """The redraft sees the ORIGINAL conversation only (None here)."""
    return not conversation


# ── Plan retry ────────────────────────────────────────────────────────────────

def test_plan_retry_is_a_fresh_answer_with_facts(monkeypatch):
    _out, calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=CLEAN)
    assert len(calls) == 1
    assert _no_draft_no_user_voice(calls[0]["conversation"]), calls[0]["conversation"]
    reqs = " ".join(calls[0]["requirements"] or [])
    assert "Barbell Curl" in reqs and "Machine Curls" in reqs and "Biceps" in reqs


def test_a_concern_alone_does_not_retry(monkeypatch):
    # No group rows: this test is about the concern, not the volume check (every
    # plan is volume-checked, and this arms plan would "cut" PKG's Chest row).
    no_groups = dict(PKG, muscle_ontology_summary={"weeks_in_window": 13.0, "muscles": []})
    out, calls, _g = run_stage(monkeypatch, PLAN_Q, CONCERN_ONLY, retry=CLEAN, pkg=no_groups)
    assert calls == []
    assert out["answer"] == CONCERN_ONLY


def test_a_plan_retry_is_fact_checked(monkeypatch):
    probe = {"kind": "grounding_probe"}
    out, _calls, grounded = run_stage(monkeypatch, PLAN_Q, CLASH, retry=CLEAN,
                                      grounding_flags=[probe])
    assert [g.strip() for g in grounded] == [CLEAN.strip()]   # strip_tags trims the ends
    assert probe in out["flagged"]


def test_a_plan_retry_passes_the_other_guards(monkeypatch):
    retry = ("Your most recent session was on 2020-01-01.\n\n" + CLEAN
             + "\nFinish with Barbell Curls.")
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=retry)
    assert LATEST in out["answer"] and "2020-01-01" not in out["answer"]   # recency
    assert "Finish with Barbell Curl." in out["answer"]                    # name check


def test_a_retry_that_still_clashes_says_so_accurately(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=CLASH)
    assert f"{PLAN_NOTE} {CLASH_TEXT}." in out["answer"]


def test_a_failed_plan_retry_does_not_hide_the_clash(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=RuntimeError("quota"))
    assert out["answer"].startswith(CLASH)
    assert f"{PLAN_NOTE} {CLASH_TEXT}." in out["answer"]


def test_a_successful_retry_leaves_no_stale_clash_flag(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=CLEAN)
    assert not [f for f in out["flagged"] if f.get("kind") == "plan_consecutive_days"]


def test_a_plan_retry_is_checked_for_invented_set_counts(monkeypatch):
    retry = "You currently perform 25 sets for Chest.\n\n" + CLEAN
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CLASH, retry=retry)
    assert COUNT_NOTE in out["answer"]


# ── Set-count retry ───────────────────────────────────────────────────────────

COUNT_Q = "how is my chest training going?"
WRONG_COUNT = "You currently perform 25 sets for Chest."


def test_set_count_retry_is_a_fresh_answer_with_the_real_figures(monkeypatch):
    fixed = "You currently perform 120 sets for Chest."
    out, calls, grounded = run_stage(monkeypatch, COUNT_Q, WRONG_COUNT, retry=fixed)
    assert len(calls) == 1
    assert _no_draft_no_user_voice(calls[0]["conversation"]), calls[0]["conversation"]
    reqs = " ".join(calls[0]["requirements"] or [])
    assert "Chest" in reqs and "120" in reqs
    assert "25" not in reqs, "the wrong number must not be repeated to the model"
    assert grounded == [fixed]
    assert out["answer"] == fixed


def test_a_failed_set_count_retry_does_not_hide_the_wrong_count(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, COUNT_Q, WRONG_COUNT, retry=RuntimeError("quota"))
    assert out["answer"].startswith(WRONG_COUNT)
    assert COUNT_NOTE in out["answer"]


# ══════════════════════════════════════════════════════════════════════════════
# The two remaining retries: coverage and the display re-frame
# ══════════════════════════════════════════════════════════════════════════════
#
# Same fault as the plan and set-count retries had: the draft went back to the
# model with a correction written as the USER ("Your previous answer did not
# fully address the question…", "Reproduce the per-set display lines…"), and the
# display re-frame was never fact-checked. Both are now fresh answers with
# [REQUIREMENTS]; both are grounded.

COVERAGE_Q = "how are my curls going, and should I add a second arm day?"


def run_coverage(monkeypatch, coverage_result, retry="A complete answer.", grounding_flags=None):
    from src.coordinator import Coordinator

    analyze_calls, grounded = _fakes(monkeypatch, retry, grounding_flags)

    async def fake_coverage(self, question, answer):
        return coverage_result

    monkeypatch.setattr(Coordinator, "_coverage_check", fake_coverage)
    state = {"question": COVERAGE_Q, "scoped_question": COVERAGE_Q, "params": {},
             "answer": "Your curls are progressing.", "flagged": []}
    out = asyncio.run(Coordinator(agent_session=None)._stage_coverage(state, _cache(PKG)))
    return out, analyze_calls, grounded


def test_coverage_retry_is_a_fresh_answer_naming_the_missing_parts(monkeypatch):
    probe = {"kind": "grounding_probe"}
    out, calls, grounded = run_coverage(
        monkeypatch, ("Your curls are progressing.", False, ["whether to add a second arm day"]),
        grounding_flags=[probe])
    assert len(calls) == 1
    assert _no_draft_no_user_voice(calls[0]["conversation"]), calls[0]["conversation"]
    assert any("whether to add a second arm day" in r for r in calls[0]["requirements"] or [])
    assert grounded == ["A complete answer."]
    assert out["answer"] == "A complete answer." and probe in out["flagged"]


def test_coverage_retry_works_with_the_old_two_value_check(monkeypatch):
    _out, calls, _g = run_coverage(monkeypatch, ("Your curls are progressing.", False))
    assert len(calls) == 1 and _no_draft_no_user_voice(calls[0]["conversation"])
    assert calls[0]["requirements"], "a generic requirement is still given"


def test_a_complete_answer_is_not_retried(monkeypatch):
    out, calls, _g = run_coverage(monkeypatch, ("Your curls are progressing.", True, []))
    assert calls == [] and out["answer"] == "Your curls are progressing."


DISPLAY_LINES = ["2026-09-09 — Barbell Curl (1 sets):", "Set 1: 100.0 lbs × 5 reps"]
DISPLAY_PKG = dict(PKG, display_sets=DISPLAY_LINES)
DISPLAY_Q = "show me my last barbell curl session"
MISSING_LINES = "Your last barbell curl session went well."


def test_display_reframe_is_a_fresh_fact_checked_answer(monkeypatch):
    probe = {"kind": "grounding_probe"}
    reframed = "Here it is:\n" + "\n".join(DISPLAY_LINES)
    out, calls, grounded = run_stage(monkeypatch, DISPLAY_Q, MISSING_LINES, retry=reframed,
                                     grounding_flags=[probe], pkg=DISPLAY_PKG)
    assert len(calls) == 1
    assert _no_draft_no_user_voice(calls[0]["conversation"]), calls[0]["conversation"]
    assert any("[DISPLAY]" in r for r in calls[0]["requirements"] or [])
    assert [g.strip() for g in grounded] == [reframed.strip()]
    assert all(line in out["answer"] for line in DISPLAY_LINES)
    assert probe in out["flagged"]


def test_a_failed_display_reframe_still_appends_the_lines(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, DISPLAY_Q, MISSING_LINES,
                                retry=RuntimeError("quota"), pkg=DISPLAY_PKG)
    assert out["answer"].startswith(MISSING_LINES)
    assert all(line in out["answer"] for line in DISPLAY_LINES)


# ══════════════════════════════════════════════════════════════════════════════
# F6 · a plan keeps every other group at maintenance (unless "only X")
# ══════════════════════════════════════════════════════════════════════════════
#
# The volume shortfall shares the plan retry: a fresh answer given the group's
# real weekly figure, then an accurate note if it is still short. A plan with a
# clash AND a shortfall still retries ONCE.

VOL_PKG = dict(PKG, muscle_ontology_summary={"weeks_in_window": 13.0, "muscles": [
    {"muscle": "Arms", "primary_sets": 130, "secondary_sets": 0, "limiting_sets": 0,
     "primary_sets_per_week": 10.0, "secondary_sets_per_week": 0.0}]})
CUT = "* **Monday:** Barbell Curl (2 sets)\n* **Wednesday:** Skull Crusher (2 sets)\n"
KEPT = "* **Monday:** Barbell Curl (5 sets)\n* **Wednesday:** Skull Crusher (5 sets)\n"
VOLUME_NOTE = "Note: this plan gives Arms 4 sets a week against your current 10"


def test_a_plan_that_cuts_a_group_is_retried_with_the_real_figure(monkeypatch):
    out, calls, _g = run_stage(monkeypatch, PLAN_Q, CUT, retry=KEPT, pkg=VOL_PKG)
    assert len(calls) == 1
    assert _no_draft_no_user_voice(calls[0]["conversation"]), calls[0]["conversation"]
    reqs = " ".join(calls[0]["requirements"] or [])
    assert "Arms" in reqs and "10" in reqs and "4 sets" not in reqs
    assert "Note: this plan gives" not in out["answer"]


def test_a_retry_that_is_still_short_says_so(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CUT, retry=CUT, pkg=VOL_PKG)
    assert VOLUME_NOTE in out["answer"]


def test_an_explicit_only_plan_is_not_volume_checked(monkeypatch):
    out, calls, _g = run_stage(monkeypatch, "build me an arms only week", CUT,
                               retry=KEPT, pkg=VOL_PKG)
    assert calls == [] and out["answer"] == CUT


def test_a_clash_and_a_shortfall_share_one_retry(monkeypatch):
    clash_and_cut = "* **Monday:** Barbell Curl (2 sets)\n* **Tuesday:** Machine Curl (2 sets)\n"
    _out, calls, _g = run_stage(monkeypatch, PLAN_Q, clash_and_cut, retry=KEPT, pkg=VOL_PKG)
    assert len(calls) == 1
    reqs = " ".join(calls[0]["requirements"] or [])
    assert "consecutive days" in reqs and "Arms: the user currently does 10" in reqs


def test_a_failed_volume_retry_does_not_hide_the_cut(monkeypatch):
    out, _calls, _g = run_stage(monkeypatch, PLAN_Q, CUT, retry=RuntimeError("quota"), pkg=VOL_PKG)
    assert out["answer"].startswith(CUT) and VOLUME_NOTE in out["answer"]
