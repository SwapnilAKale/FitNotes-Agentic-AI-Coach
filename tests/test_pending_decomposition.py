"""
tests/test_pending_decomposition.py — Decomposition Arc Stage 2: the
pending-disambiguation slot.

Lifecycle under test (user rulings 2026-07-09/12):
  - armed when the analytical lane exits on a disambiguation ask (full
    candidate list + deep-copied params incl. the Stage-1 requests array);
  - a candidate reply is matched DETERMINISTICALLY (no LLM) and resumes the
    original question with patched params, skipping classify entirely;
  - a reply naming a real off-list exercise is accepted as an override;
  - a multi-candidate reply gets ONE clarify re-ask (no strike);
  - a non-answer message strikes the slot (reminder appended to that turn's
    real answer, once); the second strike drops it;
  - 48h staleness expiry (checkpoint's MAX_AGE_HOURS), silent;
  - the reminder never enters recorded/extraction history.

Harness mirrors tests/test_demographic_followup.py (stubbed genai client,
stubbed inner route) and tests/test_routing_step_b.py (_spy_resolver).
No Gemini, no server.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from langgraph.graph import END                          # noqa: E402

from src import coordinator as coordinator_mod          # noqa: E402
from src import analysis_agent                          # noqa: E402
from src import memory                                  # noqa: E402
from src.coordinator import Coordinator                 # noqa: E402
from src.graph.coordinator_graph import _route_after_entry   # noqa: E402


# ── Harness ──────────────────────────────────────────────────────────────────

class _FakeAgent:
    def __init__(self):
        self.recorded = []

    def record_external_exchange(self, q, a):
        self.recorded.append((q, a))


MAIN = "Your deadlift is up 5% this period."

SQUAT_CANDIDATES = ["Sumo Squats", "Dumbbell Squats", "Smith Machine Squats"]

ORIGINAL_Q = ("Is my squat progressing, and also show me my last "
              "bench press session?")


def _slot_params():
    """A classify-params dict as Stage 1 emits it for the compound question."""
    return {
        "route": "analytical", "display_intent": False,
        "exercise_names": ["squat", "bench press"], "muscle_groups": None,
        "query_period_days": 90, "rep_target": None, "cardio_lock": None,
        "needs_custom_sql": False, "custom_sql_intent": None,
        "requests": [
            {"index": 0, "lane": "analytical",
             "intent_text": "Is my squat progressing?",
             "display_intent": False, "exercise_names": ["squat"],
             "muscle_groups": None, "query_period_days": 90,
             "rep_target": None, "cardio_lock": None,
             "needs_custom_sql": False, "custom_sql_intent": None},
            {"index": 1, "lane": "analytical",
             "intent_text": "Show me my last bench press session.",
             "display_intent": True, "exercise_names": ["bench press"],
             "muscle_groups": None, "query_period_days": 90,
             "rep_target": None, "cardio_lock": None,
             "needs_custom_sql": False, "custom_sql_intent": None},
        ],
    }


def _seed_slot(coord, **over):
    slot = {
        "question":   ORIGINAL_Q,
        "params":     _slot_params(),
        "name":       "squat",
        "candidates": list(SQUAT_CANDIDATES),
        "created":    datetime.now().isoformat(),
        "strikes":    0,
        "reminded":   False,
        "clarified":  False,
    }
    slot.update(over)
    coord._pending_decomposition = slot
    return slot


@pytest.fixture()
def coord(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "mem.json")
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    c = Coordinator(agent_session=_FakeAgent())

    async def fake_inner(q, **kw):
        return {"answer": MAIN, "route": "analytical",
                "flagged_claims": [], "error": None}
    monkeypatch.setattr(c, "_route_with_checkpoint", fake_inner)
    return c


def _spy_resolver(monkeypatch, *, match=None, candidates=None, side_effect=None):
    """Patch resolve_exercise_name at its source module; record every call."""
    import src.shared.resolver as resolver_mod
    calls: list = []

    def fake(name, db_path, permissive=False):
        calls.append(name)
        if side_effect is not None:
            return side_effect(name)
        return {"match": match, "candidates": candidates or []}

    monkeypatch.setattr(resolver_mod, "resolve_exercise_name", fake)
    return calls


def _stub_run_analytical(coord, monkeypatch):
    """Replace _run_analytical, capturing (question, params)."""
    captured = {}

    async def fake_ra(q, p, resume=None):
        captured["question"] = q
        captured["params"] = p
        return "RESUMED", []
    monkeypatch.setattr(coord, "_run_analytical", fake_ra)
    return captured


def _forbid_classify(coord, monkeypatch):
    async def boom(q):
        raise AssertionError("_classify must NOT be called on a resume turn")
    monkeypatch.setattr(coord, "_classify", boom)


# ═══ 1. Arming ════════════════════════════════════════════════════════════════

def test_arm_on_ambiguity_captures_everything(coord, monkeypatch):
    _spy_resolver(monkeypatch, candidates=SQUAT_CANDIDATES)
    params = _slot_params()

    answer, flagged = asyncio.run(coord._run_analytical(ORIGINAL_Q, params))

    assert "which one did you mean" in answer.lower()
    slot = coord._pending_decomposition
    assert slot is not None
    assert slot["question"] == ORIGINAL_Q
    assert slot["name"] == "squat"
    assert slot["candidates"] == SQUAT_CANDIDATES        # FULL list
    assert slot["strikes"] == 0 and slot["reminded"] is False
    assert slot["clarified"] is False
    datetime.fromisoformat(slot["created"])              # parses
    # deep copy — never aliases the live params/state
    assert slot["params"] == params
    assert slot["params"] is not params
    assert slot["params"]["requests"] is not params["requests"]
    assert slot["params"]["requests"] == params["requests"]


# ═══ 2. Consume-resume (no classify) ═════════════════════════════════════════

def test_consume_resumes_with_patched_params_no_classify(coord, monkeypatch):
    _seed_slot(coord)
    _forbid_classify(coord, monkeypatch)
    captured = _stub_run_analytical(coord, monkeypatch)

    res = asyncio.run(coord.route("Sumo Squats"))

    assert res["answer"] == "RESUMED"
    assert res["route"] == "analytical"
    assert res["resolved_question"] == ORIGINAL_Q
    assert captured["question"] == ORIGINAL_Q
    p = captured["params"]
    assert p["exercise_names"] == ["Sumo Squats", "bench press"]   # flat patched
    assert p["requests"][0]["exercise_names"] == ["Sumo Squats"]   # chunk patched
    assert p["requests"][1]["exercise_names"] == ["bench press"]   # sibling intact
    assert coord._pending_decomposition is None                    # cleared
    # finalize recorded the ORIGINAL question + resumed answer into history
    assert coord._history[-2] == {"role": "user", "content": ORIGINAL_Q}
    assert coord._history[-1] == {"role": "assistant", "content": "RESUMED"}


def test_no_reminder_on_resume_turn(coord, monkeypatch):
    _seed_slot(coord)
    _stub_run_analytical(coord, monkeypatch)
    res = asyncio.run(coord.route("Sumo Squats"))
    assert "Still pending" not in res["answer"]


# ═══ 3. Matching tiers ═══════════════════════════════════════════════════════

def test_match_exact_and_case_space_insensitive(coord):
    kind, val, src = coord._match_disambiguation_reply(
        "Sumo Squats", SQUAT_CANDIDATES)
    assert (kind, val, src) == ("match", "Sumo Squats", "exact")
    kind, val, src = coord._match_disambiguation_reply(
        "  sumo   SQUATS. ", SQUAT_CANDIDATES)
    assert (kind, val) == ("match", "Sumo Squats")


def test_match_unique_substring_both_directions(coord):
    kind, val, src = coord._match_disambiguation_reply(
        "sumo", SQUAT_CANDIDATES)
    assert (kind, val, src) == ("match", "Sumo Squats", "substring")
    kind, val, src = coord._match_disambiguation_reply(
        "I meant Sumo Squats please", SQUAT_CANDIDATES)
    assert (kind, val) == ("match", "Sumo Squats")


def test_match_multi_substring_is_ambiguous(coord):
    kind, subset, _ = coord._match_disambiguation_reply(
        "squats", SQUAT_CANDIDATES)
    assert kind == "ambiguous"
    assert set(subset) == set(SQUAT_CANDIDATES)


def test_resolver_fallback_typo_within_candidates(coord, monkeypatch):
    calls = _spy_resolver(monkeypatch, match="Sumo Squats")
    kind, val, src = coord._match_disambiguation_reply(
        "sumo squt", SQUAT_CANDIDATES)
    assert (kind, val, src) == ("match", "Sumo Squats", "resolver")
    assert calls == ["sumo squt"]


def test_resolver_off_list_accepted_as_override(coord, monkeypatch):
    _spy_resolver(monkeypatch, match="Barbell Squat")
    kind, val, src = coord._match_disambiguation_reply(
        "Barbell Squat", SQUAT_CANDIDATES)
    assert (kind, val, src) == ("match", "Barbell Squat", "override")


@pytest.mark.parametrize("reply", [
    "how is my squat volume trending this month?",   # question mark
    "tell me about every single squat variation",    # > 5 words
    "log squats 100 lbs for 5 reps",                 # write intent
    "continue",                                      # checkpoint intent
])
def test_name_shape_gate_blocks_resolver(coord, monkeypatch, reply):
    calls = _spy_resolver(monkeypatch, match="Sumo Squats")
    kind, _, _ = coord._match_disambiguation_reply(reply, SQUAT_CANDIDATES)
    assert kind in ("miss", "ambiguous")   # never a resolver match
    assert calls == []                     # resolver NOT consulted


# ═══ 4/5. Strikes, clarify, reminder ═════════════════════════════════════════

def test_clarify_once_no_strike_then_miss(coord):
    slot = _seed_slot(coord)
    r1 = asyncio.run(coord.route("squats"))          # matches all 3
    assert r1["route"] == "decomposition_clarify"
    assert "did you mean" in r1["answer"]
    assert slot["clarified"] is True and slot["strikes"] == 0
    # second ambiguous attempt → treated as a miss → strike 1, routed normally
    r2 = asyncio.run(coord.route("squats"))
    assert r2["answer"].startswith(MAIN)
    assert slot["strikes"] == 1
    assert coord._pending_decomposition is slot      # survives strike 1


def test_strike1_reminder_then_strike2_drop(coord):
    slot = _seed_slot(coord)
    r1 = asyncio.run(coord.route("how's my deadlift?"))
    assert r1["answer"].startswith(MAIN)
    assert "Still pending" in r1["answer"]
    assert ORIGINAL_Q[:40] in r1["answer"]           # truncated question shown
    assert slot["reminded"] is True and slot["strikes"] == 1
    r2 = asyncio.run(coord.route("and my bench press history?"))
    assert r2["answer"] == MAIN                      # clean — no second reminder
    assert coord._pending_decomposition is None      # dropped at strike 2


def test_reminder_only_once_even_across_real_answers(coord):
    slot = _seed_slot(coord, reminded=True)
    r = asyncio.run(coord.route("how's my deadlift?"))
    assert "Still pending" not in r["answer"]
    assert slot["strikes"] == 1


def test_no_reminder_on_non_real_answer_turn(coord, monkeypatch):
    slot = _seed_slot(coord)

    async def fake_inner(q, **kw):
        return {"answer": "Resume or discard?", "route": "checkpoint_confirm",
                "flagged_claims": [], "error": None}
    monkeypatch.setattr(coord, "_route_with_checkpoint", fake_inner)

    r = asyncio.run(coord.route("what about my volume this year?"))
    assert "Still pending" not in r["answer"]
    assert slot["strikes"] == 1 and slot["reminded"] is False


# ═══ 6. Staleness ════════════════════════════════════════════════════════════

def test_48h_expiry_silent_drop(coord):
    _seed_slot(coord,
               created=(datetime.now() - timedelta(hours=49)).isoformat())
    r = asyncio.run(coord.route("Sumo Squats"))      # even a perfect answer
    assert r["answer"] == MAIN                       # routed normally
    assert coord._pending_decomposition is None
    assert "Still pending" not in r["answer"]


def test_corrupt_created_counts_as_stale(coord):
    _seed_slot(coord, created="not-a-timestamp")
    r = asyncio.run(coord.route("Sumo Squats"))
    assert r["answer"] == MAIN
    assert coord._pending_decomposition is None


# ═══ 7. Chain re-arm ═════════════════════════════════════════════════════════

def test_chain_second_ambiguity_rearms_with_patched_params(coord, monkeypatch):
    old_slot = _seed_slot(coord)

    def side_effect(name):
        if "sumo" in name.lower():
            return {"match": name, "candidates": []}
        return {"match": None, "candidates": [
            "Flat Dumbbell Bench Press", "Incline Dumbbell Bench Press"]}
    _spy_resolver(monkeypatch, side_effect=side_effect)

    res = asyncio.run(coord.route("Sumo Squats"))

    assert "which one did you mean" in res["answer"].lower()
    assert "bench press" in res["answer"].lower()
    new_slot = coord._pending_decomposition
    assert new_slot is not None and new_slot is not old_slot
    assert new_slot["name"] == "bench press"
    assert new_slot["strikes"] == 0 and new_slot["reminded"] is False
    # the re-armed slot carries the FIRST patch already applied
    assert new_slot["params"]["exercise_names"] == ["Sumo Squats", "bench press"]
    assert new_slot["params"]["requests"][0]["exercise_names"] == ["Sumo Squats"]


# ═══ 8. Coexistence with the demographic follow-up ══════════════════════════

def test_exercise_reply_consumed_followup_untouched(coord, monkeypatch):
    _seed_slot(coord)
    coord._pending_followup = {"key": "birthdate", "clarified": False}
    _stub_run_analytical(coord, monkeypatch)
    res = asyncio.run(coord.route("Sumo Squats"))
    assert res["answer"] == "RESUMED"
    assert coord._pending_followup == {"key": "birthdate", "clarified": False}


def test_date_reply_strikes_decomposition_feeds_followup(coord, monkeypatch):
    slot = _seed_slot(coord)
    coord._pending_followup = {"key": "birthdate", "clarified": False}
    _spy_resolver(monkeypatch, match=None)           # date resolves to nothing
    res = asyncio.run(coord.route("2003-06-18"))
    assert res["route"] == "followup_ack"            # demographics consumed it
    assert slot["strikes"] == 1                      # decomposition struck
    assert coord._pending_decomposition is slot      # but survives


# ═══ 9. Graph plumbing regressions ═══════════════════════════════════════════

def test_entry_boundary_pass_through_skips_all_guards(coord, monkeypatch):
    def boom(msg):
        raise AssertionError("write-intent regex must not run on a resume")
    monkeypatch.setattr(coordinator_mod, "_is_write_intent", boom)
    params = {"route": "analytical", "exercise_names": ["Sumo Squats"]}
    out = coord._node_entry_boundary(
        {"question": "log squats 100x5", "params": params})
    assert out == {"params": params}
    assert out["params"] is params


def test_route_after_entry_truth_table():
    assert _route_after_entry({"result": {"answer": "hi"}}) == END
    assert _route_after_entry(
        {"params": {"route": "operational"}}) == "dispatch_operational"
    assert _route_after_entry(
        {"params": {"route": "analytical"}}) == "dispatch_analytical"
    assert _route_after_entry({"params": None}) == "classify"
    assert _route_after_entry({}) == "classify"


def test_synthetic_write_params_still_dispatch_operational(coord):
    out = coord._node_entry_boundary({"question": "/log bench 100x5"})
    assert out["params"]["route"] == "operational"
    assert _route_after_entry({"params": out["params"]}) == "dispatch_operational"


# ═══ 10. Reminder never enters recorded/extraction history ═══════════════════

def test_reminder_not_in_recorded_history(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "mem.json")
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    agent = _FakeAgent()
    c = Coordinator(agent_session=agent)

    monkeypatch.setattr(coordinator_mod, "prepare_analysis_package",
                        lambda **kw: {"scope": "broad", "exercises": []})

    async def fake_analyze(*a, **k):
        return "Your deadlift is up 5%."
    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)

    async def fake_ground(draft, gctx):
        return draft, []
    monkeypatch.setattr(analysis_agent, "ground_check", fake_ground)

    async def fake_cov(self, q, answer):
        return answer, True
    monkeypatch.setattr(Coordinator, "_coverage_check", fake_cov)

    async def fake_classify(self, q):
        return {"route": "analytical", "exercise_names": None,
                "muscle_groups": None, "query_period_days": 90,
                "needs_custom_sql": False, "custom_sql_intent": None,
                "requests": None}
    monkeypatch.setattr(Coordinator, "_classify", fake_classify)

    _seed_slot(c)
    res = asyncio.run(c.route("how's my deadlift trending?"))

    # the RETURNED answer carries the reminder …
    assert "Still pending" in res["answer"]
    # … but the RECORDED exchange and self._history stay clean
    assert agent.recorded, "analytical turn should have recorded an exchange"
    assert "Still pending" not in agent.recorded[-1][1]
    assert all("Still pending" not in t["content"] for t in c._history)


# ═══ 11. Containment gate: override must stay in context ════════════════════

def _spy_categories(monkeypatch, mapping):
    """Patch exercise_categories at its source module; record every call."""
    import src.shared.resolver as resolver_mod
    calls: list = []

    def fake(names, db_path):
        calls.append(list(names))
        return {n: mapping[n] for n in names if n in mapping}

    monkeypatch.setattr(resolver_mod, "exercise_categories", fake)
    return calls


def test_override_in_context_by_term(coord, monkeypatch):
    # "Barbell Squat" contains the ambiguous term "squat" → accepted without
    # even needing the category lookup.
    _spy_resolver(monkeypatch, match="Barbell Squat")
    cat_calls = _spy_categories(monkeypatch, {})
    kind, val, src = coord._match_disambiguation_reply(
        "barbell squat", SQUAT_CANDIDATES, "squat")
    assert (kind, val, src) == ("match", "Barbell Squat", "override")
    assert cat_calls == []                       # term containment sufficed


def test_override_in_context_by_category(coord, monkeypatch):
    # "Leg Press" shares the Legs category with the squat candidates.
    _spy_resolver(monkeypatch, match="Leg Press")
    _spy_categories(monkeypatch, {
        "Leg Press": "Legs", "Sumo Squats": "Legs",
        "Dumbbell Squats": "Legs", "Smith Machine Squats": "Legs"})
    kind, val, src = coord._match_disambiguation_reply(
        "leg press", SQUAT_CANDIDATES, "squat")
    assert (kind, val, src) == ("match", "Leg Press", "override")


def test_override_out_of_context_detected(coord, monkeypatch):
    _spy_resolver(monkeypatch, match="Dumbbell Hammer Curl")
    _spy_categories(monkeypatch, {
        "Dumbbell Hammer Curl": "Biceps", "Sumo Squats": "Legs",
        "Dumbbell Squats": "Legs", "Smith Machine Squats": "Legs"})
    kind, val, _ = coord._match_disambiguation_reply(
        "dumbbell hammer curl", SQUAT_CANDIDATES, "squat")
    assert (kind, val) == ("out_of_context", "Dumbbell Hammer Curl")


def test_out_of_context_pushback_holds_ground_no_strike(coord, monkeypatch):
    slot = _seed_slot(coord)
    _spy_resolver(monkeypatch, match="Dumbbell Hammer Curl")
    _spy_categories(monkeypatch, {
        "Dumbbell Hammer Curl": "Biceps", "Sumo Squats": "Legs",
        "Dumbbell Squats": "Legs", "Smith Machine Squats": "Legs"})

    res = asyncio.run(coord.route("dumbbell hammer curl"))

    assert res["route"] == "decomposition_pushback"
    assert "isn't one" in res["answer"]
    assert "Sumo Squats" in res["answer"]              # candidates restated
    assert ORIGINAL_Q[:40] in res["answer"]            # original question shown
    assert "say it again" in res["answer"]             # the relent offer
    assert slot["strikes"] == 0                        # engagement, not a strike
    assert slot["rejected_override"] == "Dumbbell Hammer Curl"
    assert coord._pending_decomposition is slot        # slot survives
    assert "Still pending" not in res["answer"]        # no reminder on push-back


def test_insistence_relents_and_resumes(coord, monkeypatch):
    _seed_slot(coord, rejected_override="Dumbbell Hammer Curl")
    _spy_resolver(monkeypatch, match="Dumbbell Hammer Curl")
    _spy_categories(monkeypatch, {
        "Dumbbell Hammer Curl": "Biceps", "Sumo Squats": "Legs",
        "Dumbbell Squats": "Legs", "Smith Machine Squats": "Legs"})
    captured = _stub_run_analytical(coord, monkeypatch)

    res = asyncio.run(coord.route("dumbbell hammer curl"))

    assert res["answer"] == "RESUMED"
    p = captured["params"]
    assert p["exercise_names"] == ["Dumbbell Hammer Curl", "bench press"]
    assert p["requests"][0]["exercise_names"] == ["Dumbbell Hammer Curl"]
    assert coord._pending_decomposition is None        # consumed


def test_different_out_of_context_after_pushback_is_strike(coord, monkeypatch):
    slot = _seed_slot(coord, rejected_override="Dumbbell Hammer Curl")
    _spy_resolver(monkeypatch, match="Lat Pulldown")
    _spy_categories(monkeypatch, {
        "Lat Pulldown": "Back", "Sumo Squats": "Legs",
        "Dumbbell Squats": "Legs", "Smith Machine Squats": "Legs"})

    res = asyncio.run(coord.route("lat pulldown"))

    assert res["route"] != "decomposition_pushback"    # no push-back loop
    assert res["answer"].startswith(MAIN)              # routed normally
    assert slot["strikes"] == 1


def test_exercise_categories_real_db():
    from src.shared.resolver import exercise_categories
    cats = exercise_categories(
        ["Sumo Squats", "Dumbbell Hammer Curl"],
        os.environ["FITNOTES_DB_PATH"])
    assert cats.get("Sumo Squats") == "Legs"
    assert cats.get("Dumbbell Hammer Curl") == "Biceps"
    # exception path: bogus db → empty dict, never raises
    assert exercise_categories(["X"], "no/such/file.db") == {}
    assert exercise_categories([], os.environ["FITNOTES_DB_PATH"]) == {}


def test_analysis_prompt_has_formatting_directive_and_exemptions():
    from src.analysis_agent import _ANALYSIS_SYSTEM
    for anchor in ("READABILITY (markdown)", "### ", "HARD EXEMPTIONS",
                   "character-for-character", "citation tag"):
        assert anchor in _ANALYSIS_SYSTEM, f"missing anchor: {anchor}"
    # legacy anchors intact
    for anchor in ("ANSWER FORMAT", "ADVICE STYLE", "CITATION TAGS",
                   "Open with the most important finding"):
        assert anchor in _ANALYSIS_SYSTEM, f"legacy anchor missing: {anchor}"
