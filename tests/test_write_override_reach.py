"""
How far the write-regex verdict reaches — ledger row E.

THE DEFECT. "I did chest and triceps today" (a session to record) and "I did
shrugs today and my grip gave out" (a complaint to explain) are the same shape.
No regex separates them, so the pre-guard called both writes and the distrust
override forced both to operational — which made _CLASSIFY_SYSTEM's own rule
("a report of how a lift went is ANALYTICAL") unreachable for that phrasing.
The prose rule lost to code it did not know existed.

THE RULE. The write signal now has two strengths, and they differ in what they
LICENSE, not merely in what they matched:

    imperative ("log ...", "delete ...", "set a goal")
        the user named the action → the regex BINDS, overruling the classifier
    narration  ("I did ... today", no write verb)
        a hint → it spends the classify call and stands in when that call
        FAILS, but a successful classification overrules it

So the only thing in the system that can read the difference — the classifier —
is the thing that decides it, while a classify FAILURE still cannot lose a
write. Both halves are asserted below; neither is safe alone.

WHY THE FAILURE HALF IS NOT OPTIONAL: the three tests that originally pinned
"I did chest and triceps today" as a write all stub the client so that classify
RAISES. They passed through the parse-fail path, not through a verdict. A change
that only narrowed the override would have left them green while silently
breaking production, where classify succeeds.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src import coordinator as coordinator_mod            # noqa: E402
from src.coordinator import (                             # noqa: E402
    WRITE_FORM_IMPERATIVE,
    WRITE_FORM_NARRATION,
    Coordinator,
    _write_intent_form,
)


@pytest.fixture()
def coord(monkeypatch):
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _params(route="analytical", parse_failed=False, requests=None):
    p = {
        "route": route, "exercise_names": None, "muscle_groups": None,
        "query_period_days": 90, "needs_custom_sql": False,
        "custom_sql_intent": None, "requests": requests,
    }
    if parse_failed:
        p["_parse_failed"] = True
    return p


def _classify_returning(coord, monkeypatch, params):
    async def fake(question):
        return dict(params)
    monkeypatch.setattr(coord, "_classify", fake)


# ══════════════════════════════════════════════════════════════════════════════
# Which form fired — pure
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("msg", [
    "log my bench 100x5",
    "record squat 80kg x5",
    "add 3 sets of deadlift",
    "delete my deadlift goal",
    "update my last set — it was 12 reps not 10",
    "set a goal for 150 lbs on Lat Pulldown",
])
def test_explicit_write_verb_is_imperative(msg):
    assert _write_intent_form(msg) == WRITE_FORM_IMPERATIVE


@pytest.mark.parametrize("msg", [
    "I did chest and triceps today",
    "I did shrugs today and my grip gave out",
    "I did legs yesterday and it destroyed me",
    "I did bench this morning",
])
def test_verbless_narration_is_narration(msg):
    assert _write_intent_form(msg) == WRITE_FORM_NARRATION


@pytest.mark.parametrize("msg", [
    "should I log bench 100x5?",       # question phrasing beats both forms
    "can I add 3 sets of squats?",
    "how is my back progressing",
    "what's my bench PR",
    "",
])
def test_no_write_signal(msg):
    assert _write_intent_form(msg) is None


def test_narration_with_data_still_reads_as_narration():
    # It carries data, but no write VERB — so it is still the deferring form.
    # The classifier routes it operational via the prompt's session rule.
    assert _write_intent_form("I did 3x10 squats today") == WRITE_FORM_NARRATION


# ══════════════════════════════════════════════════════════════════════════════
# The entry boundary records the strength
# ══════════════════════════════════════════════════════════════════════════════

def test_entry_boundary_marks_imperative_as_binding(coord):
    out = coord._node_entry_boundary({"question": "log my bench press 100x5"})
    assert out["write_intent_hint"] is True
    assert out["write_intent_hard"] is True
    assert out["fallback_write"] is True
    assert out["params"] is None            # deferred to classify (Stage 3)


def test_entry_boundary_marks_narration_as_non_binding(coord):
    out = coord._node_entry_boundary({"question": "I did chest and triceps today"})
    assert out["write_intent_hint"] is True
    assert out["write_intent_hard"] is False
    assert out["fallback_write"] is True


def test_entry_boundary_no_write_signal_at_all(coord):
    out = coord._node_entry_boundary({"question": "how is my back progressing"})
    assert out["write_intent_hint"] is False
    assert out["write_intent_hard"] is False


def test_log_boundary_is_still_classify_free(coord):
    # Unchanged: the trusted /log prefix never spends a classify call.
    out = coord._node_entry_boundary({"question": "/log bench 100x5"})
    assert out["params"]["route"] == "operational"
    assert out["write_intent_hint"] is False


# ══════════════════════════════════════════════════════════════════════════════
# How far the override reaches
# ══════════════════════════════════════════════════════════════════════════════

def _route_of(coord, monkeypatch, question, hard, classify_params):
    _classify_returning(coord, monkeypatch, classify_params)
    state = {"question": question, "write_intent_hint": True,
             "write_intent_hard": hard}
    out = asyncio.run(coord._node_classify(state))
    return out["params"]


class TestNarrationDefers:
    def test_successful_classify_wins_over_narration(self, coord, monkeypatch):
        # THE ledger-E case: the prompt's rule is reachable again.
        params = _route_of(coord, monkeypatch,
                           "I did shrugs today and my grip gave out",
                           hard=False, classify_params=_params("analytical"))
        assert params["route"] == "analytical"

    def test_classify_saying_operational_is_honored_too(self, coord, monkeypatch):
        # The other direction: a session-to-log still logs.
        params = _route_of(coord, monkeypatch, "I did chest and triceps today",
                           hard=False, classify_params=_params("operational"))
        assert params["route"] == "operational"

    def test_classify_FAILURE_still_forces_operational(self, coord, monkeypatch):
        # The safety net. An errored call is no evidence — a write must never be
        # lost to one, so the regex verdict stands in.
        params = _route_of(coord, monkeypatch, "I did chest and triceps today",
                           hard=False,
                           classify_params=_params("analytical", parse_failed=True))
        assert params["route"] == "operational"
        assert "_parse_failed" not in params      # consumed, not left to short-circuit


class TestImperativeBinds:
    def test_write_verb_overrules_a_successful_classify(self, coord, monkeypatch):
        # Unchanged write-safety rule: the user named the action, so a classifier
        # that says "analytical" does not get to discard the write.
        params = _route_of(coord, monkeypatch, "delete my deadlift goal",
                           hard=True, classify_params=_params("analytical"))
        assert params["route"] == "operational"

    def test_write_verb_overrules_a_failed_classify(self, coord, monkeypatch):
        params = _route_of(coord, monkeypatch, "log my bench 100x5", hard=True,
                           classify_params=_params("analytical", parse_failed=True))
        assert params["route"] == "operational"
        assert "_parse_failed" not in params

    def test_decomposable_mixed_turn_is_never_collapsed(self, coord, monkeypatch):
        # The override exists to stop a write leaking analytical, NOT to stop
        # decomposition: a genuine mixed-lane turn still splits.
        reqs = [
            {"index": 0, "lane": "operational", "intent_text": "log bench 100x5"},
            {"index": 1, "lane": "analytical",  "intent_text": "how is my back"},
        ]
        params = _route_of(coord, monkeypatch,
                           "log bench 100x5 and how is my back going", hard=True,
                           classify_params=_params("operational", requests=reqs))
        assert params["route"] == "operational"
        assert len(params["requests"]) == 2


def test_no_hint_means_no_override_at_all(coord, monkeypatch):
    # A turn the regex never touched is left completely alone.
    _classify_returning(coord, monkeypatch, _params("analytical"))
    out = asyncio.run(coord._node_classify(
        {"question": "how is my back progressing", "write_intent_hint": False}))
    assert out["params"]["route"] == "analytical"
