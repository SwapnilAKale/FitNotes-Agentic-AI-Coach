"""
The routing gates, frozen as an explicit input -> output table.

WHY THIS EXISTS. The lane decision is written down twice — in English inside
_CLASSIFY_SYSTEM, and in Python inside the regex gates in front of it. Neither
copy knows the other exists, so they drifted, and the drift was invisible from
either side. Two measured examples at the time of writing:

    _is_write_intent("I did shrugs today and my grip gave out")  -> True
    while _CLASSIFY_SYSTEM line 414 declares that exact sentence ANALYTICAL

    the write guard's own question test said "am I overtraining" was NOT a
    question, while "Am I overtraining?" is a verbatim analytical example in
    the prompt  (the two rival detectors are now merged into _is_question)

Every gate below is a PURE FUNCTION of its input — no LLM, no DB, no mocks. So
the whole deterministic half of routing can be pinned exactly, and the prompt
rewrite that follows can be judged against something that does not move.

THIS TABLE IS A SAFETY NET, NOT A SPEC. It was written to pass against the
UNMODIFIED gates and committed in that state, precisely so it cannot be a
restatement of the behavior we were about to introduce. Rows that the cleanup
deliberately changes are quarantined in the CHANGED block at the bottom of each
section, each carrying its pre-change value and the ledger row that authorized
the flip. A row that moves without such a label is a regression.

Both directions everywhere: a gate that fires is paired with the near-miss that
must not, because "it works" for a guard means it also declines to work.
"""

import pytest

from langgraph.graph import END

from src.coordinator import (
    _filler_reply,
    _is_write_intent,
    _log_carry_unrelated,
    _split_log_tail,
)
from src.graph.coordinator_graph import _route_after_classify, _route_after_entry


# ══════════════════════════════════════════════════════════════════════════════
# _is_write_intent — the deterministic write pre-guard
#
# Precedence, from the guard's own docstring: question phrasing WINS. A false
# negative is recoverable (the classifier still sees it, and the operational
# confirmation gate still stands between it and the DB); a false positive
# strands a coaching question in a lane that cannot answer reads. So when torn,
# this guard is supposed to decline.
# ══════════════════════════════════════════════════════════════════════════════

_WRITE_FIRES = [
    # (A) goal-setting
    "set a goal for 150 lbs on Lat Pulldown",
    "set goal 100kg squat",
    # (B) write verb + quantity shorthand
    "log 3x10 shrugs at 60kg",
    "record squat 80kg x5",
    "add 3 sets of deadlift",
    "save bench 100x5",
    # (C) write verb + explicit data noun
    "log today's workout",
    "delete my deadlift goal",
    "remove my last set",
    "update my last set",
    "correct my last set",
    # (D) narration of a completed session that carries actual data
    "I did 3x10 squats today",
    "I did 5 sets of squats yesterday",
    "I did bench this morning 100x5",
]

_WRITE_DECLINES = [
    # Questions ABOUT writing — the write verb is present, the intent is not.
    "should I log bench 100x5?",
    "can I add 3 sets of squats?",
    "should I add weight to my squat",
    "how many sets should I log",
    "do I need to record warmup sets",
    "is it ok to add a set",
    # Reports of how a lift went. _CLASSIFY_SYSTEM L414 routes these analytical:
    # they describe training that already happened and want an explanation.
    "my grip gave out on shrugs",
    "my grip gave out on shrugs at 60kg",
    "I failed the last rep on 3 sets",
    "that felt heavy, 100x5 was brutal",
    "my squat stalled at 120kg",
    # Ordinary reads — no write verb at all.
    "am I overtraining",
    "how has my squat gone",
    "how is my back progressing",
    "what's my bench PR",
    "my squat?",
]


@pytest.mark.parametrize("msg", _WRITE_FIRES)
def test_write_intent_fires(msg):
    assert _is_write_intent(msg) is True


@pytest.mark.parametrize("msg", _WRITE_DECLINES)
def test_write_intent_declines(msg):
    assert _is_write_intent(msg) is False


def test_write_intent_empty_is_never_a_write():
    assert _is_write_intent("") is False
    assert _is_write_intent(None) is False


# ── OPEN — ledger row E, unresolved ───────────────────────────────────────────
# Form (D) matches "i did ... today|yesterday|..." with no data requirement, so
# bare narration of a COMPLAINT is caught as a write and — via the distrust
# override at coordinator.py:1695 — kept there, overruling _CLASSIFY_SYSTEM L414
# which declares these analytical.
#
# The obvious fix (require set/rep/weight data) also releases "I did chest and
# triceps today", which test_routing_step_c and test_routing_response_cluster
# both pin as a WRITE. Both sides were deliberate; the contradiction is a
# product decision, not a code one, and is pending.
#
# Recorded here as it behaves TODAY so the open question stays visible and the
# tree stays honest. Do not flip without settling the pin.
_LEDGER_E_BARE_NARRATION = [
    "I did shrugs today and my grip gave out",
    "I did legs yesterday and it destroyed me",
    "I did chest this morning and felt weak",
]

# Narration that DOES carry set/rep/weight data is a write under every proposed
# resolution — this half is settled and must not move.
_LEDGER_E_NARRATION_WITH_DATA = [
    "I did 5 sets of squats yesterday and it felt awful",
    "I did 100x5 bench today and my shoulder ached",
]


@pytest.mark.parametrize("msg", _LEDGER_E_BARE_NARRATION)
def test_ledger_e_bare_narration_currently_reads_as_a_write(msg):
    assert _is_write_intent(msg) is True


@pytest.mark.parametrize("msg", _LEDGER_E_NARRATION_WITH_DATA)
def test_ledger_e_narration_with_data_is_a_write(msg):
    assert _is_write_intent(msg) is True


# ══════════════════════════════════════════════════════════════════════════════
# _filler_reply — the pre-classify short-circuit
#
# Conservative by construction: exact match against a curated set after
# normalizing, length-gated. Anything carrying real content must fall through,
# including a real question hiding behind a polite prefix.
# ══════════════════════════════════════════════════════════════════════════════

_FILLER_THANKS_CASES = ["thanks", "thank you", "thanks!", "Thanks.", "appreciate it"]
_FILLER_GENERIC_CASES = [
    "hi", "hello", "yo", "good morning", "ok", "cool", "yes", "no",
    "perfect", "sounds good", "ok thanks", "", "   ",
]
_FILLER_FALLS_THROUGH = [
    "thanks, now how's my squat",        # filler prefix + a real question
    "hi, how is my bench",
    "thanks so much for all the detailed help",   # >20 chars, fails length gate
]


@pytest.mark.parametrize("msg", _FILLER_THANKS_CASES)
def test_filler_thanks_reply(msg):
    reply = _filler_reply(msg)
    assert reply is not None and "welcome" in reply


@pytest.mark.parametrize("msg", _FILLER_GENERIC_CASES)
def test_filler_generic_reply(msg):
    reply = _filler_reply(msg)
    assert reply is not None and "welcome" not in reply


@pytest.mark.parametrize("msg", _FILLER_FALLS_THROUGH)
def test_filler_declines_real_content(msg):
    assert _filler_reply(msg) is None


# ══════════════════════════════════════════════════════════════════════════════
# _split_log_tail — peeling an analytical tail off a /log turn
#
# A tail carrying set/rep/weight data is WORKOUT CONTENT and must never peel;
# if every sentence would peel, nothing peels at all.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text,head,split", [
    # peels: question mark
    ("bench 100x5, 3 sets. How is my back progressing?",
     "bench 100x5, 3 sets.", True),
    # peels: connector + interrogative lead, no question mark
    ("bench 100x5, 3 sets. Also, how is my back progressing",
     "bench 100x5, 3 sets.", True),
    ("bench 100x5. Am I overtraining", "bench 100x5.", True),
    ("bench 100x5. btw what's my squat PR", "bench 100x5.", True),
    # does NOT peel: no tail at all
    ("bench 100x5, 3 sets", "bench 100x5, 3 sets", False),
    # does NOT peel: tail carries a quantity -> workout content
    ("squat 3 sets of 5. also did 100x5 on bench",
     "squat 3 sets of 5. also did 100x5 on bench", False),
    # does NOT peel: not interrogative
    ("bench 100x5, 3 sets. It felt heavy",
     "bench 100x5, 3 sets. It felt heavy", False),
    # does NOT peel: head would be empty
    ("how is my back progressing?", "how is my back progressing?", False),
])
def test_split_log_tail_both_directions(text, head, split):
    assert _split_log_tail(text) == (head, split)


# ══════════════════════════════════════════════════════════════════════════════
# _log_carry_unrelated — clear-without-consume test for the single-turn carry
#
# True  = this message does NOT answer the pending logging clarification, so the
#         carry clears without being consumed.
# False = it plausibly answers it, so it joins the /log flow.
# ══════════════════════════════════════════════════════════════════════════════

_CARRY_JOINS_FLOW = [
    "yesterday",              # date clarification
    "3 sets of 12",           # sets/reps clarification
    "the dumbbell one",       # name disambiguation
    "100kg",
    "bench press",
    "the one with dumbbells",
]

_CARRY_CLEARS = [
    "how is my squat progressing?",   # a question is not a clarification answer
    "what's my PR",
    "thanks",                         # filler is abandonment
    "ok",
]


@pytest.mark.parametrize("msg", _CARRY_JOINS_FLOW)
def test_carry_is_consumed_by_a_plausible_answer(msg):
    assert _log_carry_unrelated(msg) is False


@pytest.mark.parametrize("msg", _CARRY_CLEARS)
def test_carry_cleared_by_question_or_filler(msg):
    assert _log_carry_unrelated(msg) is True


# ── CHANGED by ledger row F ───────────────────────────────────────────────────
# Two question detectors existed; the weaker one was wired here, and it returned
# False for an unpunctuated question — so an unmistakable analytical question was
# swallowed into the logging flow instead of clearing the carry. Note the first
# case is the SAME sentence as in _CARRY_CLEARS above, minus the "?": the
# punctuation alone used to decide. WAS False for both; now True.
_LEDGER_F_UNPUNCTUATED_QUESTION = [
    "how is my squat progressing",
    "am I overtraining",
]


@pytest.mark.parametrize("msg", _LEDGER_F_UNPUNCTUATED_QUESTION)
def test_ledger_f_unpunctuated_question_clears_the_carry(msg):
    assert _log_carry_unrelated(msg) is True


# ══════════════════════════════════════════════════════════════════════════════
# The graph's conditional edges — pure functions of state, no runtime needed.
# ══════════════════════════════════════════════════════════════════════════════

def _mixed(*lanes):
    return {"params": {"requests": [{"lane": l} for l in lanes]}}


class TestRouteAfterEntry:
    def test_filler_result_short_circuits_to_end(self):
        assert _route_after_entry({"result": {"answer": "hi"}}) == END

    def test_no_params_goes_to_classify(self):
        assert _route_after_entry({}) == "classify"
        assert _route_after_entry({"params": None}) == "classify"

    def test_preseeded_operational_skips_classify(self):
        assert _route_after_entry(
            {"params": {"route": "operational"}}) == "dispatch_operational"

    def test_preseeded_analytical_resume_dispatches_analytical(self):
        assert _route_after_entry(
            {"params": {"route": "analytical"}}) == "dispatch_analytical"

    def test_mixed_lane_resume_re_enters_decomposed(self):
        # Dispatching a mixed-lane resume by its flat route would silently drop
        # the sibling chunk — the reason this test exists in both edge funcs.
        assert _route_after_entry(
            _mixed("analytical", "operational")) == "dispatch_decomposed"

    def test_uniform_lane_multi_chunk_is_not_decomposed(self):
        state = _mixed("analytical", "analytical")
        state["params"]["route"] = "analytical"
        assert _route_after_entry(state) == "dispatch_analytical"

    def test_single_chunk_is_not_decomposed(self):
        state = _mixed("operational")
        state["params"]["route"] = "operational"
        assert _route_after_entry(state) == "dispatch_operational"

    def test_result_wins_over_params(self):
        assert _route_after_entry(
            {"result": {"answer": "hi"}, "params": {"route": "operational"}}) == END


class TestRouteAfterClassify:
    @pytest.mark.parametrize("route,expected", [
        ("analytical",   "dispatch_analytical"),
        ("operational",  "dispatch_operational"),
        ("recall",       "dispatch_recall"),
        ("out_of_scope", "out_of_scope"),
    ])
    def test_each_lane(self, route, expected):
        assert _route_after_classify({"params": {"route": route}}) == expected

    def test_missing_route_defaults_analytical(self):
        assert _route_after_classify({"params": {}}) == "dispatch_analytical"
        assert _route_after_classify({}) == "dispatch_analytical"

    def test_parse_failure_wins_over_everything(self):
        # Checked before the mixed-lane test: garbage never builds a package.
        state = _mixed("analytical", "operational")
        state["params"]["_parse_failed"] = True
        assert _route_after_classify(state) == "unparseable"

    def test_mixed_lane_multi_chunk_decomposes(self):
        assert _route_after_classify(
            _mixed("analytical", "operational")) == "dispatch_decomposed"

    def test_uniform_lane_multi_chunk_stays_one_run(self):
        # All-analytical multi-chunk is deliberately ONE pipeline, not two.
        state = _mixed("analytical", "analytical")
        state["params"]["route"] = "analytical"
        assert _route_after_classify(state) == "dispatch_analytical"

    def test_single_chunk_never_decomposes(self):
        state = _mixed("operational")
        state["params"]["route"] = "operational"
        assert _route_after_classify(state) == "dispatch_operational"
