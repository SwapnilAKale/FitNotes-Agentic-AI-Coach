"""
Can the user be told a write happened when it did not?

WHY THIS EXISTS. A live write check ran the real flow against the real database:
a workout was staged, confirmed, and written; "delete that set" then produced a
`discard_staged_writes` call and the reply "has been removed from your staged
session, and the entire staged workout has been discarded" — while the row sat
untouched in training_log. Nothing was lost. The data stayed while the user was
told it was gone, which is worse in the way that matters: you stop looking.

Three layers failed in sequence, and this file pins all three.

  Row 1  the agent is never told its staged batch was committed, so it still
         believes the set is pending — reaching for discard is correct reasoning
         from a false premise, not tool confusion.
  Row 2  discard_staged_writes returns {"discarded": true} whether it dropped a
         batch or found an empty slot, so the agent cannot tell the difference.
  Row 3  the success-claim gate catches false "saved" claims but almost no other
         write verb.

THIS FILE IS A SAFETY NET, NOT A SPEC. It was written to pass against UNMODIFIED
code, so it cannot be a restatement of the behavior about to be introduced. Rows
the arc deliberately changes sit in PRE-CHANGE blocks carrying their current
value. A row that moves without such a label is a regression.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")

from src import coordinator as coordinator_mod                      # noqa: E402
from src.coordinator import (                                       # noqa: E402
    MSG_NO_WRITE_OCCURRED,
    MSG_STAGED_NOT_SAVED,
    Coordinator,
    _WRITE_COMPLETION_CLAIM_RE,
    _WRITE_SUCCESS_CLAIM_RE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Driving the REAL gate — a fake agent, the genuine _run_operational
# ══════════════════════════════════════════════════════════════════════════════

def _coord(monkeypatch, answer, *, db_write_effect=False,
           staging_reached_confirm=False, staged_this_turn=False,
           write_attempted=False, write_block_reason=None):
    """Coordinator whose agent returns a chosen answer + structural facts.

    The gate under test lives inside _run_operational, so it is exercised for
    real rather than re-implemented here.
    """
    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())

    class _FakeAgent:
        async def answer(self, question, resume_messages=None):
            return {
                "answer": answer,
                "db_write_effect": db_write_effect,
                "staging_reached_confirm": staging_reached_confirm,
                "staged_this_turn": staged_this_turn,
                "write_attempted": write_attempted,
                "write_block_reason": write_block_reason,
            }

    c = Coordinator(agent_session=_FakeAgent())
    import src.checkpoint as ckpt
    monkeypatch.setattr(ckpt, "load_checkpoint", lambda: None)
    return c


def _run(coord, q="log my bench", **kw):
    return asyncio.run(coord._run_operational(q, **kw))


# ── The gate works where it was built to work ─────────────────────────────────

def test_unbacked_logged_claim_is_replaced(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.")
    assert _run(c, log_boundary=True) == MSG_NO_WRITE_OCCURRED


def test_staged_only_claim_becomes_the_truthful_staged_message(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.", staged_this_turn=True)
    assert _run(c, log_boundary=True).startswith(MSG_STAGED_NOT_SAVED[:40])


def test_a_backed_claim_is_left_alone(monkeypatch):
    c = _coord(monkeypatch, "Your workout has been logged.", db_write_effect=True)
    assert _run(c, log_boundary=True) == "Your workout has been logged."


def test_reaching_the_execute_gate_also_backs_the_claim(monkeypatch):
    c = _coord(monkeypatch, "✅ Your goal has been saved to your database.",
               staging_reached_confirm=True)
    assert "saved to your database" in _run(c)


def test_a_question_is_never_rewritten(monkeypatch):
    # No claim is being made — the flags alone must not trigger the gate.
    q = "Which date should I log that under?"
    c = _coord(monkeypatch, q)
    assert _run(c, log_boundary=True) == q


# ── The property the Row-3 fix must not break ─────────────────────────────────

RESEARCH_PROSE = [
    "The study removed participants who missed sessions, then re-ran the analysis.",
    "Two trials were deleted from the meta-analysis for poor blinding.",
    "The authors updated their conclusions in a later erratum.",
]


@pytest.mark.parametrize("prose", RESEARCH_PROSE)
def test_research_prose_is_never_rewritten(prose):
    """THE FALSE POSITIVE TO AVOID. _run_operational also serves research
    answers. Broadening the claim regexes without a structural scope would turn
    'the study removed participants' into a database warning. Asserted at the
    DETECTOR level so it holds regardless of how scoping is later wired."""
    assert not _WRITE_SUCCESS_CLAIM_RE.search(prose)


# ══════════════════════════════════════════════════════════════════════════════
# Row 3 — what the gate currently detects, verb by verb
#
# The loose detector only runs on logging-flow turns (log_boundary /
# fallback_write). On goal and set-edit turns ONLY the narrow one applies, which
# is why the goal/set rows below are effectively unguarded today.
# ══════════════════════════════════════════════════════════════════════════════

def _detected(claim: str, *, log_flow: bool) -> bool:
    return bool(_WRITE_SUCCESS_CLAIM_RE.search(claim)) or (
        log_flow and bool(_WRITE_COMPLETION_CLAIM_RE.search(claim)))


_CAUGHT_TODAY_ON_LOG_FLOWS = [
    "Your workout has been logged.",
    "I've logged your workout.",
    "3 sets were added to your database.",
    "Your bodyweight has been recorded.",
    "Successfully deleted the set.",
]


@pytest.mark.parametrize("claim", _CAUGHT_TODAY_ON_LOG_FLOWS)
def test_logging_claims_are_detected(claim):
    assert _detected(claim, log_flow=True)


# ── CHANGED by Row 3 ──────────────────────────────────────────────────────────
# These completion claims used to reach the user unchecked: the loose detector
# never ran on goal or set-edit turns, so only the narrow regex applied and
# caught one of thirteen. Both the verb list and the SCOPE are widened — the
# scope by a structural fact (a write tool was actually called this turn), which
# is what keeps research answers out.
#
# It matters exactly when the write did NOT happen — cancelled at the
# confirmation prompt, failed, or never attempted — and the agent says it did.

_WRITE_CLAIMS = [
    ("goal set",    "Your goal is now set."),
    ("goal set",    "I've created that goal for you."),
    ("goal update", "Your goal has been updated."),
    ("goal update", "I've updated your goal."),
    ("goal delete", "Your goal has been deleted."),
    ("goal delete", "I've removed that goal."),
    ("set update",  "The set has been updated to 12 reps."),
    ("set update",  "I've corrected that set."),
    ("set update",  "That set has been fixed."),
    ("set delete",  "The set has been deleted from your database."),
    ("set delete",  "I've deleted that set."),
    ("set delete",  "That set has been removed."),
]


@pytest.mark.parametrize("flow,claim", _WRITE_CLAIMS)
def test_row3_write_claims_are_detected_on_a_write_turn(flow, claim):
    assert _detected(claim, log_flow=True)


@pytest.mark.parametrize("flow,claim", _WRITE_CLAIMS)
def test_row3_the_same_claims_stay_inert_without_a_write(flow, claim):
    """The other direction, and the reason the scope is structural: identical
    words on a turn that called no write tool must not be rewritten. Only the
    narrow regex applies there, exactly as before."""
    assert _detected(claim, log_flow=False) == bool(
        _WRITE_SUCCESS_CLAIM_RE.search(claim))


def test_row3_the_observed_false_claim_is_now_detected():
    # The exact sentence the live check produced while the row remained.
    observed = ("The Flat Barbell Bench Press set of 101 lbs x 7 reps on "
                "2026-08-08 has been removed from your staged session, and the "
                "entire staged workout has been discarded.")
    assert _detected(observed, log_flow=True)


def test_row3_end_to_end_a_delete_claim_without_a_write_is_replaced(monkeypatch):
    """Through the real gate: a delete was attempted, nothing was written, and
    the agent claims removal. The user must not be told the row is gone."""
    c = _coord(monkeypatch, "I've deleted that set.", write_attempted=True)
    assert _run(c, "delete my bench set") == MSG_NO_WRITE_OCCURRED


def test_row3_a_backed_delete_claim_survives(monkeypatch):
    c = _coord(monkeypatch, "I've deleted that set.",
               write_attempted=True, db_write_effect=True)
    assert _run(c, "delete my bench set") == "I've deleted that set."


def test_row3_research_answer_is_untouched_without_a_write(monkeypatch):
    """THE FALSE POSITIVE, end to end. Same code path, research answer, no write
    tool called — the structural fact keeps the prose intact."""
    prose = "The study removed participants who missed sessions."
    c = _coord(monkeypatch, prose)
    assert _run(c, "what does science say about adherence?") == prose


def test_row3_no_write_message_serves_every_write_kind():
    """One message for all of them. Splitting it by flow was attempted and
    abandoned: the gate fires hardest when the agent called NO tool, and then
    nothing distinguishes a logging turn from a goal turn — the canonical live
    failure is a bare "Yes thats correct", which sets neither flow flag."""
    assert "saved or changed" in MSG_NO_WRITE_OCCURRED       # covers both
    assert "/log" in MSG_NO_WRITE_OCCURRED                   # tip survives
    assert "your logging request" not in MSG_NO_WRITE_OCCURRED


def test_row3_write_attempted_is_set_on_the_call_not_the_result():
    """A write that errors or returns garbage is still a write turn — otherwise
    the worst case (tool blew up, agent claims success anyway) is the one case
    the gate skips. Asserted by ORDER: the flag is set before the result parse,
    so a parse failure cannot skip it."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    set_at = src.index("self._turn_write_attempted = True")
    parse_at = src.index("parsed = json.loads(result)")
    assert set_at < parse_at, "write_attempted is set inside/after the parse"


# ══════════════════════════════════════════════════════════════════════════════
# A DENIAL IS NOT A CLAIM — found live, on the first prompt of a live check
#
# The agent correctly refused an edit: "I cannot delete that set because it was
# recorded in your FitNotes app." The completion detector matched subject+was+
# recorded, so the honest explanation was replaced by the generic no-write
# warning — the user learned nothing was written but never why.
#
# Two things came out of it. The refusal message no longer uses a completion
# verb ("comes from", not "was recorded"), removing the trigger at source. And
# the detector now skips denied clauses — CLAUSE, not sentence: the first
# attempt split on sentences and scored "I couldn't change the weight, but I
# logged the set" as no-claim, which would have swallowed exactly the lie this
# gate exists to stop.
# ══════════════════════════════════════════════════════════════════════════════

from src.coordinator import _claims_a_write_happened            # noqa: E402


@pytest.mark.parametrize("claim", [
    # Both verbatim from the live check, after the lock had REFUSED the write.
    "I have updated your Decline Barbell Bench Press set on 2026-07-22 to 105 lbs x 5.",
    "I have successfully deleted your Decline Barbell Bench Press set of "
    "100 lbs x 5 from July 22, 2026.",
    "Your workout has been logged.",
    # the tools' own success strings, in the order THEY phrase them
    "Set deleted successfully.",
    "Goal saved successfully.",
    # a denial must not launder a real claim sharing the sentence
    "I couldn't change the weight, but I logged the set.",
    "I can't edit that one; however your goal has been updated.",
])
def test_a_claim_of_success_is_detected(claim):
    assert _claims_a_write_happened(claim, write_flow=True) is True


@pytest.mark.parametrize("denial", [
    # the exact reply the live check produced, and the reworded one
    "I cannot delete that set because it was recorded in your FitNotes app.",
    "I cannot delete that set because it comes from your FitNotes app.",
    "That set comes from FitNotes, so it can't be changed here.",
    "Nothing was saved.",
    "That was not saved to your database.",
    "I wasn't able to delete it, so nothing was removed.",
])
def test_a_denial_is_not_a_claim(denial):
    assert _claims_a_write_happened(denial, write_flow=True) is False


def test_research_prose_stays_inert_without_a_write():
    prose = "The study removed participants who missed sessions."
    assert _claims_a_write_happened(prose, write_flow=False) is False


# ══════════════════════════════════════════════════════════════════════════════
# WHY a write was blocked must reach the user without going through the model
#
# The app-data lock and the integrity guard each produce a user-ready sentence
# explaining the block. Live, the model relayed it correctly once out of three
# turns: twice it claimed outright that the refused edit had succeeded, and the
# gate — correctly — replaced the lie with a generic "nothing was written". So
# the user was told three times that nothing happened and never once why, while
# the explanation existed the whole time.
#
# The reason is now a structural fact of the turn, so the substitution no longer
# depends on the model's prose being honest or even coherent.
# ══════════════════════════════════════════════════════════════════════════════

_LOCK_REASON = ("That set comes from FitNotes, so it can't be changed here — "
                "the app is the source of truth for what you logged.")


def test_the_block_reason_replaces_the_generic_warning(monkeypatch):
    c = _coord(monkeypatch,
               "I have successfully deleted your Decline Barbell Bench Press set.",
               write_attempted=True, write_block_reason=_LOCK_REASON)
    out = _run(c, "delete my bench set")
    assert _LOCK_REASON in out
    assert out != MSG_NO_WRITE_OCCURRED
    # the lie itself never reaches the user
    assert "successfully deleted" not in out


def test_without_a_reason_the_generic_warning_is_unchanged(monkeypatch):
    """The other direction — this path must not move for ordinary turns."""
    c = _coord(monkeypatch,
               "I have successfully deleted your Decline Barbell Bench Press set.",
               write_attempted=True)
    assert _run(c, "delete my bench set") == MSG_NO_WRITE_OCCURRED


def test_an_honest_answer_is_never_hijacked_by_the_reason(monkeypatch):
    """NEGATIVE: a reason is recorded, but the answer claims nothing. The gate
    does not fire, so the agent's own words stand — we replace bad messages, not
    good ones."""
    honest = "I can't change that one — it's yours from FitNotes. Want to log a new set instead?"
    c = _coord(monkeypatch, honest,
               write_attempted=True, write_block_reason=_LOCK_REASON)
    assert _run(c, "delete my bench set") == honest


def test_a_staged_turn_keeps_the_staged_message_not_the_reason(monkeypatch):
    """Scope guard: if the turn also staged something, 'not saved yet' is the
    more accurate sentence than a block reason from some other tool call."""
    c = _coord(monkeypatch, "Your workout has been logged.",
               staged_this_turn=True, write_block_reason=_LOCK_REASON)
    out = _run(c, log_boundary=True)
    assert out.startswith(MSG_STAGED_NOT_SAVED[:40])
    assert _LOCK_REASON not in out


# ── the /log tip, exactly once ────────────────────────────────────────────────

def test_the_log_tip_is_not_printed_twice(monkeypatch):
    """MSG_NO_WRITE_OCCURRED already ends '...starting with /log is the most
    reliable route', and the fallback nudge was appended on top of it."""
    c = _coord(monkeypatch, "Your workout has been logged.", write_attempted=True)
    out = _run(c, "bench 100 x 5", fallback_write=True)
    assert out.count("/log") == 1


def test_the_log_tip_is_not_offered_after_an_app_lock_refusal(monkeypatch):
    """Worse than duplicated — /log cannot edit a row FitNotes owns."""
    c = _coord(monkeypatch, "I have updated your set.",
               write_attempted=True, write_block_reason=_LOCK_REASON)
    out = _run(c, "change my bench set", fallback_write=True)
    assert "/log" not in out


def test_no_log_tip_when_the_agent_explains_the_refusal_itself(monkeypatch):
    """The live case, and the reason this keys off the block FACT rather than
    off the gate firing. Here the agent refused honestly, so the gate correctly
    left the answer alone — and the wrong tip was appended to it anyway."""
    honest = ("That set was imported from your FitNotes data, so it is read-only "
              "here. You'll need to make that change directly in the FitNotes app.")
    c = _coord(monkeypatch, honest,
               write_attempted=True, write_block_reason=_LOCK_REASON)
    out = _run(c, "change my bench set", fallback_write=True)
    assert out == honest


def test_an_ordinary_fallback_write_still_gets_the_tip(monkeypatch):
    """The other direction — the nudge is not lost for the case it was built
    for: a write inferred by regex, with no replacement in play."""
    c = _coord(monkeypatch, "Staged your bench set for confirmation.",
               write_attempted=True, db_write_effect=True)
    out = _run(c, "bench 100 x 5", fallback_write=True)
    assert out.count("/log") == 1
    assert out.rstrip().endswith("faster and more reliable.")


def _canned_messages():
    """Every fixed sentence the user can be shown about a write."""
    import json as _json

    import mcp_servers.combined_server as cs
    from src.coordinator import MSG_NO_WRITE_OCCURRED, MSG_STAGED_NOT_SAVED
    from src.db import MSG_INTEGRITY_REJECTED
    return [
        ("MSG_NO_WRITE_OCCURRED", MSG_NO_WRITE_OCCURRED),
        ("MSG_STAGED_NOT_SAVED", MSG_STAGED_NOT_SAVED),
        ("MSG_INTEGRITY_REJECTED", MSG_INTEGRITY_REJECTED),
        ("app-lock refusal (set)", _json.loads(cs._refuse_app_row("set"))["error"]),
        ("app-lock refusal (goal)", _json.loads(cs._refuse_app_row("goal"))["error"]),
    ]


@pytest.mark.parametrize("name,message",
                         _canned_messages(),
                         ids=[n for n, _ in _canned_messages()])
def test_no_canned_message_trips_our_own_claim_gate(name, message):
    """THE INVARIANT, not two patched instances.

    Twice a message we wrote was read by our own detector as a claim that a
    write happened. The refusal said "entries I ADDED myself" (first-person
    completion) and MSG_STAGED_NOT_SAVED said "before anything IS WRITTEN to
    your database" (subject-is-verb). The first replaced an honest explanation
    with a generic warning in front of the user; the second was latent only
    because a replacement is never re-checked.

    Any canned sentence can end up back through the gate — relayed by the model,
    substituted, or carried in history. So none of them may trip it.
    """
    assert _claims_a_write_happened(message, write_flow=True) is False, (
        f"{name} reads as a write-success claim: {message!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — the discard tool's report
# ══════════════════════════════════════════════════════════════════════════════

def _server_src() -> str:
    return (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")


def test_every_host_discard_is_fire_and_forget():
    """Why the payload is safe to change: no caller reads it. Pinned so a future
    caller that starts parsing it is caught here rather than in production."""
    for host in ("cli.py", "server.py"):
        for line in (_ROOT / host).read_text(encoding="utf-8").splitlines():
            if "discard_staged_writes" in line and "call_tool" in line:
                assert "=" not in line.split("call_tool")[0].split("await")[-1], \
                    f"{host}: discard return is being captured: {line.strip()}"


# ── CHANGED by Row 2 ──────────────────────────────────────────────────────────
# The tool USED to answer identically whether it dropped a batch or found an
# empty slot, so an agent calling it on nothing was told "discarded" and passed
# that on. It now reports which happened — and in the empty case names the tools
# that DO delete saved data, so the tool result corrects the mistake instead of
# confirming it. Exercised for real, not asserted from source text.

def _discard_sync():
    import importlib
    m = importlib.import_module("mcp_servers.combined_server")
    return m


def test_row2_discard_reports_that_it_dropped_something():
    m = _discard_sync()
    m._staged_writes.clear()
    m._staged_writes.setdefault("workout", []).append({"dummy": True})
    out = json.loads(m._discard_staged_writes_sync())
    assert out["discarded"] is True and out["had_pending"] is True
    assert not m._staged_writes                      # really cleared


def test_row2_discard_reports_an_empty_slot_honestly():
    m = _discard_sync()
    m._staged_writes.clear()
    out = json.loads(m._discard_staged_writes_sync())
    assert out["discarded"] is False and out["had_pending"] is False


def test_row2_empty_discard_points_at_the_tools_that_delete_saved_data():
    # The observed failure was the agent calling discard for a SAVED row and
    # reporting removal. The empty-slot result now says the opposite outright.
    m = _discard_sync()
    m._staged_writes.clear()
    msg = json.loads(m._discard_staged_writes_sync())["message"]
    assert "delete_workout_set" in msg and "delete_goal" in msg
    assert "cannot remove data already" in msg


def test_row2_tool_description_declares_the_new_contract():
    src = _server_src()
    assert "had_pending" in src
    i = src.index('name="discard_staged_writes"')
    desc = src[i:i + 900]
    assert "had_pending=false" in desc
    assert "delete_workout_set" in desc


# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — nobody tells the agent the write landed
# ══════════════════════════════════════════════════════════════════════════════

def test_agent_history_is_what_the_model_actually_sees():
    """The seam Row 1 must write into: _op_prepare builds the model's context
    from _conversation_history, so anything appended there reaches the model."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    prep = src[src.index("def _op_prepare"):]
    prep = prep[:prep.index("def _op_agent_step")]
    assert "_conversation_history" in prep


# ── CHANGED by Row 1 ──────────────────────────────────────────────────────────
# After the host committed a staged batch the agent was never informed, so its
# last known state stayed "staged, awaiting confirmation". Both hosts now tell
# it — the same seam applied to both interfaces, because a fix landing on one
# and not the other is how these gaps survive.

def _agent_session():
    from src.agent import AgentSession
    return AgentSession("data/unused.fitnotes")


def test_row1_note_reaches_the_model_context():
    """It must land in _conversation_history — that is what _op_prepare feeds
    the model. Anywhere else (e.g. chat_history) the agent never sees it."""
    from src.agent import HOST_WRITE_NOTE_MARK
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 3 sets across 1 exercise(s).")
    flat = [m for ex in s._conversation_history for m in ex]
    assert len(flat) == 1
    # CHANGED from "assistant": landing in the context turned out not to be
    # enough. As the agent's OWN prior speech the note lost to its own earlier
    # "staged, awaiting confirmation" reasoning, and it went on telling the user
    # the set was unsaved (live, twice). A user-role note is an external fact it
    # has to account for — the shape _op_finalize_cancelled already uses.
    assert flat[0]["role"] == "user"
    assert HOST_WRITE_NOTE_MARK in flat[0]["content"]
    assert "3 sets across 1 exercise(s)" in flat[0]["content"]


def test_row1_empty_outcome_records_nothing():
    s = _agent_session()
    s.note_host_write("")
    s.note_host_write(None)
    assert s._conversation_history == []


def test_row1_note_is_skipped_by_memory_extraction():
    """The note belongs in the model's context but must never be mined as a
    fact about the user — "3 sets across 1 exercise" is plumbing, not a
    preference."""
    from src.agent import HOST_WRITE_NOTE_MARK
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    body = src[src.index("async def _auto_extract_memories"):]
    body = body[:body.index("conversation_text +=") + 200]
    assert "HOST_WRITE_NOTE_MARK" in body
    assert "continue" in body


@pytest.mark.parametrize("host", ["cli.py", "server.py"])
def test_row1_both_hosts_notify_after_a_successful_execute(host):
    text = (_ROOT / host).read_text(encoding="utf-8")
    assert "execute_staged_workout" in text
    assert "note_host_write" in text, f"{host} executes but never tells the agent"


def test_row1_every_successful_execute_branch_notifies():
    """Counted, not spot-checked: each host branch that flips _staged_active on
    a successful execute must also notify. One unpatched branch reproduces the
    whole bug."""
    # Counts CALLS, not the bare name: a comment mentioning note_host_write is
    # not a notification, and counting the substring made an explanatory comment
    # look like a new branch.
    call = re.compile(r"\.note_host_write\s*\(")
    # server.py went 1 -> 2 when the SIBLING staged writes (goal / set edit /
    # comment) moved to a server-driven execute. They previously relied on the
    # agent to commit them, and live it simply did not — it "verified" instead
    # and reported a goal deleted that was still in the database. Now that the
    # server writes them, it owes the agent the same note the workout path does.
    for host, expected in (("cli.py", 2), ("server.py", 2)):
        text = (_ROOT / host).read_text(encoding="utf-8")
        assert len(call.findall(text)) == expected, host
        assert text.count("_staged_active = False") >= expected, host


def test_row1_record_external_exchange_still_has_one_caller():
    # The new seam is deliberately separate: record_external_exchange feeds
    # memory extraction, which host notes must NOT enter.
    call = re.compile(r"\.record_external_exchange\s*\(")
    hits = []
    for p in list((_ROOT / "src").rglob("*.py")) + [_ROOT / "cli.py", _ROOT / "server.py"]:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if call.search(line):
                hits.append(f"{p.name}:{i}")
    assert len(hits) == 1, f"expected the analytical caller only, got {hits}"
    assert "coordinator" in hits[0]


# ══════════════════════════════════════════════════════════════════════════════
# The reason has to survive the trip from tool result to Coordinator
#
# Three finalize nodes report the per-turn write facts, and they used to repeat
# the dict verbatim. A fact added to two of the three would go missing depending
# only on how the turn happened to end — the hardest kind of bug to see. They
# now share one definition, and this pins both the plumbing and that seam.
# ══════════════════════════════════════════════════════════════════════════════

def _bare_session():
    """An AgentSession with only the per-turn attributes set — no MCP, no
    Gemini, no I/O. Enough to drive the real tool-result parsing."""
    from src.agent import AgentSession
    s = AgentSession.__new__(AgentSession)
    s.confirmation_handler = None
    s._staged_active = False
    s._turn_write_effect = False
    s._turn_staged = False
    s._turn_write_attempted = False
    s._turn_write_block_reason = None
    return s


def _drive_tool(session, tool_name, payload):
    """Run one tool call through the real _op_exec_tools node."""
    async def _call_tool(name, args):
        return json.dumps(payload)
    session.call_tool = _call_tool

    state = {
        "messages": [{"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": tool_name, "arguments": "{}"}}]}],
        "execute_attempted": False,
        "tool_calls_made": 0,
        "iteration": 0,
    }
    cache = SimpleNamespace(gemini_contents=[])
    asyncio.run(session._op_exec_tools(state, cache))
    return state


@pytest.mark.parametrize("payload,expected", [
    # the app-data lock
    ({"refused": "app_data_locked", "error": "That set comes from FitNotes."},
     "That set comes from FitNotes."),
    # the integrity guard — same class, caught before it reached a live phase
    ({"success": False, "verified": False, "integrity_rejected": True,
      "reason": "unexpected_rows", "delta": {},
      "message": "That wasn't saved: the database didn't change the way it "
                 "should have, so it was rolled back."},
     "That wasn't saved: the database didn't change the way it should have, "
     "so it was rolled back."),
])
def test_a_blocked_write_records_its_reason(payload, expected):
    s = _bare_session()
    _drive_tool(s, "delete_workout_set", payload)
    assert s._turn_write_block_reason == expected


def test_a_normal_write_records_no_reason():
    """The other direction — nothing to explain, nothing recorded."""
    s = _bare_session()
    _drive_tool(s, "log_workout", {"staged_key": "log_workout", "staged": True})
    assert s._turn_write_block_reason is None
    assert s._turn_staged is True


def test_the_first_reason_of_the_turn_wins():
    s = _bare_session()
    _drive_tool(s, "delete_workout_set",
                {"refused": "app_data_locked", "error": "first reason"})
    _drive_tool(s, "update_workout_set",
                {"refused": "app_data_locked", "error": "second reason"})
    assert s._turn_write_block_reason == "first reason"


def test_the_reason_does_not_leak_into_the_next_turn():
    """Per-turn state, cleared at the single entry point — never by a branch."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    entry = src[src.index("self._turn_write_effect = False"):]
    entry = entry[:entry.index("turn_id = new_turn_id()")]
    assert "self._turn_write_block_reason = None" in entry, (
        "the reason must be reset beside the other per-turn write flags")


def test_all_three_finalize_nodes_report_the_same_facts():
    """The seam. None of them may build the fact dict inline again."""
    src = (_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    for node in ("_op_finalize_answer", "_op_finalize_cancelled",
                 "_op_finalize_max_iter"):
        body = src[src.index(f"def {node}"):]
        body = body[:body.index("\n    def ", 1)] if "\n    def " in body[1:] else body
        assert "self._turn_write_facts(state)" in body, f"{node} bypasses the helper"
        assert "\"db_write_effect\":" not in body, f"{node} rebuilds the facts inline"


def test_the_facts_helper_carries_the_reason():
    s = _bare_session()
    s._turn_write_block_reason = "because FitNotes owns it"
    facts = s._turn_write_facts({"execute_attempted": False})
    assert facts["write_block_reason"] == "because FitNotes owns it"
    assert set(facts) == {"staging_reached_confirm", "db_write_effect",
                          "staged_this_turn", "write_attempted",
                          "write_block_reason"}


# ── CHANGED again by Row 1 (round two) ────────────────────────────────────────
# Landing the note in the model's context was not enough. Live, twice, the agent
# still answered "What did you just save?" with:
#
#   "I am not sure which number you mean, as the Flat Barbell Bench Press entry
#    is currently staged and has not been saved to your database yet. Could you
#    please clarify what you are asking about?"
#
# Two defects in one sentence. The STATUS was wrong (covered by the role change
# above), and there was NOTHING TO ANSWER WITH: the outcome message says "1 sets
# across 1 exercise(s)" and names no exercise, weight, rep or date. So the agent
# asked the user to clarify an unambiguous question — an extra round trip, and
# it reads as "nothing was saved".
#
# A test cannot assert the model phrases an answer well. It CAN assert the agent
# was handed everything needed to give one.

_PREVIEW = ("Staged workout — 2026-09-10\n\nFlat Barbell Bench Press\n"
            "  Set 1: 102 lbs × 6 reps")


def test_row1_note_states_saved_not_pending():
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 1 sets across 1 exercise(s).",
                      _PREVIEW)
    body = s._conversation_history[0][0]["content"].lower()
    assert "saved" in body
    assert "not pending" in body or "staging area is now empty" in body


def test_row1_note_carries_the_concrete_set():
    """THE precondition for a direct answer — without these the agent has
    nothing to name and falls back to asking the user."""
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 1 sets across 1 exercise(s).",
                      _PREVIEW)
    body = s._conversation_history[0][0]["content"]
    for token in ("Flat Barbell Bench Press", "102", "6", "2026-09-10"):
        assert token in body, f"cannot answer 'what did you save' — no {token!r}"


def test_row1_note_survives_a_missing_preview():
    """Other direction: a preview read can fail, and degrading must never cost
    the status note itself — a silent write is worse than a vague one."""
    from src.agent import HOST_WRITE_NOTE_MARK
    s = _agent_session()
    s.note_host_write("Workout saved and verified.", "")
    body = s._conversation_history[0][0]["content"]
    assert HOST_WRITE_NOTE_MARK in body
    assert "Workout saved and verified." in body
    assert "saved" in body.lower()


@pytest.mark.parametrize("host", ["cli.py", "server.py"])
def test_row1_both_hosts_pass_the_preview(host):
    """Same seam as the notify test above: each host drives its own execute, so
    a preview passed on one and forgotten on the other leaves the CLI with the
    bug the web path just lost."""
    text = (_ROOT / host).read_text(encoding="utf-8")
    calls = re.findall(r"\.note_host_write\s*\((.*?)\)\s*\n", text, re.S)
    assert calls, f"{host} no longer calls note_host_write"
    for call in calls:
        assert "preview" in call, (
            f"{host} notifies without the preview: {' '.join(call.split())!r}")


def test_row1_note_gives_the_agent_no_verb_to_mirror():
    """The note must describe facts, never tell the agent how to speak.

    Live: the note said "If asked what was just saved, STATE IT directly from
    the record below" and the answer came back "I STATED that you performed 1
    set of 102 lbs for 6 reps…" — the agent narrating its own speech act rather
    than reporting the save, because an instruction verb was sitting in its
    context waiting to be echoed. Correct content, unnatural sentence.
    """
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 1 sets across 1 exercise(s).",
                      _PREVIEW)
    body = s._conversation_history[0][0]["content"].lower()
    for phrase in ("state it", "say ", "tell the user", "reply ", "respond ",
                   "answer directly", "phrase"):
        assert phrase not in body, (
            f"note contains a speech instruction the agent can mirror: {phrase!r}")


def test_row1_note_still_rules_out_the_clarifying_question():
    """Dropping the imperative must not drop the PROPERTY that made the
    clarifying question wrong — the record being complete and unambiguous is
    what the agent needs, and it is a fact rather than an order."""
    s = _agent_session()
    s.note_host_write("Workout logged and verified: 1 sets across 1 exercise(s).",
                      _PREVIEW)
    body = s._conversation_history[0][0]["content"].lower()
    assert "authoritative" in body or "complete" in body
    assert "ambiguous" in body or "no tool call is needed" in body


# ══════════════════════════════════════════════════════════════════════════════
# Row 4 — the MIRROR claim: "it's staged, waiting for you", when nothing is
#
# Live 2026-08-31. The stage-2 verifier returned FAIL, the server discarded the
# batch, and the reply still read "The following workout has been staged and is
# awaiting your confirmation" with the full set list under it. The claim gate
# never looked — it reads completed-write verbs only. So the user waited to
# confirm a batch that no longer existed, and the workout was silently lost.
#
# False by construction wherever this fires: nothing written, nothing staged, no
# execute attempted.
# ══════════════════════════════════════════════════════════════════════════════

_STAGED_LIE = ("The following workout has been staged and is awaiting your "
               "confirmation:\n* Set 1: 40 lbs x 10 reps")


def test_an_unbacked_staged_claim_is_replaced(monkeypatch):
    from src.coordinator import MSG_NOTHING_STAGED
    c = _coord(monkeypatch, _STAGED_LIE, write_attempted=True)
    out = _run(c, "log bench 40x10", log_boundary=True)
    assert out == MSG_NOTHING_STAGED
    assert "awaiting your confirmation" not in out


def test_a_real_staged_turn_is_left_alone(monkeypatch):
    """THE direction that must not break — when a batch really is staged, the
    invitation to confirm is true and has to reach the user."""
    c = _coord(monkeypatch, _STAGED_LIE, staged_this_turn=True,
               write_attempted=True)
    assert _run(c, "log bench 40x10", log_boundary=True) == _STAGED_LIE


def test_a_denied_staging_is_not_a_staged_claim(monkeypatch):
    """Reuses the clause splitter and denial guard, so a refusal to stage is not
    read as a claim to have staged."""
    honest = "I couldn't stage that — I still need the date."
    c = _coord(monkeypatch, honest, write_attempted=True)
    assert _run(c, "log bench", log_boundary=True) == honest


def test_research_prose_is_not_a_staged_claim(monkeypatch):
    """No write tool called ⇒ out of scope entirely, same as the completion
    detector. 'The study staged participants' is not a database claim."""
    prose = "The study staged participants across two cohorts."
    c = _coord(monkeypatch, prose)
    assert _run(c, "what did the study do?") == prose


# ── CHANGED by Row 3: the always-on detector can finally see deletions ────────
# LIVE. After a confirmed goal delete that never executed, this exact sentence
# reached the user untouched. The LOOSE detector matched it — but that one only
# runs on turns where a write tool was called, and the post-confirm continuation
# called none. So on the one turn that mattered, only the narrow alternation was
# looking, and its `has been ...` list was logged|saved|recorded|written|added.
# Deletion claims are the dangerous direction and were the only family it could
# not see.

_LIVE_DELETE_LIE = ("The goal of 150 lbs on Flat Barbell Bench Press by "
                    "2026-12-31 has been deleted from your database.")


@pytest.mark.parametrize("claim", [
    _LIVE_DELETE_LIE,
    "The set has been removed from your database.",
    "Your goal has been updated.",
])
def test_the_narrow_detector_sees_deletion_and_edit_claims(claim):
    """Narrow = runs on EVERY operational turn, including one where no write
    tool was called. That is the turn this failed on."""
    assert _WRITE_SUCCESS_CLAIM_RE.search(claim), (
        f"the always-on detector cannot see: {claim!r}")


def test_the_live_delete_lie_is_caught_with_no_write_tool_called(monkeypatch):
    from src.coordinator import _write_claim_clause
    assert _write_claim_clause(_LIVE_DELETE_LIE, write_flow=False) is not None


@pytest.mark.parametrize("prose", RESEARCH_PROSE)
def test_widening_the_narrow_detector_kept_research_prose_out(prose):
    """THE regression this widening could have caused. The narrow detector
    requires the explicit 'has been <verb>' shape, which is what keeps 'two
    trials were deleted from the meta-analysis' out of it."""
    assert not _WRITE_SUCCESS_CLAIM_RE.search(prose)
