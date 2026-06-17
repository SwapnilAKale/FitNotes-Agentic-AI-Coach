"""
src/coordinator.py
Coordinator — Entry Point and Router

Every user message comes here first. The Coordinator:
  1. Classifies: analytical vs operational (one Gemini call)
  2. Extracts parameters: exercise_names, muscle_groups, query_period_days
  3. Routes to one of two paths:
       analytical  → prepare_analysis_package() → Analysis Agent → coverage check
       operational → Single Agent (existing AgentSession.answer())
  4. Runs a coverage check after analytical answers (1 retry max)
  5. Maintains conversation history for context passing to Analysis Agent

Shared modules wired into the analytical path:
  shared/resolver.py → exercise name resolution before package build
  shared/rag.py      → research pre-fetch (best-effort, None on failure)
  shared/memory.py   → read-only memory retrieval (best-effort)
  shared/sql_executor (via data_agent.query) → supplementary custom SQL

NOT YET WIRED:
  Memory extraction from mixed analytical+memory messages

ROUTING RULE (Step C): reads default ANALYTICAL. Operational is a positive
allowlist — writes (caught first by the write-intent regex pre-guard),
research/RAG, and specific-date session display — plus out_of_scope refusals.
Everything else, and anything uncertain, routes analytical. The analytical
package is deterministic and validated (it hard-stops on an integrity failure),
so it is the safer default for a read; the old "a wrong analytical package is
confusing" reasoning is obsolete now that the package fails loud and the
operational hand-rolled-SQL path is the riskier read surface.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from google import genai
from google.genai import types

from src.data_agent import (
    prepare_analysis_package, DataAgentIntegrityError, match_muscle_group,
)
from src import analysis_agent
from src import checkpoint as _ckpt
from src import citations as _cite

# Rate-limit error types: re-raised instead of swallowed so CLI/server
# countdown UX works for analytical-path quota exhaustion.
try:
    from google.genai import errors as _genai_errors
    _GenaiClientError = _genai_errors.ClientError
except (ImportError, AttributeError):
    _GenaiClientError = None  # type: ignore[assignment]

try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
except ImportError:
    _ResourceExhausted = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
COORDINATOR_MODEL  = "gemini-3.1-flash-lite"
CONTEXT_WINDOW     = 6     # number of recent turns passed to Analysis Agent

# ── Per-minute 429 silent-retry tuning ────────────────────────────────────────
# A per-minute quota refills in ~60s, so a brief in-request wait clears it
# without checkpointing. Diagnosis confirmed /chat has no server timeout and the
# frontend fetch has no client timeout, so holding the request open is safe.
PER_MINUTE_WAIT_CAP    = 70   # s — cap on ONE wait; bounds the held request even if retryDelay is large
PER_MINUTE_MAX_RETRIES = 2    # per stage — 2 waits clear all but pathological bursts before the daily fallback
PER_MINUTE_BUFFER      = 2    # s added to retryDelay so we retry just AFTER the window resets

# ── Write-intent guard (#3/#8 rework) ─────────────────────────────────────────
# The pre-guard's job is to catch IMPERATIVE writes (commands to RECORD data) so
# a write can never reach the analytical path. It must NOT fire on coaching
# QUESTIONS about writing ("should I add weight to my squat") — post-Step-C those
# get force-routed into the now-impoverished operational agent and strand as
# non-answers.
#
# PRECEDENCE (stated): a question/modal phrasing ALWAYS wins — if the message
# looks like a question, the guard does NOT fire, even when it also contains a
# write verb + a quantity. The reasoning: a missed regex-write still gets caught
# by the LLM classifier and is gated by the operational confirmation prompt
# before it can touch the DB, so a false NEGATIVE is recoverable. A false
# POSITIVE strands a coaching question with no recourse (operational can't
# answer analytical reads). So when ambiguous between "question about writing"
# and "command to write", we prefer NOT firing (let the classifier decide).

# Interrogative / modal-coaching phrasing → this is a QUESTION, not a command.
# Matched anywhere (a polite "..., should I add a set?" is still a question).
_WRITE_QUESTION_RE = re.compile(
    r"(?i)(?:"
    r"\?"                                                       # any question mark
    r"|\bshould\s+i\b|\bcan\s+i\b|\bcould\s+i\b|\bmay\s+i\b"
    r"|\bdo\s+i\b|\bwould\s+it\b|\bdo\s+you\s+think\b"
    r"|\bis\s+it\s+(?:ok|okay|fine|worth|better|good|bad|safe)\b"
    r"|\bwhen\s+should\b|\bhow\s+(?:much|many|often|do|should|can)\b"
    r"|^\s*(?:is|are|do|does|did|was|were|will|what|why|when|where|which|who)\b"
    r")"
)

# Weight×reps / sets×reps / unit shorthand — a strong signal a write is being
# DICTATED ("bench 100x5", "3 sets", "80kg", "12 reps").
_WRITE_QUANTITY = (
    r"(?:"
    r"\d+\s*[x×]\s*\d+"                              # 100x5, 3x5
    r"|\d+\s*sets?\b|\d+\s*reps?\b"                  # 3 sets, 12 reps
    r"|\d+\s*(?:lbs?|kgs?|pounds?|kilos?)\b"         # 100 lbs, 80kg
    r")"
)

# Imperative writes that DO fire (only after the question guard says "not a
# question"). Three forms:
#   (A) "set a goal" / "set goal …"
#   (B) write verb + quantity shorthand → bare "log bench 100x5", "record squat
#       80kg x5", "add 3 sets of deadlift" (the #8 false-negatives)
#   (C) write verb + an explicit data noun → "delete my deadlift goal",
#       "log today's workout", "update my last set"
#   (D) "I did … today/yesterday/…" narration of a completed session
# NOTE: standalone "weight" is deliberately NOT a (C) noun — "add weight" is the
# canonical coaching phrasing ("should I add weight"), so it must not anchor a write.
_WRITE_IMPERATIVE_RE = re.compile(
    r"(?i)(?:"
    r"\bset\s+(?:a\s+|an\s+|my\s+|the\s+)?goal\b"                          # (A)
    r"|\b(?:log|record|save|add|delete|remove|update|change|correct)\b"
    r".{0,40}?" + _WRITE_QUANTITY +                                        # (B)
    r"|\b(?:log|record|save|add|delete|remove|update|change|correct)\b"
    r".{0,50}\b(?:workout|sets?|reps?|goal|bodyweight|exercise|session)\b" # (C)
    r"|\bi\s+did\b.{0,80}\b(?:today|yesterday|this\s+morning|this\s+week)\b"  # (D)
    r")"
)


def _is_write_intent(message: str) -> bool:
    """
    True only for IMPERATIVE writes (commands to record data). Question/modal
    phrasing wins (precedence above): if the message reads as a question, return
    False so the classifier — not this pre-guard — decides the route.
    """
    if not message:
        return False
    if _WRITE_QUESTION_RE.search(message):
        return False
    return bool(_WRITE_IMPERATIVE_RE.search(message))


# ── Filler short-circuit (#5a) ────────────────────────────────────────────────
# Bare greetings / acknowledgments / thanks must NOT spend a classify call, a
# package build, or an analytical/operational turn. A small, conservative,
# exact-match set (after stripping surrounding punctuation/whitespace) returns a
# cheap canned reply. Anything with real content fails the exact match and falls
# through to normal routing — so "thanks, now how's my squat" is NOT a filler.
_FILLER_THANKS = frozenset({
    "thanks", "thank you", "thank u", "thx", "ty", "tysm", "thanks so much",
    "thank you so much", "cheers", "appreciate it", "much appreciated",
})
_FILLER_OTHER = frozenset({
    "hi", "hello", "hey", "yo", "hiya", "heya", "hey there", "hi there",
    "hello there", "good morning", "good evening", "good afternoon",
    "ok", "okay", "k", "kk", "cool", "nice", "got it", "gotcha", "sure",
    "yes", "yep", "yeah", "yup", "no", "nope", "great", "awesome", "perfect",
    "sounds good", "sweet", "alright", "fine", "ok thanks", "okay thanks",
})

_FILLER_THANKS_REPLY = (
    "You're welcome! Anything else about your training I can help with?"
)
_FILLER_GENERIC_REPLY = "What can I help you with about your training?"


def _filler_reply(message: str) -> Optional[str]:
    """
    Canned reply for a bare filler message, or None if the message has real
    content and should route normally. Conservative: exact match against a
    small curated set after normalizing surrounding punctuation/whitespace,
    length-gated so only short fillers qualify.
    """
    norm = re.sub(r"\s+", " ", message.strip().lower()).strip(" !.,…?")
    if not norm:
        # Empty / whitespace-only — cheap generic, no pipeline.
        return _FILLER_GENERIC_REPLY
    if len(norm) > 20:
        return None
    if norm in _FILLER_THANKS:
        return _FILLER_THANKS_REPLY
    if norm in _FILLER_OTHER:
        return _FILLER_GENERIC_REPLY
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """True if exc is a Gemini/google-api 429 / ResourceExhausted error."""
    if _ResourceExhausted is not None and isinstance(exc, _ResourceExhausted):
        return True
    if _GenaiClientError is not None and isinstance(exc, _GenaiClientError):
        if getattr(exc, "code", None) == 429:
            return True
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


# ── Classification prompt ─────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """
You are a routing classifier for a fitness coaching AI.

Classify each message as "analytical" or "operational". Return JSON only.

ANALYTICAL — needs trend analysis, progression tracking, or pattern detection
across multiple sessions. Requires computing statistics over training history.

  Examples → analytical:
    "How is my Lat Pulldown progressing?"
    "Why am I plateauing on bench?"
    "How has my back training been over the last 3 months?"
    "Am I overtraining?"
    "Will I hit my Lat Pulldown goal?"
    "What patterns do you see in my training?"
    "How consistent have I been?"
    "Which muscle groups am I neglecting?"

OPERATIONAL — use only for questions that require MCP tools:
  - Write operations: logging, goal setting, corrections, deletions
  - Research: fitness science questions
  - Session display: "show me my last X session"

  Examples → operational:
    "Show me my last chest session"       (session display)
    "Log today's workout"                 (write operation)
    "Set a goal for 150 lbs on Lat Pulldown" (write operation)
    "Fix my last set — it was 12 reps not 10" (correction)
    "What does science say about training frequency?" (research)
    "Delete my deadlift goal"             (write operation)

  Route ANALYTICAL for any question the Data Agent can answer.
  The Data Agent computes all of the following deterministically:

    Personal records     — all-time PR, period PR, PR history
    Progression          — has weight or rep count changed over time
    Plateaus             — how long stuck, when stagnation started
    Volume               — total weight lifted, sets, per muscle group
    Frequency            — sessions per week, consistency, missed weeks
    Form trends          — technique quality over time from training notes
    Fatigue patterns     — pain sessions, failed attempts
    Rest day effects     — does performance vary with days of rest
    Day-of-week patterns — which days tend to produce better sessions
    Muscle group balance — push/pull ratio, neglected groups
    Exercise comparisons — fastest improving, most stagnant
    Goal projections     — will current rate of progress reach a target
    Any historical trend — any question about what happened over time

  Route OPERATIONAL only for things that require MCP tools:
    Writes      — log workout, set goal, update set, delete anything
    Research    — fitness science questions, what does science say
    Session display — show me exactly what I did on a specific date
                      with full set breakdown

OUT_OF_SCOPE — refuse politely WITHOUT any tool, search, or analysis. Decide
this by a FITNESS-CONNECTION test, NOT a keyword blocklist:
  "Is this about fitness, training, nutrition-for-training, fitness
   science/history, or the user's own training data?"
  If yes → route analytical/operational as above. If no → route "out_of_scope".

  IN SCOPE (route normally, NOT out_of_scope):
   - The user's own logs / training data — any phrasing, even with no fitness
     words ("how many days have I trained excluding Sundays").
   - Fitness science, exercise physiology, and DEFINITIONS of fitness terms
     ("what is progressive overload / hypertrophy / RPE / a superset").
   - Nutrition for training/performance, incl. food prep with a nutrition/goal
     angle ("cook chicken keeping protein high", "good pre-workout meal", macros).
   - Fitness history & culture ("Ronnie Coleman's diet", "who won Mr. Olympia 1998").
   - Program design, splits, recovery, periodization, rest days.
   - Training / rehab / mobility / warmups around a stated symptom (see MEDICAL).

  OUT OF SCOPE (route "out_of_scope"):
   - Coding / software / SQL-for-its-own-sake / tech support.
   - AI / technology topics, writing about AI models.
   - Politics, news, current events, geography, economics.
   - Arts & humanities, NON-fitness history, literature, philosophy, music, film.
   - Linguistics / etymology / translation of NON-fitness terms ("what does
     ad-hoc mean", "translate this"). NOTE: fitness-term definitions are IN.
   - Puzzles, riddles, math-for-itself, brain teasers, general (non-fitness) trivia.
   - Creative writing, INCLUDING motivational poems / hype / "write me a poem".
   - General cooking / recipes with NO training or nutrition angle.
   - Personal-life advice unrelated to training (relationships, career, finance).

  When GENUINELY ambiguous, lean IN — a false refusal of a real fitness
  question is worse than answering something borderline.

MEDICAL questions are NEVER out_of_scope. A stated symptom, pain, or condition
is in-domain training territory — route it analytical/operational as normal. The
answer (governed by the coach's system prompt) gives training adaptations plus a
see-a-professional redirect and refuses only to DIAGNOSE or TREAT. Never send a
medical/symptom question to out_of_scope. A medical/symptom question is a
read/coaching question → ANALYTICAL by the default below.

DEFAULT: operational is a POSITIVE allowlist — route "operational" ONLY for the
three cases listed above (writes/corrections/goals, research/RAG, and
specific-date session display), and "out_of_scope" only per the test above.
EVERYTHING ELSE is "analytical": every read, trend, stat, PR, volume, frequency,
plateau, comparison, projection, and coaching question — including terse ones
("my squat?", "Lat Pulldown PR"). When uncertain, default to "analytical". The
analytical package is deterministic and validated (it hard-stops on integrity
failure), whereas the operational hand-rolled-SQL read path is the riskier
surface for a read.

If a [PREVIOUS TURNS] block is present, use it ONLY to resolve pronouns
and follow-up references in the current message ("what about my squat?",
"and over the last year?"). Classify and extract parameters for the
CURRENT MESSAGE, carrying over the topic from previous turns when the
current message is an elliptical follow-up.

PARAMETER EXTRACTION (analytical route only):
  exercise_names:    list of specific exercise names mentioned, or null
  muscle_groups:     muscle groups mentioned → map to exact names:
                     Back, Chest, Shoulders, Biceps, Triceps, Legs,
                     Forearms, Abs, Cardio
                     or null if none mentioned
  query_period_days: convert time references to integer days, or null for all-time
    "last week"          → 7
    "last month"         → 30
    "last 2/3 months"    → 60 / 90
    "last 6 months"      → 180
    "last year"          → 365
    "all time" / "ever"  → null
    not specified        → 90  (default)

CUSTOM SQL (analytical route only):
  Set needs_custom_sql=true ONLY for analytical questions that require a
  cross-cutting query the per-exercise package cannot answer: queries
  spanning multiple exercises on the same day, gaps between sessions,
  day-of-week patterns, streaks, or total counts/aggregates across all
  exercises. For these, set custom_sql_intent to a short description of
  what to query. For normal per-exercise questions (progression, PRs,
  plateaus, form), set needs_custom_sql=false and custom_sql_intent=null.
  Custom SQL is for counts, dates, gaps, and patterns — never for
  reporting individual set weights, which the package already covers.

Return ONLY valid JSON, no preamble, no markdown fences:
{
  "route": "analytical" | "operational" | "out_of_scope",
  "exercise_names": ["..."] | null,
  "muscle_groups": ["..."] | null,
  "query_period_days": 90 | null,
  "needs_custom_sql": false,
  "custom_sql_intent": null
}
""".strip()


# Out-of-scope refusal — warm, brief, redirects to purpose. Returned directly
# by route() the moment the classifier says out_of_scope, so it costs nothing
# beyond the classify call: no package build, no agent turn, no search, no
# analysis LLM call.
OUT_OF_SCOPE_REFUSAL = (
    "I'm your fitness coach, so I'll stick to your training, workouts, "
    "nutrition, and fitness questions — that one's outside what I'm built for. "
    "Ask me anything about your lifts, progress, programming, or recovery."
)


# ── Coverage check prompt ─────────────────────────────────────────────────────

_COVERAGE_SYSTEM = """
You are verifying that an answer fully addresses a user's question.

Check ONLY coverage — whether all parts of the question were addressed.
Do NOT check facts, numbers, or correctness. You have no access to the
underlying data and cannot verify whether specific values are accurate.

A complete answer need not be exhaustive — it just needs to address
every distinct thing the user asked about. Be lenient: only flag
genuinely unanswered parts, not stylistic gaps or level of detail.

Return ONLY valid JSON:
  {"complete": true,  "missing": []}
  {"complete": false, "missing": ["first unanswered part", "second..."]}
""".strip()


# ── Coordinator ───────────────────────────────────────────────────────────────

class Coordinator:
    """
    Entry point for every user message. Routes to analytical or operational path.
    Maintains conversation history for context passing.
    """

    def __init__(self, agent_session):
        """
        agent_session: an initialised AgentSession instance (from src/agent.py).
                       The Coordinator calls agent_session.answer() for operational
                       questions. Pass None to disable the operational path (testing).
        """
        self._agent   = agent_session
        self._client  = genai.Client(api_key=_GEMINI_API_KEY)
        self._history: list[dict] = []   # {role, content} pairs, last CONTEXT_WINDOW*2

    # ── Public entry point ────────────────────────────────────────────────────

    async def route(self, question: str) -> dict:
        """
        Route a question to the appropriate pipeline.

        Returns:
            {
              "answer":        str,
              "route":         "analytical" | "operational",
              "flagged_claims": list,   # grounding check edits (analytical only)
              "error":         str | None,
            }
        """
        # ── 0. Checkpoint: resume, confirm-before-discard, or pass through ────
        # Continue-intent is checked BEFORE the write-intent guard so "continue"
        # always resumes the interrupted question, never classifies as new.
        # A live slot is never silently discarded — a NEW question prompts for
        # confirmation first. (Stale >48h slots are dropped silently on load.)
        cp = _ckpt.load_checkpoint()
        if _ckpt.is_continue_intent(question):
            if cp is not None:
                return await self._resume(cp)
            # Nothing-to-resume guard: a resume request with no live slot must
            # NOT fall through to classification and run a fresh, expensive
            # question. Return a plain notice — no LLM call.
            return self._no_resume_response()
        elif cp is not None:
            if cp.get("awaiting_discard_confirm"):
                # This message answers the discard prompt for the live slot.
                if _ckpt.is_discard_intent(question):
                    # Discard saved; process the stashed new question.
                    question = cp.get("pending_question") or question
                    _ckpt.clear_checkpoint()
                elif _ckpt.is_ambiguous_reply(question):
                    # Resolves neither way — re-ask, keep the slot (never
                    # silently discard). The prompt makes no LLM call, so this
                    # reply path cannot be re-checkpointed or loop.
                    return self._confirm_response(_ckpt.discard_confirm_prompt(cp))
                else:
                    # A re-sent / substantive new question: discard, process it.
                    _ckpt.clear_checkpoint()
                # fall through to normal routing with `question`
            else:
                # First NEW question while a live slot exists: prompt and stash,
                # do NOT process the new question or discard the slot yet.
                _ckpt.mark_awaiting_discard(cp, question)
                return self._confirm_response(_ckpt.discard_confirm_prompt(cp))

        return await self._route_fresh(question)

    async def _route_fresh(self, question: str) -> dict:
        """
        Classify and dispatch a NEW (or classify-resumed) question. Separated
        from route()'s checkpoint handling so a resume from
        completed_stage='classify' can re-enter here directly (re-running the
        cheap classify call) without re-triggering checkpoint logic.
        """
        # ── 0b. Filler short-circuit (#5a) ───────────────────────────────────
        # Bare greeting / ack / thanks / empty-ish → cheap canned reply, NO
        # classify call, NO package, NO analytical/operational pipeline. Runs
        # BEFORE classification. Conservative: anything with real content (incl.
        # a question behind a polite prefix) falls through to normal routing.
        filler = _filler_reply(question)
        if filler is not None:
            return self._filler_response(filler)

        # ── 1. Classify (or short-circuit for obvious write operations) ──────
        # Misrouting a write to analytical bypasses the confirmation gate.
        if _is_write_intent(question):
            params = {
                "route":             "operational",
                "exercise_names":    None,
                "muscle_groups":     None,
                "query_period_days": 90,
                "needs_custom_sql":  False,
                "custom_sql_intent": None,
            }
        else:
            # Per-minute 429s retry silently in-request; a DAILY 429 (or
            # exhausted per-minute retries) closes the classify-stage gap by
            # checkpointing at 'classify' (nothing paid → draft=null) and
            # surfacing the resume status, instead of re-raising unprotected.
            try:
                params = await self._call_with_per_minute_retry(
                    self._classify, question)
            except Exception as e:
                if _is_rate_limit(e):
                    _ckpt.save_checkpoint(
                        route="analytical", question=question,
                        params=None, completed_stage="classify",
                    )
                    raise _ckpt.QuotaInterrupted(e, _ckpt.msg_draft_interrupted(e))
                raise

        # ── 1b. Parse-failure cheap default (#5b) ────────────────────────────
        # If _classify could not PARSE the model output (vs. parsed-but-uncertain,
        # which Step C correctly defaults analytical), the input is effectively
        # garbage — do NOT build a ~358KB package and run analyze+ground on it
        # (a bare "ok" would have the classifier invent muscle_groups=['Legs']).
        # Return a cheap rephrase prompt instead. This ONLY changes the
        # unparseable/errored case; parsed-but-uncertain still goes analytical.
        if params.get("_parse_failed"):
            return self._unparseable_response()

        route  = params.get("route", "analytical")

        # ── Out-of-scope: refuse at classification time. No package build, no
        # agent turn, no search, no analysis LLM call — the refusal IS the
        # classifier's output, so a non-fitness question costs only the classify.
        if route == "out_of_scope":
            return self._out_of_scope_response()

        # ── 2. Route ──────────────────────────────────────────────────────────
        flagged = []
        error   = None

        if route == "analytical":
            try:
                answer, flagged = await self._run_analytical(question, params)
            except DataAgentIntegrityError as e:
                ids_str = ", ".join(v.invariant_id for v in e.violations)
                logger.error(
                    "[coordinator] data integrity check failed: %s\n  %s",
                    ids_str,
                    "\n  ".join(
                        f"{v.invariant_id}: {v.message}" for v in e.violations
                    ),
                )
                error  = str(e)
                answer = (
                    f"I cannot answer this question right now. "
                    f"A data integrity check failed ({ids_str}). "
                    f"The analytical pipeline was stopped to prevent "
                    f"incorrect analysis from reaching you. "
                    f"Please try again or contact support if this persists."
                )
            except Exception as e:
                if _is_rate_limit(e):
                    raise
                logger.error("[coordinator] analytical pipeline failed: %s", e)
                # Fall back to operational on pipeline failure
                route  = "operational"
                error  = str(e)
                answer = await self._run_operational(question)
        else:
            answer = await self._run_operational(question)

        # ── 3. Update conversation history ────────────────────────────────────
        self._history.append({"role": "user",      "content": question})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > CONTEXT_WINDOW * 2:
            self._history = self._history[-(CONTEXT_WINDOW * 2):]

        return {
            "answer":         answer,
            "route":          route,
            "flagged_claims": flagged,
            "error":          error,
        }

    # ── Per-minute silent retry ───────────────────────────────────────────────

    async def _call_with_per_minute_retry(self, fn, *args, **kwargs):
        """
        Run an async LLM stage, absorbing PER-MINUTE 429s silently: wait the
        provider's retryDelay (capped at PER_MINUTE_WAIT_CAP) and retry the SAME
        call up to PER_MINUTE_MAX_RETRIES, holding the request open — no
        checkpoint, no message, the user keeps seeing 'thinking…'. A DAILY 429,
        or exhausted per-minute retries, propagates unchanged so the caller's
        daily path (checkpoint + QuotaInterrupted) runs.

        NOTE: agent_lock is held for the duration of the wait (single-user
        assumption — a concurrent /chat gets the existing 'busy' 429).
        """
        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if (_is_rate_limit(e) and _ckpt.is_per_minute_quota(e)
                        and attempt < PER_MINUTE_MAX_RETRIES):
                    wait = min(_ckpt.retry_delay_seconds(e) or 55,
                               PER_MINUTE_WAIT_CAP) + PER_MINUTE_BUFFER
                    logger.warning(
                        "[coordinator] per-minute 429 — waiting %ds then retrying "
                        "(attempt %d/%d), request held open",
                        wait, attempt + 1, PER_MINUTE_MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    attempt += 1
                    continue
                raise

    # ── Checkpoint resume ─────────────────────────────────────────────────────

    def _no_resume_response(self) -> dict:
        """Resume requested but no live slot — plain notice, no LLM call."""
        return {
            "answer":         "There's no saved question to resume.",
            "route":          "none",
            "flagged_claims": [],
            "error":          None,
        }

    def _out_of_scope_response(self) -> dict:
        """
        Non-fitness question — refuse immediately with the coach redirect. No
        tool, package, search, or analysis call is made (the dispatch returns
        before any of them), so an out-of-scope turn spends only the classify.
        """
        return {
            "answer":         OUT_OF_SCOPE_REFUSAL,
            "route":          "out_of_scope",
            "flagged_claims": [],
            "error":          None,
        }

    def _filler_response(self, text: str) -> dict:
        """
        Bare-filler reply (#5a). No classify, package, or pipeline was run — this
        is the entire cost of a greeting/ack turn. Not recorded in history.
        """
        return {
            "answer":         text,
            "route":          "filler",
            "flagged_claims": [],
            "error":          None,
        }

    def _unparseable_response(self) -> dict:
        """
        Cheap default for input the classifier could not PARSE (#5b). Avoids
        building a large package + analyze/ground on garbage; asks to rephrase.
        Not recorded in history.
        """
        return {
            "answer":         "I didn't catch that — could you rephrase?",
            "route":          "unparseable",
            "flagged_claims": [],
            "error":          None,
        }

    def _confirm_response(self, text: str) -> dict:
        """
        Transient control-flow reply (the discard-confirmation prompt).
        Not recorded in conversation history — it is a meta-prompt about a
        pending checkpoint, not a question/answer turn.
        """
        return {
            "answer":         text,
            "route":          "checkpoint_confirm",
            "flagged_claims": [],
            "error":          None,
        }

    async def _resume(self, cp: dict) -> dict:
        """
        Resume a quota-interrupted question from the checkpoint slot.

        Completed stages are never re-paid: the package rebuilds free (pure
        Python, re-validated as normal), and a stored draft skips the draft
        LLM call entirely — only the remaining verification stages run.
        Clears the slot on success. If the resume itself hits the quota, the
        stage handlers update the checkpoint and the status flows to the user
        again (slot intact).
        """
        orig_q  = cp.get("question") or ""
        route   = cp.get("route") or "analytical"
        flagged: list = []
        error = None

        logger.info(
            "[coordinator] resuming checkpoint: route=%s completed_stage=%s",
            route, cp.get("completed_stage"),
        )

        # True classify-stage gap: classify itself was interrupted, so the route
        # is not yet known and no params exist. Re-run the cheap classify+dispatch
        # path. (A draft-stage interruption also records completed_stage="classify"
        # but carries real params — that one resumes into _run_analytical below to
        # re-run only the draft, without re-classifying.)
        if cp.get("completed_stage") == "classify" and not cp.get("params"):
            _ckpt.clear_checkpoint()
            return await self._route_fresh(orig_q)

        if route == "operational":
            if self._agent is None:
                return {
                    "answer": ("Operational path is not available in this "
                               "configuration, so the saved request cannot be "
                               "resumed."),
                    "route": "operational", "flagged_claims": [], "error": None,
                }
            result = await self._agent.resume(cp)   # QuotaInterrupted propagates
            _ckpt.clear_checkpoint()
            answer = result.get("answer", "")
        else:
            params = cp.get("params") or {}
            try:
                answer, flagged = await self._run_analytical(
                    orig_q, params, resume=cp
                )
                _ckpt.clear_checkpoint()
            except DataAgentIntegrityError as e:
                # Deterministic failure — retrying the slot cannot help.
                _ckpt.clear_checkpoint()
                ids_str = ", ".join(v.invariant_id for v in e.violations)
                logger.error(
                    "[coordinator] resume: data integrity check failed: %s", ids_str
                )
                error  = str(e)
                answer = (
                    f"I cannot resume this question. A data integrity check "
                    f"failed ({ids_str}) while rebuilding the analysis. "
                    f"Please ask the question again."
                )

        self._history.append({"role": "user",      "content": orig_q})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > CONTEXT_WINDOW * 2:
            self._history = self._history[-(CONTEXT_WINDOW * 2):]

        return {
            "answer":         answer,
            "route":          route,
            "flagged_claims": flagged,
            "error":          error,
        }

    # ── Classification ────────────────────────────────────────────────────────

    async def _classify(self, question: str) -> dict:
        """
        Single Gemini call: route + extract parameters.
        Defaults to ANALYTICAL on parse/exception failure (Step C flip): reads
        are the common case and the analytical package is deterministic and
        validated, while the operational hand-rolled-SQL read path is the riskier
        surface for a read. Genuine operational intents (writes, research,
        session-display) are caught positively by the write-intent pre-guard and
        the classifier's operational allowlist; a write that somehow defaults
        analytical simply won't write (the confirmation gate is operational-only),
        so it cannot corrupt data.
        """
        default = {
            "route":              "analytical",
            "exercise_names":     None,
            "muscle_groups":      None,
            "query_period_days":  90,
            "needs_custom_sql":   False,
            "custom_sql_intent":  None,
            # Marks an UNPARSEABLE/errored classify (vs. parsed-but-uncertain).
            # The caller (#5b) returns a cheap rephrase instead of running the
            # full analytical pipeline on garbage input.
            "_parse_failed":      True,
        }
        # Follow-up questions ("what about my squat?") are unclassifiable
        # without the previous turn — give the classifier a compact window.
        classify_input = question
        if self._history:
            recent = self._history[-2:]
            ctx_lines = [
                f"{t.get('role', '?')}: {(t.get('content') or '')[:300]}"
                for t in recent
            ]
            classify_input = (
                "[PREVIOUS TURNS]\n" + "\n".join(ctx_lines)
                + "\n\n[CURRENT MESSAGE]\n" + question
            )
        try:
            config = types.GenerateContentConfig(
                system_instruction=_CLASSIFY_SYSTEM,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=COORDINATOR_MODEL,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=classify_input)],
                )],
                config=config,
            )
            raw = ""
            _cand  = response.candidates[0] if getattr(response, "candidates", None) else None
            _parts = (_cand.content.parts
                      if _cand and _cand.content and _cand.content.parts else [])
            for part in _parts:
                if getattr(part, "text", None):
                    raw += part.text

            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.strip().startswith("```")
                ).strip()

            params = json.loads(raw)
            # Ensure required keys are present (route defaults analytical — Step C flip)
            params.setdefault("route",             "analytical")
            params.setdefault("exercise_names",    None)
            params.setdefault("muscle_groups",     None)
            params.setdefault("query_period_days", 90)
            params.setdefault("needs_custom_sql",  False)
            params.setdefault("custom_sql_intent", None)
            return params

        except Exception as e:
            if _is_rate_limit(e):
                raise
            logger.warning("[coordinator] classify failed: %s — defaulting to analytical", e)
            return default

    # ── Stage 1 citation health (log only; never alters answer/grounding) ──────

    def _log_citation_health(self, cited: list) -> None:
        """
        Stage-1 verification log: report how the draft's citation tags resolved
        against the package. Pure observability — does NOT change the answer or
        grounding (full-package grounding is still the safety net). Surfaces the
        two silent failure modes loud: a bad match-key (draft invented/altered a
        name) and a false ABSENT claim.
        """
        if not cited:
            return
        problems  = [c for c in cited if c["status"] in _cite.FLAG_STATUSES]
        # Count non-scalar resolutions separately — a tag that resolves to a
        # list/dict (e.g. a ranked list) "exists" but gives grounding no scalar
        # to check, so it must not be hidden inside the clean "resolved" count.
        nonscalar = sum(1 for c in cited if c["status"] == _cite.OK_NONSCALAR)
        clean     = len(cited) - len(problems)
        logger.info(
            "[coordinator] citation health: %d tag(s), %d clean, %d non-scalar, "
            "%d flagged", len(cited), clean, nonscalar, len(problems),
        )
        for c in problems:
            logger.warning(
                "[coordinator] citation FLAG %s — %s|%s|%s",
                c["status"], c["collection"], c["match_key"], c["field_path"],
            )

    # ── Analytical pipeline ───────────────────────────────────────────────────

    async def _run_analytical(
        self,
        question: str,
        params:   dict,
        resume:   Optional[dict] = None,
    ) -> tuple[str, list]:
        """
        Full analytical pipeline:
          1. prepare_analysis_package() with extracted params
          2. analysis_agent.analyze() → draft
          3. analysis_agent.ground_check() → verified answer + edits
          4. coverage check → if incomplete, one retry with gap context

        resume: a checkpoint slot dict. The package always rebuilds (free,
        re-validated — G6 etc. still apply); a stored draft skips the draft
        LLM call and is verified VERBATIM.

        Stage-boundary 429 handling: each LLM stage saves a checkpoint of
        the last COMPLETED stage and raises QuotaInterrupted (status text
        only — never draft content).

        Returns (final_answer, flagged_claims).
        """
        exercise_names    = params.get("exercise_names")
        if exercise_names:
            exercise_names = [n.strip() for n in exercise_names]

        # ── Muscle-group Category guard (belt-and-suspenders) ──────────────────
        # A term that names a muscle-group Category (Triceps, Chest, Back, …) is
        # NOT an exercise. "Triceps" substring-matches 5 real exercise names, so
        # running resolve_exercise_name on it emits a bogus "which one did you
        # mean?" disambiguation prompt and never builds the package. Regardless of
        # which slot the classifier used, route every category term to
        # muscle_groups (canonical form) → GROUP scope, and resolve/disambiguate
        # only the genuine exercise names that remain.
        muscle_groups: list = []
        for g in (params.get("muscle_groups") or []):
            canon = match_muscle_group(g) or (g.strip() if isinstance(g, str) else g)
            if canon and canon not in muscle_groups:
                muscle_groups.append(canon)
        if exercise_names:
            kept_names: list = []
            for name in exercise_names:
                canon = match_muscle_group(name)
                if canon:
                    if canon not in muscle_groups:
                        muscle_groups.append(canon)
                else:
                    kept_names.append(name)
            exercise_names = kept_names or None

        if exercise_names:
            from src.shared.resolver import resolve_exercise_name
            db_path = os.environ.get(
                "FITNOTES_DB_PATH",
                os.path.join(os.path.dirname(__file__), "..", "data", "FitNotes_Backup.fitnotes")
            )
            resolved = []
            for name in exercise_names:
                result = resolve_exercise_name(name, db_path)
                match = result.get("match")
                candidates = result.get("candidates") or []
                if match:
                    resolved.append(match)
                elif len(candidates) == 1:
                    # One clear best match — use it (same as before for single-candidate case)
                    resolved.append(candidates[0])
                elif len(candidates) >= 2:
                    # Genuine ambiguity: surface candidates so the user can clarify
                    # rather than silently guessing the first one.
                    names_list = "\n".join(f"- {c}" for c in candidates[:5])
                    return (
                        f"I found multiple exercises matching **{name}**. "
                        f"Which one did you mean?\n\n{names_list}\n\n"
                        f"Please let me know and I'll answer your question.",
                        [],
                    )
                else:
                    # No match — keep original so the package reports it as unresolved
                    resolved.append(name)
            exercise_names = resolved
        # muscle_groups already computed above by the Category guard (canonical,
        # including any category terms moved out of exercise_names).
        muscle_groups     = muscle_groups or None
        query_period_days = params.get("query_period_days", 90)

        # Build compact package (scope-aware trim: BROAD 365d ≈ 396 KB)
        pkg = await asyncio.to_thread(
            prepare_analysis_package,
            query_period_days=query_period_days,
            exercise_names=exercise_names,
            muscle_groups=muscle_groups,
            include_phase2=True,
        )

        # Pop unresolved exercise names before the LLM sees the package.
        # These are names the filter failed to match; the package covers
        # overall training as a broad fallback in that case.
        unresolved_names = pkg.pop("unresolved_exercise_names", None)

        # Effective names: those that actually appeared in the package
        # (used to build scope notes — unresolved names are excluded).
        effective_names = (
            [n for n in (exercise_names or []) if n not in (unresolved_names or [])]
            if exercise_names else None
        ) or None

        conversation_context = self._history[-CONTEXT_WINDOW:] or None

        # Prepend package scope note so the Analysis Agent knows
        # it is working with a filtered subset, not the full database.
        # Without this, filtered packages produce misleading statements
        # like "100% of your training volume" when only one muscle group
        # was fetched.
        pkg_scope   = pkg.get("scope", "broad")
        scope_parts = []
        if effective_names:
            scope_parts.append(
                f"Note: the data package covers only these exercises: "
                f"{', '.join(effective_names)}."
            )
        elif muscle_groups:
            scope_parts.append(
                f"Note: the data package covers only "
                f"{', '.join(muscle_groups)} exercises — "
                f"volume and set counts shown are for this group only, "
                f"not total training."
            )
        if exercise_names or muscle_groups:
            scope_parts.append(
                "The training_consistency field in this package counts "
                "total gym days across all exercises, not days specific "
                "to the filtered group. Do not use it to state how many "
                "times a muscle group or exercise was trained."
            )

        # Scope-specific analytical limitations note
        if pkg_scope == "broad":
            scope_parts.append(
                "Note: broad analytical package — individual comment "
                "histories are excluded. Use pain_analysis and "
                "comment_keyword_trends for comment-based insights; "
                "do NOT claim comment detail is unavailable in the data."
            )
        elif pkg_scope == "group":
            scope_parts.append(
                "Note: group analytical package — comment histories are "
                "capped at 30 per exercise plus all pain-flagged entries."
            )

        if scope_parts:
            scoped_question = " ".join(scope_parts) + " " + question
        else:
            scoped_question = question

        # Analytical path is for personal workout data — RAG (fitness science knowledge
        # base) is irrelevant here and would burn Gemini calls unnecessarily.
        # search_fitness_knowledge only fires on the operational path (agent ReAct loop).
        research = None
        try:
            from src.shared.memory import retrieve_relevant_memories
            memories = retrieve_relevant_memories(question) or None
        except Exception:
            memories = None

        # ── Stage: DRAFT (supplementary SQL + analyze) ─────────────────────────
        # A stored draft (resume) is used VERBATIM — the draft LLM call is
        # skipped entirely; only the remaining verification stages run.
        custom_query = None
        draft = (resume or {}).get("draft") if resume else None
        if draft is not None:
            logger.info(
                "[coordinator] resume: stored draft (%d chars) used verbatim — "
                "draft LLM call skipped", len(draft),
            )
            # The stored draft is already stripped (no tags), so there are no
            # cited values to extract — grounding uses the full-package fallback.
            gctx = _cite.build_grounding_context([], pkg)
        else:
            try:
                # Custom SQL generation is an LLM call — it belongs to the
                # draft stage (nothing paid for yet if it 429s).
                if params.get("needs_custom_sql") and params.get("custom_sql_intent"):
                    custom_query = await self._call_with_per_minute_retry(
                        self._generate_custom_sql,
                        question, params["custom_sql_intent"],
                    )
                draft_tagged = await self._call_with_per_minute_retry(
                    analysis_agent.analyze,
                    pkg, scoped_question, research, memories,
                    conversation_context, custom_query,
                )
            except Exception as e:
                if _is_rate_limit(e):
                    _ckpt.save_checkpoint(
                        route="analytical", question=question, params=params,
                        completed_stage="classify",
                    )
                    raise _ckpt.QuotaInterrupted(e, _ckpt.msg_draft_interrupted(e))
                raise

            # ── Citation layer: resolve tags → grounding context → strip ────
            # The draft carries inline [[collection|match-key|field]] tags. Resolve
            # them, log health, then build the Stage-2 grounding context: if every
            # claim is cleanly cited the grounding call gets only the small
            # cited-scalar payload (the ~10–40× input cut); if any claim isn't
            # cleanly cited it falls back to the full package (today's behaviour).
            # Strip the tags so the clean prose flows to grounding + checkpoint +
            # the user.
            cited = _cite.extract_cited_values(draft_tagged, pkg)
            self._log_citation_health(cited)
            draft = _cite.strip_tags(draft_tagged)
            gctx  = _cite.build_grounding_context(cited, pkg)

        # ── Stage: GROUNDING ───────────────────────────────────────────────────
        # POLICY: the user never sees unverified draft text. On interruption
        # the verbatim draft is checkpointed and only a status message ships.
        flagged: list = []
        if resume and resume.get("completed_stage") == "coverage":
            # Grounding completed before the interruption — the stored text
            # is already verified; only the coverage stage remains.
            answer = draft
        else:
            logger.info("[coordinator] grounding path: %s", gctx.get("mode"))
            try:
                answer, flagged = await self._call_with_per_minute_retry(
                    analysis_agent.ground_check, draft, gctx)
            except Exception as e:
                if _is_rate_limit(e):
                    _ckpt.save_checkpoint(
                        route="analytical", question=question, params=params,
                        completed_stage="draft", draft=draft,
                    )
                    raise _ckpt.QuotaInterrupted(e, _ckpt.MSG_VERIFY_INTERRUPTED)
                raise

        # ── Stage: COVERAGE (question + answer only, no data; 1 retry) ────────
        try:
            answer, complete = await self._call_with_per_minute_retry(
                self._coverage_check, question, answer)

            if not complete:
                # One retry: add first draft + gaps to context, re-run analysis
                logger.info("[coordinator] coverage incomplete — retrying analysis")
                retry_context = list(conversation_context or []) + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not fully address the question. "
                            "Please revise to cover all parts."
                        ),
                    },
                ]
                retry_draft_tagged = await self._call_with_per_minute_retry(
                    analysis_agent.analyze,
                    pkg, scoped_question, research, memories, retry_context, custom_query)
                # Same citation path + Stage-2 split as the first draft.
                retry_cited = _cite.extract_cited_values(retry_draft_tagged, pkg)
                self._log_citation_health(retry_cited)
                retry_draft = _cite.strip_tags(retry_draft_tagged)
                retry_gctx  = _cite.build_grounding_context(retry_cited, pkg)
                logger.info("[coordinator] grounding path (retry): %s",
                            retry_gctx.get("mode"))
                answer, flagged2 = await self._call_with_per_minute_retry(
                    analysis_agent.ground_check, retry_draft, retry_gctx)
                flagged.extend(flagged2)
                # No second coverage check — return best effort
        except Exception as e:
            if _is_rate_limit(e):
                # `answer` is the grounded text — store it verbatim so resume
                # re-runs only the coverage stage.
                _ckpt.save_checkpoint(
                    route="analytical", question=question, params=params,
                    completed_stage="coverage", draft=answer,
                )
                raise _ckpt.QuotaInterrupted(e, _ckpt.MSG_VERIFY_INTERRUPTED)
            raise

        # Prefix answer when requested exercises weren't found in the DB.
        # Partial resolution (some names matched) covers the matched
        # exercises; total failure falls back to the broad package.
        if unresolved_names:
            names_str = ", ".join(unresolved_names)
            if effective_names:
                coverage = ", ".join(effective_names)
                answer = (
                    f"Note: {names_str} wasn't found in your workout history, "
                    f"so this answer covers {coverage}.\n\n"
                    + answer
                )
            else:
                answer = (
                    f"Note: {names_str} wasn't found in your workout history, "
                    f"so this answer covers your overall training instead.\n\n"
                    + answer
                )

        return answer, flagged

    # ── Custom SQL ────────────────────────────────────────────────────────────

    async def _generate_custom_sql(self, question: str, intent: str) -> dict | None:
        """
        Generate and run one supplementary SQL query for a cross-cutting
        question. Returns {"intent", "rows", "row_count"} or None on failure.
        """
        try:
            from src.llm import generate_sql
            from src.schema_prompt import build_schema_prompt
            from src.data_agent import query as run_custom_query

            schema = build_schema_prompt()
            prompt_question = (
                f"{question}\n\n"
                f"Focus: {intent}. Custom SQL is for counts, dates, gaps, "
                f"streaks, and patterns ONLY. Do NOT aggregate weight and do NOT "
                f"compute volume (no SUM/AVG/MIN/MAX/TOTAL over metric_weight or "
                f"metric_weight*reps) — weights and volume come from the "
                f"package's authoritative fields, not from this query. Do not "
                f"return individual set weights."
            )
            sql = await asyncio.to_thread(generate_sql, prompt_question, schema)
            result = await asyncio.to_thread(run_custom_query, sql)
            # Refused (weight/volume aggregate) → no custom data; the analysis
            # falls back to the package's authoritative weight/volume fields.
            if result.get("refused"):
                logger.info("[coordinator] custom SQL refused: %s", result.get("reason"))
                return None
            if result.get("row_count", 0) == 0:
                return None
            return {
                "intent":    intent,
                "rows":      result.get("rows", []),
                "row_count": result.get("row_count", 0),
            }
        except Exception as e:
            # Rate limits propagate so the CLI/server countdown UX fires;
            # any other failure just means no supplementary data.
            if _is_rate_limit(e):
                raise
            return None

    # ── Operational path ──────────────────────────────────────────────────────

    async def _run_operational(self, question: str) -> str:
        """
        Pass question to the existing Single Agent (AgentSession).
        The Single Agent manages its own ReAct loop and conversation history.
        """
        if self._agent is None:
            return (
                "Operational path is not available in this configuration. "
                "Please initialise the Coordinator with an AgentSession."
            )
        result = await self._agent.answer(question)
        return result.get("answer", "")

    # ── Coverage check ────────────────────────────────────────────────────────

    async def _coverage_check(
        self,
        question: str,
        answer:   str,
    ) -> tuple[str, bool]:
        """
        Verify the answer covers all parts of the question.
        No data access — checks coverage only, not correctness.
        Returns (answer, is_complete).
        On failure: returns original answer and True (avoid false retries).
        """
        prompt = f"[QUESTION]\n{question}\n\n[ANSWER]\n{answer}"
        try:
            config = types.GenerateContentConfig(
                system_instruction=_COVERAGE_SYSTEM,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=COORDINATOR_MODEL,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )],
                config=config,
            )
            raw = ""
            _cand  = response.candidates[0] if getattr(response, "candidates", None) else None
            _parts = (_cand.content.parts
                      if _cand and _cand.content and _cand.content.parts else [])
            for part in _parts:
                if getattr(part, "text", None):
                    raw += part.text

            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.strip().startswith("```")
                ).strip()

            result   = json.loads(raw)
            complete = result.get("complete", True)
            missing  = result.get("missing", [])

            if not complete and missing:
                logger.info(
                    "[coordinator] coverage gaps: %s",
                    "; ".join(missing),
                )
            return answer, complete

        except Exception as e:
            if _is_rate_limit(e):
                raise
            logger.warning(
                "[coordinator] coverage check failed: %s — returning original", e
            )
            return answer, True   # fail open: avoid false retry
