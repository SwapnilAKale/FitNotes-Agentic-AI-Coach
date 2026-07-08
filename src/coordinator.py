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
  memory extraction → _run_analytical records (question, final stripped answer)
    via agent.record_external_exchange so session-end extraction sees analytical
    turns too (operational turns already record via agent.answer()).

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
from src import demographic_followup as _followup
from src import memory as _memory

# Rate-limit error types: re-raised instead of swallowed so CLI/server
# countdown UX works for analytical-path quota exhaustion.
try:
    from google.genai import errors as _genai_errors
    _GenaiClientError = _genai_errors.ClientError
    _GenaiServerError = _genai_errors.ServerError
except (ImportError, AttributeError):
    _GenaiClientError = None  # type: ignore[assignment]
    _GenaiServerError = None  # type: ignore[assignment]

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

# ── Transient 503 (model-overload) retry tuning ───────────────────────────────
# A 503 UNAVAILABLE is a transient infra error, unrelated to routing. It carries
# no retryDelay, so we use a fixed small wait (capped by PER_MINUTE_WAIT_CAP) and a
# small retry bound. If it survives the retries, the caller fails clean — it must
# NEVER fall back to the operational lane (operational has no read tools post-strip
# and would fabricate an analytical answer).
TRANSIENT_MAX_RETRIES = 2    # 503 is transient — a couple of capped waits, then clean-fail
TRANSIENT_BACKOFF      = 5    # s — 503 carries no retryDelay; fixed small wait

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


# ── /log deterministic write boundary ─────────────────────────────────────────
# A user-typed "/log" prefix is a TRUSTED write boundary: no inference, no
# regex guessing. The write-intent regex above stays as the graceful-degradation
# fallback for un-prefixed logging messages (never silent failure) — in that
# fallback case only, the answer carries a one-line /log suggestion.
_LOG_PREFIX_RE = re.compile(r"(?i)^\s*/log\b[:,]?\s*")
_LOG_FALLBACK_NUDGE = (
    "Tip: starting your message with /log makes logging faster and more reliable."
)
_LOG_TRAILING_NOTE = (
    "(Noted your other question — ask it again after confirming this log.)"
)

# Tail-question test for _split_log_tail. Dedicated regex (NOT a change to
# _WRITE_QUESTION_RE): "Also, how is my back progressing" has no "?", "how is"
# isn't in _WRITE_QUESTION_RE's alternations, and the leading "Also" breaks its
# ^ anchor. Optional connector, then an interrogative lead — or a "?" anywhere.
_LOG_TAIL_QUESTION_RE = re.compile(
    r"(?i)^\s*(?:also,?\s+|and\s+also,?\s+|btw,?\s+|by\s+the\s+way,?\s+"
    r"|oh\s+and\s+|plus,?\s+)?"
    r"(?:how|what|why|when|where|which|who|is|are|am|do|does|did"
    r"|can|could|should|would|will)\b"
    r"|\?"
)
_LOG_QUANTITY_RE = re.compile(r"(?i)" + _WRITE_QUANTITY)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_log_tail(text: str) -> tuple[str, bool]:
    """
    Split a /log request into (workout head, has_analytical_tail). Peels
    trailing sentences that read as questions (and carry no set/rep/weight
    quantity — those are workout content, never a tail). Conservative: false
    negatives are fine; the tail is never routed or decomposed — the only
    acknowledgment is _LOG_TRAILING_NOTE appended to the answer. If every
    sentence would peel (head would be empty), don't split at all.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    keep = len(parts)
    while keep > 1:
        seg = parts[keep - 1]
        if _LOG_TAIL_QUESTION_RE.search(seg) and not _LOG_QUANTITY_RE.search(seg):
            keep -= 1
        else:
            break
    if keep == len(parts):
        return text, False
    return " ".join(parts[:keep]).strip(), True


def _log_carry_unrelated(message: str) -> bool:
    """
    Clear-without-consume test for the single-turn /log carry: a question is
    not a clarification answer, and a bare filler ("thanks") is abandonment.
    Anything else ("yesterday", "3 sets of 12", "the dumbbell one") plausibly
    answers the pending logging clarification and joins the /log flow.
    """
    return bool(_WRITE_QUESTION_RE.search(message)) or _filler_reply(message) is not None


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


# ── Clean-fail messages for analytical-pipeline failures ──────────────────────
# These ship to the user when the analytical pipeline cannot complete. The lane is
# NEVER downgraded to operational (operational has no read tools post-strip and
# would fabricate an analytical answer). No "say continue" — these are terminal
# clean failures, not checkpoint/resume statuses.
_MSG_MODEL_BUSY = (
    "The analysis service is busy right now (high demand). "
    "I couldn't complete that analysis — please try again in a moment."
)
_MSG_PIPELINE_ERROR = (
    "I couldn't complete that analysis right now. "
    "Please try again, or contact support if this keeps happening."
)


def _is_rate_limit(exc: Exception) -> bool:
    """True if exc is a Gemini/google-api 429 / ResourceExhausted error."""
    if _ResourceExhausted is not None and isinstance(exc, _ResourceExhausted):
        return True
    if _GenaiClientError is not None and isinstance(exc, _GenaiClientError):
        if getattr(exc, "code", None) == 429:
            return True
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _is_transient_server_error(exc: Exception) -> bool:
    """
    True ONLY for a transient 503 / UNAVAILABLE model-overload error (retryable).

    Deliberately NOT 500/INTERNAL: a 500 may be a genuine bug, not a transient
    overload, so it must fail clean rather than be retried (and never reroute).
    502/504 are likewise excluded for now — match 503/UNAVAILABLE only. 5xx errors
    surface as google.genai ServerError (4xx, incl. 429, are ClientError), so the
    isinstance check disambiguates 503 from 500 via .code/.status; the string
    fallback covers cases where the genai types are unavailable.
    """
    if _GenaiServerError is not None and isinstance(exc, _GenaiServerError):
        code   = getattr(exc, "code", None)
        status = (getattr(exc, "status", "") or "").upper()
        return code == 503 or status == "UNAVAILABLE"   # ServerError(500) → False
    msg = str(exc)
    return "503" in msg or "UNAVAILABLE" in msg          # "500" not matched


def _normalize_cardio_lock(raw: Optional[dict]) -> Optional[dict]:
    """
    Deterministic unit normalization for a cardio PR lock (6b) — NOT the LLM.
    The classifier emits {field, value, unit} (e.g. {"distance",5,"km"},
    {"duration",10,"min"}); convert to base units the data layer expects:
      duration → SECONDS  (min→×60; s/sec→as-is)
      distance → KM       (km→as-is; m→/1000; mi/mile→×1.609)
    Returns {"field","value"} or None when field/value are missing/invalid.
    """
    if not isinstance(raw, dict):
        return None
    field = raw.get("field")
    value = raw.get("value")
    if field not in ("distance", "duration") or not isinstance(value, (int, float)):
        return None
    unit = (raw.get("unit") or "").strip().lower()
    if field == "duration":
        secs = value * 60 if unit in ("min", "minute", "minutes") else value
        return {"field": "duration", "value": secs}
    # distance → km
    if unit in ("m", "meter", "meters", "metre", "metres"):
        km = value / 1000
    elif unit in ("mi", "mile", "miles"):
        km = value * 1.609
    else:                                   # km (default) / unspecified
        km = value
    return {"field": "distance", "value": km}


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
    "Show me my last chest session"          (session display — muscle group)
    "Show me my last Lat Pulldown session"   (session display — one exercise)
    "How was my back ROM split in the last back session"  (category session display)

OPERATIONAL — use only for questions that require MCP tools:
  - Write operations: logging, goal setting, corrections, deletions
  - Research: fitness science questions

  Examples → operational:
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
    Session display      — what was done on a specific date / most-recent
                           session, with full set breakdown, for ONE exercise
                           OR all exercises in a muscle group

  Route OPERATIONAL only for things that require MCP tools:
    Writes      — log workout, set goal, update set, delete anything
    Research    — fitness science questions, what does science say

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
two cases listed above (writes/corrections/goals, and research/RAG), and
"out_of_scope" only per the test above. Session display — single-exercise or
muscle-group level — is ANALYTICAL.
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
  rep_target:        integer rep count ONLY when the question asks for a PR at a
                     specific rep count ("5-rep PR", "PR for 3 reps", "best set of 8")
                     → 5 / 3 / 8. null otherwise (an ordinary "what's my bench PR" is null).
  cardio_lock:       for a cardio PR that fixes one quantity, an object
                     {"field": "distance"|"duration", "value": <number>, "unit": "<unit>"};
                     null otherwise. Examples: "fastest 5km" → {"field":"distance","value":5,"unit":"km"};
                     "most distance in 10 minutes" → {"field":"duration","value":10,"unit":"min"}.
                     Emit the value and unit as stated — do NOT convert units yourself.

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
  "rep_target": 5 | null,
  "cardio_lock": {"field": "distance"|"duration", "value": 5, "unit": "km"} | null,
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


# ── Stage-2 staging-verify prompt (write-verification layer) ──────────────────
# ONE LLM diff call per logging turn: the assembled /log-flow user turns ⇄ the
# staged JSON slot (+ its deterministic rendering — the raw slot alone is
# un-diffable: exercise_id is an opaque int and metric_weight is kg, typed/2.2046).
# Verification, NOT re-extraction: the model checks whether B faithfully
# represents A; it never re-derives B from A. No retries (quota: +1 call/turn).

_VERIFY_SYSTEM = """
You are a verification checker for a workout-logging system.

You receive:
  [LOGGING REQUEST]  — the user's own words asking to log a workout (possibly
                       assembled from several turns, in order).
  [STAGED JSON]      — the machine-parsed workout batch that will be written.
  [STAGED RENDERING] — a deterministic human-readable rendering of that same
                       JSON (exercise names and typed weights recovered from
                       the database).

Your task is a DIFF, not a re-parse: check whether the staged JSON faithfully
represents the request. Do NOT re-derive the workout from the request yourself
and do NOT judge whether the workout is sensible. Only compare.

FAIL when:
  - a set, exercise, or per-set comment in the request is missing from the JSON
  - the JSON contains a set, exercise, or comment the request never asked for
  - a value is misattributed (weight/reps/comment on the wrong set or exercise)
  - the date contradicts the request

Do NOT flag:
  - resolved exercise names ("bench" resolved to "Flat Barbell Bench Press")
  - unit conversion (metric_weight in the JSON is kilograms = typed pounds
    divided by 2.2046; the RENDERING shows the typed value — trust the
    rendering for weights)
  - exercise_id integers (opaque database ids; the RENDERING carries the name)
  - non-logging content in the request (questions, asides) — verify the
    logging portion only

Return ONLY valid JSON, no preamble, no markdown fences:
  {"verdict": "PASS"}
  {"verdict": "FAIL", "reason": "<one short sentence: what is missing / extra / misattributed>"}
""".strip()

# FAIL response — a re-state prompt in the existing message style, not a raw error.
MSG_VERIFY_RESTATE = "I may have misread that — could you re-state the workout?"

# ── Write-success claim gate (Fix: false "successfully logged") ───────────────
# Claim-PRESENCE detector only. The DECIDING gate is structural — the guard in
# _run_operational can fire only when the turn deterministically wrote nothing
# (db_write_effect False), staged nothing (staged_this_turn False), and
# attempted no execute (staging_reached_confirm False). In that state any
# completed-write claim is false by construction, so a regex false-positive is
# harmless (the replacement is still truthful) and a false-negative merely
# preserves the old behavior. The regex can never suppress a legitimate answer
# (e.g. a clarification question), because legitimacy is decided by the flags.
_WRITE_SUCCESS_CLAIM_RE = re.compile(
    r"(?i)(?:"
    r"successfully\s+(?:logg|sav|record|writ|add|updat|delet)"
    r"|(?:logged|saved|recorded|written|added)\b[^.\n]{0,60}?"
    r"(?:successfully|to\s+your\s+(?:database|log|workout|fitnotes))"
    r"|has\s+been\s+(?:logged|saved|recorded|written|added)"
    r"|✅[^\n]{0,80}(?:logged|saved|recorded|written)"
    r")"
)

MSG_NO_WRITE_OCCURRED = (
    "⚠️ Nothing was written to your database this turn — no write was "
    "executed. Please re-state your logging request (tip: start with /log)."
)


def format_verify_fail_message(preview: str) -> str:
    return (
        "Here's what I staged — but it didn't match your request:\n\n"
        f"{preview}\n\n"
        "Could you confirm this is right, or re-state the workout?"
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
        # Stage B: a one-turn pending demographic follow-up — {"key", "clarified"}.
        # In-memory, per session. Lives exactly the immediate next turn (+1 turn
        # only if the user attempts an answer that needs one clarification).
        self._pending_followup: dict | None = None
        # /log follow-up carry: set when a /log-boundary turn ends WITHOUT a
        # complete staged batch (the agent asked a logging clarification — date,
        # name disambiguation, ambiguous sets/reps, any of them). The immediate
        # next turn joins the /log flow without the prefix. Single-turn by
        # construction: route() snapshot-and-clears it at every entry, and only
        # _run_operational re-arms it (on a boundary turn that again ended
        # pending clarification). Coordinator-level (not server _state) so the
        # CLI and web interfaces share the same seam.
        self._pending_log_carry: bool = False
        # Stage-2 verify Input A: the assembled WORKOUT-PORTION user turns of the
        # current /log flow (turn-1 stripped head + carry replies, in order).
        # Same scope-by-construction discipline as _pending_log_carry: route()
        # snapshot-and-clears it at every entry; only the three flow-shaped
        # branches in _route_fresh rebuild it (prefix, carry-consume, fallback
        # write). Any other turn shape leaves it cleared — a flow never leaks
        # stale turns into the next one.
        self._log_flow_turns: list[str] = []

    # ── Public entry point ────────────────────────────────────────────────────

    async def route(self, question: str) -> dict:
        """
        Public entry point. Wraps the checkpoint+routing flow with the Stage-B
        demographic follow-up layer:
          PRE  — if a follow-up is pending, the user's reply either answers it
                 (ack/clarify, this IS the turn's response) or doesn't (drop it,
                 route the message normally).
          POST — after a real answer, optionally append ONE gentle follow-up
                 offering to remember a demographic anchor the user just mentioned.
        The follow-up is a pure presentation addendum: it is appended AFTER the
        answer was already recorded into history (record_external_exchange /
        agent.answer), so an unanswered aside never enters extraction/grounding
        history.
        """
        # ── PRE: consume or drop a pending follow-up (before checkpoint logic).
        # A bare follow-up answer ("2003-06-18") is never a continue/discard
        # intent, so handling it here can't collide with the checkpoint block.
        if self._pending_followup is not None:
            consumed = self._consume_pending_followup(question)
            if consumed is not None:
                return consumed
            self._pending_followup = None     # not an answer → drop, route normally

        # ── /log carry: snapshot-and-clear at every entry. The flag can never
        # outlive one turn — a followup-consumed turn, checkpoint prompt, or
        # filler reply all count as abandonment (cleared, never consumed).
        # _route_fresh decides consume vs. clear-without-consume; only
        # _run_operational can re-arm it.
        log_carry = self._pending_log_carry
        self._pending_log_carry = False

        # ── Flow turns: same snapshot-and-clear as the carry. The snapshot holds
        # the boundary turn's workout portion for a carry turn to extend; every
        # non-flow turn shape (followup, checkpoint prompt, filler, analytical)
        # leaves the member cleared, so the list's scope is one flow by
        # construction — never a branch-local reset obligation.
        flow_turns = self._log_flow_turns
        self._log_flow_turns = []

        result = await self._route_with_checkpoint(
            question, log_carry=log_carry, flow_turns=flow_turns)

        # ── POST: append at most one follow-up (only on a real answer).
        if (result.get("route") in ("analytical", "operational")
                and result.get("answer")):
            self._append_followup_if_relevant(question, result)
        return result

    # ── Stage B: demographic follow-up helpers ─────────────────────────────────

    def _consume_pending_followup(self, question: str) -> dict | None:
        """
        Interpret the user's immediate next reply against the pending follow-up.
        Returns the turn's response dict (ack / clarification) when the reply is
        an answer-attempt, or None when it isn't (caller drops + routes normally).
        """
        pending = self._pending_followup
        key = pending["key"]
        r = _followup.interpret_answer(key, question)

        if r["kind"] == "answer":
            res = _memory.set_demographic(key, r["value"], unit=r.get("unit"))
            if res.get("status") == "saved":
                self._pending_followup = None
                return self._followup_response(_followup.ACK, "followup_ack")
            # parsed but the validator rejected it → treat like a malformed attempt
            r = {"kind": "clarify"}

        if r["kind"] == "clarify":
            if not pending.get("clarified"):
                pending["clarified"] = True          # one clarification, one more turn
                return self._followup_response(
                    _followup.CLARIFY.get(key, ""), "followup_clarify")
            self._pending_followup = None            # already clarified → give up
            return None

        return None                                   # not_answer → drop, route normally

    def _append_followup_if_relevant(self, question: str, result: dict) -> None:
        """
        After a real answer, if the user mentioned a demographic anchor that ISN'T
        already stored, append ONE gentle new-line offer to remember it and mark
        it pending. (Pending is always None here — the PRE-step cleared it — so at
        most one follow-up is ever active: no stacking.)
        """
        for key in _followup.detect_mentions(question):
            if _memory.get_demographic(key) is None:
                result["answer"] = result["answer"].rstrip() + "\n\n" + \
                    _followup.followup_text(key)
                self._pending_followup = {"key": key, "clarified": False}
                return

    def _followup_response(self, text: str, route: str) -> dict:
        return {"answer": text, "route": route, "flagged_claims": [], "error": None}

    async def _route_with_checkpoint(self, question: str,
                                     log_carry: bool = False,
                                     flow_turns: Optional[list] = None) -> dict:
        """
        The checkpoint + routing flow (resume / confirm-before-discard / classify
        + dispatch). Wrapped by route(), which adds the Stage-B follow-up layer.

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

        return await self._route_fresh(question, log_carry=log_carry,
                                       flow_turns=flow_turns)

    async def _route_fresh(self, question: str, log_carry: bool = False,
                           flow_turns: Optional[list] = None) -> dict:
        """
        Classify and dispatch a NEW (or classify-resumed) question. Separated
        from route()'s checkpoint handling so a resume from
        completed_stage='classify' can re-enter here directly (re-running the
        cheap classify call) without re-triggering checkpoint logic.

        Facade over the parent coordinator graph
        (src/graph/coordinator_graph.py): entry_boundary evaluates the
        deterministic /log-boundary, carry, filler, and write-intent
        pre-guards BEFORE any LLM; a conditional edge routes to the classify
        router node or short-circuits straight to the operational dispatch.
        There is no analytical→operational edge — the clean-fail handling
        lives inside the dispatch_analytical node. Node bodies are the
        _node_* methods below (verbatim code motion); return-dict contract
        unchanged.
        """
        from src.graph.coordinator_graph import get_coordinator_graph
        from src.graph.persistence import cleanup_turn, new_turn_id, turn_config
        from src.graph.state import GraphRunContext, RunCache

        turn_id = new_turn_id()
        state = await get_coordinator_graph().ainvoke(
            {
                "question": question,
                "log_carry": log_carry,
                "flow_turns": list(flow_turns or []),
            },
            turn_config(turn_id),
            context=GraphRunContext(coordinator=self, cache=RunCache()),
        )
        cleanup_turn(turn_id)
        return state["result"]

    # ── Parent graph node bodies (verbatim code motion) ──────────────────────

    def _node_entry_boundary(self, state: dict) -> dict:
        """Node: the deterministic pre-LLM guards — /log boundary, carry
        consume, filler short-circuit, write-intent regex — evaluated BEFORE
        the classify router (the conditional edge reads this node's output)."""
        question   = state["question"]
        log_carry  = state.get("log_carry", False)
        flow_turns = state.get("flow_turns")

        # ── 0a. /log deterministic write boundary ────────────────────────────
        # Detected HERE (not route()) so a "/log ..." stashed as the checkpoint
        # discard-confirm pending_question re-detects intact when re-processed.
        # Prefix turn: strip the prefix, peel a trailing analytical tail (the
        # note is its only acknowledgment — never routed, never decomposed).
        # Carry turn: the previous /log turn ended pending a logging
        # clarification, so this reply joins the flow — unless it reads as
        # unrelated (question / filler), which clears without consuming.
        log_boundary = False
        trailing_note = False
        m = _LOG_PREFIX_RE.match(question)
        if m:
            log_boundary = True
            question = question[m.end():].strip()
            question, trailing_note = _split_log_tail(question)
            # Fresh flow: Input A restarts at this turn's workout portion (the
            # stripped head — the peeled tail is never part of the verify).
            self._log_flow_turns = [question]
        elif log_carry and not _log_carry_unrelated(question):
            log_boundary = True                       # consume the carry
            # Carry turn extends the flow: the route() snapshot holds the
            # boundary turn's workout portion; this reply joins it in order.
            self._log_flow_turns = list(flow_turns or []) + [question]

        # ── 0b. Filler short-circuit (#5a) ───────────────────────────────────
        # Bare greeting / ack / thanks / empty-ish → cheap canned reply, NO
        # classify call, NO package, NO analytical/operational pipeline. Runs
        # BEFORE classification. Conservative: anything with real content (incl.
        # a question behind a polite prefix) falls through to normal routing.
        # Skipped on a /log-boundary turn: a trusted write boundary is never
        # filler (a bare "/log" should reach the agent and get a clarification).
        if not log_boundary:
            filler = _filler_reply(question)
            if filler is not None:
                return {
                    "question":      question,
                    "log_boundary":  log_boundary,
                    "trailing_note": trailing_note,
                    "result":        self._filler_response(filler),
                }

        # ── 1. Deterministic write short-circuit (pre-guard, BEFORE classify) ─
        # Misrouting a write to analytical bypasses the confirmation gate.
        # /log boundary = trusted user signal, no inference. The regex guard
        # below it is UNTOUCHED and now the fallback: it fires only when no
        # /log prefix (and no carry), and only then the answer gets the nudge.
        # A non-None params here makes the conditional edge skip the classify
        # node entirely — the LLM router is never spent on a real write.
        fallback_write = False
        params = None
        if log_boundary:
            params = {
                "route":             "operational",
                "exercise_names":    None,
                "muscle_groups":     None,
                "query_period_days": 90,
                "needs_custom_sql":  False,
                "custom_sql_intent": None,
            }
        elif _is_write_intent(question):
            fallback_write = True
            # Regex-inferred write: the flow is this single message.
            self._log_flow_turns = [question]
            params = {
                "route":             "operational",
                "exercise_names":    None,
                "muscle_groups":     None,
                "query_period_days": 90,
                "needs_custom_sql":  False,
                "custom_sql_intent": None,
            }

        return {
            "question":       question,
            "log_boundary":   log_boundary,
            "trailing_note":  trailing_note,
            "fallback_write": fallback_write,
            "params":         params,
        }

    async def _node_classify(self, state: dict) -> dict:
        """Node: the classify LLM router — reached only when no deterministic
        pre-guard fired (the conditional edge enforces the old statement
        order structurally)."""
        question = state["question"]
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

        # Parse-failure (#5b) and out_of_scope are routed by the conditional
        # edge on this node's output: unparseable input never builds a
        # ~358KB package, an out-of-scope question costs only the classify —
        # the refusal IS the classifier's output.
        return {"params": params}

    async def _node_dispatch_analytical(self, state: dict) -> dict:
        """Node: the analytical lane dispatch + clean-fail contract. There is
        NO edge from here to the operational lane: an exception aborts or
        degrades to a clean-fail message, it never switches lanes."""
        question = state["question"]
        params   = state["params"]
        flagged  = []
        error    = None

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
            # An analytical-pipeline failure NEVER falls back to operational:
            # post-strip, operational has no read tools and would fabricate an
            # answer. Mirror the resume path's contract — retry/clean-fail only.
            if _is_rate_limit(e):
                raise                              # 429 → server countdown (unchanged)
            elif _is_transient_server_error(e):
                # The per-stage retry already tried; this 503 is persistent.
                # Clean-fail (no checkpoint, no reroute) — say try again.
                logger.warning(
                    "[coordinator] analytical pipeline hit a transient 503: %s", e)
                error  = str(e)
                answer = _MSG_MODEL_BUSY
            else:
                # Genuine pipeline bug — surface a clean failure, never operational.
                # logger.exception captures the traceback: the user only sees the
                # clean _MSG_PIPELINE_ERROR, so the log is the sole debug surface.
                logger.exception("[coordinator] analytical pipeline failed: %s", e)
                error  = str(e)
                answer = _MSG_PIPELINE_ERROR
            # route stays "analytical": the request WAS analytical and simply
            # could not complete.
        return {"answer": answer, "flagged_claims": flagged, "error": error}

    async def _node_dispatch_operational(self, state: dict) -> dict:
        """Node: the operational lane dispatch. _run_operational resolves on
        the instance at call time (byte-identical method — it arms the /log
        carry and enriches the checkpoint on QuotaInterrupted exactly as
        before)."""
        answer = await self._run_operational(
            state["question"],
            log_boundary=state.get("log_boundary", False),
            fallback_write=state.get("fallback_write", False),
            trailing_note=state.get("trailing_note", False),
        )
        return {"answer": answer, "flagged_claims": [], "error": None}

    def _node_finalize_turn(self, state: dict) -> dict:
        """Node: conversation-history update + the contracted return dict.
        Terminal short-circuits (filler / unparseable / out-of-scope) bypass
        this node, so they never enter history — the hand-built behavior."""
        question       = state["question"]
        answer         = state["answer"]
        log_boundary   = state.get("log_boundary", False)
        fallback_write = state.get("fallback_write", False)

        # ── 3. Update conversation history ────────────────────────────────────
        self._history.append({"role": "user",      "content": question})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > CONTEXT_WINDOW * 2:
            self._history = self._history[-(CONTEXT_WINDOW * 2):]

        return {"result": {
            "answer":         answer,
            "route":          (state.get("params") or {}).get("route", "analytical"),
            "flagged_claims": state.get("flagged_claims") or [],
            "error":          state.get("error"),
            "log_boundary":   log_boundary,
            # Stage-2 verify Input A: the assembled workout-portion turns of
            # this flow, present only on write-shaped turns (callers fall back
            # to the raw message when absent).
            "log_flow_turns": (list(self._log_flow_turns)
                               if (log_boundary or fallback_write) else None),
        }}

    # ── Per-minute silent retry ───────────────────────────────────────────────

    async def _call_with_per_minute_retry(self, fn, *args, **kwargs):
        """
        Run an async LLM stage, absorbing PER-MINUTE 429s silently: wait the
        provider's retryDelay (capped at PER_MINUTE_WAIT_CAP) and retry the SAME
        call up to PER_MINUTE_MAX_RETRIES, holding the request open — no
        checkpoint, no message, the user keeps seeing 'thinking…'. A DAILY 429,
        or exhausted per-minute retries, propagates unchanged so the caller's
        daily path (checkpoint + QuotaInterrupted) runs.

        Also absorbs TRANSIENT 503s (model overload): a fixed capped wait, retried
        up to TRANSIENT_MAX_RETRIES (503 carries no retryDelay). An exhausted 503
        propagates unchanged so the caller fails clean — it is NEVER rerouted to
        operational.

        NOTE: agent_lock is held for the duration of the wait (single-user
        assumption — a concurrent /chat gets the existing 'busy' 429).
        """
        attempt = 0
        transient_attempt = 0
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
                if (_is_transient_server_error(e)
                        and transient_attempt < TRANSIENT_MAX_RETRIES):
                    wait = min(TRANSIENT_BACKOFF, PER_MINUTE_WAIT_CAP) + PER_MINUTE_BUFFER
                    logger.warning(
                        "[coordinator] transient 503 — waiting %ds then retrying "
                        "(attempt %d/%d), request held open",
                        wait, transient_attempt + 1, TRANSIENT_MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    transient_attempt += 1
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
            # Three-way write-path resume: a checkpointed staged batch is
            # RESTORED (cases 1 and 2 — with/without a PASS verdict), never
            # re-staged by the agent (a fresh probabilistic extraction that
            # can diverge from what stage-2 already verified). Only a slot-less
            # checkpoint (case 3) re-enters the agent loop.
            if cp.get("staged_slot"):
                return await self._resume_staged_workout(cp)
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

    async def _resume_staged_workout(self, cp: dict) -> dict:
        """
        Restore a checkpointed staged workout batch (write-path resume, cases
        1 and 2). The exact pre-interruption slot — id-validated against the
        CURRENT DB by the restore tool (a backup upload inside the slot's 48h
        window can invalidate exercise_ids) — is written back into the MCP
        subprocess, and the caller (server/CLI) arms its confirm gate from the
        returned restore signal WITHOUT any agent call.

        Case 1 (verify_verdict PASS stored): the batch was already verified —
        the caller skips the verify LLM call entirely. Case 2 (no verdict —
        the 429 hit after staging, before verify): the caller's existing
        stage-2 verify block runs NOW on the restored slot, with the restored
        log_flow_turns as Input A (never ["continue"]).

        Checkpoint lifecycle is keep-until-confirm: the slot is NOT cleared
        here — /confirm execute-success, cancel, or a verify FAIL clears it
        (guarded), so an abandoned panel or a restart stays resumable and a
        committed batch can never be restored twice.
        """
        orig_q  = cp.get("question") or ""
        verdict = cp.get("verify_verdict")
        flow    = cp.get("log_flow_turns") or ([orig_q] if orig_q else [])
        try:
            restore = json.loads(await self._agent.call_tool(
                "restore_staged_workout_slot",
                {"staged_workouts": cp["staged_slot"], "verify": verdict},
            ))
        except Exception as e:
            logger.warning("[coordinator] staged-slot restore failed: %s", e)
            restore = {"error": str(e)}

        if restore.get("error"):
            # Invalid ids (DB replaced since checkpoint) or a restore failure:
            # never confirm a batch with dangling references — drop the slot
            # and ask for a re-log.
            logger.warning(
                "[coordinator] staged-slot restore refused: %s", restore)
            _ckpt.clear_checkpoint()
            answer = ("Your data changed since this workout was staged — "
                      "please re-log it with /log.")
            self._history.append({"role": "user",      "content": orig_q})
            self._history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer, "route": "operational",
                "flagged_claims": [], "error": None,
            }

        self._agent._staged_active = True
        answer = "Restored your staged workout — confirm below to save it."
        self._history.append({"role": "user",      "content": orig_q})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > CONTEXT_WINDOW * 2:
            self._history = self._history[-(CONTEXT_WINDOW * 2):]
        return {
            "answer":            answer,
            "route":             "operational",
            "flagged_claims":    [],
            "error":             None,
            "log_boundary":      True,
            "log_flow_turns":    flow,
            "restore_staged":    True,        # non-agent gate-arming signal
            "restored_verify":   verdict,     # PASS dict (case 1) or None (case 2)
            "restored_question": orig_q,      # checkpoint-2 re-save must never
        }                                     # degrade question to "continue"

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
            params.setdefault("rep_target",        None)
            params.setdefault("cardio_lock",       None)
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
        Full analytical pipeline — facade over the analytical subgraph
        (src/graph/analytical.py), a linear node chain:
          resolve_scope → build_package → draft → ground → coverage →
          display_fidelity
        Each node body is a _stage_* method below (verbatim code motion).

        resume: a checkpoint slot dict. The package always rebuilds (free,
        re-validated — G6 etc. still apply) inside build_package; a stored
        draft (state resume_draft) skips the draft LLM call and is verified
        VERBATIM; completed_stage=="coverage" skips grounding too. The
        package itself rides in the non-persisted RunCache — it never enters
        graph state, so the SqliteSaver checkpoints hold only params + the
        verbatim draft/answer.

        Stage-boundary 429 handling: each LLM stage saves a checkpoint of
        the last COMPLETED stage and raises QuotaInterrupted (status text
        only — never draft content); the exception aborts the graph run and
        propagates unchanged.

        Returns (final_answer, flagged_claims).
        """
        from src.graph.analytical import get_analytical_graph
        from src.graph.persistence import cleanup_turn, new_turn_id, turn_config
        from src.graph.state import GraphRunContext, RunCache

        turn_id = new_turn_id()
        state = await get_analytical_graph().ainvoke(
            {
                "question": question,
                "params": params,
                "resume_completed_stage": (resume or {}).get("completed_stage"),
                "resume_draft": (resume or {}).get("draft"),
            },
            turn_config(turn_id, "analytical"),
            context=GraphRunContext(coordinator=self, cache=RunCache()),
        )
        cleanup_turn(turn_id)
        if state.get("early_answer") is not None:
            return state["early_answer"], []
        return state["answer"], state.get("flagged", [])

    # ── Analytical graph node bodies (verbatim code motion) ──────────────────

    async def _stage_resolve_scope(self, state: dict) -> dict:
        """Node: Category guard + exercise-name resolution + canonical
        muscle groups (old pipeline head). Disambiguation produces
        early_answer — the graph exits without building the package."""
        params = state["params"]
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
                # READ path → permissive: auto-resolve a clear-margin LIKE-tier
                # winner (e.g. "walk" → Walking) instead of disambiguating. The
                # write/MCP path keeps the strict default (a wrong write is
                # unrecoverable).
                result = resolve_exercise_name(name, db_path, permissive=True)
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
                    return {"early_answer": (
                        f"I found multiple exercises matching **{name}**. "
                        f"Which one did you mean?\n\n{names_list}\n\n"
                        f"Please let me know and I'll answer your question."
                    )}
                else:
                    # No match — keep original so the package reports it as unresolved
                    resolved.append(name)
            exercise_names = resolved
        # muscle_groups already computed above by the Category guard (canonical,
        # including any category terms moved out of exercise_names).
        return {
            "exercise_names": exercise_names,
            "muscle_groups": muscle_groups or None,
        }

    async def _stage_build_package(self, state: dict, cache) -> dict:
        """Node: ONE pure call builds the analytical package (Data Agent
        fetch/process/validate stay inside prepare_analysis_package — never
        decomposed); scope notes + memories + conversation context follow.
        The package lands in the non-persisted RunCache, NOT in state."""
        question          = state["question"]
        params            = state["params"]
        exercise_names    = state.get("exercise_names")
        muscle_groups     = state.get("muscle_groups")
        query_period_days = params.get("query_period_days", 90)

        # 6b: parameterized PR targets. rep_target threads straight through; the cardio
        # lock is unit-normalized here (deterministic Python — the LLM emits {value,unit},
        # never seconds). Both default None → the unchanged static PR.
        reps_floor  = params.get("rep_target")
        cardio_lock = _normalize_cardio_lock(params.get("cardio_lock"))

        # Build compact package (scope-aware trim: BROAD 365d ≈ 396 KB)
        pkg = await asyncio.to_thread(
            prepare_analysis_package,
            query_period_days=query_period_days,
            exercise_names=exercise_names,
            muscle_groups=muscle_groups,
            include_phase2=True,
            reps_floor=reps_floor,
            cardio_lock=cardio_lock,
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

        cache.pkg                  = pkg
        cache.research             = research
        cache.memories             = memories
        cache.conversation_context = conversation_context
        return {
            "scoped_question":  scoped_question,
            "unresolved_names": unresolved_names,
            "effective_names":  effective_names,
        }

    async def _stage_draft(self, state: dict, cache) -> dict:
        """Node: supplementary SQL + analyze → stripped draft + grounding
        context (in cache). A stored draft (resume) is used VERBATIM — the
        draft LLM call is skipped entirely; only the remaining verification
        stages run."""
        question        = state["question"]
        params          = state["params"]
        scoped_question = state["scoped_question"]
        pkg             = await cache.ensure_package(self, state)
        research        = cache.research
        memories        = cache.memories
        conversation_context = cache.conversation_context

        custom_query = None
        draft = state.get("resume_draft")
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

        cache.gctx         = gctx
        cache.custom_query = custom_query
        return {"draft": draft}

    async def _stage_ground(self, state: dict, cache) -> dict:
        """Node: grounding verification. POLICY: the user never sees
        unverified draft text. On interruption the verbatim draft is
        checkpointed and only a status message ships."""
        question = state["question"]
        params   = state["params"]
        draft    = state["draft"]
        flagged: list = []
        if state.get("resume_completed_stage") == "coverage":
            # Grounding completed before the interruption — the stored text
            # is already verified; only the coverage stage remains.
            answer = draft
        else:
            gctx = cache.gctx
            if gctx is None:
                # Defensive resume path: grounding context is package-derived
                # and never persisted — rebuild the whole-package fallback,
                # exactly like the stored-draft branch.
                gctx = _cite.build_grounding_context(
                    [], await cache.ensure_package(self, state))
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
        return {"answer": answer, "flagged": flagged}

    async def _stage_coverage(self, state: dict, cache) -> dict:
        """Node: coverage check (question + answer only, no data; 1 retry
        re-running analyze+ground INSIDE this node — the chain stays
        linear, exactly as the hand-built code did)."""
        question        = state["question"]
        params          = state["params"]
        scoped_question = state["scoped_question"]
        answer          = state["answer"]
        flagged         = list(state.get("flagged") or [])
        pkg             = await cache.ensure_package(self, state)
        research        = cache.research
        memories        = cache.memories
        conversation_context = cache.conversation_context
        custom_query    = cache.custom_query
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
        return {"answer": answer, "flagged": flagged}

    async def _stage_display_fidelity(self, state: dict, cache) -> dict:
        """Node: deterministic display-sets verbatim-integrity check +
        unresolved-names prefix + memory-extraction recording (old pipeline
        tail)."""
        question         = state["question"]
        scoped_question  = state["scoped_question"]
        answer           = state["answer"]
        unresolved_names = state.get("unresolved_names")
        effective_names  = state.get("effective_names")
        pkg              = await cache.ensure_package(self, state)
        research         = cache.research
        memories         = cache.memories
        conversation_context = cache.conversation_context
        custom_query     = cache.custom_query

        # ── Stage: DISPLAY SETS CHECK (deterministic verbatim-integrity) ───────
        # When the package carries pre-formatted display strings, grounding
        # ignores them (correct — not citable). This check guarantees they
        # survive verbatim into the answer: re-prompt ONCE for a verbatim
        # reproduction, else assemble them raw from the package (no LLM).
        if pkg.get("display_sets"):
            base_answer = answer

            async def _reframe_display():
                reframe_context = list(conversation_context or []) + [
                    {"role": "assistant", "content": base_answer},
                    {"role": "user", "content": (
                        "Reproduce the per-set display lines from the package EXACTLY "
                        "and VERBATIM — character-for-character, including weights, "
                        "reps, comments, and arrows. Do not paraphrase, round, or "
                        "summarise them."
                    )},
                ]
                tagged = await self._call_with_per_minute_retry(
                    analysis_agent.analyze,
                    pkg, scoped_question, research, memories,
                    reframe_context, custom_query)
                return _cite.strip_tags(tagged)

            try:
                answer = await analysis_agent.enforce_display_fidelity(
                    base_answer, pkg, _reframe_display)
            except Exception as e:
                # The re-frame is an LLM call; if it fails, APPEND the deterministic
                # raw block to the existing answer so the strings reach the user
                # WITHOUT discarding the generated analysis (repair, never replace).
                logger.warning(
                    "[coordinator] display-fidelity re-frame failed (%s) — appending raw block", e)
                answer = (analysis_agent.append_missing_display(base_answer, pkg)
                          if analysis_agent.display_sets_missing(base_answer, pkg)
                          else base_answer)

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

        # Record the analytical turn so memory extraction sees it. The analytical
        # path runs the analysis pipeline directly and never enters the agent's
        # operational answer() loop, so without this the (question, answer) is
        # invisible to _auto_extract_memories. Feed the FINAL STRIPPED answer (the
        # text the user saw) + the original question — not the tagged draft, the
        # cited-values payload, or the package. out_of_scope / filler /
        # parse-failure short-circuit before _run_analytical, so they record
        # nothing. (Operational turns already record via agent.answer().)
        if self._agent is not None and answer:
            self._agent.record_external_exchange(question, answer)

        return {"answer": answer}

    async def _rebuild_package_for_state(self, state: dict) -> dict:
        """Rebuild the analytical package from persisted params + resolved
        scope (RunCache.ensure_package's rebuild-on-resume path). Pure
        Python, re-validated — the same call expression as
        _stage_build_package, through the same module-global seam."""
        params = state.get("params") or {}
        pkg = await asyncio.to_thread(
            prepare_analysis_package,
            query_period_days=params.get("query_period_days", 90),
            exercise_names=state.get("exercise_names"),
            muscle_groups=state.get("muscle_groups"),
            include_phase2=True,
            reps_floor=params.get("rep_target"),
            cardio_lock=_normalize_cardio_lock(params.get("cardio_lock")),
        )
        pkg.pop("unresolved_exercise_names", None)
        return pkg

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

    async def _run_operational(self, question: str, *,
                               log_boundary: bool = False,
                               fallback_write: bool = False,
                               trailing_note: bool = False) -> str:
        """
        Pass question to the existing Single Agent (AgentSession).
        The Single Agent manages its own ReAct loop and conversation history.

        This is the single append chokepoint for the /log surface text (serves
        cli.py and server.py identically — both render this answer verbatim):
          - trailing_note: the /log message carried an analytical tail — the
            note is its only acknowledgment (never routed, never decomposed).
          - fallback_write: the write was inferred by regex, not /log — append
            the suggestive nudge (fallback turns only, never /log turns).
        Carry set-site: a boundary turn whose staged batch never reached the
        confirmation gate (agent.answer's staging_reached_confirm is False)
        ended in a logging clarification — arm the single-turn carry so the
        user's next reply joins the /log flow. Staging-complete ⇒ gate reached
        ⇒ flag stays down; cancel only exists at the confirm panel, which only
        exists when the batch was complete ⇒ flag already down.
        """
        if self._agent is None:
            return (
                "Operational path is not available in this configuration. "
                "Please initialise the Coordinator with an AgentSession."
            )
        try:
            result = await self._agent.answer(question)
        except _ckpt.QuotaInterrupted:
            # Boundary 1 of the write-path checkpoint arc: the agent just
            # checkpointed its transcript (agent.py's on-429 save). Enrich that
            # slot with the write-path state only the Coordinator holds — the
            # staged batch as it exists RIGHT NOW (read from the MCP slot; may
            # be partial if the 429 hit mid-staging — resume restores it and
            # the stage-2 verify FAILs a partial batch against the full
            # request) and the assembled flow turns (verify Input A). Resume
            # then RESTORES instead of re-entering the agent loop.
            try:
                slot = json.loads(await self._agent.call_tool(
                    "read_staged_workout_slot", {}))
                if slot.get("staged_workouts"):
                    _ckpt.enrich_checkpoint({
                        "staged_slot":    slot["staged_workouts"],
                        "log_flow_turns": list(self._log_flow_turns),
                    })
            except Exception as e:
                # Enrichment failure must never mask the 429 — the un-enriched
                # checkpoint still resumes via the agent path.
                logger.warning("[coordinator] checkpoint enrichment failed: %s", e)
            raise
        answer = result.get("answer", "")
        # ── Write-success claim gate: a success claim can never ship unless a
        # write/stage/execute actually happened this turn. Structural facts
        # decide (all three False ⇒ any completed-write claim is false by
        # construction); the regex only detects that a claim is being made,
        # so a clarification question with the same flags is never touched.
        if (not result.get("db_write_effect")
                and not result.get("staged_this_turn")
                and not result.get("staging_reached_confirm")
                and _WRITE_SUCCESS_CLAIM_RE.search(answer)):
            logger.warning(
                "[coordinator] suppressed unbacked write-success claim: %r",
                answer[:120])
            answer = MSG_NO_WRITE_OCCURRED
        if log_boundary or fallback_write:
            # Fallback (regex-inferred) writes are the same flow as /log turns:
            # a turn that ends pending a logging clarification must carry the
            # originating flow-turn text into the next reply, or the stage-2
            # verify diffs the staged batch against the bare reply ("Today").
            self._pending_log_carry = not result.get("staging_reached_confirm", False)
        if trailing_note:
            answer = answer.rstrip() + "\n\n" + _LOG_TRAILING_NOTE
        if fallback_write:
            answer = answer.rstrip() + "\n\n" + _LOG_FALLBACK_NUDGE
        return answer

    # ── Stage-2 staging verify (write-verification layer) ─────────────────────

    async def verify_log_staging(self, flow_turns: list, slot_json: str,
                                 preview: Optional[str]) -> dict:
        """
        ONE LLM diff call: the assembled /log-flow user turns (Input A) against
        the staged slot JSON + its deterministic rendering (Input B). Returns
        {"verdict": "PASS"|"FAIL"|"ERROR", "reason": str}.

        FAIL is the model's verdict (B misrepresents A) — the caller suppresses
        the confirm panel and discards the slot. ERROR is the verify machinery
        failing (no preview, LLM exception, unparseable output) — fail-OPEN: the
        caller proceeds to the panel (itself a human check of the slot-rendered
        preview) and stage 3 sees a not-verified signal. Deliberately NO
        _call_with_per_minute_retry and no retry loop: +1 call per logging turn
        is the quota budget; a 429/503 here is an ERROR, never a blocker.
        """
        if not flow_turns:
            # No originating request text (a lost/absent flow thread) is a
            # NON-VERIFIABLE state, never a mismatch: diffing the slot against
            # a bare reply ("Today") produces a confidently-wrong FAIL that
            # discards a good batch. ERROR fails open to the human confirm
            # gate, which still shows the slot-rendered preview.
            return {"verdict": "ERROR",
                    "reason": "originating request unavailable — verify skipped"}
        if not preview:
            # A name-blind diff (opaque exercise_ids, kg-converted weights, no
            # rendering) could spuriously FAIL a good batch — skip instead.
            return {"verdict": "ERROR",
                    "reason": "preview unavailable — verify skipped"}
        request_block = "\n".join(
            f"{i}. {t}" for i, t in enumerate(flow_turns or [], 1))
        content = (
            "[LOGGING REQUEST]\n" + request_block
            + "\n\n[STAGED JSON]\n" + (slot_json or "")
            + "\n\n[STAGED RENDERING]\n" + preview
        )
        try:
            config = types.GenerateContentConfig(
                system_instruction=_VERIFY_SYSTEM,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=COORDINATOR_MODEL,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)],
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

            parsed  = json.loads(raw)
            verdict = str(parsed.get("verdict", "")).strip().upper()
            if verdict not in ("PASS", "FAIL"):
                return {"verdict": "ERROR",
                        "reason": f"unrecognized verdict: {parsed.get('verdict')!r}"}
            return {"verdict": verdict, "reason": parsed.get("reason", "") or ""}
        except Exception as e:
            # Fail-open (incl. 429/503): the panel is still a human check and
            # stage 3 sees not-verified. No retries by design.
            logger.warning("[coordinator] staging verify errored: %s", e)
            return {"verdict": "ERROR", "reason": str(e)}

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
