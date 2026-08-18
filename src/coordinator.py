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
import copy
import json
import logging
import os
import re
from datetime import datetime
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

# ── THE question test — one detector, used everywhere ─────────────────────────
# There used to be two of these. _WRITE_QUESTION_RE gated the write pre-guard;
# _LOG_TAIL_QUESTION_RE (added later, for /log tail peeling) existed because the
# first one could not recognize "Also, how is my back progressing" — and the two
# were then left to disagree. Measured at the time of merging, they returned
# different answers for 4 of 7 ordinary coaching questions: the write-guard copy
# said "am I overtraining" and "how has my squat gone" were NOT questions, though
# both are verbatim analytical examples in _CLASSIFY_SYSTEM.
#
# This is their union, and it is the only question test in the module. Three
# shapes count as a question:
#   (1) a question mark anywhere;
#   (2) an interrogative/modal lead at the start, optionally behind a connector
#       ("Also, how is my back progressing" — no "?" and not sentence-initial);
#   (3) modal-coaching phrasing anywhere ("..., should I add a set").
_QUESTION_CONNECTOR = (
    r"(?:also,?\s+|and\s+also,?\s+|btw,?\s+|by\s+the\s+way,?\s+"
    r"|oh\s+and\s+|plus,?\s+)?"
)
_QUESTION_RE = re.compile(
    r"(?i)(?:"
    r"\?"                                                       # (1)
    r"|^\s*" + _QUESTION_CONNECTOR +                            # (2)
    r"(?:how|what|why|when|where|which|who|is|are|am|do|does|did"
    r"|was|were|can|could|should|would|will)\b"
    r"|\bshould\s+i\b|\bcan\s+i\b|\bcould\s+i\b|\bmay\s+i\b"    # (3)
    r"|\bdo\s+i\b|\bwould\s+it\b|\bdo\s+you\s+think\b"
    r"|\bis\s+it\s+(?:ok|okay|fine|worth|better|good|bad|safe)\b"
    r"|\bwhen\s+should\b|\bhow\s+(?:much|many|often|do|should|can)\b"
    r")"
)


def _is_question(message: str) -> bool:
    """True when the message reads as a question rather than a command.

    THE single question test for the module: the write pre-guard, the /log tail
    peeler, and the /log carry test all call this, so they can no longer drift
    apart into three different answers for the same sentence.
    """
    return bool(message) and bool(_QUESTION_RE.search(message))

# Weight×reps / sets×reps / unit shorthand — a strong signal a write is being
# DICTATED ("bench 100x5", "3 sets", "80kg", "12 reps").
_WRITE_QUANTITY = (
    r"(?:"
    r"\d+\s*[x×]\s*\d+"                              # 100x5, 3x5
    r"|\d+\s*sets?\b|\d+\s*reps?\b"                  # 3 sets, 12 reps
    r"|\d+\s*(?:lbs?|kgs?|pounds?|kilos?)\b"         # 100 lbs, 80kg
    r")"
)
# The one compiled form of the above. Shared by the write pre-guard's narration
# test and by the /log tail peeler — both ask the same question ("does this text
# carry actual set/rep/weight data?") and must not answer it differently.
_QUANTITY_RE = re.compile(r"(?i)" + _WRITE_QUANTITY)

# Imperative writes that DO fire (only after the question guard says "not a
# question"). Three forms, each anchored by an explicit write verb:
#   (A) "set a goal" / "set goal …"
#   (B) write verb + quantity shorthand → bare "log bench 100x5", "record squat
#       80kg x5", "add 3 sets of deadlift" (the #8 false-negatives)
#   (C) write verb + an explicit data noun → "delete my deadlift goal",
#       "log today's workout", "update my last set"
# NOTE: standalone "weight" is deliberately NOT a (C) noun — "add weight" is the
# canonical coaching phrasing ("should I add weight"), so it must not anchor a write.
_WRITE_IMPERATIVE_RE = re.compile(
    r"(?i)(?:"
    r"\bset\s+(?:a\s+|an\s+|my\s+|the\s+)?goal\b"                          # (A)
    r"|\b(?:log|record|save|add|delete|remove|update|change|correct)\b"
    r".{0,40}?" + _WRITE_QUANTITY +                                        # (B)
    r"|\b(?:log|record|save|add|delete|remove|update|change|correct)\b"
    r".{0,50}\b(?:workout|sets?|reps?|goal|bodyweight|exercise|session)\b" # (C)
    r")"
)

# (D) Narration of a completed session — "I did … today/yesterday/…". Split out
# from the imperative forms above because it is a materially WEAKER signal: it
# carries no write verb, so on its own it cannot tell "I did chest and triceps
# today" (a session the user wants logged) from "I did shrugs today and my grip
# gave out" (a complaint about how a lift went, which _CLASSIFY_SYSTEM L414
# routes analytical).
#
# RESOLVED — ledger row E. "I did chest and triceps today" (a session to log) and
# "I did shrugs today and my grip gave out" (a complaint to explain) are the same
# shape; no regex separates them, and this one used to call both writes and force
# the verdict through, making _CLASSIFY_SYSTEM's rule unreachable. So narration
# no longer DECIDES — it only HINTS. The classifier, which can read the
# difference, decides; the hint still hard-forces operational when the classify
# call FAILS, so a write is never lost to an error. See _write_intent_form.
_WRITE_NARRATION_RE = re.compile(
    r"(?i)\bi\s+did\b.{0,80}\b(?:today|yesterday|this\s+morning|this\s+week)\b"
)

# The two strengths of write signal. They differ in what they license, not just
# in what they matched:
#   "imperative" — an explicit write verb ("log", "delete", "set a goal"). The
#                  user named the action, so this BINDS: the regex verdict wins
#                  over the classifier (the long-standing write-safety rule).
#   "narration"  — no write verb, just "I did … today". A hint only: it spends
#                  the classify call and stands in on classify FAILURE, but a
#                  successful classification overrules it.
WRITE_FORM_IMPERATIVE = "imperative"
WRITE_FORM_NARRATION  = "narration"


def _write_intent_form(message: str) -> Optional[str]:
    """Which write signal fired, or None. Question/modal phrasing wins over both
    (precedence above): if the message reads as a question, no write signal
    fires and the classifier decides."""
    if not message or _is_question(message):
        return None
    if _WRITE_IMPERATIVE_RE.search(message):
        return WRITE_FORM_IMPERATIVE
    if _WRITE_NARRATION_RE.search(message):
        return WRITE_FORM_NARRATION
    return None


def _is_write_intent(message: str) -> bool:
    """
    True when ANY write signal fired — imperative or narration. Callers that
    need to know whether the signal BINDS must ask _write_intent_form.
    """
    return _write_intent_form(message) is not None


# Strips a single leading markdown header line ("### …") from a chunk answer.
# In a multi-part decomposed merge the merge itself is the sole header authority
# (it prepends a canonical "### <intent>" per part); when the model also opens
# its answer with its own "### …" the two stack into a duplicate header. Remove
# exactly the first header line so the merge's canonical header stands alone.
_LEADING_MD_HEADER_RE = re.compile(r"^\s*#{1,6}[ \t]+[^\n]*(?:\r?\n)+")


def _strip_leading_md_header(text: str) -> str:
    return _LEADING_MD_HEADER_RE.sub("", text, count=1) if text else text


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
        if _is_question(seg) and not _QUANTITY_RE.search(seg):
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
    return _is_question(message) or _filler_reply(message) is not None


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
You are a routing classifier for a fitness coaching AI. Return JSON only.

A message contains one or more REQUESTS. Your job is to give every request a
LANE and extract its parameters. The lane rules below define what a lane means
for ONE request; they are the only lane rules, and they apply identically
wherever you assign a lane — to the message as a whole (the top-level "route")
and to each entry in "requests".

════ THE LANES ════

ANALYTICAL — any READ of the user's own training data, and any coaching question
about their training. One session or a whole history, terse or long ("my
squat?", "Lat Pulldown PR" are analytical). This is where most messages belong.

  The Data Agent answers all of these deterministically:
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

  Also analytical: any MEDICAL or symptom question. A stated pain, injury, or
  condition is in-domain training territory — never out_of_scope. (The coach's
  own prompt handles the see-a-professional redirect and refuses only to
  diagnose or treat.)

OPERATIONAL — a CLOSED list of exactly two cases. If a request is not one of
these two, it is not operational:
    1. WRITES — log a workout, set a goal, update or correct a set, delete
       anything.  "Log today's workout" · "Set a goal for 150 lbs on Lat
       Pulldown" · "Fix my last set — it was 12 reps not 10" · "Delete my
       deadlift goal"
    2. RESEARCH — fitness science questions.  "What does science say about
       training frequency?"

RECALL — ONLY to REPEAT or RESTATE a specific NUMBER or FIGURE you already gave
earlier in this conversation:
    "what was that number you just mentioned?"   "what did you say again?"
    "remind me what that percentage was"         "repeat the figure you gave"
  This lane re-quotes a value from [PREVIOUS TURNS]; it never looks anything up.
  It is NOT recall if the message asks you to EXPLAIN, IDENTIFY, DESCRIBE, or
  INVESTIGATE anything, even when it points back at a prior mention ("what was
  the same pain that re-occurred?", "what did that plateau mean?", "which
  exercise was that?"), or asks for ANY new stat/PR/trend/date/volume ("and my
  squat?", "what's my bench PR").

OUT_OF_SCOPE — refuse politely, with no tool, search, or analysis. Decide by a
FITNESS-CONNECTION test, NOT a keyword blocklist: "Is this about fitness,
training, nutrition-for-training, fitness science/history, or the user's own
training data?"

  IN SCOPE — route by the lanes above, never out_of_scope:
   - The user's own logs / training data — any phrasing, even with no fitness
     words ("how many days have I trained excluding Sundays").
   - Fitness science, exercise physiology, and DEFINITIONS of fitness terms
     ("what is progressive overload / hypertrophy / RPE / a superset").
   - Nutrition for training/performance, incl. food prep with a nutrition/goal
     angle ("cook chicken keeping protein high", "good pre-workout meal", macros).
   - Fitness history & culture ("Ronnie Coleman's diet", "who won Mr. Olympia 1998").
   - Program design, splits, recovery, periodization, rest days.
   - Anything medical: symptoms, pain, injury, rehab, mobility, warmups.

  OUT OF SCOPE:
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

════ THE WRITE BOUNDARY ════
The one genuinely hard call, and the only thing that can judge it is you.

A REPORT OF HOW A LIFT WENT IS ANALYTICAL, NOT A WRITE. "My grip gave out on
shrugs", "I failed the last rep", "that felt heavy", "my squat stalled" describe
training that ALREADY HAPPENED — they are the fatigue/failure questions named in
the analytical list, and the user wants them EXPLAINED. Offering to save a note
about a complaint is not an answer to it.

A SESSION REPORTED FOR THE RECORD IS A WRITE. "I did … today/yesterday" with NO
complaint and NO outcome attached is the user telling you what to record — even
with no logging verb and no numbers.

The difference is the trailing clause, not the words "I did":
    "I did chest and triceps today"              → operational (a session to log)
    "I did 3x10 squats today"                    → operational (a session to log)
    "log 3x10 shrugs at 60kg"                    → operational (a write)
    "I did shrugs today and my grip gave out"    → analytical (explain it)
    "I did legs yesterday and it destroyed me"   → analytical (explain it)
    "my grip gave out on shrugs"                 → analytical (why, and what to do)
A message that does BOTH ("I did 5 sets of squats yesterday and it felt awful")
is TWO requests — a write and a question. Split it in "requests" below.

════ WHEN YOU ARE UNSURE ════
Route ANALYTICAL. This is the single default for every uncertainty on this page
— analytical vs. operational, analytical vs. recall, in-scope vs. out_of_scope.
Two reasons, and they point the same way: a false refusal of a real fitness
question is worse than answering a borderline one, and the analytical package is
deterministic and validated (it hard-stops on integrity failure) where the
operational read path is hand-rolled SQL and the riskier surface for a read.

════ CONTEXT ════
If a [PREVIOUS TURNS] block is present, use it ONLY to resolve pronouns
and follow-up references in the current message ("what about my squat?",
"and over the last year?"). Classify and extract parameters for the
CURRENT MESSAGE, carrying over the topic from previous turns when the
current message is an elliptical follow-up.

PARAMETER EXTRACTION (analytical route only):
  display_intent:    a pure PHRASING test, independent of the lane: does the
                     request ask for a session or day laid out set-by-set?
                     true  — "show me my last leg day", "what did I do on Monday",
                             "lay out / breakdown of my last chest session",
                             "how did that session go". Lean TRUE whenever such
                             display phrasing is present (a real display question
                             must keep its display).
                     false — every stat / trend / PR / plateau / volume / "why" /
                             "when" question ("how has my Lat Pulldown progressed",
                             "total back volume", "when did I first reach 145").
                     Default false.
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

════ THE "requests" ARRAY ════
Everything above describes ONE request. Emit both scopes:

  - The top-level fields describe the WHOLE message.
  - "requests" holds one entry per DISTINCT request, each carrying the same
    fields for that request alone — the lane rules, the write boundary, the
    unsure-default, PARAMETER EXTRACTION and CUSTOM SQL, applied to it. The
    analytical-only fields take their defaults/null on operational, recall, and
    out_of_scope entries.

  A request is distinct when it could be answered or actioned on its own and
  asks for something different from its siblings — an analysis question plus a
  logging instruction, or two unrelated questions joined by "and"/"also".

  NEVER split a single request into artificial pieces. One question about
  several exercises, periods, or stats is ONE request ("compare my squat and
  bench over 3 months" → one entry, both exercise_names). Most messages are a
  single request, so the array usually has exactly ONE entry mirroring the
  top-level fields.

  Two fields exist only per entry:
      index:       0-based position in message order.
      intent_text: a self-contained restatement of this request. Resolve
                   pronouns and elliptical references using the sibling requests
                   and [PREVIOUS TURNS] ("is it progressing" after a squat
                   question → "Is my squat progressing?"). Someone reading only
                   intent_text must be able to answer it.

Return ONLY valid JSON, no preamble, no markdown fences:
{
  "route": "analytical" | "operational" | "out_of_scope" | "recall",
  "display_intent": false,
  "exercise_names": ["..."] | null,
  "muscle_groups": ["..."] | null,
  "query_period_days": 90 | null,
  "rep_target": 5 | null,
  "cardio_lock": {"field": "distance"|"duration", "value": 5, "unit": "km"} | null,
  "needs_custom_sql": false,
  "custom_sql_intent": null,
  "requests": [
    {
      "index": 0,
      "lane": "analytical" | "operational" | "recall" | "out_of_scope",
      "intent_text": "self-contained restatement of this request",
      "display_intent": false,
      "exercise_names": ["..."] | null,
      "muscle_groups": ["..."] | null,
      "query_period_days": 90 | null,
      "rep_target": 5 | null,
      "cardio_lock": {"field": "distance"|"duration", "value": 5, "unit": "km"} | null,
      "needs_custom_sql": false,
      "custom_sql_intent": null
    }
  ]
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


# ── Decomposition Stage 1: the emit-inert per-chunk `requests` array ──────────
# The classifier additionally emits params["requests"] — one entry per distinct
# request in the message, each with its own `lane` (per-chunk route), a
# self-contained `intent_text`, and per-chunk parameter fields. NOTHING consumes
# it yet: Stage 2 builds the pending-decomposition slot on it, Stage 3 runs each
# chunk through its lane and merges answers in index order. Until then the flat
# fields stay the whole-message source of truth, so the sanitizer's contract is
# strict: params["requests"] is either None or a FULLY valid list — a partially
# valid array is dropped whole rather than half-trusted (Stage 2 must never
# inherit a chunk list that silently lost entries).

def _norm_name(s: str) -> str:
    """Exercise-name normalization for deterministic matching: lowercase,
    strip, collapse internal whitespace (same discipline as the resolver's
    Tier-0 space-normalized compare)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_VALID_LANES = {"analytical", "operational", "recall", "out_of_scope"}

# Stage 3: at most this many chunks execute per turn (quota guard — each
# analytical chunk costs a draft + grounding call; real messages rarely
# exceed 2-3 requests).
_DECOMP_CHUNK_CAP = 4

# Per-chunk parameter defaults — mirrors the flat setdefault block in _classify.
_CHUNK_PARAM_DEFAULTS = {
    "display_intent":    False,
    "exercise_names":    None,
    "muscle_groups":     None,
    "query_period_days": 90,
    "rep_target":        None,
    "cardio_lock":       None,
    "needs_custom_sql":  False,
    "custom_sql_intent": None,
}


def _is_mixed_lane_multi(params: dict) -> bool:
    """True when params carries 2+ requests spanning 2+ lanes — the single
    condition under which a turn executes per-chunk instead of whole.

    ONE definition, three callers: both of the graph's conditional edges and the
    write-hint distrust override. All three used to carry an inline copy (one of
    them annotated "same test as _route_after_classify"), and the test decides
    whether a turn keeps or drops its sibling chunks — not something to state
    three times and hope they stay equal.
    """
    reqs = (params or {}).get("requests") or []
    return len(reqs) >= 2 and len({c.get("lane") for c in reqs}) >= 2


def _sanitize_requests(params: dict) -> None:
    """Validate params["requests"] IN PLACE to None-or-fully-valid.

    Runs inside _classify's try block, so it must never raise — an escaped
    exception would degrade a good classification to the _parse_failed
    default. Never touches the flat fields.
    """
    try:
        reqs = params.get("requests")
        if reqs is None:
            params["requests"] = None
            return
        if not isinstance(reqs, list) or not reqs:
            logger.warning(
                "[coordinator] classify: malformed requests dropped "
                "(not a non-empty list: %s)", type(reqs).__name__)
            params["requests"] = None
            return
        for entry in reqs:
            if not isinstance(entry, dict):
                logger.warning(
                    "[coordinator] classify: malformed requests dropped "
                    "(non-dict entry: %s)", type(entry).__name__)
                params["requests"] = None
                return
            if entry.get("lane") not in _VALID_LANES:
                logger.warning(
                    "[coordinator] classify: malformed requests dropped "
                    "(invalid lane: %r)", entry.get("lane"))
                params["requests"] = None
                return
            intent = entry.get("intent_text")
            if not isinstance(intent, str) or not intent.strip():
                logger.warning(
                    "[coordinator] classify: malformed requests dropped "
                    "(missing/empty intent_text)")
                params["requests"] = None
                return
        for i, entry in enumerate(reqs):
            entry["index"] = i          # list order is authoritative
            for key, default_val in _CHUNK_PARAM_DEFAULTS.items():
                entry.setdefault(key, default_val)
    except Exception as e:              # pragma: no cover — backstop only
        logger.warning("[coordinator] classify: requests sanitize error: %s", e)
        params["requests"] = None


def _log_requests_divergence(params: dict) -> None:
    """Log-only observation of flat-vs-chunks disagreement (Stage-2 field
    data). Never mutates, never raises."""
    try:
        reqs = params.get("requests")
        if not reqs:
            return
        if len(reqs) > 1:
            logger.info("[coordinator] classify: %d request chunk(s)", len(reqs))
        lanes = {c.get("lane") for c in reqs}
        if params.get("route") not in lanes:
            logger.warning(
                "[coordinator] decomposition divergence: flat route %r not "
                "among chunk lanes %s", params.get("route"), sorted(lanes))
        for field in ("exercise_names", "muscle_groups"):
            flat = set(params.get(field) or [])
            union = set()
            for c in reqs:
                union.update(c.get(field) or [])
            if flat != union:
                logger.warning(
                    "[coordinator] decomposition divergence: flat %s %s != "
                    "chunk union %s", field, sorted(flat), sorted(union))
    except Exception as e:              # pragma: no cover — backstop only
        logger.warning("[coordinator] classify: divergence log error: %s", e)


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

# Hole B: looser completion phrasings the narrow regex misses ("I've logged
# your workout.", "Your workout is saved.", "You're all set!"). Globally these
# are ordinary speech ("you're all set for tomorrow", "I've saved that to
# memory"), and _run_operational also serves RESEARCH answers where "the study
# removed participants" is a sentence about a paper, not a database. So the
# detector never runs alone — a STRUCTURAL fact scopes it (see the gate).
#
# It used to be scoped to logging flows only, and its verb list was the logging
# verbs. An audit across every write verb found 11 of 17 plausible completion
# claims undetected — and on goal/set-edit turns, where the loose detector never
# ran at all, 12 of 13. The gate protected workout logging and almost nothing
# else. Both the verb list and the scope are widened below; the scope widens to
# "a write tool was actually called this turn", which still excludes research.
_WRITE_COMPLETION_CLAIM_RE = re.compile(
    r"(?i)(?:"
    # Unqualified first-person completions — no trailing qualifier required.
    r"\bi(?:'ve|\s+have)?\s+"
    r"(?:logged|saved|recorded|added|written|updated|deleted|removed"
    r"|corrected|fixed|created)\b"
    # Subject-state completions ("Your workout is saved.", "3 sets were added",
    # "Your goal has been deleted.", "Your goal is now set."). The subject list
    # is what keeps research prose out: "the study was removed" has no match
    # because "study" is not one of our data nouns.
    #
    # The noun and the verb are NOT required to be adjacent. Real answers put
    # the details in between — the live failure was "...set of 101 lbs x 7 reps
    # on 2026-08-08 has been removed...", which an adjacency-only pattern misses
    # entirely. Bounded, and [^.\n] keeps it inside one sentence.
    r"|\b(?:workouts?|sets?|exercises?|goals?|entry|entries|it)\b[^.\n]{0,80}?"
    r"\b(?:is|was|are|were|(?:has|have)\s+been)\s+(?:now\s+)?"
    r"(?:logged|saved|recorded|added|written|updated|deleted|removed"
    r"|corrected|fixed|created|set|in\s+the\s+books)\b"
    # "You're all set!" / "All set." — \b already excludes "all sets".
    r"|\ball\s+set\b"
    r")"
)

MSG_NO_WRITE_OCCURRED = (
    "⚠️ Nothing was written to your database this turn — no write was "
    "executed. Please re-state what you wanted saved or changed "
    "(for workouts, starting with /log is the most reliable route)."
)

# Hole A's truthful replacement: the turn DID stage a batch but no execute was
# attempted and nothing was written — a completed-write claim is premature,
# not baseless. Never claims failure (the CLI executes right after this text;
# the web panel supersedes it when the slot is real).
MSG_STAGED_NOT_SAVED = (
    "⚠️ That isn't saved yet — it's staged and still needs your confirmation "
    "before anything is written to your database."
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


# ── Recall prompt (package-free: restate a prior figure, never re-derive) ──────

_RECALL_SYSTEM = """
You are a fitness coach answering a follow-up that refers to something YOU said
earlier in this conversation (e.g. "what was that number?", "what did you say
again?", "remind me / repeat that").

Using ONLY the [CONVERSATION] provided, restate the specific figure or fact the user
is asking about — verbatim as you already stated it. NEVER compute, estimate, look
up, or introduce a NEW number; your only job is to repeat what was already said.
# ONE message, phrased for every write. It used to end "re-state your logging
# request (tip: start with /log)", which is nonsense after a failed goal or set
# edit — but splitting it in two turned out to be undoable: the gate fires
# hardest when the agent called NO tool (it invented the success outright), and
# in that case nothing distinguishes a logging turn from a goal turn.
# log_boundary/fallback_write are not that signal either — the canonical live
# failure is a bare "Yes thats correct" confirming a log, which trips neither.
# So the tip is made conditional in the WORDING instead of in the code.

If the [CONVERSATION] does not contain a figure matching what they are asking about,
say you are not sure which number they mean and ask them to clarify — do not guess.

Reply in one or two plain sentences. No JSON, no lists.
""".strip()

_RECALL_FALLBACK = (
    "I'm not sure which number you're referring to — could you tell me which figure "
    "you'd like me to repeat?"
)


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
        # Decomposition Stage 2: cross-turn pending-disambiguation slot.
        # In-memory, per session (like _pending_followup). Armed when the
        # analytical lane exits on a disambiguation ask; consumed by a
        # deterministic candidate reply (no LLM). Lifecycle: survives ONE
        # non-answer message (reminder appended), dropped on the second;
        # 48h staleness; one clarify re-ask for a multi-candidate reply.
        # Shape: {"question", "params" (deep copy incl. requests), "name",
        #         "candidates", "created" iso, "strikes", "reminded",
        #         "clarified"}.
        self._pending_decomposition: dict | None = None
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
        # ── PRE: decomposition slot (before the demographic follow-up —
        # candidate matching is strict/deterministic, so a date/height reply
        # falls through to the followup PRE untouched). decomp_before is the
        # POST reminder's arm-this-turn guard: never remind on the arming
        # turn, and never about a chain re-arm (different dict object).
        decomp_before = self._pending_decomposition
        if self._pending_decomposition is not None:
            resumed = await self._consume_pending_decomposition(question)
            if resumed is not None:
                return resumed
            # expired / struck / dropped — handled internally; fall through.

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

        # ── POST: one-line pending-decomposition reminder (remind-once).
        # Identity check: the slot must be the SAME object seen at entry —
        # a slot armed or re-armed THIS turn never gets a reminder. Appended
        # after _route_with_checkpoint recorded history, so the line never
        # enters extraction/grounding history (same invariant as the
        # demographic follow-up above). Strike accounting (PRE) stays
        # authoritative; this line is courtesy only.
        if (decomp_before is not None
                and self._pending_decomposition is decomp_before
                and not decomp_before["reminded"]
                and result.get("route") in ("analytical", "operational")
                and result.get("answer")):
            q = decomp_before["question"]
            trunc = q[:80] + ("…" if len(q) > 80 else "")
            result["answer"] = result["answer"].rstrip() + (
                f"\n\n(Still pending: your earlier question \"{trunc}\" — "
                f"tell me which **{decomp_before['name']}** you meant and "
                f"I'll answer it.)")
            decomp_before["reminded"] = True
            logger.info("[decomposition] reminder appended (strike %d)",
                        decomp_before["strikes"])
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

    # ── Decomposition Stage 2: pending-disambiguation slot helpers ─────────────

    async def _consume_pending_decomposition(self, question: str) -> dict | None:
        """
        Interpret the user's reply against the pending disambiguation.
        Returns the turn's response dict (the RESUMED answer, or a clarify
        re-ask) when the reply was consumed, or None when it wasn't — the
        slot's lifecycle (expiry / strikes / drop) is handled internally and
        the caller routes the message normally.
        """
        slot = self._pending_decomposition
        # 1. Staleness — same rule + constant as the interrupted-question
        # checkpoint (48h); a corrupt timestamp counts as stale.
        try:
            age_h = (datetime.now()
                     - datetime.fromisoformat(slot["created"])
                     ).total_seconds() / 3600
        except Exception:
            age_h = _ckpt.MAX_AGE_HOURS + 1
        if age_h > _ckpt.MAX_AGE_HOURS:
            logger.info("[decomposition] discarded stale slot (%.0fh old)", age_h)
            self._pending_decomposition = None
            return None

        kind, value, source = self._match_disambiguation_reply(
            question, slot["candidates"], slot["name"])

        # Insistence relent (user ruling): repeating the SAME out-of-context
        # name after the push-back is a deliberate scope switch — accept it.
        if (kind == "out_of_context"
                and slot.get("rejected_override") == value):
            logger.info("[decomposition] override accepted on insistence: %r",
                        value)
            kind, source = "match", "override-insisted"

        if kind == "match":
            logger.info("[decomposition] consumed: %r -> %r (%s)",
                        question, value, source)
            patched = slot["params"]        # the slot's own deep copy
            self._patch_resolved_name(patched, slot["name"], value)
            # Clear BEFORE re-entry: a chain (second ambiguity) re-arms
            # fresh, and a 429 mid-resume leaves the stage checkpoint
            # holding the patched params — "continue" recovers through the
            # normal checkpoint path, which the slot must not shadow.
            self._pending_decomposition = None
            return await self._resume_decomposition(slot["question"], patched)

        if kind == "out_of_context":
            if slot.get("rejected_override") is not None:
                # A DIFFERENT out-of-context name after a push-back — one
                # push-back total, never a loop: treat as a non-answer.
                kind = "miss"
            else:
                # Stern push-back: hold ground, restate the ask (engagement,
                # NOT a strike — mirrors the clarify-once precedent).
                slot["rejected_override"] = value
                q = slot["question"]
                trunc = q[:80] + ("…" if len(q) > 80 else "")
                opts = "\n".join(f"- {c}" for c in slot["candidates"][:5])
                logger.info("[decomposition] push-back: %r is not a %r",
                            value, slot["name"])
                return self._followup_response(
                    f"You asked about **{slot['name']}** — **{value}** isn't "
                    f"one. To answer your original question (\"{trunc}\") I "
                    f"need one of these:\n\n{opts}\n\nIf you've changed your "
                    f"mind and want **{value}** instead, just say it again "
                    f"and I'll switch.",
                    "decomposition_pushback")

        if kind == "ambiguous" and not slot["clarified"]:
            # Engagement, not ignoring: one clarify re-ask, no strike
            # (mirrors _consume_pending_followup's clarify-once).
            slot["clarified"] = True
            opts = " or ".join(f"**{c}**" for c in value[:5])
            logger.info("[decomposition] clarify re-ask (%d still match)",
                        len(value))
            return self._followup_response(
                f"I still can't tell which one — did you mean {opts}?",
                "decomposition_clarify")

        # miss (or a second ambiguous attempt after the clarify)
        slot["strikes"] += 1
        if slot["strikes"] >= 2:
            logger.info("[decomposition] dropped after %d strikes: %r",
                        slot["strikes"], slot["question"][:60])
            self._pending_decomposition = None
        else:
            logger.info("[decomposition] strike %d — slot survives",
                        slot["strikes"])
        return None

    def _match_disambiguation_reply(self, reply: str, candidates: list,
                                    ambiguous_name: str | None = None) -> tuple:
        """
        Deterministic reply→candidate matching — zero LLM calls.
        Returns ("match", exact_db_name, source) | ("ambiguous", subset, None)
        | ("out_of_context", exact_db_name, None) | ("miss", None, None).
        Tiers: normalized equality → unique substring (both directions) →
        resolver fallback, gated to NAME-SHAPED replies only (the hijack
        guard: a full-sentence new question must never be swallowed). A
        resolver match outside the candidate list is accepted as an override
        ONLY when it stays in the original context — it contains the
        ambiguous term ("Barbell Squat" for "squat") or shares a muscle
        group (Category) with the candidates; otherwise it is out_of_context
        and the caller pushes back (user ruling 2026-07-12: a clarification
        answer must not teleport out of the question's context).
        """
        r = _norm_name(reply)
        r = re.sub(r"[.!?,]+$", "", r).strip()
        for prefix in ("i meant ", "i mean ", "it's ", "it is ", "the "):
            if r.startswith(prefix):
                r = r[len(prefix):].strip()
                break
        if not r:
            return ("miss", None, None)

        norm_c = {c: _norm_name(c) for c in candidates}
        for c, nc in norm_c.items():                     # tier 1: exact
            if r == nc:
                return ("match", c, "exact")
        hits = [c for c, nc in norm_c.items()            # tier 2: substring
                if r in nc or nc in r]
        if len(hits) == 1:
            return ("match", hits[0], "substring")
        if len(hits) >= 2:
            return ("ambiguous", hits, None)

        # tier 3: resolver fallback — name-shaped replies only.
        if ("?" in reply or len(r.split()) > 5
                or _is_write_intent(reply)
                or _ckpt.is_continue_intent(reply)
                or _filler_reply(reply) is not None):
            return ("miss", None, None)
        from src.shared.resolver import resolve_exercise_name
        db_path = os.environ.get(
            "FITNOTES_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..",
                         "data", "FitNotes_Backup.fitnotes"))
        try:
            result = resolve_exercise_name(r, db_path, permissive=True)
        except Exception as e:
            logger.warning("[decomposition] resolver fallback failed: %s", e)
            return ("miss", None, None)
        match = result.get("match")
        if match:
            if match in candidates:
                return ("match", match, "resolver")
            # Containment gate: an off-list override must stay in context.
            if ambiguous_name and _norm_name(ambiguous_name) in _norm_name(match):
                return ("match", match, "override")        # term containment
            from src.shared.resolver import exercise_categories
            cats = exercise_categories([match] + list(candidates), db_path)
            cand_groups = {cats[c] for c in candidates if c in cats}
            if match in cats and cats[match] in cand_groups:
                return ("match", match, "override")        # same muscle group
            return ("out_of_context", match, None)
        return ("miss", None, None)

    @staticmethod
    def _patch_resolved_name(params: dict, old: str, new: str) -> None:
        """Swap the ambiguous name for the resolved exact DB name, in the
        flat exercise_names AND every requests chunk (normalized compare).

        Two channels, because the two lanes read the name differently: the
        analytical lane resolves from params.exercise_names (patched via the
        list swap), but the operational (write) lane resolves the name from
        the chunk's natural-language intent_text inside the agent — so the
        exact name must be substituted THERE too, or the write agent re-asks
        "which one?" on resume. The text substitution is case-insensitive and
        applies to every chunk's intent_text (rewriting an analytical chunk's
        question to the exact name is correct and harmless)."""
        old_n = _norm_name(old)

        def swap(lst):
            if not lst:
                return lst
            return [new if _norm_name(x) == old_n else x for x in lst]

        params["exercise_names"] = swap(params.get("exercise_names"))
        for chunk in (params.get("requests") or []):
            chunk["exercise_names"] = swap(chunk.get("exercise_names"))
            it = chunk.get("intent_text")
            if it and old:
                chunk["intent_text"] = re.sub(
                    re.escape(old), new, it, flags=re.IGNORECASE)

    def _disambiguation_payload(self) -> dict | None:
        """Structured groups for the current pending-disambiguation slot, or
        None. The single source the server reads to raise the disambiguation
        panel: {"groups": [{"name", "candidates"}, ...]} in ask order. Only the
        display fields (name + candidate list) cross the wire; the full params
        and lifecycle stay server-side in the slot."""
        slot = self._pending_decomposition
        if not slot or not slot.get("groups"):
            return None
        return {"groups": [{"name": g["name"],
                            "candidates": list(g["candidates"])}
                           for g in slot["groups"]]}

    def _collect_decomposition_ambiguities(self, reqs: list) -> list:
        """Pre-dispatch name resolution across ALL chunks. Resolve every
        chunk's exercise_names via the shared resolver — permissive (data-first
        auto-pick) for read lanes, STRICT for the operational (write) lane (a
        wrong write is unrecoverable — the locked auto-pick-removal rule). Any
        name that resolves cleanly is patched into the chunk in place (so the
        chunk dispatches with the exact name); any that stays ambiguous becomes
        a group {name, candidates, lane}, in chunk/index order, deduped by
        normalized name (the same term in two chunks asks once)."""
        from src.shared.resolver import resolve_exercise_name
        db_path = self._db_path()
        groups: list = []
        seen: set = set()
        for chunk in reqs:
            lane = chunk.get("lane")
            permissive = lane != "operational"
            for name in list(chunk.get("exercise_names") or []):
                try:
                    result = resolve_exercise_name(name, db_path,
                                                   permissive=permissive)
                except Exception as e:
                    logger.warning(
                        "[decomposition] pre-resolve failed for %r: %s", name, e)
                    continue
                match = result.get("match")
                candidates = result.get("candidates") or []
                if match:
                    # Clean single match — bind it into the chunk now (both
                    # channels) so no lane re-asks downstream.
                    self._patch_resolved_name({"requests": [chunk]}, name, match)
                elif candidates:
                    key = _norm_name(name)
                    if key not in seen:
                        seen.add(key)
                        groups.append({"name": name,
                                       "candidates": list(candidates),
                                       "lane": lane})
                # 0 candidates → not found: leave as-is (a genuinely new write
                # exercise, or a name the package will report unresolved).
        return groups

    @staticmethod
    def _db_path() -> str:
        return os.environ.get(
            "FITNOTES_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..",
                         "data", "FitNotes_Backup.fitnotes"))

    async def _resume_decomposition(self, question: str, params: dict) -> dict:
        """
        Re-run the original question with pre-seeded params: the graph's
        entry pass-through + route-aware dispatch skip every pre-guard AND
        the classify call (net −1 LLM call vs a re-classify). The whole turn
        machinery (dispatch, finalize, history, record_external_exchange)
        runs normally. A chain ambiguity re-arms the slot via
        _run_analytical's arm seam.
        """
        from src.graph.coordinator_graph import get_coordinator_graph
        from src.graph.persistence import cleanup_turn, new_turn_id, turn_config
        from src.graph.state import GraphRunContext, RunCache

        turn_id = new_turn_id()
        state = await get_coordinator_graph().ainvoke(
            {"question": question, "log_carry": False, "flow_turns": [],
             "params": params},
            turn_config(turn_id),
            context=GraphRunContext(coordinator=self, cache=RunCache()),
        )
        cleanup_turn(turn_id)
        result = state["result"]
        result.setdefault("resolved_question", question)
        return result

    async def resolve_disambiguation(self, selections: list) -> dict | None:
        """
        Structured (panel) resolution of the pending-disambiguation slot — the
        PRIMARY path, distinct from the prose _consume_pending_decomposition
        (kept as a typed-reply fallback). `selections` is the panel's answer,
        one entry per group: {"name", "choice", "other_text"?}. A `choice` that
        is a listed candidate is exact; a `choice` of "__other__" re-resolves
        `other_text` through the resolver at the group's own permissiveness
        (STRICT for a write group — never auto-pick). Every cleanly-resolved
        name is patched into the FULL turn params (both channels); groups that
        stay unresolved (an ambiguous / not-found "Other") re-arm and the panel
        re-prompts JUST them. When nothing is left unresolved the whole turn
        resumes and runs to completion. Returns the turn's result dict, or None
        when there is no live slot (caller ships a "nothing to resolve" notice).
        """
        slot = self._pending_decomposition
        if slot is None:
            return None
        from src.shared.resolver import resolve_exercise_name
        db_path = self._db_path()
        params = slot["params"]
        by_name = {s.get("name"): s for s in (selections or [])}
        remaining: list = []
        for g in slot["groups"]:
            sel = by_name.get(g["name"]) or {}
            choice = (sel.get("choice") or "").strip()
            chosen: str | None = None
            if choice and choice != "__other__" and choice in g["candidates"]:
                chosen = choice                       # exact candidate pick
            elif choice == "__other__":
                other = (sel.get("other_text") or "").strip()
                if other:
                    permissive = g.get("lane") != "operational"
                    try:
                        res = resolve_exercise_name(other, db_path,
                                                    permissive=permissive)
                    except Exception as e:
                        logger.warning(
                            "[decomposition] Other re-resolve failed: %s", e)
                        res = {}
                    if res.get("match"):
                        chosen = res["match"]
                    elif res.get("candidates"):
                        # Still ambiguous — re-prompt this group with the NEW
                        # candidates for the term the user typed.
                        remaining.append({"name": other,
                                          "candidates": list(res["candidates"]),
                                          "lane": g.get("lane")})
                        continue
                    else:
                        # Not found — keep the original group so the panel can
                        # re-ask (the user can pick a listed candidate instead).
                        remaining.append(g)
                        continue
            if chosen:
                self._patch_resolved_name(params, g["name"], chosen)
                logger.info("[decomposition] panel resolved %r -> %r",
                            g["name"], chosen)
            else:
                remaining.append(g)                   # no answer given → keep
        if remaining:
            slot["groups"] = remaining
            slot["name"] = remaining[0]["name"]
            slot["candidates"] = list(remaining[0]["candidates"])
            logger.info("[decomposition] panel re-prompt: %d group(s) left",
                        len(remaining))
            return {
                "answer": "", "route": "analytical", "flagged_claims": [],
                "error": None, "disambiguation": self._disambiguation_payload(),
            }
        # All resolved → resume the whole turn with exact names bound in.
        question = slot["question"]
        self._pending_decomposition = None
        logger.info("[decomposition] panel complete — resuming whole turn")
        return await self._resume_decomposition(question, params)

    def cancel_disambiguation(self) -> bool:
        """Panel Cancel: drop the pending slot and the /log carry. Returns True
        if a slot was actually cleared (caller tailors the chat notice)."""
        had = self._pending_decomposition is not None
        self._pending_decomposition = None
        self._pending_log_carry = False
        return had

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
        # An interrupted-question slot is never silently discarded — a NEW
        # question prompts for confirmation first. A staged-slot checkpoint
        # (an unconfirmed /log panel) IS silently discarded on a new question.
        # (Stale >48h slots are dropped silently on load.)
        cp = _ckpt.load_checkpoint()
        if _ckpt.is_continue_intent(question):
            if cp is not None:
                return await self._resume(cp)
            # Nothing-to-resume guard: a resume request with no live slot must
            # NOT fall through to classification and run a fresh, expensive
            # question. Return a plain notice — no LLM call.
            return self._no_resume_response()
        elif cp is not None:
            if cp.get("staged_slot"):
                # An unconfirmed staged batch is cheap to recreate by re-logging
                # — a new question means the user moved on, so drop it silently,
                # mirroring the in-session turn-start discard_staged_writes
                # clear-on-entry (its persistent shadow must follow the same
                # lifecycle). The confirm-before-discard prompt is reserved for
                # interrupted questions whose paid LLM stages are worth
                # protecting; 'continue' (handled above) still restores the
                # staged batch until then.
                logger.info("[coordinator] discarded unconfirmed staged-slot "
                            "checkpoint on new question")
                _ckpt.clear_checkpoint()
            elif cp.get("awaiting_discard_confirm"):
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

        result = await self._route_fresh(question, log_carry=log_carry,
                                         flow_turns=flow_turns)
        # The resolved question can differ from the transport message (a 'new'
        # discard-confirm turn processes the stashed pending question) — expose
        # it so the server's checkpoint-2 save never stores the control word.
        result.setdefault("resolved_question", question)
        return result

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
        # Pre-seeded params = a decomposition resume: the classify already
        # happened on the original turn. Skip EVERY pre-guard — in particular
        # the write-intent regex must never re-inspect a question whose lane
        # is already decided ("did I log squats..." must not hijack to
        # operational on resume).
        if state.get("params") is not None:
            return {"params": state["params"]}
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
        # `requests` deliberately absent from these synthetic dicts — the
        # Stage-1 emit-inert chunk array lives in _classify only; whether the
        # deterministic write path synthesizes a single operational chunk is a
        # Stage-2 decision.
        fallback_write = False
        write_intent_hint = False
        write_intent_hard = False
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
        else:
            form = _write_intent_form(question)
            if form is not None:
                # Stage 3 (user-approved): a regex-caught write SPENDS the
                # classify call so a mixed analytical+write message decomposes
                # into chunks. Write safety is preserved structurally: the write
                # chunk still runs the operational lane with all its gates, and
                # the DISTRUST OVERRIDE in _node_classify sends a
                # non-decomposable turn operational-whole.
                #
                # Ledger row E: how far that override reaches now depends on
                # WHICH form fired. An explicit write verb binds as it always
                # did; verb-less narration only hints, because no regex can tell
                # "I did chest and triceps today" from "I did shrugs today and
                # my grip gave out" — the classifier can, so it decides.
                # Only the explicit /log boundary above stays classify-free.
                fallback_write = True
                write_intent_hint = True
                write_intent_hard = (form == WRITE_FORM_IMPERATIVE)
                # Regex-inferred write: the flow is this single message. The
                # decomposed executor re-arms this per write CHUNK when it runs.
                self._log_flow_turns = [question]

        return {
            "question":          question,
            "log_boundary":      log_boundary,
            "trailing_note":     trailing_note,
            "fallback_write":    fallback_write,
            "write_intent_hint": write_intent_hint,
            "write_intent_hard": write_intent_hard,
            "params":            params,
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

        # Stage-3 distrust override: this turn hit the write-intent regex and
        # spent the classify call ONLY to discover chunks. If the result is
        # not decomposable (single request, uniform lanes, or parse failure),
        # the regex verdict wins — operational-whole, exactly the pre-Stage-3
        # behavior. A write must never die unparseable or leak analytical.
        if state.get("write_intent_hint"):
            reqs = params.get("requests") or []
            decomposable = _is_mixed_lane_multi(params)
            parse_failed = bool(params.get("_parse_failed"))
            # A classify FAILURE always falls back to the regex verdict, whatever
            # form fired: an errored call is no evidence, and a write must never
            # be lost to one. With a successful classify, only the binding form
            # (an explicit write verb) overrules it — verb-less narration defers,
            # because the classifier is the only thing here that can tell a
            # session-to-log from a complaint-to-explain (ledger row E).
            if parse_failed or (state.get("write_intent_hard") and not decomposable):
                params["route"] = "operational"
                params.pop("_parse_failed", None)
                logger.info(
                    "[decomposition] write-hint distrust override — "
                    "operational-whole (%d chunk(s), %s)", len(reqs),
                    "classify failed" if parse_failed else "explicit write verb")

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

    async def _node_dispatch_decomposed(self, state: dict) -> dict:
        """Node: Stage-3 per-chunk execution. Reached only when the classify
        emitted ≥2 request chunks with MIXED lanes. Each chunk runs its own
        lane with its self-contained intent_text; the parts merge back in
        index order under ### headers (multi-part readability is enforced
        HERE, by code, not by the draft prompt). Per-chunk failures follow
        the analytical clean-fail contract — one broken part never kills its
        siblings. A rate limit propagates whole-turn (single checkpoint
        slot; completed parts re-run on resume — known residual).
        """
        params = state["params"]
        reqs = params.get("requests") or []

        # ── Pre-resolution gate (name disambiguation BEFORE any chunk runs).
        # A write chunk's name ambiguity is otherwise agent-internal and
        # invisible here, and an analytical chunk arming mid-loop loses the
        # sibling write. Resolve every name up front: clean names bind into
        # their chunks in place; any that stay ambiguous arm ONE slot holding
        # the FULL turn (requests intact) + all groups, and the whole turn is
        # held until the panel resolves them. Nothing executes, so no write is
        # staged-then-lost and the /log carry is never reached.
        # Guard: only arm when no slot is already live (never clobber a slot
        # mid-resolution). On a RESUME re-entry the slot was cleared by
        # resolve_disambiguation and the names are now exact, so this pass
        # re-resolves them cleanly (Tier-1) and finds nothing ambiguous — a
        # cheap idempotent no-op that falls through to normal dispatch.
        if self._pending_decomposition is None:
            groups = self._collect_decomposition_ambiguities(reqs)
            if groups:
                self._pending_decomposition = {
                    "question":   state["question"],
                    "params":     copy.deepcopy(params),
                    "groups":     groups,
                    # Back-compat single-name fields (first group) so the prose
                    # _consume_pending_decomposition fallback still works for a
                    # typed reply; the panel/structured path reads `groups`.
                    "name":       groups[0]["name"],
                    "candidates": list(groups[0]["candidates"]),
                    "created":    datetime.now().isoformat(),
                    "strikes":    0,
                    "reminded":   False,
                    "clarified":  False,
                    "rejected_override": None,
                }
                logger.info(
                    "[decomposition] pre-resolve armed: %d group(s) %r",
                    len(groups), [g["name"] for g in groups])
                opts = "\n\n".join(
                    f"**{g['name']}** — did you mean:\n"
                    + "\n".join(f"- {c}" for c in g["candidates"][:5])
                    for g in groups)
                return {
                    "answer": ("A couple of exercises need clarifying before "
                               "I can continue:\n\n" + opts),
                    "flagged_claims": [], "error": None, "decomposed": True,
                }

        headed_parts: list[tuple[str, str]] = []   # (header, text)
        # Parallel to headed_parts: True marks a STAGED-WRITE part, whose text
        # is staging-time ("staged, needs confirmation" / MSG_STAGED_NOT_SAVED).
        # After /confirm that text is stale — the "✅ logged" line supersedes it
        # — so it is excluded from the write-excluded merge the panel stash and
        # the CLI finalize path use (#19). Membership is the code path that ran
        # the chunk, never a regex on the prose.
        write_flags: list[bool] = []
        flagged_all: list = []
        error: str | None = None
        ran_write_chunk = False

        for chunk in reqs[:_DECOMP_CHUNK_CAP]:
            lane = chunk.get("lane")
            intent = (chunk.get("intent_text") or "").strip() or state["question"]
            head = intent if len(intent) <= 60 else intent[:57].rstrip() + "…"
            logger.info("[decomposition] executing chunk %d/%d lane=%s",
                        chunk.get("index", 0) + 1, len(reqs), lane)
            if lane == "out_of_scope":
                headed_parts.append((head, OUT_OF_SCOPE_REFUSAL))
                write_flags.append(False)
                continue
            part_is_write = False
            try:
                if lane == "analytical":
                    # A decomposed chunk is self-contained (intent_text is a full
                    # restatement), so it must NOT inherit shared conversation
                    # history — prior turns naming a different exercise made the
                    # draft volunteer an unsolicited "I lack that data" disclaimer
                    # about a sibling chunk's subject (#27). recall is exempt (it
                    # answers FROM history) — only analytical suppresses it.
                    chunk_params = {**chunk, "route": "analytical",
                                    "requests": None, "suppress_history": True}
                    answer, flagged = await self._run_analytical(
                        intent, chunk_params)
                    flagged_all.extend(flagged or [])
                elif lane == "recall":
                    answer = await self._call_with_per_minute_retry(
                        self._run_recall, intent)
                else:                              # operational
                    part_is_write = _is_write_intent(intent)
                    if part_is_write:
                        # Verify Input A = this chunk only, never the whole
                        # multi-part message.
                        self._log_flow_turns = [intent]
                        ran_write_chunk = True
                    answer = await self._run_operational(
                        intent, fallback_write=part_is_write)
                headed_parts.append((head, answer))
                write_flags.append(part_is_write)
            except DataAgentIntegrityError as e:
                ids_str = ", ".join(v.invariant_id for v in e.violations)
                logger.error(
                    "[decomposition] chunk integrity failure: %s", ids_str)
                error = error or str(e)
                headed_parts.append((head, (
                    f"I cannot answer this part right now — a data "
                    f"integrity check failed ({ids_str}).")))
                write_flags.append(False)          # a failed part staged nothing
            except Exception as e:
                if _is_rate_limit(e):
                    raise                          # 429 → countdown, whole turn
                if _is_transient_server_error(e):
                    logger.warning(
                        "[decomposition] chunk hit a transient 503: %s", e)
                    headed_parts.append((head, _MSG_MODEL_BUSY))
                else:
                    logger.exception("[decomposition] chunk failed: %s", e)
                    error = error or str(e)
                    headed_parts.append((head, _MSG_PIPELINE_ERROR))
                write_flags.append(False)          # a failed part staged nothing

        if len(reqs) > _DECOMP_CHUNK_CAP:
            headed_parts.append(("", (
                f"(I've answered the first {_DECOMP_CHUNK_CAP} parts — "
                f"ask the remaining {len(reqs) - _DECOMP_CHUNK_CAP} again.)")))
            write_flags.append(False)

        def _merge(parts: list[tuple[str, str]]) -> str:
            # Single-part turns drop the header (existing convention); multi-part
            # turns carry "### <header>" per part.
            if len(parts) == 1:
                return parts[0][1]
            # Multi-part: the merge owns the "### <intent>" header, so drop any
            # header the model opened its own answer with (else it doubles).
            return "\n\n".join(
                (f"### {h}\n\n{_strip_leading_md_header(t).strip()}"
                 if h else t.strip())
                for h, t in parts)

        merged = _merge(headed_parts)
        # Write-excluded merge (#19): the analytical/recall/refusal parts only.
        # Used by the confirm-panel stash and the CLI staged-finalize path so a
        # committed write is never re-described with its stale staging text.
        nonwrite_parts = [p for p, w in zip(headed_parts, write_flags) if not w]
        merged_nonwrite = _merge(nonwrite_parts) if nonwrite_parts else ""
        logger.info("[decomposition] merged %d part(s)", len(headed_parts))
        # A5 (defense in depth): a decomposed turn's continuation is owned by
        # the disambiguation slot, never the /log carry. A write chunk that
        # staged cleanly already leaves the carry down (#13), but a chunk that
        # stalled for any other reason must NOT arm a cross-turn carry that
        # would hijack the user's next message — force it down here.
        self._pending_log_carry = False
        out = {"answer": merged, "flagged_claims": flagged_all,
               "error": error, "decomposed": True,
               "decomposed_nonwrite_answer": merged_nonwrite}
        if ran_write_chunk:
            # The envelope's log_flow_turns gate reads fallback_write — a
            # write chunk found by the CLASSIFIER (regex missed the compound
            # message) must still ship its chunk-scoped verify Input A.
            out["fallback_write"] = True
        return out

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

    async def _node_recall_dispatch(self, state: dict) -> dict:
        """Node: the recall lane — restate a figure the assistant already gave,
        answered from conversation history ONLY (no package, no analytical
        re-derivation, no grounding). Flows through finalize so it lands in
        history; it does NOT record for memory extraction (a re-statement of an
        already-recorded number is noise)."""
        answer = await self._call_with_per_minute_retry(
            self._run_recall, state["question"])
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
            # Stage-3: True when this answer is a per-chunk MERGE — the
            # server must not let a confirmation panel swallow it.
            "decomposed":     state.get("decomposed", False),
            # #19: the same merge with staged-write parts excluded. The panel
            # stash and CLI finalize prepend THIS (not `answer`) to the write
            # outcome so a committed write is never re-described as "staged".
            "decomposed_nonwrite_answer":
                state.get("decomposed_nonwrite_answer"),
            # Structured disambiguation: when this turn armed the pending slot
            # (name ambiguity, single or decomposed), surface the candidate
            # groups so the server can raise the disambiguation panel instead
            # of shipping the fallback prose. None on every non-ambiguous turn.
            "disambiguation": self._disambiguation_payload(),
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
        # Built from the SAME defaults the success path applies below, so a
        # failed classify hands downstream code the same SHAPE as a successful
        # one. It used to omit display_intent/rep_target/cardio_lock, which made
        # the failure dict a quietly different object — harmless only because
        # every consumer happens to use .get() with a fallback.
        default = {
            "route":     "analytical",
            "requests":  None,
            # Marks an UNPARSEABLE/errored classify (vs. parsed-but-uncertain).
            # The caller (#5b) returns a cheap rephrase instead of running the
            # full analytical pipeline on garbage input.
            "_parse_failed": True,
            **_CHUNK_PARAM_DEFAULTS,
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
                # 1024, not 256: the Stage-1 requests array adds ~90-140 output
                # tokens per chunk, and a truncated response degrades the whole
                # classify to the _parse_failed analytical default (which would
                # silently no-write a write intent). ~6-chunk headroom.
                max_output_tokens=1024,
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
            # Ensure required keys are present (route defaults analytical — Step C flip).
            # The per-field defaults come from the one constant the per-chunk
            # sanitizer also uses, so the flat and per-chunk shapes cannot drift.
            params.setdefault("route", "analytical")
            for _k, _v in _CHUNK_PARAM_DEFAULTS.items():
                params.setdefault(_k, _v)
            _sanitize_requests(params)          # requests → None or fully valid
            _log_requests_divergence(params)    # log-only Stage-2 field data
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
            # Stage 2: arm the pending-decomposition slot on a disambiguation
            # ask. Single seam covers fresh dispatch AND _resume. A chain
            # (second ambiguity on the resumed run) re-arms with the
            # already-patched params. Deep copy: the slot must never alias
            # graph-checkpointed state.
            if state.get("disambiguation"):
                d = state["disambiguation"]
                self._pending_decomposition = {
                    "question":   question,
                    "params":     copy.deepcopy(params),
                    "name":       d["name"],
                    "candidates": list(d["candidates"]),
                    # groups: the structured/panel representation. A single
                    # analytical ambiguity is a 1-entry list, so the same
                    # panel + resolve_disambiguation path serves it.
                    "groups":     [{"name": d["name"],
                                    "candidates": list(d["candidates"]),
                                    "lane": "analytical"}],
                    "created":    datetime.now().isoformat(),
                    "strikes":    0,
                    "reminded":   False,
                    "clarified":  False,
                    "rejected_override": None,
                }
                logger.info(
                    "[decomposition] armed: name=%r, %d candidate(s), %d chunk(s)",
                    d["name"], len(d["candidates"]),
                    len(params.get("requests") or []) or 1)
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
                    # rather than silently guessing the first one. The structured
                    # payload lets _run_analytical arm the pending-decomposition
                    # slot (full candidate list, not just the shown 5).
                    names_list = "\n".join(f"- {c}" for c in candidates[:5])
                    return {
                        "early_answer": (
                            f"I found multiple exercises matching **{name}**. "
                            f"Which one did you mean?\n\n{names_list}\n\n"
                            f"Please let me know and I'll answer your question."
                        ),
                        "disambiguation": {"name": name,
                                           "candidates": list(candidates)},
                    }
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
            # Display sets are attached ONLY for display-shaped questions; an analytical
            # question gets no verbatim session/day block force-injected (root fix for
            # the whole-day dump). Default False = no display unless the classifier said so.
            display_intent=params.get("display_intent", False),
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

        # Decomposed analytical chunks opt out of shared conversation history
        # (#27): their intent_text is self-contained, and inherited history let
        # the draft editorialize about a sibling chunk's exercise.
        conversation_context = (
            None if params.get("suppress_history")
            else (self._history[-CONTEXT_WINDOW:] or None)
        )

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

        # ── Recency guard (deterministic, pure — no LLM) ───────────────────────
        # A "most recent / latest / last session" date is a computed fact
        # (progression.latest_session_date); the draft model must not override it.
        # Prompt guidance lowers the failure rate but can't guarantee it, so this
        # enforces the invariant: any such claim whose date isn't the exercise's
        # true latest is rewritten in place. Only dates inside a most-recent clause
        # are touched; ordinary date mentions are untouched.
        answer, recency_flags = _cite.recency_guard(answer, pkg)
        for f in recency_flags:
            logger.warning(
                "[coordinator] recency guard: corrected '%s' → '%s' for %s "
                "(most-recent claim carried a non-latest session date)",
                f.get("original"), f.get("corrected"), f.get("exercise"))

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

        return {"answer": answer,
                "flagged": (state.get("flagged") or []) + recency_flags}

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
        Carry set-site: a boundary turn that staged NOTHING and never reached
        the execute gate ended in a logging clarification — arm the
        single-turn carry so the user's next reply joins the /log flow.
        Anything staged (staged_this_turn) ⇒ the confirm panel takes over the
        flow ⇒ flag stays down. staging_reached_confirm stays in the OR for
        the sibling agent-driven-execute flows (goal/set edits); it is
        structurally always False for workouts under Fix 5 (the SERVER calls
        execute after /confirm, never the agent), which is why it could
        never be the sole signal (#13: carry mis-armed on every successful
        staged write).
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
        # decide; the regex only detects that a claim is being made, so a
        # clarification question with the same flags is never touched. Tiers:
        #   - wrote (db_write_effect) or reached the execute gate
        #     (staging_reached_confirm) → untouched, the claim has backing;
        #   - staged only (staged_this_turn, hole A) → the claim is premature,
        #     not baseless: replace with the truthful staged-not-saved text
        #     (never "nothing was written" — the CLI executes immediately
        #     after this answer, and a false-failure line would contradict
        #     the ✅ that follows);
        #   - none of the three → any completed-write claim is false by
        #     construction: replace with the no-write message.
        # Claim presence: the narrow regex everywhere; the looser one wherever a
        # write was actually ATTEMPTED this turn. `write_attempted` is the
        # structural widening — it covers goal and set-edit flows, which the old
        # logging-only scope left unguarded (an audit found 12 of 13 completion
        # claims there reaching the user unchecked), while still excluding
        # research answers, which call no write tool at all.
        _write_flow = (log_boundary or fallback_write
                       or bool(result.get("write_attempted")))
        _claim_made = bool(_WRITE_SUCCESS_CLAIM_RE.search(answer)) or (
            _write_flow and bool(_WRITE_COMPLETION_CLAIM_RE.search(answer)))
        if (not result.get("db_write_effect")
                and not result.get("staging_reached_confirm")
                and _claim_made):
            if result.get("staged_this_turn"):
                logger.warning(
                    "[coordinator] rewrote premature saved-claim on a "
                    "staged-only turn: %r", answer[:120])
                answer = MSG_STAGED_NOT_SAVED
            else:
                logger.warning(
                    "[coordinator] suppressed unbacked write-success claim: %r",
                    answer[:120])
                answer = MSG_NO_WRITE_OCCURRED
        if log_boundary or fallback_write:
            # Fallback (regex-inferred) writes are the same flow as /log turns:
            # a turn that ends pending a logging clarification must carry the
            # originating flow-turn text into the next reply, or the stage-2
            # verify diffs the staged batch against the bare reply ("Today").
            self._pending_log_carry = not (
                result.get("staging_reached_confirm", False)
                or result.get("staged_this_turn", False))
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

    async def _run_recall(self, question: str) -> str:
        """
        Answer a recall/meta follow-up ("what was that number you just mentioned?")
        from conversation history ONLY — no package build, no re-derivation. The
        model is handed just the recent turns + the question, so it cannot reach for
        a salient package scalar (the wrong-predicate temptation is removed, not
        merely discouraged). Returns the answer text. Fail-open → a gentle clarify.
        """
        history_txt = "\n".join(
            f"{t.get('role', '?').upper()}: {t.get('content', '')}"
            for t in self._history[-CONTEXT_WINDOW:]
        ) or "(no earlier turns in this session)"
        prompt = f"[CONVERSATION]\n{history_txt}\n\n[CURRENT MESSAGE]\n{question}"
        try:
            config = types.GenerateContentConfig(
                system_instruction=_RECALL_SYSTEM,
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
            return raw.strip() or _RECALL_FALLBACK
        except Exception as e:
            if _is_rate_limit(e):
                raise
            logger.warning("[coordinator] recall failed: %s — returning clarify", e)
            return _RECALL_FALLBACK
