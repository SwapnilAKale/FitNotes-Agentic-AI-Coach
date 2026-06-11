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

ROUTING RULE: when uncertain, default to operational.
The single agent can always answer something; a wrong analytical
package produces a confusing analysis.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from google import genai
from google.genai import types

from src.data_agent import prepare_analysis_package, DataAgentIntegrityError
from src.analysis_agent import run as analysis_run

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

# Write-intent guard: questions matching this are routed OPERATIONAL without
# calling the classifier so a write can never reach the analytical path.
_WRITE_INTENT_RE = re.compile(
    r"(?i)"
    r"(?:"
    r"\b(?:log|record|add|save|delete|remove|update|change|correct)\b"
    r".{0,50}"
    r"\b(?:workout|set|goal|bodyweight|weight|exercise|session|reps?)\b"
    r"|"
    r"\bset\s+a?\s*goal\b"
    r"|"
    r"\bI\s+did\b.{0,80}\b(?:today|yesterday|this\s+morning|this\s+week)\b"
    r")"
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

DEFAULT: when uncertain, use "operational".

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
  "route": "analytical" | "operational",
  "exercise_names": ["..."] | null,
  "muscle_groups": ["..."] | null,
  "query_period_days": 90 | null,
  "needs_custom_sql": false,
  "custom_sql_intent": null
}
""".strip()


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
        # ── 1. Classify (or short-circuit for obvious write operations) ──────
        # Misrouting a write to analytical bypasses the confirmation gate.
        if _WRITE_INTENT_RE.search(question):
            params = {
                "route":             "operational",
                "exercise_names":    None,
                "muscle_groups":     None,
                "query_period_days": 90,
                "needs_custom_sql":  False,
                "custom_sql_intent": None,
            }
        else:
            params = await self._classify(question)
        route  = params.get("route", "operational")

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

    # ── Classification ────────────────────────────────────────────────────────

    async def _classify(self, question: str) -> dict:
        """
        Single Gemini call: route + extract parameters.
        Defaults to operational on any failure — safe fallback.
        """
        default = {
            "route":              "operational",
            "exercise_names":     None,
            "muscle_groups":      None,
            "query_period_days":  90,
            "needs_custom_sql":   False,
            "custom_sql_intent":  None,
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
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    raw += part.text

            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.strip().startswith("```")
                ).strip()

            params = json.loads(raw)
            # Ensure required keys are present
            params.setdefault("route",             "operational")
            params.setdefault("exercise_names",    None)
            params.setdefault("muscle_groups",     None)
            params.setdefault("query_period_days", 90)
            params.setdefault("needs_custom_sql",  False)
            params.setdefault("custom_sql_intent", None)
            return params

        except Exception as e:
            if _is_rate_limit(e):
                raise
            logger.warning("[coordinator] classify failed: %s — defaulting to operational", e)
            return default

    # ── Analytical pipeline ───────────────────────────────────────────────────

    async def _run_analytical(
        self,
        question: str,
        params:   dict,
    ) -> tuple[str, list]:
        """
        Full analytical pipeline:
          1. prepare_analysis_package() with extracted params
          2. analysis_agent.run() → draft + grounding edits
          3. coverage check → if incomplete, one retry with gap context

        Returns (final_answer, flagged_claims).
        """
        exercise_names    = params.get("exercise_names")
        if exercise_names:
            exercise_names = [n.strip() for n in exercise_names]
        if exercise_names:
            from src.shared.resolver import resolve_exercise_name
            db_path = os.environ.get(
                "FITNOTES_DB_PATH",
                os.path.join(os.path.dirname(__file__), "..", "data", "FitNotes_Backup.fitnotes")
            )
            resolved = []
            for name in exercise_names:
                result = resolve_exercise_name(name, db_path)
                match = result.get("match") or (result.get("candidates") or [None])[0]
                # Keep the original name when resolution fails entirely —
                # the package filter then reports it in
                # unresolved_exercise_names and the user is told it wasn't
                # found. Dropping it here made the name vanish silently:
                # the answer covered the other exercises with no mention
                # of the one that didn't exist.
                resolved.append(match or name)
            exercise_names = resolved
        muscle_groups     = params.get("muscle_groups")
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

        # Best-effort enrichment — research and memories never block the pipeline
        try:
            from src.shared.rag import search_fitness_knowledge
            research = search_fitness_knowledge(question) or None
        except Exception:
            research = None
        try:
            from src.shared.memory import retrieve_relevant_memories
            memories = retrieve_relevant_memories(question) or None
        except Exception:
            memories = None

        # Supplementary cross-cutting SQL (counts, dates, gaps, patterns)
        custom_query = None
        if params.get("needs_custom_sql") and params.get("custom_sql_intent"):
            custom_query = await self._generate_custom_sql(
                question, params["custom_sql_intent"]
            )

        # First pass
        answer, flagged = await analysis_run(
            pkg, scoped_question, research, memories, conversation_context, custom_query
        )

        # Coverage check (question + answer only, no data)
        answer, complete = await self._coverage_check(question, answer)

        if not complete:
            # One retry: add first draft + gaps to context, re-run Analysis Agent
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
            answer, flagged2 = await analysis_run(
                pkg, scoped_question, research, memories, retry_context, custom_query
            )
            flagged.extend(flagged2)
            # No second coverage check — return best effort

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
                f"Focus: {intent}. Return counts, dates, gaps, or patterns. "
                f"Do not return individual set weights."
            )
            sql = await asyncio.to_thread(generate_sql, prompt_question, schema)
            result = await asyncio.to_thread(run_custom_query, sql)
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
            for part in response.candidates[0].content.parts:
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
