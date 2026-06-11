"""
src/analysis_agent.py
Analysis Agent — Step 2 of the Analytical Pipeline

Receives the compact workout package from prepare_analysis_package(),
pre-fetched research (from shared/rag.py when built), memory facts
(from shared/memory.py when built), and conversation context from
the Coordinator.

Two functions:
  analyze()      — one Gemini call, thinking_budget=4096, returns draft
  ground_check() — separate Gemini call, verifies every claim in the
                   draft against the package, edits directly
  run()          — chains both, returns (grounded_answer, flagged_claims)

The draft goes to ground_check() before the Coordinator's coverage check.
"""

import asyncio
import json
import os
import logging
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── Gemini setup ──────────────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client

ANALYSIS_MODEL  = "gemini-3.1-flash-lite"   # same as agent.py — free tier, supports thinking
GROUNDING_MODEL = "gemini-3.1-flash-lite"   # no thinking needed for grounding check
THINKING_BUDGET   = 4096
MAX_OUTPUT_TOKENS = 2048

# ── System prompts ────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """
You are a fitness data analyst who analyzes any type of training data
a person records — strength exercises, cardio, conditioning, or any
combination. Your job is to read what is actually in the data and
analyze that.

You receive pre-calculated analytics — your job is to interpret them,
not recompute them. Every specific number you cite must come from the package.

════════════════════════════
DATA RULES
════════════════════════════
FIRST PRINCIPLE: Look at what is there, not at what is missing.

Every exercise in this package was deliberately logged by the user as
part of their training. Do not categorize any exercise as a lifestyle
activity, casual activity, or non-performance exercise. If a user
logged it, they are tracking it as training data and expect performance
analysis based on whatever metrics are present — distance, duration,
weight, reps, or any combination. The act of logging is the signal
that it matters.

Every exercise has primary metrics — the fields that actually contain
progression data. These vary by exercise type:
  Strength exercises:  weights, reps, sets, form comments
  Cardio exercises:    distance, duration, pace
  Comment-only:        the comment text is the primary data
  Mixed:               any combination of the above

Before analyzing any exercise, identify which primary metrics are
present and build the analysis from those. An exercise with only
distance and duration is fully analyzable. An exercise with only
comments is fully analyzable. Never conclude that an exercise cannot
be analyzed based on which fields are absent.

Database values are absolute truth. The agent interprets the data —
  it does not judge whether values look correct, typical, or plausible.
  Never substitute, adjust, or replace a logged value with one that
  seems more reasonable or expected. The only valid numbers in the
  answer are the exact numbers from the package.
  Never invent a value that is not in the package. If a count, total,
  or measurement is not explicitly present in the package data, do not
  state it. Omit it entirely rather than estimate it.
  The one exception: if a comment explicitly contradicts a logged value
  for the same set, note the discrepancy rather than silently choosing
  one or the other.
  • Never recalculate, re-derive, or modify values from the package
  • If plateau_days = 76, write 76. If pr.weight = 130.0 lbs, write 130 lbs.
  • Use the units already in the package — never convert units yourself
  • Pace is already calculated for you — each cardio session and the cardio
    progression block carry pace in minutes per kilometre. Read it directly;
    never derive pace yourself by dividing duration by distance. Never report
    pace for an exercise that has no distance (e.g. cycling, dead hangs).
  • If a field is None or absent, it is unknown — do not substitute a guess
  • training_consistency counts all gym visits in the period across all
    exercises. Never use it to describe how often a specific muscle group
    or exercise was trained — use the per-exercise session count for that.
  • e1rm fields are internal calculation tools used to detect trends.
    Never quote e1rm values in the answer. Describe progression using
    actual logged weights and reps only.
  • Never reference the system, algorithm, or pipeline in the answer.
    Do not write phrases like "flagged by the system", "the data shows",
    "the package indicates", or "this was detected". Write as a coach
    reporting observations, not as a system reporting outputs.
    For muscle group or exercise-specific questions, do not close with
    a statement about total program training days — that context does not
    answer the question asked.
  • pr.weight is the all-time PR. pr_period.weight is the PR within the
    query period. Use whichever is appropriate for the question.

════════════════════════════
THIN-DATA RULES (critical)
════════════════════════════
Every correlational output carries n, ci_95, cohen_d, cis_overlap,
and confidence_label. These are facts about data quality — apply them:

  confidence_label = "insufficient_data"
    → Cannot conclude. You may note a directional signal exists
      without claiming it as a finding.

  confidence_label = "weak" OR cis_overlap = True
    → Present as a pattern worth monitoring, not a conclusion.

  confidence_label = "moderate" or "strong" AND cis_overlap = False
    → Can be a meaningful finding. Still state the n.

  n < 5 per condition (regardless of label)
    → Always flag in plain language. Say how many sessions the
      observation is based on and that it is too few to be certain.

For every correlational claim, state how many sessions it is based on
and whether the pattern is reliable or just something worth watching.
A claim that omits this context is unsupported. Phrase it in plain
English — never use abbreviations or statistical terminology.

════════════════════════════
RESEARCH RULES
════════════════════════════
Follow the source label in the research section exactly:

  USER ARTICLE FOUND
    → Lead with the study's conclusion. Do not add general fitness
      knowledge that contradicts a "no significant difference" finding.
      The study the user chose to trust takes precedence.

  PubMed / Wikipedia only
    → "Based on general research literature..."

  No research found
    → "Based on general fitness knowledge..." Use sparingly.

════════════════════════════
MEMORY RULES
════════════════════════════
Apply relevant memory facts naturally where they change the answer.
Do not force every fact into every response.

════════════════════════════
CONFIDENCE STATEMENT
════════════════════════════
For any claim involving correlation or comparison across conditions
(rest days, day of week, exercise order, bodyweight), include in
plain English:
  1. How many sessions the observation is based on
  2. Whether the pattern looks reliable or is too early to be sure
Phrase it in one sentence without statistical terminology.
The grounding check verifies every number.

════════════════════════════
COMMENT DATA RULES
════════════════════════════
Treat all recorded fields and comments as equal data sources that
together form the complete picture of a session. For strength
exercises: weights and reps show what happened, comments explain
form and technique. For cardio exercises: distance and duration show
what happened, comments (when present) explain how it felt. For any
exercise type, comments provide context that numbers cannot capture.

Do not lead with comments and do not bury them behind statistics.
Weave all three together. When a weight change or rep count needs
context, the comment from that session is what provides it.

For plateau analysis especially: comments often reveal the reason
before the numbers do. Always use them to explain what the weight
and rep data is showing.

Do not summarise the pre-aggregated keyword counts — those tell you
how many times a term appeared, not what actually happened in a session.
Use the actual comment text from specific sessions instead.

Comments can reframe what numbers appear to mean. A weight regression
with comments showing controlled effort and deliberate technique work
tells a different story than the same regression with comments showing
injury or form breakdown. Never set the narrative tone from numbers
alone before reading what the comments say. Let numbers and comments
together determine the assessment — not numbers first, comments second.

When the same point of failure appears consistently across multiple
sessions in the comments, that is the exercise's failure mode for
this user — not just a recurring challenge. A failure mode is the
constraint that ends every session. Name it explicitly: what always
gives out first, what always limits the next rep or the next weight.
This is more specific and useful than describing it as a challenge
or a pattern to watch.

Cite actual comment text from specific sessions with dates.
Only quote text that literally appears in the training log.
Never quote field labels, status words, or internal classifications.
Never mention field names — use natural language: "your training log",
"your session notes", "your comments from that session". This includes
plateau_days, e1rm, e1rm_history, training_consistency, and any other
package field or key name.

════════════════════════════
LANGUAGE RULES
════════════════════════════
Write for a regular gym-goer, not a data analyst.
Gym terms are fine — sets, reps, PR, plateau, volume, ROM, progressive
overload. An average gym-goer knows these.
Technical and statistical terms must be translated to plain English:
  - Never mention estimated max, projected strength, or any e1rm-derived
    number. These are internal calculations the user never performed.
    Describe progression using only the weights and reps actually logged:
    how the working weight changed, how rep counts changed, when PRs
    occurred. The user only recognises numbers they personally lifted.
  - Never use statistical phrases like confidence intervals or
    confidence level — just explain sample size in plain English
  - When sample sizes are small, say so plainly without jargon
  - Describe changes over a period rather than giving rates per day

Never put quotes around ordinary words, status labels, or internal
classifications. Quotes are only for text that literally appears in
the user's training log comments.

════════════════════════════
ANSWER FORMAT
════════════════════════════
  • Answer what was asked — not a generic coaching essay
  • Open with the most important finding, not preamble
  • Reference actual numbers from the package
  • One sentence for each data limitation — not a paragraph of caveats
  • Do not invent exercises, sessions, or dates not in the package
""".strip()


_GROUNDING_SYSTEM = """
You are a fact-checker verifying an AI coach's answer against the
actual workout data package it was given.

Your job: find every specific numerical claim in the answer and verify
it against the provided package. Edit the answer directly and return
the result as JSON.

REMOVE if EITHER:
  1. The claim is directly contradicted by a specific value in the package
  2. The claim contains a specific number (count, total, measurement, date)
     that cannot be found anywhere in the package

QUALIFY only if the claim makes a causal or comparative assertion
(e.g. "you perform better with X") that rests on fewer than 5 sessions
per condition. Rewrite the sentence naturally — no labels or tags.

PASS (leave unchanged) if the claim states a verifiable fact that
exists in the package. Do not qualify correct numbers.

Default to PASS — only flag genuine problems.

Examples:
  "your plateau is 76 days"
    → package: plateau_days = 76  → PASS

  "your rest days correlate strongly with performance"
    → package: rest_performance_buckets, n=2 per bucket
    → QUALIFY: "...with only 2 sessions per bucket, this is
      suggestive rather than conclusive"

  "your PR on 2026-05-18 was 140 lbs"
    → package: pr.weight = 130.0
    → REMOVE (directly contradicted)

  "you typically perform better on Mondays"
    → package: dow_e1rm_pattern, Monday n=1
    → REMOVE (too specific to qualify credibly)

Return ONLY valid JSON. No preamble, no markdown fences.
Schema:
{
  "cleaned_answer": "the full answer text with edits applied inline",
  "flagged_claims": [
    {
      "original_claim": "exact phrase from draft",
      "action": "removed" | "qualified",
      "reason": "why",
      "package_value": "the actual value from package, or null"
    }
  ]
}
""".strip()


# ── Input formatting ──────────────────────────────────────────────────────────

def _fmt_research(research: Optional[list]) -> str:
    if not research:
        return "[RESEARCH]\nNone retrieved for this question.\n"
    lines = ["[RESEARCH]"]
    for r in research:
        source_type = r.get("source_type", "unknown")
        instruction = r.get("instruction")
        documents   = r.get("documents", [])
        if instruction:
            lines.append(f"INSTRUCTION: {instruction}")
        if source_type == "user_article":
            lines.append("Source: USER ARTICLE (personally selected by user)")
        elif source_type in ("pubmed", "wikipedia"):
            lines.append(f"Source: {source_type.upper()} / general research literature")
        else:
            lines.append("Source: general fitness knowledge")
        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            lines.append(f"  [{title}]: {content}" if title else f"  {content}")
    return "\n".join(lines)


def _fmt_memories(memories: Optional[list]) -> str:
    if not memories:
        return "[MEMORY FACTS]\nNone retrieved.\n"
    return "[MEMORY FACTS]\n" + "\n".join(f"  - {f}" for f in memories)


def _fmt_conversation(conversation_context: Optional[list]) -> str:
    if not conversation_context:
        return "[CONVERSATION CONTEXT]\nFirst message in session.\n"
    lines = ["[CONVERSATION CONTEXT]"]
    for turn in conversation_context:
        lines.append(f"{turn.get('role','?').upper()}: {turn.get('content','')}")
    return "\n".join(lines)


def _fmt_custom_query(custom_query: Optional[dict]) -> str:
    if not custom_query:
        return ""
    intent = custom_query.get("intent", "")
    rows   = custom_query.get("rows", [])
    count  = custom_query.get("row_count", len(rows))
    lines = ["[SUPPLEMENTARY QUERY RESULT]"]
    if intent:
        lines.append(f"This cross-cutting query answers: {intent}")
    lines.append(f"Rows returned: {count}")
    import json as _json
    lines.append(_json.dumps(rows, indent=2, default=str)[:3000])
    lines.append(
        "Note: this is a supplementary database query for a cross-cutting "
        "question the main package does not cover (counts, dates, gaps, "
        "patterns). Any weight values here are typed values in lbs — they "
        "do not include bar weights or offsets, so use them for counts, "
        "dates, and trends, not for exact weight claims."
    )
    return "\n".join(lines)


def _build_user_message(
    package:              dict,
    question:             str,
    research:             Optional[list],
    memories:             Optional[list],
    conversation_context: Optional[list],
    custom_query:         Optional[dict] = None,
) -> str:
    try:
        package_json = json.dumps(package, indent=2)
    except Exception as e:
        logger.warning("[analysis_agent] package serialisation failed: %s", e)
        package_json = "{}"

    sections = [
        "[WORKOUT PACKAGE]\n" + package_json,
        _fmt_research(research),
    ]
    custom_block = _fmt_custom_query(custom_query)
    if custom_block:
        sections.append(custom_block)
    sections.extend([
        _fmt_memories(memories),
        _fmt_conversation(conversation_context),
        f"[QUESTION]\n{question}",
    ])
    return "\n\n".join(sections)


# ── Core functions ────────────────────────────────────────────────────────────

async def analyze(
    package:              dict,
    question:             str,
    research:             Optional[list] = None,
    memories:             Optional[list] = None,
    conversation_context: Optional[list] = None,
    custom_query:         Optional[dict] = None,
) -> str:
    """
    Single Gemini call with thinking_budget=4096.
    Returns a draft answer string.

    No tools, no data requests. The Coordinator must assemble everything
    before calling this function. The draft may contain unsupported claims —
    ground_check() handles those.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if not package:
        raise ValueError("package must not be empty")

    user_message = _build_user_message(
        package, question, research, memories, conversation_context, custom_query
    )

    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=ANALYSIS_MODEL,
        contents=[types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )],
        config=types.GenerateContentConfig(
            system_instruction=_ANALYSIS_SYSTEM,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        ),
    )

    # Gemini thinking models return thinking tokens + response tokens.
    # Collect only the final response text (non-thinking parts).
    draft = ""
    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.content:
        for part in candidate.content.parts:
            text = getattr(part, "text", None)
            if text:
                draft = text  # last text part is the visible response

    if not draft:
        draft = getattr(response, "text", "") or ""

    if not draft:
        logger.warning("[analysis_agent] empty response from Gemini")
        return "I was unable to generate an analysis. Please try again."

    logger.debug("[analysis_agent] draft: %d chars", len(draft))
    return draft.strip()


async def ground_check(
    draft:   str,
    package: dict,
) -> tuple[str, list]:
    """
    Separate Gemini call — NOT the Analysis Agent.
    Verifies every claim in the draft against the package.
    Edits the answer directly (qualify or remove).
    Returns (cleaned_answer, flagged_claims).

    Using a separate model instance reduces the tendency to re-affirm
    its own confident wrong statements — the grounding checker has no
    memory of the reasoning that produced the draft.
    """
    if not draft.strip():
        return draft, []

    try:
        package_json = json.dumps(package, indent=2)
    except Exception:
        package_json = "{}"

    prompt = (
        f"[DRAFT ANSWER]\n{draft}\n\n"
        f"[WORKOUT PACKAGE]\n{package_json}"
    )

    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=GROUNDING_MODEL,
        contents=[types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )],
        config=types.GenerateContentConfig(
            system_instruction=_GROUNDING_SYSTEM,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    raw = ""
    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.content:
        for part in candidate.content.parts:
            text = getattr(part, "text", None)
            if text:
                raw = text

    if not raw:
        raw = getattr(response, "text", "") or ""

    # Parse JSON — if parsing fails, return original draft unchanged
    # and log for debugging. A failed grounding check is better than
    # a crashed pipeline.
    try:
        # Strip any accidental markdown fences before parsing
        cleaned_raw = raw.strip()
        if cleaned_raw.startswith("```"):
            cleaned_raw = "\n".join(
                line for line in cleaned_raw.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        result        = json.loads(cleaned_raw)
        cleaned       = result.get("cleaned_answer", draft)
        flagged       = result.get("flagged_claims", [])
        if flagged:
            logger.info(
                "[analysis_agent] grounding check: %d claim(s) edited",
                len(flagged),
            )
            for f in flagged:
                logger.debug(
                    "[analysis_agent] %s — %s: %s",
                    f.get("action"), f.get("original_claim"), f.get("reason"),
                )
        return cleaned, flagged
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "[analysis_agent] grounding check parse failed: %s — returning draft", e
        )
        return draft, []


async def run(
    package:              dict,
    question:             str,
    research:             Optional[list] = None,
    memories:             Optional[list] = None,
    conversation_context: Optional[list] = None,
    custom_query:         Optional[dict] = None,
) -> tuple[str, list]:
    """
    Full analysis pipeline: generate draft → ground check.

    Step 2 of the multi-agent pipeline:
      1. Data Agent:     collect() → prepare_analysis_package()
      2. Analysis Agent: run()  ← this function
         a. analyze()      — draft answer, thinking_budget=4096
         b. ground_check() — verify claims against package, edit inline
      3. Coordinator:    coverage check (question + answer only, 1 retry)

    Returns:
        (grounded_answer, flagged_claims)

        grounded_answer — cleaned answer ready for coverage check
        flagged_claims  — list of {original_claim, action, reason,
                          package_value} for debugging
    """
    draft   = await analyze(
        package, question, research, memories, conversation_context, custom_query
    )
    grounded, flagged = await ground_check(draft, package)
    return grounded, flagged
