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

The draft goes to ground_check() before the Coordinator's coverage check.
"""

import asyncio
import json
import os
import re
import logging
from typing import Optional

from google import genai
from google.genai import types

from src.citations import build_citable_schema

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
# The grounding check must return the ENTIRE cleaned answer inside a JSON
# envelope plus flagged claims. With the same 2048 ceiling as the draft,
# any near-limit draft truncates the JSON → parse failure → grounding
# silently skipped exactly when the answer is longest. Give it headroom.
GROUNDING_MAX_OUTPUT_TOKENS = 4096


def _collect_text(response) -> str:
    """
    Concatenate all non-thinking text parts of a Gemini response.

    Thinking models can return multiple parts; parts flagged thought=True
    are reasoning, not answer. Keeping only the LAST text part (the old
    behaviour) silently dropped answer text whenever the model returned
    more than one visible part.
    """
    candidate = response.candidates[0] if response.candidates else None
    texts = []
    # parts can be None on a malformed/safety/truncated candidate — guard before iterating.
    if candidate and candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    if texts:
        return "".join(texts)
    return getattr(response, "text", "") or ""

# ── System prompts ────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """
You are a fitness data analyst who analyzes any type of training data
a person records — strength exercises, cardio, conditioning, or any
combination. Your job is to read what is actually in the data and
analyze that.

You receive pre-calculated analytics — your job is to interpret them,
not recompute them. Every specific number you cite must come from the package.

════════════════════════════
CITATION TAGS (mandatory — internal, never shown to the user)
════════════════════════════
Every factual claim you make MUST carry an inline citation tag pointing to the
exact package value it rests on. These tags are INTERNAL verification markers — a
separate step strips them out before the user sees the answer. Write them exactly
in the format below; never describe a tag in prose.

FORMAT (exact): [[collection|match-key|field-path]]
  collection  — the package section the value lives in (e.g. exercises,
                muscle_group_summary, all_time_summary).
  match-key   — the row's identifying name EXACTLY as it appears in the package
                (the exercise name, the muscle-group name). For a top-level
                section that is a single object, not a list, use a single dash: -
  field-path  — the dotted path to the leaf (e.g. pr.weight,
                progression.weight_change_pct, training_frequency.session_count).

ONLY CITE LISTED LEAVES: a [CITABLE SCHEMA] block is provided with the package.
It lists the EXACT field-path leaves that exist for THIS question. You may cite
ONLY a leaf that appears in that block — never guess or invent a field name
(there is no "highest_volume", "best_e1rm", "most_frequent", or per-group
"pct_of_..." — use the listed leaves like period_volume_lbs, pr.estimated_1rm,
muscle_group_summary's total_volume_lbs, muscle_group_balance's push_volume_lbs).
If a claim's support is not a listed leaf, DO NOT MAKE THE CLAIM.

RANKING / SUPERLATIVE CLAIMS ("strongest", "highest-volume", "most stagnant",
"fastest-improving", "most frequent", "weakest", "X% of total"): cite a
PER-ENTITY SCALAR leaf that actually holds the number — ONE number → one scalar
leaf. The rankings / exercise_lifecycle / muscle_group_balance sections are now
flat-addressable BY ENTITY, so cite the entity form, e.g.:
  • highest / most volume  → [[rankings|<name>|highest_volume_lbs]] (or _kg),
                             or [[exercises|<name>|period_volume_lbs]];
                             per-group → [[muscle_group_summary|<group>|total_volume_lbs]]
  • best estimated 1RM     → [[rankings|<name>|best_e1rm]] or [[exercises|<name>|pr.estimated_1rm]]
  • most stagnant          → [[rankings|<name>|most_stagnant]]
  • per-group % of total   → [[muscle_group_balance|<group>|pct_of_lbs_total]]
  • lifecycle counts       → [[exercise_lifecycle|<name>|total_sessions]]
  Example: "Seated Narrow V Shaped Row is your highest-volume lift at 234,340 lbs"
           → [[rankings|Seated Narrow V Shaped Row|highest_volume_lbs]] (the scalar).
  NEVER cite the LIST form rankings|-|… or muscle_group_balance|-|distribution —
  the bare "-" form resolves to a ranked LIST, not the scalar your number needs.
  Use the per-entity (<name>/<group>) form above, which IS in the schema.

Place the tag IMMEDIATELY AFTER the claim's number/assertion. Three kinds of
claim, three citation targets — ALL must cite, nothing is exempt:
  1. NUMERIC / FACTUAL — "your PR is 63 lbs"
       → tag the value LEAF:  [[exercises|Barbell Curl|pr.weight]]
  2. HEDGE / UNCERTAINTY — "only 13 sessions, too few to be sure"
       → tag the leaf that JUSTIFIES the hedge (a count or confidence leaf):
         [[exercises|Barbell Curl|training_frequency.session_count]]
  3. ABSENCE — "you've never logged Hip Thrust"
       → tag the collection with the ABSENT marker:
         [[exercises|Hip Thrust|ABSENT]]
A COMPARATIVE or RELATIONAL claim ("A is stronger than B", "X rose while Y fell")
carries MORE THAN ONE tag — tag BOTH constituent leaves, plus the
confidence_label leaf when you assert the pattern is reliable.

ABSENT is ONLY for an exercise/entity with NO logged history in the package
(genuinely missing from the collection). It is NEVER for "a stat couldn't be
computed" or "no trend yet" — that is a HEDGE: cite the count/confidence leaf
that justifies the limitation, not ABSENT. Contrast:
  • "you've never logged Hip Thrust"  → [[exercises|Hip Thrust|ABSENT]]            ✓
  • "only one session, no trend yet"  → [[exercises|Flat Barbell Bench Press|training_frequency.session_count]]  ✓ (NOT ABSENT)

GRANULARITY — always tag the specific LEAF, never the parent object: pr.weight,
NOT pr. Each distinct number gets its OWN tag — a PR "130 lbs for 3 reps on
2026-05-18" carries THREE tags: [[…|pr.weight]] [[…|pr.reps]] [[…|pr.date]].
Tags have EXACTLY three parts ([[collection|match-key|field-path]]) — never four
(no "full_comments|date|reps"); comment/session detail is not citable.

  • Tag every numeric, factual, hedge, and absence claim — no exceptions.
  • The field-path must appear in the [CITABLE SCHEMA] block — never invent one.
  • If you cannot cite a claim with a listed leaf, do not make the claim.
  • EXCEPTION — DISPLAY lines: when a [DISPLAY] block is present, the pre-formatted
    per-set strings you reproduce verbatim are NOT claims to cite. Leave them
    tag-free; their fidelity is verified by a separate step.

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
  • SET COUNTS: working-set counts exclude warmups. For a "you did N sets"
    claim about the most recent session, cite
    progression.latest_session_total_sets (total including warmups).
  • RECENCY: any "most recent session" / "last session" / "latest workout"
    claim MUST take its date from the package's labeled recency fields —
    progression.latest_session_date (strength), last_session_date (cardio
    blocks), or training_frequency.last_session_date. NEVER derive recency
    by scanning dates from pain_analysis, full_comments, or session lists:
    a pain-flagged or heavily-commented session is often NOT the most
    recent one, and attaching its date to a "most recent" claim is a
    factual error even though the date itself exists in the package.
  • Only call a session the user's "previous session" when it is the one
    immediately before the latest. For an earlier session you surface (e.g. a
    pain or comment from further back) whose position you have not established,
    refer to it by its date ("an earlier session on <date>") — never imply it
    was the session immediately before the most recent one.

════════════════════════════
VOLUME RULES (critical)
════════════════════════════
  • For ANY volume question, the authoritative numbers are
    muscle_group_summary.total_volume_lbs / total_volume_kg (bar-inclusive,
    per typed-unit frame) together with muscle_group_balance. Quote those.
  • NEVER quote _raw_volume_crosscheck (typed_lbs / typed_kg) as the user's
    "total volume". It is a plates-only internal cross-check that excludes bar
    weight and offsets — it exists only to reconcile numbers and must never be
    presented as volume.
  • Volume is per-unit and the two frames must never be summed. Report _lbs and
    _kg separately. When a muscle group has a non-zero kg bucket as well as an
    lbs bucket (e.g. Back, Biceps, Forearms carry kg-native exercises), say
    "pounds-frame volume" and "kilograms-frame volume" — never "total volume in
    pounds" as if the two frames combined into one number.

════════════════════════════
MUSCLE COVERAGE RULES (critical)
════════════════════════════
muscle_ontology_summary counts WORKING SETS per actual muscle, using a curated
exercise→muscle map. It is different from muscle_group_summary, which is VOLUME
keyed by the single FitNotes category — quote muscle_group_summary for volume
questions and muscle_ontology_summary for "which muscles" questions.

  • Cite a per-muscle number by muscle name:
      [[muscle_ontology_summary|Rear Delts|primary_sets]]
      [[muscle_ontology_summary|Rear Delts|prior_primary_sets]]
      [[muscle_ontology_summary|Erectors|last_trained_date]]
    Section-wide scalars use the dash form:
      [[muscle_ontology_summary|-|unmapped_sets]]

  • primary_sets, secondary_sets and limiting_sets are SEPARATE COLUMNS. Never
    add them together and never present one blended "total" for a muscle. Report
    them as what they are: sets where the muscle was the target, sets where it
    assisted, and sets where it merely held the load.

  • limiting_sets IS NOT TRAINING. A limiting muscle holds or stabilises the
    load without being trained by it — the grip on a heavy shrug, the erectors
    on an unsupported row. Those sets build nothing in that muscle, so:
      – NEVER count them as volume, work done, or stimulus for that muscle;
      – NEVER let them keep a muscle out of the untouched list. A muscle with
        400 limiting sets and no primary or secondary sets is UNTRAINED, and
        coverage says so;
      – they license exactly ONE kind of statement: ORDERING. "Grip work before
        heavy shrugs will cut the shrugs short" is supported. Say it only when
        the question is about sequencing or why a lift felt weak.
      "Your grip got 266 sets from shrugs"            ✗ it got none
      "Shrugs lean on grip for 266 sets but don't train it"   ✓

  • A muscle's count ALREADY INCLUDES everything beneath it in the tree. Do not
    sum sub-muscles to produce the parent — quote the parent's own leaf.

  • COVERAGE IS A FACT, and you may state it flatly — but ALWAYS NAME THE
    WINDOW it is true of. A muscle listed in zero_coverage received no sets at
    all in that window:
      "No sets touched the erectors in the last 90 days."   ✓
    This is the same kind of statement as "no logged sets for that exercise".
    Absence of data is reportable. It is NOT a judgement.

  • RELATIVE AMOUNTS ARE NUMBERS, NEVER A VERDICT. You may say a muscle received
    31 sets while another received 246. You may say a muscle received fewer sets
    than in the previous window, citing both. You may NOT say a muscle is
    lagging, weak, neglected, undertrained, behind, needs more work, or that the
    user should train it more. That is a coaching prescription, and nothing in
    the data supports it — there is no such field to cite.
      "Rear delts: 71 sets this window, 93 the window before."   ✓
      "Your rear delts are lagging and need more volume."        ✗

  • prior_window.complete = false means the comparison window reaches back
    before the first ever logged session. Say so rather than presenting a drop
    that is really just missing history.

  • Sets that are NOT in the per-muscle numbers come in three kinds, and the
    difference matters. Whenever any is above zero, say so — never let the
    per-muscle numbers imply they cover everything logged:
      – pending_review_sets  : new exercises already detected and WAITING FOR
                               THE USER TO APPROVE their muscle mapping. Say
                               they are excluded until approved, and name them.
      – unmapped_sets        : exercises the muscle map has no entry for at all.
      – unattributed_sets    : cardio, which carries no muscle attribution by
                               design. This is NOT a gap; do not call it one.

  • Sub-head attribution (triceps long vs lateral head, upper vs lower chest) is
    genuinely contested in the research. Phrase those as sets on exercises that
    EMPHASISE the head — "66 sets on long-head-emphasis work" — never as if the
    head's involvement were directly measured.

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

A sample-size or reliability statement belongs ONLY to a correlational, pattern,
or trend claim. For a SIMPLE LOOKUP — a most-recent session, a specific record or
PR, a date, a single logged value — do NOT add a "based on N sessions" or
"reliable overview" caveat. Answer the fact directly; it is not a finding whose
confidence needs qualifying.

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

When you report pain, discomfort, or a form issue from the comments, name the
exercise it was logged under. Group occurrences together ONLY when they describe
the same symptom in the same place; never merge distinct complaints — a hip pain
and a knee cramp, or comments from different exercises — into one recurring issue.
If they differ in symptom, location, or exercise, report them separately rather
than presenting them as the same pain.

When you state HOW MANY times a specific symptom or body-part pain occurred, count
ONLY the pain comments tagged with that location (the pain_locations on each
occurrence / the pain_by_location grouping). A pain comment carrying no location tag
is NOT evidence of that specific symptom — do not add it to the total. If you cannot
pin an exact per-symptom count this way, enumerate the dated occurrences or say "at
least N" rather than stating a precise number that lumps different complaints.

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

  READABILITY (markdown):
  • Use markdown to make the answer scannable: "### " section headers when
    the answer covers multiple aspects or parts of a multi-part question,
    **bold** for key figures and for the verdict sentence, and "- " bullet
    lists for enumerations (factors, recommendations, notes).
  • Short answers (one or two sentences) stay plain prose — never force
    headers or bullets onto them.
  • HARD EXEMPTIONS: the pre-formatted display lines from a [DISPLAY] block
    are reproduced character-for-character — NEVER bold, bullet, indent, or
    otherwise decorate those lines (a header line ABOVE a display block is
    fine; the block's own lines are untouchable). Never place markdown
    inside a [[...]] citation tag.

════════════════════════════
COACH CHARACTER
════════════════════════════
You are a direct, warm coach who is opinionated BECAUSE the numbers are
trustworthy. Four principles work together:

  1. DIRECT & WARM — state a clear recommendation plainly, framed supportively,
     not buried under hedges. Prefer "Your Overhead Press has stalled for 6
     sessions — I'd drop the volume 20% for two weeks" over "you might consider
     possibly looking at reducing volume."
  2. ALWAYS EXPLAIN THE WHY — every opinion carries its reasoning and the data
     behind it, so the user can judge whether it applies. "Your volume is down
     three weeks running and you logged wrist pain twice — that's why, not one
     off session."
  3. USER HOLDS THE FINAL CALL — you advise and reason; you do not dictate. You
     know the user through numbers only; you can't see their sleep, mood, or how
     a joint actually feels. State the view, give the reasoning, leave the
     decision to them.
  4. BIAS TOWARD TRAINING, NEVER TOWARD EXCUSES — advise rest or a deload when
     the DATA genuinely supports it, but never volunteer "take today off" as a
     casual option and never validate skipping the data doesn't justify. Default
     posture is "show up." Recovery advice is earned by evidence, not offered as
     an easy out.

GROUNDING (overrides all four when in tension): every strong claim must trace to
the user's data or an established fitness principle. When the data is thin
(small n, few sessions), say so directly — "there isn't enough data to tell you
this confidently" is itself a direct answer, NOT a hedge and NOT a licence to
fabricate confidence. Never be confidently wrong just to sound decisive.
Directness is about training/recovery decisions grounded in data — never blanket
negativity, discouragement, or anything promoting unhealthy restriction.

════════════════════════════
MEDICAL LINE (diagnose vs adapt)
════════════════════════════
NEVER diagnose, name, or treat a medical condition or prescribe medication.
"What spinal injury do I have", "what's causing my knee pain", "how do I treat
my herniated disc" → refuse the diagnostic/treatment part and redirect to a
qualified professional.
ALWAYS allowed (this is your job): training adaptations, exercise substitutions,
form cues, warmups, mobility/flexibility work, and load management AROUND a
stated symptom — WHILE adding a see-a-professional note. "My neck hurts during
chest" → suggest warmups/mobility/form or exercise swaps to ease it, plus "see a
professional if it persists." "Wrist pain on biceps" → grip changes,
substitutions, a deload, plus the redirect. THE LINE: talking about EXERCISES
and TRAINING ADJUSTMENTS = always allowed (with redirect when a symptom is
named); DIAGNOSING or TREATING a condition = refuse + redirect. Never cross into
"here's what's medically wrong with you."
""".strip()


_GROUNDING_SYSTEM = """
You are a fact-checker verifying an AI coach's answer against its source data.

You are given the answer plus EITHER a [CITED VALUES] block (each numeric claim
in the answer is backed by one source value, written "location = value") OR a
full [WORKOUT PACKAGE]. Find every specific numerical claim in the answer and
verify it against whichever source block is present. Edit the answer directly and
return the result as JSON.

  MISQUOTE: the claim's number ≠ its cited value (or the package value) → REMOVE
  or correct it to the source value.

PLACEHOLDER LINES: a line of the form ⟦D0⟧, ⟦D1⟧, … stands in for a verbatim
training-log line that is verified by a separate step. Keep every ⟦D…⟧ line
EXACTLY where it is — never edit, move, merge, or remove one, and never count
its content as a claim.

SET-COUNT SEMANTICS: `working_sets_count` counts WORKING sets only and excludes
warmups. A session's total set count INCLUDING warmups is `total_sets_count`
(per session) / `latest_session_total_sets` (progression). A claim's total-set
number differing from `working_sets_count` is NOT a misquote — verify it
against the total-sets field.

REMOVE if EITHER:
  1. The claim is directly contradicted by a specific value in the source
     (CITED VALUES or the package)
  2. The claim contains a specific number (count, total, measurement, date)
     with no backing value — not among the [CITED VALUES] and (when given a
     package) not found anywhere in it. A fabricated number is removed.

QUALIFY only if the claim makes a causal or comparative assertion
(e.g. "you perform better with X") that rests on fewer than 5 sessions
per condition. Rewrite the sentence naturally — no labels or tags.

PASS (leave unchanged) if the claim states a verifiable fact that
exists in the package. Do not qualify correct numbers.

Default to PASS — only flag genuine problems.

Examples:
  "your plateau is 76 days"
    → cited: exercises|Lat Pulldown|progression.plateau_span_days = 76  → PASS

  "your rest days correlate strongly with performance"
    → cited: …rest_performance_buckets…confidence_label = "weak", …n = 2
    → QUALIFY: "...with only 2 sessions per bucket, this is
      suggestive rather than conclusive"

  "your PR on 2026-05-18 was 140 lbs"
    → cited: exercises|Lat Pulldown|pr.weight = 130.0
    → REMOVE (claim number 140 ≠ cited value 130 — misquote)

  "you typically perform better on Mondays"
    → cited value present but based on n = 1
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
    lines = [
        "[CONVERSATION CONTEXT] — background continuity ONLY. Answer strictly the "
        "current [QUESTION] using the [WORKOUT PACKAGE]. Do NOT comment on, "
        "analyze, or note missing/absent data for any exercise that appears only "
        "here and not in the current QUESTION; a prior logging action is not a "
        "request to analyze that exercise.",
    ]
    for turn in conversation_context:
        lines.append(f"{turn.get('role','?').upper()}: {turn.get('content','')}")
    return "\n".join(lines)


def _fmt_display(package: dict) -> str:
    """
    Approach (b): when the package carries pre-formatted session-display strings
    (`display_sets`), guide the model to reproduce them VERBATIM for display-shaped
    questions and to leave them tag-free (their fidelity is checked separately, and
    grounding ignores them). Empty string when no display_sets are present.
    """
    display = package.get("display_sets") if package else None
    if not display:
        return ""
    listing = "\n".join(display)
    return (
        "[DISPLAY] — the package contains pre-formatted, per-set display strings "
        "for the session(s) in scope (listed below). If the QUESTION is "
        "display-shaped (\"show me\", \"what did I do\", \"my last session\"), "
        "reproduce these strings VERBATIM — character-for-character, including "
        "weights, reps, comments, and any arrows — adding only light framing prose "
        "around them. Show each display block EXACTLY ONCE — never repeat a block "
        "or re-render its lines a second time in your own prose. "
        "If the question is analytical, reason over the data normally. "
        "These display lines are NOT citable: do NOT attach citation tags to them. "
        "Cite only the analytical claims you make about the data.\n"
        f"{listing}"
    )


def _fmt_pr_targets(package: dict) -> str:
    """One-line pointer (6b): if a parameterized PR field is present, tell the model it
    exists and that null means 'no qualifying set'. No behavioral rules."""
    exes = (package or {}).get("exercises") or []
    if not any(("pr_repfloor" in e or "pr_cardio_locked" in e) for e in exes):
        return ""
    return (
        "[PR TARGET] — the question asked for a specific PR. An exercise may carry "
        "`pr_repfloor` (heaviest set for a rep floor) or `pr_cardio_locked` (a locked "
        "cardio PR). Answer from that field when present; if it is null, say there is "
        "no qualifying set at that rep count / lock. The default `pr` is also present."
    )


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
        # Compact serialization — this is LLM input, not human-read; indent=2
        # was ~34% wasted input tokens on a broad package.
        package_json = json.dumps(package, separators=(",", ":"))
    except Exception as e:
        logger.warning("[analysis_agent] package serialisation failed: %s", e)
        package_json = "{}"

    # Citable-field schema — generated FROM this exact package (per-scope correct),
    # so the draft can only cite leaves that actually exist. Prevents the model
    # confabulating plausible-but-nonexistent field names (the Stage-1 YELLOW).
    try:
        citable_schema = build_citable_schema(package)
    except Exception as e:
        logger.warning("[analysis_agent] citable schema build failed: %s", e)
        citable_schema = "(citable schema unavailable)"

    sections = [
        "[WORKOUT PACKAGE]\n" + package_json,
        "[CITABLE SCHEMA] — you may ONLY cite these exact field-path leaves; "
        "never cite a field not listed here:\n" + citable_schema,
        _fmt_research(research),
    ]
    display_block = _fmt_display(package)
    if display_block:
        sections.append(display_block)
    pr_target_block = _fmt_pr_targets(package)
    if pr_target_block:
        sections.append(pr_target_block)
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
    # Collect all non-thinking text parts.
    draft = _collect_text(response)

    if not draft:
        logger.warning("[analysis_agent] empty response from Gemini")
        return "I was unable to generate an analysis. Please try again."

    logger.debug("[analysis_agent] draft: %d chars", len(draft))
    return draft.strip()


# ── Display guard (structural, pure — no LLM) ────────────────────────────────
# Grounding must never edit the package's verbatim display_sets lines, but told
# so in prose it still deleted a warmup line live. Structural guard: excise the
# known lines from the draft BEFORE the grounding LLM sees it (each replaced by
# a sentinel), reinsert verbatim after. The checker cannot edit what it never
# receives. Sentinels use ⟦…⟧ — NOT [[…]] (strip_tags' _ANY_BRACKET_RE would
# eat those) — and survive strip_tags' whitespace tidy. A lost sentinel is
# non-fatal: the downstream display-fidelity stage already repairs missing
# lines, so the worst case equals today's behavior (safe, complete fallback).

def _excise_display(draft: str, display_lines: list) -> tuple[str, dict]:
    """
    Replace the FIRST occurrence of each display line in the draft with a
    sentinel line ⟦D<i>⟧. Longest lines first so a line that is a substring of
    a sibling can't partially match inside it. Lines absent from the draft get
    no sentinel. Returns (excised_draft, {sentinel: verbatim_line}).
    """
    mapping: dict = {}
    if not draft or not display_lines:
        return draft, mapping
    out = draft
    # Stable sentinel numbering by original position; excise longest-first.
    numbered = list(enumerate(display_lines))
    for i, line in sorted(numbered, key=lambda p: -len(p[1])):
        if line and line in out:
            sentinel = f"⟦D{i}⟧"
            out = out.replace(line, sentinel, 1)
            mapping[sentinel] = line
    return out, mapping


def _reinsert_display(text: str, mapping: dict) -> tuple[str, bool]:
    """
    Substitute each sentinel back with its verbatim line. Returns
    (restored_text, all_restored) — all_restored is False when the grounding
    model dropped a sentinel (the display-fidelity stage repairs those lines).
    """
    if not mapping:
        return text, True
    out = text or ""
    all_restored = True
    for sentinel, line in mapping.items():
        if sentinel in out:
            out = out.replace(sentinel, line, 1)
        else:
            all_restored = False
    return out, all_restored


def _build_grounding_prompt(draft: str, grounding_context: dict) -> str:
    """
    Build the grounding user prompt from the Stage-2 grounding context
    (citations.build_grounding_context). Pure — no LLM — so it is unit-testable.

      mode == "cheap": [DRAFT ANSWER] + a small [CITED VALUES] block (one line per
                       claim: location = value (answer stated: N)). A few KB.
      mode == "full":  [DRAFT ANSWER] + the whole [WORKOUT PACKAGE] (today's exact
                       input — the safety fallback when any claim isn't cleanly cited).
    """
    if grounding_context.get("mode") == "cheap":
        lines = [
            f"- {cv['location']} = {cv['value']}   (answer stated: {cv['claim_number']})"
            for cv in grounding_context.get("cited_values", [])
        ]
        cited_block = "\n".join(lines) if lines else "(none)"
        return (
            f"[DRAFT ANSWER]\n{draft}\n\n"
            "[CITED VALUES] — every numeric/factual claim in the answer is backed "
            "by one of these source values (location = value). Verify each claim's "
            "number against its cited value; there is no package this turn.\n"
            f"{cited_block}"
        )

    # Full-package fallback — byte-for-byte today's behavior.
    try:
        package_json = json.dumps(grounding_context.get("package") or {},
                                  separators=(",", ":"))
    except Exception:
        package_json = "{}"
    return (
        f"[DRAFT ANSWER]\n{draft}\n\n"
        f"[WORKOUT PACKAGE]\n{package_json}"
    )


async def ground_check(
    draft:             str,
    grounding_context: dict,
) -> tuple[str, list]:
    """
    Separate Gemini call — NOT the Analysis Agent.
    Verifies every claim in the draft against its cited value (cheap path) or the
    full package (fallback). Edits the answer directly (qualify or remove).
    Returns (cleaned_answer, flagged_claims).

    grounding_context comes from citations.build_grounding_context (Stage 2): the
    small cited-scalar payload when the answer is fully clean, else the whole
    package. Using a separate model instance reduces the tendency to re-affirm
    its own confident wrong statements — the grounding checker has no memory of
    the reasoning that produced the draft.
    """
    if not draft.strip():
        return draft, []

    # Display guard: the grounding LLM never sees the verbatim display lines —
    # they are excised to sentinels here and reinserted after the check.
    guarded_draft, display_map = _excise_display(
        draft, grounding_context.get("display_sets") or [])

    prompt = _build_grounding_prompt(guarded_draft, grounding_context)

    response = await asyncio.to_thread(
        _get_client().models.generate_content,
        model=GROUNDING_MODEL,
        contents=[types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )],
        config=types.GenerateContentConfig(
            system_instruction=_GROUNDING_SYSTEM,
            max_output_tokens=GROUNDING_MAX_OUTPUT_TOKENS,
        ),
    )

    raw = _collect_text(response)

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
        cleaned       = result.get("cleaned_answer", guarded_draft)
        # Reinsert the verbatim display lines behind the sentinels. A lost
        # sentinel is logged and left to the display-fidelity stage's repair.
        cleaned, all_restored = _reinsert_display(cleaned, display_map)
        if not all_restored:
            logger.warning(
                "[analysis_agent] display guard: grounding dropped %d sentinel(s)"
                " — display-fidelity stage will repair",
                sum(1 for s in display_map if s not in (result.get("cleaned_answer") or "")),
            )
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


# ── DISPLAY SETS CHECK (deterministic — NO LLM in the check itself) ─────────────
# Approach (b): grounding deliberately ignores `display_sets` (it is not a citable
# scalar). This separate check owns VERBATIM fidelity of the pre-formatted display
# strings: every one must survive into the final answer as a substring. If one was
# altered/paraphrased/dropped, re-prompt the Analysis Agent ONCE to reproduce them
# verbatim; if it still fails, emit the strings directly from the package (raw
# assembly — no LLM). This is a containment check, not equality: light framing
# prose around the strings is fine.


def _norm_ws(s: str) -> str:
    """Collapse every whitespace run (incl. newlines/indents) to one space — used
    only for containment tests, never to mutate displayed text."""
    return re.sub(r"\s+", " ", s or "").strip()


def display_sets_missing(answer: str, package: dict) -> list:
    """
    The display strings whose CONTENT is not present in the answer. Whitespace-
    tolerant: a line reproduced with different indentation/newlines still counts as
    present, so the fidelity fallback won't re-append a whole block over a spacing
    diff (the live per-set-block duplication). A genuinely omitted line is still
    flagged. The containment test normalizes whitespace only — displayed text is
    never mutated.
    """
    display = (package or {}).get("display_sets") or []
    norm_answer = _norm_ws(answer)
    return [s for s in display if _norm_ws(s) not in norm_answer]


def raw_display_assembly(package: dict) -> str:
    """The no-LLM fallback: the package's display_sets joined plainly."""
    return "\n".join((package or {}).get("display_sets") or [])


def append_missing_display(answer: str, package: dict) -> str:
    """
    Non-destructive fallback: keep the generated answer and APPEND only the display
    entries whose content is still missing — NOT the whole block (that re-pasted
    already-present lines over a spacing diff and duplicated them). Repair, never
    replace.

    Nothing missing (or no display_sets) → answer unchanged. Empty/blank answer →
    the missing block alone (no leading blank line).
    """
    missing = display_sets_missing(answer, package)
    if not missing:
        return answer
    block = "\n".join(missing)
    return f"{answer.rstrip()}\n{block}" if answer and answer.strip() else block


async def enforce_display_fidelity(answer: str, package: dict, reframe) -> str:
    """
    Guarantee every `display_sets` string appears verbatim in the answer.

      reframe: async callable () -> str — re-prompts the Analysis Agent to
               reproduce the display strings verbatim (already tag-stripped by the
               caller). Injected so this is testable without a live model.

    Clean → return as-is. Missing → reframe ONCE; still missing → APPEND the raw
    block to the original answer (repair, never replace — never discard analysis).
    """
    if not (package or {}).get("display_sets"):
        return answer
    if not display_sets_missing(answer, package):
        return answer
    logger.info("[analysis_agent] display-sets check: missing strings — re-framing once")
    reframed = await reframe()
    if not display_sets_missing(reframed, package):
        return reframed
    # Re-frame failed: APPEND the verbatim block to the ORIGINAL answer (not the
    # failed reframe) so the real analysis survives and the exact sets are present.
    logger.info("[analysis_agent] display-sets check: re-frame failed — appending raw block")
    return append_missing_display(answer, package)
