# Changelog — session notes archive

Working notes accumulated while the system was built, moved here verbatim from
`README.md` on 2026-08-21. They are kept for provenance: each entry records what
was changed and, more usefully, *why the first approach was wrong*.

**These notes are historical and are not maintained.** Where a note contradicts
the README, the README is current. In particular several entries below predate
the LangGraph conversion, the muscle ontology, and the decomposition arc, and
describe components as "planned" that have since shipped.

For the transferable version of the same material see [`lessons.md`](../lessons.md).
For current design references see [`routing-rules.md`](routing-rules.md),
[`data_agent_spec.md`](data_agent_spec.md), and [`../ontology/README.md`](../ontology/README.md).

---

## Tech Stack — Multi-Agent Additions (update existing table)

| Component | Technology |
|-----------|-----------|
| Analysis Agent | Gemini 3.1 Flash Lite, thinking_budget=4096, grounding check |
| Coordinator | Gemini 3.1 Flash Lite, temperature=0, routing + coverage check |
| Shared Resolver | src/shared/resolver.py — 5-tier exercise name resolution |
| Cardio Package | Clean per-type data structure, no strength fields for cardio |

---

## Project Structure — add to src/ section

```
src/
├── analysis_agent.py     # Analysis Agent — analyze(), ground_check(), run()
├── coordinator.py        # Coordinator — route(), classify, analytical pipeline
├── shared/
│   ├── __init__.py
│   └── resolver.py       # Shared 5-tier exercise name resolver
```

---

## Multi-Agent System — replace planned sections with verified

**Analysis Agent (`src/analysis_agent.py`) — complete**

Receives `prepare_analysis_package()` output. One Gemini call with
`thinking_budget=4096`. Returns a draft answer. A separate grounding
check call verifies every numerical claim against the package and edits
inline — removes claims directly contradicted by the data, qualifies
correlational claims with small sample sizes. Never requests more data.

System prompt is type-agnostic: the agent analyzes whatever metrics are
present — weights and reps for strength, distance and duration for
cardio, comments for any exercise type. "Every exercise logged was
deliberately tracked and deserves performance analysis."

Session-display is a normal package field, not a separate pipeline:
`prepare_analysis_package()` builds a display block for every resolved exercise
and muscle group in scope and concatenates them into one flat `display_sets`
`list[str]` of pre-formatted per-set lines (built by
`src/data_agent/session_display.py`). No exclusive choice between scopes — a
question that names an exercise and its parent group gets both blocks. The agent
decides display-vs-analyze from the question — there is no routing flag, and
superset display data is safe (analytical questions ignore it). Grounding ignores
`display_sets` (it is not a citable scalar); a separate deterministic DISPLAY SETS
CHECK owns verbatim fidelity (re-prompt once, then raw assembly — no LLM).

**Coordinator (`src/coordinator.py`) — complete**

Single entry point for every user message. One classification call
(temperature=0, thinking_budget=0) extracts route, exercise_names,
muscle_groups, query_period_days, and PR targets — `rep_target` (e.g.
"5-rep PR") and `cardio_lock` (e.g. "fastest 5km"). The coordinator
normalizes lock units deterministically (min→seconds, km→km — the LLM
never emits seconds) and threads them into the package, which then carries
`pr_repfloor` / `pr_cardio_locked` alongside the unchanged default `pr`
(a null value means "no qualifying set", not an error). The rep-floor PR is
deliberately NOT warmup-filtered (unlike the default max-weight PR), so a
wrongly-flagged warmup can never drop a real working-set PR. Routes to:
- Analytical: Data Agent → prepare_analysis_package() → Analysis Agent
  → grounding check → coverage check (1 retry if incomplete)
  → display-sets check (verbatim integrity, only when `display_sets` is present)
- Operational: existing AgentSession.answer() via MCP tools

Exercise names from the classifier are resolved through
`src/shared/resolver.py` before building the package. This handles
case differences, typos, compound words, and plural/singular variants.

**Shared Resolver (`src/shared/resolver.py`) — complete**

Extracted from `mcp_servers/combined_server.py`. Same 5-tier logic:
space-normalised exact → case-insensitive exact → partial LIKE →
plural/singular expansion → fuzzy difflib ≥ 0.75. Used by both the
Coordinator (before building the analytical package) and the MCP server
(single-agent tool calls). One implementation, two callers.

**Cardio exercise data structure — complete**

Cardio exercises (is_cardio=True) receive a clean, purpose-built
structure in `prepare_analysis_package()` with no strength fields:
`sessions` (date, distance_km, duration_seconds), `progression`
(distance and duration trend), `last_session_date`, `days_since_last`,
`all_time_sessions`, `total_sessions_period`, and a cardio-native `pr`
(distance/duration, never weight/reps — default = farthest distance, or
longest duration for duration-only exercises like Cycling). Mirroring strength's
dual field, `pr` is **all-time** (computed over the all-time session list threaded
from `process_data`) while `pr_period` is the within-window stat; a parameterized
`pr_cardio_locked` ("fastest 5km") is likewise all-time. Every cardio PR carries
its session comment — including an all-time PR whose date falls outside the query
period (the comment enrichment sources from the all-time session superset).
Session display renders cardio as "{distance} km in {duration}s ({pace} min/km)"
(or "{duration}s" for duration-only), never the strength weight×reps template.
Strength-specific
fields (max_working_weight, reps_at_max, estimated_1rm, volume, etc.) are
stripped entirely. The exercise is also removed from
`exercise_lifecycle` so the Analysis Agent cannot anchor on all-time
summary counts instead of period-specific progression data.

---

## What Each Component Built — add rows

| Stage / Component | What Was Built |
|---|---|
| **multi-agent** | **Analysis Agent — grounding check, type-agnostic system prompt, thinking_budget=4096** |
| **multi-agent** | **Coordinator — routing, coverage check, shared resolver integration** |
| **multi-agent** | **shared/resolver.py — 5-tier name resolver extracted from MCP server** |
| **multi-agent** | **Cardio data structure — clean per-type package, no strength fields for cardio exercises** |

---

## Key Lessons — add

**The name resolver should be the first thing wired into any new data pipeline.**
A single case mismatch ("walking" vs "Walking") caused the classifier to pass an
unmatched exercise name to the Data Agent. The filter silently fell back to all
67 exercises, the package was built for the wrong scope, and the Analysis Agent
produced wrong answers. Hours of debugging the Analysis Agent's "understanding"
of cardio data. The resolver already existed and solved the problem in one step.
Wire it early, not as an afterthought.

**Test the component in isolation before debugging the pipeline.**
Running `chat_analysis_agent.py` — a direct call to the Analysis Agent with the
correct package — answered the Walking question perfectly in one shot. The bug
was never in the Analysis Agent. Direct isolation testing reveals this in seconds;
pipeline debugging can chase it for hours.

**Training priors fill gaps in the data.**
When the Analysis Agent couldn't find a total session count for Walking (the field
didn't exist in the package), it hallucinated a plausible number (78) from its
training data about fitness apps. Adding `all_time_sessions: 78` to the package
immediately stopped the hallucination. The model wasn't broken — it was filling
in a field it expected to exist. The fix is to give it the data, not to suppress
the output.

**Type-agnostic analysis requires explicit instruction.**
The Analysis Agent defaulted to treating Walking as a "lifestyle activity" with
no performance metrics. Adding "every exercise logged was deliberately tracked —
the act of logging is the signal that it matters" to the system prompt fixed this.
Models have strong priors about what counts as "real" training data. Override them
explicitly or the model will silently ignore valid exercise data.

**Grounding checks can hallucinate their own verification sources.**
When the Analysis Agent fabricated "78 sessions," the grounding check passed it
with "Verified against exercise_lifecycle.active for Walking" — even after Walking
had been removed from exercise_lifecycle. The grounding checker invented a source
to justify keeping a plausible-sounding number. Grounding checks catch direct
contradictions reliably; they do not reliably catch hallucinated values that happen
to be plausible.

**Two different output structures for two different exercise types.**
Cardio exercises and strength exercises produce fundamentally different data.
Putting both through the same package structure (strength fields set to zero for
cardio) confused the Analysis Agent because fitness-app training data associates
all-zero strength fields with "empty" or "failed" records. A clean, purpose-built
structure for each type eliminates the ambiguity.

---

# README Additions — Shared Modules & Custom SQL (Session 7, Part 2)

Append to the relevant sections of README.md.

---

## Project Structure — add to src/shared/

```
src/shared/
├── __init__.py
├── resolver.py        # 5-tier exercise name resolution
├── memory.py          # Read-only ChromaDB fact retrieval
├── rag.py             # Fitness knowledge search with source labeling
└── sql_executor.py    # Safe read-only SQL execution
```

---

## Shared Modules — complete

**src/shared/memory.py**
`retrieve_relevant_memories(question) -> list[str]`. Read-only ChromaDB
retrieval — memory writes stay on the single agent exclusively. Embeds the
question against the user_memory collection (BAAI/bge-small-en-v1.5),
returns up to 5 facts with cosine distance < 0.8. Returns [] on any error,
never blocks the pipeline. Wired into the Coordinator's analytical path.

**src/shared/rag.py**
`search_fitness_knowledge(question) -> list | None`. Query rewriting →
hybrid search (ChromaDB + BM25) → cross-encoder reranking → three-tier
source labeling (user_article / pubmed-wikipedia / none). Returns results
in the format the Analysis Agent's _fmt_research expects. Used for analytical
questions where research context supports analysis of personal data;
standalone research questions still route operational to the single agent's
search_fitness_knowledge MCP tool.

**src/shared/sql_executor.py**
`run_query(sql, db_path) -> list[dict]`. Sanitization (curly quotes, em-dash),
SELECT/WITH-only guard (rejects all write statements), LIMIT 10000 injection
if absent, 30-second timeout via connection timer + interrupt. Errors
propagate to caller. data_agent.query() was refactored to use this executor
for execution, keeping the typed_weight conversion layer on top.

---

## Custom SQL Pipeline (Option B) — complete

The Coordinator generates one supplementary SQL query for cross-cutting
questions the per-exercise package cannot answer: gaps between sessions,
day-of-week patterns, same-day exercise combinations, total counts and
aggregates across all exercises.

Flow: _classify() returns needs_custom_sql and custom_sql_intent →
_generate_custom_sql() builds SQL via src/llm.generate_sql() with the
schema from src/schema_prompt.build_schema_prompt() → runs via
data_agent.query() (which uses shared/sql_executor) → result passed to the
Analysis Agent as a labeled supplementary query result.

Custom SQL is scoped to counts, dates, gaps, and patterns — never individual
set weights, which the standard package already polishes (offsets, bar
weights, units). Aggregate weights receive typed_weight conversion (kg→lbs);
the result is labeled so the Analysis Agent treats those as typed values, not
display values.

---

## Schema — Distance Column Correction

The training_log.distance column is stored in KILOMETERS (REAL), not meters.
The schema description previously documented it as integer meters. This was
never caught because the data_agent package reads distance directly (correct)
and no other code queried the column until custom SQL. The schema now
correctly states distance is in km — SUM(distance) gives kilometers directly,
never divide by 1000.

---

## Temporal Interpretation in SQL Generation

build_schema_prompt() now injects a DATE CONTEXT block with today's date and
the latest workout entry in the database, plus interpretation rules:
- "this year" → current calendar year (date >= 'YYYY-01-01')
- "over the past year" / "in a year" → rolling 12 months back from the latest
  DB entry (not from today, since the user's data may not be current)
- "lately" / "recently" → 30 days back from the latest entry

Relative ranges anchor to the latest workout entry because the user's data may
not extend to today.
---

# Session 9 — Full-Project Audit: Fixes & Hardening

A line-by-line audit of the multi-agent pipeline (coordinator → data agent →
analysis agent → grounding → coverage) against the documented behaviour.
Seven fixes landed; all 73 tests pass.

## Fixed

**Demographics feature status.** Where the memory/demographics work stands:
- **Done** — memory-extraction wired into the analytical path (coaching turns now
  reach extraction, stripped answer only); **Stage A** storage layer (3-tier model,
  anchors-not-computed-values, `set_demographic`/`get_demographic`/`get_derived`,
  sensitive-data policy, demographics out of ChromaDB); **Stage B** follow-up asking
  (mention-triggered, answer-first one-line aside, parse→write, drop-if-unanswered,
  clarify-once). See the entries below for each.
- **Deferred — Stage C** (RAG personalization gate, Option B) and **Stage D**
  (volatile-stat history). *Why:* the RAG path has no personalization step at all —
  stored demographics never reach the answer context (the prompt injects facts, not
  demographics), there's no notion of a paper's required population, the gate's two
  halves straddle the operational(RAG)/coordinator(follow-up) boundary, and the
  gate's canonical stat (bodyfat %) is tier-3 never-store. Building the gate now =
  building the whole personalization feature on nothing. Deferred until RAG
  personalization exists.
- **Resume sequence** (upcoming, when RAG is built out): (1) inject demographics
  into the operational answer context — the single hard blocker; (2) paper-requirement
  extraction (answer-time LLM vs ingestion-time pre-tagging); (3) boundary bridge —
  agent emits a "missing anchor K" signal the coordinator reads to arm Stage B's
  follow-up; (4) a tier-3 ask-fresh path if bodyfat/bodyweight gating is in scope.

**Demographics follow-up asking (Stage B of 3).** The conversational layer that
collects the anchors Stage A stores. When the user mentions a demographic in
passing and that anchor isn't already stored — "for a 22-year-old…", "I've been
training 3 years", "I'm 5'9" — the agent **answers the question fully first**,
then appends **one** gentle, new-line aside offering to remember the *precise*
anchor ("…if you tell me your birthday I can factor your exact age in going
forward"). It does **not** scrape the ambiguous mentioned value (a passing "22");
it asks for the exact anchor. On the user's **immediate next reply** the answer is
parsed deterministically (no LLM) and written via `set_demographic` ("2003-06-18",
"male", "176cm", a bare "2021" for a start-date → Jan 1) with a brief ack; a
malformed attempt gets **one** clarification, then it gives up — no looping. If
the next message isn't an answer (the user moved on), the follow-up is **dropped**
— and because it's appended *after* the clean answer was recorded into history, an
unanswered aside never accretes in the memory-extraction/grounding history. The
follow-up is rule-based, conservative (a bare number from reps/weight never
fires), one-at-a-time, and lives exactly one turn. (Stage C — the point-of-use
RAG/answer gate that asks when a personalized answer *needs* a missing anchor —
and Stage D — volatile-stat history — are upcoming.)

**Demographics storage layer (Stage A of 3).** Foundation for personalizing
coaching with user demographics. **Anchors only** are stored — birthdate (not
age), training-start-date (not years-trained), sex, height — and derived values
(age, years-trained) are computed **fresh at point-of-use**, never cached: the
input is today's date, so a stored derived value would go stale daily for zero
perf gain (same "one source of truth, compute fresh" discipline as `src/units.py`).
A **three-tier model** (the single source of truth, `src/demographics.py`) governs
storage and use: **T1 use-freely** (sex; birthdate→age; training-start→years-trained —
stable or accurately derivable), **T2 confirm-on-use** (height — monotonic, the
stored value is a safe floor; a later stage confirms "changed since?" before
applying it), **T3 never-stored** (bodyfat %, bodyweight — volatile, asked fresh
at use). `memory.set_demographic(key, value)` validates the value (real date,
accepted sex, positive height+unit) and **refuses tier-3** writes; `get_derived`
computes age/years-trained fresh and never persists them. A lightweight
**sensitive-data storage policy** (new — none existed): only T1/T2 keys store, as
a structured `demographics` map (not free-text facts, **not** embedded in
ChromaDB); tier-3 never stored; the anchor stored, never the identity-derived
value. (Stage B — follow-up asking that writes these — and Stage C — the
RAG/answer gate that reads them — are upcoming.)

**Memory extraction wired into the analytical path.** End-of-session memory
extraction (`_auto_extract_memories`) scans `AgentSession._conversation_history`,
which was populated only by the operational `answer()` loop. The Coordinator's
**analytical** path runs the analysis pipeline directly and never appended to it,
so analytical/coaching Q&A was invisible to extraction — the coordinator docstring
marked this "NOT YET WIRED." Now `_run_analytical`, at its final return, records
the `(question, final stripped answer)` via a new
`AgentSession.record_external_exchange` — the same exchange shape operational
turns use, so extraction treats them identically. It records the **stripped**
user-facing answer only (never the tagged draft, the cited-values payload, or the
package); `out_of_scope` / filler / parse-failure turns short-circuit before
`_run_analytical` and record nothing. (Scope note: this closes the *path* gap —
the foundation for capturing durable user facts from coaching questions. The
end-of-session extraction *timing* model, e.g. an abrupt terminal kill skipping
`close()`, is a separate known item deferred to the web-deployment rework.)

**Dead-code cleanup (post-#7 audit).** A read-only audit after the #7 arc found
code with zero callers, removed in two passes (full suite green throughout, 367):
- `analysis_agent.run()` — an unused `analyze→ground_check` convenience wrapper
  (zero callers, grep-proven); the module docstring now correctly says "Two
  functions". The `CONFABULATED` test constant — orphaned when Stage 1.7 renamed
  its only test — was removed too.
- The three Step-C-unexposed read functions whose "kept for reuse/evals"
  justification never materialized: `_get_personal_record_sync`,
  `_query_workout_data_sync`, and `_run_read_only_sql` (each with its async
  wrapper and its now-unreachable `call_tool` dispatch branch). A grep confirmed
  **no analytical/eval caller** — they were pinned only by an existence-asserting
  test and the dead dispatch. Of the four Step-C-unexposed reads, **only
  `_get_weekly_volume_sync` was genuinely reused** (`tests/test_operational_volume.py`
  calls it directly) and is kept; the existence test was retargeted to assert the
  three stay removed. `list_tools` is unchanged (these were already unexposed).
  A follow-up pass then removed the two helpers left transitively orphaned —
  `combined_server`'s local `_get_bar_weight` and `ALLOWED_TABLES` (their only
  callers were the removed functions); the shared `process._get_bar_weight_lbs`
  is distinct and was untouched.

**Broad-question latency — citation layer (#7, DONE; Stages 1 → 2 complete).** A
broad analytical answer takes ~133s, dominated by the **grounding** stage
re-sending the full ~415 KB / ~106K-token package a second time (the draft
already sent it once) to fact-check the draft's numbers — a job that only needs
the *cited* values, not the whole package. The fix is two-staged:

- **Stage 1 (done):** the Analysis Agent's draft now emits an internal citation
  tag after every factual claim — `[[collection|match-key|field-path]]` (numeric →
  the value leaf; hedge → the `n`/`session_count`/`confidence_label` leaf;
  absence → the `ABSENT` marker). A deterministic, no-LLM citation layer
  (`src/citations.py`) parses, indexes (the package's `exercises` /
  `muscle_group_summary` are *lists*, so it builds name→row maps), resolves each
  tag against the package, and **strips** the tags so the user never sees one.
  The cleaned (tag-free) prose is what flows to grounding, the checkpoint, and the
  user. **Grounding is UNCHANGED this stage** — it still receives the full package
  + the stripped draft (identical prose shape to before), so it remains the safety
  net while the citation machinery is proven on every real turn (resolution health
  is logged; a bad match-key or a false `ABSENT` claim flags loud, never silently).
  **No user-visible change and no grounding behavior change.**
- **Stage 1.5 (done) — tag reliability:** a live re-check showed the draft model
  reliably emits a tag on nearly every claim and never drifts names, but ~27% of
  tags (≈50% on broad/ranking questions) resolved to nothing because it was taught
  the tag *format* but not the package's real field *schema* — so it invented
  plausible-but-nonexistent leaf names (`highest_volume`, `best_e1rm`,
  `pct_of_lbs_total`). Fix: `citations.build_citable_schema(package)` generates a
  compact (~6 KB) whitelist of the **real** citable leaves **from the live
  package** (always in sync — no hand-maintained list that could drift), injected
  per-question into the draft prompt; the model may cite only listed leaves. Each
  scope gets its own correct menu automatically (broad-dropped fields aren't
  listed). Also tightened: `ABSENT` is only for a never-logged entity (not "no
  derived stat" — that's a hedge citing a count/confidence leaf), and per-leaf tag
  granularity (`pr.weight`, not `pr`; each number its own tag).
- **Stage 1.6 (done) — usable, not just resolvable:** the post-1.5 re-check
  showed `NOT_FOUND` down to 3.7%, but the ranking/strongest-weakest class cited
  ranked-aggregate *lists* (`rankings|-|highest_volume`,
  `muscle_group_balance|-|distribution`) that resolved "OK" while giving grounding
  **no scalar to check** — hiding the problem. Two fixes: (1) the draft prompt now
  steers ranking/superlative claims to the **per-entity scalar leaf** that holds
  the number (`exercises|<name>|period_volume_lbs`, `pr.estimated_1rm`,
  `progression.plateau_span_days`) instead of a `rankings`/`exercise_lifecycle`/
  `distribution` aggregate; (2) `resolve_tag` now distinguishes a **scalar** leaf
  (`OK`) from a **list/dict** (`OK_NONSCALAR`), so a resolves-to-list no longer
  counts as clean — it's added to `FLAG_STATUSES` and counted separately in the
  health log. `OK_NONSCALAR` (alongside `NOT_FOUND`/`UNKNOWN_COLLECTION`) is the
  "not cleanly cited" signal Stage 2's graceful fallback will trigger on.
- **Stage 1.7 (done) — flat-address the ranking sections at the source:** the
  ranking class still resolved `OK_NONSCALAR` because `rankings`,
  `exercise_lifecycle`, and `muscle_group_balance.distribution` are dict-of-lists
  with no flat scalar leaf. `build_index` now builds an **entity view** for each —
  inverting them into `{entity: {leaf: scalar}}` — so a claim cites a scalar
  directly: `[[rankings|<name>|highest_volume_lbs]]` → `234340.0`,
  `[[muscle_group_balance|<group>|pct_of_lbs_total]]` → `39.5`,
  `[[exercise_lifecycle|<name>|total_sessions]]` → a count. The original section
  dict is kept as `dict_row`, so the old `|-|` forms still resolve exactly as
  before (backward-compatible). `build_citable_schema` lists the new flat leaves
  (schema ~6.9 KB), and the steer points at them. Accepted trade-off (locked with
  the user): a ranking number is now citable two ways (entity-view leaf *and* the
  per-exercise leaf) — a duplicated-but-correct citation path beats an unreliable
  section-fallback.
- **Stage 2 (DONE) — the latency win.** The post-1.7 live re-check came back
  GREEN (0 `OK_NONSCALAR`, 0 `NOT_FOUND`, 7/7 answers 100% clean), so grounding's
  input was switched from the whole package to the **extracted cited-scalar
  payload**. `citations.build_grounding_context(cited, package)` does a **binary
  per-answer split**: if every claim is cleanly cited (`OK` scalar / `ABSENT_OK`)
  → grounding gets the draft + a small `[CITED VALUES]` block (the cheap path —
  measured **0.6–~8 KB** vs the ~415 KB / ~106K-token package, **~50–600× smaller**,
  killing the second full-package send and the per-minute-429 it triggered); if
  ANY claim isn't cleanly cited (or the draft has no tags, e.g. a resumed stripped
  draft) → it falls back to the **full package** — byte-for-byte today's grounding.
  `ground_check` verifies each claim against its cited value (misquote = claim
  number ≠ cited scalar; fabrication = a number with no backing value) on the
  cheap path, and uses the unchanged REMOVE/QUALIFY/PASS package check on the
  fallback; same `{cleaned_answer, flagged_claims}` output. The fallback is the
  safety path and fires ~never, so it's deliberately the **complete** check, not
  an optimized subset (no subset-trap). The coordinator logs which path each turn
  took. **The draft (analyze) still gets the full package — only grounding's input
  shrank. #7 is DONE (all stages).**

**Hygiene sweep (#9–#13).** Five independent low-risk cleanups:

- **#9 — one canonical SQL sanitizer (`src/shared/sql_sanitize.py`), em-dash bug
  fixed.** Three copies had drifted (`src.db.sanitize_sql`,
  `sql_executor._sanitize_sql`, `fetch.sanitize_sql`); two of them converted an
  em-dash `—` into `--`, a SQL **line comment** that silently truncated the rest
  of the query (including any injected `LIMIT`). All three now delegate to one
  canonical sanitizer that does the union of the legitimate behaviors (curly/smart
  quotes `' ' ‚ ‛ " "` → straight) and replaces `—` with a **space**, never `--`.
  Only the *text normalization* is shared; the SELECT/WITH guard, LIMIT/row-cap
  injection, and the analytical weight-aggregate fence stay intentionally
  separate (they're policy, not text cleanup).
- **#10 — dead code removed.** Deleted `src/answer.py`, `src/router.py`,
  `mcp_servers/fitnotes_server.py`, `mcp_servers/knowledge_server.py`, and
  `scripts/test_stage2.py`. Each was confirmed unreferenced (the two old servers
  were replaced by `combined_server.py`; the only importers of `answer`/`router`
  were the dead files themselves — dead-importing-dead). `combined_server.py` is
  the only server `agent.py` launches.
- **#11 — duplicate knowledge-base prompt block merged.** The operational
  `SYSTEM_PROMPT` had two near-identical headers ("USER KNOWLEDGE BASE" listing
  only `list_user_articles`, and "KNOWLEDGE BASE" listing both
  `list_user_articles` + `delete_user_article`). Merged to the single complete
  block; verified against the live 31-tool exposed list that both named tools
  exist and are exposed.
- **#12 — exercise-count doc drift corrected.** Prose said "all 51 / 50
  exercises"; the live DB has **67** distinct exercises with logged sets (136
  total exercise rows). Corrected the four trained/analyzed-exercise instances
  (README ×2, lessons.md ×2) to 67.
- **#13 — `resolve_exercise_name` input guards.** Empty / whitespace-only /
  too-short (< 2 chars after strip) input now returns a clean no-match
  (`{match: None, candidates: []}`) **before** any broad match, instead of `""`
  surviving to a `LIKE '%%'` that dumped 8 arbitrary candidates ("multiple
  exercises matching \*\*\*\*"). LIKE metacharacters `%` and `_` in a term are
  escaped (`ESCAPE '\'`) so a literal term matches literally.

**Routing / server-response cluster (#2, #3+#8, #5).** Four post-Step-C
regressions in the routing front door and the `/chat` response path, fixed
together:

- **#2 — `/chat` surfaces the coordinator's graceful answer, not the raw error.**
  The handler discarded the friendly text the Coordinator had already built (for
  both the operational pipeline-failure fallback and the `DataAgentIntegrityError`
  case) and returned `result["error"]` verbatim — leaking internal invariant IDs
  like `B3: …` to the user. `/chat` now mirrors `cli.py`: it returns the answer
  whenever one exists, logs the real error server-side, and only falls back to a
  generic "Something went wrong" when there is genuinely no answer. Relatedly,
  `_reinitialize_session` now **logs** an unexpected `close()` failure instead of
  `except Exception: pass` (close shouldn't raise after the owner-task lifecycle
  fix, so a raise is worth surfacing — without breaking the reload).
- **#3 + #8 — the write-intent pre-guard reworked, both directions.** The old
  noun-list regex both over- and under-fired post-consolidation. It now (a) does
  **not** fire on coaching QUESTIONS about writing — "should I add weight to my
  squat", "can I add a set", "is it ok to remove a set", "when should I update my
  goal" — which were being force-routed into the now-impoverished operational
  agent and stranding as non-answers (#3); and (b) **does** fire on bare
  imperative writes the old regex missed — "log my bench 100x5", "record squat
  80kg x5", "add 3 sets of deadlift" — via weight×reps / sets×reps / unit
  shorthand recognition (#8). **Precedence:** question/modal phrasing always
  wins — when a message is ambiguous between "question about writing" and
  "command to write", the guard does **not** fire and lets the classifier decide.
  A missed write is still caught by the classifier and blocked by the operational
  confirmation gate (recoverable); a false-positive strands a coaching question
  (no recourse).
- **#5 — trivial input no longer hits the expensive analytical pipeline.** Two
  cheap short-circuits run in `route()` before classification: (a) a **filler**
  guard returns a canned reply for bare greetings/acks/thanks ("hi", "ok",
  "thanks", "cool", …) with no classify call, no package build, and no pipeline;
  (b) an **unparseable-classify** guard returns "I didn't catch that — could you
  rephrase?" instead of building a ~358 KB package and running analyze+ground on
  garbage (a bare "ok" would otherwise have the classifier invent
  `muscle_groups=['Legs']`). The Step-C "parsed-but-uncertain → analytical"
  default is unchanged; only the unparseable/errored case is short-circuited.

**Custom SQL execution is now actually read-only.**
The docs (and a commit message) claimed `data_agent.query()` had been
refactored onto `shared/sql_executor`. The code never was — LLM-generated SQL
ran on a read-write connection behind a space-delimited keyword blacklist
that `WITH c AS (SELECT 1)INSERT INTO ...` walks straight past. `query()`
now executes through `shared/sql_executor.run_query`: `mode=ro` URI
connection (writes fail at the database level no matter what the text guard
misses), SELECT/WITH-only guard, `LIMIT 10000` injection, 30-second
interrupt timeout. The `typed_weight` conversion layer is unchanged.
`fetch.py`'s own connection is read-only too — a plain `sqlite3.connect()`
also silently *creates* an empty DB file when the path is wrong; `mode=ro`
makes misconfiguration loud.

**Cross-unit comparisons in progression are fully normalized.**
`_compute_progression` normalized only the start/end pair. The plateau loop,
`sessions_at_max`, and peak/regression detection still compared raw numbers
across the Deadlift lbs→kg switch. All comparisons now go through per-session
kg normalization; reported values stay in the end session's unit. Same
results for current data (the latent bug never triggered) — correct for any
future data where it would have.

**Phase 2 triggers on regressions, not just improvements.**
`weight_change_pct > 20` only fired on improvements. A 20 % *drop* is
exactly when the comment history matters most (injury, deload, technique
rebuild). Now `abs(weight_change_pct) > 20`.

**Unresolvable exercise names are reported, not silently dropped.**
When the Coordinator resolved ["Bench Press", "Foobar"] and Foobar matched
nothing, it was removed from the filter list — the answer covered Bench with
no mention that Foobar doesn't exist. Unresolvable names now flow through to
the package, come back as `unresolved_exercise_names`, and the answer is
prefixed with what was and wasn't found.

**Grounding check no longer silently skips long answers.**
The grounding call must return the entire cleaned answer inside a JSON
envelope, but shared the draft's 2048-token output ceiling — any near-limit
draft truncated the JSON, failed parsing, and returned ungrounded. Grounding
now has a 4096-token ceiling. Response parsing across the agent also collects
*all* non-thinking text parts instead of keeping only the last part.

**`/reload-db` set a dead local variable.**
`agent_ready = False` without a `global` declaration — the flag never
changed and `/chat` kept serving against a session about to be torn down.

**`cli.py` imported `groq` — a package not in requirements.txt.**
Leftover from the eliminated Groq stack; fresh installs crashed at import.
Removed. Rate limits are handled by the existing generic 429 path.

## Improved

- Classifier sees the previous turn — follow-ups like "what about my squat?"
  were previously classified with no context.
- Classifier muscle-group list now includes Abs and Cardio (both are valid
  categories the Data Agent already filters on; the prompt omitted them).
- Rate limits raised inside custom-SQL generation now propagate to the
  countdown UX instead of being swallowed.
- Stale "stubs in place" comments removed from the Coordinator — the shared
  modules have been wired since Session 7.

---

## Session 9 additions — Data Agent hardening, WAL, and Coordinator wiring

### Data Agent hardening — complete

The Data Agent is now a four-module package (`fetch` / `process` / `validate` / `__init__`). The validator runs in raising mode: any integrity violation raises `DataAgentIntegrityError`, which the Coordinator catches before passing data to the Analysis Agent. Soft violations (G4 size ceiling, G5 scope leaks) are logged as warnings. 73 tests pass across the golden and per-invariant suites. A `@pytest.mark.xfail(strict=True)` strategy kept the suite green while known violations were open — a landed fix flips the marker to a loud failure automatically.

### Scope-aware packaging — BROAD 365d = 396 KB

`prepare_analysis_package()` derives scope from query filters and trims accordingly:

| Scope | Condition | Retained |
|-------|-----------|---------|
| FOCUSED | ≤ 3 named exercises | Full detail — sessions, full_comments, all stat blocks, all aggregation levels |
| GROUP | muscle-group filter | full_comments capped 30 most recent + all pain-flagged; one aggregation level |
| BROAD | no filter | full_comments removed; deep-stat blocks removed; one aggregation level |

BROAD 365-day package: 1 443 KB → 396 KB (72 % reduction). This resolved the free-tier 429 failures on general questions ("how has my training been?").

### Coordinator wired into cli.py and server.py

Every user message now routes through the Coordinator. A write-intent regex guard routes operational questions (logging, corrections, goal-setting) to the single-agent path without an LLM classification call. `/confirm` bypasses routing — it is the continuation of an in-flight staged write whose context already lives in the agent session.

**`/log` — deterministic write boundary.** A message starting with `/log` (case-insensitive) is a TRUSTED write boundary: the prefix is stripped and the rest routes operational directly — no regex guessing, no classify call. The write-intent regex above stays as the graceful-degradation fallback for un-prefixed logging messages; on that fallback path only, the answer carries a one-line suggestion to use `/log`. A `/log` message may end in an unrelated analytical question ("… Also, how is my back progressing") — only the workout is handled; the tail is acknowledged with a note and never routed. If a `/log` turn ends pending a logging clarification (date, name disambiguation, ambiguous sets/reps), the immediately-next reply joins the `/log` flow without the prefix (single-turn carry; a question or filler reply abandons it). The frontend input shows a `/log` autocomplete when a message starts with `/`.

**Verify-at-staging (write-verification stage 2).** After a workout batch is staged and before the confirmation panel is shown (web) / before execute (CLI), one LLM diff call verifies that the staged slot faithfully represents the user's request. Input A is the assembled workout portion of the `/log` flow — the stripped turn-1 request plus any carry-turn replies, in order — so a multi-turn flow (where the agent re-stages the batch from history) is checked whole. Input B is the raw staged slot JSON plus its deterministic slot-rendered preview (the raw slot alone stores opaque exercise ids and metric-kg weights, so the rendering supplies names and typed values). The call is a diff, not a re-extraction: the model only checks whether B matches A, flagging missing / extra / misattributed sets or comments. On PASS the panel proceeds and the verdict is stored alongside the batch (in the staged-writes dict, cleared with it on discard). On FAIL the panel is suppressed, the staged batch is discarded immediately, and the coach asks the user to re-state the workout — a stray confirm afterwards finds no pending kind and an empty slot, so nothing can reach execute. If the verify call itself errors (quota, unparseable output, missing preview), it fails open: the panel — itself a deterministic rendering of the slot for human review — is shown, with the not-verified verdict recorded.

**Write-path checkpoint-and-restore.** The write path checkpoints at both of its LLM boundaries, so a quota-interrupted /log turn resumes by RESTORING the exact staged batch — never by letting the agent re-stage it from conversation history (a fresh probabilistic extraction that can diverge from what was already verified). A 429 escaping the staging loop enriches the interruption checkpoint with the staged slot and the assembled /log-flow turns; a verify PASS checkpoints the verified batch (slot + turns + verdict) before the confirmation panel is shown. On "continue", the coordinator restores the slot into the MCP subprocess — every exercise id re-validated against the current database first, since a backup upload may have replaced it (validation failure asks for a re-log rather than confirming dangling references) — and the server/CLI arms its confirm gate directly, with no agent call: a checkpointed PASS verdict skips the verify entirely, a missing verdict runs the verify now on the restored slot with the restored turns as input. The checkpoint lives until the confirmation resolves — execute-success, cancel, or a verify FAIL clears it (guarded so an unrelated interrupted question's checkpoint is never touched) — so an abandoned panel or a server restart stays resumable while a committed batch can never be restored twice.

### Write-Ahead Log

Every confirmed `execute_*` DB write is journaled to `data/agent_writes.json` (gitignored). On FitNotes backup upload, the server replays the WAL onto the new database before reinitializing the agent — agent-written sets survive backup uploads.

The upload+replay race is closed in both directions:
- **New requests** — `/chat` and `/confirm` return a clean 503 while `_upload_lock` is held, rather than silently racing the file swap.
- **In-flight turns** — `/upload` acquires `agent_lock` (inside `_upload_lock`) before replacing the DB file, so an ongoing MCP subprocess write drains before the file is touched.

The frontend chatbox is disabled the moment a file is selected for upload and stays locked until `/status` reports ready.

### Correctness invariants (validator) — A–G + G6

The validator asserts deterministic post-conditions on the finished package and
**raises** `DataAgentIntegrityError` for the integrity class (A/B/C/D/E/G + G6),
so a wrong number can never silently reach the Analysis Agent. Each invariant
names the bug it catches:

| ID | Invariant | Bug it catches |
|----|-----------|----------------|
| A1 | `unit` is exactly `"kg"` or `"lbs"`, never null/other | silent null/garbage unit |
| A2 | `kg` only if in `exercises_in_kg` (Deadlift on/after 2025-12-26) or a comment-override | wrong-unit headline |
| A3 | no weight appears without a unit label travelling with it | unlabeled number misquoted |
| A4 | one unit per exercise, except Deadlift's split + comment-override sets | Hand Gripper "Pounds" mislabel (M3) |
| B1 | plate weight = `metric_weight*2.2046 + offset`, 1 dp | non-deterministic weight |
| B1a | headline = plates + bar (exercise's unit); bar in **both** headline and volume; non-barbell bar = 0 | Deadlift PR showing 65 kg not 85 |
| B2 | PRs from full set history (highest headline → most reps → most recent), never the app `is_personal_record` flag | flag dependence in PR selection |
| B2a | PR carries its set's comment | PR missing its context |
| B3 | `pr.weight` ≥ `pr_period.weight` ≥ every in-period `max_working_weight` | PR below a session max |
| B4 | `pr.weight > 0` for weight-based; `pr is None` only for non-weight | weight exercise with no PR |
| B5 | every `max_working_weight` equals an actual set headline in that session | invented session max |
| B6 | no negative weight/reps/distance/duration | sign/parse errors |
| B7 | cross-unit comparisons are kg-normalized (Deadlift lbs→kg), never raw display | 150 lbs ranked above 70 kg |
| B8 | `total_prs_alltime`/`prs_per_month_alltime`/`is_pr_session`/`pr_velocity`/`pr_count`/`pr_context`/`learning_curve` recomputed from PR-EVENT dates (`_pr_event_dates`: running-max walk over working sets — new max weight OR more reps at the top weight — same kg-normalized headline basis as the all-time PR), never the `is_personal_record` flag. The flag under-counts (it misses same-weight-more-reps beats: real-DB 155 flagged vs 559 true events). Excluded only for CARDIO (own distance/duration PR object), by `category == "Cardio"` — NOT a weight=0 proxy, which would wrongly drop bodyweight reps-progression | flag leakage (flag fetch/bundle removal is sub-stage 2) |
| C1 | cardio `distance>0` ⇒ `distance_progression` non-null and > 0 | H3 single-session distance null |
| C2 | cardio `duration_seconds>0` ⇒ duration progression populated | duration dropped |
| C3 | `all_time_sessions` non-null when ≥1 all-time session | H2 `total_sessions_alltime` key typo |
| C4 | cardio `sessions` non-empty when `total_sessions_period>0` | H3 >90-day cardio wiping sessions |
| C5 | a period comment survives into the cardio block + sets pain flag | H1 `ex.clear()` destroying "kidney started paining" |
| C6 | `pace` only on sessions with `distance>0` (none for Cycling/Dead Hang) | H4 divide-by-zero / fabricated pace |
| D1 | `comment_count` == sets with a non-null comment (from the `LEFT JOIN Comment`) | approximate comment matching |
| D2 | `has_pain_flag` iff a set comment matches the pain vocabulary | missed pain note |
| D3 | every unit/bar/warmup token is applied **or** logged unclassified | "next time use black rod" silently corrupting the bar |
| D4 | every `full_comments` entry has a real date + text from a real row | fabricated comment |
| E1 | every session date within `[query_start_date, query_end_date]` | row outside window |
| E2 | no session dated after the end anchor | future data |
| E3 | weekly/monthly/yearly volume reconciles to member-session sums (per unit frame) | aggregation not summing to parts |
| E4 | unfiltered `distinct_training_days` == distinct dates across exercises | miscounted consistency |
| E5 | ISO-week keys are year-boundary correct | Dec-29 mapped to W53 not next-year W01 |
| F1 | every correlational block carries `n`, `ci_95`, `cohen_d`/`pearson`, `cis_overlap`, `confidence_label` | naked comparative claim |
| F2 | `confidence_label ∈ {insufficient_data, weak, moderate, strong}` | bad label |
| F3 | a bucket's `n` equals its real session count | inflated sample size |
| F4 | CI `None` when `n<2`; Pearson CI `None` when `n<4` | overstated certainty on thin data |
| G1 | required keys present per exercise type | missing `progression`/cardio block |
| G2 | no `None` where a value is required given data exists | silent null |
| G3 | package serializes to JSON | stray non-serializable object |
| G4 | (soft) size ceiling per scope — BROAD ≤ 500 KB, GROUP ≤ 400 KB, FOCUSED ≤ 250 KB | ~986 KB unfiltered package that 429s |
| G5 | (soft) BROAD omits the deep-stat fields | `trim_package` leak |
| G6 | (integrity) scope consistency: focused→≤3 exercises; broad→no leaked deep-stat fields + exactly one aggregation level | mislabeled scope bypassing the trim |

### Scope derived from effective package contents

Scope is computed from what actually survives filtering (`_derive_scope_from_package`),
not from classifier intent: a filter that resolves to **zero** exercises falls
back to BROAD so the broad trim still runs. Unresolved exercise names are popped
before the LLM sees the package, and the answer is prefixed with a plain note
("*X wasn't found in your workout history, so this answer covers …*") — partial
resolution covers the matched exercises, total failure falls back to the broad
package.

### Coordinator details

Every message routes through the Coordinator. A **write-intent regex guard**
routes logging/corrections/goal-setting to the operational single-agent path
**without** an LLM classification call (a misrouted write would bypass the
confirmation gate). `/confirm` bypasses routing entirely — it is the
continuation of an in-flight staged write whose context lives in the agent
session. Analytical turns are mirrored into `session.chat_history` so `/history`
shows them. Multi-candidate name resolution **surfaces the candidates and asks**
on the analytical path instead of guessing the first. **RAG
(`search_fitness_knowledge`) is suppressed on analytical (personal-data)
questions** — `research=None` — and only fires on the operational ReAct loop.

### Token / quota engineering

The package is serialized **compactly** for LLM input (`separators=(",",":")`,
no indent) — about **−34 %** input tokens on a broad package. The grounding
check re-sends the **full compact package** for every scope (~366 KB ≈ 92k
tokens) rather than a subset, so the REMOVE rule can never strip a true claim
for a missing source field; per-question input stays under the 250k/min ceiling.
The retry countdown parses Gemini's own `'retryDelay': '26s'` JSON hint (not just
the api-core "try again in 26.5s" phrasing).

### Bar-weight consistency (Pass 1)

Every shipped weight/volume/PR/e1RM is **bar-inclusive** (plates + bar in the
exercise's unit) or **explicitly labeled plates-only**:

- `goal_projection` previously compared a plates-only `target_weight` against a
  bar-inclusive `current_e1rm` — a mixed frame that understated the gap and made
  `is_on_track`/`months_needed` optimistic. Fixed by adding the exercise's bar
  before computing `target_e1rm`; a barbell goal's gap widens by ~44 lbs (the
  bar's e1RM contribution at 1 rep), a no-bar goal is unchanged.
- `daily_workouts` `estimated_1rm`/`max_weight` are now bar-inclusive, matching
  the per-session values (previously plates-only; `workout_position_effect`
  derives from this block in BROAD scope).
- `all_time_summary.total_volume_raw_lbs` → `…_typed` with a `total_volume_raw_note`
  stating it is plates-only, excludes bar + offsets, and is not unit-normalized.
- Legitimately plates-only display fields now carry a marker:
  `warmup_weight` (`warmup_weight_plates_only`), `goals[].target_weight`
  (`target_weight_plates_only`), `pain_analysis.failed_attempts[].weight`
  (`weight_plates_only`).

### Cross-unit volume (Pass 2) — per-unit buckets

Volume is reported **per typed-unit frame** everywhere it crosses
exercises/sessions, and the two frames are **never added together** — switch-proof
across lbs↔kg gym moves: `*_lbs`/`*_kg` on weekly/monthly/yearly aggregations,
per-exercise `period_volume_*` and `volume_trend_*`, `muscle_group_summary`
(totals, `weekly_volumes`, `trend_*`), `muscle_group_balance` (push/pull/ratio/
dominant/distribution per frame + a `note`), `rankings.highest_volume`
(`{exercise, volume_lbs, volume_kg}`, ordered by an internal kg-equivalent key
that is never emitted), `all_time_summary.total_volume_raw_typed_lbs/_kg`, and
`training_density`. Deadlift is the only genuinely mixed exercise: its
monthly 2025-12 old blended total **4539.6 = 1943.6 lbs + 2596.0 kg** — proof the
old field was adding raw kg numbers onto lbs. Validator E3 now reconciles each
bucket against member sessions of the same frame. A read-only deep-diff harness
(`scripts/diff_volume_passes.py`) proved the change touched only volume fields
and surfaced latent **cross-process nondeterminism** (set-iteration tie-breaks in
`form_quality_mode` and `inter_exercise_correlation` ordering), now fixed with
deterministic sort keys.

### Checkpoint / resume for quota-interrupted questions

A rate-limit 429 mid-pipeline saves a single-slot checkpoint
(`data/checkpoint.json`, gitignored) so the user can say **"continue"** when
quota resets and resume **without re-paying for completed stages**. The
analytical package is never stored — it **rebuilds free** (pure Python,
re-validated; G6 still applies). The draft is stored **verbatim** (grounding must
verify the exact text the user will see), and under the option-a policy the user
**never sees unverified draft text** — a grounding/coverage interruption returns
a status message only and ships the answer solely after verification on resume.
Stage-boundary catches save the last *completed* stage: a draft-time 429 →
`classify` (no draft), a grounding-time 429 → `draft` + verbatim draft, a
coverage-time 429 → `coverage` + grounded text; the operational ReAct loop saves
its message list with **mechanical** head+tail pruning (no LLM summarization). A
new question never silently discards a live slot — it triggers a
**confirm-before-discard** prompt naming the saved question (continue → resume,
"new"/re-send → discard + process, ambiguous → re-ask; stale >48h slots are
dropped silently on load). `/chat` 429 responses carry `checkpoint_saved` and
`retry_after_seconds`; `GET /checkpoint-status` exposes the slot for debugging
(never the draft text).

### Plateau / regression / current-ability — rep-aware, cadence-scaled

The progression detector was rewritten (`_compute_progression`) after it
reported a false **89-day plateau** and a false **7.7% regression** on a Lat
Pulldown that was actually progressing (130×5 → 130×9 with a lighter back-off
day last). Root cause: it anchored "current" on the single latest session and
compared `max_working_weight` only, so rep gains at the same weight were
invisible and a back-off day redefined current ability downward.

- **"New best" = literal weight → reps, never e1RM.** A session is a new best
  iff its top working weight is heavier, or the same weight with more reps —
  matching the PR rule. e1RM is deliberately *not* the basis: Epley inflates
  light high-rep sets (a 20×15 warmup outscores a real 25×3 PR by e1RM), which
  would crown a warmup as the best.
- **"Current ability" = robust recent best**, the best (weight→reps) over the
  exercise's last `CURRENT_ABILITY_SESSIONS` sessions — a lone back-off/high-rep
  day can't lower it.
- **Plateau** is declared only when all of: `sessions_since_best ≥
  PLATEAU_MIN_SESSIONS_SINCE_BEST` (scaled to the exercise's own session
  cadence), the recent e1RM slope is *not* rising (trend gate over
  `PLATEAU_SLOPE_WINDOW` sessions), and there are `≥ MIN_SESSIONS_FOR_TREND`
  sessions; otherwise it reports the observed facts or refuses to opine (thin
  data). e1RM is used *only* as this trend-direction gate, never to pick the best.

Named constants (`process.py`): `CURRENT_ABILITY_SESSIONS = 3`,
`PLATEAU_MIN_SESSIONS_SINCE_BEST = 5`, `PLATEAU_SLOPE_WINDOW = 6`,
`MIN_SESSIONS_FOR_TREND = 4`, `NEW_BEST_WEIGHT_TOL_KG = 0.05`.

### Comment binding at source (D5)

Each set's comment is bound to its row by the DB foreign key
(`Comment.owner_id = training_log._id`) at **fetch time** and carried as
`set["comment"]` / `set["set_db_id"]` — never re-matched downstream. The binding
is 1:1 (no set has >1 comment); ~46% of sets have no comment, which stays `None`
and is never inferred. The operational chat path (`combined_server.py`) had a
reps/weight heuristic that re-fetched comments and matched them by `reps ==
reps AND |weight−offset−typed| < 1.0` — first-match-wins, which **swapped
comments between same-weight×reps sets** (the live-chat misattribution that
quoted one set's notes against another). That heuristic is removed; the bound
comment is read directly. Validator invariant **D5** independently verifies every
set's carried comment against its own `Comment` row and **raises** on a mismatch.

### Operational-path per-unit volume

`get_weekly_volume` (the chat tool) now returns `total_volume_lbs` /
`total_volume_kg` per muscle group instead of one blended
`SUM(metric_weight * 2.2046 * reps)` — the same cross-unit bug as Pass 2, on the
chat-tool surface (Back, Forearms, Biceps were blending kg-native exercises'
kilograms into pounds). It reuses the kg-native rule from the existing
`KG_NATIVE` constant + `DEADLIFT_KG_SWITCH = "2025-12-26"`. Per category,
`lbs + kg` reconciles to the old blended sum (no volume created or lost; only
split). `run_read_only_sql` passes agent SQL through verbatim — it can't be
bucketed, so its response now carries a caveat that kg-native typed values are
kilograms and must not be summed across frames.

> **Superseded by Session 10:** `get_weekly_volume` was subsequently made
> **bar-inclusive** (plates + bar + numeric offset), so it now matches the
> analytical `muscle_group_summary` rather than reconciling to the old
> plates-only blended sum, and the kg-native rule moved to the single source of
> truth in `src/units.py`. See "Operational bar-inclusive volume + kg-native
> predicate unification" below.

### Per-minute vs daily 429 handling

A 429's `quotaId` distinguishes the two (read structurally from `exc.details`,
falling back to `str(exc)`):

- **Per-minute** 429 (`…PerMinute…`, `retryDelay` ~53s) is absorbed
  **silently** — the backend waits the `retryDelay` and retries the same stage
  in place (`PER_MINUTE_MAX_RETRIES = 2`, each wait capped at
  `PER_MINUTE_WAIT_CAP = 70s` + a 2s buffer), holding the request open with no
  checkpoint and no message; the "thinking…" spinner persists. Diagnosis
  confirmed `/chat` has no server timeout and the frontend fetch has none, so a
  ~70s hold is safe.
- **Daily** 429 (or exhausted per-minute retries) checkpoints and surfaces a
  status. The frontend now consumes the server's `message` and renders a
  **Resume** button only when `checkpoint_saved` is true (it POSTs `/resume`,
  which reuses the Coordinator's continue-intent path) — the hardcoded "Daily
  quota reached… 12:30 PM IST" string is gone.
- **Classify-stage gap closed:** the first (classify) call of a question is now
  wrapped too — a daily 429 there checkpoints `completed_stage="classify"`
  (`draft=null`, `params=null`) and resume re-runs the cheap classify.
- **Nothing-to-resume guard:** a resume request with no live slot returns
  `route="none"` ("There's no saved question to resume.") with zero downstream
  LLM calls, instead of running a fresh expensive question.

---

## Session 10 — Progression end-anchor, operational bar-inclusive volume, units unification, response-shape guard, routing Step A

### Progression end-anchor — "current ability" vs "latest session"

`_compute_progression` (`src/data_agent/process.py`) no longer treats the literal
last session as the end of a progression trend. The trend's **end anchor is
current ability** — the robust recent best (`_recent_best_session` over
`CURRENT_ABILITY_SESSIONS`), by the weight→reps PR rule — so a deliberate back-off
day no longer reads as a decline. The latest session is reported **separately** as
`latest_session_date` / `latest_session_weight` / `latest_session_reps` with a
`latest_session_is_backoff` flag, so the Analysis Agent can say "your most recent
session was lighter" without narrating it as a taper. **Cross-frame % guard:**
`weight_change_pct` is computed only within the end (current) unit frame and is set
to `None` when the window spans a unit switch with fewer than two same-frame
sessions, with a `progression_note` explaining why — this removes the ~185%
Deadlift artifact that came from computing a percentage across the 2025-12-26
lbs→kg switch. (Deadlift end now 70→85 = PR; the false Lat Pulldown −7.7% → 0,
held.)

### Operational bar-inclusive volume + kg-native predicate unification

`get_weekly_volume` (the operational chat tool) now returns **bar-inclusive**
volume (plates + bar + numeric offset) in per-unit buckets `total_volume_lbs` /
`total_volume_kg`, matching the analytical `muscle_group_summary`. 9 of 10
categories reconcile exactly; Legs runs ~0.5% high because the operational pass
does not replicate Smith-machine counterbalance reductions (documented residual,
below). The kg-native rule is now a **single source of truth** in
`src/units.py` (`KG_NATIVE_NAMES`/`KG_NATIVE_EXERCISES`, `DEADLIFT_KG_SWITCH`/
`DEADLIFT_KG_SWITCH_DATE`, `is_kg_native`, `kg_native_sql_predicate`,
`kg_native_volume_case`), imported by `validate.py`, `process.py`, and
`combined_server.py` — previously three independent copies of the same predicate.

### Bar-inclusive display reads — get_exercise_history + get_exercise_sessions (#6)

The two operational "recent sets" reads both returned **plates-only** weights
(`metric_weight * 2.2046 + offset`, **no bar**), so barbell/Smith exercises read
~bar-weight light — e.g. recent Barbell Curl showed **10/20/30 lbs** when the
bar-inclusive headline is **43/53/63 lbs** (the ~33 lbs date-ranged curl bar by
2026). This drifted from the analytical package, which is bar-inclusive
(`pr.weight` = 63.07). `get_exercise_history` was the more visible regression
once Step C kept it as a primary read, but **`get_exercise_sessions` was also
plates-only** — it carried an explicit "add the bar" `bar_weight_note` rather
than adding it. Both are now bar-inclusive.

The fix introduces **one** shared conversion,
`combined_server._bar_inclusive_weight(ctx, name, date, metric_weight)`
(`combined_server.py` ~792), that *composes* the analytical-path primitives
(`process._recover_typed_weight` + `_get_numeric_offset` + date-ranged
`_get_bar_weight_lbs` + `_is_kg_native`) — the **single source of truth** the
package and `get_weekly_volume` already use — and returns
`(headline_weight, unit, plates)`. Both `_get_exercise_history_sync` (~954) and
`_get_exercise_sessions_sync` (~2374) call it, so they now report identical,
bar-inclusive numbers in each set's **own unit frame** (kg-native exercises in
kg, date-ranged for Deadlift; others lbs). No bar/offset/kg rule is re-copied.
Notes: the `weight` shown is bar-inclusive but the **warmup ratio stays
plates-based** (the constant bar would otherwise shift which opener counts as a
warmup — the analytical path also flags warmups on plates), so `plates` rides
alongside each set for that check only; and `get_exercise_sessions`'
`bar_weight_note` now says weights are bar-inclusive instead of "add the bar".
Output shapes are unchanged (a per-row/`session` `unit` is added). Same Smith
caveat as `get_weekly_volume` (counterbalance reductions not applied), so
non-Smith barbell sets match the package exactly; Smith sets read very slightly
high.

### E2 warmup 0-opener headline-frame fix

The 0-weight-opener warmup gate (`process.py` ~357) now compares `working_max` on
the **bar-inclusive headline frame** against the bar-inclusive
`exercise_alltime_max` (was plates-only vs bar-inclusive — too strict, so genuine
empty-bar warmups were missed). 43 real missed empty-bar warmups are now correctly
flagged. Safe direction (adds correctly-missed warmups, creates no false
positives); weight / volume / e1RM are unchanged — only rep-range and working-set
counts shift.

### Malformed-response crash guard

Every place that iterates `candidate.content.parts` over a Gemini response is now
guarded against no-candidates / `content=None` / `parts=None` / a non-STOP
terminal `finish_reason` (`MALFORMED_RESPONSE`, `SAFETY`): `agent.py`
(`_run_collect`, `_reflect`, `_auto_extract_memories`), `analysis_agent.py`
(`_collect_text`), and `coordinator.py` (`_classify`, `_coverage_check`). A
malformed envelope degrades to a clean user message ("I wasn't able to form a
clear answer…") instead of a 500 / traceback, and is **not** blindly retried.

### Routing Step A — out_of_scope gate + medical carve-out + coach character

- **out_of_scope route** refuses non-fitness questions at classification with
  **zero downstream spend** — no package build, no agent turn, no search, no
  analysis call (`_out_of_scope_response` returns `OUT_OF_SCOPE_REFUSAL`
  directly). The classifier uses a **fitness-connection test**, not a topic
  blocklist: user data, fitness science, fitness-term definitions,
  training/nutrition, fitness history/culture, and program design are IN; coding,
  AI/tech, politics, non-fitness arts/history, non-fitness
  definitions/translation, puzzles, trivia, and creative writing (including
  motivational poems) are OUT. When genuinely ambiguous, it leans IN.
- **Medical carve-out:** medical/symptom questions are **never** routed
  out_of_scope. The coach never diagnoses, names, or treats a condition (refuses
  + redirects to a professional) but always advises training adaptations,
  substitutions, form cues, warmups, mobility, and load management around a
  stated symptom (with a see-a-professional note).
- **Coach character** added to both `_ANALYSIS_SYSTEM` (`analysis_agent.py`) and
  the operational system prompt (`agent.py`): direct & warm, always explain the
  why with the data, the user holds the final call, bias toward training (never
  toward excuses) — with grounding overriding all four (strong claims trace to
  data or principle; thin data → say so; never confidently wrong; within
  wellbeing bounds).

---

## Routing Step B — muscle-group Category guard + plates-only volume steering

Two targeted analytical-path fixes (Step C is still planned).

### Fix 1 — muscle-group-vs-exercise-name Category guard

A question like "my strength drops when I train triceps after chest" used to break
the analytical path: the classifier slotted "triceps" into `exercise_names`, the
resolver loop (`coordinator._run_analytical`) ran `resolve_exercise_name("triceps")`,
got 5 candidate exercises (substring matches), and emitted "I found multiple
exercises matching triceps. Which one did you mean?" — a category error, because
"triceps" is a **muscle group**, not an exercise.

Before resolving any extracted term, the coordinator now checks it against the
canonical muscle-group Category names via `match_muscle_group` (case-insensitive,
singular/plural tolerant). A term that names a Category is moved to
`muscle_groups` in canonical form (→ GROUP scope) and is **never** sent to
`resolve_exercise_name` or a disambiguation prompt. This is a belt-and-suspenders
guard: it fires regardless of which slot the classifier used. Only genuine
exercise names still resolve/disambiguate — "dumbbell bench press" still surfaces
its 3 real variants. The canonical list is the single source of truth
`MUSCLE_GROUP_NAMES` (derived from `process.CATEGORY_NAMES`, exported via
`src.data_agent`), not a new inline copy. Substring-trap counts that motivated the
guard: "Triceps" matches 5 exercise names, "Chest" 2, "Back" 1. Ordering/fatigue
questions naming a muscle now build a GROUP-scope package and draw on
`inter_exercise_correlation` / `workout_position_effect` (already present in group
scope) instead of pinning to one exercise.

### Fix 2 — plates-only volume steering + footgun demotion

"Total volume by muscle group" used to return plates-only numbers with a "does not
include bar weights" caveat: the model quoted the package's all-time raw
cross-check field instead of the authoritative bar-inclusive volume. The
`muscle_group_summary` already carries bar-inclusive per-unit volume
(`total_volume_lbs` / `total_volume_kg`); the raw plates-only field was a footgun
sitting in `all_time_summary`. Two changes:

- **Steering** (`_ANALYSIS_SYSTEM`, new VOLUME RULES block): for any volume
  question use `muscle_group_summary.total_volume_lbs/_kg` (bar-inclusive,
  per-unit) and `muscle_group_balance`; **never** quote the raw plates-only field
  as "total volume"; report `_lbs` and `_kg` separately and say "pounds-frame
  volume" / "kilograms-frame volume" (never "total volume in pounds") when a
  muscle group carries both frames (Back, Biceps, Forearms).
- **Demotion** (package hygiene): the former top-level
  `total_volume_raw_typed_lbs` / `_kg` / `total_volume_raw_note` keys are moved
  under `all_time_summary._raw_volume_crosscheck` (`typed_lbs`, `typed_kg`,
  `note`) so the plates-only number no longer sits beside authoritative volume
  fields. Values are unchanged; the only other consumer (a validate test fixture)
  was updated. The package is JSON-dumped wholesale to the Analysis Agent, so no
  named-field consumer needed changing.

---

## Routing Step C, Part 1 (prerequisite) — fence the custom-SQL lane to non-weight aggregates

Step C consolidates reads onto the analytical path, which makes the analytical
custom-SQL lane the only ad-hoc fallback. Before that consolidation, that lane
must not become a weight-blend surface. The lane (`coordinator._generate_custom_sql`
→ `src.data_agent.query` → `fetch.query`) runs LLM-generated SQL and only
auto-converts `typed_weight = metric_weight * 2.2046` (plates-only, no bar, no
offset, no per-unit bucketing) — the same blend-prone shape as
`run_read_only_sql`. Its intended lane is counts / dates / gaps / streaks /
patterns; weights and volume have authoritative package fields
(`muscle_group_summary`, `pr` / `pr_period`, `progression`, `e1rm_*`).

Because arbitrary SQL output cannot be reliably unit-typed, `fetch.query`
**hard-refuses** weight-aggregating SQL, returning a structured
`{"refused": true, "reason": …}` instead of executing. The detector matches on
the **invariant** that makes a blend possible, not on a surface form: it
**REFUSES iff `metric_weight` appears anywhere in the query AND a blend-prone
aggregate — `SUM` / `AVG` / `TOTAL` / `MIN` / `MAX` / `GROUP_CONCAT` — appears
anywhere.** `COUNT` is deliberately exempt (it counts rows, never combines the
weight values), so "how many sets + their weights" and `COUNT(*) … WHERE
metric_weight > 0` still pass. Per-row `metric_weight` SELECT (no aggregate) is
still allowed and keeps the existing plates-only caveat; counts/dates that never
mention `metric_weight` pass untouched.

> **#4 fix — alias-without-`AS` and `GROUP_CONCAT` bypass closed.** The previous
> detector tracked only aliases bound with an explicit `AS`, so
> `SELECT SUM(v) FROM (SELECT metric_weight v FROM training_log) t` (alias *without*
> `AS`) slipped through and executed to a meaningless blended kg+lbs sum (~155k);
> `GROUP_CONCAT(metric_weight)` (string-form blend) was unguarded too. The
> invariant rule closes both: to blend weights the SQL must still name
> `metric_weight` in the projection that feeds the alias, so "`metric_weight`
> present + blend aggregate present" catches the no-`AS` form, and `GROUP_CONCAT`
> is now in the refused set. Over-refusal is only ever in the safe direction
> (e.g. `SUM(reps)` in a query that also touches `metric_weight`) — the package
> answers any weight/volume question that gets refused here.

Belt-and-suspenders on the generation side: the coordinator's custom-SQL prompt
now explicitly states the lane is counts/dates/gaps/streaks/patterns only and
must not aggregate weight or compute volume. (The shared `_SQL_SYSTEM` /
`generate_sql` is left untouched — the operational text-to-SQL pipeline
legitimately answers weight questions through it.) On refusal,
`_generate_custom_sql` returns `None`, so the Analysis Agent simply falls back to
the package's authoritative weight/volume fields — no crash, no silent empty
result.

---

## Routing Step C, Parts 2 & 3 (DONE) — flip the read default, strip the operational read tools

With the custom-SQL lane fenced (Part 1), reads can safely consolidate onto the
analytical path. Parts 2 and 3 land together — flip first, strip second — so no
intermediate state strands a request.

### Part 2 — reads default ANALYTICAL

Routing is now a **positive operational allowlist** rather than an operational
default. Operational means exactly: writes (caught first by the unchanged
write-intent regex pre-guard), research/RAG, and specific-date session display —
plus out_of_scope refusals (Step A). **Everything else is analytical**, and the
default when uncertain is now **analytical** (flipped from operational) in all
the spots that pick a route: the `_CLASSIFY_SYSTEM` DEFAULT line, `_classify`'s
parse/exception fallback (`default` dict + `setdefault` + the `_route_fresh`
`params.get` guard), and the module docstring rationale. The old "a wrong
analytical package is confusing, so default operational" reasoning is obsolete:
the analytical package is deterministic and validated (it hard-stops on an
integrity failure), while the operational hand-rolled-SQL read path is the
riskier surface for a read. Medical/symptom questions are reads → analytical by
the new default (the diagnose-vs-adapt prompt rule is unchanged). Safety note: a
verb-less write that slips the regex and defaults analytical simply *won't write*
(the confirmation gate is operational-only), so an ambiguous write can never
corrupt data — worst case it isn't logged and the user rephrases.

### Part 3 — unexpose the four analysis-read tools

`get_personal_record`, `get_weekly_volume`, `query_workout_data`, and
`run_read_only_sql` are removed from the operational agent's exposed tool list
(`combined_server.list_tools`, 35 → 31 tools). At the time they were unexposed
rather than deleted, on the rationale that their `_sync` handlers might be reused
by the analytical path or evals. **(Post-#7 update: that reuse never materialized
for three of the four — a grep found no analytical/eval caller — so
`_get_personal_record_sync`, `_query_workout_data_sync`, and `_run_read_only_sql`
were removed (handler + async wrapper + the dead dispatch branch). Only
`_get_weekly_volume_sync` proved genuinely reused — `tests/test_operational_volume.py`
calls it directly — and is kept. See "Dead-code cleanup" in the Fixed section.)**
Kept exposed: `get_exercise_sessions`,
`resolve_exercise_name`, `read_exercise_comments`, `get_exercise_history`, all
write tools, RAG, memory, and quirks. The operational `SYSTEM_PROMPT` was updated
to match: the four tools are dropped from the READ-WORKOUT group; the
`run_read_only_sql`-specific blocks (DATABASE NOTE / SQL COLUMN RULES / OFFSET
WARNING) and the PR-answer flow (which told the agent to call
`get_personal_record` + a single-rep `read_exercise_comments` caveat) are removed
— PR questions now route analytical and the package's `pr` / `pr_period` carry
the comment and pain flag. The write/edit and session-display flows still
instruct `get_exercise_sessions` + `resolve_exercise_name` (kept), so nothing is
stranded.

**Why this is safe (no strand):** with the default flipped first, reads route
analytical, so the operational agent never receives a read needing a stripped
tool; writes/RAG/display use only kept tools; single-exercise reads (one PR, one
exercise's history) are covered by focused-scope package fields; and no eval
calls the stripped tools through the exposed list (they call the `_sync`
functions directly, which remain). There is now **one read path** —
deterministic and validated.

---

### Known open gaps (updated)

- **Legs Smith-counterbalance residual** — operational `get_weekly_volume` runs
  ~0.5% high on Legs because the chat-tool pass does not replicate the Smith-squat
  counterbalance reductions the analytical path applies. Bar + per-unit buckets
  are otherwise reconciled; the counterbalance correction is the only remaining
  delta.
- **`run_read_only_sql` blend edge** — arbitrary agent SQL can still emit a
  blended `SUM(...*2.2046)` across kg-native and lbs exercises. It can't be
  bucketed (verbatim passthrough), so it's only **annotated** with a unit
  caveat, not corrected.
- **`agent_lock` held during a per-minute wait** — the silent retry holds the
  single agent lock for up to ~2×70s, so a concurrent `/chat` gets the existing
  "busy" 429. Acceptable under the single-user assumption; flagged for multi-user.
- **Routing redesign (Steps A–C) — DONE.** Step A (out_of_scope + medical
  carve-out + coach character), Step B (muscle-group guard + plates-only volume
  steering), and Step C (Part 1 custom-SQL weight-aggregate fence, Part 2
  analytical default flip, Part 3 unexpose the four analysis-read tools) are all
  landed. The duplication / operational-read-leak root cause — low-confidence
  reads falling into the operational hand-rolled-SQL path — is **CLOSED**: there
  is one deterministic, validated read path. Small future cleanup: the two
  borderline display/dump tools kept exposed for now
  (`get_exercise_history`, `read_exercise_comments`) can be revisited later.
- **No token streaming** — answers return whole (non-streaming `_run_collect`);
  long answers have no progressive render.
- **Research / paper fetcher** (PubMed / RAG) is best-effort and uncached — it can
  be slow or empty and still spends an API call on the operational path.
- **No LangGraph / graph orchestration** — routing and the stage pipeline are
  hand-rolled in the Coordinator; a graph framework is not used.
- `cli.py` 429 handling catches `ClientError` from a different code path than the
  server's string-match guard — some error shapes may not surface as rate limits.
