FitNotes Personal Strength Coach
A full-stack agentic AI coaching system built over a personal FitNotes SQLite database. The agent answers natural language questions about workout history, provides fitness science knowledge via RAG, and can log, update, and delete workout data with human-in-the-loop confirmation.
The project has two branches:

main — single-agent system with web UI, fully complete (Stages 1–11 + Post-Stage polish)
multi-agent — multi-agent analytics system, Data Agent complete, Analysis Agent in progress

Built as a learning project covering the full agentic AI stack from scratch — no LangChain, no LangGraph, no abstractions. Every component is hand-built and understood.

What It Does
Single-agent system (main branch):

Ask anything about your training history — PRs, volume trends, exercise frequency, progression over time
Get fitness science answers — backed by a RAG pipeline over 160 PubMed abstracts and Wikipedia articles
Log workouts and goals — with a two-phase confirmation gate before any data is written
Fix mistakes — update or delete logged sets with full audit trail
Remember your preferences — long-term memory persists across sessions via ChromaDB
Handle non-standard exercises — store plain-English quirks for exercises with unusual logging conventions
Upload fitness research articles — add PDF papers to the RAG knowledge base directly from the UI
Upload FitNotes backups — replace the workout database with integrity and row count validation

Multi-agent system (multi-agent branch, in progress):

Complete analytical coverage — every exercise in the database analyzed, not a sampled subset
Deterministic data collection — pure Python pipeline with no LLM decisions in the data layer
Plateau detection — exercises stuck for > 4 weeks automatically trigger full comment history fetch
Progression analysis — e1RM curves, weight change rates, volume trends across all muscle groups
Pain and form tracking — comment-derived pain analysis, technique variants, form quality trends
Phase 2 depth — for exercises with significant changes, full session detail is automatically included


Tech Stack
Single-agent (main branch):
ComponentTechnologyAgent LLMGemini 3.1 Flash Lite (500 RPD free tier)Vector DBChromaDB (local persistent)EmbeddingsBAAI/bge-small-en-v1.5 (sentence-transformers, local)Rerankercross-encoder/ms-marco-MiniLM-L-6-v2 (local)Tool ProtocolMCP (Model Context Protocol)DatabaseSQLite (.fitnotes)Web ServerFastAPI + uvicorn (port 8000 API, port 3000 frontend)FrontendHTML/CSS/JavaScript (vanilla, Tailwind CDN)
Multi-agent additions (multi-agent branch):
ComponentTechnologyData AgentPure Python, zero LLM calls, deterministic pipelineAnalysis AgentGemini Flash 2.0, thinking_budget=4096 (planned)CoordinatorGemini Flash Lite, simple router (planned)Exercise quirksdata/user_context.json (exercise_quirks array)

Project Structure
fitnotes_coach/
├── data/
│   ├── FitNotes_Backup.fitnotes      # Your SQLite workout DB (gitignored)
│   ├── chroma_db/                     # ChromaDB (fitness knowledge + memory, gitignored)
│   ├── user_context.json              # Personal data conventions + exercise quirks (gitignored)
│   └── memory.json                    # Long-term memory store (gitignored)
├── mcp_servers/
│   └── combined_server.py             # All 31 MCP tools (single-agent)
├── src/
│   ├── agent.py                       # AgentSession, ReAct loop, Gemini client
│   ├── data_agent.py                  # [multi-agent branch] Data Agent — deterministic collection pipeline
│   ├── db.py                          # DB connections + sanitize_sql()
│   ├── llm.py                         # Gemini SQL generation + explanation
│   ├── memory.py                      # Memory store + ChromaDB sync
│   ├── rag.py                         # Hybrid search + reranker
│   ├── schema_prompt.py               # Schema + user context injection
│   ├── text_to_sql.py                 # Text-to-SQL pipeline
│   ├── answer.py                      # ⚠️ Dead code — Stage 2-3 legacy, replaced by MCP agent loop
│   └── router.py                      # ⚠️ Dead code — Stage 2-3 legacy, replaced by MCP agent loop
├── scripts/
│   ├── build_corpus.py                # PubMed + Wikipedia ingestion
│   ├── regenerate_ground_truth.py     # Eval ground truth refresh
│   └── test_memory.py                 # Standalone memory test
├── evals/
│   ├── eval_set.json                  # 20 evaluation questions (gitignored)
│   ├── run_evals.py                   # SQL + LLM-judge scoring
│   ├── row_compare.py                 # ORDER-BY agnostic comparator
│   └── stress_test.py                 # 19-test adversarial suite
├── server.py                          # FastAPI web server — run this to start everything
├── frontend/
│   ├── server.py                      # Standalone frontend server (UI-only testing)
│   └── index.html                     # Chat UI — no sidebar, paperclip upload in input bar
└── cli.py                             # Async CLI + confirmation gate

Setup
Prerequisites

Python 3.11+
A FitNotes backup file (.fitnotes) exported from the FitNotes app
A Gemini API key (free tier) from aistudio.google.com

Installation
bashgit clone https://github.com/yourusername/fitnotes-coach
cd fitnotes-coach
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
Configuration
Create a .env file in the project root:
envGEMINI_API_KEY=your_gemini_api_key_here
HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING=1
Add your FitNotes database
Export your FitNotes backup:

Open FitNotes → Menu → Backup → Export Backup
Copy the .fitnotes file to data/FitNotes_Backup.fitnotes

Build the fitness knowledge corpus (one-time)
bashpython scripts/build_corpus.py
This fetches ~160 PubMed abstracts and Wikipedia articles on strength training, hypertrophy, and fitness science. Takes 2-3 minutes. Only needs to be run once.
Run (single-agent, web UI)
bashpython server.py
Starts both the API server (port 8000) and frontend (port 3000), opens browser automatically.
Run (single-agent, CLI)
bashpython cli.py
For memory testing without loading all tools:
bashpython cli.py --memory-only
For debug output (tool calls, results, traces):
bashpython server.py --debug
Run frontend only (UI testing without agent)
bashcd frontend
python server.py
Starts only port 3000. Use this for testing upload flows, layout changes, and UI features without burning Gemini quota or waiting for MCP initialization.
Run the Data Agent standalone (multi-agent branch)
bash# Test: last 90 days, specific exercises
python src/data_agent.py 90 "Lat Pulldown" "Deadlift"

# Test: last 365 days, all exercises
python src/data_agent.py 365

# Test: 30 days, muscle group filter
python src/data_agent.py 30 --muscle_group Back

Usage Examples (single-agent)
You: What is my deadlift PR?
→ Your deadlift personal record is 85 kg for 5 reps, achieved on 2026-04-20.

You: How many sets of back exercises did I do last month?
→ You completed 47 sets across 8 back exercises in the last 30 days.

You: What does research say about optimal training frequency for hypertrophy?
→ [RAG-backed answer citing PubMed abstracts]

You: Log today's workout: flat dumbbell bench press, 3 sets — 50 lbs x 10, 55 lbs x 8, 55 lbs x 6
→ What date was this workout?

You: yesterday
→ [Confirmation gate fires — you type yes]
→ [Execute gate fires — you type yes]
→ ✅ 3 sets of Flat Dumbbell Bench Press logged for 2026-05-19.

You: I have a new exercise called Dumbbell Hold. Reps stores seconds, weight is lbs.
→ Understood — I'll interpret Dumbbell Hold sets as hold duration, not rep count.

Tools — Single-Agent (31 total)
Organized into 8 groups:
Read — Workout Data: query_workout_data, get_personal_record, get_exercise_history, get_weekly_volume, run_read_only_sql, get_exercise_sessions, read_exercise_comments, resolve_exercise_name
Read — Knowledge: search_fitness_knowledge
Write — Logging: log_workout, execute_staged_workout, log_bodyweight
Write — Goals: set_goal, execute_staged_goal, update_goal, execute_staged_goal_update, delete_goal, execute_staged_goal_delete
Write — Corrections: update_workout_set, execute_staged_set_update, delete_workout_set, execute_staged_set_delete
Verify: verify_workout_logged, verify_goal_set, verify_set_updated, verify_set_deleted
Memory: remember_fact, recall_memories, forget_fact
Exercise Quirks: add_exercise_quirk, update_exercise_quirk, delete_exercise_quirk, list_exercise_quirks

Architecture
Single-Agent System
Agent Loop (ReAct)
Hand-built ReAct loop — no LangChain or LangGraph. Each question: Thought → Tool Selection → Tool Execution → Observation → repeat until answer. Max 7 iterations. A reflection step reviews the answer before returning it. Building the loop from scratch teaches what frameworks abstract away: context accumulation costs, tool schema sizing, graceful error handling, and why confirmation gates must live outside the agent.
RAG Pipeline
Three-stage retrieval: query rewriting (casual English → academic terms) → BM25 + dense hybrid search → cross-encoder reranking (threshold 0.0). A relevance gate filters topically adjacent but irrelevant documents before answer composition. Section-aware chunking splits academic papers on headers first (Introduction, Methods, Results, Discussion, Conclusion) before word-count chunking within sections — conclusion chunks surface directly rather than being buried in 6000-char mixed-content blocks.
Exercise Session Display
get_exercise_sessions returns pre-formatted display_sets strings rather than raw weight values. Each string has set number, weights in correct units, inline comments, drop sets merged with →, and warmup labeled. The agent copies these strings verbatim — no arithmetic, no formatting decisions. All unit conversions, bar weights, and quirk offsets are applied at the tool level before returning.
Write Operations
Two-phase pattern: stage (validate + preview) → CLI confirmation gate → execute (DB write) → verify (read-back). The agent cannot bypass the gate. Write connections are separate from read connections at the SQLite level.
Long-Term Memory (Option B)
Facts in memory.json (source of truth, 30-fact cap). ChromaDB user_memory collection is the search index. Per question: embed question → retrieve top 5 semantically relevant facts (cosine distance < 0.8) → inject only those into system_instruction. Token cost stays constant at ~100 tokens regardless of total memory size.
User Article Upload
PDF articles stored in data/user_articles/ and chunked into ChromaDB using section-aware splitting. Article lifecycle is self-healing: list_user_articles auto-syncs ChromaDB and disk on every call.
MCP (Model Context Protocol)
All tools exposed via a single combined_server.py subprocess. Single server avoids Windows IOCP deadlock from multiple concurrent stdio sessions. Sentence-transformers pre-loaded in main thread before server.run() to avoid OpenMP deadlock.

Multi-Agent System (multi-agent branch)
Architecture overview
User question
      │
      ▼
 Coordinator
(pure router)
      │
      ├── Simple lookup → single-agent path (existing tools)
      │
      └── Analytical question
                │
                ▼
          Data Agent
    (pure Python, no LLM)
         collect()
              │
              ▼
    prepare_analysis_package()
    compact summaries + Phase 2
              │
              ▼
       Analysis Agent
    (Gemini Flash, thinking=4096)
    reasons over complete dataset
              │
              ▼
         Final answer
Data Agent (src/data_agent/)
Pure Python, zero LLM calls. Structured as a four-module package:

- fetch.py — all SQLite access; returns typed raw rows, no interpretation
- process.py — pure function: (raw_rows, user_context) → package; no DB, no clock, no I/O
- validate.py — independent post-condition checks; never recomputes, only asserts invariants
- __init__.py — thin facades: collect(), prepare_analysis_package(); exports DataAgentIntegrityError

Correctness spec and test suites:

docs/data_agent_spec.md — invariants named after the bug each one catches, plus golden cases pinned to a DB snapshot
tests/test_data_agent_golden.py — pinned results against the real database
tests/test_data_agent_validate.py — synthetic per-invariant defect injection tests

Validator behavior — runs on every package produced by collect() and prepare_analysis_package():

Soft violations (G4 size ceiling, G5 scope leaks) are logged as warnings
Any integrity violation (wrong units, PR below session max, negative weights, etc.) raises DataAgentIntegrityError
The Coordinator catches DataAgentIntegrityError before calling the Analysis Agent and returns a clean failure message to the user — a wrong package can never reach the LLM

Scope-aware packaging — the Coordinator derives scope from query filters; trim_package() builds accordingly:

FOCUSED (≤ 3 named exercises): full detail — sessions, full_comments, all stat blocks, all aggregation levels
GROUP (muscle-group filter): full_comments capped to 30 most recent + all pain-flagged entries; one aggregation level
BROAD (no filter): full_comments removed; deep-stat blocks removed; one aggregation level (< 180 days → weekly, 180–730 days → monthly, > 730 days → yearly)
BROAD 365-day package: 1 443 KB → 396 KB — resolves the free-tier 429 failures on general questions

Phase 1 — Always runs for all exercises active in the query period:

Session-level aggregation: max weight, e1RM, volume, form quality, pain flag per session
Plateau detection: date of first session at current max weight, days since
PR history and e1RM projections
Learning curve: sessions_to_first_pr, first_30d_weight_gain
Technique variant detection from comments
Pain analysis: session count, occurrences with comment text
Duration and distance progressions for non-weight exercises
Muscle group summary: weekly volume, push/pull ratio, form distribution
Training consistency: sessions per week, missed weeks, day-of-week patterns
All-time summary: total training days, sets, exercises, streaks, gaps
PR rankings across all exercises
Substitution detection
Fastest improving and most stagnant exercises

Phase 2 — Triggered by deterministic Python conditions:

plateau_days > 28 → fetch full comment history for that exercise
weight_change_pct > 20 → fetch full session detail with all set-level data
Triggers are Python constants, not LLM decisions

Output aggregation is time-based:

≤ 90 days → session-level detail
≤ 365 days → weekly aggregation
All-time → monthly aggregation

Analysis Agent (planned)
Receives prepare_analysis_package() output — compact summaries with all comment-derived analytics, Phase 2 full comments for triggered exercises. Reasoning only, no data decisions. thinking_budget=4096. Never asks for more data — the package is complete.
Coordinator (planned)
Single routing decision: is this an analytical question (multi-agent pipeline) or a simple lookup (single-agent tools)? No data strategy decisions. No intelligence about what data to request. Pure router.
prepare_analysis_package() (planned)
Wrapper over collect(). Strips raw sets arrays (the main bloat) while keeping all analytics derived from them: pain_analysis, technique_variants, form_quality per session, comment_keyword_trends, full_comments for Phase 2 exercises. Returns a compact, Analysis-Agent-ready package.

What Each Component Built
Stage / ComponentWhat Was BuiltStage 1Text-to-SQL pipeline — natural language → SQL → execute → explainStage 1.5Eval harness — execution-based SQL scoring + LLM-as-judgeStage 2Naive RAG — PubMed + Wikipedia corpus, ChromaDB, LLM routerStage 3Better retrieval — query rewriting, BM25 hybrid, cross-encoder rerankingStage 4MCP servers + agent loop — replaced hardcoded router with tool-calling agentStage 5ReAct + new tools — resolve_exercise_name, read_exercise_comments, reflectionStage 6Write actions — two-phase writes with human-in-the-loop confirmation gateStage 7Long-term memory — ChromaDB-backed per-question retrieval (Option B)Stage 8Stress testing — 19/19 adversarial tests passStage 9Polish — unit fixes, 1-rep PR warnings, relevance gate, error handlingStage 10Gemini migration, Memory Option B, ground truth regenerationStage 11Exercise quirks system, tool groupingPost-11FastAPI web UI, streaming fixes, upload validation, debug mode, rate limit UXPost-11RAG section-aware chunking, article lifecycle sync, Groq eliminationmulti-agentData Agent — complete deterministic collection pipeline, all exercises, Phase 2 triggersnextAnalysis Agent — receive complete package, reason without requesting more datanextCoordinator — pure router, wire into FastAPI /analyze endpoint

Key Lessons Learned
Reasoning models ≠ instruction-following models. Use reasoning models for reasoning tasks. Use instruction-following models for SQL generation, classification, and format-constrained tasks. Mixing them causes over-thinking on simple tasks.
Agent initialization is expensive. Each startup sends the full system prompt + all tool schemas. On free-tier APIs this is 10-20% of your daily budget before asking a single question. Condense system prompts aggressively.
Silent write failures are worse than noisy confirmations. A system that fails silently while appearing to succeed corrupts the user's mental model. Always explicitly state whether a write succeeded or failed.
Context window accumulation is an exponential cost risk. Each tool result is re-sent on every subsequent API call in the loop. Three questions with large tool results can burn a daily token budget. Prune context aggressively in production agents.
Wrong ground truth means every downstream eval is wrong. Ground truth must be in the user's units and schema. A correct pipeline that returns correct data in a different column order will fail evals written for the old schema.
Build from scratch before using frameworks. Building the agent loop, confirmation gate, and context management by hand teaches what LangGraph, LangChain, and similar frameworks abstract away. The abstractions make sense once you have hit the problems they solve.
Move formatting decisions to the tool, not the agent. When the agent is responsible for formatting structured data (grouping drop sets, applying unit conversions, matching comments to sets), it produces inconsistent results across sessions. Pre-format at the tool level and have the agent copy verbatim.
Two sources of truth for the same fact will diverge. Unit preferences defined in both user_context.json and memory.json caused conflicting answers. Pick one authoritative source and enforce it explicitly in the system prompt.
Pre-aggregation is architecturally superior to agent-directed data requests. The Coordinator becomes a pure router. The Analysis Agent receives a complete self-contained package. Phase 2 triggers are deterministic Python. Output stays compact because it is summaries, not raw session arrays.
Pure Python for the data layer eliminates an entire class of bugs. When the data layer is deterministic, the LLM's job is to reason, not to do arithmetic. Every "agent math" inconsistency disappears.
Exhaustive code review finds bugs that testing cannot. Phantom training days, false pain flags, wrong week numbering, unused constants — none of these appeared in happy-path tests. They required reading every line and reasoning about edge cases.
Domain knowledge is as important as schema knowledge. Schema tells you what fields exist. Domain knowledge tells you what they mean. A reps=0 set is a failed attempt for a strength exercise and a normal logging convention for a Farmers Walk. The schema cannot tell you which.
Data-level instructions beat system prompt rules. When a model has strong training priors about a topic (e.g. "cables provide constant tension"), system prompt rules that contradict those priors are consistently ignored. The fix is to embed the override instruction in the tool result data itself.
Chunk boundaries matter as much as chunk size. A 200-word Conclusion chunk scores better in retrieval than a 6000-char chunk containing the conclusion plus everything else.

Running the Evals
bashpython evals/run_evals.py        # SQL correctness + answer quality
python evals/stress_test.py      # 19 adversarial tests
Current scores:

SQL eval: 15/20 (75%) — remaining failures are column-name non-determinism
Stress test: 19/19 (100%)
Answer judge: 8/8 (100%) where judge quota was available


Future Work
Multi-Agent System (actively in progress on multi-agent branch)
Analysis Agent — receives prepare_analysis_package() output, reasons over complete dataset with thinking_budget=4096. Answers questions the single agent can never answer: complete plateau analysis across all 51 exercises, overtraining signal detection, progressive overload quality assessment, pattern detection across muscle groups.
prepare_analysis_package() — wrapper over collect() that strips raw set arrays and returns compact analysis-ready summaries. Target size: 100-300 KB for any question, regardless of database size.
Coordinator routing logic — single LLM call: is this analytical (multi-agent pipeline) or simple lookup (single-agent tools)? Wire into FastAPI /analyze endpoint.
End-to-end testing — stress test the full Data Agent → Analysis Agent → Coordinator pipeline. Git commit on multi-agent branch after full verification.
Single-Agent (near-term)
Write-ahead log + DB merge — Currently agent-written workout data lives in the SQLite DB. When user uploads a fresh FitNotes backup, agent-written sets are overwritten. Fix: log every agent write to agent_writes.json. On new DB upload, replay the write log onto the new file before replacing the old one. Teaches transaction logging, SQLite conflict resolution, and data integrity across two sources.
Session & Memory Architecture (Pre-Deployment)
The current auto-extraction runs at CLI session end. On the web server, sessions never terminate — there is no natural trigger for extraction.
Three options for web session management:

Option A — Inactivity timer: trigger extraction after X minutes of no /chat requests. Most production-like but requires asyncio background tasks with cancellation logic.
Option B — Message count trigger: run extraction every N messages (e.g. every 10 exchanges). Simple, 5 lines.
Option C — Explicit end session button: UI button triggers extraction before clearing context. User controls when the session ends.
Recommended: Option B + Option C combined.

Conversational memory architecture (three-layer):

Short-term: last N messages (active context per request)
Mid-term: session summaries — compressed LLM summaries of past conversations stored and retrieved at session start
Long-term: extracted facts in memory.json (existing)

Per-message delete button: Add delete button under each message. Pressing it removes that message from the agent's conversation history so it no longer influences future responses.
Implement all of the above before deploying to web — the growing context window will hit quota limits without it.
Learning Extensions
LangGraph — Rebuild the agent loop using LangGraph's state machine framework. The current hand-built ReAct loop in agent.py does exactly what LangGraph provides — but explicitly, without abstractions. Rebuilding in LangGraph will make the framework's design decisions immediately obvious. The right order: build from scratch first (done) → add real features → then use the framework knowing exactly what it abstracts.
Agent Memory with Knowledge Graphs — Replace the flat memory.json fact store with a graph database (NetworkX locally, Neo4j for production). Store relationships between facts: "trained chest → leads to → shoulder fatigue → affects → overhead press performance." The agent can reason over connections, not just isolated facts.
Computer Use for corpus updates — Automate build_corpus.py by having the agent call the PubMed API directly (no browser needed — PubMed has a free API). The agent searches for new papers on a topic, fetches abstracts, embeds them, adds to ChromaDB. Scheduled weekly.

Notes for Interviewers
This project deliberately avoids high-level frameworks (LangChain, LangGraph, LlamaIndex) to demonstrate understanding of the underlying components:

The ReAct agent loop, context management, and tool selection are hand-built in src/agent.py
The RAG pipeline (query rewriting, BM25, dense retrieval, cross-encoder reranking) is built from components in src/rag.py
The MCP server protocol is implemented directly using the mcp Python SDK
The confirmation gate for write operations is an explicit CLI-level intercept, not a framework feature
Long-term memory uses ChromaDB for semantic retrieval — same vector search used for the knowledge base
The Data Agent (src/data_agent/, multi-agent branch) is a four-module pure-Python package (fetch / process / validate / __init__); zero LLM calls; the validator hard-stops on integrity violations before any wrong data can reach the LLM

Every architectural decision has a documented reason in lessons.md including what went wrong when the first approach was tried. Every stage was implemented iteratively, verified against real data, and refactored when the design proved wrong.

License
MIT

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

**Coordinator (`src/coordinator.py`) — complete**

Single entry point for every user message. One classification call
(temperature=0, thinking_budget=0) extracts route, exercise_names,
muscle_groups, query_period_days. Routes to:
- Analytical: Data Agent → prepare_analysis_package() → Analysis Agent
  → grounding check → coverage check (1 retry if incomplete)
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
`all_time_sessions`, `total_sessions_period`. Strength-specific fields
(max_working_weight, reps_at_max, estimated_1rm, volume, etc.) are
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
50 exercises, the package was built for the wrong scope, and the Analysis Agent
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
| B8 | `is_pr_session`/`pr_velocity`/`pr_count`/`pr_context`/`learning_curve` from real weights, not the flag | flag leakage (tracked gap until fully refactored) |
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

### Known open gaps (updated)

- **Bar weight in operational volume** — `get_weekly_volume` is now per-unit but
  still **plates-only** (`metric_weight * 2.2046 * reps`, no bar), inconsistent
  with the analytical path's bar-inclusive volume. Deliberately out of scope for
  the cross-unit fix; tracked separately.
- **`run_read_only_sql` blend edge** — arbitrary agent SQL can still emit a
  blended `SUM(...*2.2046)` across kg-native and lbs exercises. It can't be
  bucketed (verbatim passthrough), so it's only **annotated** with a unit
  caveat, not corrected.
- **kg-native predicate duplication** — the rule (`KG_NATIVE` +
  `DEADLIFT_KG_SWITCH` / `validate._KG_NATIVE` / `process._is_kg_native`) is
  copied across `process.py`, `validate.py`, and both MCP servers; flagged
  in-code to unify into one shared helper.
- **`agent_lock` held during a per-minute wait** — the silent retry holds the
  single agent lock for up to ~2×70s, so a concurrent `/chat` gets the existing
  "busy" 429. Acceptable under the single-user assumption; flagged for multi-user.
- **Warmup 0-opener gate** (`process.py` ~342) compares a plates-only working max
  against the bar-inclusive `exercise_alltime_max` — an internal cross-frame
  heuristic (not a shipped value), left as-is.
- **No token streaming** — answers return whole (non-streaming `_run_collect`);
  long answers have no progressive render.
- **Research / paper fetcher** (PubMed / RAG) is best-effort and uncached — it can
  be slow or empty and still spends an API call on the operational path.
- **No LangGraph / graph orchestration** — routing and the stage pipeline are
  hand-rolled in the Coordinator; a graph framework is not used.
- `cli.py` 429 handling catches `ClientError` from a different code path than the
  server's string-match guard — some error shapes may not surface as rate limits.
