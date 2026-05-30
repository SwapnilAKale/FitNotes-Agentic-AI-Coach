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
Data Agent (src/data_agent.py)
Pure Python, zero LLM calls. Always runs a fixed, complete pipeline regardless of the question.
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

Stress-tested sizes:

All exercises, 90-day, no Phase 2: 2.6 MB, 0.46s
Single exercise + Phase 2: 112 KB, 0.09s
Back group, 365-day: 282 KB, 0.17s
All-time + Phase 2: 4.7 MB, 0.83s

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
The Data Agent (src/data_agent.py, multi-agent branch) is pure Python with zero LLM calls — deterministic, testable, debuggable

Every architectural decision has a documented reason in lessons.md including what went wrong when the first approach was tried. Every stage was implemented iteratively, verified against real data, and refactored when the design proved wrong.

License
MIT