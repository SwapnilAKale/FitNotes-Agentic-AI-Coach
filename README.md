# FitNotes Personal Strength Coach

A full-stack agentic AI coaching system built over a personal FitNotes SQLite database. The agent answers natural language questions about workout history, provides fitness science knowledge via RAG, and can log, update, and delete workout data with human-in-the-loop confirmation.

Built as a learning project covering the full agentic AI stack from scratch — no LangChain, no LangGraph, no abstractions. Every component is hand-built and understood.

---

## What It Does

- **Ask anything about your training history** — PRs, volume trends, exercise frequency, progression over time
- **Get fitness science answers** — backed by a RAG pipeline over 160 PubMed abstracts and Wikipedia articles
- **Log workouts and goals** — with a two-phase confirmation gate before any data is written
- **Fix mistakes** — update or delete logged sets with full audit trail
- **Remember your preferences** — long-term memory persists across sessions via ChromaDB
- **Handle non-standard exercises** — store plain-English quirks for exercises with unusual logging conventions
- **Upload fitness research articles** — add PDF papers to the RAG knowledge base directly from the UI
- **Upload FitNotes backups** — replace the workout database with integrity and row count validation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent LLM | Gemini 3.1 Flash Lite (500 RPD free tier) |
| Vector DB | ChromaDB (local persistent) |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers, local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| Tool Protocol | MCP (Model Context Protocol) |
| Database | SQLite (.fitnotes) |
| CLI | Python asyncio with graceful shutdown |
| Web Server | FastAPI + uvicorn (two processes: port 8000 API, port 3000 frontend) |
| Frontend | HTML/CSS/JavaScript (vanilla, Tailwind CDN) |

---

## Project Structure

```
fitnotes_coach/
├── data/
│   ├── FitNotes_Backup.fitnotes      # Your SQLite workout DB
│   ├── chroma_db/                     # ChromaDB (fitness knowledge + memory)
│   ├── user_context.json              # Personal data conventions + exercise quirks
│   └── memory.json                    # Long-term memory store
├── mcp_servers/
│   └── combined_server.py             # All 31 MCP tools
├── src/
│   ├── agent.py                       # AgentSession, ReAct loop, Gemini client
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
│   ├── eval_set.json                  # 20 evaluation questions
│   ├── run_evals.py                   # SQL + LLM-judge scoring
│   ├── row_compare.py                 # ORDER-BY agnostic comparator
│   └── stress_test.py                 # 19-test adversarial suite
├── server.py                          # FastAPI web server — run this to start everything
├── frontend/
│   ├── server.py                      # Standalone frontend server (UI-only testing)
│   └── index.html                     # Chat UI — no sidebar, paperclip upload in input bar
└── cli.py                             # Async CLI + confirmation gate
```

---

## Setup

### Prerequisites
- Python 3.11+
- A FitNotes backup file (`.fitnotes`) exported from the FitNotes app
- A Gemini API key (free tier) from [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
git clone https://github.com/yourusername/fitnotes-coach
cd fitnotes-coach
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING=1
```

### Add your FitNotes database

Export your FitNotes backup:
- Open FitNotes → Menu → Backup → Export Backup
- Copy the `.fitnotes` file to `data/FitNotes_Backup.fitnotes`

### Build the fitness knowledge corpus (one-time)

```bash
python scripts/build_corpus.py
```

This fetches ~160 PubMed abstracts and Wikipedia articles on strength training, hypertrophy, and fitness science. Takes 2-3 minutes. Only needs to be run once.

### Run

```bash
python cli.py
```

For memory testing without loading all tools:
```bash
python cli.py --memory-only
```

### Run the Web Interface (recommended)

```bash
python server.py
```

Starts both the API server (port 8000) and frontend (port 3000), opens browser automatically.

### Run frontend only (UI testing without agent)

```bash
cd frontend
python server.py
```

Starts only port 3000. Use this for testing upload flows, layout changes, and UI features without burning Gemini quota or waiting for MCP initialization.

---

## Usage Examples

```
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
```

---

## Tools (31 total)

Organized into 8 groups:

**Read — Workout Data:** `query_workout_data`, `get_personal_record`, `get_exercise_history`, `get_weekly_volume`, `run_read_only_sql`, `get_exercise_sessions`, `read_exercise_comments`, `resolve_exercise_name`

**Read — Knowledge:** `search_fitness_knowledge`

**Write — Logging:** `log_workout`, `execute_staged_workout`, `log_bodyweight`

**Write — Goals:** `set_goal`, `execute_staged_goal`, `update_goal`, `execute_staged_goal_update`, `delete_goal`, `execute_staged_goal_delete`

**Write — Corrections:** `update_workout_set`, `execute_staged_set_update`, `delete_workout_set`, `execute_staged_set_delete`

**Verify:** `verify_workout_logged`, `verify_goal_set`, `verify_set_updated`, `verify_set_deleted`

**Memory:** `remember_fact`, `recall_memories`, `forget_fact`

**Exercise Quirks:** `add_exercise_quirk`, `update_exercise_quirk`, `delete_exercise_quirk`, `list_exercise_quirks`

---

## Architecture

### Agent Loop (ReAct)
The agent uses a hand-built ReAct loop — no LangChain or LangGraph. Each question goes through: Thought → Tool Selection → Tool Execution → Observation → repeat until answer. Max 7 iterations. A reflection step reviews the answer before returning it.

Building the loop from scratch was intentional — it teaches what frameworks like LangGraph abstract away: context accumulation costs, tool schema sizing, graceful error handling, and why confirmation gates must live outside the agent.

### RAG Pipeline
Three-stage retrieval: query rewriting (casual English → academic terms) → BM25 + dense hybrid search → cross-encoder reranking (threshold 0.0). A relevance gate filters topically adjacent but irrelevant documents before answer composition.

### Exercise Session Display
get_exercise_sessions returns pre-formatted display_sets strings rather than
raw weight values. Each string has set number, weights in correct units,
inline comments in parentheses, drop sets merged with →, and warmup labeled.
The agent copies these strings verbatim — no arithmetic, no formatting
decisions. Exercise quirks with a numeric_offset field are applied at the
tool level before returning results.

### Write Operations
Two-phase pattern: stage (validate + preview) → CLI confirmation gate → execute (DB write) → verify (read-back). The agent cannot bypass the gate. All write connections are separate from read connections at the SQLite level.

### Long-Term Memory (Option B)
Facts stored in `memory.json` (source of truth, 30 fact cap). ChromaDB `user_memory` collection is the search index. Per-question: embed the question → retrieve top 5 semantically relevant facts (cosine distance < 0.8) → inject only those into `system_instruction`. Token cost stays constant at ~100 tokens regardless of total memory size.

### User Article Upload
PDF articles uploaded through the UI are stored in `data/user_articles/` and chunked into ChromaDB using section-aware splitting (splits on academic section headers first, then paragraph word-count within each section). Article lifecycle is self-healing: `list_user_articles` auto-syncs ChromaDB and disk on every call. Files on disk not in ChromaDB are auto-ingested. Chunks in ChromaDB with no file on disk are auto-removed. The two stores are always consistent after any list call.

### MCP (Model Context Protocol)
All tools exposed via a single `combined_server.py` subprocess. Single server avoids Windows IOCP deadlock that occurs with multiple concurrent stdio MCP sessions. Sentence-transformers pre-loaded in main thread before `server.run()` to avoid OpenMP deadlock.

---

## What Each Stage Built

| Stage | What Was Built |
|-------|---------------|
| 1 | Text-to-SQL pipeline — natural language → SQL → execute → explain |
| 1.5 | Eval harness — execution-based SQL scoring + LLM-as-judge |
| 2 | Naive RAG — PubMed + Wikipedia corpus, ChromaDB, LLM router |
| 3 | Better retrieval — query rewriting, BM25 hybrid, cross-encoder reranking |
| 4 | MCP servers + agent loop — replaced hardcoded router with tool-calling agent |
| 5 | ReAct + new tools — resolve_exercise_name, read_exercise_comments, reflection |
| 6 | Write actions — two-phase writes with human-in-the-loop confirmation gate |
| 7 | Long-term memory — ChromaDB-backed per-question retrieval (Option B) |
| 8 | Stress testing — 19/19 adversarial tests pass |
| 9 | Polish — unit fixes, 1-rep PR warnings, relevance gate, error handling |
| 10 | Gemini migration, Memory Option B, ground truth regeneration |
| 11 | Exercise quirks system, tool grouping |
| Session 6 | Analytics prep — memory extraction rewrite, schema analytical patterns, context pruning, reflection checks, thinking budget, SQL column rules, time range inference |
| Multi-agent | Separate branch — Data Agent + Analysis Agent + Coordinator for complete DB analysis |

---

## Key Lessons Learned

**Reasoning models ≠ instruction-following models.** Use reasoning models for reasoning tasks. Use instruction-following models for SQL generation, classification, and format-constrained tasks. Mixing them causes over-thinking on simple tasks.

**Agent initialization is expensive.** Each startup sends the full system prompt + all tool schemas. On free-tier APIs this is 10-20% of your daily budget before asking a single question. Condense system prompts aggressively.

**Silent write failures are worse than noisy confirmations.** A system that fails silently while appearing to succeed corrupts the user's mental model. Always explicitly state whether a write succeeded or failed.

**Context window accumulation is an exponential cost risk.** Each tool result is re-sent on every subsequent API call in the loop. Three questions with large tool results can burn a daily token budget. Prune context aggressively in production agents.

**Wrong ground truth means every downstream eval is wrong.** Ground truth must be in the user's units and schema. A correct pipeline that returns correct data in a different column order will fail evals written for the old schema.

**Build from scratch before using frameworks.** Building the agent loop, confirmation gate, and context management by hand teaches what LangGraph, LangChain, and similar frameworks abstract away. The abstractions make sense once you've hit the problems they solve.

**Move formatting decisions to the tool, not the agent.** When the agent
is responsible for formatting structured data (grouping drop sets, applying
unit conversions, matching comments to sets), it produces inconsistent results
across sessions. Pre-format at the tool level and have the agent copy verbatim.

**Two sources of truth for the same fact will diverge.** Unit preferences
defined in both user_context.json and memory.json caused conflicting answers.
Pick one authoritative source and enforce it explicitly in the system prompt.

**Schema prompt column names are not enough when training priors override them.**
The agent kept using wrong SQL column names despite correct schema descriptions.
Fix: add explicit SQL COLUMN RULES to the system prompt as a hard override,
not just a schema description.

**Deterministic data collection beats agent-directed data requests.**
For analytical completeness, define what data to collect in Python logic,
not LLM decisions. The agent deciding what to fetch introduces inconsistency.
A fixed collection pipeline ensures complete coverage every time.

---

## Running the Evals

```bash
python evals/run_evals.py        # SQL correctness + answer quality
python evals/stress_test.py      # 19 adversarial tests
```

Current scores:
- SQL eval: 15/20 (75%) — remaining failures are column-name non-determinism
- Stress test: 19/19 (100%)
- Answer judge: 8/8 (100%) where judge quota was available

---

## Future Work

### Near-term (planned)

**Streaming responses** — Gemini supports `generate_content_stream()`. Currently the agent thinks for 3-5 seconds then dumps the full answer. Streaming shows words appearing as the model generates — dramatically better UX. Teaches async streaming patterns and SSE (Server-Sent Events) for the web UI.

**FastAPI + Web UI** ✅ Complete — FastAPI server with chat UI, `.fitnotes` file upload
with integrity validation, PDF article upload to RAG knowledge base, background
initialization, rate limit UX, and `--debug` mode. Run with `python server.py`.

**Write-ahead log + DB merge** — Currently agent-written workout data (logged via `log_workout`) lives in the SQLite DB. When the user exports a fresh FitNotes backup and uploads it, those agent-written sets would be overwritten. Fix: log every agent write to `agent_writes.json`. On new DB upload, replay the write log onto the new file before replacing the old one. Teaches transaction logging, SQLite conflict resolution, and data integrity across two sources.

### Analytical AI Coaching (prerequisites required)

The agent currently handles lookup queries well (PRs, recent workouts, volume
stats). Analytical questions — plateau detection, overtraining signals,
progression rate analysis — require additional groundwork before they can be
trusted:

- Memory auto-extraction rewrite: store insights, not conversation transcripts
- Schema prompt analytical patterns: teach the agent what aggregate queries are possible
- Context pruning: 5-month history queries return hundreds of rows that must not
  accumulate across API calls
- Reflection step update: verify the agent looked at enough data, not just unit correctness
- Gemini thinking budget > 0: complex multi-step reasoning benefits from thinking tokens

Enable only after simple lookups are fully verified — analytical answers cannot
be cross-checked against the raw data the way PR lookups can.

### Session & Memory Architecture (Pre-Deployment)

The current auto-extraction runs at CLI session end via the `finally` block
in cli.py. On the web server, sessions never terminate — the AgentSession
runs continuously and there is no natural trigger for extraction.

**Three options for web session management:**
- Option A — Inactivity timer: trigger extraction after X minutes of no
  /chat requests. Most production-like but requires asyncio background
  tasks with cancellation logic.
- Option B — Message count trigger: run extraction every N messages
  (e.g. every 10 exchanges) silently in the background. Simple, 5 lines.
- Option C — Explicit end session button: UI button triggers extraction
  before clearing context. User controls when the session ends.
- Recommended: Option B + Option C combined.

**Conversational memory architecture (three-layer):**
Currently the agent has short-term context (growing message history) and
long-term facts (memory.json, 30 fact cap). Missing middle layer:
- Short-term: last N messages (active context per request)
- Mid-term: session summaries — compressed LLM summaries of past
  conversations stored and retrieved at session start
- Long-term: extracted facts in memory.json (existing)

**Per-message delete button:**
Add a delete button under each message in the chat UI. Pressing it removes
that specific message from the agent's conversation history so it no longer
influences future responses. Gives the user explicit control over what the
agent remembers from the current session.

Implement all of the above before deploying to web — the growing context
window will hit quota limits and produce inconsistent behavior without it.

### Learning extensions

**LangGraph** — Rebuild the agent loop using LangGraph's state machine framework. The current hand-built ReAct loop in `agent.py` does exactly what LangGraph provides — but explicitly, without abstractions. Rebuilding in LangGraph will make the framework's design decisions immediately obvious: why nodes, why edges, why checkpointers. The right order was always: build from scratch first, then use the framework.

**Multi-Agent Systems** — Split the single agent into specialized agents: a query agent for read questions, a coach agent for trend analysis and advice, a logging agent for write operations. Agents communicate via a shared message bus. Teaches agent orchestration, message passing, and how to avoid the context accumulation problems that plague single large agents.

**Agent Memory with Knowledge Graphs** — Replace the flat `memory.json` fact store with a graph database (NetworkX locally, Neo4j for production). Store relationships between facts: "trained chest → leads to → shoulder fatigue → affects → overhead press performance." The agent can reason over connections, not just isolated facts. Teaches graph data modeling and relationship-aware retrieval.

**Computer Use for corpus updates** — Automate `build_corpus.py` by having the agent call the PubMed API directly (no browser needed — PubMed has a free API). The agent searches for new papers on a topic, fetches abstracts, embeds them, and adds them to ChromaDB. Scheduled weekly. The credibility problem is already solved — PubMed only indexes peer-reviewed literature. Teaches API-driven automation and scheduled agent tasks.

---

## Notes for Interviewers

This project deliberately avoids high-level frameworks (LangChain, LangGraph, LlamaIndex) to demonstrate understanding of the underlying components:

- The ReAct agent loop, context management, and tool selection are hand-built in `src/agent.py`
- The RAG pipeline (query rewriting, BM25, dense retrieval, cross-encoder reranking) is built from components in `src/rag.py`
- The MCP server protocol is implemented directly using the `mcp` Python SDK
- The confirmation gate for write operations is an explicit CLI-level intercept, not a framework feature
- Long-term memory uses ChromaDB for semantic retrieval — same vector search used for the knowledge base

Every architectural decision has a documented reason in `lessons.md` including what went wrong when the first approach was tried.

---

## License

MIT


