# FitNotes Personal Strength Coach

A full-stack agentic AI coaching system built over a personal FitNotes SQLite
database. It answers natural-language questions about training history, reasons
over a complete analytical package rather than a sampled subset, grounds every
number it reports against the source data, and can log, update and delete
workout data behind a human-in-the-loop confirmation gate.

It was built from scratch first — the ReAct loop, the RAG pipeline, the
confirmation gate and the context management were all hand-written with no
framework — and then deliberately converted to LangGraph once the problems those
abstractions solve had actually been hit. Both versions are preserved as
branches, which is the point: the diff between them is the lesson.

| Branch | What it is |
|---|---|
| `main` | stable baseline |
| `Single-Agent-System` | the hand-built single agent: web UI, MCP tools, RAG, memory, writes |
| `Multi-Agent-System` | **current** — LangGraph orchestration, Coordinator / Data Agent / Analysis Agent, muscle ontology, request decomposition |

**Scale:** 36 source modules, 27 MCP tools, a 39-muscle ontology over 175
curated exercise→muscle edges, and **1,533 tests**.

---

## What it does

- **Answers analytical questions over the complete dataset** — plateau
  detection, progression and e1RM curves, per-muscle coverage, push/pull
  balance, training consistency — with every exercise in the database analyzed,
  never a sampled subset.
- **Grounds every number it states.** The draft answer cites each factual claim
  against a specific field in the data package; a deterministic layer resolves
  those citations, and a grounding pass removes or qualifies anything the data
  does not support.
- **Knows which muscles a lift actually trains.** FitNotes allows one category
  per exercise, so "how much back work did I do" was silently wrong. A
  hand-curated muscle ontology fixes that, and a 3D anatomical viewer renders it.
- **Answers fitness-science questions** from a RAG pipeline over ~160 PubMed
  abstracts and Wikipedia articles, plus any PDFs the user uploads.
- **Logs, updates and deletes workout data** through stage → confirm → execute →
  verify, with the write committed by the host, never by the agent.
- **Handles compound messages** — "is my squat progressing, and log bench press
  100x5 today" is split, each part routed to its own lane, and the answers
  merged back in the order asked.
- **Remembers preferences** across sessions via ChromaDB-backed long-term memory.

---

## Architecture

```
                        User message
                             │
                    ┌────────▼────────┐
                    │ CoordinatorGraph│   LangGraph parent graph
                    │  (entry guards) │
                    └────────┬────────┘
                             │  one classify call → lane
        ┌────────────────────┼────────────────────┬──────────────┐
        ▼                    ▼                    ▼              ▼
   ANALYTICAL           OPERATIONAL            RECALL       OUT_OF_SCOPE
        │                    │                    │              │
        ▼                    ▼               restate a      canned refusal
  Data Agent           ReAct subgraph        prior figure    (no spend)
 (pure Python)         (MCP tools)           from history
        │                    │
        ▼                    ▼
 prepare_analysis_     stage → confirm →
     package()          execute → verify
        │
        ▼
  Analysis Agent  ──► grounding ──► coverage ──► display fidelity ──► answer
```

A message containing **two or more requests in different lanes** takes a fifth
path: it is split into chunks, each chunk runs its own lane with a
self-contained restatement of its intent, and the parts merge back under
headings in ask-order.

**Six things can decide a lane, and they are strictly ordered.** That ladder is
documented once, in [`docs/routing-rules.md`](docs/routing-rules.md) — read that
before adding a routing rule anywhere. It exists because the rule was previously
written twice, in English in the classifier prompt and in Python in the regex
gates, and the two had drifted into disagreement that was invisible from either
side.

### Orchestration (LangGraph)

The parent `CoordinatorGraph` dispatches by conditional edge to an
`OperationalSubgraph` (a ReAct loop with a custom Gemini-native tool node, max 7
iterations) or an `AnalyticalSubgraph` (a deterministic linear node chain).
`AsyncCompatSqliteSaver` works around a langgraph-checkpoint-sqlite 3.x
async/event-loop incompatibility. Expensive artifacts — the analysis package can
reach ~400 KB — are held in a non-persisted `RunCache` and excluded from
checkpoints, then rebuilt for free from simple inputs on resume.

### Data Agent (`src/data_agent/`) — pure Python, zero LLM calls

| Module | Responsibility |
|---|---|
| `fetch.py` | all SQLite access; typed raw rows, no interpretation |
| `process.py` | pure function `(raw_rows, user_context) → package`; no DB, no clock, no I/O |
| `validate.py` | independent post-condition checks; never recomputes, only asserts |
| `session_display.py` | pre-formatted per-set display lines |
| `__init__.py` | facades `collect()` / `prepare_analysis_package()` |

The validator runs on every package. Soft violations (size ceiling, scope leaks,
muscle visibility) log warnings; any integrity violation — wrong units, a PR
below a session max, negative weights — raises `DataAgentIntegrityError`, which
the Coordinator catches **before** the Analysis Agent is called. A wrong package
cannot reach the LLM. Invariants are named after the bug each one catches; see
[`docs/data_agent_spec.md`](docs/data_agent_spec.md).

Packaging is scope-aware, because a broad 365-day package started at 1.4 MB and
blew the free-tier quota:

| Scope | Trigger | Contents |
|---|---|---|
| `FOCUSED` | ≤ 3 named exercises | full detail — sessions, comments, all stat blocks |
| `GROUP` | muscle-group filter | comments capped to 30 most recent + all pain-flagged |
| `BROAD` | no filter | comments and deep stats dropped; one aggregation level |

Phase 2 depth is triggered by **Python constants, not model judgment**: a
plateau over 28 days pulls that exercise's full comment history, and a weight
change over 20% in *either* direction pulls full session detail — a 20% drop is
exactly when the comments matter most.

### Analysis Agent + grounding

One `thinking_budget=4096` call produces a draft in which every factual claim
carries an internal citation tag — `[[collection|match-key|field-path]]`. A
deterministic, no-LLM layer (`src/citations.py`) indexes the package, resolves
each tag, and strips the tags so the user never sees one. The model may only
cite leaves that exist: `build_citable_schema()` generates the whitelist **from
the live package**, so it cannot drift.

That buys the latency win. When every claim resolves cleanly, grounding receives
the draft plus a small extracted `[CITED VALUES]` block — measured at **0.6–8 KB
against a ~415 KB package, 50–600× smaller** — instead of a second full-package
send. If *any* claim is not cleanly cited, it falls back to the complete
full-package check. The fallback is the safety path and fires rarely, so it is
deliberately the complete check rather than an optimized subset.

Some invariants cannot be enforced by citation at all, because a citation proves
which field was read, never what the sentence means. Those get deterministic
post-draft guards — the recency guard, for instance, enforces the positive
invariant `date == latest_session_date` rather than blocklisting known-bad dates.

---

## Muscle ontology

`exercise.category_id` in FitNotes is a single NOT NULL column: one category per
exercise. Multi-muscle reality cannot be expressed, and the resulting answers
were already wrong — `Smith Machine Shrugs` (266 sets) filed under **Back** when
it is upper traps; `Reverse Cable Curls` (296 sets) filed under **Forearms**
when it is elbow flexion.

The store is plain CSV in [`ontology/`](ontology/README.md) — not a graph DB, not
a second SQLite file, and deliberately **not** inside the `.fitnotes` file, which
is wiped on every backup upload.

| File | Rows |
|---|---|
| `muscles.csv` | 39 — hierarchical, and a **closed set**. Every human has the same muscles. |
| `exercises.csv` | 77 canonical exercises |
| `exercise_muscle.csv` | 175 edges, each with mandatory provenance |
| `aliases.csv` | 77 exact DB-name → exercise bindings |

Edges carry one of **three roles**, and the distinction is the interesting part:

| Role | Count | Meaning |
|---|---|---|
| `primary` | 95 | the target of the lift |
| `secondary` | 65 | genuinely trained; interference runs **both** ways. Counts as volume. |
| `limiting` | 15 | **held, not trained**; interference runs **one** way. Never counts as volume, never as coverage. |

Grip on Smith Machine Shrugs is the case that forced the third role. Grip
genuinely caps how much you can shrug, but 266 sets of shrugs train the grip for
nothing. Calling the edge `secondary` credits false volume; cutting it loses a
real scheduling fact. So a muscle with 400 limiting sets and no primary or
secondary sets is **untrained**, and coverage says so — `limiting` licenses
exactly one kind of coaching statement, ordering, and never volume.

The bridge from user data to the ontology is an **exact alias lookup, never
fuzzy matching**: a near-miss would silently attribute hundreds of sets to the
wrong muscle. Unmapped exercises are named and visibly excluded, never dropped.
When a backup upload introduces new exercises, reconciliation auto-aliases only
on categorical evidence (normalised-exact, or a sole exact word permutation);
everything else queues for human review behind an approval gate.

### 3D anatomical viewer

`http://localhost:3000/graph` renders the ontology as a body where **the figure
is the graph** — each of the 39 nodes is its actual anatomical volume, so you
click the rear delt's real shape rather than a sphere hovering near it. The 9
grouping nodes are the union of their descendants. Rollup is computed
**server-side** and read by the viewer; the first version reimplemented it in JS
and diverged, overstating Back by 588 primary sets.

`scripts/preview_body.py` rasterises the same `figure.json` to an ASCII
silhouette — the feedback loop that was missing when three shipped defects in a
row turned out to be blind failures rather than thinking failures.

---

## Write safety

Two-phase throughout: **stage** (validate + preview) → **confirmation gate** →
**execute** (DB write) → **verify** (read-back).

For workouts the agent only *stages*. It holds no execute tool and no verify
tool; the host (`server.py` / `cli.py`) commits the staged slot after explicit
user confirmation, reads the inserted rows back **inside the same transaction**,
and rolls back unless every staged set is present. The agent stages, the host
commits, and the agent cannot bypass the gate. A full multi-exercise day stages
as one batch and writes in one transaction — all-or-nothing.

The confirmation panel renders the staged slot itself — the exact payload
`execute` will write — not a re-generated summary. Abandoning or cancelling a
pending log discards the batch, so it can never be carried into a later confirm.

Because the host executes outside any agent turn, the agent used to be left
believing its batch was still awaiting confirmation. `note_host_write` now
records the host's own verified outcome back into the agent's history, and a
success-claim gate scoped by a structural fact — *was a write tool actually
called this turn* — blocks completion claims that no write backs.

Every confirmed write is journaled to a write-ahead log, so uploading a fresh
FitNotes backup replays agent-written sets onto the new file instead of losing
them. Replay is user-controllable from the settings panel.

---

## Setup

**Prerequisites:** Python 3.11+, a `.fitnotes` backup exported from the FitNotes
app, and a free Gemini API key from [aistudio.google.com](https://aistudio.google.com).

```bash
git clone https://github.com/SwapnilAKale/FitNotes-Agentic-AI-Coach
cd FitNotes-Agentic-AI-Coach
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create `.env` in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING=1
```

Export your backup (FitNotes → Menu → Backup → Export Backup) to
`data/FitNotes_Backup.fitnotes`, then build the knowledge corpus once:

```bash
python scripts/build_corpus.py    # ~160 PubMed abstracts + Wikipedia, 2-3 min
```

### Running

```bash
python server.py            # API on :8000 + UI on :3000, opens a browser
python server.py --debug    # + tool calls, results and routing traces
python cli.py               # CLI, same pipeline
cd frontend && python server.py   # UI only — no agent, no quota burn
```

The 3D ontology viewer is at `http://localhost:3000/graph`.

```bash
python -m src.data_agent 90 "Lat Pulldown" "Deadlift"   # Data Agent standalone
python -m pytest                                        # 1,533 tests
```

---

## Tech stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`, conditional edges, SQLite checkpointer) |
| All LLM calls | `gemini-3.1-flash-lite` (free tier, 500 RPD) |
| Analysis Agent | `thinking_budget=4096` + citation-backed grounding |
| Coordinator | `temperature=0`, `thinking_budget=0` |
| Data Agent | pure Python, zero LLM calls |
| Tool protocol | MCP (Model Context Protocol) — one combined stdio server |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Database | SQLite (`.fitnotes`), read-only `mode=ro` connections for generated SQL |
| Web | FastAPI + uvicorn; vanilla HTML/CSS/JS frontend |
| 3D viewer | three.js, vendored for offline use |

---

## Project structure

```
├── src/
│   ├── graph/                 # LangGraph — coordinator_graph, operational,
│   │                          #   analytical, state, persistence
│   ├── data_agent/            # fetch / process / validate / session_display
│   ├── shared/                # resolver, memory, rag, sql_executor, sql_sanitize
│   ├── coordinator.py         # routing, classification, decomposition, pipeline
│   ├── analysis_agent.py      # analyze() + ground_check()
│   ├── citations.py           # citation resolution + deterministic answer guards
│   ├── agent.py               # operational ReAct session + MCP lifecycle
│   ├── ontology*.py           # ontology load / propose / reconcile
│   ├── prompt_blocks.py       # prompt text shared by both agents
│   ├── memory.py  demographics.py  wal.py  settings.py  units.py  checkpoint.py
│   └── rag.py  llm.py  db.py  schema_prompt.py  text_to_sql.py  stdio_utf8.py
├── mcp_servers/combined_server.py   # all 27 exposed MCP tools
├── ontology/                  # muscles, exercises, edges, aliases (CSV)
├── frontend/                  # index.html (chat), graph.html (3D), body/, vendor/
├── docs/                      # routing-rules, data_agent_spec, changelog
├── tests/                     # 76 files, 1,533 tests
├── evals/  scripts/  corpus/
├── server.py                  # FastAPI web server
└── cli.py                     # async CLI
```

## Tools (27 exposed)

| Group | Tools |
|---|---|
| Read | `resolve_exercise_name`, `search_fitness_knowledge`, `list_user_articles`, `delete_user_article` |
| Write — logging | `log_workout`, `log_bodyweight`, `discard_staged_writes` |
| Write — goals | `set_goal`, `execute_staged_goal`, `update_goal`, `execute_staged_goal_update`, `delete_goal`, `execute_staged_goal_delete` |
| Write — corrections | `update_workout_set`, `execute_staged_set_update`, `delete_workout_set`, `execute_staged_set_delete` |
| Verify | `verify_goal_set`, `verify_set_updated`, `verify_set_deleted` |
| Memory | `remember_fact`, `recall_memories`, `forget_fact` |
| Exercise quirks | `add_exercise_quirk`, `update_exercise_quirk`, `delete_exercise_quirk`, `list_exercise_quirks` |

Seven read tools — `get_personal_record`, `get_weekly_volume`,
`query_workout_data`, `run_read_only_sql`, `get_exercise_history`,
`read_exercise_comments`, `get_exercise_sessions` — were **unexposed, not
deleted**. Their reads now route to the analytical lane, where the validated
package answers them deterministically. There is one read path, not two. The
`_sync` handlers remain for evals and direct tests.

---

## Testing

```bash
python -m pytest                    # 1,533 tests
python evals/run_evals.py           # SQL correctness + LLM-as-judge
python evals/stress_test.py         # adversarial suite
```

The suite is layered deliberately:

- **Golden tests** pinned against a real DB snapshot — catch drift in real data.
- **Per-invariant defect injection** — synthetic packages that violate one
  invariant each, proving the validator actually fires.
- **Gate tables** (`test_routing_gates_golden.py`) — written to pass against
  *unmodified* code first, so a fix turns red exactly the rows it targets and
  nothing else.
- **Isolation fixtures at the conftest root** — every shared mutable file (DB,
  WAL, checkpoints, settings) is redirected per-test. This was added after an
  audit found the suite had written 1,025 WAL records and 77 rows into the real
  database over weeks of green runs. The proof a fix works is byte-equality:
  hash the real files, run the whole suite, hash again.

> **Eval scores:** the last recorded figures — SQL 15/20, stress test 19/19 —
> were measured in the **single-agent era** and predate the analytical lane and
> the muscle ontology. They have not been re-run and are **not** presented as
> current numbers.

---

## Currently in flight

Work in the working tree, not yet committed as of 2026-08-21:

- **Two-layer write integrity.** Five of six write operations had no read-back —
  `success: true` meant only "the SQL raised no exception", and a DELETE
  matching zero rows raises nothing. Layer 1 checks blast radius
  (`conn.total_changes` must move by exactly the intended row count); Layer 2
  checks identity (the focus table's added/removed/modified id-sets must equal
  exactly what was intended, every other table hashed whole and unmoved). Both
  run *inside* the transaction, so a rejection rolls back.
- **Ontology-backed answers.** Sets-per-week as the default dose measure rather
  than volume; `never_trained` vs `dormant` reported as the different facts they
  are; a `limiting_claim_guard` that stops a held muscle being described as
  trained; and a `plan_guard` that rejects a generated plan giving one muscle
  direct work on consecutive days.
- **Scope-filter intersection fix.** `exercise_names` and `muscle_groups` were
  ANDed as row filters, so "do my lat pulldowns train my biceps?" was empty by
  construction — Lat Pulldown is filed under Back. When exercise names are
  present they *are* the scope; a muscle group in the same sentence is the
  subject of the question, not a second narrowing.
- **Chat markdown rendering.** The analysis prompt has always asked for bold,
  headers and bullets; nothing rendered them, so they reached the user as
  literal asterisks for the entire life of the app.

## Known gaps

- **No token streaming** — answers return whole; long answers have no
  progressive render.
- **Session/memory architecture for web deployment** — memory extraction runs at
  CLI session end, and a web session never terminates. Needs a message-count
  trigger plus an explicit end-session control before deployment.
- **Layer 2 write integrity is not concurrency-safe** — a second writer's
  unrelated change would look like corruption. Correct under the single-user
  local assumption; must narrow to the transaction's own scope before multi-user.
- **`agent_lock` held during a per-minute rate-limit wait** — a concurrent
  request gets a "busy" response for up to ~2×70s. Same single-user assumption.
- **RAG personalization** — stored demographics never reach the answer context,
  so the personalization gate (Stage C) is blocked on it and deferred.
- **Research fetching is best-effort and uncached** — it can be slow or empty
  and still spends an API call.
- **Non-name write clarifications inside a decomposed chunk** (a missing date,
  say) stay agent-internal rather than surfacing as a structured panel.
- **Smith-counterbalance residual** — the operational volume path runs ~0.5%
  high on Legs; the analytical path applies the correction and it does not.

---

## Notes for interviewers

The interesting decisions in this project are mostly about **not trusting the
model**, and each has a documented reason — including what went wrong when the
first approach was tried:

- **The data layer makes no LLM decisions.** `process.py` is a pure function
  with no DB, no clock and no I/O. Phase 2 depth triggers are Python constants.
  Every "agent math" inconsistency disappears when the model's job is to reason
  rather than to compute.
- **A wrong package cannot reach the LLM.** The validator hard-stops on an
  integrity violation before the Analysis Agent is called.
- **Structural signals gate; prose never does.** A success claim is gated on
  whether a write tool was actually called, not on what the answer says.
  Attempt-time arming is a time bomb that detonates the day the action learns to
  say no.
- **Deterministic guards enforce what citations cannot.** A citation proves
  which field was read, never what the sentence means.
- **Build from scratch, then adopt the framework.** The hand-built ReAct loop
  came first; the LangGraph conversion came after, so the framework's design
  decisions were legible rather than magic. Both are on branches.

Design references live next to the code: [`docs/routing-rules.md`](docs/routing-rules.md),
[`docs/data_agent_spec.md`](docs/data_agent_spec.md),
[`ontology/README.md`](ontology/README.md).
The engineering lessons — stated as transferable principles, each with the bug
that produced it — are in [`lessons.md`](lessons.md).
The full session-by-session history is archived in
[`docs/changelog.md`](docs/changelog.md).

---

## License

MIT
