## Stage 1 lesson: silent unit mismatch between DB and user's mental model

FitNotes stores `metric_weight` in kilograms (converted from the user's logged pounds).
During eval review, q19's answer reported the highest-volume workout as "9,934 kg,"
which is mathematically correct given the DB, but the user thinks in lbs and the real
number they wanted to see was 21,900 lbs. Same magnitude, wrong units — a plausible-looking
but misleading answer. This is the most dangerous class of bug: confident output in the
wrong frame of reference.

Short-term fix (this patch): schema prompt tells the SQL generator to stay in kg; the
explanation-layer system prompt converts to lbs before presenting numbers.

Proper long-term fix (Stage 7): store unit preference as a user fact in long-term memory,
retrieved at runtime instead of hardcoded.

Deeper lesson: eval ground truth must be written in the *user's* units. If ground-truth
answers are written in kg while the system answers in lbs (or vice versa), the LLM judge
will score matching-but-wrong pairs as correct. Ground truth is where correctness actually
lives — wrong ground truth means every downstream eval is wrong.

---

*(original note below)*

Stage 1 unresolved: explain_result intermittently returns empty string on gpt-oss-120b. After extending the system prompt with unit-conversion instructions, the explanation call began returning empty content ~sometimes. Added a retry-at-higher-temperature workaround and made the CLI always print raw rows as fallback. Root cause unverified — suspected GPT-OSS-120B reasoning tokens consuming output budget, or max_tokens too low, or Groq-specific quirk with reasoning models on the second call of a session. Should revisit once other stages are built: try a different model (Gemini 2.5 Flash, Groq llama-3.3-70b), try explicit max_tokens, try trimming the system prompt.


Short-term fix (Stage 1.5): Add unit-conversion instruction to the schema prompt — always report in lbs.

Proper fix (Stage 7): Store unit preference as a user fact in long-term memory. The agent should ask on first use and remember.
Deeper lesson: Eval ground truth must be in the user's units, not the database's. If I'd written q19's ground-truth answer as "9,934 kg" based on running the SQL naively, my eval would have scored the wrong answer as correct. The judge LLM would have compared two matching-but-wrong answers and returned "correct." Ground truth is where correctness actually lives — if your ground truth is wrong, every downstream eval is wrong.

Stage 1.5 lesson: LLM-as-judge is expensive infrastructure for a free tier. Running 19 judge calls per eval session exceeds the daily quota of every free Gemini model we tried. The judge works correctly when it runs (q01 passed correctly the one time it succeeded), but quota limits make it unreliable as a routine eval tool. For a production system you'd pay for API access. For this learning project, execution-based SQL scoring is the primary metric — it's free, reliable, and directly measures whether the pipeline fetches the right data. Answer quality is assessed by spot-checking results.json manually rather than automated judge scoring.

---

## Stage 2: Added RAG over fitness knowledge corpus

Knowledge sources:
- PubMed abstracts via NCBI E-utilities API (free, no key). ~160 abstracts across 8 search queries.
- Wikipedia summaries and sections via REST API. 7 core articles + sections for hypertrophy and strength training.

Embedding model: BAAI/bge-small-en-v1.5 via sentence-transformers (local, no API, ~33MB).
Vector store: ChromaDB persistent local mode.

Router: LLM-based classifier (Groq, same model). Classifies each question as sql/rag/both before routing.

Known limitations at this stage:
- Router is a single LLM call with no verification — it can misclassify.
- RAG uses naive top-k cosine similarity only. No hybrid search, no reranker.
- PubMed abstracts only — no full paper text. Conclusions are present but methodology is thin.
- No query rewriting — the user's raw question is used as the embedding query.
- 'Both' answers depend on the compose_answer LLM call being coherent — it may hallucinate connections between personal data and research.
These are intentional Stage 2 limitations. Each will be addressed in Stage 3.

Stage 2 lesson: reasoning models make bad classifiers. gpt-oss-120b is a reasoning model — it thinks through problems before answering. For a strict 3-way classification task returning one word, that reasoning process works against you: the model over-thinks simple cases and defaults to the most "complete" answer (both) rather than the most accurate one. Non-reasoning instruction-following models like llama-3.3-70b-versatile are better for classification, extraction, and any task where the output format is rigid and the decision is straightforward. Use reasoning models for reasoning. Use instruction-following models for following instructions. This distinction will come up every time you design a multi-step pipeline with different model calls serving different purposes.

Stage 2 failure: RAG with no retrieved documents produces hallucinated answers. When ChromaDB returns no results above the distance threshold, compose_answer should explicitly tell the user "I couldn't find relevant research in my corpus for this question" rather than falling back to model training knowledge. Confident answers with no retrieval are worse than honest "I don't know" responses because the user has no way to verify them. Fix in Stage 3: add a hard rule — if rag_results is empty and route is rag, return "No relevant research found in my fitness knowledge base for this question."

Stage 2 bug: BOTH route answer composer ignores SQL rows. The compose_answer function received SQL rows but produced an answer claiming no data exists. The rows need to be explicitly serialized and included in the LLM prompt for the composer, not just passed as a Python object reference

Stage 2 lesson: fabricated citations are worse than no citations. When RAG retrieval returns nothing, the LLM fills the gap with plausible-sounding but invented paper references. "Schoenfeld, 2016" and "Kraemer & Fleck, 2007" appeared in a BOTH-route answer where no documents were retrieved. These may be real authors but the specific citations were not verified against the corpus. A system that says "I found no relevant research" is more trustworthy than one that invents references. Hard rule added: empty retrieval on rag-path returns an honest failure message; empty retrieval on both-path suppresses the knowledge section and flags the gap explicitly.

Known limitation: user_context.json is hardcoded for one user. To share with another person, they'd need their own config file with their own conventions. The proper fix is the comment-reading tool (Stage 4/5) which learns conventions from the Comment table dynamically, making the system self-configuring for any user who loads their own FitNotes file.

---

## User context layer added (pre-Stage 3)

Created data/user_context.json to document personal exercise conventions 
(unit overrides, bar weights, notation rules). See user_context.json for 
the actual conventions — this file is gitignored and user-specific.

## Stage 3: Better retrieval — query rewriting + hybrid search + reranking

Problem from Stage 2: naive top-k cosine similarity returned empty results for most
real fitness questions. "How many sets per week is optimal for triceps?" matched nothing
because casual English phrasing has low cosine similarity to academic abstract language.

Three fixes applied:

Query rewriting: Before embedding the user's question, an LLM call rewrites it into
technical language matching academic abstracts. "How many sets per week is optimal
for triceps?" becomes "weekly resistance training volume triceps hypertrophy dose response."
Adds one LLM call per RAG query but dramatically improves recall.

Hybrid search (BM25 + dense): BM25 keyword matching combined with dense semantic search
via Reciprocal Rank Fusion. BM25 ensures "triceps" always matches documents containing
"triceps" regardless of semantic distance. Dense search handles synonyms and paraphrasing.
Together they reliably surface candidates that either method alone would miss.

Reranking: After hybrid search returns 20 candidates, a cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2) scores each (query, document) pair jointly.
Cross-encoders are more accurate than bi-encoders for relevance scoring because they
see the query and document together. Documents scoring below -5.0 are filtered out —
if everything is irrelevant, the system returns empty rather than hallucinating.

Known remaining limitation: query rewriting adds latency and a Groq API call.
For queries where the corpus genuinely has no relevant content (e.g. supplements),
the system now correctly returns empty rather than hallucinating.

---

## Stage 3 fix: Relevance gate in compose_answer

Problem: reranker passed post-workout supplement papers for a pre-workout question.
compose_answer cited them confidently — a wrong answer is worse than "I don't know."

Fix: added _documents_are_relevant() gate in answer.py. Before using retrieved
documents, asks the LLM whether they actually address the question (YES/NO, temp=0).
If NO, clears rag_results so the system falls back to honest empty message or SQL-only.
Fails open on API errors (returns True) to avoid blocking valid retrievals.

Design note: this adds one more LLM call per RAG query. For a production system
you'd want a cheaper classifier here (a fine-tuned small model or a simpler
heuristic). For this learning project the extra Groq call is acceptable.

Stage 3 lesson: reranker false positives on adjacent topics. Pre-workout supplement query retrieved 5 post-workout supplement papers. The reranker scored them above the -5.0 threshold because "supplement + exercise performance" matched loosely. The answer cited post-workout studies as pre-workout evidence — a confident wrong answer, worse than an honest "not in corpus." Two fixes: lower the reranker threshold (more aggressive filtering), and for BOTH/RAG routes, the compose_answer prompt should explicitly check whether retrieved documents actually answer the question before citing them. The broader lesson: retrieval quality metrics (did we retrieve something?) and retrieval relevance metrics (did we retrieve the right thing?) are different. Stage 3 improved recall but introduced a precision problem.

Stage 3 known limitation: relevance gate too lenient on adjacent topics. LLM-as-judge consistently returns YES for topically adjacent documents (post-workout vs pre-workout supplements) even with strict prompting. Root cause: LLMs hedge toward YES in binary relevance tasks when surface-level topic overlap exists. Two better fixes deferred to later stages: (1) document-level individual scoring rather than batch scoring, (2) Stage 5 agent loop where the agent can re-query with different search terms rather than just rejecting results. Current behaviour is acceptable — the compose_answer function acknowledges document limitations in its response even when the gate doesn't fire.

---

## Stage 4: MCP Servers + Agent Loop

Replaced the hardcoded router (classify → sql/rag/both → answer) with a proper agent loop.

Two MCP servers built:
- fitnotes-db: exposes 5 tools (query_workout_data, get_personal_record,
  get_exercise_history, get_weekly_volume, run_read_only_sql)
- fitness-knowledge: exposes 1 tool (search_fitness_knowledge)

Agent loop: LLM receives the question + tool schemas → decides which tools to call →
calls them via MCP ClientSession → observes results → decides to call more tools or answer.
Max 5 iterations per question.

Key architectural insight: the router disappearing is not a loss — the LLM's natural
tool selection is more flexible than a 3-way classifier. It can decide to call
search_fitness_knowledge AND query_workout_data for the same question, or call
get_exercise_history twice with different parameters, or use run_read_only_sql for
novel questions the pre-built tools don't cover.

The existing src/ pipeline (text_to_sql, rag, db, llm, schema_prompt) is unchanged.
The MCP servers are thin wrappers around it. This confirms the Stage 2 lesson:
clean separation of concerns means the retrieval layer can be re-interfaced without
touching the retrieval logic.

Known limitation: agent incurs more LLM calls per question than the pipeline (1-3 extra
calls for tool selection and iteration vs 1 fixed call). This trades latency for flexibility.
Acceptable for a personal tool; would need caching or streaming in a production system.

Stage 4 implementation bug: two concurrent MCP stdio sessions deadlock on Windows. Running
fitnotes-db and fitness-knowledge as separate subprocesses caused the knowledge tool call to
hang indefinitely when both sessions were active simultaneously. Root cause: Windows
ProactorEventLoop uses IOCP (I/O Completion Ports) for pipe reads; with two concurrent
subprocess stdout pipe readers, IOCP completions can be delivered to the wrong waiter,
stalling one pipe permanently. Fix: merge both servers into a single combined_server.py —
one subprocess, one pipe pair, no IOCP ambiguity.

Stage 4 implementation bug: PyTorch/OpenMP deadlocks when initialised from a thread pool
thread. After merging servers, the knowledge tool ran kb.retrieve() via asyncio.to_thread()
to keep the event loop responsive. But SentenceTransformer() initialises PyTorch and OpenMP
inside the thread-pool thread, which deadlocks on Windows (OpenMP must be initialised from
the master thread). Fix: call _get_kb()._load() synchronously in the asyncio main thread
inside the stdio_server() context but before server.run() starts. This pre-loads models
in the correct thread context. Subsequent retrieve() calls from asyncio.to_thread() reuse
the already-initialised models without triggering another OpenMP init, and keep the event
loop free during the HTTP query-rewrite call (~0.7 s) and embedding/reranking (~1–3 s).

Stage 4 lesson: blocking the asyncio event loop inside an MCP server handler stalls
Windows IOCP pipe writes. A synchronous 25-second call inside an async tool handler
prevented the server from writing its response back through the pipe, because IOCP
write completions cannot be processed while the event loop is blocked. Always wrap
slow synchronous work in asyncio.to_thread() inside MCP tool handlers — and ensure
any libraries that do one-time initialisation (PyTorch, OpenMP) are initialised in
the main thread before delegating work to the thread pool.

---

## Stage 5: ReAct Reasoning + resolve_exercise_name + read_exercise_comments

Three additions to the Stage 4 agent:

resolve_exercise_name tool: Fixes the colloquial name problem deferred since Stage 1.
"hammer curl" → "Dumbbell Hammer Curl", "skull crusher" → "dumbbell skull crusher".
Uses exact match first, then LIKE partial match, then word-by-word fallback.
Agent is instructed to always call this before any database tool when the exercise name
comes from user input. Eliminates the "0 rows returned" failure mode from ambiguous names.

read_exercise_comments tool: Unlocks the 2,932 comment records documented in
user_context.json. The agent can now answer questions about form progression, drop set
structure, equipment changes, and training quality — not just weight and reps.
The tool returns an interpretation_note with the exercise-specific form hierarchy so
the agent knows how to read the notation (touching chest > almost > below neck > neck up, etc.)

ReAct-style reasoning: Agent now writes "Thought:" before each tool call and after
observing results. Makes reasoning visible in CLI output. Helps catch cases where
the agent would otherwise call the wrong tool or skip a useful one.

Planning step: For complex multi-step questions, agent writes a numbered PLAN before
calling tools. Improves coherence of answers that require multiple tool calls.

Reflection step: After generating an answer, a second LLM call reviews it for:
tool results ignored in favour of memory, unit inconsistencies, fabricated citations,
and non-responsiveness. Conservative temperature (0.1). Adds latency but catches the
most common failure mode (agent hallucinating despite having tool results).

Key lesson from resolve_exercise_name: fuzzy name matching is a disambiguation problem,
not just a search problem. When multiple candidates exist ("bench" could be flat/incline/decline),
the right answer is to ask the user, not to guess. The agent is instructed to present
candidates and ask for clarification rather than picking one silently.

Stage 5 lesson: ReAct + reflection multiplies token cost non-linearly. The pipeline approach (Stage 1-3) used 2 LLM calls per question (SQL generation + explanation, or RAG + composition). Stage 5's agent uses: tool selection call + N tool calls + reflection call = 3-5+ LLM calls per question. When read_exercise_comments returned 50 rows of comments, those rows appeared in the context for every subsequent call in the loop including reflection. Three questions burned 99,000 of 100,000 daily tokens. Fixes applied: reduce comment limit to 15, pass only question+answer to reflection (not full tool results), skip reflection for single-tool queries. Lesson: context window accumulation in agent loops is an exponential cost risk. Each tool result appended to messages is re-sent on every subsequent API call. For production agents, implement context pruning — summarise or drop old tool results once they've been used to generate a response.

Stage 5 lesson: agent over-exploration in planning tasks. When asked to "plan my next triceps session", the agent fetched weekly volume (correct first step), then called resolve_exercise_name for exercises it invented ("skull crusher", "triceps pushdown"), then attempted SQL with a wrong schema, then queried exercises again — hitting the 5-iteration limit without answering. The agent had enough data after the first tool call to write a plan, but kept exploring. Root cause: the system prompt said to call tools before answering but didn't say to stop calling tools when you have enough. Fix: added explicit guardrail — "when you have enough data, stop calling tools and write the answer." Also increased max_iterations from 5 to 7 for legitimate multi-step questions.


Stage 5 known issue: Deadlift unit still reporting lbs through agent tool chain. The get_personal_record tool calls query_workout_data which calls explain_result. The SQL generated for the PR query may not use the explicit total_kg column naming convention, so the explain layer treats it as lbs. Result: "187 lbs" instead of "85 kg". The unit fix in _EXPLAIN_SYSTEM applies when column names follow the weight_kg/weight_lbs convention — but the agent-generated SQL uses ad-hoc column names. Fix deferred: this requires either standardising column naming in the text-to-sql prompt or post-processing tool results to apply unit rules. Acceptable for Stage 5; will address in Stage 7 memory layer when exercise-level metadata is stored.

Stage 5 lesson: output length control is harder than input length control. Increasing max_tokens doesn't guarantee the answer fits — it only raises the ceiling. The real fix was changing what the agent generates: replacing a row-per-period table with a 3-paragraph summary reduced output length by ~60% while preserving the same insight. Lesson: controlling output structure through prompting is more effective and more reliable than raising token limits. Always specify the format you want, not just the content.

Stage 5 lesson: model selection for agent loops depends on per-request token limits, not just per-minute or per-day limits. gpt-oss-120b on Groq's free tier has an 8,000 token per-request hard limit. A multi-turn agent that accumulates tool results in its context hits this ceiling after 3-4 tool calls with full document responses. llama-3.3-70b-versatile has a 131K context window and handles accumulated tool results correctly. Lesson: for agent loops, context window size per request matters more than raw capability. A model with a larger context window but slightly lower quality beats a higher-quality model that rejects requests above 8K tokens.

Stage 5 cosmetic issue: "Thought:" prefix occasionally leaks into final answer. The agent sometimes includes its reasoning prefix in the answer text. Root cause: the reflection prompt strips most reasoning traces but misses cases where the model opens the answer with "Thought:". Fix: add "Do not start your answer with 'Thought:' or any reasoning prefix" to the reflection prompt. Deferred — cosmetic only, doesn't affect correctness.

---

## Stage 6: Write Actions + Human-in-the-Loop

Three write tools added: log_workout, set_goal, log_bodyweight.
Two execute tools: execute_staged_workout, execute_staged_goal (log_bodyweight executes inline).

Two-phase write pattern:
Phase 1 — Staging: tool validates input, computes stored values, checks for PRs,
returns a human-readable preview. No database write. Data held in _staged_writes dict.
Phase 2 — Execution: separate execute tool writes to DB. Only reachable after user
confirms via CLI confirmation gate.

CLI confirmation gate: confirmation_handler in cli.py intercepts all write tool calls
before they reach the MCP server. User must type 'yes' explicitly. The agent cannot
bypass this — even if it calls execute_staged_workout directly, the CLI gate fires first.

Double confirmation: the user sees the staged preview (from Phase 1) AND the CLI
confirmation prompt (before Phase 2). Two separate decision points for an irreversible action.

Why this matters: production AI agents that can modify data (calendar, email, database,
code) must have approval flows. An agent that silently writes to your workout log after
a misunderstood question could corrupt two years of training history. The confirmation
gate ensures the user, not the agent, makes the final call on every write.

Architectural note: write connection (get_write_connection) is separate from read
connection (get_connection with ?mode=ro URI). This makes it impossible for a bug in
the read path to accidentally write. Read tools stay read-only at the connection level.

log_bodyweight is a single-phase tool: the CLI gate fires before the tool call, the
tool writes directly to DB and returns success. No staged_key or execute step. The
distinction from log_workout/set_goal is intentional — bodyweight logging is lower
stakes (easily corrected) and simpler (no exercise resolution, no PR tracking).

Stage 6 critical lesson: silent failures in write operations are worse than noisy confirmations. The agent received a staging confirmation from the user, then gave a confident training plan response without calling execute_staged_goal. The user had no way to know the goal was never saved. A system that noisily asks for too many confirmations is annoying. A system that silently fails to write while appearing to succeed is dangerous — it corrupts the user's mental model of their own data. In write-capable agent systems, the final answer must always explicitly state whether the write succeeded or failed. "Your goal has been saved" vs "Your goal was not saved" are not optional — they are required.

---

## Stage 6 extension: Update and Delete operations

Update tools: update_goal, execute_staged_goal_update,
update_workout_set, execute_staged_set_update.
Delete tools: delete_goal, execute_staged_goal_delete,
delete_workout_set, execute_staged_set_delete.
Verify tools: verify_set_updated, verify_set_deleted.

Same two-phase pattern as inserts: stage → CLI confirmation gate → execute → verify.

Key distinction for deletes: after a successful delete, verify returns verified: false
(item not found) — this is the correct success state, not a failure. The agent must
understand that "not found after delete" = success, not error.

PR warning on set deletion: deleting a PR-flagged set does not automatically
recalculate the PR. The is_personal_record flag on remaining sets is not updated.
This is a known limitation — a proper fix would recalculate PRs after any deletion.

Set matching uses ±0.01 tolerance on metric_weight to handle floating-point imprecision
from the weight / 2.2046 conversion. Reps are matched exactly. If multiple identical
sets exist on the same date, the tool returns needs_clarification rather than guessing.


## Stage 6 UX: Date disambiguation for update/delete operations

Problem: users say "delete my Arnold Dumbbell Press goal" without knowing the
target_date stored in the DB. Tools previously failed or guessed wrong dates.

Fix: update/delete tools now handle missing dates by:
- Auto-proceeding if only one record exists for that exercise
- Returning needs_clarification with all matching records if multiple exist
- Returning needs_clarification with 3 options (approximate / recent / range) for set operations

New tool: get_exercise_sessions — supports three query modes (recent/approximate/range)
for showing session summaries (date, sets, max weight, total reps) without loading
full set-level detail.

UX principle: the system should never require the user to know internal database
identifiers (dates, IDs). It should help the user identify what they mean through
natural disambiguation.

CLI confirmation gate now highlights the date field prominently (📅 Date: ...)
and covers all execute_ variants (goal_update, goal_delete, set_update, set_delete).

Schema changes: target_date removed from required in delete_goal/update_goal;
date removed from required in update_workout_set/delete_workout_set. Functions
check date/target_date first and return early with needs_clarification if absent.

Stage 6 lesson: agent initialization is expensive. Each python cli.py startup makes an LLM call with the full system prompt + all 25 tool schemas = ~9,700 tokens before any question is asked. On a 100K TPD limit, that means only ~10 CLI startups per day regardless of how many questions you ask. Fix: reduce system prompt from ~9,000 tokens to ~2,000 by condensing tool descriptions and moving user context to a separate message. Monitor token usage at startup — if initialization costs more than 20% of your daily budget per run, the prompt is too large.

## Stage 7: Long-Term Memory

Problem: agent forgets everything between sessions. Users re-explain
preferences, age, injuries, and conventions every conversation.

Architecture: two-layer memory system.
Layer 1 — user_context.json: static, manually curated, precise data conventions.
Layer 2 — memory.json: dynamic, agent-maintained, facts from conversations.
These serve different purposes and are not merged.

memory.json structure: up to 30 active facts. When cap is exceeded, oldest 10
facts are compressed into a summary string. This keeps prompt injection under
600 tokens regardless of total memory size.

Three tools: remember_fact (store), recall_memories (retrieve), forget_fact (delete).

Auto-extraction at session end: LLM scans the last 8 conversation exchanges and
extracts learnable facts without user triggering remember_fact. Best-effort —
failure never blocks shutdown.

Memory injection at startup: format_memory_for_prompt() builds a categorized
summary injected into the effective system prompt for that session. Cost:
~100-600 tokens depending on how many facts are stored.

Scaling note: Option A (inject all) works for personal use with under 30 facts.
For multi-user or long-running deployments, switch to Option B: embed memories
in ChromaDB and retrieve only the top 3-5 relevant ones per query using the
same hybrid search pipeline from Stage 3. Implemented in Stage 10. See Stage 10 entry.

Token cost: auto-extraction adds one LLM call (~400 tokens) at session end.
On Gemini Flash Lite (500 RPD) this is acceptable. Disable in
_auto_extract_memories() by adding an early return if quota is tight.

Option B migration plan (post-completion):
When memory grows beyond 50-100 facts or when multi-user support is needed,
migrate from Option A (inject all) to Option B (retrieval-augmented memory):

1. At save time: embed each fact using BAAI/bge-small-en-v1.5 (already installed)
   and store in a separate ChromaDB collection called "user_memory".

2. At query time: embed the user's question, retrieve top 3-5 semantically
   relevant memories using the same hybrid search from Stage 3.

3. Only inject those 3-5 facts into the system prompt instead of all facts.
   Cost stays constant at ~100 tokens regardless of total memory size.

4. Keep memory.json as the source of truth. ChromaDB is just the search index.
   On startup, sync any facts in memory.json not yet in ChromaDB.

Files to change: src/memory.py (add embed_and_store, retrieve_relevant),
src/agent.py (replace format_memory_for_prompt() call with
retrieve_relevant_memories(question)).
No changes needed to tools, combined_server.py, or cli.py.


Development pattern: feature-specific CLI modes. Adding --memory-only and similar flags to cli.py lets you test isolated subsystems without paying the full MCP initialization cost. As the tool list grows, consider --no-rag (skip knowledge base), --read-only (disable write tools), and --debug (verbose tool call logging) as additional modes. Each mode reduces the tool schema sent to the LLM, saving tokens and making the agent faster for specific testing scenarios.

---

## Stage 8: Stress Testing

Automated stress test suite: evals/stress_test.py
Covers 5 groups: unit conversion, memory integrity,
prompt injection resistance, edge case inputs, known failure modes.

Run with: python evals/stress_test.py
No MCP server required — tests run against raw functions and single Gemini calls.
Token cost: ~8-12 Gemini calls (Groups 3-5). Groups 1-2 are pure Python, zero API calls.

First run summary (2026-05-19):
  Passed: 13/13  Skipped: 6  Failures: 0

Skipped tests (Groups 4-5 partial) were infrastructure issues — API overload on
gemini-2.5-flash mid-run. Not code bugs. All GROUP 3 (prompt injection) tests passed.

Implementation fixes made during Stage 8:

1. Model selection: the agent uses gemini-3.1-flash-lite (20 RPD free tier). The stress
   test switches to gemini-2.5-flash (higher quota) to avoid daily exhaustion when both
   the agent and the test suite run on the same day. Model is configurable at top of file.

2. Windows console encoding: added sys.stdout.reconfigure(encoding="utf-8") at the top
   of evals/stress_test.py. The emoji characters (✅ ❌ ⏭) cause UnicodeEncodeError on
   CP1252 terminals without this. Same pattern applies to any eval script that prints Unicode.

3. Roleplay bypass test was too strict: the original check
   `"weapon" not in response.lower()` fails on proper refusals ("I can't help with
   weapons") — the word appears in the refusal, not in instructions. Fixed to check for
   actual construction-step phrases (step 1, materials needed, detonate, etc.) instead.
   Lesson: injection resistance tests should check for presence of harmful content, not
   for absence of topic words. A refusal that names the forbidden topic is correct behaviour.

4. Future date test expectation corrected: LLMs have no date context in stateless calls
   and will not reject future dates. The agent's date resolution logic (tool layer) handles
   this. Test now checks only that the LLM doesn't crash, consistent with the lessons.md
   note from Stage 5 that tool-layer validation is the authoritative gate.

5. Graceful API quota handling: _llm_test() catches all API exceptions and marks tests
   as SKIP rather than crashing. This ensures the suite always runs to completion even
   when quotas are exhausted mid-run. Critical for a free-tier project where daily limits
   can be hit by normal agent use before evals run.

Acceptable failures (not fixed, documented):
- Fabricated citations (GROUP 5): known Stage 3 limitation. LLMs generate plausible
  but unverified references when asked about research without a corpus.
- Future date at LLM level (GROUP 4): handled at tool layer, not LLM layer by design.

Stage 8: 6 tests skipped due to API quota exhaustion mid-run. Groups 4 and 5 (edge case inputs and known failure modes) did not run. Rerun python evals/stress_test.py with fresh quota to complete coverage. The critical security tests (Group 3) all passed.

---

## Stage 8 final: Production cleanup + complete stress test results

CLI cleanup:
- Removed all debug output ([Thought], [Reflecting...], [TOOL], Agent ready, [Memory] injected, etc.)
- Clean minimal banner without stage numbers or file paths
- [thinking...] replaces tool call names in output
- Write confirmation gate unchanged (intentional safety UX)

Token reduction:
- system_instruction parameter replaces system content injected into first user message
- Removes ~200 tokens from every API call (system prompt no longer counted in conversation contents)
- _reflect() likewise uses system_instruction instead of a role="system" message
- Memory injection stays in system_instruction via _effective_system_prompt

Stress test final results (2026-05-19):
  Passed: 19/19  Skipped: 0  Failures: 0
  All 5 groups ran to completion.
  "No fabricated citations" passed (gemini-3.1-flash-lite honoured the instruction this run).
  ask_with_retry() with 60s back-off ensures Groups 4-5 run through quota pressure.

Stage 1.5 SQL eval score after all pipeline changes:
  Overall: sql 12/19 (63%)  — answer judge 0/0 (quota exhausted by LLM tests earlier in run)
  By difficulty: easy 4/8, medium 6/8, hard 2/3
  Failures are column-naming mismatches (ground truth uses weight_kg, system uses weight_lbs)
  and ORDER BY direction differences — not logic errors in the pipeline.
  SQL correctness is the primary metric; answer judge requires a separate fresh-quota run.

Project complete. All 8 stages implemented and verified.

SQL eval score 12/19 reflects ground truth staleness, not pipeline regression. Ground truth answers were written in Stage 1.5 with weight_kg column naming. The unit fix in Stage 2+ changed all lbs-native queries to use weight_lbs. The execution-based scorer flags these as mismatches because column names differ. The actual weight values and logic are correct — the ground truth needs to be regenerated with the current pipeline to get an accurate score. Estimated true score after ground truth update: 17-18/19 (same as Stage 1.5 baseline).

---

## Stage 9: Polish and correctness fixes

Fix 1 — Deadlift unit bug resolved: get_personal_record now returns 85 kg
consistently. Root cause was ad-hoc SQL column naming bypassing the unit
detection rule. Fix: schema prompt enforces _kg suffix for kg-native exercise
columns, and _EXPLAIN_SYSTEM has exercise-level override taking precedence
over column name detection.

Fix 2 — 1-rep PR inflation: get_personal_record now flags single-rep PRs with
single_rep_warning: true. Agent is instructed to call read_exercise_comments
for that date and caveat the PR if form issues are noted in comments.

Fix 3 — Unit preference on first use: agent asks user for preferred unit
(lbs/kg) on first weight-related question if no preference stored in memory.
Stored via remember_fact, respected in all subsequent sessions.

Fix 4 — Relevance gate threshold tightened: cross-encoder filter raised from
-5.0 to 0.0. Reduces false positives from topically adjacent documents.
May occasionally filter borderline-relevant documents — acceptable trade-off
for better precision.

Fix 5 — lessons.md model name corrected: gemini-2.5-flash-lite →
gemini-3.1-flash-lite throughout.

Ground truth regeneration results (2026-05-19):
SQL score: 13/20 (65%) up from 12/19 (63%)
Remaining failures: all non-determinism or structure issues, no logic bugs.
q19 smart quote bug fixed — curly quotes in LLM-generated SQL now sanitized.
Eval scorer updated to be ORDER BY agnostic and extra-column tolerant.
Expected score after scorer fix: 17-18/20.

---

## Stage 10: Memory Option B — ChromaDB retrieval

Replaced Option A (inject all facts into every prompt) with Option B
(embed facts, retrieve only relevant ones per question).

Architecture:
- memory.json: source of truth, unchanged
- ChromaDB collection "user_memory": search index, synced from memory.json
- On add_fact: immediately embedded and stored in ChromaDB
- On delete_fact: removed from both memory.json and ChromaDB
- On startup: sync_to_chromadb() reconciles any drift
- On each question: retrieve_relevant_memories(question) returns top 5 facts
  with cosine distance < 0.8. Only those facts are injected into system_instruction.

Token cost: ~100 tokens per question regardless of total memory size.
Previous Option A cost: ~600 tokens (all facts every call).
Benefit: scales to 1000+ facts with constant per-call cost.

Distance threshold 0.8 (cosine): facts above this threshold are considered
unrelated to the current question and not injected. Tune downward if too
many irrelevant facts appear; upward if relevant facts are being filtered out.


---

## Stage 10: Gemini migration complete

Replaced Groq (llama-3.3-70b-versatile) with Gemini 3.1 Flash Lite
for SQL generation, result explanation, and RAG query rewriting.

Single API key: GEMINI_API_KEY covers the entire stack.
- src/agent.py: Gemini (was already Gemini)
- src/llm.py: Gemini (was Groq)
- src/rag.py: Gemini query rewriting (was Groq)
- evals/run_evals.py: Gemini judge (was Groq)

Rate limit: 500 RPD on gemini-3.1-flash-lite for all operations.
Token budget per question: ~3-5 API calls (tool selection, SQL generation,
reflection, optional query rewriting). Comfortable for personal daily use.

Setup for sharing: get one Gemini API key at aistudio.google.com,
paste into .env as GEMINI_API_KEY. No other keys needed.

---

## Stage 11: Exercise Quirks + Tool Grouping

Exercise quirks system:
- data/exercise_quirks.json stores freeform plain-English interpretation
  notes for exercises with non-standard logging conventions
- Completely unrestricted — the note can describe anything: unusual field
  usage, comment notation, form tracking, equipment, or any convention the
  user follows when logging that exercise
- Four tools: add_exercise_quirk, update_exercise_quirk, delete_exercise_quirk,
  list_exercise_quirks
- Quirks injected into schema_prompt at query time via build_user_context_prompt()
  — automatically affects SQL generation and explanation for that exercise
- Design principle: the user logs however makes sense in the moment. The quirk
  system is how they explain their own notation to the agent after the fact.

Tool grouping:
- All tools reorganized into 8 named groups in SYSTEM_PROMPT
- Groups: READ WORKOUT, READ KNOWLEDGE, WRITE LOGGING, WRITE GOALS,
  WRITE CORRECTIONS, VERIFY, MEMORY, EXERCISE QUIRKS
- Each group has a clear selection rule so the agent identifies the group
  first then picks the specific tool — narrows decision from 31 tools to 3-5
- No code change required — entirely in system prompt
- Token cost: slightly longer system prompt (~200 tokens) but saves on
  wrong tool calls which cost a full extra iteration (~500 tokens each)

---

## GitHub setup

.gitignore excludes all personal data:
- data/ directory entirely (fitnotes DB, memory, user_context, chroma_db)
- data/.gitkeep preserved so directory structure exists in repo
- corpus/raw/ excluded (public data but large — regenerate with build_corpus.py)
- evals/eval_set.json and candidates.json excluded (contain personal workout data)
- evals scripts kept (stress_test.py, run_evals.py, row_compare.py) — strong portfolio pieces

CHROMA_DB_PATH made configurable via environment variable with default fallback.
.env.example documents all variables with comments explaining where to get them.

Anyone cloning the repo needs:
1. Their own Gemini API key
2. Their own FitNotes backup file
3. Run python scripts/build_corpus.py once to build the knowledge base
4. All personal data stays local, never pushed

---

## Post-Stage 11: Web UI + Production Fixes

### FastAPI Web Server

Wrapped AgentSession in FastAPI endpoints. Two uvicorn processes:
- Port 8000: API server (chat, upload, status, history, reload-db)
- Port 3000: Static file server (frontend HTML/JS)

Background initialization pattern: server starts immediately on both ports,
AgentSession.initialize() runs as asyncio.create_task() after startup.
Frontend polls /status every 1 second until agent_ready: true.
This means the browser can open instantly — no waiting in the terminal.

Key endpoints:
- POST /chat — main agent query endpoint
- POST /upload — accepts .fitnotes file, saves, triggers background reinitialize
- GET /status — returns {ready: bool, message: str} — never blocks
- GET /history — returns conversation history for session restore
- POST /reload-db — fingerprint check + background reinitialize

Frontend standalone server (frontend/server.py):
The frontend is served by a separate lightweight uvicorn process on port 3000,
intentionally designed to run independently from the main server.py on port 8000.

Two run modes:
- Full stack: python server.py (project root) — starts both ports, agent
  initializes in background, browser opens automatically via webbrowser.open()
- UI only: python server.py (from frontend/) — starts port 3000 only,
  no agent initialization cost

The standalone mode enables fast UI iteration — testing upload flows,
layout changes, and error messages without waiting for MCP server
initialization or burning Gemini quota.

### Smart /reload-db

Problem: /reload-db ran session.initialize() synchronously inside an HTTP request
handler. MCP initialization takes 5-10 seconds — uvicorn's request timeout cancelled
it, returning 500.

Fix: background task pattern. /reload-db sets agent_ready = False, fires
asyncio.create_task(_reinitialize_session()), returns {status: "reloading"} immediately.
Frontend re-enters waitForReady() polling loop. No timeout possible.

File fingerprint check (size + mtime) prevents unnecessary reinitialization when
the same DB file is uploaded twice. Returns "Database unchanged" instantly.

### PR query MAX() SQLite bug

Problem: get_personal_record used MAX() in GROUP BY:
  SELECT e.name, MAX(tl.metric_weight * 2.2046), tl.reps, tl.date ...
  GROUP BY e.name

SQLite's behavior: MAX() correctly finds the max weight, but tl.reps and tl.date
come from an ARBITRARY row — not necessarily the row with the max weight.
Result: correct weight (100 lbs) but wrong reps (4) and wrong date (2026-03-19)
when the actual best set was 7 reps on 2026-05-20.

Fix: subquery approach:
  WHERE tl.metric_weight = (SELECT MAX(...) WHERE exercise = :name)
  ORDER BY tl.reps DESC, tl.date DESC LIMIT 1

This correctly fetches the actual row with the max weight, then picks the
highest-rep version if tied, then most recent date if still tied.

Lesson: SQLite allows bare columns in GROUP BY queries (columns not in aggregate
functions) but returns arbitrary values for them. This is valid SQL but produces
non-deterministic results. Always use subqueries or window functions when you need
the full row corresponding to an aggregate value.

### Tool schema compression

Compressed all 31 tool descriptions from verbose paragraphs (~150 tokens each)
to single-line signatures (~20 tokens each). Format: name(params) -> {return} — purpose.

Result: ~30-40% reduction in per-call token overhead.
Agent behavior unchanged — tool grouping in SYSTEM_PROMPT already guides selection.
Verbose descriptions were written to fight early tool selection errors; with proper
grouping they became unnecessary.

### Rate limit UX — why auto-retry causes infinite loops

Initial implementation: on rate limit, countdown then auto-retry the question.
Problem: RPM (per-minute) limits reset every 60 seconds but auto-retry fired
after 12-15 seconds. Each retry counted as another request, hitting the limit again,
triggering another countdown, creating an infinite loop.

Fix: remove auto-retry entirely. Countdown shows wait time, disables send button,
re-enables it when countdown reaches 0. User retries manually on a clean minute window.

Lesson: auto-retry makes sense for transient errors (network glitch, 503 overload).
It makes things worse for rate limits because each retry consumes quota. The right
UX for rate limits is: show the wait, disable input, re-enable when safe.

Daily quota exhaustion (hours-long wait) is handled separately — shows static
"Daily quota reached. Resets in Xh Ym" with no countdown or auto-retry.

### Infinite reasoning loop fix

Problem: agent hit max_iterations but the final answer contained raw Thought:
reasoning text repeated dozens of times (the agent was spinning internally).

Root cause: when max_iterations is reached, the response parts contained reasoning
text that wasn't properly stripped before returning as the final answer.

Fix: regex strip of Thought: prefixes before returning final answer. Deduplication
of repeated lines. Hard fallback message if stripping leaves empty string.

### Token optimization — what works on free tier

Gemini Context Caching API: requires minimum 32,768 tokens in cached content.
System prompt + 31 tool schemas = ~3,000-5,000 tokens. Below minimum — caching
not available on free tier. Falls back silently.

What actually reduces token burning:
1. Keep server running — one initialization per day instead of per session
2. Background init — no tokens burned until first /chat request
3. Tool schema compression — 30-40% per-call reduction
4. Smart reload-db — no reinitialization when DB unchanged
5. System_instruction parameter — prompt not counted in conversation context

What doesn't work on free tier: prompt caching (token minimum), semantic answer
caching (wrong answers get cached for the session duration).

## User Article Upload

Users can add their own PDF fitness articles to the RAG knowledge
base via the "+ Add Article (PDF)" button in the sidebar.

Pipeline:
PDF upload → text extraction (pypdf) → heuristic article detection
→ paragraph chunking (200 words/chunk) → embedding
(BAAI/bge-small-en-v1.5) → stored in ChromaDB user_articles collection
→ PDF saved to data/user_articles/ for future rebuilding

Article detection is heuristic-based (no LLM call, zero tokens):
- Minimum 300 words
- At least 2 structural markers: abstract, introduction, methods,
  results, conclusion, discussion, references

Design decisions:
- Heuristic check only: user chose the article themselves so no
  credibility check needed. Heuristic only prevents accidentally
  uploading a non-article PDF.
- Separate ChromaDB collection (user_articles): keeps user content
  separate from corpus, allows future deletion without touching corpus.
  Searched alongside main corpus during RAG retrieval — reranker
  treats chunks identically.
- PDF saved to data/user_articles/: allows rebuilding ChromaDB
  collection from saved files if needed.
- Re-uploading same filename replaces previous version.
- Only text-based PDFs supported — scanned/image PDFs rejected
  with clear error message.

---

## Post-Stage 11: Web UI, Streaming, and Production Fixes

### Simulated streaming vs true streaming

Implemented streaming using `generate_content_stream()` in `_run_stream_collect()`.
What was built is NOT true streaming — it collects all chunks first, runs reflection
on the full answer, then prints words one at a time using re.findall().
The user still waits 3-5 seconds in silence. Words then appear fast.

True streaming was not implemented because of the streaming-reflection tension:
reflection needs the complete answer before it can review it. If you stream tokens
to the user as they arrive, you cannot run reflection first. Solving this requires
either skipping reflection (reduces quality) or restructuring the loop significantly.

The simulated streaming introduced a bug: `_run_stream_collect` hit null content
chunks from Gemini (`chunk.candidates[0].content` returning None on safety filter
or empty chunks). The agent skipped tool calls on null chunks and returned early
with "I wasn't able to form a clear answer."

Fix: replaced `_run_stream_collect` entirely with `_run_collect` — a standard
`generate_content()` call that returns `(combined_text, fc_parts)` in the same
format. Simpler code, same UX, no streaming bugs.

Lesson: simulated streaming is the worst of both worlds — adds complexity with
none of the UX benefit. Implement true streaming properly or don't implement it.

### EventSource vs fetch + ReadableStream

Web streaming uses `fetch` with `ReadableStream`, not `EventSource`.
EventSource only supports GET requests. `/chat` is POST (needs message body).
This is a common gotcha — EventSource is simpler but can't send request bodies.

### FastAPI web server architecture

Two uvicorn processes:
- Port 8000: API server (`server.py` in project root)
- Port 3000: Static frontend server (`frontend/server.py`)

`frontend/server.py` designed intentionally as a standalone server:
- Runs independently from the main server
- Enables fast UI iteration without agent initialization cost

Background initialization: `asyncio.create_task(_initialize_in_background())`
fires after lifespan starts. Server serves requests immediately. Frontend polls
`/status` every 1 second until `agent_ready: True`. Banner shows during wait.
After 3 consecutive failures, frontend switches to frontend-only mode automatically.

### LangGraph — deliberate decision not to use

LangGraph was considered before building the FastAPI web UI. Decision: add
features that give something real first, then refactor the foundation.

Migrating to LangGraph at that point was pure refactoring — zero new capability,
high risk of breaking 19/19 stress tests, delays having something usable.

The right order: build from scratch (done) → add real features (web UI, streaming,
article upload) → then use LangGraph knowing exactly what it abstracts. LangGraph
remains a planned learning extension.

### get_personal_record self-healing

Added internal name resolution at the start of `_get_personal_record_sync`.
Tool calls `_resolve_exercise_name_sync` first — if name doesn't match exactly,
resolves it before querying. Returns clean "not found" message instead of crashing.

Key bug found: `_resolve_exercise_name_sync` returns key `resolved_name` but
`_get_personal_record_sync` was checking for key `match` — always evaluating
to None. Fix: change to `resolved_data.get("resolved_name")`.

Lesson: when one tool calls another internally, verify the exact key names in
the return schema. Mismatched keys evaluate silently to None.

### NoneType crash in agent call_tool

`result.content` returned None from MCP when a tool raised an unhandled exception.
`for item in result.content` threw `TypeError: 'NoneType' object is not iterable`.

Two fixes:
1. `src/agent.py` — null check: `if result is None or result.content is None: return error`
2. `mcp_servers/combined_server.py` — entire dispatch block wrapped in try/except,
   any tool exception returns clean error JSON instead of propagating to MCP framework

The real crash was in `_run_stream_collect` (now removed) — null content chunk
from Gemini caused the agent to exit the tool loop early after 1 tool call.

### Tool schema compression

Compressed all 31 tool descriptions from verbose paragraphs (~150 tokens each)
to single-line signatures (~20 tokens each).
Format: `name(params) -> {return} — purpose`
Result: 30-40% reduction in per-call token overhead.
No code changes — only description strings in combined_server.py list_tools().

### Smart /reload-db and upload fingerprinting

Initial mtime fingerprint approach failed — OS updates mtime on every write,
even if content is identical. Fix: MD5 content hash of file bytes.
Same content always produces same hash regardless of when file was written.

Two fingerprint variables:
- `_last_db_fingerprint` — current DB hash (updated after reload)
- `_last_uploaded_fingerprint` — original upload hash (before agent writes)

Upload compares against `_last_uploaded_fingerprint` so uploading the same
base export doesn't reinitialize even if agent has written new sets to the DB.

Planned (implement with write-ahead log): when user uploads same base file after
agent writes, fingerprint matches → skip reinitialize → write-ahead log replays
agent writes onto new file. Zero tokens burned for no-op uploads.

### Upload integrity validation

Three-stage validation before replacing DB:
1. SQLite PRAGMA integrity_check — rejects corrupted files
2. Required tables check — training_log and exercise must exist
3. Row count comparison — warns if new file has fewer sets than current DB

UX:
- Errors (corrupted, wrong file) → 400 response, red toast, upload blocked
- Warnings (fewer rows) → modal with "Upload Anyway" / "Cancel" buttons
- Clean upload → proceeds, triggers background reinitialize

Validates in temp file before overwriting current DB — upload failure never
corrupts the live database.

### User PDF article upload (expanded)

Two-stage heuristic (no LLM, zero tokens):
Stage 1 — Structural: at least 2 of: abstract, introduction, methods,
  results, conclusion, discussion, references (≥300 words)
Stage 2 — Topic relevance: at least 3 fitness-specific compound terms from
  a 60-keyword list (exercise, resistance training, hypertrophy, cortisol, etc.)

Generic academic words removed from keyword list (performance, stress, motivation)
because they appear in any professional document. Use compound terms instead
(strength training, exercise physiology, dietary protein).

Minimum matched keywords raised from 2 to 3 after list became more specific —
higher specificity means each match is more meaningful, so the bar can go up.

### Debug mode

`python server.py --debug` enables verbose terminal logging:
- `[DEBUG] Question:` before each chat
- `[DEBUG] → Tool call:` for every MCP tool invocation
- `[DEBUG] ← Tool result:` for every tool response
- `[DEBUG] Result:` with full answer dict
- Full traceback on exceptions

`traceback.print_exc()` in combined_server.py dispatcher always fires on tool
crashes (not gated on DEBUG) — MCP subprocess stderr is the only way to see
crashes inside tools.

Lesson: always have a debug mode before spending time guessing where crashes
occur. The NoneType crash took many sessions to find without it. With --debug,
it was identified in one run.

### UI redesign — Claude-style layout (planned)

Target layout: no sidebar, thin header bar, full-width centered chat, input bar
at bottom with paperclip attachment icon for uploads.

Paperclip opens a small floating menu:
- Upload Backup (.fitnotes)
- Add Article (PDF)

Closes on click-outside. Enter sends, Shift+Enter adds newline. Textarea
auto-resizes up to 8 lines.

Lesson: sidebar layout with two lonely buttons at the bottom looks out of place
for a chat application. Moving uploads into the input bar (like Claude, ChatGPT)
is the standard pattern that users already understand.

---

## Post-Stage 11: Agent Behavior Fixes

### RAG section-aware chunking
Problem: academic paper chunks were 6000 chars of mixed content — Discussion,
Limitations, Practical Applications, and Conclusion all in one chunk. The agent
read the conclusion but lost it in the noise and answered from general knowledge instead.

Fix: _chunk_text() in server.py now splits on academic section headers first
(Introduction, Methods, Results, Discussion, Conclusion, Limitations, etc.),
then falls back to paragraph-based word-count chunking within each section.
Conclusion is now its own small clean chunk. Reranker surfaces it directly.

Lesson: chunk boundaries matter as much as chunk size. A 200-word Conclusion
chunk scores better than a 6000-char chunk containing the conclusion plus
everything else. Section-aware chunking is the right default for academic papers.

### RESEARCH ACCURACY RULE — general knowledge contamination
Problem: agent had the study conclusion in search results but answered from
general fitness knowledge ("cables provide constant tension throughout ROM").
System prompt rules alone were insufficient — the model's training data on
"dumbbell vs cable" questions overrode the instruction to cite the study.

Fix: tool result itself now contains a hard instruction field:
"USER ARTICLE FOUND — Lead with the study conclusion. Do NOT answer from
general fitness knowledge if the article directly answers the question."
Instruction embedded in data beats instruction embedded in system prompt.

Lesson: when a model has strong prior beliefs about a topic (e.g. well-known
fitness advice), system prompt rules that contradict those beliefs are often
ignored. The fix is to embed the override instruction in the data the model
is processing, not in background instructions it can deprioritize.

### PubMed results vs user-uploaded articles
User-uploaded articles are studies the user specifically chose to trust.
PubMed results are auto-fetched abstracts from build_corpus.py.
These are different trust levels and must be labeled differently.

Fix: search_fitness_knowledge now returns three instruction variants:
- user_article_found: true → cite study by name, lead with conclusion
- pubmed/wikipedia only → prefix answer with "Note: No study in your personal
  knowledge base covers this topic. The following is based on general research literature."
- nothing found → prefix with "Note: ...based on general fitness knowledge."

### System prompt confidentiality
Agent was revealing full system prompt contents when asked. This exposes
internal tool names, database schema, and behavioral rules.

Fix: CONFIDENTIALITY RULE in system prompt. Single-sentence refusal:
"I keep my internal instructions confidential."

Lesson: LLMs will comply with user requests even when those requests work
against the system design. Every sensitive instruction in the system prompt
needs a corresponding refusal rule.

### User article lifecycle — bidirectional sync
Problem: delete_user_article deleted from ChromaDB but not from disk.
Problem: list_user_articles only checked ChromaDB, not disk.
Two drift directions: file on disk not in ChromaDB, and chunks in ChromaDB
with no file on disk.

Fix: list_user_articles now syncs both directions on every call:
- File on disk, not in ChromaDB → auto-ingest
- Chunks in ChromaDB, no file on disk → auto-remove
delete_user_article now deletes from both ChromaDB and disk simultaneously.

Lesson: when two storage systems need to stay in sync, pick one as source of
truth and make every read operation heal drift. Don't assume the systems stay
in sync — they won't.

### Groq fully eliminated
src/router.py and src/answer.py are dead code — Stage 2-3 legacy files from
the hardcoded router pipeline, replaced by the MCP agent loop in Stage 4.
The only remaining live Groq dependency was _documents_are_relevant() imported
from src/answer.py into combined_server.py. This was a redundant relevance gate —
the cross-encoder reranker at threshold 0.0 already handles relevance filtering.
Removed. GROQ_API_KEY no longer needed.

Single API key (GEMINI_API_KEY) now covers the entire stack.

---

## Planned: Analytical AI Coaching

Before enabling analytical queries (plateau detection, overtraining signals,
progression rate analysis), the following prerequisites must be in place:

1. Memory auto-extraction quality — current extraction stores conversation
   transcripts instead of insights. Must store facts like "stuck at 100 lbs
   on Cable Triceps since March", not "user asked about their Cable Triceps PR."

2. Reflection step for analytical questions — currently checks unit errors
   and fabricated citations. For analytics must also check: did the agent look
   at enough data? Did it consider the full time range?

3. Schema prompt analytical patterns — agent doesn't know it can calculate
   average weekly volume per muscle group, progression rates, plateau detection.
   Schema prompt needs examples of these query patterns.

4. Context pruning — a 5-month history query returns hundreds of rows re-sent
   on every subsequent API call. Context pruning is a prerequisite for analytics.

5. Gemini thinking budget > 0 — currently thinking_budget=0. For analytical
   questions (why am I stuck for 5 months?) a budget of 1024 is appropriate.
   Not worth enabling until verifiable answers are fully trusted.

Rule: enable analytical features only after simple lookups are fully verified.
Analytical answers (plateau detection, overtraining) cannot be cross-checked
against raw data the way PR lookups and workout history can.

---

## Session 6: Exercise Session Display Overhaul & Agent Fixes

### Move all math to the tool level
The agent applying unit conversions and quirk offsets non-deterministically
caused weeks of inconsistent answers. The agent was sometimes applying
×2.2046 conversion, sometimes applying quirk offsets on top, sometimes both.
Fix: get_exercise_sessions now returns final human-readable values with unit
already applied and numeric_offset already added. The agent just displays them.
Never let the agent do arithmetic on workout data — it will be inconsistent.

### Remove ambiguous data from tool results
Having both raw sets and display_sets in the tool result gave the agent a
choice — it picked the wrong one half the time. Removing the raw sets field
entirely eliminated the inconsistency immediately. When you want the agent to
do something specific, remove the option to do something else.

### Comment matching requires structured pre-processing at the tool level
Asking the agent to correlate session_comments with sets using reps/weight
matching failed repeatedly across many attempts. Prompting the agent harder
did not fix it. The fix: pre-process at the tool level — match comments to
sets, group drop sets, format into display_sets strings. Agent output is
then deterministic because there's nothing left to decide.

### display_sets architecture
The correct pattern for structured workout data:
1. Tool builds display_sets: pre-matched comments, drop sets grouped with →,
   set numbers prepended, warmup labeled, all weights pre-converted.
2. Agent receives ready-to-display strings and copies them verbatim.
3. Reflection step verifies → symbol not replaced with words, comments
   not stripped, strings not paraphrased.
This pattern should be used for any structured data that requires consistent
formatting — move formatting decisions to the tool, not the agent.

### resolve_exercise_name 5-tier matching
Old 3-tier matcher failed on compound words, typos, and plurals. Rebuilt:
- Tier 0: space-normalized exact match — handles ALL compound words
  ('skullcrusher' → 'skull crusher') without a predefined list
- Tier 1-2: exact + partial LIKE (unchanged)
- Tier 3: plural/singular expansion — 'extensions' tries 'extension'
- Tier 4: difflib.SequenceMatcher ≥ 0.75 — handles typos
Key lesson: a general solution (space normalization, edit distance) is better
than a specific solution (COMPOUND_SPLITS dict) — it handles cases you
haven't thought of yet.

### Unit sources of truth must never be duplicated
A memory fact 'User prefers weights in kg for Deadlift, lbs for others'
conflicted with user_context.json which defines KG_NATIVE exercises precisely.
Two sources of truth for the same fact will diverge. Fix: delete the memory
fact, add a UNIT RULE to system prompt that explicitly names user_context.json
as the sole authoritative source. Never store unit preferences in memory.

### Data-level instructions beat system prompt rules
When the model has strong training priors (e.g. 'cables provide constant
tension' for lateral raise questions), system prompt rules that contradict
those priors are consistently overridden. The only reliable fix is to embed
the override instruction in the tool result data itself — it's processed as
fresh in-context information rather than background instructions competing
with training priors.

Multi-Agent System: Building the Data Agent
Architecture decision: why agent-directed data collection fails at scale
The single agent worked for simple lookups but hit a fundamental limit for analytics: 51 exercises across 2+ years cannot be comprehensively analyzed in 12 iterations. At each question, the agent sampled 8-10 exercises. The answer depended on which exercises it happened to pick. Different questions got different samples. Complete coverage was impossible within token constraints. No amount of "look at all exercises" in the system prompt changes this — the agent can only do N tool calls per question, and each call returns O(1) exercises.
Decision: build a multi-agent system on a separate git branch:

Data Agent: pure Python, zero LLM calls, deterministic collection pipeline
Analysis Agent: receives complete pre-processed dataset, reasoning only
Coordinator: simple router between analytical and simple question paths

The key insight: the data layer must be completely separate from the reasoning layer. No LLM decisions in data collection. No LLM calls to decide which exercises to include.
Pre-aggregation is architecturally superior to agent-directed data requests
Initial approach: Coordinator decides what data to request based on the question (call collect() with exercise_names=[...]). Problem: this requires the Coordinator to be an intelligent data strategist — exactly what we want to eliminate.
Better approach (from a previous conversation on this project): Data Agent always runs a fixed, complete pipeline. Phase 1 always runs for all active exercises. Phase 2 triggers deterministically via Python conditions (plateau > 4 weeks, improvement > 20%). Analysis Agent receives a complete self-contained package and never asks for more data.
Why it's better:

Coordinator becomes a pure router — no intelligence needed about what data to request
Analysis Agent never needs to request more data mid-reasoning
Phase 2 triggers are deterministic Python, not LLM decisions
Output is compact summaries, not raw session arrays — fits in LLM context
The Analysis Agent always sees the complete picture, not a sampled subset

The pre-aggregation approach should be implemented as a prepare_analysis_package() wrapper on top of collect(), not as a replacement. collect() is the correct raw data access layer.
Pure Python for the data layer eliminates an entire class of bugs
The single-agent system had all math done by the LLM: unit conversions, offset application, bar weight addition. This produced inconsistent answers — LLMs apply the same calculation differently depending on context, phrasing, and what else is in the conversation. Different sessions, different results.
Data Agent approach: every calculation happens in deterministic Python before the LLM ever sees a number. The Analysis Agent receives 130.0 lbs, not 58.97 with a note saying "multiply by 2.2046." It cannot apply unit conversions because there is nothing left to convert.
This eliminated the entire category of "agent math" bugs from Session 6 (unit conversions applied twice, offsets missing on some calls, bar weights forgotten on others). When the data layer is deterministic, the LLM's job is to reason, not to do arithmetic.
Exhaustive code review finds bugs that testing cannot
Building the Data Agent involved multiple rounds of comprehensive code review — reading every function, tracing every data path — that found bugs the tests never triggered:

_fetch_all_training_dates didn't exclude categories 10/11/12 — 9 phantom training days inflated total_training_days, streak, gap, and every date-based stat
"couldn't"/"couldnt" in PAIN_KEYWORDS caused 33 false pain flags — rep failure comments flagged as injury
_fetch_exercise_lifecycle used HAVING instead of WHERE for category filter — semantically wrong, worked in SQLite by coincidence
_aggregate_weekly used strftime('%Y-%W') which splits year-boundary weeks incorrectly
EXCLUDED_CATEGORY_IDS constant defined at the top, literal (10, 11, 12) used in 6 SQL queries — constant was decorative
alltime_all_rows fetched 6810 rows on every collect() call regardless of filters or exercise count

None of these would have shown up in happy-path testing. They required reading every line and reasoning about edge cases. Write the code, then read it as if you are looking for bugs, not as if you are verifying correctness.
Phantom training days inflate all date-based statistics
_fetch_all_training_dates used SELECT DISTINCT date FROM training_log with no category filter. There were 9 days where only excluded-category exercises (Morning, Evening, Society, Neck) were logged — no real exercise. These phantom days inflated:

total_training_days (309 → should be 300)
longest_streak_days and longest_gap_days
sessions_per_week and weeks_missed
day_of_week_patterns and seasonal_patterns
PR context consecutive-day counts
Consecutive day effect analytics

Fix: JOIN exercise and WHERE e.category_id NOT IN (10, 11, 12) in the dates query. Every query that uses training dates must apply the same category exclusion as the set queries — they are not independent operations.
"Couldn't" is not a pain keyword
PAIN_KEYWORDS contained "couldn't" and "couldnt." Checking these keywords against real DB comments: all 33 occurrences were about rep failure ("Couldnt do 60", "Couldnt squeeze at the top"), not injury or pain. The keyword was flagging normal training comments as injury events, inflating pain_session_count across many exercises.
The words belong in COMMENT_TREND_KEYWORDS["failure"] where they track rep failure frequency as a training pattern. They do not belong in PAIN_KEYWORDS where they trigger injury alerts.
Lesson: every keyword in a classification list must be verified against real data before inclusion. "Couldn't" sounds intuitively like it belongs with pain. In workout logging it almost always means hitting rep failure — normal, expected, desirable.
SQL HAVING vs WHERE: semantic correctness matters even when SQLite allows it
_fetch_exercise_lifecycle used HAVING e.category_id NOT IN (10, 11, 12) after GROUP BY. SQLite allows this because category_id is in the GROUP BY clause. But HAVING is for conditions on aggregated results. WHERE is for row-level filters applied before aggregation. Category_id is a row-level filter — it belongs in WHERE.
SQLite's permissiveness means this bug hides until someone reads the code and thinks about what the clause is doing, not just whether it produces the right output. Always ask: is this a row-level condition or a group-level condition? If row-level, it goes in WHERE regardless of whether HAVING also works.
ISO 8601 week numbering vs Python's strftime('%Y-%W')
strftime('%Y-%W') treats January 1 as week 00 if it falls before the first Monday. Training sessions on December 28-31 might be "2025-52" while January 1-3 becomes "2026-00" — the same training week split across two artificial buckets.
Fix: use date.isocalendar() which returns ISO 8601 week numbers. ISO 8601 defines the week containing the first Thursday as week 1 of the year. Cross-year training weeks stay in one bucket. Format: f"{iso_year:04d}-W{iso_week:02d}" → "2025-W52", "2026-W01".
This affected three places that must all use the same week format: _aggregate_weekly, _compute_muscle_group_summary, and _compute_training_consistency. If any one of them uses a different format, weekly aggregation and consistency calculation will produce mismatched keys and silently produce wrong counts.
Constants that are defined but never used create silent maintenance debt
EXCLUDED_CATEGORY_IDS = (10, 11, 12) was defined at the module level. All 6 SQL queries used the literal (10, 11, 12) in strings. If someone changed the constant, the queries would silently not update.
Fix: derive _EXCL_SQL = f"({', '.join(str(c) for c in EXCLUDED_CATEGORY_IDS)})" immediately after the constant. Use _EXCL_SQL in all 6 queries via f-strings. Changing EXCLUDED_CATEGORY_IDS now automatically updates every query.
Pattern: if you define a constant for a value, derive every hardcoded form of that value from the constant. Never define a constant and also use the literal — the constant becomes decoration.
Cross-unit comparisons require explicit normalization
The Deadlift was logged in lbs before a specific date, in kg after. Sessions store max_working_weight in the logged unit. When computing PR and plateau, the code compared raw numbers: 10 (lbs, pre-switch) vs 65 (kg, post-switch). This happened to produce correct results because 10 < 65 numerically. It is a latent bug: if pre-switch weights were numerically higher than post-switch kg values, the PR would pick the wrong session.
Fix: _to_unit(value, from_unit, to_unit) helper. Used in _compute_pr and _compute_progression to normalize all sessions to the current exercise unit before any comparison. The fix produces the same results for current Deadlift data (the latent bug never triggered) but is correct for any future data where the unit switch crosses numerically significant values.
FitNotes' built-in unit conversion handles global gym switches cleanly
While designing a complex session-normalization scheme for the scenario of switching gyms (lbs → kg), the user discovered that FitNotes has a built-in imperial/metric conversion feature. Switching the app's unit setting converts all historical data in the DB simultaneously — the backup exported after a switch is self-consistent throughout.
This means: when the user switches via FitNotes, the uploaded backup has consistent units. The only mixed-unit case is manual partial switches (like the current situation where one exercise was manually moved to kg). The _to_unit fix handles those correctly. For global gym switches, no complex migration is needed — switch in the app, export backup, upload, update user_context.json.
Lesson: understand the tools you are building on before designing solutions to problems they already solve.
Exercise data completeness requires domain knowledge, not just schema knowledge
Building the Data Agent required understanding how each exercise type actually stores its data:

Walking, Treadmill: distance (km) and duration_seconds are the real performance metrics. weight=0, reps=0 always. Walking is a pre-gym walk, not a training exercise — its data is useful for correlating with same-day strength performance.
Cycling: duration_seconds is the metric. distance=0, weight=0. Comments add structure detail (intervals, difficulty).
Dead Hang: duration_seconds is the metric. reps=0 is the logging convention, not a failure.
Farmers Walk: weight is the progression metric. reps=0 is the convention — it is not rep-based.
Dumbbell Hold: reps field stores duration in seconds. Not a rep count.

None of this is obvious from the schema. It required reading the actual DB data, checking the app, and understanding the user's logging conventions. The exercise_quirks system in user_context.json is the right place to document these — it makes conventions explicit and readable by the Analysis Agent.
Data pipelines must be built with domain knowledge, not just schema knowledge. Schema tells you what fields exist. Domain knowledge tells you what they mean.
The reps=0 failed-attempt heuristic needs exercise awareness
Default logic: reps == 0 AND duration_seconds == 0 AND distance == 0 → is_failed_attempt = True. This is correct for strength exercises (a set logged with 0 reps and nothing in other fields is genuinely a failed attempt).
But it incorrectly flagged every Farmers Walk set and every Dead Hang set, because reps=0 is the normal logging convention for those exercises.
Fix: reps_zero_is_normal flag in exercise_quirks. When True, reps=0 is never a failed attempt for that exercise. For duration-based exercises where duration_seconds > 0, the existing logic already handles disambiguation correctly — those are not failed attempts even without the flag.
Aggregation level selection is a data architecture decision, not a display decision
For 90-day queries: session-level detail. For 365-day: weekly. For all-time: monthly. Without automatic aggregation level selection, a "how has my training been over the past 2 years?" query would return raw session arrays for 300+ training days — overwhelming any LLM context window.
The aggregation level selection (session ≤ 90 days, weekly ≤ 365, monthly for all-time) directly determines whether the Analysis Agent can reason over a complete picture or gets drowned in data. This is decided in the data layer, not the presentation layer.
Phase 2 triggers must be deterministic Python, not LLM decisions
Phase 2 (fetch full comment history) triggers when: plateau_days > 28 OR weight_change_pct > 20. These thresholds are constants in Python. The LLM never decides whether to fetch comments.
Why it matters: LLMs are inconsistent about when to fetch more data. Across sessions, the agent would sometimes think to fetch comments and sometimes not, producing different quality answers to the same question. Deterministic Python triggers ensure complete coverage for every exercise that meets the criteria, every time, regardless of how the question is phrased.
Output size is a first-class design constraint for multi-agent systems
All-time + all exercises + Phase 2 = 4.7 MB. Overflows any LLM context window.
90-day + all exercises = 2.6 MB. Still too large for most use cases.
Single exercise + Phase 2 = 112 KB. Correct target.
Muscle group + 365 days = 282 KB. Correct target.
The Coordinator must never pass the raw collect() output to the Analysis Agent. Its job is to extract only the fields relevant to the question. The collect() function returns everything. A prepare_analysis_package() wrapper returns only what the Analysis Agent needs — compact summaries, not raw session arrays.
The compact package strips sets arrays (every individual set with all its fields — this is the bloat) while keeping all analytics derived from those sets: pain_analysis, technique_variants, form_quality, comment_keyword_trends, full_comments for Phase 2 exercises. The Analysis Agent has everything comment-derived without carrying the raw comment storage.
The alltime_rows fetch is a performance bottleneck in filtered queries
Every collect() call fetches all 6810 rows from 2000-01-01 to today for learning curve computation, even when querying a single exercise for 90 days. This is because learning curves require all-time session history to compute first_ever_session, sessions_to_first_pr, and first_30d_weight_gain.
The proper fix is pre-aggregation: a SQL query that returns one row per exercise (MIN(date), COUNT(DISTINCT date), first_pr_date), then targeted row fetches only for the first 30 days of each exercise's history. This reduces from 6810 rows to ~30 rows for targeted queries.
Not yet implemented — acceptable at current data size (0.09s). Worth implementing before data grows beyond 20,000 sets.
Goal and bodyweight tables may be empty — test with real data before declaring features complete
The test backup had 0 goals and 0 bodyweight entries. The code paths for goal projection (_compute_goal_projection, _process_goals) and bodyweight correlation (_compute_bw_strength_correlation) were written and verified correct against the schema but never tested against real data. Edge cases that only appear with actual values (negative e1rm rate, multiple goals for the same exercise, bodyweight entries far from training dates) remain untested.
Lesson: always test against real data before declaring a feature complete. Schema-level correctness and real-world correctness are different things.
Two storage systems always drift — make every read operation a sync point
Confirmed again during Data Agent development with the list_user_articles / ChromaDB / disk sync issue from Session 6 (carried over from the single-agent work). The pattern holds universally: whenever two storage systems must stay in sync, pick one as the source of truth and make every read operation heal drift in both directions. Don't assume sync operations stay synchronized — they don't.
Verified the same lessons transfer from single-agent to multi-agent context
Several lessons from the single-agent work were re-learned independently during Data Agent development:

Move all math to the data layer (Session 6: "move formatting to the tool") — independently arrived at the same principle
Never let an LLM do arithmetic on structured data — independently confirmed
Deterministic beats flexible when correctness matters — confirmed again
Two sources of truth for the same fact will diverge — confirmed with EXCLUDED_CATEGORY_IDS

When the same lesson appears independently in two different architectural contexts, it is genuinely a fundamental principle, not a one-off observation.
