# FitNotes Coach — Stage 1: Text-to-SQL

A minimal pipeline that answers natural-language questions about your FitNotes workout history by generating SQL, running it against your local SQLite database, and returning a plain-English explanation.

**Two LLM calls per question. No agent loop. No frameworks. Just Python.**

---

## What Stage 1 does

```
Your question
     │
     ▼
 generate_sql()  ──►  Groq (kimi-k2-instruct)  ──►  SELECT ...
     │
     ▼
 run_query()     ──►  SQLite (read-only)         ──►  [{...}, ...]
     │
     ▼
 explain_result() ──► Groq (kimi-k2-instruct)   ──►  "Your PR is 80 kg..."
```

---

## Requirements

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier works)
- Your `FitNotes_Backup.fitnotes` file exported from the FitNotes Android app

---

## Installation

```bash
cd fitnotes_coach

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_...your_key_here...
FITNOTES_DB_PATH=./data/FitNotes_Backup.fitnotes
```

---

## Drop in your database

Export a backup from FitNotes (app menu → Backup & Restore → Backup to device), then copy the `.fitnotes` file:

```
fitnotes_coach/
└── data/
    └── FitNotes_Backup.fitnotes   ← put it here
```

---

## Run the CLI

```bash
python cli.py
```

Example session:

```
FitNotes Coach — Stage 1 (Text-to-SQL)
Type a question about your workout history. Type 'exit' or 'quit' to leave.

You: What's my PR for Lat Pulldown?

[SQL]
SELECT tl.metric_weight AS pr_kg, tl.reps, tl.date
FROM training_log tl
JOIN exercise e ON tl.exercise_id = e._id
WHERE e.name = 'Lat Pulldown' AND tl.is_personal_record = 1
ORDER BY tl.metric_weight DESC LIMIT 1;

[Answer]
Your personal record for Lat Pulldown is 80 kg for 8 reps, set on 2024-11-03.

[Rows returned: 1]
```

---

## Run the eval harness (Stage 1.5)

The eval harness uses two complementary scoring methods — execution-based SQL
comparison and an LLM-as-judge for natural-language answers — replacing the
naive substring matching from Stage 1.

### Step 1 — Populate ground truth (one-time, ~20 minutes)

```bash
python evals/build_ground_truth.py
```

This interactive script walks through each of the 20 eval cases. For each one
you paste a hand-written SQL query (ended with `;;;`), inspect the rows it
returns, and enter a one-sentence plain-English answer. Progress is saved
atomically after every case; type `exit` at any prompt to quit and resume
later.

**Do not skip this step.** Cases without ground truth are automatically skipped
during scoring.

### Step 2 — Score

```bash
python evals/run_evals.py
```

Prints a table and a failure report, then saves `evals/results.json`:

```
id    difficulty  sql_ok   answer_ok  gt_rows        sys_rows     time_ms
─────────────────────────────────────────────────────────────────────────
q01   easy        PASS     PASS       1              1            1823
q02   easy        PASS     FAIL       1              1            1541
...

Pass rates by difficulty:
  easy      sql 8/8   answer 7/8
  medium    sql 6/8   answer 6/8
  hard      sql 3/4   answer 3/4

Overall (20 run):  sql 17/20 (85%)  answer 16/20 (80%)

────────────────────────────────────────────────────────────────
FAILURES (3):
────────────────────────────────────────────────────────────────

q02 [easy]: judge marked incorrect
  Q: When did I last train Flat Dumbbell Bench Press?
  Judge: System said 2024-10-15 but ground truth is 2024-11-03.
```

### How the two scoring methods work

**Execution-based SQL scoring:** Both the hand-written `ground_truth_sql` and
the system's generated SQL are executed against the real SQLite database. The
resulting row sets are compared with `evals/row_compare.py`: column names are
ignored (only values matter), row order is ignored unless the ground-truth SQL
contains `ORDER BY`, and floats are compared within 1% relative tolerance. A
ground-truth query that returns a single scalar value is matched if that value
appears anywhere in the system's result rows.

**LLM-as-judge answer scoring:** The judge receives the original question, the
hand-written `ground_truth_answer`, and the system's natural-language answer.
It is instructed to mark the system correct if the core facts match, allowing
for differences in phrasing, unit (kg vs lbs — with conversion), rounding
within 1%, and additional commentary. It marks the system incorrect if a
numeric value differs by more than 1%, the wrong exercise or date is reported,
or a key fact is missing. Temperature is set to 0 for determinism.

### Verify row-comparison logic

```bash
python evals/row_compare.py
```

Runs 10 built-in assertion tests covering identical rows, reordered rows,
scalar matching, float tolerance, and edge cases. No test framework required.

---

---

## Project structure

```
fitnotes_coach/
├── data/
│   └── FitNotes_Backup.fitnotes   # you provide this
├── src/
│   ├── db.py                      # read-only SQLite connection, schema introspection, query runner
│   ├── llm.py                     # Groq client — generate_sql(), explain_result(), judge_answer()
│   ├── schema_prompt.py           # curated schema description fed to the LLM
│   └── text_to_sql.py             # the full pipeline: question → SQL → rows → answer
├── evals/
│   ├── eval_set.json              # 20 test questions with ground_truth_sql / ground_truth_answer
│   ├── build_ground_truth.py      # one-time interactive helper to populate ground truth
│   ├── row_compare.py             # row-set comparison logic + self-tests
│   ├── run_evals.py               # execution-based + LLM-judge eval harness
│   └── results.json               # output from last eval run
├── cli.py                         # interactive REPL
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Known limitations of the eval harness

- **LLM judge is not infallible.** The judge can misread unit conversions,
  accept wrong answers that sound plausible, or reject correct answers that
  use unexpected phrasing. Treat judge scores as a signal, not ground truth.

- **Ground truth is only as good as the hand-written SQL.** If your reference
  SQL is wrong (wrong filter, off-by-one date, wrong join), every future run
  will score against a bad baseline. Review the SQL carefully before accepting
  it in `build_ground_truth.py`.

- **20 questions is a small sample.** The eval set covers common patterns but
  cannot represent every query type or edge case in your workout history. A
  passing score does not guarantee correctness on unseen questions.

- **Calibration is not tested.** The harness checks whether the system gives
  the right answer when one exists, but does not test whether the system
  appropriately says "I don't know" or "no data found" for questions that have
  no answer in the database.

---

## Intentionally missing (future stages)

| Feature | Stage |
|---|---|
| MCP server (expose tools to Claude Desktop / other agents) | Stage 2 |
| RAG / semantic search over exercise notes | Stage 3 |
| Agent loop with multi-step reasoning and tool use | Stage 3 |
| Write operations (log a set, update a goal) | Stage 4 |
| Conversation memory across turns | Stage 3 |
| Streaming responses | Stage 2+ |

Stage 1 is deliberately minimal so failure modes are easy to see — if the SQL is wrong, you see it printed; if the LLM hallucinates a table name, the query error surfaces immediately.
