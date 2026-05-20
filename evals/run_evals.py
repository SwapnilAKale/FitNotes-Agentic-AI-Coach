#!/usr/bin/env python3
"""
Stage 1.5 eval harness.

Scoring strategy:
  - SQL correctness: execute both ground_truth_sql and the system's SQL against
    the real DB, then compare result sets with rows_match().
  - Answer correctness: call judge_answer() (LLM-as-judge) to compare the
    system's natural-language answer against the hand-written ground_truth_answer.

Cases where ground_truth_sql or ground_truth_answer is null are skipped.

Usage (from project root):
    python evals/run_evals.py
"""
import json
import os
import re
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _EVALS_DIR)

from src.db import get_connection, run_query, sanitize_sql  # noqa: E402
from src.llm import judge_answer                  # noqa: E402
from src.text_to_sql import answer_question       # noqa: E402
from row_compare import rows_match                # noqa: E402

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
EVAL_SET_PATH = os.path.join(_EVALS_DIR, "eval_set.json")
RESULTS_PATH = os.path.join(_EVALS_DIR, "results.json")

_COL = "{:<5} {:<10} {:<8} {:<10} {:<14} {:<12} {:<10}"
_HEADER = _COL.format("id", "difficulty", "sql_ok", "answer_ok", "gt_rows", "sys_rows", "time_ms")
_SEP = "─" * len(_HEADER)


def _has_order_by(sql: str) -> bool:
    return bool(re.search(r"\bORDER\s+BY\b", sql, re.IGNORECASE))


def _run_gt_sql(conn, sql: str) -> tuple[list[dict], str | None]:
    try:
        return run_query(conn, sql), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


_JUDGE_ERROR_PHRASES = (
    "judge call failed",
    "judge output unparseable",
    "GEMINI_API_KEY not set",
    "skipped —",
)


def _is_judge_error(verdict: dict) -> bool:
    r = verdict.get("reasoning", "")
    return any(r.startswith(p) for p in _JUDGE_ERROR_PHRASES)


def run_evals() -> None:
    with open(EVAL_SET_PATH, encoding="utf-8") as fh:
        eval_set = json.load(fh)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Drop FitNotes_Backup.fitnotes into data/ and set FITNOTES_DB_PATH.")
        sys.exit(1)

    gt_conn = get_connection(DB_PATH)

    total_run = 0
    total_skipped = 0
    total_sql_pass = 0
    total_answer_pass = 0
    total_answer_judged = 0
    totals: dict[str, dict[str, int]] = {}
    raw_results: list[dict] = []
    failures: list[dict] = []

    print(f"\nRunning evals against {DB_PATH}\n")
    print(_HEADER)
    print(_SEP)

    for case in eval_set:
        qid = case["id"]
        difficulty = case["difficulty"]
        question = case["question"]

        # ── Skip cases without ground truth ──────────────────────────────
        if not case.get("ground_truth_sql") or not case.get("ground_truth_answer"):
            print(_COL.format(qid, difficulty, "SKIP", "SKIP", "─", "─", "─"))
            total_skipped += 1
            raw_results.append({
                "id": qid,
                "difficulty": difficulty,
                "question": question,
                "skipped": True,
            })
            continue

        gt_sql = sanitize_sql(case["ground_truth_sql"])
        gt_answer = case["ground_truth_answer"]

        # ── Run system pipeline ───────────────────────────────────────────
        start = time.monotonic()
        result = answer_question(question, DB_PATH)
        time.sleep(15)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        sys_sql = result.get("sql", "")
        sys_rows = result.get("rows", [])
        sys_answer = result.get("answer", "")
        sys_error = result.get("error")

        # ── Execute ground-truth SQL ──────────────────────────────────────
        gt_rows, gt_error = _run_gt_sql(gt_conn, gt_sql)

        # ── SQL correctness (execution-based row comparison) ──────────────
        sql_ok = False
        sql_fail_reason = ""
        if sys_error:
            sql_fail_reason = f"system error: {sys_error}"
        elif gt_error:
            sql_fail_reason = f"ground-truth SQL error: {gt_error}"
        else:
            ordered = _has_order_by(gt_sql)
            sql_ok = rows_match(gt_rows, sys_rows, has_order_by=ordered)
            if not sql_ok:
                sql_fail_reason = "row mismatch"

        # ── Answer correctness (LLM judge) ────────────────────────────────
        # answer_ok is True/False for a real verdict, None when the judge itself erred.
        # None cases are excluded from the pass-rate denominator.
        answer_ok: bool | None = None
        judge_reasoning = ""
        if sys_error:
            judge_reasoning = f"skipped — system error: {sys_error}"
        else:
            verdict = judge_answer(question, gt_answer, sys_answer)
            time.sleep(12)
            judge_reasoning = verdict["reasoning"]
            if _is_judge_error(verdict):
                answer_ok = None
            else:
                answer_ok = verdict["correct"]

        # ── Accumulate totals ─────────────────────────────────────────────
        total_run += 1
        if difficulty not in totals:
            totals[difficulty] = {"sql_pass": 0, "answer_pass": 0, "answer_judged": 0, "total": 0}
        totals[difficulty]["total"] += 1
        totals[difficulty]["sql_pass"] += int(sql_ok)
        total_sql_pass += int(sql_ok)
        if answer_ok is not None:
            totals[difficulty]["answer_pass"] += int(answer_ok)
            totals[difficulty]["answer_judged"] += 1
            total_answer_pass += int(answer_ok)
            total_answer_judged += 1

        answer_col = "PASS" if answer_ok is True else ("FAIL" if answer_ok is False else "ERR")
        print(_COL.format(
            qid,
            difficulty,
            "PASS" if sql_ok else "FAIL",
            answer_col,
            len(gt_rows),
            len(sys_rows),
            elapsed_ms,
        ))

        entry = {
            "id": qid,
            "difficulty": difficulty,
            "question": question,
            "skipped": False,
            "system_sql": sys_sql,
            "system_rows": sys_rows,
            "system_answer": sys_answer,
            "system_error": sys_error,
            "ground_truth_sql": gt_sql,
            "ground_truth_rows": gt_rows,
            "ground_truth_answer": gt_answer,
            "ground_truth_error": gt_error,
            "sql_ok": sql_ok,
            "answer_ok": answer_ok,
            "judge_reasoning": judge_reasoning,
            "time_ms": elapsed_ms,
        }
        raw_results.append(entry)

        if not sql_ok or answer_ok is False:
            failures.append({**entry, "_sql_fail_reason": sql_fail_reason})

    print(_SEP)

    # ── Summary ───────────────────────────────────────────────────────────
    if total_run:
        print("\nPass rates by difficulty:")
        for diff in ("easy", "medium", "hard"):
            if diff not in totals:
                continue
            t = totals[diff]
            judged = t["answer_judged"]
            err_note = f" ({t['total'] - judged} judge err)" if judged < t["total"] else ""
            print(
                f"  {diff:<8}  sql {t['sql_pass']}/{t['total']}"
                f"  answer {t['answer_pass']}/{judged}{err_note}"
            )
        skipped_note = f", {total_skipped} skipped" if total_skipped else ""
        judge_err = total_run - total_answer_judged
        judge_err_note = f", {judge_err} judge err" if judge_err else ""
        ans_pct = (100 * total_answer_pass // total_answer_judged) if total_answer_judged else 0
        print(
            f"\nOverall ({total_run} run{skipped_note}):  "
            f"sql {total_sql_pass}/{total_run} "
            f"({100 * total_sql_pass // total_run}%)  "
            f"answer {total_answer_pass}/{total_answer_judged} "
            f"({ans_pct}%){judge_err_note}"
        )
    else:
        print(f"\nAll {total_skipped} cases skipped — no ground truth populated yet.")
        print("Run:  python evals/build_ground_truth.py")

    # ── Failure report ────────────────────────────────────────────────────
    if failures:
        print(f"\n{'─' * 64}")
        print(f"FAILURES ({len(failures)}):")
        print("─" * 64)
        for entry in failures:
            reasons = []
            if not entry["sql_ok"]:
                reasons.append(entry.get("_sql_fail_reason") or "SQL row mismatch")
            if entry["answer_ok"] is False:
                reasons.append("judge marked incorrect")

            print(f"\n{entry['id']} [{entry['difficulty']}]: {' + '.join(reasons)}")
            print(f"  Q: {entry['question']}")
            if not entry["sql_ok"] and not entry["system_error"] and not entry["ground_truth_error"]:
                print(f"  GT rows (first 3):  {entry['ground_truth_rows'][:3]}")
                print(f"  SYS rows (first 3): {entry['system_rows'][:3]}")
            if entry["answer_ok"] is False and entry["judge_reasoning"]:
                print(f"  Judge: {entry['judge_reasoning']}")

    # ── Save raw results ──────────────────────────────────────────────────
    # Strip internal _sql_fail_reason from saved output
    save_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in raw_results
    ]
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(save_results, fh, indent=2)
    print(f"\nRaw results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evals()
