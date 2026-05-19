#!/usr/bin/env python3
"""
One-time interactive helper to populate ground_truth_sql and ground_truth_answer
in evals/eval_set.json.

Usage (from project root):
    python evals/build_ground_truth.py

For each eval case you will be prompted to:
  1. Paste a SQL query (end with ;;; on its own line).
  2. Inspect the rows returned by running that SQL against the DB.
  3. Enter a one-sentence plain-English answer based on those rows.

Progress is saved atomically after every case. Type 'exit' at any prompt to quit.
"""
import json
import os
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.db import get_connection, run_query  # noqa: E402

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(eval_set: list) -> None:
    """Atomic write: temp file → rename."""
    dir_ = os.path.dirname(EVAL_SET_PATH)
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_, delete=False, suffix=".tmp", encoding="utf-8"
    ) as fh:
        json.dump(eval_set, fh, indent=2)
        tmp = fh.name
    os.replace(tmp, EVAL_SET_PATH)


def _prompt(prompt_text: str) -> str | None:
    """Print prompt and read a line. Returns None on EOF or if user types 'exit'."""
    try:
        val = input(prompt_text)
    except EOFError:
        return None
    if val.strip().lower() == "exit":
        return None
    return val


def _read_sql() -> str | None:
    """
    Read multi-line SQL from stdin until a line containing only ';;;'.
    Returns the SQL string, or None if the user exits.
    """
    print("Paste SQL (end with ;;; on its own line, or type 'exit' to quit):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return None
        if line.strip().lower() == "exit":
            return None
        if line.strip() == ";;;":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _run_sql(sql: str) -> tuple[list[dict] | None, str | None]:
    """
    Open a fresh connection and run sql. Returns (rows, error_string).
    Fresh connection each attempt to avoid any interrupt-state carryover.
    """
    if not os.path.exists(DB_PATH):
        return None, f"Database not found: {DB_PATH}"
    try:
        conn = get_connection(DB_PATH)
        rows = run_query(conn, sql)
        return rows, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _show_rows(rows: list[dict]) -> None:
    print(f"\n  Rows returned: {len(rows)}")
    for i, row in enumerate(rows[:20]):
        print(f"  [{i + 1}] {row}")
    if len(rows) > 20:
        print(f"  ... ({len(rows)} total, showing first 20)")


# ---------------------------------------------------------------------------
# Per-case workflow
# ---------------------------------------------------------------------------

def _process_case(case: dict, eval_set: list) -> bool:
    """
    Walk the user through filling in one eval case.
    Returns False if the user wants to exit, True otherwise.
    """
    print(f"\n{'=' * 64}")
    print(f"  [{case['id']}]  {case['question']}")
    print(f"  Difficulty: {case['difficulty']}")
    if case.get("notes"):
        print(f"  Notes: {case['notes']}")

    already_filled = bool(case.get("ground_truth_sql") and case.get("ground_truth_answer"))
    if already_filled:
        print(f"\n  Existing SQL:\n    {case['ground_truth_sql']}")
        print(f"\n  Existing answer: {case['ground_truth_answer']}")
        choice = _prompt("\n  [S]kip / [R]edo? (default: skip) > ")
        if choice is None:
            return False
        if choice.strip().lower() not in ("r", "redo"):
            print("  Skipped.")
            return True

    # SQL entry loop
    while True:
        sql = _read_sql()
        if sql is None:
            return False
        if not sql:
            print("  Empty SQL — try again.")
            continue

        rows, err = _run_sql(sql)
        if err:
            print(f"\n  Error: {err}")
            retry = _prompt("  Retry? [Y/n] > ")
            if retry is None or retry.strip().lower() in ("n", "no"):
                return retry is not None  # 'n' → skip case, None → exit
            continue

        _show_rows(rows)

        accept = _prompt("\n  Accept this SQL? [Y/n] > ")
        if accept is None:
            return False
        if accept.strip().lower() not in ("n", "no"):
            break
        # user said no → loop to re-enter SQL

    case["ground_truth_sql"] = sql

    # Answer entry
    print("\n  Enter a one-sentence ground_truth_answer based on the rows above:")
    answer = _prompt("  > ")
    if answer is None:
        return False
    case["ground_truth_answer"] = answer.strip()

    _save(eval_set)
    print("  Saved.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(EVAL_SET_PATH):
        print(f"ERROR: eval_set.json not found at {EVAL_SET_PATH}")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Set FITNOTES_DB_PATH in your .env or copy the .fitnotes file to data/")
        sys.exit(1)

    with open(EVAL_SET_PATH, encoding="utf-8") as fh:
        eval_set = json.load(fh)

    total = len(eval_set)
    filled = sum(
        1 for c in eval_set
        if c.get("ground_truth_sql") and c.get("ground_truth_answer")
    )

    print("\nFitNotes Coach — Build Ground Truth")
    print(f"{total} eval cases found, {filled} already complete.")
    print("Type 'exit' at any prompt to quit (progress is saved after each case).\n")

    for case in eval_set:
        try:
            cont = _process_case(case, eval_set)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break
        if not cont:
            print("\nExiting. Progress saved.")
            break

    filled_after = sum(
        1 for c in eval_set
        if c.get("ground_truth_sql") and c.get("ground_truth_answer")
    )
    print(f"\nDone. {filled_after}/{total} cases have ground truth.")


if __name__ == "__main__":
    main()
