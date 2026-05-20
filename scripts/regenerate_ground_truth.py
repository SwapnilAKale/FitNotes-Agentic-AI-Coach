"""
Ground truth regeneration script.
Runs all 19 eval questions through the current pipeline and saves
candidate ground truth to evals/candidates.json for review.

Usage:
  python scripts/regenerate_ground_truth.py

Review candidates.json, then run with --apply to update eval_set.json
with approved candidates.
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.text_to_sql import answer_question

DB_PATH = "./data/FitNotes_Backup.fitnotes"
EVAL_SET_PATH = "evals/eval_set.json"
CANDIDATES_PATH = "evals/candidates.json"

def load_eval_set():
    with open(EVAL_SET_PATH) as f:
        data = json.load(f)
    # eval_set.json is a top-level array
    if isinstance(data, list):
        return {"questions": data, "_raw_list": True}
    return data

async def run_question(question: str) -> dict:
    """Run a single question through the current pipeline."""
    try:
        result = answer_question(question, DB_PATH)
        return {
            "sql": result.get("sql", ""),
            "rows": result.get("rows", []),
            "answer": result.get("answer", ""),
            "error": result.get("error")
        }
    except Exception as e:
        return {"sql": "", "rows": [], "answer": "", "error": str(e)}

def main():
    apply_mode = "--apply" in sys.argv

    eval_set = load_eval_set()
    questions = eval_set.get("questions", [])

    if apply_mode:
        # Apply approved candidates to eval_set.json
        if not os.path.exists(CANDIDATES_PATH):
            print("No candidates.json found. Run without --apply first.")
            return
        with open(CANDIDATES_PATH) as f:
            candidates = json.load(f)

        updated = 0
        for q in questions:
            qid = q.get("id")
            if qid in candidates and candidates[qid].get("approved"):
                candidate = candidates[qid]
                q["ground_truth_sql"] = candidate.get("new_sql", q.get("ground_truth_sql", ""))
                q["ground_truth_rows"] = candidate.get("new_rows", q.get("ground_truth_rows", []))
                q["ground_truth_answer"] = candidate.get("new_answer", q.get("ground_truth_answer", ""))
                updated += 1

        payload = questions if eval_set.get("_raw_list") else eval_set
        with open(EVAL_SET_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Updated {updated} questions in eval_set.json")
        return

    # Run all questions and save candidates
    print(f"Running {len(questions)} eval questions through current pipeline...\n")
    candidates = {}

    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i+1:02d}")
        question_text = q.get("question", "")

        print(f"[{i+1}/{len(questions)}] {qid}: {question_text[:60]}...")

        result = answer_question(question_text, DB_PATH)

        new_sql = result.get("sql", "")
        new_rows = result.get("rows", [])
        new_answer = result.get("answer", "")

        old_sql = q.get("ground_truth_sql") or ""
        old_rows = q.get("ground_truth_rows") or []
        old_answer = q.get("ground_truth_answer") or ""

        # Detect changes
        sql_changed = new_sql.strip() != old_sql.strip()
        rows_changed = str(new_rows) != str(old_rows)
        answer_changed = new_answer.strip() != old_answer.strip()

        status = "CHANGED" if (sql_changed or rows_changed) else "SAME"
        print(f"  Status: {status}")

        if sql_changed:
            print(f"  OLD SQL: {old_sql[:100]}")
            print(f"  NEW SQL: {new_sql[:100]}")
        if rows_changed:
            print(f"  OLD rows: {str(old_rows)[:100]}")
            print(f"  NEW rows: {str(new_rows)[:100]}")

        candidates[qid] = {
            "question": question_text,
            "status": status,
            "approved": False,  # User must manually set to True to approve
            "new_sql": new_sql,
            "new_rows": new_rows,
            "new_answer": new_answer,
            "old_sql": old_sql,
            "old_rows": old_rows,
            "old_answer": old_answer,
            "sql_changed": sql_changed,
            "rows_changed": rows_changed,
            "answer_changed": answer_changed
        }
        print()

    with open(CANDIDATES_PATH, "w") as f:
        json.dump(candidates, f, indent=2)

    changed = sum(1 for c in candidates.values() if c["status"] == "CHANGED")
    print(f"\nDone. {changed}/{len(questions)} questions have changes.")
    print(f"Candidates saved to {CANDIDATES_PATH}")
    print(f"\nNext steps:")
    print(f"1. Open evals/candidates.json")
    print(f"2. For each changed question, verify the new_sql and new_rows are correct")
    print(f"3. Set approved: true for questions where the new output is correct")
    print(f"4. Run: python scripts/regenerate_ground_truth.py --apply")

if __name__ == "__main__":
    main()
