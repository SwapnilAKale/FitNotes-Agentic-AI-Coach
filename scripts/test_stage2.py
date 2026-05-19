#!/usr/bin/env python3
"""Quick smoke test for Stage 2 — runs the three required questions."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.answer import answer_question

DB = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")

QUESTIONS = [
    "What is my PR for Lat Pulldown?",
    "What is progressive overload?",
    "Given my recent triceps volume, am I overtraining?",
]

for q in QUESTIONS:
    print("=" * 70)
    print(f"Q: {q}")
    r = answer_question(q, DB)
    print(f"[ROUTE] {r['route'].upper()}")
    if r.get("sql"):
        print(f"[SQL]\n{r['sql']}")
    if r.get("rag_results") is not None:
        docs = r["rag_results"]
        if docs:
            print(f"[KNOWLEDGE] {len(docs)} docs retrieved:")
            for d in docs:
                print(f"  - {d['title']} ({d['source']}, {d['year']})")
        else:
            print("[KNOWLEDGE] No relevant documents found.")
    if r.get("error"):
        print(f"[ERROR] {r['error']}")
    else:
        print(f"[Answer]\n{r['answer']}")
    if r.get("sql_rows") is not None:
        print(f"[Rows returned: {len(r['sql_rows'])}]")
    print()
