import json
import uuid
from datetime import datetime
from pathlib import Path

MEMORY_PATH = Path("data/memory.json")
MAX_FACTS = 30


def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {"facts": [], "last_updated": None, "summary": None}
    with open(MEMORY_PATH) as f:
        return json.load(f)


def save_memory(memory: dict):
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def add_fact(category: str, content: str,
             source: str = "user_stated",
             confidence: str = "high") -> dict:
    memory = load_memory()

    for fact in memory["facts"]:
        if fact["content"].strip().lower() == content.strip().lower():
            return {"status": "duplicate", "message": "Already stored."}

    fact = {
        "id": str(uuid.uuid4())[:8],
        "category": category,
        "content": content,
        "source": source,
        "created_at": datetime.now().isoformat(),
        "confidence": confidence,
    }
    memory["facts"].append(fact)

    # If over cap, compress oldest 10 facts into summary
    if len(memory["facts"]) > MAX_FACTS:
        oldest = memory["facts"][:10]
        memory["facts"] = memory["facts"][10:]
        compressed = "; ".join(f["content"] for f in oldest)
        existing_summary = memory.get("summary") or ""
        memory["summary"] = (existing_summary + " | " + compressed).strip(" |")

    save_memory(memory)
    return {"status": "saved", "fact": fact}


def get_all_facts() -> list:
    return load_memory().get("facts", [])


def delete_fact(fact_id: str) -> dict:
    memory = load_memory()
    before = len(memory["facts"])
    memory["facts"] = [f for f in memory["facts"] if f["id"] != fact_id]
    if len(memory["facts"]) < before:
        save_memory(memory)
        return {"status": "deleted"}
    return {"status": "not_found", "message": f"No fact found with id {fact_id}"}


def format_memory_for_prompt() -> str:
    """Format stored memories for injection into system prompt.
    Total output kept under 600 tokens."""
    memory = load_memory()
    facts = memory.get("facts", [])
    summary = memory.get("summary")

    if not facts and not summary:
        return ""

    lines = ["WHAT I KNOW ABOUT YOU FROM PREVIOUS SESSIONS:"]

    if summary:
        lines.append(f"(Earlier sessions summary): {summary}")

    by_category: dict = {}
    for f in facts:
        by_category.setdefault(f["category"], []).append(f["content"])

    category_labels = {
        "user_fact": "Personal",
        "preference": "Preferences",
        "training_pattern": "Training Patterns",
        "injury": "Physical Notes",
        "convention": "Data Conventions",
    }

    for cat, items in by_category.items():
        label = category_labels.get(cat, cat.replace("_", " ").title())
        lines.append(f"{label}: {'; '.join(items)}")

    return "\n".join(lines)
