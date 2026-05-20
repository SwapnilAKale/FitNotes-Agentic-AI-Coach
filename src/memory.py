import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import chromadb

CHROMA_MEMORY_PATH = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
MEMORY_COLLECTION = "user_memory"

_embed_model = None
_chroma_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        import os
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_MEMORY_PATH)
        _chroma_collection = client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection

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
    try:
        embed_and_store_fact(fact)
    except Exception:
        pass  # ChromaDB failure never blocks fact storage
    return {"status": "saved", "fact": fact}


def get_all_facts() -> list:
    return load_memory().get("facts", [])


def delete_fact(fact_id: str) -> dict:
    memory = load_memory()
    before = len(memory["facts"])
    memory["facts"] = [f for f in memory["facts"] if f["id"] != fact_id]
    if len(memory["facts"]) < before:
        save_memory(memory)
        try:
            collection = _get_chroma_collection()
            collection.delete(ids=[fact_id])
        except Exception:
            pass
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


def embed_and_store_fact(fact: dict):
    """Embed a fact and store in ChromaDB. Skips if already stored."""
    collection = _get_chroma_collection()
    model = _get_embed_model()

    fact_id = fact["id"]

    existing = collection.get(ids=[fact_id])
    if existing["ids"]:
        return

    embedding = model.encode(fact["content"]).tolist()
    collection.add(
        ids=[fact_id],
        embeddings=[embedding],
        documents=[fact["content"]],
        metadatas=[{
            "category": fact.get("category", ""),
            "source": fact.get("source", ""),
            "confidence": fact.get("confidence", "high"),
            "created_at": fact.get("created_at", "")
        }]
    )


def sync_to_chromadb():
    """Sync all facts in memory.json to ChromaDB.
    Adds missing facts. Removes deleted facts from ChromaDB."""
    facts = get_all_facts()
    if not facts:
        return

    collection = _get_chroma_collection()

    for fact in facts:
        embed_and_store_fact(fact)

    current_ids = {f["id"] for f in facts}
    all_in_chroma = collection.get()
    for chroma_id in all_in_chroma["ids"]:
        if chroma_id not in current_ids:
            collection.delete(ids=[chroma_id])


def retrieve_relevant_memories(question: str, n_results: int = 5) -> list[dict]:
    """Retrieve the most relevant stored memories for a given question."""
    collection = _get_chroma_collection()
    model = _get_embed_model()

    total = collection.count()
    if total == 0:
        return []

    n = min(n_results, total)
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    relevant = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        if distance < 0.8:
            relevant.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category", ""),
                "distance": distance
            })

    return relevant


def format_relevant_memories_for_prompt(question: str) -> str:
    """Retrieve relevant memories and format for system prompt injection.
    Returns empty string if no relevant memories found."""
    memory = load_memory()
    summary = memory.get("summary")

    relevant = retrieve_relevant_memories(question)

    if not relevant and not summary:
        return ""

    lines = ["RELEVANT MEMORIES FROM PREVIOUS SESSIONS:"]

    if summary:
        lines.append(f"(Earlier sessions): {summary}")

    for fact in relevant:
        lines.append(f"- [{fact['category']}] {fact['content']}")

    return "\n".join(lines)
