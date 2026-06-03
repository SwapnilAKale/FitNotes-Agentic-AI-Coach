import os

CHROMA_MEMORY_PATH = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
MEMORY_COLLECTION = "user_memory"

_embed_model = None
_chroma_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_MEMORY_PATH)
        _chroma_collection = client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


def retrieve_relevant_memories(question: str) -> list[str]:
    """
    Retrieve memory facts relevant to the question.
    Read-only — no writes. Memory writes stay on the single agent.

    Uses ChromaDB user_memory collection.
    Embeds the question, retrieves top 5 facts with cosine
    distance < 0.8 (same threshold as src/memory.py Option B).

    Returns a list of fact strings, empty list if none found or
    if memory is unavailable.
    """
    try:
        collection = _get_chroma_collection()
        model = _get_embed_model()

        total = collection.count()
        if total == 0:
            return []

        n = min(5, total)
        query_embedding = model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "distances"]
        )

        facts = []
        for i in range(len(results["ids"][0])):
            if results["distances"][0][i] < 0.8:
                facts.append(results["documents"][0][i])

        return facts
    except Exception:
        return []
