import os
from collections import defaultdict

_kb = None


def _get_kb():
    global _kb
    if _kb is None:
        from src.rag import FitnessKnowledgeBase
        _kb = FitnessKnowledgeBase(
            chroma_path=os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
        )
    return _kb


def _get_article_boundary_chunks(collection, filename: str) -> list[dict]:
    """Return first chunk + last 3 chunks of a user article (mirrors combined_server)."""
    try:
        all_chunks = collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"]
        )
        if not all_chunks["ids"]:
            return []
        chunks = sorted(
            zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"]),
            key=lambda x: x[2].get("chunk_index", 0)
        )
        boundary_indices = [0]
        if len(chunks) > 1:
            for i in range(max(1, len(chunks) - 3), len(chunks)):
                boundary_indices.append(i)
        return [{"text": chunks[i][1]} for i in boundary_indices]
    except Exception:
        return []


def search_fitness_knowledge(question: str) -> list | None:
    """
    Search the fitness knowledge base for content relevant to the question.

    Returns a list of result dicts matching the format expected by
    analysis_agent._fmt_research():
        [
            {
                "source_type": "user_article" | "pubmed" |
                               "wikipedia" | "general",
                "instruction": str | None,
                "documents": [{"title": str, "content": str}]
            }
        ]

    Returns None if nothing relevant found.
    Never raises — swallows all errors and returns None.
    """
    try:
        kb = _get_kb()
        results = kb.retrieve(question, n_results=5)

        if not results:
            return None

        # Find user article filenames in reranked results
        user_article_filenames = set(
            doc.get("metadata", {}).get("filename") or doc.get("title")
            for doc in results
            if doc.get("source") == "user_article"
            or doc.get("metadata", {}).get("source_type") == "user_article"
        )

        # Append first + last chunks for each user article found
        try:
            import chromadb
            chroma_path = os.environ.get("CHROMA_DB_PATH", "data/chroma_db")
            client = chromadb.PersistentClient(path=chroma_path)
            user_col = client.get_collection("user_articles")
            for filename in user_article_filenames:
                if filename:
                    for chunk in _get_article_boundary_chunks(user_col, filename):
                        results.append({
                            "title": filename,
                            "source": "user_article",
                            "year": "",
                            "url": "",
                            "text": chunk["text"],
                        })
        except Exception:
            pass

        # Three-tier instruction
        if user_article_filenames:
            instruction = (
                "USER ARTICLE FOUND — Lead with the study conclusion. "
                "Do NOT answer from general fitness knowledge if the article "
                "directly answers the question."
            )
        else:
            instruction = (
                "Note: No study in your personal knowledge base covers this topic. "
                "The following is based on general research literature."
            )

        # Group documents by source type, priority: user_article > pubmed > wikipedia > general
        groups: dict[str, list] = defaultdict(list)
        for doc in results:
            source = doc.get("source", "general")
            if source not in ("pubmed", "wikipedia", "user_article"):
                source = "general"
            text = doc.get("text", "")
            groups[source].append({
                "title": doc.get("title", ""),
                "content": text[:6000] if len(text) > 6000 else text,
            })

        priority = {"user_article": 0, "pubmed": 1, "wikipedia": 2, "general": 3}
        sorted_groups = sorted(groups.items(), key=lambda x: priority.get(x[0], 99))

        result_list = [
            {
                "source_type": src_type,
                "instruction": instruction if i == 0 else None,
                "documents": docs_list,
            }
            for i, (src_type, docs_list) in enumerate(sorted_groups)
        ]

        return result_list or None

    except Exception:
        return []
