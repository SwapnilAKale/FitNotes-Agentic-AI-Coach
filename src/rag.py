import re

COLLECTION_NAME = "fitness_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_THRESHOLD = 0.0


def _tokenize(text: str) -> list[str]:
    return re.split(r"[\s\W]+", text.lower())


class FitnessKnowledgeBase:
    def __init__(self, chroma_path: str = "./data/chroma_db"):
        self._chroma_path = chroma_path
        self._collection = None
        self._embed_model = None
        self._reranker = None
        self._bm25_index = None
        self._bm25_corpus = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer, CrossEncoder
            from rank_bm25 import BM25Okapi

            client = chromadb.PersistentClient(path=self._chroma_path)
            try:
                self._collection = client.get_collection(name=COLLECTION_NAME)
            except Exception:
                print("Warning: ChromaDB collection not found. Run scripts/build_corpus.py first.")
                return

            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
            self._reranker = CrossEncoder(RERANKER_MODEL)

            all_docs = self._collection.get(include=["documents", "metadatas"])
            texts = all_docs["documents"]
            metas = all_docs["metadatas"]

            self._bm25_corpus = [
                {
                    "text": texts[i],
                    "title": metas[i].get("title", ""),
                    "url": metas[i].get("url", ""),
                    "source": metas[i].get("source", ""),
                    "year": metas[i].get("year", ""),
                    "topic": metas[i].get("topic", ""),
                }
                for i in range(len(texts))
            ]

            tokenized = [_tokenize(t) for t in texts]
            self._bm25_index = BM25Okapi(tokenized)

            print(
                f"Knowledge base loaded: {len(texts)} documents, "
                "BM25 index ready, reranker ready"
            )
        except ImportError as e:
            print(f"Warning: missing dependency for RAG — {e}")

    def _rewrite_query(self, query: str) -> str:
        from src.llm import rewrite_query_for_retrieval
        return rewrite_query_for_retrieval(query)

    def _hybrid_search(self, query: str, n_candidates: int = 20) -> list[dict]:
        # --- Dense (semantic) search ---
        embedding = self._embed_model.encode(query).tolist()
        sem_results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        sem_ids = sem_results["ids"][0]
        sem_texts = sem_results["documents"][0]
        sem_metas = sem_results["metadatas"][0]

        # Build rank map for semantic results (rank 1 = best)
        sem_rank: dict[str, int] = {doc_id: rank + 1 for rank, doc_id in enumerate(sem_ids)}

        # Build a lookup by doc id for text/metadata
        doc_lookup: dict[str, dict] = {}
        for i, doc_id in enumerate(sem_ids):
            doc_lookup[doc_id] = {
                "text": sem_texts[i],
                "title": sem_metas[i].get("title", ""),
                "url": sem_metas[i].get("url", ""),
                "source": sem_metas[i].get("source", ""),
                "year": sem_metas[i].get("year", ""),
                "topic": sem_metas[i].get("topic", ""),
            }

        # --- BM25 (keyword) search ---
        tokenized_query = _tokenize(query)
        bm25_scores = self._bm25_index.get_scores(tokenized_query)

        # Get all ChromaDB doc ids in corpus order
        all_ids = self._collection.get(include=[])["ids"]

        # Rank BM25 results by score descending, take top n_candidates
        scored = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_rank: dict[str, int] = {}
        for rank, (idx, _score) in enumerate(scored[:n_candidates]):
            doc_id = all_ids[idx]
            bm25_rank[doc_id] = rank + 1
            if doc_id not in doc_lookup and idx < len(self._bm25_corpus):
                doc_lookup[doc_id] = self._bm25_corpus[idx]

        # --- Reciprocal Rank Fusion ---
        all_doc_ids = set(sem_rank.keys()) | set(bm25_rank.keys())
        rrf_scores: list[tuple[str, float]] = []
        for doc_id in all_doc_ids:
            r_sem = sem_rank.get(doc_id, n_candidates + 1)
            r_bm25 = bm25_rank.get(doc_id, n_candidates + 1)
            rrf = 1 / (60 + r_sem) + 1 / (60 + r_bm25)
            rrf_scores.append((doc_id, rrf))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for doc_id, rrf_score in rrf_scores[:n_candidates]:
            if doc_id not in doc_lookup:
                continue
            entry = dict(doc_lookup[doc_id])
            entry["rrf_score"] = rrf_score
            results.append(entry)

        return results

    def _rerank(self, query: str, candidates: list[dict], n_results: int = 5) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        filtered = [
            {**doc, "score": float(score)}
            for score, doc in ranked
            if score >= RERANKER_THRESHOLD
        ]
        return filtered[:n_results]

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        if not self._loaded:
            self._load()

        if self._collection is None or self._bm25_index is None:
            return []

        total_docs = self._collection.count()
        if total_docs == 0:
            return []

        technical_query = self._rewrite_query(query)
        print(f'[RAG] Original: "{query}" → Rewritten: "{technical_query}"')

        n_candidates = min(20, total_docs)
        candidates = self._hybrid_search(technical_query, n_candidates=n_candidates)
        return self._rerank(technical_query, candidates, n_results=n_results)

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return "No relevant research found in fitness knowledge base for this question."
        lines = [f"Retrieved {len(results)} documents from fitness research corpus.\n"]
        for i, doc in enumerate(results, 1):
            lines.append(f"[{i}] {doc['title']} ({doc['year']}) — {doc['source']}")
            lines.append(doc["text"])
            lines.append("")
        return "\n".join(lines)
