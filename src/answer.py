import os
from openai import OpenAI
from src.rag import FitnessKnowledgeBase

_knowledge_base: FitnessKnowledgeBase | None = None

_RAG_SYSTEM = (
    "You are a helpful fitness coach. "
    "Answer the user's question using the provided research documents. "
    "Synthesize and summarize — do not copy chunks verbatim. "
    "Cite sources by title where relevant (e.g., 'according to Smith et al.'). "
    "If no documents were retrieved, say so honestly and answer from general knowledge. "
    "Weights in lbs, not kg. Be direct and friendly."
)

_BOTH_SYSTEM = (
    "You are a helpful fitness coach with access to both the user's personal training data "
    "and fitness research. Combine them into one coherent, actionable answer. "
    "Lead with the user's specific numbers from the database, then provide the research context. "
    "Example: 'Your weekly triceps volume is 51 sets, which is above the commonly cited upper "
    "threshold of 20-25 sets per week (Smith et al., 2022). This may indicate you are approaching "
    "a recovery ceiling.' "
    "Cite sources by title. Synthesize — do not reproduce large text blocks verbatim. "
    "Weights in lbs, not kg. Be direct and friendly."
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _get_knowledge_base() -> FitnessKnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        chroma_path = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
        _knowledge_base = FitnessKnowledgeBase(chroma_path=chroma_path)
    return _knowledge_base


def _format_rows(rows: list | None) -> str:
    if not rows:
        return "(no rows)"
    lines = []
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items()]
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _documents_are_relevant(question: str, rag_results: list[dict]) -> bool:
    """
    Asks the LLM whether the retrieved documents actually address the question.
    Returns True if relevant, False if not.
    Falls back to True on error (don't block on judge failure).
    """
    doc_lines = []
    for i, doc in enumerate(rag_results, 1):
        snippet = doc.get("text", "")[:100]
        doc_lines.append(f"{i}. {doc.get('title', '')}: {snippet}")

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved documents:\n" + "\n".join(doc_lines) + "\n\n"
        "Assess whether these documents would help answer the question directly.\n"
        "A document is relevant ONLY if it contains information that specifically addresses what is being asked.\n"
        "Topically adjacent documents that discuss related but different subjects do NOT count as relevant.\n\n"
        "Examples of NOT relevant:\n"
        "- Question asks about pre-workout supplements → documents are about post-workout supplements (different timing, different purpose)\n"
        "- Question asks about triceps volume → documents are about general periodization\n"
        "- Question asks about deadlift form → documents are about squat biomechanics\n\n"
        "Answer YES only if at least 2 of the retrieved documents directly address the specific question.\n"
        "Answer NO if the documents are mostly about adjacent topics that don't answer the question.\n\n"
        "Answer only: YES or NO"
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        if "YES" in answer:
            return True
        if "NO" in answer:
            return False
        return True
    except Exception:
        return True


def compose_answer(
    question: str,
    route: str,
    sql_answer: str | None,
    sql_rows: list | None,
    rag_results: list | None,
) -> str:
    # SQL-only: Stage 1's explain_result already composed the answer — return as-is
    if route == "sql":
        return sql_answer or "(No SQL answer available.)"

    has_rag = bool(rag_results)

    if route == "rag":
        if not has_rag:
            return (
                "I couldn't find relevant research in my fitness knowledge base for this "
                "question. Try rephrasing with more specific fitness terminology."
            )
        kb = _get_knowledge_base()
        user_content = (
            f"Question: {question}\n\n"
            f"Research context:\n{kb.format_context(rag_results)}"
        )
        system = _RAG_SYSTEM
    else:  # both
        kb = _get_knowledge_base()
        rows_text = _format_rows(sql_rows)
        if has_rag:
            knowledge_block = kb.format_context(rag_results)
        else:
            knowledge_block = (
                "NO RESEARCH DOCUMENTS WERE RETRIEVED from the fitness knowledge corpus. "
                "Do not fabricate, invent, or reference any studies, papers, or authors. "
                "Answer using only the SQL data above and acknowledge honestly that no "
                "supporting research is available in the current knowledge base."
            )
        user_content = (
            f"Question: {question}\n\n"
            f"Personal training data (from database):\n{sql_answer or 'No SQL data available.'}\n\n"
            f"Raw data rows:\n{rows_text}\n\n"
            f"Research context:\n{knowledge_block}"
        )
        system = _BOTH_SYSTEM

    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.5,
        max_tokens=1500,
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(question: str, db_path: str) -> dict:
    from src.router import classify_question
    from src.text_to_sql import answer_question as sql_pipeline

    result: dict = {
        "question": question,
        "route": "both",
        "sql": None,
        "sql_rows": None,
        "rag_results": None,
        "answer": "",
        "error": None,
    }

    try:
        route = classify_question(question)
        result["route"] = route

        sql_answer: str | None = None
        sql_rows: list | None = None

        if route in ("sql", "both"):
            sql_result = sql_pipeline(question, db_path)
            if sql_result.get("error"):
                result["error"] = sql_result["error"]
                return result
            result["sql"] = sql_result.get("sql")
            sql_rows = sql_result.get("rows", [])
            result["sql_rows"] = sql_rows
            sql_answer = sql_result.get("answer")

        rag_results: list | None = None
        if route in ("rag", "both"):
            kb = _get_knowledge_base()
            rag_results = kb.retrieve(question)
            if rag_results:
                if _documents_are_relevant(question, rag_results):
                    print("[RAG] Relevance gate: documents passed")
                else:
                    print("[RAG] Relevance gate: documents rejected as off-topic for this question")
                    rag_results = []
            result["rag_results"] = rag_results if rag_results else []

        result["answer"] = compose_answer(question, route, sql_answer, sql_rows, rag_results)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result
