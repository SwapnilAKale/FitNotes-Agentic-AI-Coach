import re
from src.db import get_connection, run_query
from src.llm import generate_sql, explain_result
from src.schema_prompt import build_schema_prompt

_schema_prompt_cache: str | None = None


def _get_schema_prompt() -> str:
    global _schema_prompt_cache
    if _schema_prompt_cache is None:
        _schema_prompt_cache = build_schema_prompt()
    return _schema_prompt_cache


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:sql)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def _validate_sql(sql: str) -> None:
    normalized = sql.lstrip().upper()
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        raise ValueError(
            f"Rejected: SQL must start with SELECT or WITH. Got: {sql[:80]!r}"
        )


def answer_question(question: str, db_path: str) -> dict:
    result: dict = {
        "question": question,
        "sql": "",
        "rows": [],
        "answer": "",
        "error": None,
    }
    try:
        conn = get_connection(db_path)
        schema_prompt = _get_schema_prompt()

        raw_sql = generate_sql(question, schema_prompt)
        sql = _strip_fences(raw_sql)
        _validate_sql(sql)
        result["sql"] = sql

        rows = run_query(conn, sql)
        result["rows"] = rows

        answer = explain_result(question, sql, rows)
        if not answer:
            answer = "(The model returned an empty explanation. The SQL ran successfully — check the rows above.)"
        result["answer"] = answer

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result
