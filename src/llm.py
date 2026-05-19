import os
from openai import OpenAI

_client: OpenAI | None = None
MODEL = "openai/gpt-oss-120b"

_SQL_SYSTEM = (
    "You are a SQL expert for a SQLite fitness database. "
    "Given a natural-language question and a schema description, output ONLY a raw SQL SELECT query. "
    "No markdown fences. No explanations. No comments. No prose. "
    "Just the SQL statement, ending with a semicolon."
)

_EXPLAIN_SYSTEM = (
    "You are a helpful fitness coach assistant. "
    "Given a user question, the SQL that was run, and the query results, "
    "write a concise natural-language answer. "
    "Do not modify non-weight values (reps, counts, dates, durations, distances). "
    "If the result is empty, clearly state that no data was found. "
    "If there are more than 20 rows, summarize trends rather than listing every row. "
    "Do not repeat the SQL. Be direct and friendly.\n\n"
    "UNIT DISPLAY RULES:\n"
    "The SQL query already handles all unit conversion. Your job is only to label and display correctly.\n\n"
    "EXERCISE-LEVEL UNIT OVERRIDE (takes precedence over column names):\n"
    "These exercises are always reported in kg regardless of column naming:\n"
    "  - Deadlift (from 2025-12-26 onwards)\n"
    "  - Seated Machine Curl (Kg)\n"
    "  - Machine Wrist Extension\n"
    "  - Hand Gripper\n"
    "When rows contain data for these exercises, report the weight values in kg.\n"
    "Do not convert to lbs. Do not apply 2.2046 multiplication.\n"
    "The SQL has already computed the correct value — just label it kg.\n\n"
    "For all other exercises: apply the normal column-name rules below.\n\n"
    "If the SQL column is named weight_lbs or total_lbs: display the value with 'lbs' label. Do not convert.\n"
    "If the SQL column is named weight_kg or total_kg: display the value with 'kg' label. Do not convert.\n"
    "Round weight values to a sensible precision: whole numbers for heavy lifts, one decimal for lighter weights.\n\n"
    "Never apply any additional multiplication or division to the value the SQL already returned.\n"
    "Never convert a kg result to lbs or a lbs result to kg in your answer.\n"
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def generate_sql(question: str, schema_prompt: str) -> str:
    client = _get_client()
    user_content = f"Schema:\n{schema_prompt}\n\nQuestion: {question}"
    for attempt in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SQL_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1 + attempt * 0.2,
            max_tokens=2048,
        )
        sql = (response.choices[0].message.content or "").strip()
        if sql:
            return sql
    return ""


def explain_result(question: str, sql: str, rows: list[dict]) -> str:
    client = _get_client()
    rows_text = str(rows[:20]) if rows else "[]"
    if len(rows) > 20:
        rows_text = f"{str(rows[:20])} ... ({len(rows)} rows total, showing first 20)"
    user_content = (
        f"Question: {question}\n\n"
        f"SQL run:\n{sql}\n\n"
        f"Result rows:\n{rows_text}"
    )
    messages = [
        {"role": "system", "content": _EXPLAIN_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(2):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5 + attempt * 0.2,
            max_tokens=512,
        )
        content = (response.choices[0].message.content or "").strip()
        if content:
            return content
    return ""


def rewrite_query_for_retrieval(question: str) -> str:
    """
    Rewrites a conversational fitness question into technical language
    that matches the phrasing of academic abstracts and Wikipedia articles.
    Returns the rewritten query string. Falls back to original question on error.
    """
    _REWRITE_SYSTEM = (
        "You are a search query specialist for fitness and exercise science literature.\n"
        "Rewrite the given question as a concise technical search query (8-15 words)\n"
        "that would match language used in peer-reviewed research abstracts or academic articles.\n"
        "Replace casual terms with scientific equivalents.\n"
        "Output ONLY the rewritten query. No explanation, no punctuation at the end.\n\n"
        "Examples:\n"
        '"How many sets per week is optimal for triceps?"\n'
        '→ "weekly resistance training volume triceps hypertrophy dose response"\n\n'
        '"Is my chest progress normal?"\n'
        '→ "pectoralis major strength gains novice resistance training progression rate"\n\n'
        '"Should I take a deload week?"\n'
        '→ "deload week resistance training recovery performance fatigue management"\n\n'
        '"What does DOMS mean?"\n'
        '→ "delayed onset muscle soreness mechanisms resistance exercise"'
    )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# LLM-as-judge (added in Stage 1.5 — do not modify functions above)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a system's answer to a fitness "
    "question conveys the same factual information as a known ground-truth answer.\n\n"
    "Rules:\n"
    "1. Mark CORRECT if both answers convey the same core facts, even with different phrasing.\n"
    "2. Weights may appear in kg OR lbs (1 kg = 2.2046 lbs). A weight that matches after "
    "unit conversion within 1% is EQUIVALENT — do not penalise unit differences.\n"
    "3. Allow rounding differences within 1% of the ground-truth value.\n"
    "4. Additional commentary or context in the system answer does not make it incorrect.\n"
    "5. Mark INCORRECT if a numeric value differs by more than 1% (after unit conversion), "
    "the wrong exercise or date is named, or a key fact is entirely missing.\n\n"
    "Respond ONLY with valid JSON on a single line:\n"
    "{\"correct\": true, \"reasoning\": \"brief explanation\"}\n\n"
    "--- Examples ---\n\n"
    "Question: \"What is my PR for Lat Pulldown?\"\n"
    "Ground truth: \"Your Lat Pulldown PR is 58.97 kg.\"\n"
    "System answer: \"You hit a personal record on Lat Pulldown at 130 lbs, set on 2024-11-03.\"\n"
    "Response: {\"correct\": true, \"reasoning\": \"130 lbs = 58.97 kg × 2.2046. Same fact, different units.\"}\n\n"
    "Question: \"What is my PR for Lat Pulldown?\"\n"
    "Ground truth: \"Your Lat Pulldown PR is 58.97 kg.\"\n"
    "System answer: \"Your Lat Pulldown PR is 62 kg.\"\n"
    "Response: {\"correct\": false, \"reasoning\": \"62 kg differs from 58.97 kg by more than 1%.\"}"
)


def judge_answer(question: str, ground_truth_answer: str, system_answer: str) -> dict:
    """Uses the LLM as a judge. Returns {"correct": bool, "reasoning": str}."""
    import json as _json
    import re as _re
    import time as _time
    import google.genai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"correct": False, "reasoning": "GEMINI_API_KEY not set"}

    client = genai.Client(api_key=api_key)

    user_content = (
        f"Question: {question}\n\n"
        f"Ground-truth answer: {ground_truth_answer}\n\n"
        f"System answer: {system_answer}"
    )

    def _call_gemini() -> tuple[str | None, Exception | None]:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_content,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_JUDGE_SYSTEM,
                    temperature=0,
                    max_output_tokens=256,
                ),
            )
            return (response.text or "").strip(), None
        except Exception as exc:
            return None, exc

    def _extract(content: str) -> dict | None:
        # Strip markdown fences (```json ... ``` or ``` ... ```)
        cleaned = _re.sub(r"^```(?:json)?\s*", "", content, flags=_re.IGNORECASE)
        cleaned = _re.sub(r"\s*```$", "", cleaned).strip()

        # Try the cleaned string first, then fall back to the raw content
        for candidate in (cleaned, content):
            try:
                parsed = _json.loads(candidate)
                return {
                    "correct": bool(parsed.get("correct", False)),
                    "reasoning": str(parsed.get("reasoning", "")),
                }
            except _json.JSONDecodeError:
                pass

        # Aggressively extract the first {...} block from whatever remains
        match = _re.search(r"\{[^{}]+\}", content, _re.DOTALL)
        if match:
            try:
                parsed = _json.loads(match.group())
                return {
                    "correct": bool(parsed.get("correct", False)),
                    "reasoning": str(parsed.get("reasoning", "")),
                }
            except _json.JSONDecodeError:
                pass

        return None

    for attempt in range(2):
        raw, exc = _call_gemini()
        if exc is not None:
            return {"correct": False, "reasoning": f"judge call failed: {exc}"}

        result = _extract(raw)
        if result is not None:
            return result

        if attempt == 0:
            _time.sleep(5)

    return {"correct": False, "reasoning": "judge output unparseable"}
