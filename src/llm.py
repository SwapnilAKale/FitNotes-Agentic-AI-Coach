import json as _json
import os
import re as _re
import time as _time

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite"

_client = None

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


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


def _call(system: str, user: str, temperature: float = 0.0,
          max_tokens: int = 2048) -> str:
    """Single synchronous Gemini call."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=user)]
            )],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        parts = response.candidates[0].content.parts
        return parts[0].text if parts else ""
    except Exception as e:
        return f"ERROR: {e}"


def generate_sql(question: str, schema_prompt: str) -> str:
    user_content = f"Schema:\n{schema_prompt}\n\nQuestion: {question}"
    for attempt in range(3):
        sql = _call(_SQL_SYSTEM, user_content, temperature=0.1 + attempt * 0.2, max_tokens=2048)
        if sql and not sql.startswith("ERROR"):
            return sql.strip()
    return ""


def explain_result(question: str, sql: str, rows: list[dict]) -> str:
    rows_text = str(rows[:20]) if rows else "[]"
    if len(rows) > 20:
        rows_text = f"{str(rows[:20])} ... ({len(rows)} rows total, showing first 20)"
    user_content = (
        f"Question: {question}\n\n"
        f"SQL run:\n{sql}\n\n"
        f"Result rows:\n{rows_text}"
    )
    for attempt in range(2):
        content = _call(_EXPLAIN_SYSTEM, user_content, temperature=0.5 + attempt * 0.2, max_tokens=512)
        if content and not content.startswith("ERROR"):
            return content.strip()
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
    result = _call(_REWRITE_SYSTEM, question, temperature=0.1, max_tokens=50)
    if not result or result.startswith("ERROR"):
        return question
    return result.strip()


# ---------------------------------------------------------------------------
# LLM-as-judge (added in Stage 1.5 — do not modify functions above)
# ---------------------------------------------------------------------------

def judge_answer(question: str, ground_truth_answer: str, system_answer: str) -> dict:
    """Uses the LLM as a judge. Returns {"correct": bool, "reasoning": str}."""
    if not os.environ.get("GEMINI_API_KEY"):
        return {"correct": False, "reasoning": "GEMINI_API_KEY not set"}

    user_content = (
        f"Question: {question}\n\n"
        f"Ground-truth answer: {ground_truth_answer}\n\n"
        f"System answer: {system_answer}"
    )

    def _extract(content: str) -> dict | None:
        cleaned = _re.sub(r"^```(?:json)?\s*", "", content, flags=_re.IGNORECASE)
        cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
        for candidate in (cleaned, content):
            try:
                parsed = _json.loads(candidate)
                return {
                    "correct": bool(parsed.get("correct", False)),
                    "reasoning": str(parsed.get("reasoning", "")),
                }
            except _json.JSONDecodeError:
                pass
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
        raw = _call(_JUDGE_SYSTEM, user_content, temperature=0.0, max_tokens=256)
        if raw.startswith("ERROR"):
            return {"correct": False, "reasoning": f"judge call failed: {raw}"}
        result = _extract(raw)
        if result is not None:
            return result
        if attempt == 0:
            _time.sleep(5)

    return {"correct": False, "reasoning": "judge output unparseable"}
