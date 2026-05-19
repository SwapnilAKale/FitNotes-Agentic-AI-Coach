import os
from openai import OpenAI

# Uses llama-3.3-70b-versatile via the same Groq client as Stage 1.
# A non-reasoning model works better here: classification is binary signal extraction,
# not multi-step reasoning. The gpt-oss-120b reasoning model over-defaults to "both".
ROUTER_MODEL = "llama-3.3-70b-versatile"

_ROUTER_SYSTEM = """\
You route a fitness coach question to exactly one of three categories.

== Categories ==

sql
  Use when the question ONLY needs the user's personal workout history to answer.
  No fitness science or general knowledge is required.
  Signals: "my PR", "my sets", "my volume", "how many times did I", "when did I", \
"my best", "what did I lift", "my history", "my streak", "did I ever".
  Examples:
    "What is my PR for Lat Pulldown?" → sql
    "How many bench press sets did I do last week?" → sql

rag
  Use when the question ONLY needs general fitness science or principles to answer.
  No database lookup is needed — the user is NOT asking about their own numbers.
  Signals: "what is", "how does", "explain", "what does X mean", "how many sets should",
           "how long should I rest", "what is the research on", "is X good for Y".
  Examples:
    "What is progressive overload?" → rag
    "How does muscle hypertrophy work?" → rag

both
  Use ONLY when the question requires the user's personal numbers AND fitness science
  to give a complete, meaningful answer.
  The user wants their specific data interpreted through a scientific lens.
  Examples:
    "Given my recent triceps volume, am I overtraining?" → both
    "Is my bench press progress normal compared to what research says?" → both

== Rules ==
- A question with "my" that asks for a specific personal number (PR, set count, date) → sql.
- A question with no reference to the user's own data → rag.
- A question asking to EVALUATE or COMPARE the user's data against science → both.
- Default to "both" only when genuinely ambiguous.

Respond with ONE word only: sql, rag, or both. No punctuation. No explanation.\
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def classify_question(question: str) -> str:
    """Classify question as 'sql', 'rag', or 'both'. Defaults to 'both' on parse failure."""
    client = _get_client()
    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=10,
    )
    raw = (response.choices[0].message.content or "").strip().lower()
    if raw in {"sql", "rag", "both"}:
        return raw
    return "both"
