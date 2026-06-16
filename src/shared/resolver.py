"""
src/shared/resolver.py
Shared exercise name resolver.

Resolves a colloquial or partial exercise name to its exact database name
using the same 5-tier matching logic as the MCP resolve_exercise_name tool.
"""

import difflib
import re
import sqlite3


def resolve_exercise_name(query: str, db_path: str) -> dict:
    """
    Resolve a colloquial exercise name to its exact database name.
    Returns {"candidates": [...], "match": "..." or None}
    Same 5-tier logic as the MCP tool.
    Uses db_path to query the exercise table directly.
    """
    conn = _connect(db_path)
    try:
        return _resolve(query, conn)
    finally:
        conn.close()


def _connect(db_path: str) -> sqlite3.Connection:
    normalized = db_path.replace("\\", "/")
    uri = f"file:{normalized}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _like_escape(term: str) -> str:
    """
    Escape LIKE metacharacters so a literal term matches literally. In SQL LIKE,
    '%' and '_' are wildcards; an un-escaped '_' in a term ("Wrist_Curl") would
    match any character. Escaped with a backslash; the LIKE clauses pass
    ESCAPE '\\'. Backslash itself is escaped first so it can be the escape char.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _resolve(query: str, conn: sqlite3.Connection) -> dict:
    # Input guard: empty / whitespace-only / too-short (after strip) is NOT a
    # disambiguation case — it's "no exercise specified". Return a clean no-match
    # BEFORE any broad LIKE match. Without this, "" survived to Tier 2 as
    # LIKE '%%', matched everything, and dumped 8 arbitrary candidates — the user
    # saw "multiple exercises matching ****". The caller treats
    # {match: None, candidates: []} as no-exercise, not a clarify prompt.
    if len((query or "").strip()) < 2:
        return {"match": None, "candidates": []}

    # Tier 0: space-normalized exact match — handles compound word variations
    # ("skullcrusher" → "skull crusher", "lateralraise" → "lateral raise", etc.)
    user_term_nospace = query.lower().replace(" ", "")
    cursor = conn.execute(
        "SELECT name FROM exercise WHERE REPLACE(LOWER(name), ' ', '') = ?",
        (user_term_nospace,),
    )
    rows = cursor.fetchall()
    if len(rows) == 1:
        return {"match": rows[0]["name"], "candidates": []}
    if len(rows) > 1:
        return {"match": None, "candidates": [r["name"] for r in rows]}

    # Tier 1: exact case-insensitive match
    cursor = conn.execute(
        "SELECT name FROM exercise WHERE LOWER(name) = LOWER(?)", (query,)
    )
    row = cursor.fetchone()
    if row:
        return {"match": row["name"], "candidates": []}

    # Equipment token pre-filter: restrict to exercises containing the same
    # equipment token (EZ, KB, DB, BB, KG) when the query includes one.
    _EQUIPMENT_TOKENS = ["EZ", "KB", "DB", "BB", "KG"]
    equipment_token = next(
        (tok for tok in _EQUIPMENT_TOKENS
         if re.search(r"(?<![A-Za-z])" + tok + r"(?![A-Za-z])", query, re.IGNORECASE)),
        None,
    )

    def _filter_by_token(names):
        if not equipment_token:
            return names
        return [
            n for n in names
            if re.search(
                r"(?<![A-Za-z])" + equipment_token + r"(?![A-Za-z])", n, re.IGNORECASE
            )
        ]

    # Tier 2: partial LIKE match (term escaped so % / _ match literally)
    cursor = conn.execute(
        "SELECT name FROM exercise WHERE LOWER(name) LIKE LOWER(?) ESCAPE '\\' ORDER BY name LIMIT 8",
        (f"%{_like_escape(query)}%",),
    )
    candidates = _filter_by_token([r["name"] for r in cursor.fetchall()])
    if candidates:
        return {"match": None, "candidates": candidates}

    # Tier 3: plural/singular expansion + word-by-word matching.
    # For each word, also try flipping its trailing-s.
    # Only keep exercises where at least 2 query words match (1 for single-word queries).
    def _expand_queries(term):
        variants = [term]
        words = term.split()
        for i, word in enumerate(words):
            flipped = word[:-1] if word.lower().endswith("s") else word + "s"
            variant = " ".join(words[:i] + [flipped] + words[i + 1:])
            if variant != term:
                variants.append(variant)
        return list(dict.fromkeys(variants))

    def _dedup(names):
        seen = {}
        for n in names:
            seen.setdefault(n, None)
        return list(seen)

    raw = []
    for variant in _expand_queries(query):
        words = variant.split()
        words_lower = [w.lower() for w in words]
        min_word_matches = min(2, len(words_lower))
        if words:
            placeholders = " OR ".join(["LOWER(name) LIKE LOWER(?) ESCAPE '\\'"] * len(words))
            params = tuple(f"%{_like_escape(w)}%" for w in words)
            cursor = conn.execute(
                f"SELECT DISTINCT name FROM exercise WHERE {placeholders} ORDER BY name LIMIT 8",
                params,
            )
            raw.extend(
                n for n in (r["name"] for r in cursor.fetchall())
                if sum(1 for w in words_lower if w in n.lower()) >= min_word_matches
            )
    candidates = _filter_by_token(_dedup(raw))[:8]
    if candidates:
        return {"match": None, "candidates": candidates}

    # Tier 4: fuzzy character-level match using difflib.SequenceMatcher.
    # Compares space-stripped lowercase strings to handle typos and abbreviations.
    query_nospace = query.lower().replace(" ", "")
    all_names = conn.execute("SELECT name FROM exercise ORDER BY name").fetchall()
    scored = sorted(
        (
            (
                difflib.SequenceMatcher(
                    None, query_nospace, row["name"].lower().replace(" ", "")
                ).ratio(),
                row["name"],
            )
            for row in all_names
        ),
        key=lambda x: -x[0],
    )
    candidates = _filter_by_token([name for ratio, name in scored if ratio >= 0.75][:5])
    if candidates:
        return {"match": None, "candidates": candidates}

    return {"match": None, "candidates": []}
