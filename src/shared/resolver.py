"""
src/shared/resolver.py
Shared exercise name resolver.

Resolves a colloquial or partial exercise name to its exact database name
using the same 5-tier matching logic as the MCP resolve_exercise_name tool.
"""

import difflib
import re
import sqlite3


# Permissive rank-and-pick thresholds (READ path only). These now serve ONLY the
# 0-data NAME-ONLY FALLBACK below (when no candidate has real logged data): a LIKE
# tier with multiple all-unlogged candidates auto-resolves to the difflib winner
# only when it clears both:
#   MARGIN    — top ratio must beat the runner-up by this much (a CLEAR winner)
#   MIN_FLOOR — top ratio must itself be at least this (not a weak best-of-bad-lot)
_PERMISSIVE_MARGIN    = 0.10
_PERMISSIVE_MIN_FLOOR = 0.5

# Real-data threshold (READ path only). A candidate with at least this many logged
# sets is a "real target" the user actually trains; below it an exercise has no
# analyzable history. The multi-candidate decision keys on HOW MANY candidates clear
# this floor — a corpus-independent signal, unlike a tuned name-similarity margin:
#   >= 2 real-data candidates -> genuine ambiguity -> ASK (return None, no auto-pick)
#   == 1 real-data candidate  -> the sole real target -> auto-pick it
#   == 0 real-data candidates -> fall back to the name-only margin pick above
# WRITE path (permissive=False) NEVER applies this filter: a first-time-logged
# exercise legitimately has 0 sets, so writes keep strict disambiguation.
_LOGGED_FLOOR = 5


def resolve_exercise_name(query: str, db_path: str, permissive: bool = False) -> dict:
    """
    Resolve a colloquial exercise name to its exact database name.
    Returns {"candidates": [...], "match": "..." or None}
    Same 5-tier logic as the MCP tool.
    Uses db_path to query the exercise table directly.

    permissive: READ-PATH ONLY. When True, a LIKE tier that yields multiple
    candidates auto-resolves to a clear-margin difflib winner instead of
    disambiguating. MUST stay False for writes — a silent wrong write is
    unrecoverable (the locked auto-pick-removal rule).
    """
    conn = _connect(db_path)
    try:
        return _resolve(query, conn, permissive=permissive)
    finally:
        conn.close()


def _ratio(query: str, name: str) -> float:
    """Space-stripped, lowercased difflib ratio — the shared name-similarity score."""
    return difflib.SequenceMatcher(
        None, query.lower().replace(" ", ""), name.lower().replace(" ", "")
    ).ratio()


def _logged_counts(conn: sqlite3.Connection, names: list) -> dict:
    """Logged-set count per candidate name (READ path only). One LEFT-JOIN query so
    a never-logged exercise returns 0 rather than being dropped. Read-only."""
    if not names:
        return {}
    placeholders = ",".join("?" * len(names))
    rows = conn.execute(
        f"""SELECT e.name AS name, COUNT(tl._id) AS sets
            FROM exercise e
            LEFT JOIN training_log tl ON tl.exercise_id = e._id
            WHERE e.name IN ({placeholders})
            GROUP BY e.name""",
        tuple(names),
    ).fetchall()
    return {r["name"]: r["sets"] for r in rows}


def _order_by_data(query: str, candidates: list, counts: dict) -> list:
    """Order the ask-list data-first: (logged_count desc, name-ratio desc) so the
    exercise the user actually trains leads the disambiguation prompt."""
    return sorted(
        candidates,
        key=lambda n: (-counts.get(n, 0), -_ratio(query, n)),
    )


def _permissive_pick(query: str, candidates: list, counts: dict) -> "str | None":
    """
    READ-path multi-candidate decision (bug 2.4(ii)). The signal is HOW MANY
    candidates have real logged data (>= _LOGGED_FLOOR), NOT a name-similarity margin:

      >= 2 real-data candidates -> None (genuine ambiguity -> caller ASKS)
      == 1 real-data candidate  -> that candidate (the sole real target)
      == 0 real-data candidates -> name-only fallback (clear-margin difflib winner,
                                    else None) — every pick here is data-sparse and
                                    degrades downstream to "not found -> broad" anyway.

    A single candidate can't be ambiguous, so it is returned directly.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    with_data = [c for c in candidates if counts.get(c, 0) >= _LOGGED_FLOOR]
    if len(with_data) >= 2:
        return None                      # genuine ambiguity → ASK
    if len(with_data) == 1:
        return with_data[0]              # sole real target → auto-pick

    # 0 candidates with real data → name-only margin pick (unchanged behavior).
    scored = sorted(
        ((_ratio(query, n), n) for n in candidates),
        key=lambda x: -x[0],
    )
    top_ratio, top_name = scored[0]
    second_ratio = scored[1][0]
    if top_ratio >= _PERMISSIVE_MIN_FLOOR and (top_ratio - second_ratio) >= _PERMISSIVE_MARGIN:
        return top_name
    return None


def _word_multiset(s: str) -> tuple:
    """Sorted lowercase word multiset — the categorical name-equality signal for
    the Tier-3 permutation rescue (word ORDER is the only thing it ignores)."""
    return tuple(sorted(s.lower().split()))


def _expand_queries(term: str) -> list:
    """The term plus one-word plural/singular flips (trailing-s), deduped.
    Shared by Tier 3's word matching and the permutation rescue."""
    variants = [term]
    words = term.split()
    for i, word in enumerate(words):
        flipped = word[:-1] if word.lower().endswith("s") else word + "s"
        variant = " ".join(words[:i] + [flipped] + words[i + 1:])
        if variant != term:
            variants.append(variant)
    return list(dict.fromkeys(variants))


def _permutation_match(query: str, candidates: list) -> "str | None":
    """
    Tier-3 READ-path rescue (bug 2.4): the user typing a word PERMUTATION of
    exactly one exercise name ("dumbbell flat bench press" → "Flat Dumbbell
    Bench Press") is naming THAT exercise — categorical name evidence, no tuned
    threshold (difflib ≥0.75 is NOT single here: Incline/Decline score 0.766).
    Plural-flip tolerant via _expand_queries. Returns the sole permutation
    match, or None (zero or ≥2 matches → fall through to ask/pick as before).

    Callers pass ALL exercise names, not the Tier-3 candidate pool: the Tier-3
    SQL LIMITs 8 alphabetically BEFORE the ≥2-word filter, so the true
    permutation match can be crowded out of the pool entirely (the live
    "dumbbell flat bench press" case).
    """
    variant_sets = {_word_multiset(v) for v in _expand_queries(query)}
    matches = [c for c in candidates if _word_multiset(c) in variant_sets]
    return matches[0] if len(matches) == 1 else None


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


def _resolve(query: str, conn: sqlite3.Connection, permissive: bool = False) -> dict:
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
        if permissive:
            counts = _logged_counts(conn, candidates)
            picked = _permissive_pick(query, candidates, counts)
            if picked:
                return {"match": picked, "candidates": []}
            # 2.3: the ≥2-real-data ASK offers ONLY real-data candidates —
            # never a 0-set name ("squat" must not offer Barbell Squat).
            # 0-real-data (name-only fallback) keeps the full list unchanged.
            with_data = [c for c in candidates if counts.get(c, 0) >= _LOGGED_FLOOR]
            candidates = _order_by_data(query, with_data or candidates, counts)
        return {"match": None, "candidates": candidates}

    # Tier 3: plural/singular expansion + word-by-word matching (module-level
    # _expand_queries). Only keep exercises where at least 2 query words match
    # (1 for single-word queries).
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
        if permissive:
            # 2.4 rescue FIRST: an exact word-permutation of ONE exercise name
            # is the user naming that exercise — beats ask AND data heuristics
            # (may legitimately pick a 0-set exercise the user named precisely).
            # Checked against ALL names — the LIMIT-8 pool can crowd out the
            # true match (see _permutation_match docstring).
            all_names = [r["name"] for r in
                         conn.execute("SELECT name FROM exercise").fetchall()]
            perm = _permutation_match(query, all_names)
            if perm:
                return {"match": perm, "candidates": []}
            counts = _logged_counts(conn, candidates)
            picked = _permissive_pick(query, candidates, counts)
            if picked:
                return {"match": picked, "candidates": []}
            # 2.3: ask offers only real-data candidates (see Tier 2).
            with_data = [c for c in candidates if counts.get(c, 0) >= _LOGGED_FLOOR]
            candidates = _order_by_data(query, with_data or candidates, counts)
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
        if permissive:
            counts = _logged_counts(conn, candidates)
            picked = _permissive_pick(query, candidates, counts)
            if picked:
                return {"match": picked, "candidates": []}
            # 2.3: ask offers only real-data candidates (see Tier 2).
            with_data = [c for c in candidates if counts.get(c, 0) >= _LOGGED_FLOOR]
            candidates = _order_by_data(query, with_data or candidates, counts)
        return {"match": None, "candidates": candidates}

    return {"match": None, "candidates": []}
