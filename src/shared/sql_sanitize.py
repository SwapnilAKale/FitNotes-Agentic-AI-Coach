"""
src/shared/sql_sanitize.py
ONE canonical SQL text-sanitizer (single source of truth) — same discipline as
src/units.py. Previously three copies had drifted (src.db.sanitize_sql,
src.shared.sql_executor._sanitize_sql, src.data_agent.fetch.sanitize_sql),
each with a slightly different rule set.

LLM-generated SQL routinely contains Unicode punctuation SQLite cannot parse:
curly/smart quotes (which raise OperationalError) and em-dashes. This normalizes
those BEFORE execution. It is the UNION of the legitimate behaviors of the three
former sanitizers.

SHARED here / intentionally SEPARATE:
  - SHARED:   pure text normalization (curly quotes, em-dash). Nothing else.
  - SEPARATE: the SELECT/WITH-only guard + LIMIT/row-cap injection (in the
              run_query executors) and the analytical weight-aggregate fence
              (fetch._weight_aggregate_reason). Those are policy/safety, not
              text cleanup, and must NOT be folded in here.

EM-DASH FIX: an em-dash must NEVER be turned into '--'. In SQL '--' starts a line
comment, so the old `'—' -> '--'` silently truncated the rest of the query —
including any LIMIT injected afterwards. We replace an em-dash with a SPACE
instead: inside a string literal it degrades to a harmless space; outside one the
query errors cleanly rather than being silently truncated. (We do NOT touch
en-dash or other dashes — that would be a new behavior, not a consolidation.)
"""

# Curly / smart single quotes -> straight apostrophe.
# ‘ ' , ’ ' , ‚ ‚ , ‛ ‛
_SINGLE_QUOTES = "‘’‚‛"
# Curly / smart double quotes -> straight double quote.
# “ " , ” "
_DOUBLE_QUOTES = "“”"
# Em-dash -> space (NEVER '--'; see module docstring).
_EM_DASH = "—"


def sanitize_sql(sql: str) -> str:
    """Normalize Unicode punctuation in LLM-generated SQL before execution.

    - curly single quotes (' ' ‚ ‛) -> '
    - curly double quotes (" ")      -> "
    - em-dash (—)                    -> space  (NOT '--' — see module docstring)
    """
    out = sql
    for ch in _SINGLE_QUOTES:
        out = out.replace(ch, "'")
    for ch in _DOUBLE_QUOTES:
        out = out.replace(ch, '"')
    out = out.replace(_EM_DASH, " ")
    return out
