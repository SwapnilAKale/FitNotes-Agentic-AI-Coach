import re
import sqlite3
import threading

# Single canonical SQL text-sanitizer (one source of truth) — replaces the local
# copy whose '—' -> '--' turned an em-dash into a SQL line comment (truncating
# the query and any injected LIMIT). See src/shared/sql_sanitize.py.
from src.shared.sql_sanitize import sanitize_sql as _sanitize_sql


def run_query(sql: str, db_path: str) -> list[dict]:
    """
    Execute a read-only SQL query against the FitNotes database.

    Protections applied before execution:
      - sanitize_sql(): replace curly quotes with straight quotes
      - SELECT-only guard: reject any non-SELECT statement
      - LIMIT injection: if no LIMIT clause present, inject LIMIT 10000
      - timeout=30 seconds via sqlite3 connection

    Returns list of row dicts.
    Raises ValueError for non-SELECT queries.
    Raises sqlite3.Error for database errors.
    Never swallows errors — callers decide how to handle them.
    """
    sql = _sanitize_sql(sql)

    normalized = sql.lstrip().upper()
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        raise ValueError(f"Rejected: only SELECT/WITH statements are allowed, got: {sql[:80]!r}")

    if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        sql = sql.rstrip().rstrip(';') + ' LIMIT 10000'

    normalized_path = db_path.replace("\\", "/")
    uri = f"file:{normalized_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row

    interrupted = threading.Event()

    def _interrupt():
        interrupted.set()
        conn.interrupt()

    timer = threading.Timer(30, _interrupt)
    try:
        timer.start()
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]
    except sqlite3.OperationalError as exc:
        if interrupted.is_set():
            raise TimeoutError("Query exceeded 30s timeout") from exc
        raise
    finally:
        timer.cancel()
        conn.close()
