"""
#9 — ONE canonical SQL sanitizer (src/shared/sql_sanitize.py), with the em-dash
bug fixed.

The three former sanitizers (src.db.sanitize_sql, src.shared.sql_executor
._sanitize_sql, src.data_agent.fetch.sanitize_sql) all now delegate to the
single canonical one. Two of them used to turn an em-dash '—' into '--', a SQL
line comment, which silently truncated the rest of the query (including any
injected LIMIT). The canonical replaces '—' with a SPACE, never '--'.

No Gemini, no server.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.shared.sql_sanitize import sanitize_sql as canon   # noqa: E402

DB_PATH = os.environ["FITNOTES_DB_PATH"]


# ── one source of truth: every former sanitizer IS the canonical ─────────────

def test_all_three_call_sites_share_the_canonical():
    import src.db as db
    import src.data_agent.fetch as fetch
    import src.shared.sql_executor as ex
    import src.data_agent as data_agent
    assert db.sanitize_sql is canon
    assert fetch.sanitize_sql is canon
    assert ex._sanitize_sql is canon              # internal alias kept for its call site
    assert data_agent.sanitize_sql is canon       # public re-export still valid


# ── em-dash fix: NEVER becomes '--' (the bug) ────────────────────────────────

@pytest.mark.parametrize("sql", [
    "SELECT 1 — trailing note",
    "SELECT date FROM training_log WHERE metric_weight > 0 — heavy sets",
    "—",
    "a—b",
])
def test_em_dash_never_becomes_sql_comment(sql):
    out = canon(sql)
    assert "—" not in out          # normalized away
    assert "--" not in out         # and NEVER turned into a line comment
    assert " " in out or out == "" # replaced with a space


# ── curly quotes still normalized (union incl. fetch's ‚ ‛) ───────────────────

def test_curly_quotes_normalized_union():
    assert canon("‘a’") == "'a'"
    assert canon("“b”") == '"b"'
    # the low-9 / high-reversed-9 singles only the fetch copy handled — now shared
    assert canon("‚c‛") == "'c'"
    # plain ASCII untouched
    assert canon("SELECT * FROM t WHERE x = 'y'") == "SELECT * FROM t WHERE x = 'y'"


# ── the bug case end-to-end: LIMIT no longer swallowed by an em-dash comment ──

def test_executor_em_dash_in_string_runs_and_keeps_limit():
    # Em-dash inside a STRING literal: previously '—' -> '--' would corrupt the
    # value to '--'; now it degrades to a harmless space and the query runs with
    # its injected LIMIT intact (proves nothing was truncated into a comment).
    from src.shared.sql_executor import run_query
    rows = run_query("SELECT '—' AS c FROM training_log", DB_PATH)
    assert rows                       # LIMIT 10000 injected, query executed
    assert rows[0]["c"] == " "        # em-dash -> space, NOT '--'


def test_db_run_query_em_dash_in_string_runs():
    from src.db import get_connection, run_query
    conn = get_connection(DB_PATH)
    try:
        rows = run_query(conn, "SELECT '—' AS c FROM training_log", 3, 5)
    finally:
        conn.close()
    assert rows and rows[0]["c"] == " "


def test_executor_curly_quote_query_still_works():
    # Regression: a curly-quoted string still normalizes and executes.
    from src.shared.sql_executor import run_query
    rows = run_query("SELECT ‘x‘ AS c FROM training_log", DB_PATH)
    assert rows and rows[0]["c"] == "x"
