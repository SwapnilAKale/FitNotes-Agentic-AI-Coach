"""
src/data_agent/fetch.py
Database boundary — ALL sqlite access lives here and ONLY here.

Public surface:
  fetch_data(end_str)  -> dict bundle of raw rows
  load_user_context()  -> dict
  query(sql)           -> dict
  sanitize_sql(sql)    -> str
"""

import sqlite3
import json
import os
import re
import logging
from typing import Optional

from src.shared.sql_executor import run_query as _run_ro_query
# Single canonical SQL text-sanitizer (one source of truth). Imported (not
# redefined) and re-exported so `from src.data_agent import sanitize_sql` and the
# call below keep working. NOTE: the analytical weight-aggregate FENCE
# (_weight_aggregate_reason) stays separate from sanitizing — it is a policy
# guard, not text cleanup.
from src.shared.sql_sanitize import sanitize_sql

logger = logging.getLogger(__name__)

# ── Custom-SQL weight/volume-aggregate guard ────────────────────────────────────
# The analytical custom-SQL lane is for counts / dates / gaps / streaks /
# patterns ONLY. Weight and volume have authoritative package fields
# (muscle_group_summary [bar-inclusive, per-unit], pr / pr_period, progression,
# e1rm_*). A raw aggregate over metric_weight here is the same blend-prone
# surface as run_read_only_sql: SUM(metric_weight * reps) adds kg-native rows'
# kilograms onto lbs, and nothing applies the bar or offset. We cannot unit-type
# arbitrary SQL output, so we REFUSE cross-row weight aggregation instead of
# caveating it. Per-row metric_weight SELECT (no aggregate) is still allowed and
# keeps the existing plates-only caveat.
# Blend-prone aggregates: each combines/serializes the raw metric_weight values
# of multiple rows, so it adds kg-native rows' kilograms onto lbs (or, for
# GROUP_CONCAT, emits a raw mixed-unit list). COUNT is deliberately NOT here — it
# counts rows and never touches the weight values, so "how many sets + their
# weights" / "count where metric_weight > 0" stay allowed. (See _weight_aggregate_reason.)
_BLEND_AGG_RE = re.compile(r"\b(?:sum|avg|total|min|max|group_concat)\s*\(")
_WEIGHT_AGG_REASON = (
    "weight/volume aggregates are not available via custom SQL — use the "
    "package's muscle_group_summary / pr / progression fields"
)


def _weight_aggregate_reason(sql: str) -> Optional[str]:
    """
    Return a refusal reason if the SQL could aggregate metric_weight across rows
    (blending kg-native and lbs frames into a meaningless number); else None.

    FINAL RULE (invariant, not surface form): REFUSE iff `metric_weight` appears
    ANYWHERE in the query AND a blend-prone aggregate — SUM / AVG / TOTAL / MIN /
    MAX / GROUP_CONCAT — appears ANYWHERE in the query. We do not (and cannot
    cheaply) fully parse SQL, so we match on the invariant that makes the blend
    possible: to combine weights you must (a) reference the weight column and
    (b) feed it to a combining aggregate. This deliberately closes the forms an
    AS-only alias check missed — e.g. `SELECT SUM(v) FROM (SELECT metric_weight v
    …)` (alias without AS) and GROUP_CONCAT(metric_weight) — because the literal
    `metric_weight` still has to appear in the projection that feeds the alias.

    Stays allowed: per-row `SELECT metric_weight … LIMIT n` (no blend aggregate),
    counts/dates that never mention metric_weight, and COUNT alongside a weight
    column or filter (COUNT is exempt). Over-refuses only in the safe direction
    (e.g. SUM(reps) in a query that also touches metric_weight) — the package
    answers any weight/volume question that gets refused here.
    """
    low = sql.lower()
    if "metric_weight" not in low:
        return None
    if _BLEND_AGG_RE.search(low):
        return _WEIGHT_AGG_REASON
    return None

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH           = os.environ.get("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
USER_CONTEXT_PATH = os.environ.get("USER_CONTEXT_PATH", "data/user_context.json")

# ── Category exclusion ─────────────────────────────────────────────────────────
# Categories 10, 11, 12 are excluded from all queries.
# Keep in sync with CATEGORY_NAMES in process.py.
EXCLUDED_CATEGORY_IDS = (10, 11, 12)
_EXCL_SQL = f"({', '.join(str(c) for c in EXCLUDED_CATEGORY_IDS)})"


# ── Connection ──────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    # Read-only at the connection level (same principle as src/db.py):
    # the Data Agent only reads, so a bug in this module must not be able
    # to write — and a plain connect() would silently CREATE an empty DB
    # file when DB_PATH is wrong, hiding the misconfiguration.
    normalized = DB_PATH.replace("\\", "/")
    conn = sqlite3.connect(f"file:{normalized}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── File I/O ───────────────────────────────────────────────────────────────────

def load_user_context() -> dict:
    with open(USER_CONTEXT_PATH, "r") as f:
        return json.load(f)


# ── Raw DB queries ─────────────────────────────────────────────────────────────

def _fetch_all_sets_in_period(conn: sqlite3.Connection,
                               start_date: str, end_date: str) -> list:
    """
    Single bulk query: ALL sets for ALL exercises in [start_date, end_date].
    LEFT JOIN Comment — comment = None = unremarkable set, valid data.
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            tl._id               AS set_id,
            tl.date,
            tl.metric_weight,
            tl.reps,
            tl.distance,
            tl.duration_seconds,
            tl.is_personal_record,
            e.name               AS exercise_name,
            e.category_id,
            c.comment
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        LEFT JOIN Comment c ON c.owner_id = tl._id
        WHERE tl.date >= ? AND tl.date <= ?
          AND e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC, tl._id ASC
    """, (start_date, end_date))
    return [dict(row) for row in cur.fetchall()]


def _fetch_exercise_lifecycle(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT e.name AS exercise_name, e.category_id,
               MIN(tl.date) AS first_date, MAX(tl.date) AS last_date,
               COUNT(*) AS total_sets,
               COUNT(DISTINCT tl.date) AS total_sessions
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
        GROUP BY e._id, e.name, e.category_id
        ORDER BY e.category_id, e.name
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_pr_history(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT tl.date, tl.metric_weight, tl.reps, e.name AS exercise_name
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE tl.is_personal_record = 1
          AND e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_all_bodyweight(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("""
        SELECT date, body_weight_metric, body_fat, comments
        FROM BodyWeight
        ORDER BY date ASC
    """)
    return [{"date": row["date"],
             "weight": row["body_weight_metric"],
             "body_fat": row["body_fat"],
             "comments": row["comments"]}
            for row in cur.fetchall()]


def _fetch_goals(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            g._id          AS goal_id,
            e.name         AS exercise_name,
            g.metric_weight,
            g.reps,
            g.target_date,
            g.start_date,
            g.title        AS notes,
            g.type_id,
            g.unit         AS unit_flag
        FROM Goal g
        JOIN exercise e ON g.exercise_id = e._id
        ORDER BY g.target_date ASC
    """)
    return [dict(row) for row in cur.fetchall()]


def _fetch_all_training_dates(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT tl.date
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.category_id NOT IN {_EXCL_SQL}
        ORDER BY tl.date ASC
    """)
    return [row["date"] for row in cur.fetchall()]


# ── Main fetch entry point ─────────────────────────────────────────────────────

def fetch_data(end_str: str) -> dict:
    """
    Run all up-front SELECTs in a single connection and return a typed bundle.

    alltime_rows covers the full DB history (from 2000-01-01 to end_str).
    process.py derives period_rows by filtering alltime_rows on start_str.

    Bundle keys:
        alltime_rows  — all sets not in excluded categories, up to end_str
        bodyweight    — all BodyWeight entries
        goals         — all Goal rows
        lifecycle     — per-exercise lifecycle summary
        pr_history    — all is_personal_record=1 rows
        training_dates — all distinct training dates (no date cap)
    """
    conn = _get_connection()
    try:
        training_dates = _fetch_all_training_dates(conn)
        return {
            "alltime_rows":    _fetch_all_sets_in_period(conn, "2000-01-01", end_str),
            "bodyweight":      _fetch_all_bodyweight(conn),
            "goals":           _fetch_goals(conn),
            "lifecycle":       _fetch_exercise_lifecycle(conn),
            "pr_history":      _fetch_pr_history(conn),
            "training_dates":  training_dates,
        }
    finally:
        conn.close()


# ── Dynamic SQL fallback ───────────────────────────────────────────────────────

def query(sql: str) -> dict:
    """
    Fallback dynamic SQL query for questions the pre-built Data Agent functions
    do not cover.

    Use only when collect() data genuinely cannot answer the question.
    The pre-built path (collect) is always preferred — it returns clean,
    unit-converted, offset-applied, bar-weight-included values.

    This function returns RAW DB values. The caller is responsible for
    understanding the conversion rules documented in the WARNING below.

    Args:
        sql: A SELECT statement. Any other statement type is rejected.

    Returns:
        {
            "rows":    list of dicts (column -> raw value),
            "columns": list of column names,
            "row_count": int,
            "warning": str   <-- always present, always read this
        }
        OR, when the SQL applies an aggregate to a weight/volume expression:
        {
            "refused": True, "reason": str,
            "rows": [], "columns": [], "row_count": 0, "warning": "REFUSED: ..."
        }
        Weight/volume aggregates are out of this lane (counts/dates/gaps only);
        the authoritative weight fields live in the package (muscle_group_summary,
        pr / pr_period, progression). The caller falls back to those.

    WARNING — Raw values returned, no automatic conversions applied:
        metric_weight : stored as kg (FitNotes always divides typed value by 2.2046).
                        To recover typed value: metric_weight * 2.2046
        numeric_offset: NOT applied. Machine Wrist Extension and similar exercises
                        have an offset in user_context.json that this query does not add.
        bar_weight    : NOT included. Barbell and Smith Machine exercises log plates
                        only. Bar weight must be added separately for true load.
        unit label    : NOT determined. KG-native exercises (Deadlift, Seated Machine
                        Curl (Kg), Machine Wrist Extension, Hand Gripper) report in kg;
                        all others in lbs. The label is not attached to raw rows.
        is_personal_record: raw integer (0 or 1), not boolean.

    The Analysis Agent must apply these conversions or explicitly note in its
    answer that weights shown are raw logged values before conversion.
    """
    sql = sanitize_sql(sql.strip())

    # Weight/volume-aggregate guard: refuse before executing. Custom SQL stays in
    # its non-weight lane; weight/volume come from authoritative package fields.
    refusal = _weight_aggregate_reason(sql)
    if refusal:
        return {
            "refused":   True,
            "reason":    refusal,
            "rows":      [],
            "columns":   [],
            "row_count": 0,
            "warning":   f"REFUSED: {refusal}",
        }

    # Execution goes through shared/sql_executor.run_query:
    #   - read-only connection (mode=ro URI) — a write statement that slips
    #     past any textual guard fails at the database level
    #   - SELECT/WITH-only guard (raises ValueError)
    #   - LIMIT 10000 injected when absent
    #   - 30-second timeout via connection interrupt
    # The previous inline guard was a space-delimited keyword blacklist on a
    # read-write connection — "WITH c AS (SELECT 1)INSERT INTO ..." passed it.
    try:
        rows = _run_ro_query(sql, DB_PATH)
    except ValueError as e:
        return {
            "rows":      [],
            "columns":   [],
            "row_count": 0,
            "warning":   f"REJECTED: only SELECT statements are permitted. {e}",
        }
    except Exception as e:
        return {
            "rows":      [],
            "columns":   [],
            "row_count": 0,
            "warning":   f"QUERY ERROR: {str(e)}",
        }

    columns = list(rows[0].keys()) if rows else []

    # Auto-convert metric_weight to typed_weight so the Analysis Agent
    # never sees raw stored kg values directly.
    has_metric_weight = bool(rows) and "metric_weight" in rows[0]
    if has_metric_weight:
        for row in rows:
            if row.get("metric_weight") is not None:
                row["typed_weight"] = round(row["metric_weight"] * 2.2046, 1)
            else:
                row["typed_weight"] = None
        columns = columns + ["typed_weight"]

    return {
        "rows":      rows,
        "columns":   columns,
        "row_count": len(rows),
        "warning":   (
            "PARTIAL CONVERSION APPLIED: typed_weight = metric_weight * 2.2046 "
            "(recovers original typed value). Still missing: numeric_offset (e.g. "
            "Machine Wrist Extension +5), bar_weight (barbell/Smith Machine exercises "
            "log plates only), and unit label (kg-native exercises: Deadlift from "
            "2025-12-26, Seated Machine Curl (Kg), Machine Wrist Extension, Hand Gripper "
            "— all others lbs). Use typed_weight for display, not metric_weight."
        ),
    }
