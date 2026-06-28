import sqlite3
import threading
from typing import Any

# Single canonical SQL text-sanitizer (one source of truth). Re-exported here so
# the public `src.db.sanitize_sql` name keeps working for existing callers.
from src.shared.sql_sanitize import sanitize_sql


def get_connection(db_path: str) -> sqlite3.Connection:
    normalized = db_path.replace("\\", "/")
    uri = f"file:{normalized}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_write_connection(db_path: str) -> sqlite3.Connection:
    """Opens a read-write connection for write operations only."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def insert_training_log_set(conn: sqlite3.Connection, exercise_id: int,
                            date: str, s: dict) -> int:
    """
    Write ONE staged set to training_log and return its new _id (lastrowid).

    The staged set dict `s` is the canonical row built at stage time (it already
    carries the final column values for its metric type):
      strength       : metric_weight, reps, unit=0, distance=0, duration_seconds=0
      cardio (dist)  : metric_weight=0, reps=0, unit=3, distance, duration_seconds
      cardio (dur)   : metric_weight=0, reps=0, unit=2, distance=0, duration_seconds

    `unit` here is the METRIC-TYPE code (0/2/3), never the lbs/kg display unit —
    the lbs/kg param is intentionally NOT bound to this column (the read path
    derives display unit from user_context.exercises_in_kg, not from here).

    distance / duration_seconds / unit / is_complete are NOT NULL DEFAULT 0, so
    absent metrics are written as 0, never NULL. Both the live execute handler and
    the WAL replay call THIS function, so a replay reproduces an identical row.
    """
    cur = conn.execute(
        """INSERT INTO training_log
           (exercise_id, date, metric_weight, reps, unit,
            is_personal_record, is_complete, distance, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
        (exercise_id, date,
         s.get("metric_weight", 0) or 0, s.get("reps", 0) or 0, s.get("unit", 0),
         int(s.get("is_personal_record", 0)),
         s.get("distance", 0) or 0, s.get("duration_seconds", 0) or 0),
    )
    return cur.lastrowid


def insert_set_comment(conn: sqlite3.Connection, owner_id: int,
                       date: str, comment: str) -> None:
    """
    Write a per-set comment bound to its training_log row by foreign key
    (Comment.owner_id == training_log._id, owner_type_id=1) — the SAME keying the
    read path joins on (LEFT JOIN Comment c ON c.owner_id = tl._id). Caller passes
    the set's lastrowid so the comment lands on the correct set (no off-by-one).
    """
    conn.execute(
        "INSERT INTO Comment (date, owner_type_id, owner_id, comment) "
        "VALUES (?, 1, ?, ?)",
        (date, owner_id, comment),
    )


def introspect_schema(conn: sqlite3.Connection) -> dict:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    schema: dict[str, list[dict]] = {}
    for table in tables:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        schema[table] = [{"name": row[1], "type": row[2]} for row in cols]
    return schema


def run_query(
    conn: sqlite3.Connection,
    sql: str,
    row_limit: int = 100,
    timeout_seconds: int = 5,
) -> list[dict]:
    interrupted = threading.Event()

    def _interrupt():
        interrupted.set()
        conn.interrupt()

    timer = threading.Timer(timeout_seconds, _interrupt)
    try:
        timer.start()
        cursor = conn.execute(sanitize_sql(sql))
        rows = cursor.fetchmany(row_limit)
        return [{k: row[k] for k in row.keys()} for row in rows]
    except sqlite3.OperationalError as exc:
        if interrupted.is_set():
            raise TimeoutError(
                f"Query exceeded {timeout_seconds}s timeout"
            ) from exc
        raise
    finally:
        timer.cancel()
