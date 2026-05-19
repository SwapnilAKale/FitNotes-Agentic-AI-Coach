import sqlite3
import threading
from typing import Any


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
        cursor = conn.execute(sql)
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
