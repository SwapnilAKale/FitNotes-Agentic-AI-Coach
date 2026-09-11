import hashlib
import sqlite3
import threading
from typing import Any

# Single canonical SQL text-sanitizer (one source of truth). Re-exported here so
# the public `src.db.sanitize_sql` name keeps working for existing callers.
from src.shared.sql_sanitize import sanitize_sql


# ── Write integrity ───────────────────────────────────────────────────────────
# A write used to be called successful when the SQL raised no exception. It is
# not the same thing: a DELETE matching zero rows raises nothing, so a staged
# delete whose row had already gone reported "Set deleted successfully" while
# nothing was deleted. And the one operation that DID verify only re-read the
# rows it inserted, so damage anywhere else passed unnoticed — as did its
# rollback, which re-read the same narrow scope.
#
# Two layers, both used INSIDE the write transaction so a rejection can roll back:
#
#   Layer 1 — BLAST RADIUS. conn.total_changes must move by exactly the number of
#     rows the caller intended. Fewer means it did nothing; more means it touched
#     rows nobody asked for. Effectively free, and per-connection, so it stays
#     correct if another writer is active.
#
#   Layer 2 — IDENTITY. The focus table (the only one an operation may touch) is
#     captured as {_id: row-hash}; the added/removed/modified id-sets must equal
#     exactly what was intended. That is what catches the right NUMBER of rows
#     changing in the wrong PLACE. Every other table is hashed whole and must not
#     move at all.
#
# sqlite_sequence is excluded by construction: training_log is AUTOINCREMENT, so
# it legitimately changes on every insert and would otherwise reject every log.
#
# TWO ASSUMPTIONS, both true today and both silent if they stop being:
#
#   COLUMN 0 IS THE PRIMARY KEY of every focus table. The focus map keys on r[0].
#   Verified for all five write-reachable tables (training_log, Goal, BodyWeight,
#   Comment, WorkoutComment) — in each, _id is both the primary key and the first
#   column. A focus table where that is not so would key the map on the wrong
#   value and compare nonsense. Pinned by
#   test_every_write_reachable_table_keys_on_id.
#
#   COST IS PROPORTIONAL TO DATABASE SIZE. Measured ~35 ms per snapshot and
#   ~70 ms per guarded write on a 0.7 MB / 11k-row database — negligible against
#   a turn that already spends seconds in LLM calls, but it is a full table scan
#   and it grows with the data.
#
# NOT concurrency-safe at Layer 2 — a second writer's unrelated change would look
# like corruption. Fine for a single-user local database; before multi-user this
# must narrow to the transaction's own scope. Layer 1 is unaffected.

INTEGRITY_IGNORED_TABLES = frozenset({"sqlite_sequence"})


class IntegrityRejected(Exception):
    """A write changed the database in a way nobody asked for.

    Carries `delta` — the observed-vs-expected detail — so the caller can log it
    while showing the user one plain sentence.
    """

    def __init__(self, reason: str, delta: dict):
        super().__init__(reason)
        self.reason = reason
        self.delta = delta


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(r[0] for r in rows)


def _row_hash(row) -> str:
    return hashlib.blake2b(repr(tuple(row)).encode(), digest_size=8).hexdigest()


def snapshot_for_integrity(conn: sqlite3.Connection, focus_tables) -> dict:
    """Capture the state a write will be checked against.

    Focus tables get a per-row map so an id-level diff is possible; everything
    else gets one hash, because all we need to know is "did this move at all".
    Taken on the SAME connection as the write, so the after-snapshot sees the
    uncommitted changes.
    """
    focus = set(focus_tables)
    snap = {"total_changes": conn.total_changes, "focus": {}, "rest": {}}
    for table in _user_tables(conn):
        if table in INTEGRITY_IGNORED_TABLES:
            continue
        if table in focus:
            snap["focus"][table] = {
                r[0]: _row_hash(r)
                for r in conn.execute(f"SELECT * FROM [{table}]")
            }
        else:
            digest = hashlib.blake2b(digest_size=16)
            for r in conn.execute(f"SELECT * FROM [{table}] ORDER BY rowid"):
                digest.update(repr(tuple(r)).encode())
            snap["rest"][table] = digest.hexdigest()
    return snap


def assert_only_expected_changed(conn: sqlite3.Connection, before: dict,
                                 expect: dict) -> None:
    """Raise IntegrityRejected unless the database changed EXACTLY as intended.

    `expect` maps a focus table to the ids it may affect::

        {"training_log": {"removed": {15211}}}
        {"training_log": {"modified": {15211}}}
        {"training_log": {"added": {15212, 15213}}}   # ids known (lastrowid)
        {"Comment": {"added_count": 2}}               # ids not tracked

    Prefer `added` over `added_count` when the caller knows the ids it created —
    it proves the rows that appeared are the ones it inserted, not merely that
    the right NUMBER appeared.

    Anything omitted must be empty. Any non-focus table must be untouched.
    """
    after = snapshot_for_integrity(conn, before["focus"].keys())

    # Layer 1 — blast radius, checked first because it is nearly free.
    intended = sum(
        len(e.get("removed", ())) + len(e.get("modified", ()))
        + (len(e["added"]) if "added" in e else e.get("added_count", 0))
        for e in expect.values()
    )
    observed = after["total_changes"] - before["total_changes"]
    if observed != intended:
        raise IntegrityRejected(
            "row count changed by an unexpected amount",
            {"rows_changed": observed, "rows_expected": intended},
        )

    # Layer 2a — nothing outside the focus table may move.
    moved = [t for t, h in after["rest"].items() if before["rest"].get(t) != h]
    if moved:
        raise IntegrityRejected(
            "a table the operation should not touch was modified",
            {"unexpected_tables": moved},
        )

    # Layer 2b — inside the focus table, the exact rows and no others.
    for table, old_rows in before["focus"].items():
        new_rows = after["focus"][table]
        added = set(new_rows) - set(old_rows)
        removed = set(old_rows) - set(new_rows)
        modified = {i for i in set(old_rows) & set(new_rows)
                    if old_rows[i] != new_rows[i]}
        e = expect.get(table, {})
        added_ok = (added == set(e["added"]) if "added" in e
                    else len(added) == e.get("added_count", 0))
        if (removed != set(e.get("removed", ()))
                or modified != set(e.get("modified", ()))
                or not added_ok):
            raise IntegrityRejected(
                "different rows changed than the operation intended",
                {
                    "table": table,
                    "added": sorted(added),
                    "added_expected": (sorted(e["added"]) if "added" in e
                                       else f"count={e.get('added_count', 0)}"),
                    "removed": sorted(removed),
                    "removed_expected": sorted(e.get("removed", ())),
                    "modified": sorted(modified),
                    "modified_expected": sorted(e.get("modified", ())),
                },
            )


MSG_INTEGRITY_REJECTED = (
    "That wasn't saved: the database didn't change the way it should have, so "
    "it was rolled back. Nothing was altered. Please try again."
)


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
