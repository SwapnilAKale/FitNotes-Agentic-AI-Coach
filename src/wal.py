"""
src/wal.py — Write-Ahead Log for agent database writes.

A FitNotes backup upload replaces the entire .fitnotes file, which wipes
every set/goal the agent wrote since the previous upload. Each confirmed
execute_* write is journaled here so /upload can replay it onto the fresh
database.

Public surface:
  append_write(tool, params)  -> str   record id; journal one confirmed write
  replay_writes(db_path)      -> dict  {"replayed": N, "conflicts": M, "errors": [...]}
  get_records()               -> list  current WAL contents (for /wal-status)

Record shape (data/agent_writes.json is a JSON list of these):
  {
    "id":        "<uuid4>",
    "timestamp": "<ISO-8601>",
    "tool":      "execute_staged_workout" | "log_workout" | ...,
    "params":    { resolved staged payload },
    "status":    "pending" | "replayed" | "conflict"
  }

params is the RESOLVED staged payload (exercise_id, metric_weight, row ids),
captured after the commit succeeded — not the user-facing tool arguments —
so replay never depends on name resolution or staging state. Replay also
accepts the friendlier shape (exercise_name + typed weight) for records
written by tests or by hand.
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DEFAULT_WAL_PATH = "data/agent_writes.json"
WAL_PATH = os.environ.get("AGENT_WRITES_PATH", _DEFAULT_WAL_PATH)

_pytest_redirect_warned = False


def _effective_path() -> str:
    """
    The WAL path all file I/O actually uses.

    Last-line guard: if we are running under pytest (PYTEST_CURRENT_TEST is
    set, and it is inherited by MCP subprocesses spawned from a test) and
    nobody isolated the WAL — no AGENT_WRITES_PATH override, WAL_PATH still
    the default — redirect to a per-process temp file so test writes can
    never pollute the real journal. Test fixtures that set WAL_PATH or the
    env var keep full control.
    """
    global _pytest_redirect_warned
    if (WAL_PATH == _DEFAULT_WAL_PATH
            and "AGENT_WRITES_PATH" not in os.environ
            and "PYTEST_CURRENT_TEST" in os.environ):
        redirect = os.path.join(tempfile.gettempdir(),
                                f"agent_writes.pytest-{os.getpid()}.json")
        if not _pytest_redirect_warned:
            _pytest_redirect_warned = True
            print(f"[wal] pytest detected with no WAL isolation — "
                  f"redirecting writes to {redirect}", file=sys.stderr)
        return redirect
    return WAL_PATH

# One lock guards every read-modify-write of the WAL file. replay_writes
# holds it for the whole replay so a concurrent append can't be lost.
_lock = threading.Lock()

# Same epsilon the live update/delete tools use to match stored kg values.
_WEIGHT_EPSILON = 0.01


class _ReplayConflict(Exception):
    """Replay-level conflict: target row missing, or data already present."""


# ── File I/O (callers must hold _lock) ─────────────────────────────────────────

def _load() -> list:
    path = _effective_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"expected a JSON list, got {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        # Never silently overwrite a corrupt WAL — move it aside so the
        # records can be recovered by hand, then start fresh.
        quarantine = f"{path}.corrupt-{int(time.time())}"
        try:
            os.replace(path, quarantine)
            logger.error("[wal] %s is unreadable (%s) — moved to %s",
                         path, exc, quarantine)
        except OSError:
            logger.error("[wal] %s is unreadable (%s) and could not be "
                         "quarantined", path, exc)
        return []


def _save(records: list) -> None:
    path = _effective_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    # Atomic on the same filesystem. On Windows, OneDrive/antivirus can hold
    # a transient lock on the destination and fail the replace with
    # PermissionError — retry briefly before giving up.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


# ── Public API ─────────────────────────────────────────────────────────────────

def append_write(tool: str, params: dict) -> str:
    """Append one confirmed write to the WAL. Returns the record id."""
    record = {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool":      tool,
        "params":    params,
        "status":    "pending",
    }
    with _lock:
        records = _load()
        records.append(record)
        _save(records)
    logger.info("[wal] journaled %s (id=%s)", tool, record["id"])
    return record["id"]


def get_records() -> list:
    """Current WAL contents (read-only snapshot, for /wal-status)."""
    with _lock:
        return _load()


def wipe() -> dict:
    """
    Empty the journal (user-initiated 'clear saved chat logs').

    The current records are archived next to the WAL file first —
    agent_writes.archive-<UTC-ts>.json — so a mistaken wipe is recoverable
    by hand. An already-empty journal produces no archive.
    """
    with _lock:
        records = _load()
        archive = None
        if records:
            path = _effective_path()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = os.path.join(os.path.dirname(path) or ".",
                                   f"agent_writes.archive-{stamp}.json")
            with open(archive, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            _save([])
    if records:
        logger.info("[wal] wiped %d records (archived to %s)", len(records), archive)
    return {"wiped": len(records), "archive": archive}


def replay_writes(db_path: str) -> dict:
    """
    Re-execute every pending WAL record against db_path.

    Each record commits independently, so one conflict never rolls back the
    records around it. Conflicts (target row missing, data already present,
    constraint errors) are marked status="conflict" with the error stored on
    the record and logged loud — never silently dropped. Records whose
    status is not "pending" are skipped, which makes replay idempotent.
    """
    summary: dict = {"replayed": 0, "conflicts": 0, "errors": []}

    with _lock:
        records = _load()
        pending = [r for r in records if r.get("status") == "pending"]
        if not pending:
            return summary

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            for record in pending:
                tool = record.get("tool", "")
                handler = _HANDLERS.get(_TOOL_ALIASES.get(tool))
                try:
                    if handler is None:
                        raise _ReplayConflict(f"no replay handler for tool {tool!r}")
                    detail = handler(conn, record.get("params") or {})
                    conn.commit()
                    record["status"] = "replayed"
                    record["replayed_at"] = datetime.now(timezone.utc).isoformat()
                    if detail:
                        record["replay_detail"] = detail
                    summary["replayed"] += 1
                except Exception as exc:
                    conn.rollback()
                    record["status"] = "conflict"
                    record["error"] = str(exc)
                    summary["conflicts"] += 1
                    summary["errors"].append(
                        {"id": record.get("id"), "tool": tool, "error": str(exc)})
                    logger.error("[wal] CONFLICT replaying %s (id=%s): %s",
                                 tool, record.get("id"), exc)
        finally:
            conn.close()

        _save(records)

    logger.info("[wal] replay complete: %d replayed, %d conflicts",
                summary["replayed"], summary["conflicts"])
    return summary


# ── Replay handlers ────────────────────────────────────────────────────────────
# Each performs the same SQL the live execute_* tool runs (combined_server.py),
# raises _ReplayConflict when the write cannot apply, and returns an optional
# detail string. The caller commits.

def _resolve_exercise_id(conn: sqlite3.Connection, params: dict) -> int:
    if params.get("exercise_id") is not None:
        return int(params["exercise_id"])
    name = params.get("exercise_name") or params.get("exercise")
    if not name:
        raise _ReplayConflict("params carry neither exercise_id nor exercise_name")
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (name,)).fetchone()
    if not row:
        raise _ReplayConflict(f"exercise {name!r} not found in the new database")
    return row["_id"]


def _set_metric_weight(s: dict) -> float:
    if s.get("metric_weight") is not None:
        return float(s["metric_weight"])
    if s.get("weight") is not None:        # typed value (lbs frame)
        return float(s["weight"]) / 2.2046
    raise _ReplayConflict("set carries neither metric_weight nor weight")


def _replay_workout(conn: sqlite3.Connection, params: dict) -> str:
    exercise_id = _resolve_exercise_id(conn, params)
    date_str = params.get("date")
    sets = params.get("sets") or []
    if not date_str or not sets:
        raise _ReplayConflict("workout record missing date or sets")

    from src.db import insert_training_log_set, insert_set_comment

    inserted = skipped = 0
    for s in sets:
        metric_weight = _set_metric_weight(s)
        reps = int(s.get("reps") or 0)          # cardio sets carry reps=0
        distance = float(s.get("distance") or 0)
        duration = int(s.get("duration_seconds") or 0)
        # training_log has no unique constraint, so a duplicate INSERT would
        # succeed and double the set. An identical existing row means the
        # data is already in the database — skip it. The predicate is
        # cardio-aware: distinct cardio sessions (weight=reps=0) differ only by
        # distance/duration, so those are part of the identity check.
        dup = conn.execute(
            """SELECT 1 FROM training_log
               WHERE exercise_id = ? AND date = ? AND reps = ?
                 AND ABS(metric_weight - ?) < ?
                 AND ABS(distance - ?) < ?
                 AND duration_seconds = ?
               LIMIT 1""",
            (exercise_id, date_str, reps, metric_weight, _WEIGHT_EPSILON,
             distance, _WEIGHT_EPSILON, duration),
        ).fetchone()
        if dup:
            skipped += 1
            continue
        # Shared writer (also used by the live execute handler) → a replay
        # reproduces an identical row. metric_weight is recovered above for the
        # dedup check, so thread it onto the dict the helper writes.
        new_id = insert_training_log_set(
            conn, exercise_id, date_str, {**s, "metric_weight": metric_weight})
        if s.get("comment"):
            insert_set_comment(conn, new_id, date_str, s["comment"])
        inserted += 1

    if inserted == 0:
        raise _ReplayConflict(
            f"all {skipped} set(s) already exist for exercise_id={exercise_id} "
            f"on {date_str}")
    return f"{inserted} set(s) inserted, {skipped} already present"


def _replay_goal(conn: sqlite3.Connection, params: dict) -> str:
    exercise_id = _resolve_exercise_id(conn, params)
    metric_weight = float(params["metric_weight"])
    reps = int(params["reps"])
    target_date = params["target_date"]

    dup = conn.execute(
        """SELECT 1 FROM Goal
           WHERE exercise_id = ? AND target_date = ? AND reps = ?
             AND ABS(metric_weight - ?) < ?
           LIMIT 1""",
        (exercise_id, target_date, reps, metric_weight, _WEIGHT_EPSILON),
    ).fetchone()
    if dup:
        raise _ReplayConflict(
            f"identical goal already exists for exercise_id={exercise_id} "
            f"target_date={target_date}")

    conn.execute(
        """INSERT INTO Goal
           (type_id, exercise_id, metric_weight, reps, unit, title, target_date,
            sort_order, distance, duration_seconds, start_date)
           VALUES (1, ?, ?, ?, 0, ?, ?, 0, 0, 0, ?)""",
        (exercise_id, metric_weight, reps,
         params.get("title", ""), target_date, params.get("start_date", "")),
    )
    return "goal inserted"


def _replay_goal_update(conn: sqlite3.Connection, params: dict) -> str:
    cur = conn.execute(
        "UPDATE Goal SET metric_weight = ?, reps = ?, target_date = ? WHERE _id = ?",
        (float(params["new_metric_weight"]), int(params["new_reps"]),
         params["new_target_date"], int(params["goal_id"])),
    )
    if cur.rowcount == 0:
        raise _ReplayConflict(f"Goal _id={params['goal_id']} not found")
    return "goal updated"


def _replay_goal_delete(conn: sqlite3.Connection, params: dict) -> str:
    cur = conn.execute("DELETE FROM Goal WHERE _id = ?", (int(params["goal_id"]),))
    if cur.rowcount == 0:
        raise _ReplayConflict(f"Goal _id={params['goal_id']} not found")
    return "goal deleted"


def _replay_set_update(conn: sqlite3.Connection, params: dict) -> str:
    set_id = int(params["set_id"])
    new_metric_weight = float(params["new_metric_weight"])
    new_reps = int(params["new_reps"])

    target = conn.execute(
        "SELECT exercise_id FROM training_log WHERE _id = ?", (set_id,)).fetchone()
    if not target:
        raise _ReplayConflict(f"training_log _id={set_id} not found")

    # Recompute the PR flag against the new database, same rule as the
    # live tool: new weight beats every other set of this exercise.
    pr_row = conn.execute(
        "SELECT MAX(metric_weight) AS max_w FROM training_log "
        "WHERE exercise_id = ? AND _id != ?",
        (target["exercise_id"], set_id),
    ).fetchone()
    other_max = pr_row["max_w"] if pr_row and pr_row["max_w"] is not None else 0.0
    is_pr = 1 if new_metric_weight > other_max else 0

    conn.execute(
        "UPDATE training_log SET metric_weight = ?, reps = ?, "
        "is_personal_record = ? WHERE _id = ?",
        (new_metric_weight, new_reps, is_pr, set_id),
    )
    return "set updated"


def _replay_set_delete(conn: sqlite3.Connection, params: dict) -> str:
    cur = conn.execute(
        "DELETE FROM training_log WHERE _id = ?", (int(params["set_id"]),))
    if cur.rowcount == 0:
        raise _ReplayConflict(f"training_log _id={params['set_id']} not found")
    return "set deleted"


# Both naming conventions map to one handler: the execute_* names are what
# the combined_server hook journals; the stage-tool names are accepted for
# records written by tests or by hand.
_TOOL_ALIASES: dict = {
    "log_workout":                 "workout",
    "execute_staged_workout":      "workout",
    "set_goal":                    "goal",
    "execute_staged_goal":         "goal",
    "update_goal":                 "goal_update",
    "execute_staged_goal_update":  "goal_update",
    "delete_goal":                 "goal_delete",
    "execute_staged_goal_delete":  "goal_delete",
    "update_workout_set":          "set_update",
    "execute_staged_set_update":   "set_update",
    "delete_workout_set":          "set_delete",
    "execute_staged_set_delete":   "set_delete",
}

_HANDLERS: dict = {
    "workout":     _replay_workout,
    "goal":        _replay_goal,
    "goal_update": _replay_goal_update,
    "goal_delete": _replay_goal_delete,
    "set_update":  _replay_set_update,
    "set_delete":  _replay_set_delete,
}
