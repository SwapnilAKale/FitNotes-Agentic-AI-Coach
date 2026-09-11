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
        # Journalled row id -> its id in THIS database, built as inserts replay.
        # Records are applied in order, so a set's insert is always seen before
        # the edit or delete that names it. See _mapped_id.
        id_map: dict = {}
        try:
            for record in pending:
                tool = record.get("tool", "")
                handler = _HANDLERS.get(_TOOL_ALIASES.get(tool))
                try:
                    if handler is None:
                        raise _ReplayConflict(f"no replay handler for tool {tool!r}")
                    # BLAST RADIUS, the same Layer-1 check live writes get.
                    # Replay ran the same SQL as the live execute tools against
                    # a database that has changed underneath it, with no
                    # integrity guard at all — a delete that removed two rows,
                    # or an update that silently hit none, would have committed.
                    # Counted around the handler rather than per-handler expect
                    # dicts: one place, and it covers the whole class.
                    before_changes = conn.total_changes
                    detail = handler(conn, record.get("params") or {}, id_map)
                    changed = conn.total_changes - before_changes
                    expected = _EXPECTED_CHANGES.get(
                        _TOOL_ALIASES.get(tool), None)
                    if expected is not None and changed != expected:
                        raise _ReplayConflict(
                            f"blast radius: {changed} row(s) changed, expected "
                            f"{expected} — rolled back")
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
    """The exercise's id in THIS database. NAME FIRST — the id is not portable.

    This preferred the journalled `exercise_id` and fell back to the name. The
    id is the one the exercise held in the database the record was WRITTEN to;
    across a fresh export it can belong to a different exercise entirely (you
    add or delete an exercise in FitNotes and the numbering moves). Replay would
    then attach the workout to whatever exercise now holds that id — no
    conflict, no error, silently filed under the wrong lift.

    Every other stale-id defect in this system LOSES data, which is visible.
    This one MISFILES it, which is not. So the stable key wins: names round-trip
    through an export, ids do not.

    The id stays as the fallback for records journalled before the name was
    recorded — that is every record written before this change, and there is no
    way to recover a name for them retroactively.
    """
    name = params.get("exercise_name") or params.get("exercise")
    if name:
        row = conn.execute(
            "SELECT _id FROM exercise WHERE name = ?", (name,)).fetchone()
        if row:
            return row["_id"]
        # A name that does not exist here is a genuine conflict — never fall
        # back to the id, or the "wrong exercise" case returns by the back door.
        raise _ReplayConflict(f"exercise {name!r} not found in the new database")
    if params.get("exercise_id") is not None:
        return int(params["exercise_id"])
    raise _ReplayConflict("params carry neither exercise_name nor exercise_id")


def _set_metric_weight(s: dict) -> float:
    if s.get("metric_weight") is not None:
        return float(s["metric_weight"])
    if s.get("weight") is not None:        # typed value (lbs frame)
        return float(s["weight"]) / 2.2046
    raise _ReplayConflict("set carries neither metric_weight nor weight")


def _mapped_id(params: dict, id_map: dict) -> int:
    """Translate a journalled row id into its id in THIS database.

    A WAL record names the id a row held in the database it was written to. On a
    fresh export that id means something else entirely — the app's own counter
    advanced independently, so the agent's local id 6 and the app's id 6 are
    different workouts. Reaching for the raw id silently rewrites the user's data.

    So a row is only reachable if replay INSERTED it earlier in this same run and
    recorded the translation. Anything else is a conflict — surfaced to the user,
    never guessed at. That is safe because the agent can no longer edit or delete
    a row it did not create (see the app-data lock in combined_server), which
    means every legitimate edit or delete has its insert in the journal too.
    """
    raw = int(params["set_id"])
    if not id_map or raw not in id_map:
        raise _ReplayConflict(
            f"set_id={raw} was not created by this replay — refusing to touch "
            f"row {raw} of the uploaded database, which is a different set"
        )
    return id_map[raw]


def _register_id(params: dict, sets: list, s: dict, row_id: int,
                 id_map: dict | None) -> None:
    """Record `this record's old id for set s` → `its id in THIS database`.

    A journalled edit or delete names the id the row held where it was written;
    this is the only way it can be rewritten to the row here. `row_ids` is
    positional, matching the staged set order, so it must line up 1:1 with
    `sets` before any of it can be trusted.

    Called from BOTH the insert and the duplicate-skip paths — a row that
    already exists is just as reachable as one we created, and skipping the
    registration is what stranded an edit and a delete on a phantom row.
    """
    old_ids = params.get("row_ids") or []
    if id_map is None or len(old_ids) != len(sets):
        return
    id_map[int(old_ids[sets.index(s)])] = row_id


def _replay_workout(conn: sqlite3.Connection, params: dict,
                    id_map: dict | None = None) -> str:
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
            """SELECT _id FROM training_log
               WHERE exercise_id = ? AND date = ? AND reps = ?
                 AND ABS(metric_weight - ?) < ?
                 AND ABS(distance - ?) < ?
                 AND duration_seconds = ?
               LIMIT 1""",
            (exercise_id, date_str, reps, metric_weight, _WEIGHT_EPSILON,
             distance, _WEIGHT_EPSILON, duration),
        ).fetchone()
        if dup:
            # A SKIP STILL HAS TO MAP ITS IDS. The row exists, so this record's
            # old id and that row are the same set — and any journalled edit or
            # delete naming the old id must resolve to it.
            #
            # Not mapping here is how a phantom row reached real training data
            # and sat there for eleven days. TWO records described one workout:
            # the first carried no row_ids (journalled before they were added)
            # and inserted the row; the second carried row_ids=[15218] and was
            # skipped as a duplicate — so 15218 entered no map, and the update
            # and the delete that named it both conflicted. The insert applied
            # and nothing could undo it.
            #
            # Refusing untrackable inserts instead would be the wrong fix: eight
            # of the user's real July workouts replayed from row_ids=None
            # records and would have been dropped.
            _register_id(params, sets, s, dup["_id"], id_map)
            skipped += 1
            continue
        # Shared writer (also used by the live execute handler) → a replay
        # reproduces an identical row. metric_weight is recovered above for the
        # dedup check, so thread it onto the dict the helper writes.
        new_id = insert_training_log_set(
            conn, exercise_id, date_str, {**s, "metric_weight": metric_weight})
        if s.get("comment"):
            insert_set_comment(conn, new_id, date_str, s["comment"])
        _register_id(params, sets, s, new_id, id_map)
        inserted += 1

    if inserted == 0:
        raise _ReplayConflict(
            f"all {skipped} set(s) already exist for exercise_id={exercise_id} "
            f"on {date_str}")
    return f"{inserted} set(s) inserted, {skipped} already present"


def _replay_goal(conn: sqlite3.Connection, params: dict,
                 id_map: dict | None = None) -> str:
    exercise_id = _resolve_exercise_id(conn, params)
    metric_weight = float(params["metric_weight"])
    reps = int(params["reps"])
    target_date = params["target_date"]

    dup = conn.execute(
        """SELECT _id FROM Goal
           WHERE exercise_id = ? AND target_date = ? AND reps = ?
             AND ABS(metric_weight - ?) < ?
           LIMIT 1""",
        (exercise_id, target_date, reps, metric_weight, _WEIGHT_EPSILON),
    ).fetchone()
    if dup:
        # Map BEFORE raising. The goal exists, so a journalled edit or delete
        # naming its old id must still reach it — skipping the registration is
        # exactly what stranded set edits on a phantom row for eleven days.
        _register_goal_id(params, dup["_id"], id_map)
        raise _ReplayConflict(
            f"identical goal already exists for exercise_id={exercise_id} "
            f"target_date={target_date}")

    cur = conn.execute(
        """INSERT INTO Goal
           (type_id, exercise_id, metric_weight, reps, unit, title, target_date,
            sort_order, distance, duration_seconds, start_date)
           VALUES (1, ?, ?, ?, 0, ?, ?, 0, 0, 0, ?)""",
        (exercise_id, metric_weight, reps,
         params.get("title", ""), target_date, params.get("start_date", "")),
    )
    _register_goal_id(params, cur.lastrowid, id_map)
    return "goal inserted"


def _register_goal_id(params: dict, row_id: int, id_map: dict | None) -> None:
    """Record `this goal's id where it was written` → `its id HERE`.

    Goals are namespaced apart from set ids in the same map: both are small
    integers from different tables, and 'Goal 3' must never resolve to
    'training_log 3'.
    """
    old = params.get("goal_id")
    if id_map is None or old is None:
        return
    id_map[f"goal:{int(old)}"] = row_id


def _mapped_goal_id(params: dict, id_map: dict | None) -> int:
    """The goal's id in THIS database, or a conflict.

    Sets have had this since the app-data lock; goals never did. They went
    straight to `WHERE _id = <journalled id>`, which means two things, one
    observed and one waiting: a goal the user created AND deleted came back to
    life because the delete could not find the remapped row (Phase 8c), and a
    delete naming an id the user's own export happens to use would destroy
    THEIR goal. Same defect, both directions.
    """
    raw = params.get("goal_id")
    if raw is None:
        raise _ReplayConflict("goal record carries no goal_id")
    key = f"goal:{int(raw)}"
    if not id_map or key not in id_map:
        raise _ReplayConflict(
            f"goal_id={raw} was not created by this replay — refusing to touch "
            f"goal {raw} of the uploaded database, which is a different goal")
    return id_map[key]


def _replay_goal_update(conn: sqlite3.Connection, params: dict,
                        id_map: dict | None = None) -> str:
    goal_id = _mapped_goal_id(params, id_map)
    cur = conn.execute(
        "UPDATE Goal SET metric_weight = ?, reps = ?, target_date = ? WHERE _id = ?",
        (float(params["new_metric_weight"]), int(params["new_reps"]),
         params["new_target_date"], goal_id),
    )
    if cur.rowcount == 0:
        raise _ReplayConflict(f"Goal _id={goal_id} not found")
    return "goal updated"


def _replay_goal_delete(conn: sqlite3.Connection, params: dict,
                        id_map: dict | None = None) -> str:
    goal_id = _mapped_goal_id(params, id_map)
    cur = conn.execute("DELETE FROM Goal WHERE _id = ?", (goal_id,))
    if cur.rowcount == 0:
        raise _ReplayConflict(f"Goal _id={goal_id} not found")
    return "goal deleted"


def _replay_set_update(conn: sqlite3.Connection, params: dict,
                       id_map: dict | None = None) -> str:
    set_id = _mapped_id(params, id_map)
    new_metric_weight = float(params["new_metric_weight"])
    new_reps = int(params["new_reps"])

    target = conn.execute(
        "SELECT exercise_id FROM training_log WHERE _id = ?", (set_id,)).fetchone()
    if not target:
        raise _ReplayConflict(f"training_log _id={set_id} not found")

    # is_personal_record is NOT written. It is FitNotes' column and means "was a
    # PR when performed"; the live tool stopped recomputing it, and replay doing
    # so anyway would mean the same edit lands differently depending on which
    # path applied it.
    conn.execute(
        "UPDATE training_log SET metric_weight = ?, reps = ? WHERE _id = ?",
        (new_metric_weight, new_reps, set_id),
    )
    return "set updated"


def _replay_set_delete(conn: sqlite3.Connection, params: dict,
                       id_map: dict | None = None) -> str:
    set_id = _mapped_id(params, id_map)
    cur = conn.execute("DELETE FROM training_log WHERE _id = ?", (set_id,))
    if cur.rowcount == 0:
        raise _ReplayConflict(f"training_log _id={set_id} not found")
    return "set deleted"


def _replay_set_comment(conn: sqlite3.Connection, params: dict,
                        id_map: dict | None = None) -> str:
    """Re-apply a note on a set. THE TARGET IS FOUND BY CONTENT, NOT BY ID.

    Comments had no replay handler at all, so every note the agent wrote was
    dropped on the next upload with `no replay handler for tool ...`. Four of
    them died that way in one real replay.

    Deliberately NOT _mapped_id. That guard refuses any id this replay did not
    create, which is correct for edits — the app-data lock means the agent can
    only edit rows it made — but a COMMENT is the carve-out: a note may sit on
    a set the FitNotes app owns. So the set is re-resolved the way
    set_set_comment found it in the first place, from the content in the
    record. An ambiguous match is a conflict; a note is never guessed onto a
    set.
    """
    raw = params.get("set_id")
    set_id = None
    if raw is not None and id_map and int(raw) in id_map:
        set_id = id_map[int(raw)]                 # replay created this row
    else:
        exercise_id = _resolve_exercise_id(conn, params)
        date_str = params.get("date")
        reps = params.get("reps")
        metric_weight = params.get("stored_weight")
        if metric_weight is None and params.get("weight") is not None:
            metric_weight = float(params["weight"]) / 2.2046
        if not date_str or reps is None or metric_weight is None:
            raise _ReplayConflict(
                "comment record cannot identify its set (needs date, reps and a weight)")
        matches = conn.execute(
            """SELECT _id FROM training_log
               WHERE exercise_id = ? AND date = ? AND reps = ?
                 AND ABS(metric_weight - ?) < ?""",
            (exercise_id, date_str, int(reps), float(metric_weight), _WEIGHT_EPSILON),
        ).fetchall()
        if not matches:
            raise _ReplayConflict(
                f"no set matching the comment's target on {date_str}")
        if len(matches) > 1:
            raise _ReplayConflict(
                f"{len(matches)} identical sets on {date_str} — refusing to "
                f"guess which one the note belongs to")
        set_id = matches[0]["_id"]

    comment = (params.get("comment") or "").strip()
    conn.execute(
        "DELETE FROM Comment WHERE owner_type_id = 1 AND owner_id = ?", (set_id,))
    if comment:
        conn.execute(
            "INSERT INTO Comment (date, owner_type_id, owner_id, comment) "
            "VALUES (?, 1, ?, ?)",
            (params.get("date"), set_id, comment))
        return f"comment set on training_log _id={set_id}"
    return f"comment cleared on training_log _id={set_id}"


def _replay_bodyweight(conn: sqlite3.Connection, params: dict,
                       id_map: dict | None = None) -> str:
    """Re-apply a body-weight entry — one per date, so this UPSERTS.

    Matching on the date rather than the journalled row id is deliberate and is
    not the stale-id shortcut fixed elsewhere: BodyWeight holds at most one
    entry per day by rule, so the date IS the stable key. The id is recorded
    only so a later delete can be mapped.
    """
    date_str = params.get("date")
    weight = params.get("body_weight_metric")
    if not date_str or weight is None:
        raise _ReplayConflict("bodyweight record missing date or weight")
    body_fat = float(params.get("body_fat") or 0)

    existing = conn.execute(
        "SELECT _id, body_weight_metric, body_fat FROM BodyWeight "
        "WHERE date = ? ORDER BY _id LIMIT 1", (date_str,)).fetchone()
    if existing:
        _register_bodyweight_id(params, existing["_id"], id_map)
        if (abs(existing["body_weight_metric"] - float(weight)) < _WEIGHT_EPSILON
                and abs(existing["body_fat"] - body_fat) < _WEIGHT_EPSILON):
            raise _ReplayConflict(
                f"identical body weight already recorded on {date_str}")
        conn.execute(
            "UPDATE BodyWeight SET body_weight_metric = ?, body_fat = ? WHERE _id = ?",
            (float(weight), body_fat, existing["_id"]))
        return f"body weight updated on {date_str}"

    cur = conn.execute(
        "INSERT INTO BodyWeight (date, body_weight_metric, body_fat) VALUES (?, ?, ?)",
        (date_str, float(weight), body_fat))
    _register_bodyweight_id(params, cur.lastrowid, id_map)
    return "body weight inserted"


def _register_bodyweight_id(params: dict, row_id: int, id_map: dict | None) -> None:
    old = params.get("row_id")
    if id_map is None or old is None:
        return
    id_map[f"bw:{int(old)}"] = row_id


def _replay_bodyweight_delete(conn: sqlite3.Connection, params: dict,
                              id_map: dict | None = None) -> str:
    """Delete a weigh-in, through the same id mapping goals and sets use.

    Written with the mapping from the start rather than retrofitted: the row-
    15771 defect has now been found in training_log and again in Goal, and a
    third table with a raw `WHERE _id = <journalled id>` would be the same bug
    a third time.
    """
    raw = params.get("row_id")
    if raw is None:
        raise _ReplayConflict("bodyweight delete record carries no row_id")
    key = f"bw:{int(raw)}"
    if not id_map or key not in id_map:
        raise _ReplayConflict(
            f"body weight row_id={raw} was not created by this replay — refusing "
            f"to touch row {raw} of the uploaded database")
    cur = conn.execute("DELETE FROM BodyWeight WHERE _id = ?", (id_map[key],))
    if cur.rowcount == 0:
        raise _ReplayConflict(f"BodyWeight _id={id_map[key]} not found")
    return "body weight deleted"


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
    # Added after both were found missing on a live upload: comments hit "no
    # replay handler" and were dropped; body weight was never journalled at all.
    # tests assert every tool _wal_append is called with appears here.
    "set_set_comment":             "set_comment",
    "execute_staged_set_comment":  "set_comment",
    "log_bodyweight":              "bodyweight",
    "execute_staged_bodyweight":   "bodyweight",
    "delete_bodyweight":           "bodyweight_delete",
    "execute_staged_bodyweight_delete": "bodyweight_delete",
}

_HANDLERS: dict = {
    "workout":     _replay_workout,
    "goal":        _replay_goal,
    "goal_update": _replay_goal_update,
    "goal_delete": _replay_goal_delete,
    "set_update":  _replay_set_update,
    "set_delete":  _replay_set_delete,
    "set_comment": _replay_set_comment,
    "bodyweight":  _replay_bodyweight,
    "bodyweight_delete": _replay_bodyweight_delete,
}

# Rows a record MUST change, for the handlers whose blast radius is fixed.
# None = variable and checked by the handler itself: a workout inserts N sets
# with per-set duplicate skips, and a comment is a delete-then-insert whose
# count depends on whether a note was already there.
_EXPECTED_CHANGES: dict = {
    "goal":        1,   # one INSERT
    "goal_update": 1,   # one row updated
    "goal_delete": 1,   # one row removed
    "set_update":  1,
    "set_delete":  1,
    "bodyweight":  None,  # upsert: 1 for an insert, 1 for an update,
                          # but 0 when the day already matches exactly
    "bodyweight_delete": 1,
    "workout":     None,
    "set_comment": None,
}
