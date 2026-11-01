"""
Per-visitor conversations for the public demo (DEMO_MODE=1). Inert otherwise.

The app was built for one user: one AgentSession, one Coordinator, one
confirmation panel, one checkpoint slot. Loading all of that per visitor would
mean one MCP tool server (embedding models, Chroma) per browser, which a
1 OCPU / 6 GB VM cannot afford. So the expensive objects stay single, and each
visitor's CONVERSATION state is swapped in and out around their turn:

    queue grant → load the visitor's state into the live objects → run the turn
    → capture it back → save → blank the live objects → next visitor

Turns are serialized anyway (one agent lock), so nothing can observe a
half-swapped state. Read-only endpoints (/history, /pending-*) answer straight
from the store and never touch the live objects.

What one visitor's state holds (see capture / restore):
  * the AgentSession's chat_history (what the page shows) and the last ten
    exchanges the agent is prompted with;
  * the Coordinator's pending follow-up / disambiguation / log-carry fields and
    its analytical history;
  * server.py's _state (the confirmation panel and its stash);
  * the src/checkpoint.py slot (an interrupted question to resume);
  * a memory overlay (remembered facts + age/sex/height), used by src/memory.py
    in place of data/memory.json while that visitor's turn runs.

Storage is one SQLite file on the data volume (default
data/visitor_sessions/sessions.sqlite), which the demo reset leaves alone.

Environment
-----------
DEMO_SESSIONS_PATH        SQLite file (default data/visitor_sessions/sessions.sqlite)
DEMO_SESSION_TTL_DAYS     delete a session after this many days without a question (30)
DEMO_MAX_SESSIONS         keep at most this many sessions; LRU evicted (5000)
DEMO_SESSIONS_MAX_MB      keep the store under this many MB; LRU evicted (200)
DEMO_SESSION_MAX_KB       per-session size; oldest chat entries trimmed past it (256)
DEMO_MAX_QUEUED_TURNS     turns that may wait behind the running one (0: none;
                          a visitor who asks while another is answered is told
                          to try again in a minute)
DEMO_TURN_TIMEOUT_SECONDS stop waiting for a turn still running after this long (300)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import datetime
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

COOKIE_NAME = "fn_sid"
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")

DEFAULT_PATH = os.path.join("data", "visitor_sessions", "sessions.sqlite")


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def sessions_path() -> str:
    return os.environ.get("DEMO_SESSIONS_PATH") or DEFAULT_PATH


def ttl_days() -> int:
    return _int_env("DEMO_SESSION_TTL_DAYS", 30, minimum=1)


def max_sessions() -> int:
    return _int_env("DEMO_MAX_SESSIONS", 5000, minimum=1)


def store_max_bytes() -> int:
    return _int_env("DEMO_SESSIONS_MAX_MB", 200, minimum=1) * 1024 * 1024


def session_max_bytes() -> int:
    return _int_env("DEMO_SESSION_MAX_KB", 256, minimum=16) * 1024


def max_queued_turns() -> int:
    return _int_env("DEMO_MAX_QUEUED_TURNS", 0)


def turn_timeout_seconds() -> float:
    """A safety net for a turn that hangs (a Gemini request that never comes
    back), not a speed limit: a slow answer, per-minute 429 waits included,
    finishes well inside it."""
    try:
        return max(0.01, float(os.environ.get("DEMO_TURN_TIMEOUT_SECONDS", 300)))
    except (TypeError, ValueError):
        return 300.0


# ── Session ids ──────────────────────────────────────────────────────────────

def new_sid() -> str:
    """128 random bits, URL-safe (22 chars)."""
    return secrets.token_urlsafe(16)


def valid_sid(value: str | None) -> bool:
    return bool(value) and bool(_SID_RE.match(value))


# ── JSON that survives the odd non-JSON value in agent history ──────────────

def _default(o):
    if isinstance(o, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(bytes(o)).decode("ascii")}
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _hook(d: dict):
    if len(d) == 1 and "__b64__" in d:
        return base64.b64decode(d["__b64__"])
    return d


def dumps(state: dict) -> str:
    return json.dumps(state, default=_default, ensure_ascii=False)


def loads(text: str) -> dict:
    return json.loads(text, object_hook=_hook)


# ── What a visitor's state is ────────────────────────────────────────────────

# Attribute → factory for its blank value.
AGENT_FIELDS = {
    "chat_history": list,
    "_conversation_history": list,
    "_staged_active": lambda: False,
}
COORD_FIELDS = {
    "_history": list,
    "_pending_followup": lambda: None,
    "_pending_decomposition": lambda: None,
    "_pending_log_carry": lambda: False,
    "_log_flow_turns": list,
}


def blank_memory() -> dict:
    return {"facts": [], "last_updated": None, "summary": None, "demographics": {}}


def blank_state(server_defaults: dict) -> dict:
    return {
        "agent": {k: f() for k, f in AGENT_FIELDS.items()},
        "coordinator": {k: f() for k, f in COORD_FIELDS.items()},
        "server": copy.deepcopy(server_defaults),
        "checkpoint": None,
        "memory": blank_memory(),
    }


def normalize(state: dict | None, server_defaults: dict) -> dict:
    """Fill in anything missing (an older row, a blank visitor)."""
    out = blank_state(server_defaults)
    if not state:
        return out
    for section in ("agent", "coordinator", "server"):
        out[section].update({k: v for k, v in (state.get(section) or {}).items()
                             if k in out[section]})
    out["checkpoint"] = state.get("checkpoint")
    mem = state.get("memory") or {}
    out["memory"].update(mem)
    out["memory"].setdefault("demographics", {})
    return out


def restore(state: dict, agent, coordinator, server_state: dict) -> None:
    """Load a visitor's (normalized) state into the live objects."""
    for k in AGENT_FIELDS:
        setattr(agent, k, copy.deepcopy(state["agent"][k]))
    if coordinator is not None:
        for k in COORD_FIELDS:
            setattr(coordinator, k, copy.deepcopy(state["coordinator"][k]))
    for k, v in state["server"].items():
        server_state[k] = copy.deepcopy(v)


def clear_unfinished(agent, coordinator, server_state: dict,
                     server_defaults: dict) -> None:
    """After a turn was stopped part-way: drop any panel or pending question it
    half-built, keeping the visitor's chat and conversation history."""
    agent._staged_active = False
    if coordinator is not None:
        for k, f in COORD_FIELDS.items():
            if k != "_history":
                setattr(coordinator, k, f())
    for k, v in server_defaults.items():
        server_state[k] = copy.deepcopy(v)


def capture(agent, coordinator, server_state: dict, server_defaults: dict) -> dict:
    """Read the live objects back into a state dict (checkpoint/memory are
    filled in by the caller, which owns those stores)."""
    state = blank_state(server_defaults)
    for k in AGENT_FIELDS:
        state["agent"][k] = copy.deepcopy(getattr(agent, k, AGENT_FIELDS[k]()))
    if coordinator is not None:
        for k in COORD_FIELDS:
            state["coordinator"][k] = copy.deepcopy(
                getattr(coordinator, k, COORD_FIELDS[k]()))
    for k in server_defaults:
        state["server"][k] = copy.deepcopy(server_state.get(k))
    return state


INTERRUPTED_TEXT = ("This answer was interrupted before it finished. "
                    "Please ask again.")


def settle_stale_pending(history: list) -> bool:
    """A `pending` entry for a visitor with no turn queued or running was cut
    off (the server restarted mid-turn). Turn it into an error the page can
    show, or the page would wait for it forever. Returns whether anything
    changed."""
    changed = False
    for m in history:
        if m.get("pending"):
            m.pop("pending", None)
            m["text"] = INTERRUPTED_TEXT
            m["kind"] = "error"
            changed = True
    return changed


# ── The store ────────────────────────────────────────────────────────────────

class SessionStore:
    """One row per visitor. Thread-safe; every call is a short transaction."""

    def __init__(self, path: str | None = None, clock=time.time):
        self.path = path or sessions_path()
        self._clock = clock
        self._lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                created_at      REAL NOT NULL,
                last_prompt_at  REAL NOT NULL,
                last_used_at    REAL NOT NULL,
                size            INTEGER NOT NULL,
                state           TEXT NOT NULL
            )""")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS sessions_used ON sessions(last_used_at)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- rows --
    def load(self, sid: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            return None
        try:
            return loads(row[0])
        except Exception:
            return None

    def exists(self, sid: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone() is not None

    def save(self, sid: str, state: dict, *, prompted: bool = False,
             protect: set | frozenset = frozenset()) -> int:
        """Write a visitor's state. `prompted` marks a new question (the TTL
        clock restarts). Trims the visitor to the per-session size cap, then
        enforces the store caps (never evicting `protect`). Returns the size."""
        text = self._fit(state)
        now = self._clock()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET state = ?, size = ?, last_used_at = ?"
                + (", last_prompt_at = ?" if prompted else "")
                + " WHERE id = ?",
                (text, len(text), now, now, sid) if prompted
                else (text, len(text), now, sid))
            created = cur.rowcount == 0
            if created:
                self._conn.execute(
                    "INSERT INTO sessions (id, created_at, last_prompt_at, "
                    "last_used_at, size, state) VALUES (?, ?, ?, ?, ?, ?)",
                    (sid, now, now, now, len(text), text))
        if created:
            self.enforce_caps(protect=set(protect) | {sid})
        return len(text)

    def delete(self, sid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    def _fit(self, state: dict) -> str:
        """Serialize, trimming the oldest chat entries until it fits."""
        limit = session_max_bytes()
        text = dumps(state)
        if len(text.encode("utf-8")) <= limit:
            return text
        state = copy.deepcopy(state)
        hist = state.get("agent", {}).get("chat_history", [])
        conv = state.get("agent", {}).get("_conversation_history", [])
        chist = state.get("coordinator", {}).get("_history", [])
        while len(text.encode("utf-8")) > limit:
            if len(hist) > 2:
                del hist[:max(1, len(hist) // 4)]
            elif len(conv) > 1:
                del conv[0]
            elif len(chist) > 2:
                del chist[:2]
            else:
                break
            text = dumps(state)
        return text

    # -- housekeeping --
    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def total_bytes(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM sessions").fetchone()[0]

    def sweep(self, protect: set | frozenset = frozenset()) -> int:
        """Delete sessions with no question for longer than the TTL."""
        cutoff = self._clock() - ttl_days() * 86400
        with self._lock:
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM sessions WHERE last_prompt_at < ?", (cutoff,))]
            ids = [i for i in ids if i not in protect]
            self._conn.executemany("DELETE FROM sessions WHERE id = ?",
                                   [(i,) for i in ids])
        return len(ids)

    def enforce_caps(self, protect: set | frozenset = frozenset()) -> int:
        """Evict least-recently-used sessions until under both caps."""
        evicted = 0
        max_n, max_b = max_sessions(), store_max_bytes()
        with self._lock:
            n, total = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM sessions").fetchone()
            if n <= max_n and total <= max_b:
                return 0
            for sid, size in self._conn.execute(
                    "SELECT id, size FROM sessions ORDER BY last_used_at ASC").fetchall():
                if n <= max_n and total <= max_b:
                    break
                if sid in protect:
                    continue
                self._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                n -= 1
                total -= size
                evicted += 1
        return evicted

    def housekeeping(self, protect: set | frozenset = frozenset()) -> tuple[int, int]:
        expired = self.sweep(protect)
        evicted = self.enforce_caps(protect)
        if expired or evicted:
            with self._lock:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return expired, evicted


# ── Turn queue ───────────────────────────────────────────────────────────────

class AlreadyActive(Exception):
    """This visitor already has a turn queued or running."""


class QueueFull(Exception):
    """Too many turns are already waiting."""



class TurnQueue:
    """FIFO of visitors' turns: one runs, up to `max_waiting` wait behind it.
    A visitor holds at most one place. Turns run back to back."""

    def __init__(self):
        self._running: str | None = None
        self._waiting: list[list] = []          # [sid, future]

    @property
    def running(self) -> str | None:
        return self._running

    def active(self) -> set:
        return ({self._running} if self._running else set()) | {w[0] for w in self._waiting}

    def is_active(self, sid: str) -> bool:
        return sid == self._running or any(w[0] == sid for w in self._waiting)

    def ahead(self, sid: str) -> int | None:
        """Turns ahead of this visitor's: 0 = theirs is running, None = none."""
        if sid == self._running:
            return 0
        for i, (s, _) in enumerate(self._waiting):
            if s == sid:
                return i + (1 if self._running else 0)
        return None

    def enter(self, sid: str, max_waiting: int):
        """Take a place, synchronously (so refusals happen before anything is
        recorded). Returns a ticket for wait()."""
        if self.is_active(sid):
            raise AlreadyActive()
        if self._running is None and not self._waiting:
            self._running = sid
            return None
        if len(self._waiting) >= max_waiting:
            raise QueueFull()
        entry = [sid, asyncio.get_running_loop().create_future()]
        self._waiting.append(entry)
        return entry

    async def wait(self, sid: str, ticket) -> None:
        if ticket is None:
            return
        try:
            await ticket[1]
        except BaseException:
            if ticket in self._waiting:
                self._waiting.remove(ticket)
            elif self._running == sid:          # granted, then cancelled
                self.release(sid)
            raise

    def release(self, sid: str) -> None:
        if self._running != sid:
            return
        self._running = None
        while self._waiting:
            nxt, fut = self._waiting.pop(0)
            if not fut.done():
                self._running = nxt
                fut.set_result(None)
                return


def queue_message(ahead: int) -> str:
    others = "1 other" if ahead == 1 else f"{ahead} others"
    return ("A few people are using the demo right now. Your question is "
            f"queued and will be answered after {others}.")


FULL_TEXT = ("Someone else is using the coach right now. "
             "Please try again in a minute.")
TIMED_OUT_TEXT = ("This answer took too long and was stopped. "
                  "Please ask again.")
DEMO_NOT_SAVED_TEXT = ("This public demo doesn't save changes, so nothing was "
                       "written. Run your own copy to log your training.")
