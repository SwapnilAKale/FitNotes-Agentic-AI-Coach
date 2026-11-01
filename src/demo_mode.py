"""
Public-demo safety gates. Inert unless DEMO_MODE=1.

When the app is hosted as a shared public demo (see DEPLOY.md), every visitor
has their own conversation (src/visitor_sessions.py) over ONE shared,
read-only synthetic database. Two things must hold here:

  1. Nobody can replace, wipe or reconfigure the shared state — the upload,
     wipe and settings endpoints answer 403 with a message the UI shows.
     (Chat-driven writes are refused by the agent, the tool server and
     src/db.py; see AgentSession.read_only_tools.)
  2. Nobody can burn the Gemini quota for everyone — the turn-running
     endpoints are rate limited per client IP (in memory; resets on restart).

Nothing here changes how a question is answered. With DEMO_MODE unset, every
function below is a no-op from the caller's point of view (`check` returns
None), so `python server.py` behaves exactly as before.

Environment
-----------
DEMO_MODE                    1 to enable the gates (default off)
DEMO_RATE_LIMIT_PER_MINUTE   turns per client IP per 60 s      (default 3)
DEMO_RATE_LIMIT_PER_DAY      turns per client IP per 24 h      (default 30)
DEMO_RATE_LIMIT_GLOBAL_PER_DAY  turns for ALL visitors per 24 h (default 0 = off)
TRUSTED_PROXY_CIDRS          comma-separated networks whose X-Forwarded-For is
                             believed (default: loopback + private ranges,
                             i.e. Caddy and the frontend proxy on the Docker
                             network). The client IP is the right-most address
                             in the chain that is NOT a trusted proxy.
"""

from __future__ import annotations

import ipaddress
import math
import os
import threading
import time
from collections import deque

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def enabled() -> bool:
    """Read on every call so tests (and a restart with a new env) see changes."""
    return _flag("DEMO_MODE")


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# ── Endpoints that overwrite, wipe or reconfigure shared state ───────────────
# (method, path) → what the visitor tried to do, for the message.
BLOCKED: dict[tuple[str, str], str] = {
    ("POST", "/upload"):          "uploading a database",
    ("POST", "/upload/confirm"):  "uploading a database",
    ("POST", "/upload/article"):  "uploading articles",
    ("POST", "/wal-wipe"):        "clearing saved chat logs",
    ("POST", "/settings"):        "changing settings",
    ("POST", "/reload-db"):       "reloading the database",
    ("POST", "/ontology-review"): "saving ontology review decisions",
}

# Endpoints that run a (Gemini-backed) agent turn.
RATE_LIMITED: frozenset[tuple[str, str]] = frozenset({
    ("POST", "/chat"),
    ("POST", "/resume"),
    ("POST", "/disambiguate"),
})


def blocked_body(what: str) -> dict:
    text = (f"This is a public demo with synthetic data, so {what} is turned off. "
            "Run your own copy to use it with your FitNotes backup.")
    # `type`/`text` let the chat render it as an error bubble; `status`/`message`
    # /`errors` let the upload and settings toasts show it unchanged.
    return {"error": "demo_disabled", "type": "error", "status": "error",
            "text": text, "message": text, "errors": [text]}


# ── Client IP behind Caddy → frontend proxy → backend ────────────────────────
_DEFAULT_TRUSTED = ("127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12",
                    "192.168.0.0/16", "fc00::/7")


def _trusted_networks() -> list:
    raw = os.environ.get("TRUSTED_PROXY_CIDRS")
    cidrs = [c.strip() for c in raw.split(",")] if raw else list(_DEFAULT_TRUSTED)
    nets = []
    for c in cidrs:
        if not c:
            continue
        try:
            nets.append(ipaddress.ip_network(c, strict=False))
        except ValueError:
            pass
    return nets


def _parse_ip(value: str):
    value = (value or "").strip().strip('"')
    if value.startswith("[") and "]" in value:          # [v6]:port
        value = value[1:value.index("]")]
    elif value.count(":") == 1:                          # v4:port
        value = value.split(":", 1)[0]
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def client_ip(peer: str | None, forwarded_for: str | None) -> str:
    """
    The visitor's address. Walk the hop chain right to left — the peer that
    connected to us, then X-Forwarded-For from its right end — skipping hops
    that are trusted proxies. The first untrusted hop is the client: a proxy
    we trust wrote it, so a visitor cannot forge it by sending their own
    X-Forwarded-For (that ends up further LEFT and is never reached).
    If every hop is trusted (a LAN visitor), the left-most one is used.
    """
    trusted = _trusted_networks()
    hops = [h for h in (forwarded_for or "").split(",") if h.strip()]
    chain = hops + ([peer] if peer else [])
    parsed = [ip for ip in (_parse_ip(h) for h in chain) if ip is not None]
    if not parsed:
        return peer or "unknown"
    for ip in reversed(parsed):
        if not any(ip in net for net in trusted if ip.version == net.version):
            return str(ip)
    return str(parsed[0])


# ── Sliding-window rate limiter ──────────────────────────────────────────────
class RateLimiter:
    """In-memory, per-key sliding windows. Thread-safe; O(hits) memory."""

    _MAX_KEYS = 50_000          # bound memory under a flood of distinct IPs

    def __init__(self, per_minute: int, per_day: int, global_per_day: int = 0,
                 clock=time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._global: deque = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _prune(q: deque, now: float, window: float) -> None:
        while q and q[0] <= now - window:
            q.popleft()

    def hit(self, key: str) -> float | None:
        """Record one turn for `key`. Returns None if allowed, else seconds to wait."""
        now = self._clock()
        with self._lock:
            self._prune(self._global, now, 86400)
            if self.global_per_day and len(self._global) >= self.global_per_day:
                return max(1.0, self._global[0] + 86400 - now)

            q = self._hits.get(key)
            if q is None:
                if len(self._hits) >= self._MAX_KEYS:
                    self._evict(now)
                q = self._hits[key] = deque()
            self._prune(q, now, 86400)

            if self.per_day and len(q) >= self.per_day:
                return max(1.0, q[0] + 86400 - now)
            if self.per_minute:
                recent = [t for t in q if t > now - 60]
                if len(recent) >= self.per_minute:
                    return max(1.0, recent[0] + 60 - now)

            q.append(now)
            self._global.append(now)
            return None

    def _evict(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or q[-1] <= now - 86400]:
            del self._hits[k]
        while len(self._hits) >= self._MAX_KEYS:          # still full: drop oldest
            del self._hits[next(iter(self._hits))]


_limiter: RateLimiter | None = None


def limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(
            per_minute=_int_env("DEMO_RATE_LIMIT_PER_MINUTE", 3),
            per_day=_int_env("DEMO_RATE_LIMIT_PER_DAY", 30),
            global_per_day=_int_env("DEMO_RATE_LIMIT_GLOBAL_PER_DAY", 0),
        )
    return _limiter


def rate_limited_body(wait_seconds: float) -> dict:
    wait = int(math.ceil(wait_seconds))
    if wait >= 3600:
        when = f"in about {math.ceil(wait / 3600)} hour(s)"
    elif wait >= 60:
        when = f"in about {math.ceil(wait / 60)} minute(s)"
    else:
        when = f"in {wait} second(s)"
    text = ("This public demo limits how many questions each visitor can ask, "
            f"to keep it free for everyone. Please try again {when}.")
    return {"error": "demo_rate_limit", "type": "error", "text": text,
            "message": text, "retry_after_seconds": wait}


# ── The one call sites make ──────────────────────────────────────────────────
def check(method: str, path: str, peer: str | None,
          forwarded_for: str | None) -> tuple[int, dict, dict] | None:
    """
    None → let the request through. Otherwise (status, json_body, headers) to
    answer with instead. Always None when DEMO_MODE is off.
    """
    if not enabled():
        return None
    key = (method.upper(), (path.rstrip("/") or "/"))
    what = BLOCKED.get(key)
    if what:
        return 403, blocked_body(what), {}
    if key in RATE_LIMITED:
        wait = limiter().hit(client_ip(peer, forwarded_for))
        if wait is not None:
            body = rate_limited_body(wait)
            return 429, body, {"Retry-After": str(body["retry_after_seconds"])}
    return None


def open_browser_allowed() -> bool:
    """Never in demo mode or a container; otherwise OPEN_BROWSER (default on)."""
    if enabled():
        return False
    return _flag("OPEN_BROWSER", "1")
