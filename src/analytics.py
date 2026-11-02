"""
Private visitor analytics for the public demo. Off unless DEMO_MODE=1 (or
ANALYTICS_ENABLED=1), so a local, personal run is never logged.

What it records, in one SQLite file on the data volume (default
data/analytics/analytics.sqlite). The demo reset, the nightly restart and the
30-day visitor-session expiry never touch it:

  visits  one row per page load whose JavaScript ran (the page sends a beacon
          after its first /status reply, so crawlers that don't run JS never
          count). ?src= tag, referrer, screen, language, timezone.
  turns   one row per question / button press that runs a turn (/chat,
          /resume, /disambiguate, /confirm), including the ones refused before
          they ran: question, answer, route, latency, outcome, Gemini tokens.
  events  UI clicks sent by the same beacon (graph button, settings button).
          `kind` is free text checked against EVENT_KINDS, so a new kind is
          one entry there plus one line in index.html.

Every row carries the visitor id (the fn_sid cookie, src/visitor_sessions.py),
the raw client IP (demo_mode.client_ip, i.e. the visitor, not Caddy), the full
user-agent, device / browser / OS parsed from it, and city / country from an
offline DB-IP Lite database (IP Geolocation by DB-IP, https://db-ip.com, CC BY 4.0).

Nothing is ever read back over HTTP: there is no endpoint that returns this
data. The owner copies a snapshot over SSH (scripts/export_analytics.py).

Writes never slow a request: record_*() puts the row on a bounded queue and
returns; one background thread does the IP lookup, the user-agent parsing and
the INSERT. If the queue is full (it never should be), the row is dropped.

Outcomes (turns.outcome)
------------------------
ok               answered (an answer, a confirm panel, a disambiguation panel)
error            anything else that went wrong (`detail` says what, if known)
turn_timeout     stopped at the per-turn limit (DEMO_TURN_TIMEOUT_SECONDS)
gemini_timeout   a Gemini request hit GEMINI_TIMEOUT_SECONDS
demo_full        refused: someone else was being answered
demo_rate_limit  refused: this visitor's per-minute / per-day demo limit
gemini_429       Gemini quota (429 RESOURCE_EXHAUSTED)
busy             refused: this visitor's own previous turn was still running
not_ready        refused: the app was still starting

Environment
-----------
ANALYTICS_ENABLED            1/0; default: the value of DEMO_MODE
ANALYTICS_DB_PATH            SQLite file (default data/analytics/analytics.sqlite)
GEOIP_DB_PATH                DB-IP City Lite .mmdb (default
                             /app/seed/geo/dbip-city-lite.mmdb, then
                             data/geo/dbip-city-lite.mmdb); missing = no location
ANALYTICS_BEACON_PER_MINUTE  beacons per client IP per minute (default 30)
ANALYTICS_BEACON_PER_DAY     beacons per client IP per day    (default 500)
"""

from __future__ import annotations

import contextvars
import datetime
import functools
import json
import os
import queue
import re
import sqlite3
import sys
import threading
from pathlib import Path

from src import demo_mode

_TRUE = {"1", "true", "yes", "on"}

DEFAULT_PATH = os.path.join("data", "analytics", "analytics.sqlite")
GEOIP_DEFAULT_PATHS = ("/app/seed/geo/dbip-city-lite.mmdb",
                       os.path.join("data", "geo", "dbip-city-lite.mmdb"))

OUTCOMES = ("ok", "error", "turn_timeout", "gemini_timeout", "demo_full",
            "demo_rate_limit", "gemini_429", "busy", "not_ready")

# Beacon kinds the page may send. "visit" fills `visits`; the rest are `events`.
VISIT_KIND = "visit"
EVENT_KINDS = frozenset({"graph_click", "settings_click"})

BEACON_MAX_BYTES = 4096
_QUEUE_MAX = 10_000

# Per-field caps, so a hostile client cannot bloat the file through the beacon.
# (Questions and answers come from the server's own turn and are kept whole.)
_CAP = {"src": 100, "referrer": 1000, "path": 300, "screen": 20,
        "language": 40, "timezone": 64, "detail": 500, "user_agent": 1000}


def enabled() -> bool:
    """ANALYTICS_ENABLED when set, else on exactly when DEMO_MODE is."""
    raw = os.environ.get("ANALYTICS_ENABLED")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in _TRUE
    return demo_mode.enabled()


def db_path() -> str:
    return os.environ.get("ANALYTICS_DB_PATH") or DEFAULT_PATH


def geoip_path() -> str | None:
    custom = os.environ.get("GEOIP_DB_PATH")
    if custom:
        return custom
    for p in GEOIP_DEFAULT_PATHS:
        if os.path.isfile(p):
            return p
    return None


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _cap(value, n: int):
    if value is None:
        return None
    value = str(value)
    return value[:n]


# ── User-agent → device / browser / OS ───────────────────────────────────────
# Deliberately small: enough to tell a phone from a laptop and Chrome from
# Safari. The raw user-agent is stored too, so the dashboard can re-parse it.

_BOT_RE = re.compile(
    r"bot\b|bot/|crawl|spider|slurp|headless|lighthouse|pagespeed|preview|"
    r"facebookexternalhit|embedly|curl/|wget/|python-requests|python-urllib|"
    r"httpx|go-http-client|okhttp|java/|axios|node-fetch|phantomjs|puppeteer|"
    r"playwright|selenium|scrapy|monitor|uptime", re.I)


def parse_user_agent(ua: str | None) -> dict:
    """{device, browser, os, is_bot} from a user-agent string."""
    ua = ua or ""
    is_bot = bool(_BOT_RE.search(ua)) or not ua.strip()

    if re.search(r"Windows NT", ua):
        os_name = "Windows"
    elif re.search(r"iPhone|iPad|iPod", ua):
        os_name = "iOS"
    elif re.search(r"Android", ua):
        os_name = "Android"
    elif re.search(r"CrOS", ua):
        os_name = "ChromeOS"
    elif re.search(r"Mac OS X|Macintosh", ua):
        os_name = "macOS"
    elif re.search(r"Linux", ua):
        os_name = "Linux"
    else:
        os_name = "Other"

    # Order matters: Edge / Opera / Samsung all claim "Chrome", and Chrome
    # claims "Safari".
    if re.search(r"Edg(e|A|iOS)?/", ua):
        browser = "Edge"
    elif re.search(r"OPR/|Opera", ua):
        browser = "Opera"
    elif re.search(r"SamsungBrowser/", ua):
        browser = "Samsung Internet"
    elif re.search(r"Firefox/|FxiOS/", ua):
        browser = "Firefox"
    elif re.search(r"Chrome/|CriOS/|Chromium/", ua):
        browser = "Chrome"
    elif re.search(r"Safari/", ua) and re.search(r"Version/", ua):
        browser = "Safari"
    else:
        browser = "Other"

    if is_bot:
        device = "bot"
    elif re.search(r"iPad|Tablet", ua) or (os_name == "Android"
                                            and not re.search(r"Mobile", ua)):
        device = "tablet"
    elif re.search(r"Mobi|iPhone|iPod|Android", ua):
        device = "mobile"
    else:
        device = "desktop"
    return {"device": device, "browser": browser, "os": os_name,
            "is_bot": int(is_bot)}


# ── IP → city / country (DB-IP City Lite, offline) ──────────────────────────

class GeoLookup:
    """Lazy MMDB reader. Every failure (no file, no package, private IP)
    returns empty fields: location is nice to have, never required."""

    def __init__(self, path: str | None = None, reader=None):
        self._path = path
        self._reader = reader
        self._tried = reader is not None

    def _open(self):
        if self._tried:
            return self._reader
        self._tried = True
        path = self._path or geoip_path()
        if not path or not os.path.isfile(path):
            return None
        try:
            import maxminddb
            self._reader = maxminddb.open_database(path)
        except Exception as exc:
            print(f"[analytics] IP database unusable ({exc}); no location",
                  file=sys.stderr)
            self._reader = None
        return self._reader

    def lookup(self, ip: str | None) -> dict:
        out = {"country": None, "country_code": None, "region": None,
               "city": None, "latitude": None, "longitude": None}
        reader = self._open()
        if reader is None or not ip:
            return out
        try:
            rec = reader.get(ip) or {}
        except Exception:
            return out

        def name(d):
            names = (d or {}).get("names") or {}
            return names.get("en") or next(iter(names.values()), None)

        country = rec.get("country") or {}
        subs = rec.get("subdivisions") or []
        loc = rec.get("location") or {}
        out.update(country=name(country), country_code=country.get("iso_code"),
                   region=name(subs[0]) if subs else None,
                   city=name(rec.get("city")),
                   latitude=loc.get("latitude"), longitude=loc.get("longitude"))
        return out

    def close(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass


# ── Schema ───────────────────────────────────────────────────────────────────

_WHO = """
    ts           TEXT NOT NULL,      -- UTC, ISO 8601
    day          TEXT NOT NULL,      -- UTC date, YYYY-MM-DD
    visitor_id   TEXT,               -- fn_sid cookie (NULL outside the demo)
    ip           TEXT,               -- raw client IP
    user_agent   TEXT,
    device       TEXT,               -- desktop / mobile / tablet / bot
    browser      TEXT,
    os           TEXT,
    is_bot       INTEGER NOT NULL DEFAULT 0,
    country      TEXT,
    country_code TEXT,
    region       TEXT,
    city         TEXT,
    latitude     REAL,
    longitude    REAL"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY,{_WHO},
    src          TEXT,               -- ?src= tag on the page URL
    referrer     TEXT,
    path         TEXT,
    screen       TEXT,
    language     TEXT,
    timezone     TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,{_WHO},
    endpoint     TEXT NOT NULL,      -- chat / resume / disambiguate / confirm / cancel
    question     TEXT,
    answer       TEXT,
    response_type TEXT,              -- answer / confirmation_required / ...
    route        TEXT,
    outcome      TEXT NOT NULL,      -- see OUTCOMES in src/analytics.py
    detail       TEXT,
    http_status  INTEGER,
    latency_ms   INTEGER,
    retry_after_seconds INTEGER,
    gemini_calls INTEGER,
    prompt_tokens   INTEGER,
    output_tokens   INTEGER,
    thinking_tokens INTEGER,
    total_tokens    INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,{_WHO},
    kind         TEXT NOT NULL,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS visits_day     ON visits(day);
CREATE INDEX IF NOT EXISTS visits_visitor ON visits(visitor_id);
CREATE INDEX IF NOT EXISTS turns_day      ON turns(day);
CREATE INDEX IF NOT EXISTS turns_visitor  ON turns(visitor_id);
CREATE INDEX IF NOT EXISTS events_day     ON events(day);
CREATE INDEX IF NOT EXISTS events_visitor ON events(visitor_id);
"""

SLOW_ANSWER_MS = 60_000

# Views are dropped and re-created on every start, so a change here reaches an
# existing file. They hold no data of their own.
VIEWS = {
    # Page loads and unique visitors per day (bots counted separately).
    "daily_visits": """
        SELECT day,
               SUM(is_bot = 0)                                   AS page_loads,
               COUNT(DISTINCT CASE WHEN is_bot = 0
                     THEN COALESCE(visitor_id, ip) END)          AS unique_visitors,
               SUM(is_bot = 1)                                   AS bot_page_loads
        FROM visits GROUP BY day""",
    # One row per visitor: first / last seen, visits, questions, bounce.
    "visitors": """
        WITH seen AS (
            SELECT visitor_id, ts, ip, country, city, device, browser, os, is_bot
              FROM visits WHERE visitor_id IS NOT NULL
            UNION ALL
            SELECT visitor_id, ts, ip, country, city, device, browser, os, is_bot
              FROM turns WHERE visitor_id IS NOT NULL
            UNION ALL
            SELECT visitor_id, ts, ip, country, city, device, browser, os, is_bot
              FROM events WHERE visitor_id IS NOT NULL
        ),
        latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY visitor_id
                                         ORDER BY ts DESC) AS rn FROM seen
        ),
        span AS (
            SELECT visitor_id, MIN(ts) AS first_seen, MAX(ts) AS last_seen,
                   MAX(is_bot) AS is_bot
              FROM seen GROUP BY visitor_id
        ),
        v AS (
            SELECT visitor_id, COUNT(*) AS visits,
                   (SELECT src FROM visits v2 WHERE v2.visitor_id = v1.visitor_id
                      AND COALESCE(src, '') <> '' ORDER BY id LIMIT 1) AS first_src,
                   (SELECT referrer FROM visits v2 WHERE v2.visitor_id = v1.visitor_id
                      AND COALESCE(referrer, '') <> '' ORDER BY id LIMIT 1) AS first_referrer
              FROM visits v1 WHERE visitor_id IS NOT NULL GROUP BY visitor_id
        ),
        t AS (
            SELECT visitor_id,
                   SUM(endpoint = 'chat')                      AS questions,
                   SUM(endpoint = 'chat' AND outcome = 'ok')   AS answered,
                   COUNT(*)                                    AS turns,
                   COALESCE(SUM(total_tokens), 0)              AS tokens
              FROM turns WHERE visitor_id IS NOT NULL GROUP BY visitor_id
        )
        SELECT span.visitor_id, span.first_seen, span.last_seen,
               COALESCE(v.visits, 0)    AS visits,
               COALESCE(t.questions, 0) AS questions,
               COALESCE(t.answered, 0)  AS answered,
               COALESCE(t.turns, 0)     AS turns,
               COALESCE(t.tokens, 0)    AS tokens,
               (COALESCE(t.questions, 0) = 0) AS bounced,
               v.first_src, v.first_referrer,
               latest.ip AS last_ip, latest.country, latest.city,
               latest.device, latest.browser, latest.os, span.is_bot
          FROM span
          JOIN latest ON latest.visitor_id = span.visitor_id AND latest.rn = 1
          LEFT JOIN v ON v.visitor_id = span.visitor_id
          LEFT JOIN t ON t.visitor_id = span.visitor_id""",
    # Did it work? Outcomes, slow answers and tokens per day.
    "daily_outcomes": f"""
        SELECT day,
               COUNT(*)                               AS turns,
               SUM(outcome = 'ok')                    AS ok,
               SUM(outcome = 'error')                 AS errors,
               SUM(outcome = 'turn_timeout')          AS turn_timeouts,
               SUM(outcome = 'gemini_timeout')        AS gemini_timeouts,
               SUM(outcome = 'demo_full')             AS demo_full,
               SUM(outcome = 'demo_rate_limit')       AS demo_rate_limited,
               SUM(outcome = 'gemini_429')            AS gemini_429,
               SUM(outcome = 'busy')                  AS busy,
               SUM(outcome = 'not_ready')             AS not_ready,
               SUM(outcome = 'ok' AND latency_ms >= {SLOW_ANSWER_MS}) AS slow_answers,
               CAST(AVG(CASE WHEN outcome = 'ok' THEN latency_ms END) AS INTEGER)
                                                      AS avg_ok_latency_ms,
               MAX(latency_ms)                        AS max_latency_ms,
               COALESCE(SUM(total_tokens), 0)         AS total_tokens
        FROM turns GROUP BY day""",
}


def open_db(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the analytics file with its schema and views."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    for name, sql in VIEWS.items():
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(f"CREATE VIEW {name} AS {sql}")
    conn.execute("PRAGMA user_version = 1")
    return conn


_COLUMNS: dict[str, list[str]] = {}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if table not in _COLUMNS:
        _COLUMNS[table] = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")
                           if r[1] != "id"]
    return _COLUMNS[table]


# ── The background writer ────────────────────────────────────────────────────

class Recorder:
    """Bounded queue + one writer thread that owns the SQLite connection."""

    _STOP = object()

    def __init__(self, path: str | None = None, geo: GeoLookup | None = None):
        self.path = path or db_path()
        self.geo = geo or GeoLookup()
        self._q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self.dropped = 0

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="analytics-writer")
            self._thread.start()

    def submit(self, table: str, row: dict) -> bool:
        """Queue one row. Never blocks; False if it had to be dropped."""
        self._ensure_thread()
        try:
            self._q.put_nowait((table, row))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        conn = None
        try:
            conn = open_db(self.path)
        except Exception as exc:
            print(f"[analytics] cannot open {self.path}: {exc}", file=sys.stderr)
        while True:
            item = self._q.get()
            try:
                if item is self._STOP:
                    return
                if conn is not None:
                    self._write(conn, *item)
            except Exception as exc:
                print(f"[analytics] write failed: {exc}", file=sys.stderr)
            finally:
                self._q.task_done()
                if item is self._STOP and conn is not None:
                    conn.close()

    def _write(self, conn: sqlite3.Connection, table: str, row: dict) -> None:
        row = dict(row)
        row.update(parse_user_agent(row.get("user_agent")))
        row.update(self.geo.lookup(row.get("ip")))
        cols = [c for c in _columns(conn, table) if c in row]
        conn.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [row[c] for c in cols])

    def flush(self) -> None:
        """Wait until everything queued so far is written (tests, shutdown)."""
        if self._thread is not None and self._thread.is_alive():
            self._q.join()

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._q.put(self._STOP)
            self._thread.join(timeout=10)
        self._thread = None
        self.geo.close()


_recorder: Recorder | None = None
_recorder_lock = threading.Lock()


def recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        with _recorder_lock:
            if _recorder is None:
                _recorder = Recorder()
    return _recorder


def set_recorder(rec: Recorder | None) -> None:
    """Swap the process-wide recorder (tests)."""
    global _recorder
    _recorder = rec


def start() -> None:
    """Called once at server start when enabled()."""
    install_usage_counter()
    recorder()._ensure_thread()


def shutdown() -> None:
    global _recorder
    if _recorder is not None:
        _recorder.close()
        _recorder = None


# ── Who is asking ────────────────────────────────────────────────────────────

def who(request, visitor_id: str | None = None) -> dict:
    """The per-row identity fields, from a Starlette request. Cheap: the IP
    lookup and user-agent parsing happen later, in the writer thread."""
    now = _now()
    ip = None
    ua = None
    if request is not None:
        ip = demo_mode.client_ip(request.client.host if request.client else None,
                                 request.headers.get("x-forwarded-for"))
        ua = request.headers.get("user-agent")
        if visitor_id is None:
            visitor_id = getattr(getattr(request, "state", None), "sid", None)
        if visitor_id is None:
            from src.visitor_sessions import COOKIE_NAME, valid_sid
            cookie = request.cookies.get(COOKIE_NAME)
            visitor_id = cookie if valid_sid(cookie) else None
    return {"ts": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "day": now.date().isoformat(), "visitor_id": visitor_id,
            "ip": ip, "user_agent": _cap(ua, _CAP["user_agent"])}


# ── Per-turn context: route and Gemini token usage ──────────────────────────

class TurnStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.route: str | None = None
        self.gemini_calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.total_tokens = 0

    def add_call(self) -> None:
        with self._lock:
            self.gemini_calls += 1

    def add_usage(self, usage) -> None:
        if usage is None:
            return

        def n(name):
            v = getattr(usage, name, None)
            return int(v) if isinstance(v, (int, float)) else 0

        with self._lock:
            self.prompt_tokens += n("prompt_token_count")
            self.output_tokens += n("candidates_token_count")
            self.thinking_tokens += n("thoughts_token_count")
            self.total_tokens += n("total_token_count")


_current: contextvars.ContextVar[TurnStats | None] = contextvars.ContextVar(
    "fitnotes_analytics_turn", default=None)


def begin_turn() -> tuple[TurnStats, contextvars.Token]:
    stats = TurnStats()
    return stats, _current.set(stats)


def end_turn(token: contextvars.Token) -> None:
    try:
        _current.reset(token)
    except ValueError:
        _current.set(None)


def note_route(result) -> None:
    """Called by the turn with the coordinator's result once it has routed the
    question. Tolerates anything (never raises into the turn)."""
    stats = _current.get()
    route = result.get("route") if isinstance(result, dict) else None
    if stats is not None and route:
        stats.route = str(route)


def install_usage_counter() -> None:
    """Wrap google-genai's Models.generate_content once, so every Gemini call
    made in this process during a turn adds its usage_metadata to that turn.
    asyncio.to_thread and new tasks copy the context, so calls made from
    worker threads are counted too. (Calls made inside the MCP tool server
    subprocess are not.) A no-op outside a turn."""
    try:
        from google.genai import models
    except Exception:
        return
    original = models.Models.generate_content
    if getattr(original, "_fitnotes_analytics", False):
        return

    @functools.wraps(original)
    def generate_content(self, *args, **kwargs):
        stats = _current.get()
        if stats is not None:
            stats.add_call()
        response = original(self, *args, **kwargs)
        if stats is not None:
            try:
                stats.add_usage(getattr(response, "usage_metadata", None))
            except Exception:
                pass
        return response

    generate_content._fitnotes_analytics = True
    models.Models.generate_content = generate_content


# ── Turn outcome ─────────────────────────────────────────────────────────────

def classify(status: int, body: dict, *, turn_timeout_text: str,
             gemini_timeout_text: str) -> tuple[str, str | None]:
    """(outcome, detail) from the response a visitor was given."""
    err = body.get("error")
    text = body.get("text")
    if err == "demo_full":
        return "demo_full", None
    if err == "demo_rate_limit":
        return "demo_rate_limit", None
    if err == "rate_limit":
        return "gemini_429", None
    if status == 429 and isinstance(err, str) and "busy" in err.lower():
        return "busy", None
    if body.get("type") == "not_ready":
        return "not_ready", None
    if text == turn_timeout_text:
        return "turn_timeout", None
    if text == gemini_timeout_text:
        return "gemini_timeout", None
    if body.get("type") == "error" or err or status >= 400:
        return "error", _cap(err or text or f"HTTP {status}", _CAP["detail"])
    return "ok", None


def _answer_of(body: dict) -> str | None:
    t = body.get("type")
    if t == "confirmation_required":
        return body.get("preview")
    if t == "disambiguation_required":
        try:
            return json.dumps(body.get("groups"), ensure_ascii=False)
        except Exception:
            return None
    return body.get("text") or body.get("message")


def record_turn(request, *, endpoint: str, question: str | None,
                status: int, body: dict | None, latency_ms: int | None,
                stats: TurnStats | None = None, turn_timeout_text: str = "",
                gemini_timeout_text: str = "",
                visitor_id: str | None = None,
                identity: dict | None = None) -> None:
    """Queue one turn row. `body` None means the turn raised. `identity` is
    who(request) taken when the turn STARTED (its time is the row's time);
    without it, the identity is taken now."""
    if not enabled():
        return
    try:
        if body is None:
            outcome, detail, body = "error", "exception", {}
        else:
            outcome, detail = classify(status, body,
                                       turn_timeout_text=turn_timeout_text,
                                       gemini_timeout_text=gemini_timeout_text)
        row = dict(identity) if identity else who(request, visitor_id)
        row.update(
            endpoint=endpoint, question=question, answer=_answer_of(body),
            response_type=body.get("type") or body.get("error"),
            route=(stats.route if stats and stats.route else body.get("route")),
            outcome=outcome, detail=detail, http_status=status,
            latency_ms=latency_ms,
            retry_after_seconds=body.get("retry_after_seconds"),
        )
        if stats is not None:
            row.update(gemini_calls=stats.gemini_calls,
                       prompt_tokens=stats.prompt_tokens,
                       output_tokens=stats.output_tokens,
                       thinking_tokens=stats.thinking_tokens,
                       total_tokens=stats.total_tokens)
        recorder().submit("turns", row)
    except Exception as exc:                 # analytics must never break a turn
        print(f"[analytics] record_turn failed: {exc}", file=sys.stderr)


# ── The beacon (page loads and clicks) ───────────────────────────────────────

_beacon_limiter: demo_mode.RateLimiter | None = None


def beacon_limiter() -> demo_mode.RateLimiter:
    global _beacon_limiter
    if _beacon_limiter is None:
        _beacon_limiter = demo_mode.RateLimiter(
            per_minute=_int_env("ANALYTICS_BEACON_PER_MINUTE", 30),
            per_day=_int_env("ANALYTICS_BEACON_PER_DAY", 500))
    return _beacon_limiter


def record_beacon(request, payload) -> str:
    """Queue a visit or event from the page's beacon. Returns what happened
    ("visit", "event", "ignored", "rate_limited") — for tests; the endpoint
    answers 204 regardless."""
    if not enabled() or not isinstance(payload, dict):
        return "ignored"
    kind = payload.get("kind")
    if kind != VISIT_KIND and kind not in EVENT_KINDS:
        return "ignored"
    row = who(request)
    if beacon_limiter().hit(row["ip"] or "unknown") is not None:
        return "rate_limited"
    if kind == VISIT_KIND:
        row.update({k: _cap(payload.get(k), _CAP[k])
                    for k in ("src", "referrer", "path", "screen",
                              "language", "timezone")})
        recorder().submit("visits", row)
        return "visit"
    detail = payload.get("detail")
    if detail is not None and not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    row.update(kind=kind, detail=_cap(detail, _CAP["detail"]))
    recorder().submit("events", row)
    return "event"


# ── Consistent snapshot (scripts/analytics_snapshot.py) ─────────────────────

def snapshot(dest: str, source: str | None = None) -> dict:
    """Copy the live database to `dest` with SQLite's backup API (consistent
    even while the server is writing in WAL mode), check it, and describe it."""
    import hashlib

    source = source or db_path()
    if not os.path.isfile(source):
        raise FileNotFoundError(f"no analytics database at {source}")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(dest):
        os.remove(dest)
    src = sqlite3.connect(f"file:{Path(source).resolve().as_posix()}?mode=ro",
                          uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")   # one self-contained file
        check = dst.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("visits", "turns", "events")}
    finally:
        dst.close()
        src.close()
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": dest, "bytes": os.path.getsize(dest), "sha256": h.hexdigest(),
            "integrity": check, "counts": counts}
