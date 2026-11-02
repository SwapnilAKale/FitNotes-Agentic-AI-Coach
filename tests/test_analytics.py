"""
Private visitor analytics for the public demo (src/analytics.py).

Covers: what a turn records and how each outcome is classified, end to end
through the real ASGI app with the demo's fake agent; the page beacon (visits,
clicks, size cap, rate limit); the views the owner's dashboard reads; that
writes never block a request; Gemini token counting; that nothing is recorded
outside DEMO_MODE; that the demo reset keeps the file; the SSH export path
(snapshot + export script, run against a local shell instead of ssh); the
build-time DB-IP download's fallback; and the page's beacon wiring.
"""

import asyncio
import datetime
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src import analytics                            # noqa: E402
from src import visitor_sessions as vs               # noqa: E402
# The demo fixture: the real server app with a fake agent and coordinator.
from test_visitor_sessions import demo, _browser     # noqa: E402,F401

UA_CHROME_WIN = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
UA_IPHONE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1")
VISITOR_HEADERS = {"user-agent": UA_CHROME_WIN, "x-forwarded-for": "203.0.113.9"}


def _rows(table):
    analytics.recorder().flush()
    path = analytics.db_path()
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id"
                                              if table in ("visits", "turns", "events")
                                              else f"SELECT * FROM {table}")]
    finally:
        conn.close()


# ── switch ───────────────────────────────────────────────────────────────────

def test_on_only_with_demo_mode_unless_overridden(monkeypatch):
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert not analytics.enabled()
    monkeypatch.setenv("DEMO_MODE", "1")
    assert analytics.enabled()
    monkeypatch.setenv("ANALYTICS_ENABLED", "")          # compose's empty default
    assert analytics.enabled()
    monkeypatch.setenv("ANALYTICS_ENABLED", "0")
    assert not analytics.enabled()
    monkeypatch.delenv("DEMO_MODE")
    monkeypatch.setenv("ANALYTICS_ENABLED", "1")
    assert analytics.enabled()


def test_nothing_is_recorded_outside_demo_mode(monkeypatch):
    import server as srv
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)

    async def go():
        async with _browser(srv) as b:
            r = await b.post("/client-event", content=json.dumps({"kind": "visit"}),
                             headers=VISITOR_HEADERS)
            assert r.status_code == 204
    asyncio.run(go())
    analytics.record_turn(None, endpoint="chat", question="private", status=200,
                          body={"type": "answer", "text": "x"}, latency_ms=1)
    analytics.recorder().flush()
    assert not os.path.exists(analytics.db_path())


# ── user-agent and location ─────────────────────────────────────────────────

@pytest.mark.parametrize("ua,device,browser,os_name,bot", [
    (UA_CHROME_WIN, "desktop", "Chrome", "Windows", 0),
    (UA_IPHONE, "mobile", "Safari", "iOS", 0),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.5 Safari/605.1.15", "desktop", "Safari", "macOS", 0),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0", "desktop", "Edge", "Windows", 0),
    ("Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
     "desktop", "Firefox", "Linux", 0),
    ("Mozilla/5.0 (Linux; Android 14; SM-X710) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/130.0.0.0 Safari/537.36", "tablet", "Chrome", "Android", 0),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/130.0.0.0 Mobile Safari/537.36", "mobile", "Chrome", "Android", 0),
    ("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
     "bot", "Other", "Other", 1),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "HeadlessChrome/130.0.0.0 Safari/537.36", "bot", "Chrome", "Linux", 1),
    ("", "bot", "Other", "Other", 1),
])
def test_user_agent_parsing(ua, device, browser, os_name, bot):
    assert analytics.parse_user_agent(ua) == {
        "device": device, "browser": browser, "os": os_name, "is_bot": bot}


class FakeReader:
    def __init__(self, table):
        self.table = table

    def get(self, ip):
        return self.table.get(ip)

    def close(self):
        pass


RECORD = {"city": {"names": {"en": "Pune"}},
          "country": {"iso_code": "IN", "names": {"en": "India"}},
          "subdivisions": [{"names": {"en": "Maharashtra"}}],
          "location": {"latitude": 18.52, "longitude": 73.86}}


def test_geo_lookup_fields_and_fallbacks(tmp_path):
    geo = analytics.GeoLookup(reader=FakeReader({"203.0.113.9": RECORD}))
    assert geo.lookup("203.0.113.9") == {
        "country": "India", "country_code": "IN", "region": "Maharashtra",
        "city": "Pune", "latitude": 18.52, "longitude": 73.86}
    assert geo.lookup("10.0.0.1")["city"] is None          # not in the database
    assert geo.lookup(None)["country"] is None
    missing = analytics.GeoLookup(path=str(tmp_path / "absent.mmdb"))
    assert missing.lookup("203.0.113.9")["country"] is None
    broken = tmp_path / "broken.mmdb"
    broken.write_bytes(b"not an mmdb")
    assert analytics.GeoLookup(path=str(broken)).lookup("203.0.113.9")["city"] is None


# ── the writer ───────────────────────────────────────────────────────────────

class SlowGeo(analytics.GeoLookup):
    def __init__(self, gate):
        super().__init__(reader=FakeReader({}))
        self.gate = gate

    def lookup(self, ip):
        self.gate.wait(5)
        return super().lookup(ip)


def _who(**extra):
    row = {"ts": "2026-11-02T10:00:00Z", "day": "2026-11-02", "visitor_id": "v1",
           "ip": "203.0.113.9", "user_agent": UA_CHROME_WIN}
    row.update(extra)
    return row


def test_recording_never_blocks_and_drops_when_full(tmp_path, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(analytics, "_QUEUE_MAX", 5)
    rec = analytics.Recorder(str(tmp_path / "a.sqlite"), geo=SlowGeo(gate))
    try:
        start = time.monotonic()
        results = [rec.submit("visits", _who()) for _ in range(20)]
        assert time.monotonic() - start < 0.5          # the writer is stuck; we are not
        assert results.count(False) >= 10 and rec.dropped == results.count(False)
        gate.set()
        rec.flush()
        conn = sqlite3.connect(rec.path)
        assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == results.count(True)
        conn.close()
    finally:
        gate.set()
        rec.close()


def test_views_daily_visits_visitor_timeline_bounces_outcomes(tmp_path):
    rec = analytics.Recorder(str(tmp_path / "a.sqlite"),
                             geo=analytics.GeoLookup(reader=FakeReader({"203.0.113.9": RECORD})))
    try:
        # Visitor A: two loads (first from ?src=resume), asks twice — one
        # slow answer and one refusal. Visitor B: one load, never asks (bounce).
        # A crawler that ran JS: flagged as a bot, not counted as a visitor.
        rec.submit("visits", _who(ts="2026-11-01T09:00:00Z", day="2026-11-01",
                                  src="resume", referrer="https://www.linkedin.com/"))
        rec.submit("visits", _who(ts="2026-11-02T09:00:00Z"))
        rec.submit("turns", _who(ts="2026-11-02T09:01:00Z", endpoint="chat",
                                 question="q1", answer="a1", outcome="ok",
                                 route="analytical", latency_ms=75_000, total_tokens=1200))
        rec.submit("turns", _who(ts="2026-11-02T09:05:00Z", endpoint="chat",
                                 question="q2", outcome="demo_full", latency_ms=3))
        rec.submit("visits", _who(visitor_id="v2", ip="198.51.100.4",
                                  user_agent=UA_IPHONE, ts="2026-11-02T11:00:00Z"))
        rec.submit("visits", _who(visitor_id="crawler", ip="192.0.2.1",
                                  user_agent="HeadlessChrome/130"))
        rec.flush()
        conn = sqlite3.connect(rec.path)
        conn.row_factory = sqlite3.Row
        daily = {r["day"]: dict(r) for r in conn.execute("SELECT * FROM daily_visits")}
        assert daily["2026-11-01"] == {"day": "2026-11-01", "page_loads": 1,
                                       "unique_visitors": 1, "bot_page_loads": 0}
        assert daily["2026-11-02"] == {"day": "2026-11-02", "page_loads": 2,
                                       "unique_visitors": 2, "bot_page_loads": 1}
        people = {r["visitor_id"]: dict(r) for r in conn.execute("SELECT * FROM visitors")}
        a = people["v1"]
        assert (a["first_seen"], a["last_seen"]) == ("2026-11-01T09:00:00Z", "2026-11-02T09:05:00Z")
        assert (a["visits"], a["questions"], a["answered"], a["bounced"]) == (2, 2, 1, 0)
        assert a["first_src"] == "resume" and a["first_referrer"] == "https://www.linkedin.com/"
        assert (a["last_ip"], a["city"], a["country"], a["device"]) == (
            "203.0.113.9", "Pune", "India", "desktop")
        b = people["v2"]
        assert (b["visits"], b["questions"], b["bounced"], b["device"]) == (1, 0, 1, "mobile")
        assert people["crawler"]["is_bot"] == 1
        out = dict(conn.execute("SELECT * FROM daily_outcomes WHERE day = '2026-11-02'").fetchone())
        assert (out["turns"], out["ok"], out["demo_full"], out["slow_answers"],
                out["total_tokens"]) == (2, 1, 1, 1, 1200)
        # Location and parsed user-agent are stored alongside the raw values.
        v = dict(conn.execute("SELECT * FROM visits ORDER BY id LIMIT 1").fetchone())
        assert (v["ip"], v["user_agent"], v["city"], v["region"], v["country_code"],
                v["browser"], v["os"]) == ("203.0.113.9", UA_CHROME_WIN, "Pune",
                                           "Maharashtra", "IN", "Chrome", "Windows")
        conn.close()
    finally:
        rec.close()


# ── outcome classification ──────────────────────────────────────────────────

@pytest.mark.parametrize("status,body,expected", [
    (200, {"type": "answer", "text": "fine"}, "ok"),
    (200, {"type": "confirmation_required", "preview": "x"}, "ok"),
    (200, {"type": "error", "text": "An error occurred: boom"}, "error"),
    (200, {"type": "error", "text": vs.TIMED_OUT_TEXT}, "turn_timeout"),
    (200, {"type": "error", "text": "GEMINI-TIMEOUT"}, "gemini_timeout"),
    (429, {"error": "demo_full", "type": "error", "text": vs.FULL_TEXT}, "demo_full"),
    (429, {"error": "demo_rate_limit", "type": "error", "text": "x"}, "demo_rate_limit"),
    (429, {"error": "rate_limit", "retry_after_seconds": 30}, "gemini_429"),
    (429, {"error": "Agent is busy, please wait"}, "busy"),
    (503, {"type": "not_ready", "text": "Still initializing."}, "not_ready"),
    (200, {"error": "overload", "text": "Gemini is under high demand"}, "error"),
])
def test_classify(status, body, expected):
    outcome, _ = analytics.classify(status, body, turn_timeout_text=vs.TIMED_OUT_TEXT,
                                    gemini_timeout_text="GEMINI-TIMEOUT")
    assert outcome == expected and outcome in analytics.OUTCOMES


# ── end to end through the demo server ──────────────────────────────────────

def test_a_turn_records_question_answer_route_identity_and_latency(demo):
    async def go():
        async with _browser(demo.srv) as a:
            await a.get("/status")
            r = await a.post("/chat", json={"message": "how is my bench?"},
                             headers=VISITOR_HEADERS)
            assert r.json()["text"].startswith("echo:how is my bench?")
            return a.cookies.get(vs.COOKIE_NAME)
    sid = asyncio.run(go())
    [t] = _rows("turns")
    assert (t["endpoint"], t["question"], t["outcome"], t["route"], t["http_status"]) == (
        "chat", "how is my bench?", "ok", "analytical", 200)
    assert t["answer"].startswith("echo:how is my bench?")
    assert t["visitor_id"] == sid
    # The real client, not the proxy that connected (127.0.0.1 is trusted).
    assert t["ip"] == "203.0.113.9" and t["user_agent"] == UA_CHROME_WIN
    assert t["latency_ms"] is not None and t["latency_ms"] >= 0
    assert t["day"] == datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def test_turn_timeout_and_demo_full_are_recorded(demo, monkeypatch):
    monkeypatch.setenv("DEMO_TURN_TIMEOUT_SECONDS", "0.3")
    original = demo.coord.route

    async def route(message):
        if message == "stuck":
            await asyncio.Event().wait()
        return await original(message)
    demo.coord.route = route

    async def go():
        async with _browser(demo.srv) as a, _browser(demo.srv) as b:
            await b.get("/status")
            ta = asyncio.create_task(a.post("/chat", json={"message": "stuck"}))
            while demo.srv._turn_queue.running is None:
                await asyncio.sleep(0.01)
            r = await b.post("/chat", json={"message": "me too"})
            assert r.json()["error"] == "demo_full"
            assert (await ta).json()["text"] == vs.TIMED_OUT_TEXT
    asyncio.run(go())
    outcomes = {t["question"]: t["outcome"] for t in _rows("turns")}
    assert outcomes == {"stuck": "turn_timeout", "me too": "demo_full"}


def test_a_turn_row_is_timed_when_it_was_asked(demo, monkeypatch):
    stamps = iter([datetime.datetime(2026, 11, 2, 9, 0, 0, tzinfo=datetime.timezone.utc),
                   datetime.datetime(2026, 11, 2, 9, 4, 0, tzinfo=datetime.timezone.utc)])
    monkeypatch.setattr(analytics, "_now", lambda: next(stamps))

    async def go():
        async with _browser(demo.srv) as a:
            await a.post("/chat", json={"message": "when?"})
    asyncio.run(go())
    [t] = _rows("turns")
    assert t["ts"] == "2026-11-02T09:00:00Z"            # the start, not the end


def test_busy_and_not_ready_are_their_own_outcomes(demo):
    async def go():
        gate = asyncio.Event()
        demo.coord.gates["slow"] = gate
        async with _browser(demo.srv) as a:
            await a.get("/status")
            ta = asyncio.create_task(a.post("/chat", json={"message": "slow"}))
            while demo.srv._turn_queue.running is None:
                await asyncio.sleep(0.01)
            r = await a.post("/chat", json={"message": "again"})
            assert r.status_code == 429 and "busy" in r.json()["error"].lower()
            gate.set()
            await ta
            demo.srv.agent_ready = False
            r = await a.post("/chat", json={"message": "too early"})
            assert r.json()["type"] == "not_ready"
    asyncio.run(go())
    outcomes = {t["question"]: t["outcome"] for t in _rows("turns")}
    assert outcomes == {"slow": "ok", "again": "busy", "too early": "not_ready"}


def test_gemini_429_timeout_and_errors_are_recorded(demo):
    original = demo.coord.route

    async def route(message):
        if message == "quota":
            raise RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '30s'}")
        if message == "slow gemini":
            raise httpx.ReadTimeout("timed out")
        if message == "boom":
            raise RuntimeError("boom")
        return await original(message)
    demo.coord.route = route

    async def go():
        async with _browser(demo.srv) as a:
            for q in ("quota", "slow gemini", "boom"):
                await a.post("/chat", json={"message": q})
    asyncio.run(go())
    rows = {t["question"]: t for t in _rows("turns")}
    assert rows["quota"]["outcome"] == "gemini_429"
    assert rows["quota"]["retry_after_seconds"] == 30 and rows["quota"]["http_status"] == 429
    assert rows["slow gemini"]["outcome"] == "gemini_timeout"
    assert rows["boom"]["outcome"] == "error" and "boom" in rows["boom"]["detail"]


def test_demo_rate_limit_refusal_is_recorded_with_its_question(demo, monkeypatch):
    monkeypatch.setenv("DEMO_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setattr(demo.srv.demo_mode, "_limiter", None)

    async def go():
        async with _browser(demo.srv) as a:
            await a.post("/chat", json={"message": "first"}, headers=VISITOR_HEADERS)
            r = await a.post("/chat", json={"message": "second"}, headers=VISITOR_HEADERS)
            assert r.status_code == 429 and r.json()["error"] == "demo_rate_limit"
            return a.cookies.get(vs.COOKIE_NAME)
    sid = asyncio.run(go())
    rows = {t["question"]: t for t in _rows("turns")}
    assert rows["first"]["outcome"] == "ok"
    second = rows["second"]
    assert (second["outcome"], second["endpoint"], second["visitor_id"], second["ip"]) == (
        "demo_rate_limit", "chat", sid, "203.0.113.9")


def test_confirm_cancel_and_disambiguate_are_recorded(demo):
    async def go():
        async with _browser(demo.srv) as a:
            r = await a.post("/chat", json={"message": "log bench 100x5"})
            assert r.json()["type"] == "confirmation_required"
            await a.post("/confirm", json={"confirmed": False})
            r = await a.post("/chat", json={"message": "ambiguous"})
            assert r.json()["type"] == "disambiguation_required"
            await a.post("/disambiguate", json={"selections": [
                {"name": "press", "choice": "Bench Press"}]})
    asyncio.run(go())
    rows = _rows("turns")
    # (The cancel's reply comes from the agent, which the fake doesn't model.)
    assert [(t["endpoint"], t["response_type"]) for t in rows if t["endpoint"] != "cancel"][:2] == [
        ("chat", "confirmation_required"), ("chat", "disambiguation_required")]
    assert rows[1]["endpoint"] == "cancel"
    assert rows[0]["answer"] == "Bench Press 100 x 5"          # the panel preview
    assert rows[0]["route"] == "operational"
    assert rows[3]["endpoint"] == "disambiguate"
    assert rows[3]["question"] == "press → Bench Press"


def test_gemini_tokens_per_turn_are_summed(demo, monkeypatch):
    from google.genai import models

    def fake_generate_content(self, *, model, contents, config=None):
        return SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20,
            thoughts_token_count=5, total_token_count=125))
    monkeypatch.setattr(models.Models, "generate_content", fake_generate_content)
    analytics.install_usage_counter()
    analytics.install_usage_counter()                         # idempotent
    original = demo.coord.route

    async def route(message):
        for _ in range(2):           # the way agent.py / coordinator.py call it
            await asyncio.to_thread(models.Models.generate_content, object(),
                                    model="m", contents="c")
        return await original(message)
    demo.coord.route = route

    async def go():
        async with _browser(demo.srv) as a:
            await a.post("/chat", json={"message": "count me"})
    asyncio.run(go())
    # Outside a turn, nothing is counted and nothing breaks.
    models.Models.generate_content(object(), model="m", contents="c")
    [t] = _rows("turns")
    assert (t["gemini_calls"], t["prompt_tokens"], t["output_tokens"],
            t["thinking_tokens"], t["total_tokens"]) == (2, 200, 40, 10, 250)


# ── the beacon ───────────────────────────────────────────────────────────────

def test_beacon_records_visits_and_clicks(demo):
    async def go():
        async with _browser(demo.srv) as a:
            await a.get("/status")
            visit = {"kind": "visit", "src": "acme", "referrer": "https://mail.example/",
                     "path": "/?src=acme", "screen": "1920x1080",
                     "language": "en-GB", "timezone": "Europe/London"}
            for payload in (visit, {"kind": "graph_click"}, {"kind": "settings_click"},
                            {"kind": "not_a_kind"}, ["not", "a", "dict"]):
                r = await a.post("/client-event", content=json.dumps(payload),
                                 headers={**VISITOR_HEADERS, "content-type": "text/plain"})
                assert r.status_code == 204
            r = await a.post("/client-event", content=b"{not json",
                             headers=VISITOR_HEADERS)
            assert r.status_code == 204
            return a.cookies.get(vs.COOKIE_NAME)
    sid = asyncio.run(go())
    [v] = _rows("visits")
    assert (v["visitor_id"], v["ip"], v["src"], v["referrer"], v["screen"],
            v["language"], v["timezone"], v["device"]) == (
        sid, "203.0.113.9", "acme", "https://mail.example/", "1920x1080",
        "en-GB", "Europe/London", "desktop")
    assert [e["kind"] for e in _rows("events")] == ["graph_click", "settings_click"]


def test_beacon_is_size_capped_and_rate_limited(demo, monkeypatch):
    monkeypatch.setenv("ANALYTICS_BEACON_PER_MINUTE", "3")
    monkeypatch.setattr(analytics, "_beacon_limiter", None)

    async def go():
        async with _browser(demo.srv) as a:
            big = json.dumps({"kind": "visit", "src": "x" * 5000})
            r = await a.post("/client-event", content=big, headers=VISITOR_HEADERS)
            assert r.status_code == 413
            for _ in range(6):
                r = await a.post("/client-event", content=json.dumps({"kind": "graph_click"}),
                                 headers=VISITOR_HEADERS)
                assert r.status_code == 204
    asyncio.run(go())
    assert len(_rows("events")) == 3 and _rows("visits") == []
    # Over-long fields that fit the size cap are truncated, not stored whole.
    row = {}
    req = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"),
                          headers={"x-forwarded-for": "198.51.100.7"}, cookies={},
                          state=SimpleNamespace())
    monkeypatch.setattr(analytics, "_beacon_limiter", None)
    assert analytics.record_beacon(req, {"kind": "visit", "src": "y" * 300}) == "visit"
    row = _rows("visits")[-1]
    assert len(row["src"]) == 100 and row["ip"] == "198.51.100.7"


# ── never wiped ──────────────────────────────────────────────────────────────

def _prepare(monkeypatch, tmp_path):
    import importlib.util
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("FITNOTES_DB_PATH", str(data / "FitNotes_Backup.fitnotes"))
    monkeypatch.setenv("CHROMA_DB_PATH", str(data / "chroma_db"))
    monkeypatch.setenv("SEED_CHROMA_DIR", str(tmp_path / "no-seed"))
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_RESET_ON_START", "1")
    monkeypatch.delenv("DEMO_SESSIONS_PATH", raising=False)
    spec = importlib.util.spec_from_file_location(
        "prepare_data_an", os.path.join(_ROOT, "deploy", "prepare_data.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return data, prep


def test_demo_reset_keeps_the_analytics_file(monkeypatch, tmp_path):
    data, prep = _prepare(monkeypatch, tmp_path)
    monkeypatch.delenv("ANALYTICS_DB_PATH", raising=False)
    (data / "analytics").mkdir()
    for name in ("analytics.sqlite", "analytics.sqlite-wal"):
        (data / "analytics" / name).write_text("keep me")
    (data / "memory.json").write_text('{"facts": ["stale"]}')
    assert prep.main() == 0
    assert (data / "analytics" / "analytics.sqlite").read_text() == "keep me"
    assert (data / "analytics" / "analytics.sqlite-wal").read_text() == "keep me"
    assert "stale" not in (data / "memory.json").read_text()


def test_demo_reset_keeps_a_custom_analytics_path(monkeypatch, tmp_path):
    data, prep = _prepare(monkeypatch, tmp_path)
    monkeypatch.setenv("ANALYTICS_DB_PATH", str(data / "visits.sqlite"))
    for name in ("visits.sqlite", "visits.sqlite-wal", "visits.sqlite-shm"):
        (data / name).write_text("keep me")
    assert {"analytics", "visits.sqlite", "visits.sqlite-wal",
            "visits.sqlite-shm"} <= prep._kept_names()
    assert prep.main() == 0
    for name in ("visits.sqlite", "visits.sqlite-wal", "visits.sqlite-shm"):
        assert (data / name).read_text() == "keep me"


# ── snapshot and export ──────────────────────────────────────────────────────

def _live_db(tmp_path):
    """An analytics DB with rows still in its WAL (a connection holds it open)."""
    path = str(tmp_path / "live" / "analytics.sqlite")
    conn = analytics.open_db(path)
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO visits (ts, day, visitor_id, ip) VALUES "
                 "('2026-11-02T10:00:00Z', '2026-11-02', 'v1', '203.0.113.9')")
    conn.execute("INSERT INTO turns (ts, day, endpoint, outcome, question) VALUES "
                 "('2026-11-02T10:01:00Z', '2026-11-02', 'chat', 'ok', 'hello')")
    return path, conn


def test_snapshot_is_consistent_while_the_database_is_live(tmp_path):
    path, conn = _live_db(tmp_path)
    try:
        assert os.path.getsize(path + "-wal") > 0          # rows are only in the WAL
        dest = tmp_path / "snap" / "s.sqlite"
        info = analytics.snapshot(str(dest), path)
        assert info["integrity"] == "ok"
        assert info["counts"] == {"visits": 1, "turns": 1, "events": 0}
        assert not os.path.exists(str(dest) + "-wal")      # one self-contained file
        copy = sqlite3.connect(dest)
        assert copy.execute("SELECT question FROM turns").fetchone()[0] == "hello"
        assert copy.execute("SELECT COUNT(*) FROM visitors").fetchone()[0] == 1
        copy.close()
    finally:
        conn.close()


def test_snapshot_script_exit_codes(tmp_path, capsys):
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import analytics_snapshot
    assert analytics_snapshot.main([str(tmp_path / "s.sqlite"),
                                    "--source", str(tmp_path / "none.sqlite")]) == 2
    assert "no analytics database" in json.loads(capsys.readouterr().out)["error"]
    path, conn = _live_db(tmp_path)
    conn.close()
    assert analytics_snapshot.main([str(tmp_path / "s.sqlite"), "--source", path]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["turns"] == 1


def _export_module():
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import export_analytics
    return export_analytics


def test_export_commands_are_argument_lists():
    ex = _export_module()
    remote = ex.remote_command("~/FitNotes-Agentic-AI-Coach", "app", "/tmp/s.sqlite")
    assert remote.startswith("cd ~/FitNotes-Agentic-AI-Coach && docker compose exec -T app sh -c ")
    assert remote.endswith(" _ /tmp/s.sqlite")
    assert ex.remote_command("/srv/my repo", "app", "/tmp/s").startswith("cd '/srv/my repo' && ")
    cmd = ex.ssh_command(r"C:\Users\ASUS\.ssh\oracle_fitnotes", "ubuntu@130.210.14.28", remote)
    assert cmd[:3] == ["ssh", "-i", r"C:\Users\ASUS\.ssh\oracle_fitnotes"]
    assert cmd[-2:] == ["ubuntu@130.210.14.28", remote]
    assert '"' not in remote                     # nothing for PowerShell 5.1 to strip


def _local_ssh(db_path, seen, tamper=False):
    """Stands in for ssh + docker compose exec: runs the real container script
    with sh, in this repo, against `db_path`."""
    ex = _export_module()

    def run(cmd, stdout, stderr):
        seen.append(cmd)
        tmp = cmd[-1].rsplit(" _ ", 1)[1]
        proc = subprocess.run(["sh", "-c", ex.CONTAINER_SCRIPT, "_", tmp], cwd=_ROOT,
                              stdout=stdout, stderr=stderr,
                              env={**os.environ, "ANALYTICS_DB_PATH": str(db_path)})
        if tamper:
            stdout.write(b"x")
        seen.append(os.path.exists(tmp))
        return proc
    return run


def test_export_end_to_end_with_a_local_shell(tmp_path):
    ex = _export_module()
    path, conn = _live_db(tmp_path)
    try:
        seen = []
        out = tmp_path / "analytics_export"
        saved = ex.export(key="k", host="ubuntu@h", out=str(out),
                          run=_local_ssh(path, seen),
                          now=datetime.datetime(2026, 11, 2, 12, 30, 0))
        assert saved == out / "analytics-20261102-123000.sqlite"
        assert seen[1] is False                 # the container's temp copy is gone
        for f in (saved, out / "latest.sqlite"):
            c = sqlite3.connect(f)
            assert c.execute("SELECT question FROM turns").fetchone()[0] == "hello"
            c.close()
        assert not list(out.glob("*.part"))
    finally:
        conn.close()


def test_export_rejects_a_corrupted_copy_and_a_missing_database(tmp_path):
    ex = _export_module()
    path, conn = _live_db(tmp_path)
    conn.close()
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ex.export(key="k", host="h", out=str(out), run=_local_ssh(path, [], tamper=True))
    with pytest.raises(RuntimeError, match="nothing recorded yet"):
        ex.export(key="k", host="h", out=str(out),
                  run=_local_ssh(tmp_path / "absent.sqlite", []))
    assert list(out.iterdir()) == []            # no partial or bad file left behind


# ── DB-IP download (build time) ─────────────────────────────────────────────

def test_geoip_download_never_fails_the_build(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_geoip", os.path.join(_ROOT, "deploy", "fetch_geoip.py"))
    fg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fg)
    assert fg.candidate_months(None, datetime.date(2026, 1, 15)) == ["2026-01", "2025-12"]
    assert fg.candidate_months("2025-06", datetime.date(2026, 1, 15)) == [
        "2025-06", "2026-01", "2025-12"]
    tried = []

    def offline(url, timeout):
        tried.append(url)
        raise OSError("network unreachable")
    dest = tmp_path / "geo" / "dbip-city-lite.mmdb"
    assert fg.fetch(str(dest), opener=offline) is False
    assert not dest.exists() and len(tried) == 2
    assert tried[0].startswith("https://download.db-ip.com/free/dbip-city-lite-")
    assert list((tmp_path / "geo").iterdir()) == []      # no temp files left


def test_dockerfile_downloads_geoip_before_copying_the_code():
    text = open(os.path.join(_ROOT, "Dockerfile"), encoding="utf-8").read()
    fetch = text.index("RUN python deploy/fetch_geoip.py")
    assert fetch < text.index("COPY . .")
    assert "maxminddb" in open(os.path.join(_ROOT, "requirements.txt")).read()


# ── the page ─────────────────────────────────────────────────────────────────

def test_page_sends_a_visit_and_click_events_only_in_the_demo():
    html = open(os.path.join(_ROOT, "frontend", "index.html"), encoding="utf-8").read()
    assert "/client-event" in html
    assert "if (!demoMode) return;" in html
    status = html.index("if (data.demo_mode) {")
    assert html.index("sendVisit();", status) < html.index("if (data.ready) break;", status)
    for kind in ("'visit'", "graph_click", "settings_click"):
        assert kind in html
    assert "new URLSearchParams(location.search).get('src')" in html
    assert "document.referrer" in html
    # Every kind the page sends is one the server accepts.
    assert analytics.EVENT_KINDS == {"graph_click", "settings_click"}
