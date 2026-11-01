"""
Public-demo plumbing (src/demo_mode.py, deploy/prepare_data.py,
scripts/make_demo_db.py) and the OPEN_BROWSER switch it relies on.

The invariant that matters most is the first test: with DEMO_MODE unset,
nothing is blocked or limited, so `python server.py` behaves exactly as before.
"""

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import demo_mode                       # noqa: E402
from src.demo_mode import RateLimiter, client_ip  # noqa: E402


@pytest.fixture
def demo_on(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr(demo_mode, "_limiter",
                        RateLimiter(per_minute=2, per_day=100))
    yield


@pytest.fixture
def demo_off(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    yield


# ── the gate itself ──────────────────────────────────────────────────────────

def test_everything_passes_when_demo_mode_is_off(demo_off):
    for (method, path) in list(demo_mode.BLOCKED) + list(demo_mode.RATE_LIMITED):
        for _ in range(50):
            assert demo_mode.check(method, path, "203.0.113.5", None) is None


def test_overwrite_and_wipe_endpoints_are_403_in_demo_mode(demo_on):
    for (method, path), what in demo_mode.BLOCKED.items():
        status, body, _ = demo_mode.check(method, path, "203.0.113.5", None)
        assert status == 403
        assert body["error"] == "demo_disabled"
        assert what in body["text"] and body["errors"] == [body["text"]]
    # trailing slash and lower-case method are the same endpoint
    assert demo_mode.check("post", "/wal-wipe/", None, None)[0] == 403
    # reads stay open
    assert demo_mode.check("GET", "/settings", None, None) is None
    assert demo_mode.check("GET", "/status", None, None) is None


def test_turn_endpoints_are_rate_limited_per_ip(demo_on):
    a, b = "203.0.113.5", "198.51.100.7"
    assert demo_mode.check("POST", "/chat", a, None) is None
    assert demo_mode.check("POST", "/resume", a, None) is None
    status, body, headers = demo_mode.check("POST", "/disambiguate", a, None)
    assert status == 429 and body["error"] == "demo_rate_limit"
    assert int(headers["Retry-After"]) >= 1
    assert demo_mode.check("POST", "/chat", b, None) is None     # other visitor


def test_rate_limiter_windows_slide():
    now = [1000.0]
    rl = RateLimiter(per_minute=2, per_day=3, clock=lambda: now[0])
    assert rl.hit("x") is None and rl.hit("x") is None
    assert rl.hit("x") is not None                     # 3rd within a minute
    now[0] += 61
    assert rl.hit("x") is None                         # minute window slid
    now[0] += 61
    wait = rl.hit("x")                                 # 4th in a day
    assert wait is not None and wait > 3600
    now[0] += 86400
    assert rl.hit("x") is None


def test_global_daily_cap():
    rl = RateLimiter(per_minute=0, per_day=0, global_per_day=2, clock=lambda: 5.0)
    assert rl.hit("a") is None and rl.hit("b") is None
    assert rl.hit("c") is not None


# ── client IP behind Caddy → frontend proxy → backend ───────────────────────

def test_client_ip_through_caddy_and_frontend_proxy(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    # Caddy sets XFF to the visitor; the frontend appends Caddy's Docker IP;
    # the backend's peer is the frontend on loopback.
    assert client_ip("127.0.0.1", "203.0.113.9, 172.18.0.3") == "203.0.113.9"


def test_client_ip_cannot_be_spoofed_with_own_header(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    # The visitor sent "X-Forwarded-For: 1.2.3.4"; the proxy appended the
    # real address. The right-most untrusted hop wins.
    assert client_ip("127.0.0.1", "1.2.3.4, 198.51.100.7, 172.18.0.3") == "198.51.100.7"


def test_client_ip_fallbacks(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    assert client_ip("198.51.100.7", None) == "198.51.100.7"
    assert client_ip("127.0.0.1", "192.168.1.20, 172.18.0.3") == "192.168.1.20"   # LAN
    assert client_ip("127.0.0.1", "[2001:db8::1]:4711") == "2001:db8::1"
    monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "127.0.0.1/32")
    assert client_ip("127.0.0.1", "10.0.0.5") == "10.0.0.5"


# ── switches ─────────────────────────────────────────────────────────────────

def test_browser_switch_defaults_on(monkeypatch):
    for var in ("OPEN_BROWSER", "DEMO_MODE"):
        monkeypatch.delenv(var, raising=False)
    assert demo_mode.open_browser_allowed()
    monkeypatch.setenv("OPEN_BROWSER", "0")
    assert not demo_mode.open_browser_allowed()
    monkeypatch.setenv("OPEN_BROWSER", "1")
    monkeypatch.setenv("DEMO_MODE", "1")
    assert not demo_mode.open_browser_allowed()        # never in the demo


# ── wired into the API ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api():
    import server as srv
    return srv


def test_api_blocks_wipe_in_demo_mode(api, demo_on):
    resp = TestClient(api.app).post("/wal-wipe")
    assert resp.status_code == 403
    assert resp.json()["error"] == "demo_disabled"
    assert TestClient(api.app).get("/status").json()["demo_mode"] is True


def test_api_wipe_still_works_without_demo_mode(api, demo_off):
    resp = TestClient(api.app).post("/wal-wipe")
    assert resp.status_code == 200 and "wiped" in resp.json()
    assert TestClient(api.app).get("/status").json()["demo_mode"] is False


def test_api_rate_limits_chat_by_forwarded_ip(api, demo_on):
    c = TestClient(api.app)
    hdr = {"X-Forwarded-For": "203.0.113.50"}
    for _ in range(2):                                 # agent not ready → 503
        assert c.post("/chat", json={"message": "hi"}, headers=hdr).status_code == 503
    r = c.post("/chat", json={"message": "hi"}, headers=hdr)
    assert r.status_code == 429 and r.json()["error"] == "demo_rate_limit"
    other = c.post("/chat", json={"message": "hi"},
                   headers={"X-Forwarded-For": "203.0.113.51"})
    assert other.status_code == 503


def _frontend():
    name = "fitnotes_frontend_server"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / "frontend" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_frontend_ontology_review_is_read_only_in_demo(demo_on, monkeypatch, tmp_path):
    fe = _frontend()
    monkeypatch.setattr(fe, "REVIEW_CUTS", tmp_path / "review_cuts.csv")
    r = TestClient(fe.app).post("/ontology-review", json={"decisions": []})
    assert r.status_code == 403 and r.json()["error"] == "demo_disabled"
    assert not (tmp_path / "review_cuts.csv").exists()


# ── synthetic demo database ─────────────────────────────────────────────────

def _load_script(rel, name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    import datetime as dt
    mk = _load_script("scripts/make_demo_db.py", "make_demo_db")
    path = str(tmp_path_factory.mktemp("demo") / "demo.fitnotes")
    info = mk.generate(path, weeks=12, end_date=dt.date(2026, 10, 30), seed=42)
    return path, info, mk


def test_demo_db_has_the_columns_the_code_reads(demo_db):
    path, info, _ = demo_db
    conn = sqlite3.connect(path)
    cols = lambda t: {r[1] for r in conn.execute(f"PRAGMA table_info([{t}])")}
    assert {"_id", "exercise_id", "date", "metric_weight", "reps", "unit",
            "is_personal_record", "is_complete", "distance",
            "duration_seconds"} <= cols("training_log")
    assert {"_id", "name", "category_id"} <= cols("exercise")
    assert {"_id", "name"} <= cols("Category")
    assert {"_id", "date", "owner_type_id", "owner_id", "comment"} <= cols("Comment")
    assert {"_id", "type_id", "exercise_id", "metric_weight", "reps", "unit", "title",
            "target_date", "sort_order", "distance", "duration_seconds",
            "start_date"} <= cols("Goal")
    assert {"_id", "date", "body_weight_metric", "body_fat"} <= cols("BodyWeight")
    n, first, last = conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM training_log").fetchone()
    assert n > 500 and first >= "2026-08-01" and last <= "2026-10-30"
    assert info["sessions"] >= 45
    # every exercise is one the ontology knows, so the muscle graph maps it
    import csv
    with open(_ROOT / "ontology" / "aliases.csv", encoding="utf-8-sig") as fh:
        known = {r["db_exercise_name"] for r in csv.DictReader(fh)}
    names = {r[0] for r in conn.execute("SELECT name FROM exercise")}
    assert names <= known, names - known
    conn.close()


def test_demo_db_pr_flags_follow_the_write_path_rule(demo_db):
    path, _, _ = demo_db
    conn = sqlite3.connect(path)
    best = {}
    for ex, w, pr in conn.execute(
            "SELECT exercise_id, metric_weight, is_personal_record FROM training_log"
            " WHERE unit = 0 ORDER BY _id"):
        expected = 1 if w > best.get(ex, 0) + 1e-9 else 0
        best[ex] = max(best.get(ex, 0), w)
        assert pr == expected
    conn.close()


def test_demo_db_is_deterministic_and_accepts_the_write_path(demo_db, tmp_path):
    import datetime as dt
    path, _, mk = demo_db
    again = str(tmp_path / "again.fitnotes")
    mk.generate(again, weeks=12, end_date=dt.date(2026, 10, 30), seed=42)
    q = "SELECT exercise_id, date, metric_weight, reps FROM training_log ORDER BY _id"
    a, b = sqlite3.connect(path), sqlite3.connect(again)
    assert a.execute(q).fetchall() == b.execute(q).fetchall()
    a.close()
    from src.db import insert_set_comment, insert_training_log_set
    b.row_factory = sqlite3.Row
    rid = insert_training_log_set(b, 1, "2026-10-31",
                                  {"metric_weight": 70.0, "reps": 5, "unit": 0})
    insert_set_comment(b, rid, "2026-10-31", "logged by the agent")
    b.commit()
    assert b.execute("SELECT reps FROM training_log WHERE _id=?", (rid,)).fetchone()[0] == 5
    b.close()


# ── container start-up seeding ──────────────────────────────────────────────

def _prepare(monkeypatch, data_dir, **env):
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("FITNOTES_DB_PATH", str(data_dir / "FitNotes_Backup.fitnotes"))
    monkeypatch.setenv("CHROMA_DB_PATH", str(data_dir / "chroma_db"))
    monkeypatch.setenv("SEED_CHROMA_DIR", str(data_dir.parent / "no-seed"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return _load_script("deploy/prepare_data.py", "prepare_data")


def test_prepare_seeds_demo_once_and_resets_on_request(monkeypatch, tmp_path):
    data = tmp_path / "data"
    prep = _prepare(monkeypatch, data, DEMO_MODE="1", DEMO_RESET_ON_START="0")
    assert prep.main() == 0
    db = data / "FitNotes_Backup.fitnotes"
    assert db.exists()
    ctx = json.loads((data / "user_context.json").read_text())
    assert ctx["unit_overrides"]["exercises_in_kg"] == []
    assert json.loads((data / "agent_writes.json").read_text()) == []
    marks = json.loads((data / "settings.json").read_text())["app_data_watermark"]
    conn = sqlite3.connect(db)
    assert marks["training_log"] == conn.execute("SELECT MAX(_id) FROM training_log").fetchone()[0]
    conn.close()

    (data / "memory.json").write_text('{"facts": ["visitor wrote this"]}')
    mtime = db.stat().st_mtime_ns
    assert _prepare(monkeypatch, data, DEMO_MODE="1").main() == 0
    assert db.stat().st_mtime_ns == mtime                       # kept
    assert "visitor" in (data / "memory.json").read_text()

    (data / "stray.json").write_text("{}")
    assert _prepare(monkeypatch, data, DEMO_MODE="1", DEMO_RESET_ON_START="1").main() == 0
    assert not (data / "stray.json").exists()                   # wiped
    assert "visitor" not in (data / "memory.json").read_text()  # fresh state


def test_prepare_never_seeds_without_demo_mode(monkeypatch, tmp_path):
    data = tmp_path / "data"
    monkeypatch.delenv("DEMO_MODE", raising=False)
    prep = _prepare(monkeypatch, data, DEMO_RESET_ON_START="1")
    assert prep.main() == 0
    assert not (data / "FitNotes_Backup.fitnotes").exists()
    assert list(data.iterdir()) == []
