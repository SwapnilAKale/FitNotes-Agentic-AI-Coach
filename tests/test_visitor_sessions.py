"""
Per-visitor conversations in the public demo (src/visitor_sessions.py, server.py).

Every visitor used to share ONE conversation: /history returned the same chat to
everyone and the agent remembered other visitors' turns. With DEMO_MODE=1 each
browser now gets a random session id (cookie) and its own stored state —
history, the agent's context, pending panels, the checkpoint slot and its
remembered facts — swapped in around its own turns. With DEMO_MODE off nothing
of this runs: the single global conversation is used exactly as before.

The server tests drive the real ASGI app in one event loop (httpx
ASGITransport), with one httpx client per "browser" so each has its own cookie
jar, and a fake agent/coordinator so no Gemini call is made.
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src import memory as _memory                    # noqa: E402
from src import visitor_sessions as vs               # noqa: E402


# ── the store ────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _state(text="hi"):
    st = vs.blank_state({"panel": None})
    st["agent"]["chat_history"] = [{"role": "user", "text": text}]
    return st


def test_store_round_trip_survives_a_new_store_object(tmp_path):
    path = str(tmp_path / "s.sqlite")
    a = vs.SessionStore(path)
    st = _state("remember me")
    st["agent"]["_conversation_history"] = [[{"role": "user", "content": "x",
                                              "sig": b"\x00\x01"}]]
    a.save("A" * 22, st, prompted=True)
    a.close()
    b = vs.SessionStore(path)                       # a server restart
    got = b.load("A" * 22)
    assert got["agent"]["chat_history"][0]["text"] == "remember me"
    assert got["agent"]["_conversation_history"][0][0]["sig"] == b"\x00\x01"
    assert b.load("B" * 22) is None


def test_idle_sessions_past_the_ttl_are_swept(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_SESSION_TTL_DAYS", "30")
    clock = Clock()
    store = vs.SessionStore(str(tmp_path / "s.sqlite"), clock=clock)
    store.save("old" * 8, _state(), prompted=True)
    clock.t += 20 * 86400
    store.save("new" * 8, _state(), prompted=True)
    # Saving without a prompt (e.g. a panel cancel) does not restart the clock.
    store.save("old" * 8, _state())
    clock.t += 11 * 86400                            # old: 31 days, new: 11
    assert store.sweep() == 1
    assert store.load("old" * 8) is None
    assert store.load("new" * 8) is not None
    # A visitor whose turn is running is never swept.
    clock.t += 40 * 86400
    assert store.sweep(protect={"new" * 8}) == 0
    assert store.sweep() == 1 and store.count() == 0


def test_session_count_cap_evicts_least_recently_used(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MAX_SESSIONS", "3")
    clock = Clock()
    store = vs.SessionStore(str(tmp_path / "s.sqlite"), clock=clock)
    ids = [f"visitor{i}".ljust(22, "x") for i in range(3)]
    for sid in ids:
        clock.t += 1
        store.save(sid, _state(), prompted=True)
    clock.t += 1
    store.save(ids[0], _state(), prompted=True)      # 0 is now the most recent
    clock.t += 1
    store.save("newcomer".ljust(22, "x"), _state(), prompted=True)
    assert store.count() == 3
    assert store.load(ids[1]) is None                # least recently used
    assert store.load(ids[0]) is not None and store.load(ids[2]) is not None


def test_store_size_cap_evicts_and_one_session_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_SESSION_MAX_KB", "16")
    monkeypatch.setenv("DEMO_SESSIONS_MAX_MB", "1")
    clock = Clock()
    store = vs.SessionStore(str(tmp_path / "s.sqlite"), clock=clock)
    big = vs.blank_state({})
    big["agent"]["chat_history"] = [{"role": "user", "text": "x" * 1000, "n": i}
                                    for i in range(100)]
    size = store.save("big".ljust(22, "x"), big, prompted=True)
    assert size <= 16 * 1024                          # trimmed to fit
    kept = store.load("big".ljust(22, "x"))["agent"]["chat_history"]
    assert kept and kept[-1]["n"] == 99               # newest entries kept
    for i in range(80):                               # ~16 KB each → > 1 MB
        clock.t += 1
        store.save(f"s{i}".ljust(22, "x"), big, prompted=True)
    assert store.total_bytes() <= 1024 * 1024
    assert store.load("s79".ljust(22, "x")) is not None
    assert store.load("big".ljust(22, "x")) is None   # oldest went first


def test_session_ids_are_random_and_validated():
    a, b = vs.new_sid(), vs.new_sid()
    assert a != b and vs.valid_sid(a) and len(a) >= 22
    for bad in ("", None, "short", "a" * 200, "x" * 21 + ";", "../" * 10):
        assert not vs.valid_sid(bad)


# ── the queue ────────────────────────────────────────────────────────────────

def test_turn_queue_is_fifo_bounded_and_one_place_per_visitor():
    async def go():
        q = vs.TurnQueue()
        assert q.enter("a", 2) is None and q.ahead("a") == 0
        tb, tc = q.enter("b", 2), q.enter("c", 2)
        assert (q.ahead("b"), q.ahead("c")) == (1, 2)
        with pytest.raises(vs.AlreadyActive):
            q.enter("a", 2)
        with pytest.raises(vs.AlreadyActive):
            q.enter("b", 2)
        with pytest.raises(vs.QueueFull):
            q.enter("d", 2)
        order = []

        async def turn(sid, ticket):
            await q.wait(sid, ticket)
            order.append(sid)
            q.release(sid)

        tasks = [asyncio.create_task(turn("c", tc)), asyncio.create_task(turn("b", tb))]
        await asyncio.sleep(0)
        q.release("a")
        await asyncio.gather(*tasks)
        assert order == ["b", "c"] and q.running is None and q.active() == set()

        # A waiter that goes away (client disconnected) gives up its place.
        assert q.enter("x", 2) is None
        ty = q.enter("y", 2)
        tz = q.enter("z", 2)
        waiter = asyncio.create_task(q.wait("y", ty))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert q.ahead("z") == 1
        q.release("x")
        await q.wait("z", tz)
        assert q.running == "z"
    asyncio.run(go())


def test_queue_message_counts_the_others():
    assert "after 1 other." in vs.queue_message(1)
    assert "after 2 others." in vs.queue_message(2)


# ── demo write refusal, layer by layer ───────────────────────────────────────

def test_mcp_server_refuses_every_shared_write_in_demo_mode(monkeypatch):
    from src.agent import SHARED_WRITE_TOOLS, MEMORY_TOOLS
    import mcp_servers.combined_server as cs
    assert SHARED_WRITE_TOOLS <= cs._DEMO_REFUSED_TOOLS
    assert (MEMORY_TOOLS - {"recall_memories"}) <= cs._DEMO_REFUSED_TOOLS
    monkeypatch.setenv("DEMO_MODE", "1")
    for name in sorted(cs._DEMO_REFUSED_TOOLS):
        out = asyncio.run(cs.call_tool(name, {}))
        assert json.loads(out[0].text)["error"] == "demo_read_only", name


def test_db_write_connection_is_refused_in_demo_mode(monkeypatch, tmp_path):
    from src.db import get_write_connection
    monkeypatch.setenv("DEMO_MODE", "1")
    with pytest.raises(PermissionError):
        get_write_connection(str(tmp_path / "x.db"))
    monkeypatch.delenv("DEMO_MODE")
    get_write_connection(str(tmp_path / "x.db")).close()


def test_agent_session_refuses_and_routes_memory_in_process(monkeypatch, tmp_path):
    from src.agent import AgentSession, SHARED_WRITE_TOOLS
    s = AgentSession(str(tmp_path / "x.db"))
    assert s.read_only_tools == frozenset() and not s.memory_in_process
    assert s.auto_extract_memories
    s.read_only_tools = SHARED_WRITE_TOOLS
    s.memory_in_process = True
    out = json.loads(asyncio.run(s.call_tool("execute_staged_workout", {})))
    assert out["error"] == "demo_read_only" and out["success"] is False
    monkeypatch.setattr(_memory, "MEMORY_PATH", tmp_path / "memory.json")
    overlay = vs.blank_memory()
    _memory.set_overlay(overlay)
    try:
        out = json.loads(asyncio.run(s.call_tool(
            "remember_fact", {"category": "preference", "content": "likes squats"})))
        assert out["status"] == "saved"
    finally:
        _memory.set_overlay(None)
    assert overlay["facts"][0]["content"] == "likes squats"
    assert not (tmp_path / "memory.json").exists()     # the shared file untouched


# ── container reset keeps visitors' conversations ───────────────────────────

def test_demo_reset_keeps_visitor_sessions(monkeypatch, tmp_path):
    import importlib.util
    data = tmp_path / "data"
    data.mkdir()
    store_dir = data / "visitor_sessions"
    store_dir.mkdir()
    (store_dir / "sessions.sqlite").write_text("keep me")
    (data / "memory.json").write_text('{"facts": ["stale"]}')
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("FITNOTES_DB_PATH", str(data / "FitNotes_Backup.fitnotes"))
    monkeypatch.setenv("CHROMA_DB_PATH", str(data / "chroma_db"))
    monkeypatch.setenv("SEED_CHROMA_DIR", str(tmp_path / "no-seed"))
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_RESET_ON_START", "1")
    monkeypatch.delenv("DEMO_SESSIONS_PATH", raising=False)
    spec = importlib.util.spec_from_file_location(
        "prepare_data_vs", os.path.join(_ROOT, "deploy", "prepare_data.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    assert prep.main() == 0
    assert (store_dir / "sessions.sqlite").read_text() == "keep me"
    assert "stale" not in (data / "memory.json").read_text()   # base data reset


# ── the server, two browsers ─────────────────────────────────────────────────

class FakeCoordinator:
    """Answers by echoing; records what conversation context it saw."""

    def __init__(self, srv):
        self.srv = srv
        self._history = []
        self._pending_followup = None
        self._pending_decomposition = None
        self._pending_log_carry = False
        self._log_flow_turns = []
        self.gates = {}

    async def route(self, message):
        if message in self.gates:
            await self.gates[message].wait()
        seen = [h["content"] for h in self._history if h["role"] == "user"]
        self._history += [{"role": "user", "content": message},
                          {"role": "assistant", "content": "ok"}]
        if message.startswith("remember "):
            _memory.add_fact("preference", message[len("remember "):])
        if message == "what do you know":
            facts = [f["content"] for f in _memory.get_all_facts()]
            return {"answer": f"facts={facts}", "route": "operational"}
        if message.startswith("log "):
            # What the real agent's log_workout call does via the handler.
            await self.srv._confirmation_handler("log_workout", {"what": message})
            return {"answer": "staged", "route": "operational",
                    "log_flow_turns": [message]}
        if message == "ambiguous":
            groups = [{"name": "press", "candidates": ["Bench Press", "Leg Press"]}]
            self._pending_decomposition = {"groups": groups}
            return {"disambiguation": {"groups": groups}}
        return {"answer": f"echo:{message} seen={seen}", "route": "analytical"}

    async def verify_log_staging(self, flow, slot_raw, preview):
        return {"verdict": "PASS"}

    def cancel_disambiguation(self):
        self._pending_decomposition = None


class FakeSession:
    def __init__(self):
        self.chat_history = []
        self._conversation_history = []
        self._staged_active = False
        self.calls = []
        self.slot = []

    async def call_tool(self, name, args):
        self.calls.append(name)
        if name == "discard_staged_writes":
            self.slot = []
            return json.dumps({"ok": True})
        if name == "read_staged_workout_slot":
            return json.dumps({"staged_workouts": self.slot or [{"x": 1}]})
        if name == "format_staged_workout_for_confirmation":
            return json.dumps({"preview": "Bench Press 100 x 5"})
        return json.dumps({"ok": True})

    def _save_exchange(self, messages, start):
        self._conversation_history.append(messages[start:])


@pytest.fixture()
def demo(monkeypatch, tmp_path):
    import server as srv
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("DEMO_RATE_LIMIT_PER_DAY", "0")
    monkeypatch.setattr(srv.demo_mode, "_limiter", None)
    monkeypatch.setattr(_memory, "MEMORY_PATH", tmp_path / "memory.json")
    session = FakeSession()
    coord = FakeCoordinator(srv)
    store = vs.SessionStore(str(tmp_path / "visitor_sessions" / "s.sqlite"))
    monkeypatch.setattr(srv, "session", session)
    monkeypatch.setattr(srv, "coordinator", coord)
    monkeypatch.setattr(srv, "agent_ready", True)
    monkeypatch.setattr(srv, "agent_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "_visitor_store", store)
    monkeypatch.setattr(srv, "_turn_queue", vs.TurnQueue())
    saved_state = dict(srv._state)
    srv._state.update(srv._STATE_DEFAULTS)
    yield SimpleNamespace(srv=srv, session=session, coord=coord, store=store,
                          path=store.path)
    srv._state.clear()
    srv._state.update(saved_state)
    _memory.set_overlay(None)
    store.close()


def _browser(srv):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=srv.app),
                             base_url="http://demo.test")


def _texts(history):
    return [(m["role"], m.get("text")) for m in history]


def test_two_visitors_never_see_each_others_conversation(demo):
    async def go():
        async with _browser(demo.srv) as a, _browser(demo.srv) as b:
            r = await a.post("/chat", json={"message": "hello from A"})
            assert r.json()["text"] == "echo:hello from A seen=[]"
            sid_a = a.cookies.get(vs.COOKIE_NAME)
            assert vs.valid_sid(sid_a)
            r = await b.post("/chat", json={"message": "hello from B"})
            # B's agent context does not contain A's turn.
            assert r.json()["text"] == "echo:hello from B seen=[]"
            assert b.cookies.get(vs.COOKIE_NAME) != sid_a
            r = await a.post("/chat", json={"message": "again"})
            assert r.json()["text"] == "echo:again seen=['hello from A']"

            ha = (await a.get("/history")).json()["history"]
            hb = (await b.get("/history")).json()["history"]
            assert _texts(ha) == [
                ("user", "hello from A"),
                ("assistant", "echo:hello from A seen=[]"),
                ("user", "again"),
                ("assistant", "echo:again seen=['hello from A']")]
            assert _texts(hb) == [("user", "hello from B"),
                                  ("assistant", "echo:hello from B seen=[]")]
            async with _browser(demo.srv) as c:          # a brand-new visitor
                assert (await c.get("/history")).json()["history"] == []
        # Between turns the live objects hold nobody's conversation.
        assert demo.session.chat_history == [] and demo.coord._history == []
    asyncio.run(go())


def test_pending_confirmation_and_checkpoint_are_per_visitor(demo):
    async def go():
        async with _browser(demo.srv) as a, _browser(demo.srv) as b:
            r = await a.post("/chat", json={"message": "log bench 100x5"})
            assert r.json()["type"] == "confirmation_required"
            assert r.json()["preview"] == "Bench Press 100 x 5"
            pa = (await a.get("/pending-confirmation")).json()
            assert pa["preview"] == "Bench Press 100 x 5"
            assert (await a.get("/checkpoint-status")).json()["exists"] is True

            # B sees no panel and no checkpoint, before and after its own turn.
            assert (await b.get("/pending-confirmation")).json() == {}
            assert (await b.get("/checkpoint-status")).json() == {"exists": False}
            await b.post("/chat", json={"message": "hi"})
            assert (await b.get("/pending-confirmation")).json() == {}
            # ...and A's panel survived B's turn.
            assert (await a.get("/pending-confirmation")).json()["preview"]

            # Confirm in the demo: answered, never executed.
            r = await a.post("/confirm", json={"confirmed": True})
            assert r.json()["type"] == "answer"
            assert "doesn't save changes" in r.json()["text"]
            assert not [c for c in demo.session.calls if c.startswith("execute_")]
            assert (await a.get("/pending-confirmation")).json() == {}
            assert (await a.get("/checkpoint-status")).json() == {"exists": False}
            hist = (await a.get("/history")).json()["history"]
            assert "doesn't save changes" in hist[-1]["text"]
    asyncio.run(go())


def test_pending_disambiguation_is_per_visitor(demo):
    async def go():
        async with _browser(demo.srv) as a, _browser(demo.srv) as b:
            r = await a.post("/chat", json={"message": "ambiguous"})
            assert r.json()["type"] == "disambiguation_required"
            ga = (await a.get("/pending-disambiguation")).json()["groups"]
            assert ga[0]["name"] == "press"
            assert (await b.get("/pending-disambiguation")).json()["groups"] is None
            r = await b.post("/disambiguate/cancel")
            assert r.status_code == 200
            assert (await a.get("/pending-disambiguation")).json()["groups"]
            await a.post("/disambiguate/cancel")
            assert (await a.get("/pending-disambiguation")).json()["groups"] is None
    asyncio.run(go())


def test_remembered_facts_are_per_visitor_and_never_shared(demo, tmp_path):
    async def go():
        async with _browser(demo.srv) as a, _browser(demo.srv) as b:
            await a.post("/chat", json={"message": "remember I am 45"})
            r = await b.post("/chat", json={"message": "what do you know"})
            assert r.json()["text"] == "facts=[]"
            r = await a.post("/chat", json={"message": "what do you know"})
            assert r.json()["text"] == "facts=['I am 45']"
    asyncio.run(go())
    assert not (tmp_path / "memory.json").exists()
    assert _memory.get_overlay() is None


def test_same_visitor_keeps_history_across_reload_and_restart(demo):
    async def go():
        async with _browser(demo.srv) as a:
            await a.post("/chat", json={"message": "first"})
            sid = a.cookies.get(vs.COOKIE_NAME)
        # A reload: a new connection carrying the same cookie.
        async with _browser(demo.srv) as again:
            again.cookies.set(vs.COOKIE_NAME, sid, domain="demo.test")
            h = (await again.get("/history")).json()["history"]
            assert _texts(h)[0] == ("user", "first")
            # The response refreshes the same id, it does not replace it.
            assert again.cookies.get(vs.COOKIE_NAME) == sid
        return sid
    sid = asyncio.run(go())

    # A server restart: a new store object over the same file, and a fresh
    # agent with nothing in memory.
    demo.store.close()
    demo.srv._visitor_store = vs.SessionStore(demo.path)
    demo.srv.coordinator = FakeCoordinator(demo.srv)

    async def after_restart():
        async with _browser(demo.srv) as a:
            a.cookies.set(vs.COOKIE_NAME, sid, domain="demo.test")
            h = (await a.get("/history")).json()["history"]
            assert _texts(h) == [("user", "first"),
                                 ("assistant", "echo:first seen=[]")]
            r = await a.post("/chat", json={"message": "second"})
            assert r.json()["text"] == "echo:second seen=['first']"
    asyncio.run(after_restart())
    demo.srv._visitor_store.close()
    demo.srv._visitor_store = demo.store = vs.SessionStore(demo.path)


def test_a_turn_cut_off_by_a_restart_is_not_left_pending(demo):
    st = vs.blank_state(demo.srv._STATE_DEFAULTS)
    st["agent"]["chat_history"] = [
        {"role": "user", "text": "q"},
        {"role": "assistant", "text": "", "pending": True}]
    sid = vs.new_sid()
    demo.store.save(sid, st, prompted=True)

    async def go():
        async with _browser(demo.srv) as a:
            a.cookies.set(vs.COOKIE_NAME, sid, domain="demo.test")
            h = (await a.get("/history")).json()["history"]
            assert "pending" not in h[-1] and h[-1]["kind"] == "error"
    asyncio.run(go())


def test_other_visitors_queue_instead_of_busy(demo, monkeypatch):
    monkeypatch.setenv("DEMO_MAX_QUEUED_TURNS", "1")

    async def go():
        gate = asyncio.Event()
        demo.coord.gates["slow"] = gate
        async with _browser(demo.srv) as a, _browser(demo.srv) as b, \
                _browser(demo.srv) as c:
            await a.get("/status")
            await b.get("/status")
            ta = asyncio.create_task(a.post("/chat", json={"message": "slow"}))
            while demo.srv._turn_queue.running is None:
                await asyncio.sleep(0.01)
            tb = asyncio.create_task(b.post("/chat", json={"message": "queued"}))
            for _ in range(200):
                q = (await b.get("/queue-status")).json()
                if q["ahead"]:
                    break
                await asyncio.sleep(0.01)
            assert q["ahead"] == 1 and "after 1 other." in q["text"]
            assert (await a.get("/queue-status")).json()["ahead"] == 0
            # B's question is already in B's history while it waits.
            hb = (await b.get("/history")).json()["history"]
            assert hb[-2]["text"] == "queued" and hb[-1].get("pending")

            # Same visitor again while queued or running: busy.
            r = await b.post("/chat", json={"message": "dup"})
            assert r.status_code == 429 and r.json()["error"] != "demo_full"
            r = await a.post("/chat", json={"message": "dup"})
            assert r.status_code == 429
            # Queue full: a third visitor is told so.
            r = await c.post("/chat", json={"message": "late"})
            assert r.status_code == 429 and r.json()["error"] == "demo_full"

            gate.set()
            ra, rb = await ta, await tb
            assert ra.json()["text"] == "echo:slow seen=[]"
            assert rb.json()["text"] == "echo:queued seen=[]"
            assert (await b.get("/queue-status")).json()["ahead"] is None
    asyncio.run(go())


def test_demo_mode_off_uses_the_single_session(monkeypatch, tmp_path):
    import server as srv
    monkeypatch.delenv("DEMO_MODE", raising=False)
    session = FakeSession()
    session.chat_history = [{"role": "user", "text": "the one conversation"}]
    monkeypatch.setattr(srv, "session", session)
    monkeypatch.setattr(srv, "_visitor_store", None)
    agent = srv._new_agent_session()
    assert agent.read_only_tools == frozenset() and agent.auto_extract_memories
    assert not agent.memory_in_process

    async def go():
        async with _browser(srv) as a, _browser(srv) as b:
            for client in (a, b):
                r = await client.get("/history")
                assert _texts(r.json()["history"]) == [("user", "the one conversation")]
                assert vs.COOKIE_NAME not in r.headers.get("set-cookie", "")
            assert (await a.get("/queue-status")).json()["ahead"] is None
    asyncio.run(go())


def test_demo_mode_session_flags(monkeypatch):
    import server as srv
    from src.agent import SHARED_WRITE_TOOLS
    monkeypatch.setenv("DEMO_MODE", "1")
    agent = srv._new_agent_session()
    assert agent.read_only_tools == SHARED_WRITE_TOOLS
    assert agent.memory_in_process and not agent.auto_extract_memories


# ── the UI's reverse proxy must not carry one visitor's cookie to another ───

def test_frontend_proxy_never_reuses_a_visitors_session_cookie(monkeypatch):
    import importlib.util
    name = "fitnotes_frontend_server"
    fe = sys.modules.get(name)
    if fe is None:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_ROOT, "frontend", "server.py"))
        fe = importlib.util.module_from_spec(spec)
        sys.modules[name] = fe
        spec.loader.exec_module(fe)

    seen = []

    async def backend(scope, receive, send):
        cookie = dict(scope["headers"]).get(b"cookie", b"").decode()
        seen.append(cookie)
        await send({"type": "http.response.start", "status": 200, "headers": [
            (b"content-type", b"application/json"),
            (b"set-cookie", b"fn_sid=" + (b"A" * 22) + b"; Path=/; HttpOnly")]})
        await send({"type": "http.response.body", "body": b"{}"})

    async def go():
        monkeypatch.setattr(fe, "_http_client", fe._make_backend_client(
            transport=httpx.ASGITransport(app=backend)))
        transport = httpx.ASGITransport(app=fe.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ui") as first:
            await first.get("/history")
            assert first.cookies.get("fn_sid") == "A" * 22   # passed through
            await first.get("/history")                       # sends its own
        async with httpx.AsyncClient(transport=transport, base_url="http://ui") as new:
            await new.get("/history")                         # no cookie yet
        await fe._http_client.aclose()
    asyncio.run(go())
    assert seen == ["", "fn_sid=" + "A" * 22, ""]


def test_buttons_from_an_unknown_visitor_create_no_session(demo):
    async def go():
        async with _browser(demo.srv) as bot:
            for path, body in (("/confirm", {"confirmed": True}), ("/resume", None),
                               ("/disambiguate", {"selections": []}),
                               ("/disambiguate/cancel", None)):
                r = await bot.post(path, json=body)
                assert r.status_code == 200, path
            await bot.get("/history")
    asyncio.run(go())
    assert demo.store.count() == 0
    assert not [c for c in demo.session.calls if c.startswith("execute_")]
