"""
MCP session lifecycle (owner-task model). The stdio_client / ClientSession
contexts must be ENTERED and EXITED in the same long-lived owner task, so anyio's
stdio_client cancel scope is never exited cross-task (the bug that orphaned the
combined_server subprocess on every reload/upload/shutdown).

No Gemini, no real subprocess — stdio_client/ClientSession and the heavy init
bits are mocked. We assert the structural invariant (enter task == exit task),
clean close, idempotent close, reload isolation, and stop-before-park.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

import src.agent as agent_mod                 # noqa: E402
from src.agent import AgentSession            # noqa: E402


class FakeStdio:
    """Async-CM stand-in for mcp.client.stdio.stdio_client. Records the task it
    is entered/exited in so the test can assert they match."""
    instances: list = []

    def __init__(self, params):
        self.params = params
        self.enter_task = None
        self.exit_task = None
        self.entered = False
        self.exited = False
        FakeStdio.instances.append(self)

    async def __aenter__(self):
        self.enter_task = asyncio.current_task()
        self.entered = True
        return ("read", "write")

    async def __aexit__(self, *exc):
        self.exit_task = asyncio.current_task()
        self.exited = True
        return False


class FakeSession:
    """Async-CM stand-in for mcp.ClientSession."""
    def __init__(self, read, write):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        tool = SimpleNamespace(name="probe", description="d",
                               inputSchema={"type": "object", "properties": {}})
        return SimpleNamespace(tools=[tool])


@pytest.fixture(autouse=True)
def patch_mcp(monkeypatch):
    FakeStdio.instances.clear()
    monkeypatch.setattr(agent_mod, "stdio_client", lambda params: FakeStdio(params))
    monkeypatch.setattr(agent_mod, "ClientSession", FakeSession)
    monkeypatch.setattr(agent_mod.genai, "Client", lambda api_key=None: SimpleNamespace())
    # Stub the heavy, non-MCP parts of initialize()
    monkeypatch.setattr(AgentSession, "_setup_cache", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(AgentSession, "_build_gemini_tools", lambda self, t: [])
    monkeypatch.setattr(agent_mod, "load_user_context", lambda p=None: {})
    monkeypatch.setattr(agent_mod, "build_user_context_prompt", lambda c: "")
    import src.memory as _m
    monkeypatch.setattr(_m, "sync_to_chromadb", lambda: None)


# ── init → usable → close: clean, same-task teardown ─────────────────────────

def test_init_then_close_same_task_no_orphan():
    async def go():
        s = AgentSession("data/FitNotes_Backup.fitnotes")
        await s.initialize()

        # session usable; owner task alive; context entered, not yet exited
        assert s._session is not None
        owner = s._owner_task
        assert owner is not None and not owner.done()
        st = FakeStdio.instances[-1]
        assert st.entered and not st.exited

        await s.close()

        # owner finished; teardown done; references cleared
        assert owner.done()
        assert s._owner_task is None
        assert s._session is None
        # the crux: stdio_client exited in the SAME task it was entered in
        assert st.exited
        assert st.enter_task is st.exit_task
        assert st.enter_task is owner          # and that task is the owner task

    asyncio.run(go())


def test_double_close_is_idempotent():
    async def go():
        s = AgentSession("data/FitNotes_Backup.fitnotes")
        await s.initialize()
        await s.close()
        # second close must be a clean no-op (no exception, no error)
        await s.close()
        assert s._owner_task is None

    asyncio.run(go())


def test_reload_isolation_old_closes_new_lives():
    async def go():
        s1 = AgentSession("data/FitNotes_Backup.fitnotes")
        await s1.initialize()
        st1 = FakeStdio.instances[-1]

        s2 = AgentSession("data/FitNotes_Backup.fitnotes")
        await s2.initialize()
        st2 = FakeStdio.instances[-1]

        owner1 = s1._owner_task
        await s1.close()   # reinit order: old closes fully before/while new runs

        assert owner1.done()
        assert st1.exited and st1.enter_task is st1.exit_task   # old torn down in-task
        assert not st2.exited                                   # new still open
        assert s2._session is not None
        assert not s2._owner_task.done()

        await s2.close()
        assert st2.exited and st2.enter_task is st2.exit_task

    asyncio.run(go())


# ── owner task edge cases ────────────────────────────────────────────────────

def test_stop_before_park_exits_cleanly():
    async def go():
        s = AgentSession("data/FitNotes_Backup.fitnotes")
        s._stop_event  = asyncio.Event()
        s._ready_event = asyncio.Event()
        s._owner_error = None
        s._stop_event.set()                    # stop already requested before park

        await s._session_owner(params=None)    # runs enter → (park returns at once) → exit

        assert s._ready_event.is_set()
        assert s._owner_error is None
        st = FakeStdio.instances[-1]
        assert st.entered and st.exited
        assert st.enter_task is st.exit_task    # still same-task, no half-entered ctx
        assert s._session is None               # cleared in finally

    asyncio.run(go())


def test_owner_startup_failure_surfaces_via_initialize():
    async def go():
        class Boom(FakeSession):
            async def initialize(self):
                raise RuntimeError("mcp init blew up")

        import src.agent as am
        am.ClientSession = Boom            # patched within this test only (autouse re-patches next test)
        s = AgentSession("data/FitNotes_Backup.fitnotes")
        with pytest.raises(RuntimeError, match="mcp init blew up"):
            await s.initialize()
        # context entered then exited in-task even on failure (no orphan)
        st = FakeStdio.instances[-1]
        assert st.exited and st.enter_task is st.exit_task

    asyncio.run(go())
