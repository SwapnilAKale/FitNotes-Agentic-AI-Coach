"""
The web server's side of the speech worker (worker.py).

  SpeechWorker   starts `python -m src.dictation.worker`, passes audio in and
                 events out. Pipe I/O runs on threads, so a slow pipe can never
                 stall the event loop the chat runs on.
  LazyEngine     loads the worker on the first recording and stops it after
                 idle_seconds without one (Confab's lazy-engine.cjs). Audio that
                 arrives while the model loads is kept and sent once it is ready,
                 so nothing said in the first seconds is lost.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from src.dictation.worker import read_frame, write_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTUP_SECONDS = 90
# Audio kept while the model loads: about 125 s of 16 kHz PCM16.
MAX_BUFFERED_BYTES = 16000 * 2 * 125


class SpeechWorker:
    def __init__(self, model_root: Path, on_event: Callable[[dict], None], log: Callable[[str], None]):
        self.loop = asyncio.get_running_loop()
        self.on_event, self.log = on_event, log
        self.ready = self.loop.create_future()
        self.outbox: queue.Queue = queue.Queue()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.dictation.worker", str(model_root)],
            cwd=str(REPO_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for target in (self._read_events, self._read_stderr, self._write):
            threading.Thread(target=target, daemon=True).start()

    def _deliver(self, event: dict) -> None:
        if event["type"] == "ready":
            if not self.ready.done():
                self.ready.set_result(event)
            return
        if event["type"] in ("error", "exit") and not self.ready.done():
            self.ready.set_exception(RuntimeError(event["message"]))
            return
        self.on_event(event)

    def _read_events(self) -> None:
        while (frame := read_frame(self.proc.stdout)) is not None:
            self.loop.call_soon_threadsafe(self._deliver, json.loads(frame[1]))
        self.loop.call_soon_threadsafe(
            self._deliver, {"type": "exit", "message": "Speech recognition stopped unexpectedly."})

    def _read_stderr(self) -> None:
        for line in self.proc.stderr:
            text = line.decode(errors="replace").rstrip()
            if text:
                self.log("worker: " + text)

    def _write(self) -> None:
        while (item := self.outbox.get()) is not None:
            try:
                write_frame(self.proc.stdin, *item)
            except (BrokenPipeError, OSError, ValueError):
                return

    def send(self, message: dict) -> None:
        self.outbox.put((b"J", json.dumps(message).encode()))

    def audio(self, pcm: bytes) -> None:
        self.outbox.put((b"A", pcm))

    def close(self) -> None:
        self.outbox.put(None)
        try:
            self.proc.stdin.close()        # the worker exits when its stdin closes
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class LazyEngine:
    """One recording at a time on one worker. state: unloaded | loading | loaded | error."""

    def __init__(self, model_root: Path, idle_seconds: float = 900,
                 on_state: Callable[[str], None] = lambda s: None,
                 log: Callable[[str], None] = lambda line: None,
                 make_worker=SpeechWorker):
        self.model_root, self.idle_seconds = model_root, idle_seconds
        self.on_state, self.log, self.make_worker = on_state, log, make_worker
        self.state = "unloaded"
        self.worker = None
        self.loading: asyncio.Task | None = None
        self.capture: dict | None = None
        self.last_used = time.monotonic()
        self.idle_task: asyncio.Task | None = None

    def _set(self, state: str) -> None:
        self.state = state
        self.on_state(state)

    def _event(self, event: dict) -> None:
        c = self.capture
        if event["type"] == "exit":
            self.worker = None
            self._set("unloaded")
            if c:
                self.capture = None
                c["emit"]({"type": "error", "message": event["message"]})
            return
        if not c or event.get("id") != c["id"]:
            return
        c["emit"](event)
        if event["type"] in ("done", "error"):
            self.capture = None
            self.last_used = time.monotonic()

    async def warmup(self) -> None:
        self.last_used = time.monotonic()
        if self.state == "loaded":
            return
        if self.loading is None:
            self.loading = asyncio.ensure_future(self._load())
        await asyncio.shield(self.loading)

    async def _load(self) -> None:
        self._set("loading")
        worker = self.make_worker(self.model_root, self._event, self.log)
        try:
            ready = await asyncio.wait_for(asyncio.shield(worker.ready), STARTUP_SECONDS)
        except Exception:
            worker.close()
            self._set("error")
            raise
        finally:
            self.loading = None
        self.worker = worker
        self.log(f"speech model loaded in {ready.get('load_ms', 0)} ms")
        self._set("loaded")
        if self.idle_task is None:
            self.idle_task = asyncio.ensure_future(self._unload_when_idle())

    async def _unload_when_idle(self) -> None:
        while self.worker is not None:
            await asyncio.sleep(min(30, self.idle_seconds))
            if (self.worker is not None and self.capture is None and self.loading is None
                    and time.monotonic() - self.last_used >= self.idle_seconds):
                self.worker.close()
                self.worker = None
                self.log("speech model unloaded after being idle")
                self._set("unloaded")
        self.idle_task = None

    # A recording opens at once; its audio waits in memory while the model loads.
    def start(self, capture_id: str, emit: Callable[[dict], None]) -> None:
        if self.capture is not None:
            raise RuntimeError("Capture already open.")
        c = self.capture = {"id": capture_id, "emit": emit, "buffer": [], "bytes": 0,
                            "ended": False, "open": False}
        asyncio.ensure_future(self._open(c))

    async def _open(self, c: dict) -> None:
        try:
            await self.warmup()
        except Exception as exc:
            if self.capture is c:
                self.capture = None
                c["emit"]({"type": "error", "message": f"Speech models could not load: {exc}"})
            return
        if self.capture is not c or self.worker is None:
            return
        self.worker.send({"type": "open", "id": c["id"]})
        for pcm in c["buffer"]:
            self.worker.audio(pcm)
        c["buffer"], c["open"] = [], True
        if c["ended"]:
            self.worker.send({"type": "close", "id": c["id"]})

    def audio(self, capture_id: str, pcm: bytes) -> None:
        c = self.capture
        if not c or c["id"] != capture_id:
            raise RuntimeError("Capture closed.")
        self.last_used = time.monotonic()
        if c["open"]:
            self.worker.audio(pcm)
            return
        c["bytes"] += len(pcm)
        if c["bytes"] > MAX_BUFFERED_BYTES:
            raise RuntimeError("Recording exceeded the capture buffer.")
        c["buffer"].append(pcm)

    def end(self, capture_id: str) -> None:
        c = self.capture
        if not c or c["id"] != capture_id:
            return
        c["ended"] = True
        if c["open"]:
            self.worker.send({"type": "close", "id": c["id"]})

    def cancel(self, capture_id: str) -> None:
        c = self.capture
        if not c or c["id"] != capture_id:
            return
        self.capture = None
        self.last_used = time.monotonic()
        if c["open"] and self.worker is not None:
            self.worker.send({"type": "cancel"})

    def close(self) -> None:
        if self.capture:
            self.cancel(self.capture["id"])
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.idle_task is not None:
            self.idle_task.cancel()
        self.state = "unloaded"


def default_model_root() -> Path:
    return Path(os.environ.get("SPEECH_MODELS") or REPO_ROOT / "models" / "speech")
