"""
Dictation over a WebSocket at /stt, served by the chat UI's own server
(frontend/server.py). Ported from confab-stt's server/index.cjs; the browser
client (frontend/vendor/confab-stt.js) speaks the same protocol:

  browser → server   {"type":"start"}  binary PCM16 frames  {"type":"stop"}  {"type":"cancel"}
  server → browser   started, loaded, final {text}, limit {reason}, done {text},
                     cancelled, busy, error {reason, message}

A browser streams 16 kHz mono PCM while the person speaks; each sentence comes
back as it is recognised, the whole text when the recording ends. Audio is held
only in memory while it is recognised; nothing is written to disk, and the log
records durations and character counts, never words.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from starlette.websockets import WebSocket, WebSocketDisconnect

from src.dictation.engine import LazyEngine, default_model_root
from src.dictation.recognizer import models_present

RATE = 16000
DEFAULTS = {
    "slots": 1,                  # recordings recognised at once; each holds its own model (~1.2 GB)
    "max_seconds": 600,          # a recording stops itself after 10 minutes
    "silent_stop_seconds": 15,   # stop a recording whose browser sends no audio for this long
    "finish_seconds": 120,       # after stop, give up on recognition after this long
    "idle_unload_seconds": 900,  # unload the model after 15 minutes without a recording
    "starts_per_minute": 6,      # per visitor IP
    "starts_per_day": 120,       # per visitor IP
    "connections_per_ip": 3,
    "max_connections": 100,
    "origins": None,             # allowed browser origins; None = this page's own origin only
}
UNAVAILABLE = ("Dictation isn't set up on this server: the speech models are missing. "
               "Run python scripts/fetch_speech_models.py, or set SPEECH_MODELS.")


class Quota:
    """Fixed-window counter per key, pruned as windows expire."""

    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_seconds, clock
        self.windows: dict = {}

    def take(self, key: str) -> int:
        """0 if allowed; otherwise the seconds until the window resets."""
        now = self.clock()
        if len(self.windows) > 10000:
            self.windows = {k: v for k, v in self.windows.items() if now - v[0] < self.window}
        since, count = self.windows.get(key, (now, 0))
        if now - since >= self.window:
            since, count = now, 0
        if count >= self.limit:
            return max(1, int(since + self.window - now + 0.999))
        self.windows[key] = (since, count + 1)
        return 0


def origin_allowed(headers, origins: list | None) -> bool:
    """A browser always sends Origin on a WebSocket handshake. Another website
    must not be able to open the visitor's microphone stream to this server."""
    origin = headers.get("origin")
    if origins and "*" in origins:
        return True
    if origin is None:
        return True                      # not a browser
    if origins:
        return origin in origins
    host = (headers.get("x-forwarded-host") or headers.get("host") or "").split(",")[0].strip()
    try:
        return urlsplit(origin).netloc == host
    except ValueError:
        return False


class _Slot:
    def __init__(self, index: int, engine):
        self.index, self.engine, self.session = index, engine, None


class DictationService:
    def __init__(self, model_root: Path | None = None, make_engine: Callable | None = None,
                 log: Callable[[str], None] = lambda line: None, **options):
        self.o = {**DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
        self.model_root = Path(model_root) if model_root else default_model_root()
        self.log = log
        self.fake = make_engine is not None
        make = make_engine or (lambda on_state: LazyEngine(
            self.model_root, idle_seconds=self.o["idle_unload_seconds"], on_state=on_state, log=log))
        self.slots = []
        for i in range(self.o["slots"]):
            slot = _Slot(i, None)
            slot.engine = make(lambda state, slot=slot: self._state(slot, state))
            self.slots.append(slot)
        self.per_minute = Quota(self.o["starts_per_minute"], 60)
        self.per_day = Quota(self.o["starts_per_day"], 86400)
        self.connections: dict = {}
        self.total = 0
        self.closed = False

    def _state(self, slot: _Slot, state: str) -> None:
        self.log(f"slot {slot.index}: model {state}")
        if state == "loaded" and slot.session:
            slot.session.send({"type": "loaded"})

    def unavailable(self) -> str | None:
        """Why dictation cannot work here, or None."""
        if self.fake or models_present(self.model_root):
            return None
        return UNAVAILABLE

    def health(self) -> dict:
        return {"ok": True, "connections": self.total,
                "slots": [{"model": s.engine.state, "busy": s.session is not None} for s in self.slots]}

    def close(self) -> None:
        self.closed = True
        for s in self.slots:
            s.engine.close()

    async def handle(self, ws: WebSocket, ip: str) -> None:
        if self.closed or not origin_allowed(ws.headers, self.o["origins"]):
            if not self.closed:
                self.log(f"refused origin {ws.headers.get('origin')}")
            await ws.close(code=1008)    # before accept: the browser sees a refused handshake
            return
        await ws.accept()
        if self.total >= self.o["max_connections"] or self.connections.get(ip, 0) >= self.o["connections_per_ip"]:
            await ws.send_text(json.dumps({"type": "error", "reason": "too_many_connections",
                                           "message": "Too many dictation connections from your network. "
                                                      "Close another tab and try again."}))
            await ws.close()
            return
        self.total += 1
        self.connections[ip] = self.connections.get(ip, 0) + 1
        try:
            await _Session(self, ws, ip).run()
        finally:
            self.total -= 1
            n = self.connections[ip] - 1
            if n:
                self.connections[ip] = n
            else:
                del self.connections[ip]


class _Session:
    def __init__(self, service: DictationService, ws: WebSocket, ip: str):
        self.svc, self.ws, self.ip = service, ws, ip
        self.slot: _Slot | None = None
        self.id: str | None = None
        self.phase = "idle"
        self.texts: list[str] = []
        self.samples = 0
        self.timers: dict = {}
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()

    def send(self, message: dict) -> None:
        self.outbox.put_nowait(message)

    async def _sender(self) -> None:
        while True:
            message = await self.outbox.get()
            try:
                await self.ws.send_text(json.dumps(message))
            except Exception:
                return

    def _clear(self) -> None:
        for t in self.timers.values():
            t.cancel()
        self.timers = {}

    def _timer(self, name: str, seconds: float, fn) -> None:
        if name in self.timers:
            self.timers[name].cancel()
        self.timers[name] = self.loop.call_later(seconds, fn)

    def _release(self) -> None:
        self._clear()
        if self.slot and self.slot.session is self:
            self.slot.session = None
        self.slot, self.id, self.phase = None, None, "idle"

    def _fail(self, reason: str, message: str) -> None:
        if self.slot and self.id:
            self.slot.engine.cancel(self.id)
        self._release()
        self.send({"type": "error", "reason": reason, "message": message})

    def _silent(self) -> None:
        self._timer("silent", self.svc.o["silent_stop_seconds"], lambda: self._finish("no_audio"))

    def _finish(self, reason: str | None) -> None:
        if self.phase != "recording":
            return
        self.phase = "finishing"
        if "silent" in self.timers:
            self.timers.pop("silent").cancel()
        if reason:
            self.send({"type": "limit", "reason": reason})
        self.slot.engine.end(self.id)
        self._timer("finish", self.svc.o["finish_seconds"] + self.samples / RATE,
                    lambda: self._fail("timeout", "Transcription took too long. Please try again."))

    def _start(self) -> None:
        o = self.svc.o
        if self.phase != "idle":
            return self.send({"type": "error", "reason": "already_started",
                              "message": "A recording is already open on this connection."})
        reason = self.svc.unavailable()
        if reason:
            return self.send({"type": "error", "reason": "unavailable", "message": reason})
        wait = self.svc.per_minute.take(self.ip) or self.svc.per_day.take(self.ip)
        if wait:
            when = f"{-(-wait // 60)} minutes" if wait > 120 else f"{wait} seconds"
            return self.send({"type": "error", "reason": "rate_limited", "retryAfterSeconds": wait,
                              "message": f"Too many recordings. Try again in {when}."})
        slot = next((s for s in self.svc.slots if s.session is None), None)
        if slot is None:
            return self.send({"type": "busy", "message": "Someone else is dictating right now. Try again in a moment."})
        slot.session = self
        self.slot, self.id, self.phase, self.texts, self.samples = slot, uuid.uuid4().hex, "recording", [], 0
        capture_id = self.id

        def on_event(event: dict) -> None:
            if self.id != capture_id:
                return
            if event["type"] == "final":
                text = str(event.get("text") or "").strip()
                if text:
                    self.texts.append(text)
                    self.send({"type": "final", "text": text})
            elif event["type"] == "done":
                text = " ".join(self.texts)
                self.svc.log(f"done: {self.samples / RATE:.1f} s of audio, {len(text)} characters")
                self._release()
                self.send({"type": "done", "text": text})
            elif event["type"] == "error":
                self.svc.log("recognition error: " + str(event.get("message")))
                self._fail("recognition_failed", event.get("message") or "Speech recognition failed.")

        loading = slot.engine.state != "loaded"
        try:
            slot.engine.start(capture_id, on_event)
        except Exception as exc:
            return self._fail("recognition_failed", str(exc))
        self._timer("limit", o["max_seconds"] + 5, lambda: self._finish("max_duration"))
        self._silent()
        self.send({"type": "started", "sampleRate": RATE, "maxSeconds": o["max_seconds"], "loading": loading})

    def _audio(self, data: bytes) -> None:
        if self.phase != "recording":
            return                         # frames still in flight after stop or a limit
        if len(data) % 2:
            return self._fail("bad_audio", "Audio must be 16-bit PCM.")
        room = self.svc.o["max_seconds"] * RATE - self.samples
        if len(data) // 2 > room:
            data = data[:room * 2]
        self.samples += len(data) // 2
        try:
            if data:
                self.slot.engine.audio(self.id, data)
        except Exception as exc:
            return self._fail("overloaded", str(exc))
        self._silent()
        if self.samples >= self.svc.o["max_seconds"] * RATE:
            self._finish("max_duration")

    def _cancel(self) -> None:
        if self.id:
            self.slot.engine.cancel(self.id)
        was = self.phase
        self._release()
        if was != "idle":
            self.send({"type": "cancelled"})

    async def run(self) -> None:
        sender = asyncio.ensure_future(self._sender())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    self._audio(msg["bytes"])
                    continue
                try:
                    m = json.loads(msg.get("text") or "")
                except ValueError:
                    self.send({"type": "error", "reason": "bad_message", "message": "Messages must be JSON."})
                    continue
                if m.get("type") == "start":
                    self._start()
                elif m.get("type") == "stop":
                    self._finish(None)
                elif m.get("type") == "cancel":
                    self._cancel()
        except WebSocketDisconnect:
            pass
        finally:
            if self.id:
                self.slot.engine.cancel(self.id)
            self._release()
            sender.cancel()
