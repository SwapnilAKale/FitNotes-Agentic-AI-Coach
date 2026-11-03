"""
The speech worker: a child process of the web server that holds the model.

    python -m src.dictation.worker <models folder>

The web server starts it on the first recording and stops it after 15 idle
minutes (engine.py), so the ~1.2 GB model is in memory only while dictation is
in use, and a crash in native code ends this process, not the chat.

Talks over stdin/stdout in frames: one kind byte, a 4-byte big-endian length,
then the payload. b"J" is a JSON message, b"A" is 16 kHz mono PCM16 audio for
the open recording.
    in:   open {id}   A...   close {id}   cancel
    out:  ready {load_ms}   final {id, text}   done {id}   error {id?, message}
The worker exits when stdin closes, so it never outlives the server.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

HEADER = struct.Struct(">cI")


def read_frame(stream) -> tuple[bytes, bytes] | None:
    head = stream.read(HEADER.size)
    if len(head) < HEADER.size:
        return None
    kind, length = HEADER.unpack(head)
    payload = stream.read(length)
    return (kind, payload) if len(payload) == length else None


def write_frame(stream, kind: bytes, payload: bytes) -> None:
    stream.write(HEADER.pack(kind, len(payload)) + payload)
    stream.flush()


def _no_network() -> None:
    """Defence in depth: audio never leaves the machine from this process."""
    def refuse(*_a, **_k):
        raise OSError("Networking is disabled in the speech worker")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse


def main(model_root: str) -> int:
    # The protocol owns the real stdout; anything native code prints goes to stderr.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    _no_network()
    lock = threading.Lock()

    def send(message: dict) -> None:
        with lock:
            write_frame(out, b"J", json.dumps(message).encode())

    began = time.monotonic()
    try:
        from src.dictation.recognizer import Capture, Recognizer
        recognizer = Recognizer(Path(model_root))
    except Exception as exc:
        send({"type": "error", "message": f"Speech models could not load: {exc}"})
        return 1
    send({"type": "ready", "load_ms": round((time.monotonic() - began) * 1000)})

    capture, capture_id = None, None
    stdin = sys.stdin.buffer
    while True:
        frame = read_frame(stdin)
        if frame is None:
            break
        kind, payload = frame
        try:
            if kind == b"A":
                if capture is not None:
                    capture.push(payload)
                continue
            m = json.loads(payload)
            if m["type"] == "open":
                if capture is not None:
                    capture.cancel()
                capture_id = m["id"]
                capture = Capture(recognizer, Path(model_root),
                                  lambda e, cid=capture_id: send({**e, "id": cid}))
            elif m["type"] == "close" and capture is not None and m.get("id") == capture_id:
                capture.close()
                capture = None
            elif m["type"] == "cancel" and capture is not None:
                capture.cancel()
                capture = None
        except Exception as exc:
            if capture is not None:
                capture.cancel()
                capture = None
            send({"type": "error", "id": capture_id, "message": f"Speech recognition failed: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
