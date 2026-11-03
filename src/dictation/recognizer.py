"""
Parakeet TDT 0.6B v2 (int8, sherpa-onnx export) and one recording's pipeline.

Ported from Confab's dictation (confab-stt core/: parakeet, manual-recognition,
recognition-queue). Runs inside the speech worker process (worker.py), never in
the web server: the model needs about 1.2 GB and a fault in native code must not
take the chat down with it.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from src.dictation.turns import (BOUNDARY_HOLD_SECONDS, OVERLAP_SECONDS, RATE, ContextTurns,
                                 Segmenter, contextual_result, pad, unpad)

PARAKEET = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
# Decoding that falls this far behind the speaker is stopped rather than left
# to grow without bound.
MAX_BACKLOG_SECONDS = 60


def models_present(model_root: Path) -> bool:
    root = Path(model_root)
    return ((root / PARAKEET / "encoder.int8.onnx").is_file()
            and (root / "silero-vad" / "silero_vad.onnx").is_file())


class Recognizer:
    """English only, greedy search, CPU. Each decode gets a fresh stream."""

    def __init__(self, model_root: Path):
        import sherpa_onnx

        d = Path(model_root) / PARAKEET
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=min(3, os.cpu_count() or 1), sample_rate=RATE, feature_dim=80,
            decoding_method="greedy_search", model_type="nemo_transducer", provider="cpu")

    def decode(self, samples: np.ndarray) -> dict:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(RATE, samples)
        self.recognizer.decode_stream(stream)
        r = stream.result
        return {"text": r.text, "tokens": list(r.tokens),
                "timestamps": list(r.timestamps), "durations": list(r.durations)}


class Capture:
    """One recording: PCM in through push() and close(), sentences out through emit.

    emit receives {"type": "final", "text"}, then {"type": "done"} once every
    turn is transcribed, or {"type": "error", "message"}. VAD runs on the
    caller's thread; decoding on its own, so audio keeps flowing during a
    decode (onnxruntime releases the GIL).
    """

    def __init__(self, recognizer: Recognizer, model_root: Path, emit: Callable[[dict], None]):
        self.recognizer, self.emit = recognizer, emit
        self.segmenter = Segmenter(model_root)
        self.turns = ContextTurns()
        self.jobs: queue.Queue = queue.Queue()
        self.backlog = 0.0                 # seconds of audio waiting to be decoded
        self.lock = threading.Lock()
        self.cancelled = False
        self.failed = False
        # Overlap handoff between the pieces of one long turn.
        self.accepted_through: float | None = None
        self.accepted_text = ""
        self.pending_text = ""
        self.thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.thread.start()

    def push(self, pcm16: bytes) -> None:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        self._queue(self.turns.push(samples, self.segmenter.push(samples)))

    def close(self) -> None:
        self._queue(self.turns.push(np.zeros(0, np.float32), self.segmenter.flush(), flush=True))
        self.jobs.put(None)                # done once everything before it is decoded

    def cancel(self) -> None:
        self.cancelled = True
        self.jobs.put(None)

    def _queue(self, pieces: list[dict]) -> None:
        for piece in pieces:
            with self.lock:
                self.backlog += len(piece["samples"]) / RATE
                behind = self.backlog > MAX_BACKLOG_SECONDS
            if behind:
                self._fail("Recognition fell behind. Your unfinished speech was not sent.")
                return
            self.jobs.put(piece)

    def _fail(self, message: str) -> None:
        if not (self.failed or self.cancelled):
            self.failed = True
            self.emit({"type": "error", "message": message})

    def _decode_loop(self) -> None:
        while True:
            piece = self.jobs.get()
            if self.cancelled or self.failed:
                return
            if piece is None:
                self.emit({"type": "done"})
                return
            try:
                text = self._transcribe(piece)
            except Exception as exc:       # native or shape errors end this recording only
                self._fail(f"Speech recognition failed: {exc}")
                return
            with self.lock:
                self.backlog -= len(piece["samples"]) / RATE
            if not self.cancelled and text.strip():
                self.emit({"type": "final", "text": text.strip()})

    def _transcribe(self, piece: dict) -> str:
        timing, samples = piece["timing"], piece["samples"]
        padded, lead = pad(samples, timing)
        result = unpad(self.recognizer.decode(padded), lead, len(samples))
        raw = result["text"]
        overlapped = timing["overlap_seconds"] > 0
        try:
            context = (max(0.0, (self.accepted_through if self.accepted_through is not None
                                 else timing["commit_start"]) - timing["start"])
                       if overlapped else 0.0)
            hold = BOUNDARY_HOLD_SECONDS if timing["reason"] == "length-limit" else 0
            trimmed = contextual_result(result, context_seconds=context, hold_seconds=hold,
                                        audio_seconds=timing["end"] - timing["start"],
                                        overlap_seconds=timing["context_overlap_seconds"] or OVERLAP_SECONDS,
                                        previous_text=self.accepted_text)
            text = trimmed["text"]
            if not raw.strip() and self.pending_text and not hold:
                text = self.pending_text
            self.pending_text = trimmed["pending_text"]
            self.accepted_through = timing["start"] + trimmed["commit_seconds"]
        except ValueError:
            # No usable word timing: keep the raw words rather than guess which to drop.
            text = raw
            self.pending_text = ""
            self.accepted_through = timing["end"]
        self.accepted_text = (((self.accepted_text if overlapped else "") + " " + text).strip())[-4096:]
        return text
