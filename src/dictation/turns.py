"""
From a stream of microphone samples to the pieces of audio that get transcribed.

Ported from Confab's dictation (confab-stt core/: silero-segmenter, context-turns,
overlap-text, recognition-padding, turn-settings, vad-settings, pcm-window) with
the settings Confab's dictation runs with. Confab-only branches (live previews,
model comparison) are left out. Pure logic except `Segmenter`, which runs the
Silero model through sherpa-onnx.

  Segmenter      where speech is (Silero VAD): "segment" when a stretch of speech
                 ends, "activity" while it is still going on.
  ContextTurns   joins those into one turn per thing said. A turn ends after
                 SETTLE seconds of silence, on stop, or at MAX_TURN_SECONDS; a
                 turn cut at that limit is decoded again from OVERLAP seconds
                 earlier, so no word is cut in half.
  contextual_result  after an overlapped decode, drops the words the previous
                 piece already gave, and holds back the last second of a piece
                 cut at the limit (the next decode hears it whole).
  pad            short finished phrases get a little silence on both sides,
                 which Parakeet transcribes more reliably.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

RATE = 16000

# Speech detector (Confab's defaults, chosen 11 October 2026 against the
# segmenter they replaced): probability above which a 32 ms frame is speech,
# audio kept before the detected start so a soft first word is not clipped, and
# the silence that ends a segment (a turn ends later, after SETTLE).
VAD_THRESHOLD = 0.3
PRE_SPEECH_SECONDS = 0.75
MIN_SILENCE_SECONDS = 0.5
MIN_SPEECH_SECONDS = 0.1

# Measured in Confab on saved recordings: the detector reports resumed speech
# late, and 2.2 s bridges a thinking pause without the extra merging seen at 2.6.
SETTLE = 2.2
MAX_TURN_SECONDS = 24
OVERLAP_SECONDS = 3
BOUNDARY_HOLD_SECONDS = 1

PAD_LEADING, PAD_TRAILING, PAD_MAX_SECONDS = 0.25, 0.5, 8


class PcmWindow:
    """The last `capacity` samples, addressed in samples since the recording began."""

    def __init__(self, capacity: int = RATE * 32):
        self.data = np.zeros(capacity, dtype=np.float32)
        self.end = 0

    @property
    def start(self) -> int:
        return max(0, self.end - len(self.data))

    def push(self, samples: np.ndarray) -> None:
        n, cap = len(samples), len(self.data)
        if n >= cap:
            self.data[:] = samples[-cap:]
            self.data = np.roll(self.data, self.end + n)   # keep index = position % cap
        else:
            at = self.end % cap
            first = min(n, cap - at)
            self.data[at:at + first] = samples[:first]
            self.data[:n - first] = samples[first:]
        self.end += n

    def slice(self, start: float, end: float | None = None) -> np.ndarray:
        start = max(self.start, round(start))
        end = min(self.end, round(self.end if end is None else end))
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(start, end) % len(self.data)
        return self.data[idx].copy()


class Segmenter:
    """Silero VAD through sherpa-onnx. Times are seconds since the recording began."""

    def __init__(self, model_root: Path):
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(Path(model_root) / "silero-vad" / "silero_vad.onnx")
        config.silero_vad.threshold = VAD_THRESHOLD
        config.silero_vad.min_silence_duration = MIN_SILENCE_SECONDS
        config.silero_vad.min_speech_duration = MIN_SPEECH_SECONDS
        config.silero_vad.max_speech_duration = 15
        config.silero_vad.window_size = 512
        config.sample_rate = RATE
        config.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        self.input = PcmWindow()
        self.pre = round(PRE_SPEECH_SECONDS * RATE)
        self.delivered_end = 0
        self.next_id = 0
        self.open_since: int | None = None

    def _drain(self) -> list[dict]:
        events = []
        while not self.vad.empty():
            front = self.vad.front
            native_start, length = front.start, len(front.samples)
            self.vad.pop()
            end = min(self.input.end, native_start + length)
            if end <= self.delivered_end:
                continue
            # Keep audio from before the detected start, but never audio the
            # previous segment already had.
            begin = max(self.input.start, self.delivered_end, native_start - self.pre)
            self.delivered_end = end
            events.append({"type": "segment", "id": str(self.next_id),
                           "start": begin / RATE, "end": end / RATE})
            self.next_id += 1
            self.open_since = None
        # sherpa reports a segment's start only once it ends: while speech goes
        # on, report activity from an estimate so the turn stays open.
        if self.vad.is_speech_detected():
            if self.open_since is None:
                self.open_since = max(self.delivered_end,
                                      self.input.end - round((MIN_SPEECH_SECONDS + 0.1) * RATE) - self.pre)
            events.append({"type": "activity", "id": str(self.next_id),
                           "start": max(self.input.start, self.open_since) / RATE,
                           "end": self.input.end / RATE})
        return events

    def push(self, samples: np.ndarray) -> list[dict]:
        self.input.push(samples)
        self.vad.accept_waveform(samples)
        return self._drain()

    def flush(self) -> list[dict]:
        self.vad.flush()
        return [e for e in self._drain() if e["type"] == "segment"]


class _Turn:
    def __init__(self, n: int, start: int, audio_start: int, end: int, open_ids: set):
        self.id = f"turn-{n}"
        self.start, self.audio_start, self.end = start, audio_start, end
        self.open = open_ids


class ContextTurns:
    """Speech-detector events in, finished turns out: {"samples", "timing"}."""

    def __init__(self, settle: float = SETTLE, max_seconds: float = MAX_TURN_SECONDS,
                 overlap_seconds: float = OVERLAP_SECONDS):
        self.overlap = round(overlap_seconds * RATE)
        self.overlap_end: int | None = None
        self.input = PcmWindow()
        self.settle, self.max_seconds = settle, max_seconds
        self.sequence = 0
        self.cut = 0
        self.group: _Turn | None = None

    def _make(self, start: int, end: int, open_ids: set | None = None) -> _Turn:
        self.sequence += 1
        audio_start = (max(self.input.start, start - self.overlap)
                       if self.overlap_end == start else start)
        return _Turn(self.sequence, start, audio_start, end, set(open_ids or ()))

    def _finish(self, out: list, reason: str, end: int | None = None) -> None:
        g = self.group
        if not g:
            return
        self.group = None
        end = g.end if end is None else end
        if end <= g.start and g.audio_start == g.start:
            return
        out.append({"samples": self.input.slice(g.audio_start, end), "timing": {
            "start": g.audio_start / RATE, "end": end / RATE, "commit_start": g.start / RATE,
            "overlap_seconds": (g.start - g.audio_start) / RATE,
            "context_overlap_seconds": self.overlap / RATE, "reason": reason}})
        self.cut = end
        self.overlap_end = end if reason == "length-limit" else None

    def push(self, samples: np.ndarray, events: list[dict], flush: bool = False) -> list[dict]:
        self.input.push(samples)
        out: list = []
        limit = round(self.max_seconds * RATE)
        if self.group and self.group.audio_start < self.input.start:
            raise RuntimeError("Speech context exceeded the audio buffer; unfinished audio was not submitted")
        for e in events:
            start = max(self.cut, self.input.start, round(e["start"] * RATE))
            end = min(self.input.end, round(e["end"] * RATE))
            if end <= start:
                if e["type"] == "segment" and self.group:
                    self.group.open.discard(e["id"])
                continue
            # A pause inside an unsealed turn stays part of it.
            if not self.group:
                self.group = self._make(start, end)
            g = self.group
            g.end = max(g.end, end)
            if e["type"] == "segment":
                g.open.discard(e["id"])
            else:
                g.open.add(e["id"])
            while g.end - g.audio_start >= limit:
                boundary, tail_end, open_ids = g.audio_start + limit, g.end, set(g.open)
                self._finish(out, "length-limit", boundary)
                if tail_end == boundary and not self.overlap:
                    break
                self.group = g = self._make(boundary, tail_end, open_ids)
                if tail_end == boundary:
                    break
        g = self.group
        if g and (flush or (not g.open and self.input.end - g.end >= self.settle * RATE)):
            self._finish(out, "flush" if flush else "silence")
        return out


def pad(samples: np.ndarray, timing: dict) -> tuple[np.ndarray, int]:
    """Silence around a short finished phrase. Returns (samples, leading samples added)."""
    if (timing.get("overlap_seconds", 0) > 0 or timing.get("reason") == "length-limit"
            or not len(samples) or len(samples) > PAD_MAX_SECONDS * RATE):
        return samples, 0
    lead, trail = round(PAD_LEADING * RATE), round(PAD_TRAILING * RATE)
    return np.concatenate([np.zeros(lead, np.float32), samples, np.zeros(trail, np.float32)]), lead


def unpad(result: dict, lead: int, source_samples: int) -> dict:
    """Token times moved back onto the captured audio, so padding cannot shift
    which words an overlapped decode treats as already given."""
    if not lead:
        return result
    offset, end = lead / RATE, source_samples / RATE

    def clamp(t):
        return max(0.0, min(end, t - offset))
    times = result["timestamps"]
    return {**result, "timestamps": [clamp(t) for t in times],
            "durations": [clamp(t + d) - clamp(t) for t, d in zip(times, result["durations"])]}


def _key(text: str) -> str:
    text = text.strip().lower().replace("‘", "'").replace("’", "'")
    text = re.sub(r"^[\"'“”]+", "", text)
    return re.sub(r"[.,!?\"“”]+$", "", text)


def contextual_result(result: dict, context_seconds: float = 0, hold_seconds: float = 0,
                      audio_seconds: float = 0, overlap_seconds: float = OVERLAP_SECONDS,
                      previous_text: str = "") -> dict:
    """Commit stable words; reserve the boundary ending for the next decode.

    Exact suffix matching only locates the handoff; it cannot rewrite stable text.
    """
    text = result.get("text") or ""
    if not text or (not context_seconds and not hold_seconds):
        return {"text": text, "pending_text": "", "commit_seconds": audio_seconds}
    tokens, times, durations = result.get("tokens"), result.get("timestamps"), result.get("durations")
    if (not isinstance(tokens, list) or not isinstance(times, list) or not isinstance(durations, list)
            or not (len(tokens) == len(times) == len(durations))
            or any(not math.isfinite(d) or d < 0 for d in durations)
            or "".join(tokens).strip() != text.strip()
            or any(not math.isfinite(t) or t < 0 or (i and t < times[i - 1]) for i, t in enumerate(times))):
        raise ValueError("Overlapping transcription needs valid token timing")
    words: list[dict] = []
    for tok, t, d in zip(tokens, times, durations):
        if not words or tok[:1].isspace():
            words.append({"raw": tok, "start": t, "end": t + d})
        else:
            words[-1]["raw"] += tok
            words[-1]["end"] = max(words[-1]["end"], t + d)
    cut = next((i for i, w in enumerate(words) if w["start"] >= context_seconds), len(words))
    previous = [k for k in (_key(w) for w in previous_text.split()) if k] if context_seconds else []
    current = [_key(w["raw"]) for w in words]
    best = None
    # Exact suffix agreement near the timestamp boundary recovers a new word
    # whose timestamp drifts into the prefix, and avoids repeating the last
    # committed word when its timestamp drifts past it. Repeats away from the
    # boundary stay untouched.
    for end in range(1, len(words) + 1):
        if words[end - 1]["start"] > context_seconds + 0.4:
            continue
        for count in range(min(len(previous), end, 12), 1, -1):
            if previous[-count:] == current[end - count:end]:
                distance, before = abs(end - cut), words[end - 1]["start"] < context_seconds
                if (not best or count > best["count"]
                        or (count == best["count"] and ((before and not best["before"])
                            or (before == best["before"] and distance < best["distance"])))):
                    best = {"end": end, "count": count, "distance": distance, "before": before}
                break
    if best:
        cut = best["end"]
    stop = len(words)
    if hold_seconds:
        stop = next((i for i, w in enumerate(words) if w["end"] > audio_seconds - hold_seconds), len(words))
        # Never withhold the start of a long word that will lie outside the
        # next decoder's audio: keep that whole word now.
        while stop < len(words) and words[stop]["start"] < audio_seconds - overlap_seconds + 0.2:
            stop += 1
        stop = max(cut, stop)

    def render(ws):
        return "".join(w["raw"] for w in ws).strip()
    return {"text": render(words[cut:stop]), "pending_text": render(words[stop:]),
            "commit_seconds": words[stop]["start"] if stop < len(words) else audio_seconds}
