"""
Dictation (src/dictation): the /stt protocol, its limits and access rules, the
turn and overlap logic, the model downloader, and, when the speech models are
present, real recognition of tests/fixtures/speech-sample.wav ("Sitara, start
the next set.", spoken by Confab's Kokoro voice).

The protocol tests use a stand-in engine, so they need no models. Ported from
confab-stt's test/server.test.cjs and test/recognition.test.cjs; the Python
port was also checked word for word against the Node original on the same
recordings, including continuous speech past the 24 s cut.
"""

import asyncio
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dictation import turns  # noqa: E402
from src.dictation.engine import LazyEngine  # noqa: E402
from src.dictation.service import DictationService, Quota, origin_allowed  # noqa: E402

RATE = 16000
FIXTURE = _ROOT / "tests" / "fixtures" / "speech-sample.wav"


# ── The protocol, with a stand-in engine ─────────────────────────────────────

class FakeEngine:
    """One "final" per second of audio received, then done shortly after end()."""

    def __init__(self, on_state):
        self.state = "loaded"
        self.samples = 0
        self.cancelled = False

    def start(self, capture_id, emit):
        self.id, self.emit, self.samples, self.said = capture_id, emit, 0, 0

    def audio(self, capture_id, pcm):
        self.samples += len(pcm) // 2
        while self.samples >= RATE * (self.said + 1):
            self.said += 1
            self.emit({"type": "final", "text": f"word{self.said}", "id": capture_id})

    def end(self, capture_id):
        asyncio.get_running_loop().call_later(0.01, lambda: self.emit({"type": "done", "id": capture_id}))

    def cancel(self, capture_id):
        self.cancelled = True

    def close(self):
        pass


def _serve(**options):
    engines = []

    def make(on_state):
        e = FakeEngine(on_state)
        engines.append(e)
        return e
    service = DictationService(make_engine=make, **options)
    app = FastAPI()

    @app.websocket("/stt")
    async def stt(ws: WebSocket):
        await service.handle(ws, ws.headers.get("x-test-ip", "203.0.113.7"))
    return TestClient(app), service, engines


def _next(ws, kind):
    while True:
        m = json.loads(ws.receive_text())
        if m["type"] == kind:
            return m


def _pcm(seconds):
    return bytes(round(seconds * RATE) * 2)


ORIGIN = {"origin": "http://testserver"}


def test_each_sentence_arrives_as_recognised_then_the_whole_text():
    client, _, _ = _serve()
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        started = _next(ws, "started")
        assert started["sampleRate"] == 16000 and started["maxSeconds"] == 600
        ws.send_bytes(_pcm(1))
        assert _next(ws, "final")["text"] == "word1"
        ws.send_bytes(_pcm(1))
        ws.send_text(json.dumps({"type": "stop"}))
        assert _next(ws, "done")["text"] == "word1 word2"
        ws.send_text(json.dumps({"type": "start"}))      # the connection can record again
        _next(ws, "started")


def test_a_second_person_is_told_it_is_busy_and_can_record_once_free():
    client, _, _ = _serve()
    with client.websocket_connect("/stt", headers=ORIGIN) as a, \
            client.websocket_connect("/stt", headers={**ORIGIN, "x-test-ip": "198.51.100.2"}) as b:
        a.send_text(json.dumps({"type": "start"}))
        _next(a, "started")
        b.send_text(json.dumps({"type": "start"}))
        assert "Someone else is dictating" in _next(b, "busy")["message"]
        a.send_text(json.dumps({"type": "stop"}))
        _next(a, "done")
        b.send_text(json.dumps({"type": "start"}))
        _next(b, "started")


def test_a_recording_stops_at_the_time_limit_and_still_returns_its_text():
    client, _, engines = _serve(max_seconds=2)
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
        ws.send_bytes(_pcm(1.5))
        ws.send_bytes(_pcm(1.5))
        assert _next(ws, "limit")["reason"] == "max_duration"
        assert _next(ws, "done")["text"] == "word1 word2"
    assert engines[0].samples == 32000                   # audio past the limit is not recognised


def test_a_browser_that_stops_sending_audio_has_its_recording_finished():
    client, _, _ = _serve(silent_stop_seconds=0.2)
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
        assert _next(ws, "limit")["reason"] == "no_audio"
        _next(ws, "done")


def test_cancel_and_a_closed_connection_free_the_recogniser():
    client, service, engines = _serve()
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
        ws.send_bytes(_pcm(0.5))
        ws.send_text(json.dumps({"type": "cancel"}))
        _next(ws, "cancelled")
        assert engines[0].cancelled
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
    assert service.health()["slots"][0]["busy"] is False
    assert service.total == 0 and service.connections == {}


def test_starts_are_rate_limited_per_visitor():
    client, _, _ = _serve(starts_per_minute=2)
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        for _ in range(2):
            ws.send_text(json.dumps({"type": "start"}))
            _next(ws, "started")
            ws.send_text(json.dumps({"type": "cancel"}))
            _next(ws, "cancelled")
        ws.send_text(json.dumps({"type": "start"}))
        e = _next(ws, "error")
        assert e["reason"] == "rate_limited" and e["retryAfterSeconds"] > 0


def test_odd_length_audio_is_refused():
    client, _, _ = _serve()
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
        ws.send_bytes(b"\x00\x00\x00")
        assert _next(ws, "error")["reason"] == "bad_audio"


def test_another_website_cannot_open_a_connection_from_a_visitors_browser():
    client, _, _ = _serve()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/stt", headers={"origin": "https://evil.example"}):
            pass
    with client.websocket_connect("/stt", headers=ORIGIN):
        pass
    client, _, _ = _serve(origins=["https://coach.example"])
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/stt", headers=ORIGIN):
            pass
    with client.websocket_connect("/stt", headers={"origin": "https://coach.example"}):
        pass


def test_connections_per_visitor_are_capped():
    client, _, _ = _serve(connections_per_ip=1)
    with client.websocket_connect("/stt", headers=ORIGIN):
        with client.websocket_connect("/stt", headers=ORIGIN) as second:
            assert _next(second, "error")["reason"] == "too_many_connections"


def test_without_models_the_server_says_so_instead_of_recording(tmp_path):
    service = DictationService(model_root=tmp_path)
    assert "speech models are missing" in service.unavailable()
    app = FastAPI()

    @app.websocket("/stt")
    async def stt(ws: WebSocket):
        await service.handle(ws, "203.0.113.7")
    with TestClient(app).websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        assert _next(ws, "error")["reason"] == "unavailable"


def test_helpers_quota_windows_and_origins():
    now = [0.0]
    q = Quota(2, 1000, clock=lambda: now[0])
    assert (q.take("a"), q.take("a"), q.take("a")) == (0, 0, 1000)
    now[0] = 1000
    assert q.take("a") == 0
    assert origin_allowed({"origin": "https://a.example", "host": "a.example"}, None)
    assert not origin_allowed({"origin": "https://b.example", "host": "a.example"}, None)
    assert origin_allowed({"origin": "https://a.example", "host": "app:3000",
                           "x-forwarded-host": "a.example"}, None)
    assert origin_allowed({}, None)                       # not a browser


# ── The chat UI's server serves it ───────────────────────────────────────────

def test_the_chat_ui_serves_dictation_on_its_own_port(monkeypatch):
    monkeypatch.setenv("OPEN_BROWSER", "0")              # its startup hook opens a tab otherwise
    name = "fitnotes_frontend_server"
    fe = sys.modules.get(name)
    if fe is None:
        spec = importlib.util.spec_from_file_location(name, _ROOT / "frontend" / "server.py")
        fe = importlib.util.module_from_spec(spec)
        sys.modules[name] = fe
        spec.loader.exec_module(fe)
    monkeypatch.setattr(fe, "_dictation", DictationService(make_engine=FakeEngine))
    client = TestClient(fe.app)
    assert client.get("/stt-status").json() == {"available": True, "reason": None}
    js = client.get("/vendor/confab-stt.js")
    assert js.status_code == 200 and "class ConfabSTT" in js.text and "analyser" in js.text
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _next(ws, "started")
        ws.send_bytes(_pcm(1))
        assert _next(ws, "final")["text"] == "word1"


# ── Turns and overlap ────────────────────────────────────────────────────────

def _speak(ct, seconds, step=1.0):
    """Continuous speech: the detector reports activity covering everything so far."""
    out = []
    for _ in range(round(seconds / step)):
        samples = np.zeros(round(step * RATE), np.float32)
        end = (ct.input.end + len(samples)) / RATE
        out += ct.push(samples, [{"type": "activity", "id": "0", "start": 0, "end": end}])
    return out


def test_long_speech_is_cut_at_24_s_and_the_next_piece_starts_3_s_earlier():
    ct = turns.ContextTurns()
    pieces = _speak(ct, 30)
    assert [p["timing"]["reason"] for p in pieces] == ["length-limit"]
    first = pieces[0]["timing"]
    assert (first["start"], first["end"]) == (0, 24) and len(pieces[0]["samples"]) == 24 * RATE
    last = ct.push(np.zeros(0, np.float32), [], flush=True)[0]["timing"]
    assert (last["start"], last["commit_start"], last["end"], last["overlap_seconds"]) == (21, 24, 30, 3)


def test_a_turn_ends_after_the_settle_silence():
    ct = turns.ContextTurns()
    out = ct.push(np.zeros(2 * RATE, np.float32), [{"type": "segment", "id": "0", "start": 0, "end": 2}])
    assert out == []
    out = ct.push(np.zeros(round(2.2 * RATE), np.float32), [])
    assert [p["timing"]["reason"] for p in out] == ["silence"]


def _result(words, step=1.0):
    tokens = [" " + w for w in words]
    return {"text": "".join(tokens).strip(), "tokens": tokens,
            "timestamps": [i * step for i in range(len(words))], "durations": [step / 2] * len(words)}


def test_overlap_drops_words_already_given_and_holds_the_cut_ending():
    r = _result(["one", "two", "three", "four", "five"])
    out = turns.contextual_result(r, context_seconds=2.0, previous_text="zero one two")
    assert out["text"] == "three four five"
    held = turns.contextual_result(r, hold_seconds=1, audio_seconds=5)
    assert (held["text"], held["pending_text"], held["commit_seconds"]) == ("one two three four", "five", 4.0)
    plain = turns.contextual_result(r)
    assert plain["text"] == "one two three four five"
    with pytest.raises(ValueError):
        turns.contextual_result({**r, "durations": []}, context_seconds=1)


def test_short_phrases_are_padded_and_token_times_moved_back():
    samples = np.ones(RATE, np.float32)
    padded, lead = turns.pad(samples, {"overlap_seconds": 0, "reason": "silence"})
    assert lead == 4000 and len(padded) == RATE + 4000 + 8000
    assert turns.pad(samples, {"overlap_seconds": 3, "reason": "silence"})[1] == 0
    moved = turns.unpad({"timestamps": [0.25, 0.75], "durations": [0.5, 0.5]}, lead, RATE)
    assert moved["timestamps"] == [0.0, 0.5]


def test_the_pcm_window_keeps_the_newest_audio_in_place():
    w = turns.PcmWindow(capacity=10)
    w.push(np.arange(7, dtype=np.float32))
    w.push(np.arange(7, 15, dtype=np.float32))
    assert w.start == 5 and list(w.slice(5, 15)) == list(range(5, 15))
    w.push(np.arange(15, 40, dtype=np.float32))           # more than the capacity at once
    assert list(w.slice(30, 40)) == list(range(30, 40))


# ── The engine: lazy loading ─────────────────────────────────────────────────

class FakeWorker:
    made = []

    def __init__(self, model_root, on_event, log):
        self.on_event, self.sent, self.audio_in = on_event, [], []
        self.ready = asyncio.get_running_loop().create_future()
        self.closed = False
        FakeWorker.made.append(self)

    def send(self, m):
        self.sent.append(m)

    def audio(self, pcm):
        self.audio_in.append(pcm)

    def close(self):
        self.closed = True


def test_audio_said_while_the_model_loads_is_kept_and_the_idle_model_unloads():
    async def scenario():
        FakeWorker.made.clear()
        states, events = [], []
        eng = LazyEngine(Path("."), idle_seconds=0.05, on_state=states.append, make_worker=FakeWorker)
        eng.start("rec1", events.append)
        eng.audio("rec1", b"\x01\x00" * 10)               # before the model is ready
        eng.end("rec1")
        while not FakeWorker.made:
            await asyncio.sleep(0)
        worker = FakeWorker.made[0]
        assert eng.state == "loading" and worker.sent == []
        worker.ready.set_result({"type": "ready", "load_ms": 1})
        await asyncio.sleep(0.01)
        assert [m["type"] for m in worker.sent] == ["open", "close"] and worker.audio_in == [b"\x01\x00" * 10]
        eng._event({"type": "final", "id": "rec1", "text": "hi"})
        eng._event({"type": "done", "id": "rec1"})
        assert [e["type"] for e in events] == ["final", "done"] and eng.capture is None
        await asyncio.sleep(0.2)
        assert worker.closed and eng.state == "unloaded"
        assert states == ["loading", "loaded", "unloaded"]
        eng.close()
    asyncio.run(scenario())


# ── The model downloader ─────────────────────────────────────────────────────

def _load_fetcher():
    spec = importlib.util.spec_from_file_location("fetch_speech_models", _ROOT / "scripts" / "fetch_speech_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_downloader_checks_unpacks_and_places_only_listed_files(tmp_path, monkeypatch):
    f = _load_fetcher()
    model, extra = b"weights" * 100, b"licence text"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in (("pack/model.onnx", model), ("pack/unlisted.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    src = tmp_path / "src"
    src.mkdir()
    (src / "pack.tar.bz2").write_bytes(buf.getvalue())
    (src / "LICENSE").write_bytes(extra)
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    manifest = {"pieces": [{"name": "Test model", "path": "test-model", "totalBytes": len(model) + len(extra),
                            "download": {"archive": {"url": (src / "pack.tar.bz2").as_uri(),
                                                     "size": len(buf.getvalue()), "sha256": sha(buf.getvalue()),
                                                     "folder": "pack"},
                                         "extra": {"LICENSE": (src / "LICENSE").as_uri()}},
                            "files": [{"path": "model.onnx", "size": len(model), "sha256": sha(model)},
                                      {"path": "LICENSE", "size": len(extra), "sha256": sha(extra)}]}]}
    (tmp_path / "models.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(f, "MANIFEST", tmp_path / "models.json")
    root = tmp_path / "models"
    assert f.main([str(root), "--check"]) == 1
    assert f.main([str(root)]) == 0
    assert (root / "test-model" / "model.onnx").read_bytes() == model
    assert not (root / "test-model" / "unlisted.txt").exists() and not (root / ".downloads").exists()
    assert f.main([str(root), "--check"]) == 0
    (root / "test-model" / "model.onnx").write_bytes(b"damaged")
    assert f.main([str(root), "--check"]) == 1


def test_the_real_manifest_lists_both_models_with_checksums():
    manifest = json.loads((_ROOT / "src" / "dictation" / "models.json").read_text(encoding="utf-8"))
    paths = {p["path"] for p in manifest["pieces"]}
    assert paths == {"sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "silero-vad"}
    for piece in manifest["pieces"]:
        for f in piece["files"]:
            assert len(f["sha256"]) == 64 and f["size"] > 0


# ── Real recognition (skipped without the models) ────────────────────────────

MODELS = Path(os.environ.get("SPEECH_MODELS") or _ROOT / "models" / "speech")


def _fixture():
    with wave.open(str(FIXTURE)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _dictate(client, pcm):
    with client.websocket_connect("/stt", headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"type": "start"}))
        events = [_next(ws, "started")]
        data = pcm.astype("<i2").tobytes()
        for i in range(0, len(data), 3200):
            ws.send_bytes(data[i:i + 3200])
        ws.send_text(json.dumps({"type": "stop"}))
        while events[-1]["type"] not in ("done", "error"):
            events.append(json.loads(ws.receive_text()))
        return events


@pytest.mark.skipif(not (MODELS / "silero-vad" / "silero_vad.onnx").is_file(),
                    reason=f"speech models not found in {MODELS} (python scripts/fetch_speech_models.py)")
def test_dictation_with_the_real_models():
    service = DictationService(model_root=MODELS)
    app = FastAPI()

    @app.websocket("/stt")
    async def stt(ws: WebSocket):
        await service.handle(ws, "203.0.113.7")
    try:
        with TestClient(app) as client:
            x = _fixture()
            first = _dictate(client, x)
            assert first[0]["loading"] is True
            assert first[-1]["type"] == "done" and "start the next set" in first[-1]["text"].lower()
            # Twice with a pause between: two sentences, both in the final text.
            pause = np.zeros(3 * RATE, np.int16)
            second = _dictate(client, np.concatenate([x, pause, x]))
            assert [e["type"] for e in second].count("final") == 2
            assert second[-1]["text"].lower().count("start the next set") == 2
            # 40 s with only short gaps: crosses the 24 s cut; nothing lost, nothing doubled.
            gap = np.zeros(int(0.3 * RATE), np.int16)
            long = np.concatenate([np.concatenate([x, gap]) for _ in range(18)])
            third = _dictate(client, long)
            assert third[-1]["text"].lower().count("start the next set") == 18, third[-1]["text"]
    finally:
        service.close()
