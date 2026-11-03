/* confab-stt browser client. One file, no dependencies, no user interface of its own: the page draws the mic button.
 * Vendored from confab-stt (MIT); in FitNotes Coach the server it talks to is src/dictation/service.py.
 *
 *   const stt = new ConfabSTT({url: '/stt'});
 *   stt.onstate = (state, detail) => ...   // 'idle' | 'connecting' | 'recording' | 'transcribing'
 *   stt.onsegment = text => ...            // each sentence as it is recognised, while the person is still speaking
 *   stt.onlevel = level => ...             // microphone level 0..1, about 10 times a second
 *   stt.onlimit = reason => ...            // 'max_duration' or 'no_audio': the recording stopped by itself
 *   stt.ondone = text => ...               // the whole text, however the recording ended
 *   stt.onerror = error => ...             // error.reason: 'busy', 'rate_limited', 'microphone', 'connection', ...
 *                                          // ('microphone' also has error.name, the browser's: NotAllowedError, ...)
 *   await stt.start();                     // asks for the microphone; call it from a click
 *   const text = await stt.stop();         // resolves with the whole text
 *   stt.cancel();                          // discards the recording
 *
 * For a live waveform, opt in with new ConfabSTT({url, analyser: true}): while the microphone is open, stt.analyser is
 * an AnalyserNode on the same audio (fftSize 1024, smoothingTimeConstant 0.6), null otherwise. Read it in an animation
 * loop with getByteTimeDomainData / getByteFrequencyData; it adds no second microphone stream.
 *
 * Audio is sent to the server as 16 kHz mono 16-bit PCM over a WebSocket while the person speaks.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ConfabSTT = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  // Runs on the audio thread: averages the microphone's samples down to 16 kHz (a box filter: enough for speech
  // recognition at 44.1 or 48 kHz input) and posts 100 ms of PCM16 at a time. 'flush' posts what is left.
  const WORKLET = `
class ConfabCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.step = sampleRate / 16000; this.pos = 0; this.sum = 0; this.n = 0; this.out = new Int16Array(1600); this.o = 0;
    this.port.onmessage = e => { if (e.data === 'flush') { this.post(); this.port.postMessage('flushed'); } };
  }
  post() { if (!this.o) return; const b = this.out.slice(0, this.o).buffer; this.port.postMessage(b, [b]); this.o = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i]; this.n++; this.pos++;
      if (this.pos >= this.step) {
        this.pos -= this.step; const v = Math.max(-1, Math.min(1, this.sum / this.n)); this.sum = 0; this.n = 0;
        this.out[this.o++] = Math.round(v * 32767);
        if (this.o === this.out.length) this.post();
      }
    }
    return true;
  }
}
registerProcessor('confab-capture', ConfabCapture);`;

  function wsUrl(url) {
    const u = new URL(url, location.href);
    if (u.protocol === 'http:') u.protocol = 'ws:'; else if (u.protocol === 'https:') u.protocol = 'wss:';
    return u.href;
  }
  function err(reason, message, extra) { return Object.assign(new Error(message), {reason}, extra); }
  // reason 'microphone'; error.name is the browser's (NotAllowedError, NotFoundError, NotReadableError, ...).
  function micError(e) {
    const message = e.name === 'NotAllowedError' || e.name === 'SecurityError' ? 'Microphone access was blocked. Allow it in the browser to dictate.'
      : e.name === 'NotFoundError' || e.name === 'OverconstrainedError' ? 'No microphone was found on this device.'
      : e.name === 'NotReadableError' || e.name === 'AbortError' ? 'The microphone could not be started. Another app may be using it.'
      : 'No microphone could be opened: ' + e.message;
    return err('microphone', message, {name: e.name});
  }

  class ConfabSTT {
    static isSupported() {
      return typeof window !== 'undefined' && !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia &&
        window.AudioContext && window.AudioWorkletNode && window.WebSocket);
    }
    // url: the WebSocket address, absolute or relative to the page ('/stt' by default).
    // audio: getUserMedia audio constraints. analyser: true exposes stt.analyser while recording.
    constructor({url = '/stt', audio = {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}, analyser = false} = {}) {
      this.url = url; this.audio = audio; this.state = 'idle'; this.wantAnalyser = analyser; this.analyser = null;
      this.onstate = this.onsegment = this.onlevel = this.onlimit = this.ondone = this.onerror = null;
      this._run = null;
    }
    _set(state, detail) { this.state = state; this.onstate && this.onstate(state, detail || {}); }

    async start() {
      if (this.state !== 'idle') throw err('already_started', 'Already recording.');
      if (!ConfabSTT.isSupported()) throw this._fail(null, err('unsupported', 'This browser cannot record audio here (it needs a secure https:// page).'));
      // The AudioContext is made first, inside the click, so the browser lets it run.
      const run = this._run = {ctx: new AudioContext(), pending: [], started: false, segments: [], stopped: null};
      this._set('connecting');
      try {
        // The microphone first, so the browser's permission prompt is the first thing the person sees and nothing
        // else can fail while it is open.
        const stream = run.stream = await navigator.mediaDevices.getUserMedia({audio: this.audio}).catch(e => { throw micError(e); });
        if (this._run !== run) { stream.getTracks().forEach(t => t.stop()); return; }
        await Promise.all([
          this._connect(run),
          run.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type: 'text/javascript'})))
        ]);
        if (this._run !== run) { stream.getTracks().forEach(t => t.stop()); return; }
        await run.ctx.resume();
        run.source = run.ctx.createMediaStreamSource(stream);
        if (this.wantAnalyser) {
          run.analyser = run.ctx.createAnalyser(); run.analyser.fftSize = 1024; run.analyser.smoothingTimeConstant = 0.6;
          run.source.connect(run.analyser); this.analyser = run.analyser;
        }
        run.node = new AudioWorkletNode(run.ctx, 'confab-capture');
        run.node.port.onmessage = e => {
          if (e.data === 'flushed') { run.flushed && run.flushed(); return; }
          this._level(new Int16Array(e.data));
          if (run.started && run.ws.readyState === 1) run.ws.send(e.data); else run.pending.push(e.data);
        };
        run.source.connect(run.node);
        // Some browsers only pull audio through a node that reaches the output; a muted gain keeps it silent.
        const mute = run.ctx.createGain(); mute.gain.value = 0; run.node.connect(mute).connect(run.ctx.destination);
        run.ws.send(JSON.stringify({type: 'start'}));
        await run.startedPromise;
        run.started = true;
        for (const b of run.pending.splice(0)) run.ws.send(b);
        this._set('recording', {loading: run.loading});
      } catch (e) {
        if (this._run !== run) { this._release(run, true); return; }// cancelled while starting
        throw this._fail(run, e);
      }
    }

    _connect(run) {
      return new Promise((resolve, reject) => {
        const ws = run.ws = new WebSocket(wsUrl(this.url)); ws.binaryType = 'arraybuffer';
        run.startedPromise = new Promise((ok, no) => { run.startedOk = ok; run.startedNo = no; });
        run.startedPromise.catch(() => {});
        ws.onopen = () => resolve();
        ws.onerror = () => reject(err('connection', 'Could not reach the dictation service.'));
        ws.onclose = () => { if (this._run === run && this.state !== 'idle') this._fail(run, err('connection', 'The connection to the dictation service closed.')); };
        ws.onmessage = e => this._message(run, JSON.parse(e.data));
      });
    }

    _message(run, m) {
      if (this._run !== run) return;
      if (m.type === 'started') { run.loading = m.loading; run.startedOk(); }
      else if (m.type === 'loaded') { if (this.state === 'recording') this._set('recording', {loading: false}); }
      else if (m.type === 'final') { run.segments.push(m.text); this.onsegment && this.onsegment(m.text); }
      else if (m.type === 'limit') { this._release(run, false); this._set('transcribing', {reason: m.reason}); this.onlimit && this.onlimit(m.reason); }
      else if (m.type === 'done') {
        const text = m.text; this._release(run, true); this._run = null; this._set('idle');
        run.stopped && run.stopped.resolve(text); this.ondone && this.ondone(text);
      }
      else if (m.type === 'busy') run.startedNo(err('busy', m.message));
      else if (m.type === 'error') {
        const e = err(m.reason || 'error', m.message, m.retryAfterSeconds ? {retryAfterSeconds: m.retryAfterSeconds} : undefined);
        if (!run.started) run.startedNo(e); else this._fail(run, e);
      }
    }

    // Stops the microphone, sends what the audio thread still holds, then asks for the text.
    stop() {
      const run = this._run;
      if (!run) return Promise.resolve('');
      if (run.stopped) return run.stopped.promise;
      let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; });
      promise.catch(() => {});// onerror reports the failure too; a caller that ignores this promise sees no unhandled rejection
      run.stopped = {promise, resolve, reject};
      if (!run.started) { this.cancel(); resolve(''); return promise; }
      if (this.state === 'recording') {
        this._set('transcribing');
        const send = () => { if (run.ws.readyState === 1) run.ws.send(JSON.stringify({type: 'stop'})); this._release(run, false); };
        run.flushed = send; run.node.port.postMessage('flush');
        setTimeout(() => { if (run.flushed === send) { run.flushed = null; send(); } }, 500);
      }
      return promise;
    }

    cancel() {
      const run = this._run; if (!run) return;
      this._run = null;
      if (run.ws && run.ws.readyState === 1) run.ws.send(JSON.stringify({type: 'cancel'}));
      this._release(run, true);
      run.stopped && run.stopped.resolve('');
      this._set('idle');
    }

    _level(pcm) {
      if (!this.onlevel) return;
      let sum = 0; for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
      this.onlevel(Math.min(1, Math.sqrt(sum / Math.max(1, pcm.length)) / 8000));
    }

    // Microphone and audio graph off; the socket too when closeSocket.
    _release(run, closeSocket) {
      if (run.flushed) run.flushed = null;
      try { run.source && run.source.disconnect(); run.node && run.node.disconnect(); run.analyser && run.analyser.disconnect(); } catch (e) {}
      if (run.analyser && this.analyser === run.analyser) this.analyser = null;
      run.stream && run.stream.getTracks().forEach(t => t.stop());
      if (run.ctx && run.ctx.state !== 'closed') run.ctx.close().catch(() => {});
      if (closeSocket && run.ws && run.ws.readyState <= 1) { run.ws.onclose = null; run.ws.close(); }
    }

    _fail(run, e) {
      if (run) { this._release(run, true); if (this._run === run) this._run = null; run.stopped && run.stopped.reject(e); }
      this._set('idle', {error: e});
      this.onerror && this.onerror(e);
      return e;
    }
  }
  return ConfabSTT;
});
