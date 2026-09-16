"""
The chat page after a reload, and when the coach is busy (live re-check, 2026-09-16).

WHY THIS EXISTS. Reloading while the coach was answering lost the turn: the page
restored finished history, unlocked the input, and never learned an answer was
on its way. A message typed then was rejected as busy with nothing said, and the
typed text was gone. A busy reply to Confirm hid the panel and added an EMPTY
coach bubble, so the confirmation was lost with no way to retry.

The server now records a turn the moment it starts (a blank assistant entry
marked `pending`, see tests/test_turn_history.py) and can restore a waiting
confirmation panel. These tests hold the page to using that: show "…" and keep
the input locked until the answer arrives, restore both panels, and say plainly
when a request was not sent.

The REAL functions are extracted from frontend/index.html and run under node,
with the page's helpers stubbed and a scripted fetch — the approach of
test_chat_rendering.py. Skipped, with a reason, when node is not installed.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed — page tests skipped")

_MD_PARTS = (r"function esc\(s\)[\s\S]*?\n\}", r"const _ROW_RE[\s\S]*?\n  \}\n\}")

# Stubs for everything the functions under test touch. `log` is what the tests read.
_PRELUDE = r"""
const API = '';
let busy = false, lastQuestion = null, autoRetryTimer = null;
let disambigGroups = [{name: 'row', candidates: ['Barbell Row']}];
const log = {msgs: [], busy: [], thinking: 0, fetches: [], unlockAfterHistory: [],
             disambig: [], confirmPreview: [], rateLimited: 0, welcome: 0};
function mkEl(id) {
  const hidden = new Set(['hidden']);
  return {id, value: '', textContent: '', innerHTML: '', style: {}, disabled: false, focus() {},
          classList: {add: c => hidden.add(c), remove: c => hidden.delete(c),
                      contains: c => hidden.has(c)}};
}
const els = {};
const document = {getElementById: id => (els[id] = els[id] || mkEl(id))};
function chatInner() { return {set innerHTML(v) { log.msgs = []; log.thinking = 0; }}; }
function addMsg(role, html) {
  const rec = {role, html, removed: false};
  log.msgs.push(rec);
  return {parentElement: {parentElement: {remove() { rec.removed = true; }}}};
}
function addThinking() { log.thinking++; }
function removeThinking() { if (log.thinking) log.thinking--; }
function setBusy(b) {
  busy = b; log.busy.push(b);
  if (!b) log.unlockAfterHistory.push(log.fetches.filter(f => f === '/history').length);
}
function addWelcomeMessage() { log.welcome++; }
function renderDisambiguation(g) { log.disambig.push(g); setBusy(true); }
function renderConfirmPreview(p) { log.confirmPreview.push(p); }
function handleRateLimitError(d) { log.rateLimited++; setBusy(false); }
function handleOverloadError() {}
function showToast() {}
function _collectSelections() { return []; }
function renderResumeNotice(message) { (log.resume = log.resume || []).push(message); }
function setTimeout(fn) { fn(); }
const SCRIPT = JSON.parse(process.argv[1]).script;
async function fetch(url) {
  const path = url.replace(API, '');
  log.fetches.push(path);
  const q = SCRIPT[path];
  if (!q || !q.length) throw new Error('unscripted ' + path);
  const r = q.length > 1 ? q.shift() : q[0];
  if (r.fail) throw new Error('server down');
  return {status: r.status || 200, json: async () => r.body};
}
"""


def _fn(src: str, name: str) -> str:
    m = re.search(r"(?:async )?function " + re.escape(name) + r"\([^)]*\)[\s\S]*?\n\}", src)
    return m.group(0) if m else f"/* {name} missing from index.html */"


def _run(functions: list, call: str, script: dict, setup: str = "") -> dict:
    src = INDEX.read_text(encoding="utf-8")
    # The shared busy helper every request uses (absent on an old page → the
    # tests fail on behaviour, not on a harness error).
    busy_text = re.search(r"const BUSY_TEXT = [^\n]*", src)
    helpers = (busy_text.group(0) if busy_text else "") + "\n" + (
        _fn(src, "isBusy") if "function isBusy(" in src else "")
    code = ("\n".join(re.search(p, src).group(0) for p in _MD_PARTS) + "\n"
            + _PRELUDE + helpers + "\n" + "\n".join(_fn(src, f) for f in functions) + "\n"
            + setup + "\n"
            + "(async () => {\n  let result;\n  try { result = await (" + call + "); }\n"
            + "  catch (e) { log.error = String(e); }\n"
            + "  process.stdout.write(JSON.stringify({log, busy, result,\n"
            + "    input: document.getElementById('input').value,\n"
            + "    confirmHidden: document.getElementById('confirm-bar').classList.contains('hidden')}));\n"
            + "})();")
    out = subprocess.run([NODE, "-e", code, json.dumps({"script": script})],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _live(msgs):
    return [(m["role"], m["html"]) for m in msgs if not m["removed"]]


READY = {"/status": [{"body": {"ready": True}}],
         "/pending-disambiguation": [{"body": {"groups": None}}],
         "/pending-confirmation": [{"body": {}}]}
PROMPT = {"role": "user", "text": "plan my week"}
PENDING = {"role": "assistant", "text": "", "pending": True}
ANSWER = {"role": "assistant", "text": "### Your week"}
BUSY = {"status": 429, "body": {"error": "Agent is busy, please wait"}}


# ── restoreHistory draws every kind of entry ─────────────────────────────────

def test_restore_shows_a_turn_in_progress_as_thinking():
    r = _run(["restoreHistory"], "restoreHistory(" + json.dumps([PROMPT, PENDING]) + ")", {})
    assert r["result"] is True
    assert r["log"]["thinking"] == 1
    assert _live(r["log"]["msgs"]) == [("user", "plan my week")]      # no empty bubble


def test_restore_draws_errors_as_errors_and_answers_as_markdown():
    hist = [{"role": "user", "text": "hi"},
            {"role": "assistant", "text": "Rate <b>limit</b>", "kind": "error"},
            ANSWER]
    r = _run(["restoreHistory"], "restoreHistory(" + json.dumps(hist) + ")", {})
    assert r["result"] is False
    assert _live(r["log"]["msgs"]) == [
        ("user", "hi"),
        ("error", "Rate &lt;b&gt;limit&lt;/b&gt;"),
        ("agent", '<span class="md-h">Your week</span>')]


QUOTA_STOP = {"role": "assistant", "text": "Daily limit reached — your progress is saved.",
              "kind": "error", "resumable": True}


def test_a_reload_after_a_resumable_quota_stop_offers_resume_again():
    """Live, this showed a Resume button; after a reload it became a plain error
    and the saved question could not be finished."""
    r = _run(["restoreHistory"], "restoreHistory(" + json.dumps([PROMPT, QUOTA_STOP]) + ")", {})
    assert r["log"].get("resume") == ["Daily limit reached — your progress is saved."]
    assert not [m for m in _live(r["log"]["msgs"]) if m[0] == "error"]


def test_an_older_quota_stop_gets_no_stale_resume_button():
    hist = [PROMPT, QUOTA_STOP, {"role": "user", "text": "later"}, ANSWER]
    r = _run(["restoreHistory"], "restoreHistory(" + json.dumps(hist) + ")", {})
    assert not r["log"].get("resume")
    assert [m for m in _live(r["log"]["msgs"]) if m[0] == "error"]


# ── A reload mid-answer waits for the answer ─────────────────────────────────

def test_a_reload_mid_answer_keeps_the_input_locked_until_the_answer_arrives():
    script = dict(READY, **{"/history": [{"body": {"history": [PROMPT, PENDING]}},
                                         {"body": {"history": [PROMPT, PENDING]}},
                                         {"body": {"history": [PROMPT, ANSWER]}}]})
    r = _run(["restoreHistory", "waitForReady"], "waitForReady()", script)
    assert r["log"]["fetches"].count("/history") == 3
    assert r["log"]["unlockAfterHistory"] and min(r["log"]["unlockAfterHistory"]) == 3, \
        "the input was unlocked before the answer arrived"
    assert _live(r["log"]["msgs"]) == [("user", "plan my week"),
                                       ("agent", '<span class="md-h">Your week</span>')]
    assert r["log"]["thinking"] == 0 and r["busy"] is False


def test_a_reload_restores_a_waiting_confirmation_panel():
    script = dict(READY, **{"/history": [{"body": {"history": [PROMPT]}}],
                            "/pending-confirmation": [{"body": {"preview": "Row 60kg x 8",
                                                                "preview_source": "slot"}}]})
    r = _run(["restoreHistory", "waitForReady"], "waitForReady()", script)
    assert r["log"]["confirmPreview"] == ["Row 60kg x 8"]
    assert r["confirmHidden"] is False


def test_losing_the_server_while_waiting_says_so_and_unlocks():
    script = dict(READY, **{"/history": [{"body": {"history": [PROMPT, PENDING]}},
                                         {"fail": True}]})
    r = _run(["restoreHistory", "waitForReady"], "waitForReady()", script)
    errors = [h for role, h in _live(r["log"]["msgs"]) if role == "error"]
    assert errors and "Lost contact" in errors[0]
    assert r["log"]["thinking"] == 0 and r["busy"] is False


# ── Busy: say the request was not sent, and lose nothing ─────────────────────

def _busy_message(r):
    return [h for role, h in _live(r["log"]["msgs"]) if role == "error" and "still answering" in h]


def test_busy_on_send_puts_the_text_back_and_says_so():
    r = _run(["sendMessage"], "sendMessage()", {"/chat": [BUSY]},
             setup="document.getElementById('input').value = 'typed after reload';")
    assert _busy_message(r)
    assert not [m for m in _live(r["log"]["msgs"]) if m[0] == "user"], "the unsent bubble stayed"
    assert r["input"] == "typed after reload"
    assert r["busy"] is False


def test_busy_on_confirm_keeps_the_panel():
    r = _run(["sendConfirm"], "sendConfirm(true)", {"/confirm": [BUSY]},
             setup="document.getElementById('confirm-bar').classList.remove('hidden');")
    assert _busy_message(r)
    assert r["confirmHidden"] is False, "the confirm panel was lost"
    assert not [m for m in _live(r["log"]["msgs"]) if m[0] == "agent"], "an empty coach bubble"


def test_busy_on_disambiguation_keeps_the_panel():
    r = _run(["submitDisambiguation"], "submitDisambiguation()", {"/disambiguate": [BUSY]})
    assert _busy_message(r)
    assert r["log"]["disambig"] == [[{"name": "row", "candidates": ["Barbell Row"]}]]
    assert not [m for m in _live(r["log"]["msgs"]) if m[0] == "agent"]


def test_busy_on_resume_says_so():
    r = _run(["resumeSavedQuestion"], "resumeSavedQuestion()", {"/resume": [BUSY]})
    assert "error" not in r["log"], r["log"].get("error")
    assert _busy_message(r)
    assert not [m for m in _live(r["log"]["msgs"]) if m[0] == "agent"]


# ── Controls: what must not change ───────────────────────────────────────────

def test_a_quota_error_still_goes_to_the_rate_limit_handler():
    r = _run(["sendMessage"], "sendMessage()",
             {"/chat": [{"status": 429, "body": {"error": "rate_limit", "message": "Daily limit"}}]},
             setup="document.getElementById('input').value = 'hello';")
    assert r["log"]["rateLimited"] == 1 and not _busy_message(r)


def test_a_normal_answer_is_unchanged():
    r = _run(["sendMessage"], "sendMessage()",
             {"/chat": [{"body": {"type": "answer", "text": "**Done**"}}]},
             setup="document.getElementById('input').value = 'hello';")
    assert _live(r["log"]["msgs"]) == [("user", "hello"), ("agent", "<strong>Done</strong>")]
    assert r["busy"] is False
