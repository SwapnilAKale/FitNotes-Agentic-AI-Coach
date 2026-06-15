"""
agent.py must not crash when Gemini returns a candidate with no usable content
(the live "MALFORMED_RESPONSE … 'NoneType' object is not iterable" trace at
_run_collect). Every degraded shape must return the graceful ("", [], None)
result with NO exception. No Gemini, no server — the LLM client is stubbed.
"""

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.agent import AgentSession           # noqa: E402
from src.analysis_agent import _collect_text  # noqa: E402


# ── stub LLM client ─────────────────────────────────────────────────────────

class _Models:
    def __init__(self, resp): self._resp = resp
    def generate_content(self, **kw): return self._resp


class _Client:
    def __init__(self, resp): self.models = _Models(resp)


def _collect(resp):
    """Call _run_collect unbound with a minimal stub (only ._client is used)."""
    stub = SimpleNamespace(_client=_Client(resp))
    return AgentSession._run_collect(stub, [], None)


def _resp(candidates):
    return SimpleNamespace(candidates=candidates)


def _cand(content, finish=None):
    return SimpleNamespace(content=content, finish_reason=finish)


# ── degraded shapes: must return ("", [], None), never raise ────────────────

def test_parts_none_is_graceful():
    # the exact live bug: candidate.content.parts is None
    r = _resp([_cand(SimpleNamespace(parts=None), finish="MALFORMED_RESPONSE")])
    assert _collect(r) == ("", [], None)


def test_content_none_is_graceful():
    assert _collect(_resp([_cand(None)])) == ("", [], None)


def test_no_candidates_is_graceful():
    assert _collect(_resp([])) == ("", [], None)
    assert _collect(_resp(None)) == ("", [], None)


def test_parts_empty_is_graceful():
    assert _collect(_resp([_cand(SimpleNamespace(parts=[]))])) == ("", [], None)


def test_malformed_finish_reason_is_graceful():
    # finish_reason MALFORMED_RESPONSE with no usable parts -> degraded, no raise
    r = _resp([_cand(SimpleNamespace(parts=None), finish="MALFORMED_RESPONSE")])
    out = _collect(r)
    assert out == ("", [], None)


def test_safety_finish_reason_is_graceful():
    r = _resp([_cand(SimpleNamespace(parts=None), finish="SAFETY")])
    assert _collect(r) == ("", [], None)


# ── healthy shapes still work (no regression) ───────────────────────────────

def test_normal_text_response():
    part = SimpleNamespace(text="hello", function_call=None)
    content = SimpleNamespace(parts=[part])
    out = _collect(_resp([_cand(content, finish="STOP")]))
    assert out[0] == "hello" and out[1] == [] and out[2] is content


def test_normal_function_call_response():
    fc = SimpleNamespace(text=None, function_call=SimpleNamespace(name="x", args={}))
    content = SimpleNamespace(parts=[fc])
    out = _collect(_resp([_cand(content)]))
    assert out[0] is None and out[1] == [fc] and out[2] is content


# ── analysis-path _collect_text mirrors the guard ──────────────────────────

def test_analysis_collect_text_parts_none_no_crash():
    r = _resp([_cand(SimpleNamespace(parts=None))])
    r.text = ""                       # fallback path
    assert _collect_text(r) == ""


def test_analysis_collect_text_normal():
    part = SimpleNamespace(text="draft", thought=False)
    r = _resp([_cand(SimpleNamespace(parts=[part]))])
    assert _collect_text(r) == "draft"
