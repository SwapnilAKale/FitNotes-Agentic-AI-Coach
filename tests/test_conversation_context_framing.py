"""
tests/test_conversation_context_framing.py — #27 prompt half.

The analysis agent receives recent conversation history as background. It must
be framed as continuity ONLY, so the draft never volunteers a data-availability
disclaimer about an exercise the current question does not ask about (the "I only
have Sumo Squats data, please update your Bench Press data" confabulation).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.analysis_agent import _fmt_conversation


def test_conversation_context_marked_background_only():
    out = _fmt_conversation([{"role": "user", "content": "log bench 100x5"}])
    assert "background continuity ONLY" in out
    assert "not a request to analyze" in out
    # the actual turn content is still passed through
    assert "log bench 100x5" in out


def test_empty_conversation_context_unchanged():
    assert _fmt_conversation(None).startswith("[CONVERSATION CONTEXT]")
    assert "First message in session." in _fmt_conversation([])
