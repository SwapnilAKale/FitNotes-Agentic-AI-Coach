"""
Every Gemini request has a time limit (src/llm.py gemini_http_options).

A request Google never answers used to wait forever, holding its worker thread
and, in the demo, every other visitor. The limit is far above any real answer
(default 30 min), so it never cuts one off; it only ends a stuck request.
"""

import re
from pathlib import Path

import httpx

from src.llm import gemini_http_options

SRC = Path(__file__).resolve().parent.parent / "src"


def test_default_limit_is_thirty_minutes(monkeypatch):
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    assert gemini_http_options().timeout == 30 * 60 * 1000


def test_limit_is_set_by_the_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "90")
    assert gemini_http_options().timeout == 90_000
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "not a number")
    assert gemini_http_options().timeout == 30 * 60 * 1000


def test_every_gemini_client_gets_the_limit():
    """A new genai.Client without the limit fails here."""
    found = 0
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"genai\.Client\(", text):
            call = text[m.end():text.find(")\n", m.end()) + 1]
            assert "http_options=gemini_http_options()" in call, (
                f"{path.name}: genai.Client without the request time limit")
            found += 1
    assert found >= 6


def test_a_timed_out_request_gets_a_plain_message():
    import server as srv
    r = srv._error_response(httpx.ReadTimeout("The read operation timed out"))
    body = r.body.decode()
    assert srv.GEMINI_TIMEOUT_TEXT in body
    assert "read operation" not in body
