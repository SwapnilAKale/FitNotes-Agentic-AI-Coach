"""
Hygiene sweep — #11 (duplicate knowledge-base prompt block merged) and
#13 (resolve_exercise_name input guards). #9 has its own file
(test_sql_sanitize_canonical.py); #10 (dead-code deletion) and #12 (doc count)
are verified by the suite staying green / by inspection.

No Gemini, no server.
"""

import asyncio
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

DB_PATH = os.environ["FITNOTES_DB_PATH"]


# ══ #11 — one knowledge-base block, naming valid + exposed tools ═════════════

def test_single_knowledge_base_block_with_valid_tools():
    from src.agent import SYSTEM_PROMPT as P
    from mcp_servers.combined_server import list_tools
    exposed = {t.name for t in asyncio.run(list_tools())}

    # exactly one knowledge-base block; the old duplicate header is gone
    assert P.count("📖 KNOWLEDGE BASE") == 1
    assert "USER KNOWLEDGE BASE" not in P

    # the block names the real article tools, and they actually exist + are exposed
    for tool in ("list_user_articles", "delete_user_article"):
        assert tool in P
        assert tool in exposed


# ══ #13 — resolve_exercise_name input guards ════════════════════════════════

@pytest.mark.parametrize("bad", ["", "   ", "\t", "a", " x "])
def test_empty_or_too_short_is_clean_no_match(bad):
    from src.shared.resolver import resolve_exercise_name
    out = resolve_exercise_name(bad, DB_PATH)
    # no-exercise, NOT a disambiguation dump (the old "" -> 8 candidates footgun)
    assert out == {"match": None, "candidates": []}


@pytest.mark.parametrize("wild", ["%", "_", "%%", "ben%ch", "bench_press", "____"])
def test_like_wildcards_treated_literally(wild):
    from src.shared.resolver import resolve_exercise_name
    out = resolve_exercise_name(wild, DB_PATH)
    # %/_ no longer act as wildcards: none of these literal terms exist as a
    # substring of a real exercise name, so no broad wildcard blow-up.
    assert out["match"] is None
    assert out["candidates"] == []


def test_normal_terms_unchanged():
    from src.shared.resolver import resolve_exercise_name
    # exact match still resolves
    assert resolve_exercise_name("deadlift", DB_PATH)["match"] == "Deadlift"
    # genuine multi-match still disambiguates (non-empty candidates)
    out = resolve_exercise_name("bench press", DB_PATH)
    assert out["match"] is None and len(out["candidates"]) > 1


def test_like_escape_helper():
    from src.shared.resolver import _like_escape
    assert _like_escape("a%b_c") == r"a\%b\_c"
    assert _like_escape("plain") == "plain"
    assert _like_escape("back\\slash") == "back\\\\slash"
