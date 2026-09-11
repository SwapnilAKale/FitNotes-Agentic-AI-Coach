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

from pathlib import Path as _Path
_ROOT_PATH = _Path(_ROOT)


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


# ══ One list of execute tools, not four ══════════════════════════════════════
#
# LIVE-CAUGHT. A note on a set staged with requires_confirmation:true and then
# executed with NO confirmation panel: server.py kept its own EXECUTE_TOOLS and
# execute_staged_set_comment was not in it, so the handler treated the execute
# as a staging call and waved it through. The same knowledge was written out by
# hand in three files and no two agreed.
#
# These assert against list_tools() — reality — rather than against another copy.

def _exposed_execute_tools() -> set:
    from mcp_servers.combined_server import list_tools
    return {t.name for t in asyncio.run(list_tools())
            if t.name.startswith("execute_staged_")}


def test_every_exposed_execute_tool_is_gated():
    """The agent can call these. Each one must be confirmable and recognised as
    an execute, or it writes to the database unchallenged."""
    from src.agent import CONFIRM_TOOLS, EXECUTE_TOOLS
    missing_gate = _exposed_execute_tools() - EXECUTE_TOOLS
    missing_confirm = _exposed_execute_tools() - CONFIRM_TOOLS
    assert not missing_gate, f"not treated as executes: {sorted(missing_gate)}"
    assert not missing_confirm, f"not confirmable: {sorted(missing_confirm)}"


def test_the_callers_do_not_keep_their_own_copies():
    """Importing the shared set is the entire fix — a local list in either caller
    is the bug growing back.

    Single call sites (`call_tool("execute_staged_workout", {})`) are fine and
    expected; what must not come back is a COLLECTION of execute names, named or
    anonymous, since that is a copy that can drift.
    """
    import re
    for fname in ("server.py", "cli.py"):
        src = (_ROOT_PATH / fname).read_text(encoding="utf-8")
        assert "EXECUTE_TOOLS = {" not in src, f"{fname} redefines EXECUTE_TOOLS"
        assert "from src.agent import" in src and "EXECUTE_TOOLS" in src, (
            f"{fname} should import the shared EXECUTE_TOOLS")
        for literal in re.findall(r"[\{\[\(][^{}\[\]()]*[\}\]\)]", src, re.S):
            # A MAPPING is not a copy. {"delete_goal": "execute_staged_goal_
            # delete", ...} routes a staged key to its one execute tool — a
            # different fact from "which tools are gated", and it cannot drift
            # from the gated set because the next test pins its values against
            # EXECUTE_TOOLS. What must not come back is a bare COLLECTION of
            # execute names, which is a second answer to the same question.
            bare = re.findall(r'(?<![:{]\s)(?<!: )"(execute_staged_[a-z_]+)"(?!\s*:)',
                              literal)
            mapped = re.findall(r':\s*"execute_staged_[a-z_]+"', literal)
            if mapped:
                continue
            assert len(bare) < 2, (
                f"{fname} lists execute tool names inline instead of importing "
                f"them: {sorted(set(bare))}")


def test_the_staged_key_map_cannot_name_an_ungated_tool():
    """server._SIBLING_EXECUTE routes a confirmed staged write to its execute.
    Every tool it names must be one the confirmation gate knows about, or the
    server would commit a write the panel never covered."""
    import server as server_mod
    from src.agent import EXECUTE_TOOLS
    unknown = set(server_mod._SIBLING_EXECUTE.values()) - EXECUTE_TOOLS
    assert not unknown, f"staged-key map names ungated tools: {sorted(unknown)}"


def test_the_host_driven_slot_set_is_not_widened_by_mistake():
    """_EXECUTE_TOOL_NAMES is NOT 'all executes' and must not be 'fixed' to match
    EXECUTE_TOOLS. It flips staging_reached_confirm, which SUPPRESSES the
    write-success claim gate — widening it to agent-driven executes would switch
    that gate off for every set-edit and goal-edit turn."""
    from src.agent import _EXECUTE_TOOL_NAMES
    assert _EXECUTE_TOOL_NAMES == {"execute_staged_workout", "execute_staged_goal"}
