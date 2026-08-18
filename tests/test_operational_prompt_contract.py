"""
The operational agent's contract with its tools and its prompt, frozen.

WHY THIS EXISTS. `agent.py SYSTEM_PROMPT` has drifted away from the system it
describes. Two measured examples at the time of writing:

  * The DATE RESOLUTION RULE tells the agent to check whether a date "exists in
    the database" before picking a year. Every read tool was stripped from this
    agent — none of the seven is declared by the MCP server any more — so the
    agent is instructed to branch on a fact it cannot obtain.

  * `discard_staged_writes` (throws away a staged workout) is handed to the
    agent, is described nowhere in the prompt, and is NOT in WRITE_TOOLS — so it
    does not reach the confirmation gate. Undocumented and ungated.

Plus the medical and advice-style rules are written twice, in `agent.py` and in
`analysis_agent.py`, and the copies have diverged.

THIS FILE IS A SAFETY NET, NOT A SPEC. It was written to pass against the
UNMODIFIED code and committed in that state, so it cannot be a restatement of
the behavior about to be introduced. Rows the cleanup deliberately changes are
quarantined in PRE-CHANGE blocks carrying their current value and the ledger row
that authorizes the flip. A row that moves without such a label is a regression.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.agent import (                                          # noqa: E402
    CONFIRM_TOOLS,
    DB_WRITE_TOOLS,
    SYSTEM_PROMPT,
    AgentSession,
)


# The tools the MCP server actually hands over. Read from source rather than by
# starting the server: the point is what is DECLARED, and a test that needs a
# live MCP session to answer that would not run in this suite.
def _declared_tools() -> set[str]:
    src = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    return set(re.findall(r'name="([a-z_]+)"', src))


_READ_TOOLS = {
    "get_exercise_sessions", "get_exercise_history", "read_exercise_comments",
    "get_personal_record", "get_weekly_volume", "query_workout_data",
    "run_read_only_sql",
}


# ══════════════════════════════════════════════════════════════════════════════
# What the agent is actually given
# ══════════════════════════════════════════════════════════════════════════════

def test_no_read_tool_is_exposed():
    # The whole premise of "YOU DO NOT READ OR ANALYZE WORKOUT DATA": the agent
    # cannot read, because the tools are gone. If one ever comes back, the
    # prompt's honesty rule and the date rule below both change meaning.
    assert _declared_tools() & _READ_TOOLS == set()


def test_prompt_advertises_no_tool_the_agent_cannot_call():
    # A prompt naming a tool the model cannot call is the fabrication mechanism.
    for tool in _READ_TOOLS:
        assert tool not in SYSTEM_PROMPT, f"prompt still names stripped tool: {tool}"


def test_every_gated_tool_is_really_exposed():
    # The gate must not be guarding names that no longer exist.
    declared = _declared_tools()
    # execute_staged_workout is server-internal by design: the host drives it,
    # the agent never sees it in list_tools, but the gate still names it.
    internal = {"execute_staged_workout"}
    assert (CONFIRM_TOOLS - internal) <= declared


# ══════════════════════════════════════════════════════════════════════════════
# The confirmation gate — which tools reach it, and by which path
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_covers_every_staging_and_execute_tool():
    for tool in ("log_workout", "set_goal", "log_bodyweight",
                 "update_workout_set", "delete_workout_set", "delete_goal",
                 "execute_staged_goal", "execute_staged_set_delete"):
        assert tool in CONFIRM_TOOLS
        assert tool in DB_WRITE_TOOLS


def test_every_db_write_also_needs_confirmation():
    # The superset relationship, asserted rather than assumed: nothing may reach
    # the database without the user having approved it.
    assert DB_WRITE_TOOLS <= CONFIRM_TOOLS


def test_raw_call_tool_is_ungated_by_design():
    """`AgentSession.call_tool` must NOT consult the confirmation handler.

    This is the property the hosts depend on. `cli.py` and `server.py` drive
    turn-start cleanup and cancel by calling the MCP session directly, one level
    below this — if the gate ever moved down into the shared helper, that
    deterministic housekeeping would start prompting the user.
    """
    session = AgentSession("data/unused.fitnotes")
    session._memory_only = True          # no MCP; unknown tools return an error dict
    asked = []
    session.confirmation_handler = lambda name, args: asked.append(name) or True

    out = asyncio.run(session.call_tool("delete_workout_set", {"x": 1}))

    assert asked == [], "raw call_tool consulted the confirmation handler"
    assert "error" in json.loads(out)    # it really did reach the memory dispatcher


def test_hosts_drive_discard_through_the_session_not_the_agent():
    # Structural: every host discard is a direct session.call_tool, which is why
    # gating the agent's path cannot disturb turn-start cleanup or cancel.
    for host in ("cli.py", "server.py"):
        text = (_ROOT / host).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "discard_staged_writes" in line and "call_tool" in line:
                assert "session.call_tool" in line, f"{host}: {line.strip()}"


# ── CHANGED by ledger row B ───────────────────────────────────────────────────
# `discard_staged_writes` destroys pending staged work. It WAS handed to the
# agent, described nowhere in the prompt, and absent from the gate — so the agent
# could silently drop a staged workout. Now: documented, and confirmed like any
# other write. It is deliberately NOT a DB write — a discard changes no stored
# data, and letting it feed the success-claim gate would let an answer say
# something was saved when nothing was.

def test_ledger_b_discard_is_documented():
    assert "discard_staged_writes" in SYSTEM_PROMPT


def test_ledger_b_discard_needs_confirmation():
    assert "discard_staged_writes" in _declared_tools()     # the agent CAN call it
    assert "discard_staged_writes" in CONFIRM_TOOLS         # ...but must ask first


def test_ledger_b_discard_is_not_a_database_write():
    assert "discard_staged_writes" not in DB_WRITE_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# Rules that must survive the rewrite
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("anchor", [
    "DO NOT READ OR ANALYZE WORKOUT DATA",   # honesty rule for misrouted reads
    "WRITE ACTIONS",
    "resolve_exercise_name",
    "update_workout_set",
    "delete_workout_set",
    "STAGED",                                # workout staging language
    "CONFIDENTIALITY RULE",
])
def test_load_bearing_sections_present(anchor):
    assert anchor in SYSTEM_PROMPT


def test_workout_staging_never_claims_past_tense():
    # The staging answer must not say "logged"/"saved" before the user confirms.
    assert 'MUST NOT say "logged"' in SYSTEM_PROMPT


def test_delete_verification_semantics_match_the_server():
    """The prompt stated this rule TWICE, in opposite directions.

    One copy said `verified: false = success`, the other `verified: true =
    success`. The server settles it: _verify_set_deleted_sync returns
    {"verified": true} when the row is GONE, and the tool description declares
    "{verified: true = deleted}". So true = the delete worked, and the other copy
    was telling the model to report a successful delete as a failure.

    Asserted against the server's own text rather than restated by hand — the
    whole defect was two hand-maintained copies of one fact.
    """
    server = (_ROOT / "mcp_servers" / "combined_server.py").read_text(encoding="utf-8")
    assert "{verified: true = deleted}" in server          # the authority

    assert "verified: true = the set is gone = the delete SUCCEEDED" in SYSTEM_PROMPT
    # ...and the inverted copy is gone, in every phrasing it appeared in.
    assert "verified: false = success" not in SYSTEM_PROMPT
    assert "(verified: false = success)" not in SYSTEM_PROMPT


# ── CHANGED by ledger row A ───────────────────────────────────────────────────
# The date rule USED to branch on whether a date "exists in the database" — a
# check this agent cannot perform, having no read tools (asserted above). It also
# carried a worked example that named the same date twice and pointed at a year
# in the future. Both are gone; year inference (pure reasoning, no data needed)
# stays, and anything it cannot settle now goes back to the user.

def test_ledger_a_date_rule_requires_no_lookup():
    assert "exists in the database" not in SYSTEM_PROMPT
    assert "If no data on 2025-12-25" not in SYSTEM_PROMPT


def test_ledger_a_year_inference_survives():
    assert "DATE RESOLUTION RULE" in SYSTEM_PROMPT
    assert "has not happened yet this year" in SYSTEM_PROMPT


def test_ledger_a_date_rule_says_ask_rather_than_guess():
    # The replacement for the impossible lookup: admit the limit, ask the user.
    # Fragments chosen to sit within one line — the prompt wraps, and an anchor
    # spanning a line break asserts the formatting, not the rule.
    assert "CANNOT check which dates have data" in SYSTEM_PROMPT
    assert "no read access to the workout" in SYSTEM_PROMPT
    assert "ask the user for the full date" in SYSTEM_PROMPT


# ══════════════════════════════════════════════════════════════════════════════
# The blocks shared with the analytical agent
# ══════════════════════════════════════════════════════════════════════════════

def _analysis_prompt() -> str:
    from src.analysis_agent import _ANALYSIS_SYSTEM
    return _ANALYSIS_SYSTEM


@pytest.mark.parametrize("anchor", [
    "MEDICAL LINE",
    "NEVER diagnose",
    "USER HOLDS THE FINAL CALL",
    "BIAS TOWARD TRAINING",
])
def test_shared_rules_live_in_both_prompts(anchor):
    # Both agents answer the user, so both carry these. After the extraction they
    # carry them from ONE definition instead of two hand-maintained copies.
    assert anchor in SYSTEM_PROMPT
    assert anchor in _analysis_prompt()


# ── CHANGED by ledger row C ───────────────────────────────────────────────────
# The two copies of the medical policy HAD diverged: the analytical one grew a
# carve-out saying a performance limit is not a symptom, and the operational one
# never got it — so "my grip gave out" could draw a medical redirect from one
# agent and not the other. Both now compose from src/prompt_blocks.py, so the
# operational agent gained the carve-out as a consequence of sharing the text.

def test_ledger_c_performance_limit_carveout_reaches_both_agents():
    assert "A PERFORMANCE LIMIT IS NOT A SYMPTOM" in _analysis_prompt()
    assert "A PERFORMANCE LIMIT IS NOT A SYMPTOM" in SYSTEM_PROMPT


def test_ledger_c_shared_blocks_are_byte_identical_in_both_prompts():
    """The point of the extraction, asserted as an invariant rather than as the
    one symptom that was reported. Equal substrings — not merely 'both mention
    the topic' — is what makes a future edit impossible to apply to one copy."""
    from src.prompt_blocks import ADVICE_STYLE, MEDICAL_LINE
    for block in (ADVICE_STYLE, MEDICAL_LINE):
        assert block in SYSTEM_PROMPT
        assert block in _analysis_prompt()
