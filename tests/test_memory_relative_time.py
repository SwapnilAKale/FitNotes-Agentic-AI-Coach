"""
Relative-time disambiguation guard on the memory write path.

Stored facts are re-read on later sessions with no per-fact date attached
(the prompt formatters emit content only), so a fact phrased relative to its
write date ("stalled over the last 70 days") turns false as time passes.
add_fact (the single chokepoint shared by the auto-extractor and the
remember_fact tool) deterministically appends an "(as of YYYY-MM-DD)" anchor
to any content carrying a relative-time phrase.

No Gemini, no ChromaDB: MEMORY_PATH is pointed at a tmp file and
embed_and_store_fact is stubbed out.
"""

import inspect
import json
import os
from datetime import date

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src import memory as memory_module
from src.memory import _has_relative_time, _strip_as_of, add_fact


@pytest.fixture
def mem_file(tmp_path, monkeypatch):
    """Isolated memory store; ChromaDB embedding stubbed to a no-op."""
    path = tmp_path / "memory_test.json"
    monkeypatch.setattr(memory_module, "MEMORY_PATH", path)
    monkeypatch.setattr(memory_module, "embed_and_store_fact", lambda fact: None)
    return path


def _stored_contents(path) -> list:
    with open(path) as f:
        return [fact["content"] for fact in json.load(f)["facts"]]


TODAY_STAMP = f"(as of {date.today().isoformat()})"


# ── 1. Relative phrase → stamped ────────────────────────────────────────────

def test_relative_phrase_gets_as_of_stamp(mem_file):
    result = add_fact(
        category="long-term trend",
        content="Lat Pulldown has 20 sessions logged over the last 3 months.",
        source="agent_inferred",
        confidence="medium",
    )
    assert result["status"] == "saved"
    assert result["fact"]["content"].endswith(TODAY_STAMP)
    # The stamp reaches the persisted store (what future sessions read).
    assert _stored_contents(mem_file)[0].endswith(TODAY_STAMP)


@pytest.mark.parametrize("content", [
    "Progress stalled over the past 70 days.",
    "User set a PR 2 weeks ago.",
    "User has been improving recently.",
    "User currently trains fasted.",
    "User hit a squat PR this month.",
])
def test_relative_variants_all_stamped(mem_file, content):
    result = add_fact(category="user_fact", content=content)
    assert result["status"] == "saved"
    assert result["fact"]["content"].endswith(TODAY_STAMP)


# ── 2. Absolute-only content → unchanged (negative direction) ───────────────

@pytest.mark.parametrize("content", [
    "User has been unable to increase Barbell Curl beyond 52 lbs since 2026-04-01.",
    "The user prefers to train in the evening.",
    "20 sessions logged between 2026-04-10 and 2026-07-09.",
    "The user has been working out since 2023.",
])
def test_absolute_content_stored_verbatim(mem_file, content):
    result = add_fact(category="user_fact", content=content)
    assert result["status"] == "saved"
    assert result["fact"]["content"] == content          # no stamp appended
    assert "(as of" not in result["fact"]["content"]


# ── 3. Already-stamped content → no double stamp ────────────────────────────

def test_already_stamped_content_not_double_stamped(mem_file):
    content = "Progress stalled over the last 70 days. (as of 2026-07-09)"
    result = add_fact(category="long-term trend", content=content)
    assert result["status"] == "saved"
    assert result["fact"]["content"] == content
    assert result["fact"]["content"].count("(as of") == 1


# ── 4. Dedup compares stamp-stripped content ────────────────────────────────

def test_dedup_same_fact_same_day(mem_file):
    content = "Squat volume dropped over the last month."
    assert add_fact(category="user_fact", content=content)["status"] == "saved"
    assert add_fact(category="user_fact", content=content)["status"] == "duplicate"
    assert len(_stored_contents(mem_file)) == 1


def test_dedup_across_differing_stamps(mem_file):
    # A fact stamped on an EARLIER day already in the store …
    add_fact(
        category="user_fact",
        content="Squat volume dropped over the last month. (as of 2026-06-01)",
    )
    # … re-extracted later without a stamp (would be stamped with today):
    result = add_fact(
        category="user_fact",
        content="Squat volume dropped over the last month.",
    )
    assert result["status"] == "duplicate"
    assert len(_stored_contents(mem_file)) == 1


# ── 5. Helper unit behavior (edge pins) ─────────────────────────────────────

def test_strip_as_of_only_trailing_stamp():
    assert _strip_as_of("Fact text. (as of 2026-07-09)") == "Fact text."
    # A mid-string parenthetical is NOT a stamp — untouched.
    s = "Improved (as of the June meet) and kept going."
    assert _strip_as_of(s) == s


def test_has_relative_time_negative_on_absolute():
    assert not _has_relative_time("Stalled from 2026-04-30 to 2026-07-09.")
    assert _has_relative_time("Stalled over the last 70 days.")


# ── 6. Prompt pins (Layer 2: extraction prompt + remember_fact tool) ────────

def test_extraction_prompt_teaches_absolute_time():
    from src.agent import AgentSession
    src = inspect.getsource(AgentSession._auto_extract_memories)
    # Today's date is injected; the rule is stated.
    assert "Today's date is" in src
    assert "TIME REFERENCES MUST BE ABSOLUTE" in src
    # The old example that MODELED relative phrasing is gone.
    assert "for the past 3 months" not in src


def test_remember_fact_tool_description_requires_absolute_dates(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.agent import AgentSession
    session = AgentSession.__new__(AgentSession)   # no init: pure method call
    tools = session._build_memory_only_tools()
    remember = next(t for t in tools if t["function"]["name"] == "remember_fact")
    desc = remember["function"]["description"]
    assert "absolute dates" in desc
    assert "never relative" in desc
