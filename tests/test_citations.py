"""
#7 Stage 1 — deterministic citation layer (src/citations.py).

The draft emits inline [[collection|match-key|field-path]] tags; this layer
parses / indexes / resolves / strips them. Grounding is UNCHANGED this stage —
these tests prove the machinery resolves cleanly and strips with no residue
before Stage 2 trusts it to replace the full-package grounding input.

No Gemini — crafted tagged-draft strings + the real analysis package.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src import citations as C                       # noqa: E402
from src.data_agent import prepare_analysis_package  # noqa: E402


@pytest.fixture(scope="module")
def pkg():
    return prepare_analysis_package(query_period_days=365)


@pytest.fixture(scope="module")
def some_exercise(pkg):
    """An exercise that genuinely has a pr.weight leaf, fetched live."""
    for ex in pkg["exercises"]:
        if isinstance(ex.get("pr"), dict) and ex["pr"].get("weight") is not None:
            return ex
    pytest.skip("no exercise with a pr.weight in the pinned DB")


# ── parse ────────────────────────────────────────────────────────────────────

def test_parse_multi_tag_midsentence():
    draft = ("PR is 63.07 lbs [[exercises|Barbell Curl|pr.weight]] and you trained "
             "it 13 times [[exercises|Barbell Curl|training_frequency.session_count]].")
    tags = C.parse_tags(draft)
    assert len(tags) == 2
    assert tags[0].collection == "exercises"
    assert tags[0].match_key == "Barbell Curl"
    assert tags[0].field_path == "pr.weight"
    assert tags[0].associated_number == "63.07"
    assert tags[1].field_path == "training_frequency.session_count"
    assert tags[1].associated_number == "13"


# ── resolve: numeric / hedge ─────────────────────────────────────────────────

def test_resolve_numeric_matches_live_value(pkg, some_exercise):
    idx = C.build_index(pkg)
    name = some_exercise["name"]
    status, value = C.resolve_tag(idx, "exercises", name, "pr.weight")
    assert status == C.OK
    assert value == some_exercise["pr"]["weight"]


def test_resolve_hedge_leaf(pkg, some_exercise):
    idx = C.build_index(pkg)
    name = some_exercise["name"]
    tf = some_exercise.get("training_frequency") or {}
    if "session_count" not in tf:
        pytest.skip("exercise has no training_frequency.session_count")
    status, value = C.resolve_tag(idx, "exercises", name,
                                  "training_frequency.session_count")
    assert status == C.OK
    assert value == tf["session_count"]


def test_resolve_top_level_dict_section(pkg):
    idx = C.build_index(pkg)
    ats = pkg.get("all_time_summary") or {}
    leaf = next((k for k, v in ats.items()
                 if isinstance(v, (int, float, str))), None)
    if leaf is None:
        pytest.skip("all_time_summary has no scalar leaf")
    status, value = C.resolve_tag(idx, "all_time_summary", "-", leaf)
    assert status == C.OK
    assert value == ats[leaf]


# ── resolve: absence ─────────────────────────────────────────────────────────

def test_absence_ok_when_genuinely_absent(pkg):
    idx = C.build_index(pkg)
    assert "Quidditch" not in {e["name"] for e in pkg["exercises"]}
    status, value = C.resolve_tag(idx, "exercises", "Quidditch", "ABSENT")
    assert status == C.ABSENT_OK


def test_absence_violation_when_present(pkg, some_exercise):
    idx = C.build_index(pkg)
    status, value = C.resolve_tag(idx, "exercises", some_exercise["name"], "ABSENT")
    assert status == C.ABSENT_VIOLATION


# ── resolve: the two silent-failure flags ────────────────────────────────────

def test_bad_match_key_flags_not_silent(pkg, some_exercise):
    idx = C.build_index(pkg)
    # case/whitespace variant of a real name must NOT silently match — it FLAGS
    drifted = some_exercise["name"].lower() + "  "
    status, _ = C.resolve_tag(idx, "exercises", drifted, "pr.weight")
    assert status == C.MATCH_KEY_FLAG


def test_bad_field_path_not_found(pkg, some_exercise):
    idx = C.build_index(pkg)
    status, _ = C.resolve_tag(idx, "exercises", some_exercise["name"], "pr.bogus_leaf")
    assert status == C.NOT_FOUND


def test_unknown_collection_flags(pkg):
    idx = C.build_index(pkg)
    status, _ = C.resolve_tag(idx, "nonsuch_section", "x", "y")
    assert status == C.UNKNOWN_COLLECTION


# ── comparative: two tags both resolve ───────────────────────────────────────

def test_comparative_two_tags_resolve(pkg):
    names = [e["name"] for e in pkg["exercises"]
             if isinstance(e.get("pr"), dict) and e["pr"].get("weight") is not None]
    if len(names) < 2:
        pytest.skip("need two exercises with pr.weight")
    a, b = names[0], names[1]
    draft = (f"{a} at 100 [[exercises|{a}|pr.weight]] is heavier than {b} "
             f"[[exercises|{b}|pr.weight]].")
    cited = C.extract_cited_values(draft, pkg)
    assert len(cited) == 2
    assert all(c["status"] == C.OK for c in cited)


# ── strip: no residue, clean prose ───────────────────────────────────────────

def test_strip_no_residue_and_clean():
    draft = ("Your PR is 63 lbs [[exercises|Barbell Curl|pr.weight]]. "
             "You trained 13 times[[exercises|Barbell Curl|training_frequency.session_count]], "
             "nice work [[exercises|Walking|ABSENT]] .")
    out = C.strip_tags(draft)
    assert "[[" not in out and "]]" not in out
    assert "  " not in out                      # no doubled spaces
    assert " ." not in out and " ," not in out  # no orphan punctuation
    assert out == ("Your PR is 63 lbs. You trained 13 times, nice work.")


def test_strip_removes_malformed_brackets_too():
    # parse only matches well-formed tags, but strip must leave NO bracket residue
    assert C.strip_tags("text [[not a tag]] more") == "text more"
    assert "[[" not in C.strip_tags("a [[x|y|z]] b [[junk]] c")


# ── extract_cited_values: full payload ───────────────────────────────────────

def test_extract_cited_values_payload(pkg, some_exercise):
    name = some_exercise["name"]
    draft = (f"PR is 99 lbs [[exercises|{name}|pr.weight]]. "
             f"No Quidditch [[exercises|Quidditch|ABSENT]]. "
             f"Bad path [[exercises|{name}|pr.nope]].")
    cited = C.extract_cited_values(draft, pkg)
    assert len(cited) == 3
    by_status = {c["status"] for c in cited}
    assert by_status == {C.OK, C.ABSENT_OK, C.NOT_FOUND}
    ok = next(c for c in cited if c["status"] == C.OK)
    assert ok["value"] == some_exercise["pr"]["weight"]
    assert ok["claim_number"] == "99"


# ── prompt-presence: the draft prompt carries the citation rule ──────────────

def test_draft_prompt_has_citation_rule():
    from src.analysis_agent import _ANALYSIS_SYSTEM as P
    assert "CITATION TAGS" in P
    assert "[[collection|match-key|field-path]]" in P
    # the three claim-kind targets
    assert "NUMERIC" in P and "HEDGE" in P and "ABSENCE" in P
    assert "ABSENT" in P
    assert "[[exercises|Barbell Curl|pr.weight]]" in P
