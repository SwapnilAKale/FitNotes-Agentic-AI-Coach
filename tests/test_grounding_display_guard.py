"""
Issue 3 / B1 — grounding false-positive fixes (live-reproduced 2026-07-11):

B1a — COUNT CONFLATION: the package had no scalar for "total sets incl.
warmups", so grounding "corrected" a TRUE "4 sets" claim against
working_sets_count (3). Fix: sessions carry total_sets_count; progression
carries the citable latest_session_total_sets.

B1b — DISPLAY-BLINDNESS: grounding edited/deleted verbatim display_sets lines
(removed "Set 1 (Warmup): 0.0 lbs × 12 reps" live). Fix: structural guard —
ground_check excises the known lines to ⟦D<i>⟧ sentinels BEFORE the grounding
LLM sees the draft and reinserts them after; build_grounding_context carries
display_sets in BOTH modes so the guard works on the cheap path too.

Live-DB tests use the standard data/FitNotes_Backup.fitnotes fixture (same as
test_citations.py). No Gemini — the wiring test fakes the client.
"""

import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

import pytest                                                  # noqa: E402

from src import citations as C                                 # noqa: E402
from src import analysis_agent                                 # noqa: E402
from src.analysis_agent import (                               # noqa: E402
    _ANALYSIS_SYSTEM,
    _GROUNDING_SYSTEM,
    _excise_display,
    _reinsert_display,
)
from src.data_agent import prepare_analysis_package            # noqa: E402
from src.data_agent.process import _compute_progression        # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# B1a — count scalars
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sumo_pkg():
    return prepare_analysis_package(
        query_period_days=90, exercise_names=["Sumo Squats"], include_phase2=True)


@pytest.fixture(scope="module")
def sumo_ex(sumo_pkg):
    return next(e for e in sumo_pkg["exercises"] if e["name"] == "Sumo Squats")


def test_total_sets_count_present_and_survives_trim(sumo_ex):
    # Every session carries the scalar; the trim removed the raw `sets` array.
    for s in sumo_ex["sessions"]:
        assert "total_sets_count" in s
        assert "sets" not in s


def test_warmup_session_total_exceeds_working(sumo_ex):
    # The live-repro session: 2026-06-25 = 1 warmup + 3 working = 4 logged sets.
    s = next(x for x in sumo_ex["sessions"] if x["date"] == "2026-06-25")
    assert s["working_sets_count"] == 3
    assert s["total_sets_count"] == 4


def test_no_warmup_session_counts_equal(sumo_ex):
    # Edge in the other direction: a session with no warmup detected has
    # total == working.
    no_warm = [s for s in sumo_ex["sessions"] if s.get("warmup_weight") is None]
    assert no_warm, "expected at least one no-warmup session in the period"
    for s in no_warm:
        assert s["total_sets_count"] == s["working_sets_count"]


def test_total_never_below_working(sumo_ex):
    # Negative: the total (incl. warmups) can never be below the working count.
    for s in sumo_ex["sessions"]:
        assert s["total_sets_count"] >= s["working_sets_count"]


def test_latest_session_total_sets_matches_last_session(sumo_ex):
    prog = sumo_ex["progression"]
    last = sumo_ex["sessions"][-1]
    assert prog["latest_session_total_sets"] == last["total_sets_count"] == 4
    assert prog["latest_session_date"] == last["date"]


def test_compute_progression_tolerates_minimal_session_dicts():
    # Tests elsewhere call _compute_progression with minimal dicts — the new
    # field must be None-tolerant, never KeyError.
    def _S(date, w, reps=5):
        e = round(w * (1 + reps / 30), 1)
        return {"date": date, "unit": "lbs", "max_working_weight": float(w),
                "reps_at_max": reps, "estimated_1rm": e}
    p = _compute_progression([_S("2026-06-01", 100), _S("2026-06-08", 105),
                              _S("2026-06-15", 110), _S("2026-06-25", 110)])
    assert p["latest_session_total_sets"] is None


def test_citable_schema_lists_latest_session_total_sets(sumo_pkg):
    s = C.build_citable_schema(sumo_pkg)
    assert "progression.latest_session_total_sets" in s


# ══════════════════════════════════════════════════════════════════════════════
# B1b — excise / reinsert (pure helpers)
# ══════════════════════════════════════════════════════════════════════════════

_DISPLAY = [
    "2026-06-25 — Sumo Squats (4 sets):",
    "Set 1 (Warmup): 0.0 lbs × 12 reps",
    "Set 2: 40.0 lbs × 12 reps",
]

_DRAFT = (
    "Your most recent session was on 2026-06-25.\n"
    "2026-06-25 — Sumo Squats (4 sets):\n"
    "Set 1 (Warmup): 0.0 lbs × 12 reps\n"
    "Set 2: 40.0 lbs × 12 reps\n"
    "You are in a plateau of 70 days."
)


def test_excise_removes_every_display_line():
    excised, mapping = _excise_display(_DRAFT, _DISPLAY)
    # Negative at the source: no display-line text survives into what the
    # grounding LLM would see.
    for line in _DISPLAY:
        assert line not in excised
    assert len(mapping) == 3
    for sentinel in mapping:
        assert sentinel in excised
    # Prose is untouched.
    assert "Your most recent session was on 2026-06-25." in excised
    assert "plateau of 70 days" in excised


def test_reinsert_round_trips_byte_for_byte():
    excised, mapping = _excise_display(_DRAFT, _DISPLAY)
    restored, all_restored = _reinsert_display(excised, mapping)
    assert restored == _DRAFT
    assert all_restored is True


def test_lost_sentinel_partial_restore_flagged():
    excised, mapping = _excise_display(_DRAFT, _DISPLAY)
    # Simulate the grounding model dropping the warmup line's sentinel.
    warm_sentinel = next(s for s, l in mapping.items() if "Warmup" in l)
    mutilated = excised.replace(warm_sentinel + "\n", "")
    restored, all_restored = _reinsert_display(mutilated, mapping)
    assert all_restored is False
    assert "Set 2: 40.0 lbs × 12 reps" in restored        # others restored
    assert "Set 1 (Warmup)" not in restored               # repaired downstream
    assert "⟦D" not in restored                           # no sentinel residue


def test_empty_display_sets_noop():
    excised, mapping = _excise_display(_DRAFT, [])
    assert excised == _DRAFT and mapping == {}
    restored, ok = _reinsert_display(_DRAFT, {})
    assert restored == _DRAFT and ok is True


def test_line_absent_from_draft_gets_no_sentinel():
    excised, mapping = _excise_display(
        "No display content here.", _DISPLAY)
    assert excised == "No display content here."
    assert mapping == {}


def test_substring_line_excised_longest_first():
    short = "Set 2: 40.0 lbs × 12 reps"
    long_ = "Set 2: 40.0 lbs × 12 reps — PR set"
    draft = f"{long_}\n{short}\n"
    excised, mapping = _excise_display(draft, [short, long_])
    assert long_ not in excised and short not in excised
    restored, ok = _reinsert_display(excised, mapping)
    assert restored == draft and ok is True


def test_sentinels_survive_strip_tags():
    # strip_tags removes [[...]] and tidies whitespace — ⟦D0⟧ must pass through.
    assert "⟦D0⟧" in C.strip_tags("text ⟦D0⟧ more [[a|b|c]] text")


# ══════════════════════════════════════════════════════════════════════════════
# B1b — build_grounding_context carries display_sets in BOTH modes
# ══════════════════════════════════════════════════════════════════════════════

def _clean_cited(n=2):
    return [{"claim_number": "1", "tag": "t", "collection": "exercises",
             "match_key": "X", "field_path": "pr.weight",
             "status": C.OK, "value": 1.0} for _ in range(n)]


def test_grounding_context_cheap_carries_display_sets():
    pkg = {"display_sets": list(_DISPLAY)}
    g = C.build_grounding_context(_clean_cited(), pkg)
    assert g["mode"] == "cheap"
    assert g["display_sets"] == _DISPLAY


def test_grounding_context_full_carries_display_sets():
    pkg = {"display_sets": list(_DISPLAY)}
    g = C.build_grounding_context([], pkg)
    assert g["mode"] == "full"
    assert g["display_sets"] == _DISPLAY


def test_grounding_context_no_display_sets_is_empty_list():
    assert C.build_grounding_context([], {})["display_sets"] == []
    assert C.build_grounding_context(_clean_cited(), {})["display_sets"] == []


# ══════════════════════════════════════════════════════════════════════════════
# B1b — ground_check wiring (fake Gemini client; captures the real prompt)
# ══════════════════════════════════════════════════════════════════════════════

class _FakeResponse:
    candidates = []
    def __init__(self, text): self.text = text


class _FakeClient:
    """Captures the grounding prompt; returns a canned JSON answer built by fn."""
    def __init__(self, respond_fn):
        self.prompts = []
        outer = self
        class _Models:
            def generate_content(self, model=None, contents=None, config=None):
                prompt = contents[0].parts[0].text
                outer.prompts.append(prompt)
                return _FakeResponse(respond_fn(prompt))
        self.models = _Models()


def _wire(monkeypatch, respond_fn):
    fake = _FakeClient(respond_fn)
    monkeypatch.setattr(analysis_agent, "_get_client", lambda: fake)
    return fake


def test_ground_check_prompt_never_contains_display_lines(monkeypatch):
    # The model echoes its input back unchanged (a PASS-everything checker).
    def echo(prompt):
        draft = prompt.split("[DRAFT ANSWER]\n", 1)[1].split("\n\n[", 1)[0]
        return json.dumps({"cleaned_answer": draft, "flagged_claims": []})
    fake = _wire(monkeypatch, echo)

    gctx = {"mode": "full", "package": {}, "display_sets": list(_DISPLAY)}
    answer, flagged = asyncio.run(analysis_agent.ground_check(_DRAFT, gctx))

    # Negative at the sink: the grounding LLM's input held sentinels, never
    # the verbatim lines.
    assert len(fake.prompts) == 1
    for line in _DISPLAY:
        assert line not in fake.prompts[0]
    assert "⟦D" in fake.prompts[0]
    # And the final answer holds the verbatim lines again, no sentinel residue.
    for line in _DISPLAY:
        assert line in answer
    assert "⟦D" not in answer
    assert flagged == []


def test_ground_check_model_drops_sentinel_others_restored(monkeypatch):
    # The model deletes one sentinel line (the live failure shape).
    def drop_one(prompt):
        draft = prompt.split("[DRAFT ANSWER]\n", 1)[1].split("\n\n[", 1)[0]
        lines = [l for l in draft.splitlines() if l != "⟦D1⟧"]
        return json.dumps({"cleaned_answer": "\n".join(lines),
                           "flagged_claims": []})
    _wire(monkeypatch, drop_one)

    gctx = {"mode": "full", "package": {}, "display_sets": list(_DISPLAY)}
    answer, _ = asyncio.run(analysis_agent.ground_check(_DRAFT, gctx))

    assert _DISPLAY[0] in answer                      # header restored
    assert _DISPLAY[2] in answer                      # Set 2 restored
    assert _DISPLAY[1] not in answer                  # dropped → fidelity stage
    assert "⟦D" not in answer                         # no residue either way


def test_ground_check_without_display_sets_unchanged(monkeypatch):
    # A gctx with no display_sets key (old shape) must behave exactly as before.
    def echo(prompt):
        draft = prompt.split("[DRAFT ANSWER]\n", 1)[1].split("\n\n[", 1)[0]
        return json.dumps({"cleaned_answer": draft, "flagged_claims": []})
    fake = _wire(monkeypatch, echo)

    gctx = {"mode": "full", "package": {}}
    answer, _ = asyncio.run(analysis_agent.ground_check(_DRAFT, gctx))
    assert answer == _DRAFT
    assert "⟦D" not in fake.prompts[0]


def test_ground_check_parse_failure_returns_original_draft(monkeypatch):
    # Grounding JSON blows up → the ORIGINAL draft (with real display lines,
    # no sentinels) is returned untouched.
    _wire(monkeypatch, lambda prompt: "NOT JSON {{{")
    gctx = {"mode": "full", "package": {}, "display_sets": list(_DISPLAY)}
    answer, flagged = asyncio.run(analysis_agent.ground_check(_DRAFT, gctx))
    assert answer == _DRAFT
    assert flagged == []
    assert "⟦D" not in answer


# ══════════════════════════════════════════════════════════════════════════════
# Prompt hardening present (secondary text — structural guards decide)
# ══════════════════════════════════════════════════════════════════════════════

def test_grounding_prompt_carries_placeholder_and_set_count_rules():
    assert "PLACEHOLDER LINES" in _GROUNDING_SYSTEM
    assert "⟦D" in _GROUNDING_SYSTEM
    assert "total_sets_count" in _GROUNDING_SYSTEM
    assert "latest_session_total_sets" in _GROUNDING_SYSTEM


def test_analysis_prompt_carries_set_count_rule():
    assert "SET COUNTS" in _ANALYSIS_SYSTEM
    assert "latest_session_total_sets" in _ANALYSIS_SYSTEM


def test_analysis_prompt_carries_pain_and_lookup_rules():
    # Issue 3 A/B — prose-quality guardrails (secondary text; behaviour verified live).
    assert "exercise it was logged under" in _ANALYSIS_SYSTEM   # A: attribute pain
    assert "recurring issue" in _ANALYSIS_SYSTEM                # A: no conflation
    assert "SIMPLE LOOKUP" in _ANALYSIS_SYSTEM                  # B: scope reliability
    assert "reliable overview" in _ANALYSIS_SYSTEM             # B: forbid the filler


def test_analysis_prompt_carries_followup_rules():
    # Issue 3 live-review follow-ups (A-count structural aid + Q0). The Q1
    # "shown session" rule was REVERTED (it caused per-set block duplication —
    # see memory issue3-q1-display-duplication); revisit via fidelity dedup.
    assert "pain_by_location" in _ANALYSIS_SYSTEM                 # A: per-symptom count
    assert "immediately before the latest" in _ANALYSIS_SYSTEM    # Q0: previous session
