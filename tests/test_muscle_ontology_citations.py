"""
muscle_ontology_summary must be addressable by the citation layer, or the
Analysis Agent cannot make a single muscle claim: every factual claim has to
cite a leaf that appears in the [CITABLE SCHEMA] block.

Also pins the structural half of the no-verdict rule — the section carries no
judgement leaf, so "you should train X more" has nothing to cite and grounding
rejects it. That is the enforcement; the prompt only states it.
"""

import pytest

from src import citations as cite

_SECTION = {
    "window":       {"start": "2026-04-23", "end": "2026-07-22", "days": 90},
    "prior_window": {"start": "2026-01-22", "end": "2026-04-22", "complete": True},
    "muscles": [
        {"muscle": "Rear Delts", "path": "Shoulders > Rear Delts",
         "size_class": "small", "primary_sets": 71, "secondary_sets": 0,
         "prior_primary_sets": 93, "prior_secondary_sets": 0,
         "exercise_count": 2, "last_trained_date": "2026-06-25"},
        {"muscle": "Erectors", "path": "Back > Erectors",
         "size_class": "medium", "primary_sets": 0, "secondary_sets": 12,
         "prior_primary_sets": 0, "prior_secondary_sets": 4,
         "exercise_count": 1, "last_trained_date": "2026-05-02"},
    ],
    "zero_coverage": ["Calves"],
    "counted_exercises": ["Deadlift"],
    "unmapped_exercises": [], "unmapped_sets": 0,
    "unattributed_exercises": ["Walking"], "unattributed_sets": 7,
    "note": "primary_sets and secondary_sets are SEPARATE columns",
}
_PACKAGE = {"muscle_ontology_summary": _SECTION, "exercises": []}


@pytest.fixture
def index():
    return cite.build_index(_PACKAGE)


def _resolve(index, key, path):
    return cite.resolve_tag(index, "muscle_ontology_summary", key, path)


# ── Per-muscle scalars resolve by muscle name ─────────────────────────────────

def test_per_muscle_set_count_resolves_to_a_scalar(index):
    status, value = _resolve(index, "Rear Delts", "primary_sets")
    assert status == cite.OK and value == 71


def test_prior_window_count_resolves(index):
    status, value = _resolve(index, "Rear Delts", "prior_primary_sets")
    assert status == cite.OK and value == 93


def test_secondary_column_resolves_independently(index):
    assert _resolve(index, "Erectors", "secondary_sets") == (cite.OK, 12)
    assert _resolve(index, "Erectors", "primary_sets") == (cite.OK, 0)


def test_last_trained_date_resolves(index):
    status, value = _resolve(index, "Erectors", "last_trained_date")
    assert status == cite.OK and value == "2026-05-02"


def test_section_wide_scalar_resolves_on_the_dash_form(index):
    assert _resolve(index, "-", "unmapped_sets") == (cite.OK, 0)
    assert _resolve(index, "-", "unattributed_sets") == (cite.OK, 7)
    assert _resolve(index, "-", "window.days") == (cite.OK, 90)
    assert _resolve(index, "-", "prior_window.complete") == (cite.OK, True)


# ── Bad citations are flagged, not silently accepted ──────────────────────────

def test_invented_muscle_name_is_flagged(index):
    status, _ = _resolve(index, "Sternocleidomastoid", "primary_sets")
    assert status == cite.MATCH_KEY_FLAG


def test_invented_field_is_flagged(index):
    status, _ = _resolve(index, "Rear Delts", "total_sets")
    assert status == cite.NOT_FOUND


def test_no_judgement_leaf_exists_to_cite(index):
    """The no-verdict rule is enforced by the SCHEMA, not by a phrase blocklist:
    there is simply nothing to point at."""
    for invented in ("lagging", "is_lagging", "deficit", "needs_work",
                     "undertrained", "recommendation", "target_sets"):
        status, _ = _resolve(index, "Rear Delts", invented)
        assert status == cite.NOT_FOUND, invented


def test_the_muscles_list_itself_is_not_a_citable_scalar(index):
    status, _ = _resolve(index, "-", "muscles")
    assert status == cite.OK_NONSCALAR


# ── The schema block the agent is shown ───────────────────────────────────────

def test_schema_advertises_the_section_by_muscle(index):
    schema = cite.build_citable_schema(_PACKAGE)
    assert "muscle_ontology_summary (match-key = the muscle)" in schema
    for leaf in ("primary_sets", "secondary_sets", "prior_primary_sets",
                 "last_trained_date"):
        assert leaf in schema


def test_schema_lists_no_judgement_leaf(index):
    schema = cite.build_citable_schema(_PACKAGE)
    for word in ("lagging", "neglect", "undertrained", "deficit", "should_train"):
        assert word not in schema.lower()


# ── End-to-end through a draft ────────────────────────────────────────────────

def test_tags_in_a_draft_extract_and_strip(index):
    draft = ("Rear delts took 71 sets [[muscle_ontology_summary|Rear Delts|primary_sets]] "
             "this window, against 93 [[muscle_ontology_summary|Rear Delts|prior_primary_sets]] "
             "in the one before.")
    cited = cite.extract_cited_values(draft, _PACKAGE)
    assert [c["value"] for c in cited] == [71, 93]
    assert all(c["status"] == cite.OK for c in cited)
    # The number the agent wrote must match the leaf it pointed at.
    assert [c["claim_number"] for c in cited] == ["71", "93"]
    assert "[[" not in cite.strip_tags(draft)


def test_degraded_empty_section_does_not_break_the_citation_layer():
    """The ontology store can be missing; the section is then {} and the rest of
    the package must still address cleanly."""
    pkg = {"muscle_ontology_summary": {}, "exercises": []}
    idx = cite.build_index(pkg)
    status, _ = cite.resolve_tag(idx, "muscle_ontology_summary", "Rear Delts",
                                 "primary_sets")
    assert status in (cite.MATCH_KEY_FLAG, cite.UNKNOWN_COLLECTION, cite.NOT_FOUND)
    assert "muscle_ontology_summary" not in cite.build_citable_schema(pkg)
