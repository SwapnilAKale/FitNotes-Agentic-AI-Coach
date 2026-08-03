"""
Stage 2 — the web-search proposal and, more importantly, everything it is not
allowed to do.

validate_proposal is pure, so every rule is tested directly against a raw model
object. The single network-touching function is covered with a fake client, the
same way the existing agent tests fake Gemini.

The governing idea: 'unsure' is a SUCCESS. A wrong "alias" silently attributes
hundreds of sets to the wrong muscle and nothing downstream can detect it, so
every ambiguity must degrade to a human decision.
"""

import csv

import pytest

from src import ontology as ont_mod
from src import ontology_propose as prop

_MUSCLES = [
    (1, "Back", "", "large"),
    (2, "Lats", 1, "large"),
    (3, "Traps", 1, "medium"),
    (4, "Biceps", "", "medium"),
]
_EXERCISES = [
    (1, "Machine Shrug", "machine", "shrug"),
    (2, "Barbell Row", "barbell", "horizontal pull"),
]
_EDGES = [(1, 3, "primary", "test"), (2, 2, "primary", "test")]
_ALIASES = [("Machine Shrug", 1), ("Barbell Row", 2)]

ROW = {"db_exercise_name": "Pendlay Row", "logged_sets": "40",
       "fitnotes_category": "", "decision": "pending", "alias_of": "",
       "muscles": "", "evidence": "", "sources": "", "approved": ""}


@pytest.fixture
def ont(tmp_path, monkeypatch):
    def _w(name, header, rows):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    _w("muscles.csv", ["id", "name", "parent_id", "size_class"], _MUSCLES)
    _w("exercises.csv", ["id", "canonical_name", "equipment", "movement_pattern"],
       _EXERCISES)
    _w("exercise_muscle.csv", ["exercise_id", "muscle_id", "role", "source"], _EDGES)
    _w("aliases.csv", ["db_exercise_name", "exercise_id"], _ALIASES)
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    ont_mod.clear_cache()
    yield ont_mod.load_ontology(force=True)
    ont_mod.clear_cache()


def _v(parsed, ont, row=None):
    return prop.validate_proposal(row or ROW, parsed, ont)


# ── Accepted proposals ────────────────────────────────────────────────────────

def test_valid_new_exercise_is_accepted(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "primary"},
                          {"muscle": "Biceps", "role": "secondary"}],
              "evidence": "A barbell row from a dead stop.",
              "sources": ["https://example.org/pendlay"]}, ont)
    assert out["decision"] == "new"
    assert out["muscles"] == "Lats:primary|Biceps:secondary"
    assert out["sources"] == "https://example.org/pendlay"
    assert out["approved"] == ""          # the gate stays closed (R10)


def test_valid_alias_is_accepted_and_canonicalised(ont):
    out = _v({"decision": "alias", "alias_of": "machine shrug",
              "evidence": "Same movement.",
              "sources": ["https://example.org/shrug"]}, ont)
    assert out["decision"] == "alias"
    assert out["alias_of"] == "Machine Shrug"    # exact store spelling
    assert out["approved"] == ""


def test_muscle_names_are_matched_case_insensitively(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "lats", "role": "PRIMARY"}],
              "sources": ["https://example.org/x"]}, ont)
    assert out["decision"] == "new" and out["muscles"] == "Lats:primary"


def test_duplicate_muscles_collapse(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "primary"},
                          {"muscle": "Lats", "role": "secondary"}],
              "sources": ["https://example.org/x"]}, ont)
    assert out["muscles"] == "Lats:primary"


# ── R7: the model cannot widen the muscle set ─────────────────────────────────

def test_muscle_outside_the_closed_set_forces_unsure(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "primary"},
                          {"muscle": "Sternocleidomastoid", "role": "secondary"}],
              "evidence": "confident", "sources": ["https://example.org/x"]}, ont)
    assert out["decision"] == "unsure"
    assert "Sternocleidomastoid" in out["evidence"]
    # Negative: not silently dropped and the rest kept — the WHOLE row is held.
    assert out["muscles"] == ""
    assert "Sternocleidomastoid" not in ont["by_muscle_name"]


def test_a_confident_hallucination_still_cannot_add_a_muscle(ont):
    before = len(ont["muscles"])
    _v({"decision": "new",
        "muscles": [{"muscle": "Serratus Anterior", "role": "primary"}],
        "evidence": "definitely", "sources": ["https://example.org/x"]}, ont)
    assert len(ont["muscles"]) == before


# ── R8: no category can enter through this path ───────────────────────────────

def test_a_category_field_in_the_model_output_is_ignored(ont):
    out = _v({"decision": "new", "category": "Rowing",
              "new_category": "Pull Machines",
              "muscles": [{"muscle": "Lats", "role": "primary"}],
              "sources": ["https://example.org/x"]}, ont)
    assert out["decision"] == "new"
    assert "category" not in out.get("muscles", "")
    # The row schema has no category output at all — only the input hint field.
    assert set(out) == set(ROW)
    assert out["fitnotes_category"] == ROW["fitnotes_category"]


def test_category_hint_is_labelled_as_weak_in_the_prompt(ont):
    text = prop._build_prompt(dict(ROW, fitnotes_category="Category_15"), ont)
    assert "Category_15" in text and "WEAK HINT" in text
    # The closed list is handed over verbatim so the model picks, not invents.
    assert "Lats" in text and "Biceps" in text


# ── Everything ambiguous degrades to unsure ───────────────────────────────────

def test_missing_sources_forces_unsure(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "primary"}],
              "sources": []}, ont)
    assert out["decision"] == "unsure" and "source" in out["evidence"]


def test_non_http_source_does_not_count(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "primary"}],
              "sources": ["I know this already"]}, ont)
    assert out["decision"] == "unsure"


def test_alias_to_unknown_exercise_forces_unsure(ont):
    out = _v({"decision": "alias", "alias_of": "Pendlay Row Machine",
              "sources": ["https://example.org/x"]}, ont)
    assert out["decision"] == "unsure" and "not a known exercise" in out["evidence"]


def test_no_primary_muscle_forces_unsure(ont):
    out = _v({"decision": "new",
              "muscles": [{"muscle": "Lats", "role": "secondary"}],
              "sources": ["https://example.org/x"]}, ont)
    assert out["decision"] == "unsure" and "primary" in out["evidence"]


def test_model_answering_unsure_is_respected(ont):
    out = _v({"decision": "unsure", "evidence": "Sources disagreed."}, ont)
    assert out["decision"] == "unsure" and "Sources disagreed" in out["evidence"]


def test_garbage_output_forces_unsure(ont):
    for junk in (None, "not json", [], 42, {}):
        assert _v(junk, ont)["decision"] == "unsure"


def test_unsure_row_is_never_approved(ont):
    out = _v({"decision": "unsure"}, ont, row=dict(ROW, approved="y"))
    assert out["approved"] == ""      # a downgrade also revokes an approval


# ── JSON extraction robustness ────────────────────────────────────────────────

def test_fenced_json_is_parsed():
    assert prop._parse_json('```json\n{"decision":"new"}\n```') == {"decision": "new"}


def test_json_wrapped_in_prose_is_parsed():
    got = prop._parse_json('Here you go:\n{"decision":"alias"}\nHope that helps.')
    assert got == {"decision": "alias"}


def test_unparseable_text_returns_none():
    assert prop._parse_json("no object here") is None


# ── propose_one with a fake client ────────────────────────────────────────────

class _FakeClient:
    def __init__(self, text=None, raises=None):
        self._text, self._raises = text, raises
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return type("R", (), {"text": self._text})()


def test_propose_one_requests_the_search_tool(ont):
    fake = _FakeClient('{"decision":"new","muscles":[{"muscle":"Lats","role":"primary"}],'
                       '"sources":["https://example.org/x"]}')
    out = prop.propose_one(ROW, ont, client=fake)
    assert out["decision"] == "new"
    tools = fake.calls[0]["config"].tools
    assert tools and tools[0].google_search is not None


def test_network_failure_degrades_to_unsure_and_writes_nothing(ont):
    fake = _FakeClient(raises=RuntimeError("connection reset"))
    out = prop.propose_one(ROW, ont, client=fake)
    assert out["decision"] == "unsure"
    assert "connection reset" in out["evidence"]
    assert out["approved"] == "" and out["muscles"] == ""


def test_propose_all_leaves_a_row_the_user_already_decided(ont):
    fake = _FakeClient('{"decision":"new"}')
    decided = dict(ROW, decision="new", muscles="Lats:primary", approved="y")
    out = prop.propose_all([decided], ont, client=fake)
    assert out == [decided]
    assert fake.calls == []          # no call, no cost


def test_propose_all_retries_an_unsure_row(ont):
    fake = _FakeClient('{"decision":"new","muscles":[{"muscle":"Lats","role":"primary"}],'
                       '"sources":["https://example.org/x"]}')
    out = prop.propose_all([dict(ROW, decision="unsure")], ont, client=fake)
    assert out[0]["decision"] == "new"
    assert len(fake.calls) == 1
