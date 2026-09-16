"""
Stage 2 — the proposal pipeline and, more importantly, everything it is not
allowed to do.

The pipeline: a GRAPH proposal (Gemini, no tools) -> a SEARCH run by our code
(it cannot be skipped) -> a READ of the results (Gemini, no tools) -> a code
COMPARISON of the two. validate_proposal, read_results and compare are pure, so
every rule is tested directly. The network is faked: a scripted Gemini client
and a fake search function.

The governing idea: 'unsure' is a SUCCESS. A wrong "alias" silently attributes
hundreds of sets to the wrong muscle and nothing downstream can detect it, so
every ambiguity must degrade to a human decision.
"""

import csv
import json
import urllib.error
from types import SimpleNamespace

import pytest

from src import ontology as ont_mod
from src import ontology_propose as prop

_MUSCLES = [
    (1, "Back", "", "large"),         # region
    (2, "Lats", 1, "large"),
    (3, "Traps", 1, "medium"),        # has a head below it, but is not a region
    (5, "Arms", "", ""),              # region
    (4, "Biceps", 5, "medium"),
    (6, "Upper Traps", 3, "medium"),
    (7, "Rhomboids", 1, "small"),
    (8, "Grip", 5, "small"),
]
_EXERCISES = [                       # two movement patterns, two exercises each
    (1, "Machine Shrug", "machine", "shrug"),
    (2, "Barbell Row", "barbell", "horizontal pull"),
    (3, "Dumbbell Shrug", "dumbbell", "shrug"),
    (4, "Cable Row", "cable", "horizontal pull"),
    (5, "Treadmill", "", "cardio"),  # cardio carries no muscles
]
_EDGES = [(1, 3, "primary", "test"), (2, 2, "primary", "test"),
          (3, 3, "primary", "test"),
          (4, 2, "primary", "test"), (4, 4, "secondary", "test")]
_ALIASES = [("Machine Shrug", 1), ("Barbell Row", 2), ("Dumbbell Shrug", 3),
            ("Cable Row", 4)]

ROW = {"db_exercise_name": "Pendlay Row", "logged_sets": "40",
       "fitnotes_category": "", "decision": "pending", "alias_of": "",
       "muscles": "", "evidence": "", "sources": "", "approved": "",
       "movement_pattern": ""}


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


_SRC = ["https://example.org/pendlay"]


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


# ── Fakes: a scripted Gemini and a search function ────────────────────────────

_GOOD = ('{"decision":"new","movement_pattern":"horizontal pull",'
         '"muscles":[{"muscle":"Lats","role":"primary"}]}')
_READ_LATS = '{"primary":["Lats"],"secondary":["Rhomboids"]}'
_RESULTS = [{"title": "Pendlay row muscles", "url": "https://www.example.org/pendlay-muscles",
             "content": "The Pendlay row works the lats and the rhomboids."}]


class _Gemini:
    """Replies in order, one per call: stage 1 first, then the read stage. An
    Exception in the script is raised instead of returned."""
    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


class _Search:
    def __init__(self, results=_RESULTS, raises=None):
        self._results, self._raises = results, raises
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        if self._raises:
            raise self._raises
        return list(self._results)


def _run(ont, *replies, search=None, row=ROW, slept=None):
    gem, srch = _Gemini(*replies), search or _Search()
    out = prop.propose_one(row, ont, client=gem, search_fn=srch,
                           sleep=(slept.append if slept is not None else (lambda s: None)))
    return out, gem, srch


def _row_answer(pattern, muscles):
    return json.dumps({"decision": "new", "movement_pattern": pattern,
                       "muscles": [{"muscle": m, "role": r} for m, r in muscles]})


# ── Failures are not judgements ───────────────────────────────────────────────

def test_a_call_failure_is_not_a_judgement(ont):
    """The model was never reached, so there is nothing to call 'unsure'.

    The first live --propose run had no API key loaded, marked all ten queued
    exercises 'unsure', and reported success — ten apparent verdicts, zero
    questions asked. A failed call leaves the row pending, with the reason
    visible, and nothing that could pass for a mapping.
    """
    out, _gem, srch = _run(ont, RuntimeError("No API key was provided."))
    assert out["decision"] == "pending"
    assert out["evidence"].startswith(prop.NOT_PROPOSED)
    assert "No API key" in out["evidence"]
    assert out["approved"] == ""
    assert out["muscles"] == "" and out["alias_of"] == "" and out["sources"] == ""
    assert srch.calls == []


def test_a_model_unsure_answer_is_still_unsure(ont):
    """THE NEGATIVE THAT MATTERS. When the model IS reached and says it cannot
    tell, that is a real verdict — and no search credit is spent on it."""
    out, _gem, srch = _run(ont, '{"decision":"unsure","evidence":"Cannot tell."}')
    assert out["decision"] == "unsure"
    assert not out["evidence"].startswith(prop.NOT_PROPOSED)
    assert srch.calls == []


def test_a_never_proposed_row_is_retried(ont):
    """A row left pending by a failed call must be picked up next run."""
    gem, srch = _Gemini(_GOOD, _READ_LATS), _Search()
    stuck = dict(ROW, decision="pending",
                 evidence=f"{prop.NOT_PROPOSED} model call failed: boom")
    out = prop.propose_all([stuck], ont, client=gem, search_fn=srch)
    assert out[0]["decision"] == "new"
    assert len(gem.calls) == 2 and len(srch.calls) == 1


def test_propose_all_leaves_a_row_the_user_already_decided(ont):
    gem, srch = _Gemini(), _Search()
    decided = dict(ROW, decision="new", muscles="Lats:primary", approved="y")
    out = prop.propose_all([decided], ont, client=gem, search_fn=srch)
    assert out == [decided]
    assert gem.calls == [] and srch.calls == []      # no call, no cost


def test_propose_all_retries_an_unsure_row(ont):
    gem, srch = _Gemini(_GOOD, _READ_LATS), _Search()
    out = prop.propose_all([dict(ROW, decision="unsure")], ont, client=gem, search_fn=srch)
    assert out[0]["decision"] == "new"


# ── The search is run by code, and cannot be skipped ──────────────────────────
#
# In run 7 the search was a Gemini tool the model chose not to use on three good
# answers, and a check threw all three away. Now code runs it, every time.

def test_search_always_runs(ont):
    _out, _gem, srch = _run(ont, _GOOD, _READ_LATS)
    assert len(srch.calls) == 1
    alias, _gem, srch = _run(ont, '{"decision":"alias","alias_of":"Machine Shrug"}',
                             '{"primary":["Traps"]}')
    assert alias["decision"] == "alias" and len(srch.calls) == 1


def test_search_uses_the_fixed_query(ont):
    _out, _gem, srch = _run(ont, _GOOD, _READ_LATS)
    assert srch.calls == ['"Pendlay Row" muscles worked']


def test_no_gemini_search_tool(ont):
    _out, gem, _srch = _run(ont, _GOOD, _READ_LATS)
    assert len(gem.calls) == 2
    assert all(not c["config"].tools for c in gem.calls)


@pytest.mark.parametrize("failure", [
    prop.SearchFailed("the search API key was rejected"),
    prop.SearchFailed("this month's search credits are used up"),
    OSError("network down"),
])
def test_search_failure_is_not_proposed(ont, failure):
    out, gem, _srch = _run(ont, _GOOD, search=_Search(raises=failure))
    assert out["decision"] == "pending"
    assert out["evidence"].startswith(prop.NOT_PROPOSED)
    assert "web search failed" in out["evidence"]
    assert len(gem.calls) == 1            # the read stage never ran


def test_no_results_is_unsure_without_a_read_call(ont):
    out, gem, _srch = _run(ont, _GOOD, search=_Search(results=[]))
    assert out["decision"] == "unsure"
    assert len(gem.calls) == 1


def test_read_stage_sees_only_the_results(ont):
    _out, gem, _srch = _run(ont, _GOOD, _READ_LATS)
    read_prompt = gem.calls[1]["contents"]
    assert "The Pendlay row works the lats" in read_prompt
    assert "START FROM THESE" not in read_prompt
    assert "[horizontal pull]" not in read_prompt


def test_tavily_request_shape(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"results": [
                {"title": "T", "url": "https://example.org/a", "content": "lats", "score": 0.9},
                {"title": "junk", "url": "not-a-url"}]}).encode()

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        return _Resp()

    out = prop.tavily_search('"Pendlay Row" muscles worked', _urlopen=fake_urlopen)
    req = seen["req"]
    assert req.full_url == prop.SEARCH_URL and req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer tvly-test"
    body = json.loads(req.data)
    assert body["query"] == '"Pendlay Row" muscles worked'
    assert body["search_depth"] == "basic"
    assert body["max_results"] == prop.SEARCH_MAX_RESULTS
    assert out == [{"title": "T", "url": "https://example.org/a", "content": "lats"}]


@pytest.mark.parametrize("code, words", [
    (401, "rejected"), (432, "credits"), (433, "spending limit")])
def test_tavily_errors_are_plain_and_never_show_the_key(monkeypatch, code, words):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(prop.SEARCH_URL, code, "err", {}, None)

    with pytest.raises(prop.SearchFailed) as err:
        prop.tavily_search("q", _urlopen=boom)
    assert words in str(err.value) and "tvly-secret" not in str(err.value)


def test_tavily_without_a_key_sends_nothing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    sent = []
    with pytest.raises(prop.SearchFailed, match="TAVILY_API_KEY"):
        prop.tavily_search("q", _urlopen=lambda *a, **k: sent.append(1))
    assert sent == []


# ── Comparing the graph with the search ───────────────────────────────────────

def test_agreeing_primary_is_accepted(ont):
    out, _gem, _srch = _run(ont, _GOOD, _READ_LATS)
    assert out["decision"] == "new"
    assert out["sources"] == "https://www.example.org/pendlay-muscles"
    assert "search says primary: Lats" in out["evidence"]
    assert "sites: example.org" in out["evidence"]


def test_a_head_agrees_with_its_parent(ont):
    """The negative: the graph says Upper Traps, a site just says traps."""
    out, _gem, _srch = _run(ont, _row_answer("shrug", [("Upper Traps", "primary")]),
                            '{"primary":["Traps"]}')
    assert out["decision"] == "new"


def test_disagreeing_primary_is_unsure_with_both_shown(ont):
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Biceps"],"secondary":[]}')
    assert out["decision"] == "unsure"
    assert "graph calls Lats a main muscle" in out["evidence"]
    assert "search results say Biceps" in out["evidence"]
    assert out["muscles"] == "" and out["approved"] == ""


# ── The tightened agreement rule ──────────────────────────────────────────────
#
# Run 8 aliased a seated CABLE row to a DUMBBELL row: every search result was a
# cable row, but one shared muscle (Lats) was enough to agree.

def test_a_region_is_ignored_when_a_specific_muscle_is_named(ont):
    ids = ont["by_muscle_name"]
    assert prop.read_results({"primary": ["Back", "Biceps"], "secondary": ["Arms"]}, ont) == {
        "primary": [ids["Biceps"]], "secondary": [], "primary_regions": [ids["Back"]]}
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Back","Biceps"],"secondary":[]}')
    assert out["decision"] == "unsure"


def test_region_only_search_supports_a_muscle_inside_it(ont):
    """Run 9: the sites for an iso-lateral press said only "chest"."""
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Back"],"secondary":[]}')
    assert out["decision"] == "new"
    assert "search says primary: (region only) Back" in out["evidence"]


def test_region_only_search_does_not_support_a_muscle_outside_it(ont):
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Arms"],"secondary":[]}')
    assert out["decision"] == "unsure"
    assert "the search only says Arms; graph said Lats" in out["evidence"]


def test_every_claimed_primary_must_be_named_by_the_search(ont):
    answer = _row_answer("horizontal pull", [("Lats", "primary"), ("Traps", "primary")])
    out, _gem, _srch = _run(ont, answer, '{"primary":["Lats"],"secondary":[]}')
    assert out["decision"] == "unsure"
    assert "graph calls Traps a main muscle; the search never names it" in out["evidence"]


def test_a_claimed_primary_named_only_as_a_helper_is_enough(ont):
    """The negative: sites often list a second main mover as 'also works'."""
    answer = _row_answer("horizontal pull", [("Lats", "primary"), ("Traps", "primary")])
    out, _gem, _srch = _run(ont, answer, '{"primary":["Lats"],"secondary":["Traps"]}')
    assert out["decision"] == "new"


def test_most_of_the_searchs_main_muscles_must_be_covered(ont):
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Lats","Rhomboids","Biceps"]}')
    assert out["decision"] == "unsure"
    assert "main muscles Rhomboids, Biceps are not in the graph answer" in out["evidence"]


def test_exactly_half_is_not_enough(ont):
    """Run 9: a dumbbell row covered exactly 2 of a seated cable row's 4."""
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Lats","Rhomboids"]}')
    assert out["decision"] == "unsure"


def test_more_than_half_is_enough(ont):
    """The negative: sites list more main movers than the graph does."""
    answer = _row_answer("horizontal pull", [("Lats", "primary"), ("Biceps", "secondary")])
    out, _gem, _srch = _run(ont, answer, '{"primary":["Lats","Rhomboids","Biceps"]}')
    assert out["decision"] == "new"


@pytest.fixture
def real_graph(monkeypatch):
    """The real, tracked graph — read only."""
    from pathlib import Path
    monkeypatch.setenv("ONTOLOGY_DIR", str(Path(__file__).resolve().parent.parent / "ontology"))
    ont_mod.clear_cache()
    yield ont_mod.load_ontology(force=True)
    ont_mod.clear_cache()


def test_run8_seated_row_replay(real_graph):
    """Run 8's exact reading for 'Single Arm Seated Row', against the real graph."""
    names = {e["canonical_name"] for e in real_graph["exercises"].values()}
    assert {"Single Hand Dumbbell Row", "Seated Narrow V Shaped Row"} <= names
    row = dict(ROW, db_exercise_name="Single Arm Seated Row")
    results = [{"title": "", "url": "https://sweat.com/exercises/single-arm-seated-cable-row",
                "content": ""}]
    read = prop.read_results({"primary": ["Lats", "Rhomboids", "Rear Delts"],
                              "secondary": ["Biceps", "Traps", "Erectors", "Core", "Obliques"]},
                             real_graph)

    def verdict(target):
        parsed = {"decision": "alias", "alias_of": target}
        out = prop.validate_proposal(row, dict(parsed, sources=[results[0]["url"]]), real_graph)
        return prop.compare(row, parsed, out, read, results, real_graph)["decision"]

    assert verdict("Single Hand Dumbbell Row") == "unsure"
    assert verdict("Seated Narrow V Shaped Row") == "alias"


def test_run9_replay(real_graph):
    """Run 9's readings, against the real graph."""
    names = {e["canonical_name"] for e in real_graph["exercises"].values()}
    assert {"Single Hand Dumbbell Row", "Seated Narrow V Shaped Row",
            "Machine Chest Press"} <= names
    url = "https://example.org/run9"

    def verdict(name, target, reading):
        row = dict(ROW, db_exercise_name=name)
        parsed = {"decision": "alias", "alias_of": target}
        out = prop.validate_proposal(row, dict(parsed, sources=[url]), real_graph)
        return prop.compare(row, parsed, out, prop.read_results(reading, real_graph),
                            [{"title": "", "url": url, "content": ""}], real_graph)["decision"]

    seated_row = {"primary": ["Lats", "Rhomboids", "Rear Delts", "Biceps"],
                  "secondary": ["Traps", "Erectors", "Obliques", "Forearms"]}
    assert verdict("Single Arm Seated Row", "Single Hand Dumbbell Row", seated_row) == "unsure"
    assert verdict("Single Arm Seated Row", "Seated Narrow V Shaped Row", seated_row) == "alias"
    # The sites said only "chest"; the reader's region was not recorded in run 9.
    assert verdict("Iso-lateral Bench Press", "Machine Chest Press",
                   {"primary": ["Chest"], "secondary": ["Triceps"]}) == "alias"


def test_alias_is_checked_against_its_targets_muscles(ont):
    alias = '{"decision":"alias","alias_of":"Machine Shrug"}'
    wrong, _g, _s = _run(ont, alias, '{"primary":["Lats"]}')
    assert wrong["decision"] == "unsure"
    right, _g, _s = _run(ont, alias, '{"primary":["Traps"]}')
    assert right["decision"] == "alias"
    assert "graph: same muscles as Machine Shrug" in right["evidence"]


def test_results_with_no_primary_are_unsure(ont):
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":[],"secondary":["Biceps"]}')
    assert out["decision"] == "unsure"
    assert "did not name a primary muscle" in out["evidence"]


def test_read_stage_cannot_widen_the_muscle_set(ont):
    before = len(ont["muscles"])
    out, _gem, _srch = _run(ont, _GOOD, '{"primary":["Serratus Anterior"]}')
    assert out["decision"] == "unsure"
    assert len(ont["muscles"]) == before
    read = prop.read_results({"primary": ["Serratus Anterior", "Lats"]}, ont)
    assert read["primary"] == [ont["by_muscle_name"]["Lats"]] and read["secondary"] == []


def test_sources_come_from_the_search_not_model_text(ont):
    text = ('{"decision":"new","movement_pattern":"horizontal pull",'
            '"muscles":[{"muscle":"Lats","role":"primary"}],'
            '"sources":["https://youtube.com/watch?v=0_f0_f10_10"]}')
    out, _gem, _srch = _run(ont, text, _READ_LATS)
    assert out["sources"] == "https://www.example.org/pendlay-muscles"


def test_search_secondaries_are_shown_never_added(ont):
    out, _gem, _srch = _run(ont, _GOOD, _READ_LATS)
    assert out["muscles"] == "Lats:primary"
    assert "also names: Rhomboids" in out["evidence"]


# ── Model choice and the retries ──────────────────────────────────────────────

def _per_minute_429(delay="7s"):
    return RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'quotaId': "
        "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}, "
        f"{{'retryDelay': '{delay}'}}]}}}}")


def _hard_429():
    return RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
        "your current quota, please check your plan and billing details.'}}")


def test_proposer_needs_no_search_grounding(monkeypatch):
    """Neither Gemini call searches, so the proposer runs on the app's own model
    — Gemini 3's zero free-tier search grounding no longer matters."""
    monkeypatch.delenv("ONTOLOGY_PROPOSE_MODEL", raising=False)
    assert prop._model() == "gemini-3.1-flash-lite"


def test_proposer_model_can_be_overridden(ont, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_PROPOSE_MODEL", "gemini-2.5-flash")
    _out, gem, _srch = _run(ont, _GOOD, _READ_LATS)
    assert [c["model"] for c in gem.calls] == ["gemini-2.5-flash"] * 2


def test_a_per_minute_429_is_waited_out_and_retried(ont):
    slept = []
    out, gem, _srch = _run(ont, _per_minute_429(), _GOOD, _READ_LATS, slept=slept)
    assert out["decision"] == "new"
    assert len(gem.calls) == 3 and len(slept) == 1


def test_a_hard_limit_429_is_not_retried(ont):
    """THE NEGATIVE THAT MATTERS. A daily limit does not clear by waiting."""
    slept = []
    out, gem, srch = _run(ont, _hard_429(), slept=slept)
    assert slept == [] and len(gem.calls) == 1 and srch.calls == []
    assert out["decision"] == "pending"
    assert out["evidence"].startswith(prop.NOT_PROPOSED)


def test_per_minute_retries_are_bounded(ont):
    slept = []
    out, gem, _srch = _run(ont, *[_per_minute_429()] * 10, slept=slept)
    assert len(gem.calls) == prop.PER_MINUTE_MAX_RETRIES + 1
    assert len(slept) == prop.PER_MINUTE_MAX_RETRIES
    assert out["decision"] == "pending"


def test_retry_wait_uses_the_providers_retry_delay(ont):
    """retryDelay 7s -> 7 + buffer, not the 55s default."""
    slept = []
    _run(ont, _per_minute_429("7s"), _GOOD, _READ_LATS, slept=slept)
    assert slept == [7 + prop.PER_MINUTE_BUFFER]


def test_a_read_stage_failure_is_not_proposed(ont):
    out, _gem, srch = _run(ont, _GOOD, _hard_429())
    assert len(srch.calls) == 1
    assert out["decision"] == "pending"
    assert out["evidence"].startswith(prop.NOT_PROPOSED)


def _503():
    return RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand.', 'status': 'UNAVAILABLE'}}")


def _500():
    return RuntimeError("500 INTERNAL. {'error': {'code': 500, 'status': 'INTERNAL'}}")


def test_a_503_is_retried(ont):
    slept = []
    out, gem, _srch = _run(ont, _503(), _GOOD, _READ_LATS, slept=slept)
    assert out["decision"] == "new"
    assert len(gem.calls) == 3
    assert slept == [prop.TRANSIENT_BACKOFF + prop.PER_MINUTE_BUFFER]


def test_503_retries_are_bounded(ont):
    slept = []
    out, gem, _srch = _run(ont, *[_503()] * 10, slept=slept)
    assert len(gem.calls) == prop.TRANSIENT_MAX_RETRIES + 1
    assert out["decision"] == "pending"
    assert out["evidence"].startswith(prop.NOT_PROPOSED)


def test_a_500_is_not_retried(ont):
    """The negative. A 500 may be a real bug — it fails at once, like the
    coordinator's rule."""
    slept = []
    out, gem, _srch = _run(ont, _500(), slept=slept)
    assert slept == [] and len(gem.calls) == 1
    assert out["decision"] == "pending"


def test_transient_predicate_matches_503_only():
    from google.genai import errors
    from src import checkpoint as ck
    assert ck.is_transient_server_error(_503())
    assert ck.is_transient_server_error(
        errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                           "message": "high demand"}}))
    assert not ck.is_transient_server_error(
        errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL",
                                           "message": "boom"}}))
    assert not ck.is_transient_server_error(_500())
    assert not ck.is_transient_server_error(_per_minute_429())


# ── The graph is the evidence ─────────────────────────────────────────────────

def test_graph_prompt_has_no_search_instructions(ont):
    p = prop._build_prompt(ROW, ont).lower()
    assert "muscles worked" not in p and "stabilizer" not in p


def test_prompt_groups_known_exercises_by_pattern(ont):
    p = prop._build_prompt(ROW, ont)
    assert ("[horizontal pull]\n"
            "- Barbell Row — Lats (primary)\n"
            "- Cable Row — Lats (primary), Biceps (secondary)\n") in p
    assert "[shrug]\n- Dumbbell Shrug — Traps (primary)\n" in p


def test_prompt_says_start_from_pattern_mates():
    assert "START FROM THE MUSCLES OF ITS PATTERN-MATES" in prop._SYSTEM


def test_prompt_shows_known_exercises_with_their_muscles(ont):
    p = prop._build_prompt(ROW, ont)
    assert "Barbell Row — Lats (primary)" in p
    assert "Machine Shrug — Traps (primary)" in p


def test_prompt_lists_the_muscles_inside_each_region(ont):
    p = prop._build_prompt(ROW, ont)
    assert "- Arms (use Biceps, Grip)" in p
    assert "- Back (use Lats, Rhomboids, Traps)" in p


def test_prompt_says_not_to_record_general_bracing():
    assert "Do NOT record general bracing" in prop._SYSTEM


def _new(pattern, muscles=({"muscle": "Lats", "role": "primary"},)):
    return {"decision": "new", "sources": _SRC, "movement_pattern": pattern,
            "muscles": list(muscles)}


def test_known_pattern_is_kept(ont):
    out = _v(_new("Horizontal Pull"), ont)
    assert out["decision"] == "new"
    assert out["movement_pattern"] == "horizontal pull"


def test_cardio_pattern_needs_a_person(ont):
    assert _v(_new("cardio"), ont)["decision"] == "unsure"


def test_unknown_pattern_is_blank_with_a_note(ont):
    """The negative: an uncommon exercise may need a pattern the graph lacks."""
    out = _v(_new("sled push"), ont)
    assert out["decision"] == "new"
    assert out["movement_pattern"] == ""
    assert "a person sets it" in out["evidence"]


def test_muscles_outside_the_group_are_noted(ont):
    out, _gem, _srch = _run(
        ont, _row_answer("horizontal pull", [("Lats", "primary"), ("Rhomboids", "secondary")]),
        _READ_LATS)
    assert out["decision"] == "new"
    assert "not in other horizontal pull exercises: Rhomboids" in out["evidence"]


def test_no_group_note_when_it_matches(ont):
    out, _gem, _srch = _run(
        ont, _row_answer("horizontal pull", [("Lats", "primary"), ("Biceps", "secondary")]),
        _READ_LATS)
    assert out["decision"] == "new"
    assert "not in other" not in out["evidence"]
    assert "graph: started from horizontal pull (Barbell Row, Cable Row)" in out["evidence"]


# ── Roles, regions and the size of a proposal ─────────────────────────────────

def test_limiting_role_is_accepted(ont):
    out = _v({"decision": "new", "sources": _SRC,
              "muscles": [{"muscle": "Lats", "role": "primary"},
                          {"muscle": "Grip", "role": "limiting"}]}, ont)
    assert out["decision"] == "new"
    assert out["muscles"] == "Lats:primary|Grip:limiting"


def test_limiting_alone_is_not_a_primary(ont):
    out = _v({"decision": "new", "sources": _SRC,
              "muscles": [{"muscle": "Grip", "role": "limiting"}]}, ont)
    assert out["decision"] == "unsure"


@pytest.mark.parametrize("muscles", [
    [{"muscle": "Back", "role": "primary"}],
    [{"muscle": "Lats", "role": "primary"}, {"muscle": "Arms", "role": "limiting"}],
])
def test_a_region_is_never_a_muscle(ont, muscles):
    out = _v({"decision": "new", "sources": _SRC, "muscles": muscles}, ont)
    assert out["decision"] == "unsure"
    assert "whole body region" in out["evidence"]


def test_a_muscle_with_heads_is_still_allowed(ont):
    """The negative. Traps has a head below it but is not a region — the real
    graph uses Triceps (which has heads) on 13 press edges."""
    out = _v({"decision": "new", "sources": _SRC,
              "muscles": [{"muscle": "Lats", "role": "primary"},
                          {"muscle": "Traps", "role": "secondary"}]}, ont)
    assert out["decision"] == "new"


def _trained(n_trained, limiting=()):
    names = ["Lats", "Traps", "Upper Traps", "Rhomboids", "Biceps"][:n_trained]
    return ([{"muscle": names[0], "role": "primary"}]
            + [{"muscle": m, "role": "secondary"} for m in names[1:]]
            + [{"muscle": m, "role": "limiting"} for m in limiting])


def test_too_many_trained_muscles_needs_a_person(ont):
    out = _v({"decision": "new", "sources": _SRC,
              "muscles": _trained(prop.MAX_TRAINED_MUSCLES + 1)}, ont)
    assert out["decision"] == "unsure"


def test_limiting_muscles_do_not_count_toward_the_cap(ont):
    out = _v({"decision": "new", "sources": _SRC,
              "muscles": _trained(prop.MAX_TRAINED_MUSCLES,
                                  limiting=("Biceps", "Grip"))}, ont)
    assert out["decision"] == "new"


# ── A rejected answer is still visible ────────────────────────────────────────

def test_a_rejected_proposal_is_shown_in_evidence(ont):
    out, _gem, _srch = _run(
        ont, '{"decision":"new","muscles":[{"muscle":"Back","role":"primary"}]}', _READ_LATS)
    assert out["decision"] == "unsure"
    assert "model proposed: Back:primary" in out["evidence"]
    assert out["muscles"] == "" and out["approved"] == ""


def test_rejected_answer_shows_its_pattern(ont):
    out, _gem, _srch = _run(ont, _row_answer("horizontal pull", [("Back", "primary")]),
                            _READ_LATS)
    assert out["decision"] == "unsure"
    assert "model proposed: (horizontal pull) Back:primary" in out["evidence"]


def test_a_rejected_alias_is_shown_in_evidence(ont):
    out, _gem, _srch = _run(ont, '{"decision":"alias","alias_of":"Nonexistent Lift"}',
                            _READ_LATS)
    assert out["decision"] == "unsure"
    assert "model proposed: ALIAS Nonexistent Lift" in out["evidence"]
    assert out["alias_of"] == ""


def test_an_accepted_proposal_has_no_rejection_note(ont):
    out, _gem, _srch = _run(ont, _GOOD, _READ_LATS)
    assert out["decision"] == "new"
    assert "model proposed:" not in out["evidence"]


# ── A failure must be diagnosable from the queue ──────────────────────────────

def test_an_unreadable_reply_is_shown(ont):
    """Run 6 lost two rows to 'did not return a JSON object' with no way to see
    what the model said instead."""
    out, _gem, srch = _run(ont, "Sorry, I think this works the lats mostly.")
    assert out["decision"] == "unsure"
    assert 'it replied: "Sorry, I think this works the lats mostly."' in out["evidence"]
    assert srch.calls == []


def test_an_unreadable_reader_reply_is_shown(ont):
    out, _gem, _srch = _run(ont, _GOOD, "The results mention lats a lot.")
    assert out["decision"] == "unsure"
    assert 'reader reply was not JSON: "The results mention lats a lot."' in out["evidence"]


def test_long_replies_are_cut():
    snippet = prop._snippet("x" * 1000)
    assert len(snippet) <= 203 and snippet.endswith('…"')
    assert prop._snippet("   ") == "(empty reply)"


def test_cardio_is_not_offered_as_a_choice(ont):
    p = prop._build_prompt(ROW, ont)
    choices = p.split("Movement patterns in the graph (choose one):\n", 1)[1].split("\n", 1)[0]
    assert "cardio" not in choices and "shrug" in choices
    assert "[cardio]" in p          # the graph itself is still shown as it is


def test_error_dumps_are_shortened(ont):
    out, _gem, _srch = _run(ont, _503(), _503(), _503())
    assert out["decision"] == "pending"
    assert "503 UNAVAILABLE: This model is currently experiencing high demand." in out["evidence"]
    assert "{'error'" not in out["evidence"]


def test_plain_errors_pass_through():
    assert prop._short_error(RuntimeError("No API key was provided.")) == \
        "No API key was provided."
