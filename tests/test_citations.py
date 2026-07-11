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


# ── B5: date-aware claim-number extraction ────────────────────────────────────

def test_claim_number_iso_date_not_fragment():
    tags = C.parse_tags(
        "Your most recent session was 2026-06-25 "
        "[[exercises|Sumo Squats|progression.latest_session_date]].")
    assert tags[0].associated_number == "2026-06-25"      # full date, not "-25"


def test_claim_number_human_dates():
    mf = C.parse_tags("It was June 25, 2026 [[exercises|X|progression.latest_session_date]].")
    assert mf[0].associated_number == "June 25, 2026"
    df = C.parse_tags("It was 25 June 2026 [[exercises|X|progression.latest_session_date]].")
    assert df[0].associated_number == "25 June 2026"


def test_claim_number_plain_numbers_unchanged():
    assert C.parse_tags("You lifted 130 lbs [[exercises|X|pr.weight]].")[0].associated_number == "130"
    assert C.parse_tags("a 11.5% gain [[exercises|X|progression.weight_change_pct]] here")[0]  # parses
    assert C.parse_tags("gained 11.5% [[exercises|X|progression.weight_change_pct]].")[0].associated_number == "11.5"
    assert C.parse_tags("count is 9 [[exercises|X|training_frequency.session_count]].")[0].associated_number == "9"


# ── B3: a resolved list-leaf cite uses the cheap grounding path ───────────────

_B3_PKG = {"exercises": [{"name": "Sumo Squats",
                          "progression": {"latest_session_date": "2026-06-25"},
                          "pain_analysis": {"pain_occurrences": [{"date": "2026-06-15",
                                                                  "comment": "knee"}]}}]}


def test_list_leaf_cite_uses_cheap_path():
    draft = ("Latest 2026-06-25 [[exercises|Sumo Squats|progression.latest_session_date]]. "
             "Knee pain [[exercises|Sumo Squats|pain_analysis.pain_occurrences]].")
    cited = C.extract_cited_values(draft, _B3_PKG)
    assert any(c["status"] == C.OK_NONSCALAR for c in cited)     # the list leaf
    gctx = C.build_grounding_context(cited, _B3_PKG)
    assert gctx["mode"] == "cheap"                                # was "full" before B3
    loc = "exercises|Sumo Squats|pain_analysis.pain_occurrences"
    carried = next(cv for cv in gctx["cited_values"] if cv["location"] == loc)
    assert isinstance(carried["value"], list) and carried["value"]  # its value rides along


def test_unresolvable_cite_still_forces_full():
    # A fabricated session-path cite (B4) does not resolve → full path stays (safe).
    draft = ("flagged [[sessions|2026-06-15|has_pain_flag]] and latest 2026-06-25 "
             "[[exercises|Sumo Squats|progression.latest_session_date]].")
    gctx = C.build_grounding_context(C.extract_cited_values(draft, _B3_PKG), _B3_PKG)
    assert gctx["mode"] == "full"


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


# ══ Stage 1.5 — generated citable-field schema (Fix 1) ══════════════════════

REAL_LEAVES = ["pr.weight", "pr.reps", "pr.date", "pr.estimated_1rm",
               "period_volume_lbs", "training_frequency.session_count",
               "progression.weight_change_pct", "total_volume_lbs",
               "push_volume_lbs"]


def test_schema_contains_real_leaves(pkg):
    s = C.build_citable_schema(pkg)
    for leaf in REAL_LEAVES:
        assert leaf in s, f"real leaf missing from schema: {leaf}"
    # labelled with the collections + match-keys
    assert "exercises (match-key = the name)" in s
    assert "muscle_group_summary (match-key = the muscle_group)" in s


def test_exercises_line_does_not_offer_ranking_leaves(pkg):
    # Stage 1.7 update: highest_volume / best_e1rm / most_frequent / pct_of_lbs_total
    # are now REAL flat leaves — but on the rankings / muscle_group_balance entity
    # views, NOT on the exercises line. The original confabulation was citing them
    # under `exercises|X|…`; assert the exercises line still doesn't offer them.
    s = C.build_citable_schema(pkg)
    ex_line = next(l for l in s.splitlines() if l.startswith("exercises (match-key"))
    for fake in ("highest_volume", "best_e1rm", "most_frequent", "pct_of_lbs_total"):
        assert fake not in ex_line, f"ranking leaf wrongly on exercises line: {fake}"


def test_schema_is_compact(pkg):
    # leaf names only, not values — a few KB, not the ~358 KB package
    assert len(C.build_citable_schema(pkg).encode()) < 12_000


def test_schema_is_scope_correct():
    # a correlational leaf exists in FOCUSED but is dropped in BROAD scope, and
    # the generated schema reflects that automatically.
    broad   = C.build_citable_schema(prepare_analysis_package(query_period_days=365))
    focused = C.build_citable_schema(
        prepare_analysis_package(query_period_days=365, exercise_names=["Lat Pulldown"]))
    leaf = "rest_performance_buckets.comparison.confidence_label"
    assert leaf in focused
    assert leaf not in broad


def test_schema_lists_entity_view_sections(pkg):
    # Stage 1.7: rankings / exercise_lifecycle / per-group muscle_group_balance
    # are now flat-addressable by entity → listed so ranking claims cite scalars.
    s = C.build_citable_schema(pkg)
    assert "rankings (match-key = the exercise)" in s
    assert "exercise_lifecycle (match-key = the exercise)" in s
    assert "muscle_group_balance (match-key = the muscle_group)" in s
    # the flat ranking/distribution leaves are present
    assert "highest_volume_lbs" in s and "most_stagnant" in s
    assert "pct_of_lbs_total" in s


def test_schema_empty_package_is_safe():
    assert C.build_citable_schema({}) == "(no citable fields available for this question)"
    assert C.build_citable_schema(None) == "(no citable fields available for this question)"


def test_build_user_message_injects_schema(pkg):
    from src.analysis_agent import _build_user_message
    msg = _build_user_message(pkg, "How is my Lat Pulldown?", None, None, None, None)
    assert "[CITABLE SCHEMA]" in msg
    schema_block = msg.split("[CITABLE SCHEMA]")[1].split("[RESEARCH]")[0]
    assert "pr.weight" in schema_block and "period_volume_lbs" in schema_block


def test_draft_prompt_has_stage15_rules():
    from src.analysis_agent import _ANALYSIS_SYSTEM as P
    # Fix 1: only-cite-listed-leaves
    assert "ONLY CITE LISTED LEAVES" in P and "[CITABLE SCHEMA]" in P
    assert "highest_volume" in P  # named as a thing NOT to invent
    # Fix 2: ABSENT only for never-logged
    assert "NO logged history" in P and "NOT ABSENT" in P
    # Fix 3: per-leaf granularity, exactly three parts
    assert "always tag the specific LEAF" in P and "EXACTLY three parts" in P


# ══ Stage 1.6 — scalar vs non-scalar resolution + ranking steer ═════════════

def test_resolve_scalar_leaf_is_ok(pkg, some_exercise):
    idx = C.build_index(pkg)
    s, v = C.resolve_tag(idx, "exercises", some_exercise["name"], "pr.weight")
    assert s == C.OK and C._is_scalar(v)
    s2, v2 = C.resolve_tag(idx, "exercises", some_exercise["name"], "period_volume_lbs")
    assert s2 == C.OK


def test_resolve_list_field_is_nonscalar(pkg):
    idx = C.build_index(pkg)
    # rankings.* and muscle_group_balance.distribution are ranked LISTS
    assert isinstance(pkg["rankings"]["highest_volume"], list)
    s, v = C.resolve_tag(idx, "rankings", "-", "highest_volume")
    assert s == C.OK_NONSCALAR and isinstance(v, list)
    s2, _ = C.resolve_tag(idx, "muscle_group_balance", "-", "distribution")
    assert s2 == C.OK_NONSCALAR


def test_resolve_dict_parent_is_nonscalar(pkg, some_exercise):
    # citing the parent object (pr) instead of its leaf (pr.weight) → OK_NONSCALAR
    idx = C.build_index(pkg)
    s, v = C.resolve_tag(idx, "exercises", some_exercise["name"], "pr")
    assert s == C.OK_NONSCALAR and isinstance(v, dict)


def test_resolve_null_leaf_is_scalar_ok(pkg):
    # a present-but-None leaf is still a scalar leaf (OK, not NOT_FOUND/NONSCALAR)
    idx = C.build_index(pkg)
    # find an exercise with a None-valued scalar leaf (e.g. cardio_note=None)
    name = next((e["name"] for e in pkg["exercises"]
                 if "cardio_note" in e and e["cardio_note"] is None), None)
    if name is None:
        pytest.skip("no exercise with a null cardio_note leaf")
    s, v = C.resolve_tag(idx, "exercises", name, "cardio_note")
    assert s == C.OK and v is None


def test_ok_nonscalar_in_flag_statuses():
    assert C.OK_NONSCALAR in C.FLAG_STATUSES


def test_extract_mixes_ok_and_nonscalar(pkg, some_exercise):
    name = some_exercise["name"]
    draft = (f"PR is 9 lbs [[exercises|{name}|pr.weight]]. "
             f"Highest volume lift [[rankings|-|highest_volume]].")
    cited = C.extract_cited_values(draft, pkg)
    statuses = [c["status"] for c in cited]
    assert C.OK in statuses and C.OK_NONSCALAR in statuses


def test_draft_prompt_has_ranking_scalar_steer():
    from src.analysis_agent import _ANALYSIS_SYSTEM as P
    assert "RANKING / SUPERLATIVE" in P
    # one number → one scalar leaf
    assert "one scalar" in P.lower()
    # Stage 1.7: now points to the flat per-entity ranking targets …
    assert "[[rankings|<name>|highest_volume_lbs]]" in P
    assert "[[exercise_lifecycle|<name>|total_sessions]]" in P
    assert "pct_of_lbs_total" in P
    # … and still forbids the bare "-" list form
    assert "rankings|-|" in P and "muscle_group_balance|-|distribution" in P


# ══ Stage 1.7 — entity-view flat addressing for rankings/lifecycle/distribution ═

def test_rankings_flat_leaf_resolves_scalar(pkg):
    idx = C.build_index(pkg)
    top = pkg["rankings"]["highest_volume"][0]   # {exercise, volume_lbs, volume_kg}
    s, v = C.resolve_tag(idx, "rankings", top["exercise"], "highest_volume_lbs")
    assert s == C.OK and v == top["volume_lbs"]
    stag = pkg["rankings"]["most_stagnant"][0]    # {exercise, value}
    s2, v2 = C.resolve_tag(idx, "rankings", stag["exercise"], "most_stagnant")
    assert s2 == C.OK and v2 == stag["value"]


def test_lifecycle_flat_leaf_resolves_scalar(pkg):
    idx = C.build_index(pkg)
    e = pkg["exercise_lifecycle"]["active"][0]
    s, v = C.resolve_tag(idx, "exercise_lifecycle", e["exercise_name"], "total_sessions")
    assert s == C.OK and v == e["total_sessions"]


def test_distribution_flat_leaf_resolves_scalar(pkg):
    idx = C.build_index(pkg)
    e = pkg["muscle_group_balance"]["distribution"][0]   # {muscle_group, pct_of_lbs_total, …}
    g = e["muscle_group"]
    s, v = C.resolve_tag(idx, "muscle_group_balance", g, "pct_of_lbs_total")
    assert s == C.OK and v == e["pct_of_lbs_total"]


def test_entity_view_preserves_old_dash_behavior(pkg):
    # backward-compat: the bare "-" list form still resolves OK_NONSCALAR (not
    # broken), and a top-level mgb scalar still resolves OK.
    idx = C.build_index(pkg)
    assert C.resolve_tag(idx, "rankings", "-", "highest_volume")[0] == C.OK_NONSCALAR
    assert C.resolve_tag(idx, "muscle_group_balance", "-", "distribution")[0] == C.OK_NONSCALAR
    assert C.resolve_tag(idx, "muscle_group_balance", "-", "push_volume_lbs")[0] == C.OK


def test_entity_view_bad_entity_flags(pkg):
    idx = C.build_index(pkg)
    s, _ = C.resolve_tag(idx, "rankings", "Nonexistent Lift", "most_stagnant")
    assert s == C.MATCH_KEY_FLAG


def test_extract_ranking_flat_tag_round_trips(pkg):
    top = pkg["rankings"]["highest_volume"][0]
    draft = (f"{top['exercise']} is your highest-volume lift at {top['volume_lbs']:.0f} "
             f"lbs [[rankings|{top['exercise']}|highest_volume_lbs]].")
    cited = C.extract_cited_values(draft, pkg)
    assert len(cited) == 1
    assert cited[0]["status"] == C.OK and cited[0]["value"] == top["volume_lbs"]


# ══ Stage 2 — grounding split (cited-scalar payload vs full-package fallback) ══

def _clean_cited(n=2):
    return [{"claim_number": "130", "tag": "[[exercises|X|pr.weight]]",
             "collection": "exercises", "match_key": f"Ex{i}", "field_path": "pr.weight",
             "status": C.OK, "value": 130.0 + i} for i in range(n)]


def test_grounding_context_cheap_when_all_clean(pkg):
    cited = _clean_cited(3) + [{"claim_number": None, "tag": "[[exercises|Z|ABSENT]]",
        "collection": "exercises", "match_key": "Z", "field_path": "ABSENT",
        "status": C.ABSENT_OK, "value": None}]
    g = C.build_grounding_context(cited, pkg)
    assert g["mode"] == "cheap"
    assert len(g["cited_values"]) == 4
    assert g["cited_values"][0]["location"] == "exercises|Ex0|pr.weight"
    assert "package" not in g


# B3: OK_NONSCALAR is NO LONGER in this list — a resolved list-leaf cite is now
# cheap-eligible (carries its own value; see test_list_leaf_cite_uses_cheap_path).
# Only genuinely-UNRESOLVABLE statuses still force the full-package fallback.
@pytest.mark.parametrize("bad", [C.NOT_FOUND, C.UNKNOWN_COLLECTION,
                                 C.MATCH_KEY_FLAG, C.ABSENT_VIOLATION])
def test_grounding_context_full_when_any_unresolvable(pkg, bad):
    cited = _clean_cited(2) + [{"claim_number": "9", "tag": "[[r|-|x]]",
        "collection": "rankings", "match_key": "-", "field_path": "highest_volume",
        "status": bad, "value": None}]
    g = C.build_grounding_context(cited, pkg)
    assert g["mode"] == "full" and g["package"] is pkg


def test_grounding_context_full_when_empty(pkg):
    # empty cited (tag-less / resumed stripped draft) → full-package fallback
    g = C.build_grounding_context([], pkg)
    assert g["mode"] == "full" and g["package"] is pkg


def test_build_grounding_prompt_cheap_is_small_no_package(pkg):
    from src.analysis_agent import _build_grounding_prompt
    g = C.build_grounding_context(_clean_cited(3), pkg)
    prompt = _build_grounding_prompt("Your PR is 130 lbs.", g)
    assert "[CITED VALUES]" in prompt
    assert "exercises|Ex0|pr.weight = 130.0" in prompt
    assert "[WORKOUT PACKAGE]" not in prompt
    # a package-only marker must be absent → the full package was NOT serialized
    assert "_raw_volume_crosscheck" not in prompt
    assert len(prompt.encode()) < 12_000


def test_build_grounding_prompt_full_has_package(pkg):
    from src.analysis_agent import _build_grounding_prompt
    g = C.build_grounding_context([], pkg)          # → full
    prompt = _build_grounding_prompt("answer", g)
    assert "[WORKOUT PACKAGE]" in prompt
    assert "_raw_volume_crosscheck" in prompt        # the full package IS present


def test_misquote_plumbing_cheap_prompt_has_both_numbers(pkg):
    # crafted misquote: answer states 999, cited scalar is 185 → the cheap
    # prompt must carry both so grounding can catch it.
    from src.analysis_agent import _build_grounding_prompt
    cited = [{"claim_number": "999", "tag": "[[exercises|X|pr.weight]]",
              "collection": "exercises", "match_key": "X", "field_path": "pr.weight",
              "status": C.OK, "value": 185.0}]
    g = C.build_grounding_context(cited, pkg)
    prompt = _build_grounding_prompt("Your PR is 999 lbs.", g)
    assert "999" in prompt and "185" in prompt
    assert "[WORKOUT PACKAGE]" not in prompt
