"""
src/citations.py
ONE deterministic citation layer for the analytical draft (pure Python, no LLM) —
same single-source discipline as src/units.py / src/shared/sql_sanitize.py.

The Analysis Agent's draft (analysis_agent.analyze) emits an inline citation tag
after every factual/numeric/hedge/absence claim:

    [[collection|match-key|field-path]]

  collection  — the package section the value lives in (e.g. "exercises",
                "muscle_group_summary", or a top-level dict section).
  match-key   — the row identifier within a list section (the exercise name, the
                muscle-group name, …); "-" for a top-level dict section.
  field-path  — dotted path to the leaf within that row (e.g. "pr.weight",
                "progression.weight_change_pct", "training_frequency.session_count"),
                OR the literal "ABSENT" marker for an absence claim.

This module parses those tags, resolves each against the (deterministically
computed + validated) package, and strips them so the user never sees a tag.

STAGE 1 (this module + wiring) only PROVES the machinery: grounding is unchanged
and still receives the full package + the STRIPPED draft. extract_cited_values is
built and tested now; STAGE 2 will feed its output to grounding in place of the
full package (the latency win).

What is deliberately NOT here: the misquote/fabrication *judgement* (Stage 2's
grounding), and any LLM call. This layer is pure addressing — it returns the
cited leaf values and a per-tag status; it does not decide if a claim is wrong.
"""

import re
from typing import Any, NamedTuple, Optional

# ── Resolution statuses ──────────────────────────────────────────────────────
OK                 = "OK"                  # field-path resolved to a SCALAR leaf value
OK_NONSCALAR       = "OK_NONSCALAR"        # field-path exists but resolves to a list/dict —
                                           # it exists, but gives grounding NO scalar to
                                           # misquote-check (e.g. a ranked list). Not usable.
NOT_FOUND          = "NOT_FOUND"           # row found, but field-path missing (bad path)
MATCH_KEY_FLAG     = "MATCH_KEY_FLAG"      # collection known, match-key not in it (name drift / invented)
UNKNOWN_COLLECTION = "UNKNOWN_COLLECTION"  # collection not addressable
ABSENT_OK          = "ABSENT_OK"           # ABSENT claim correct — match-key genuinely absent
ABSENT_VIOLATION   = "ABSENT_VIOLATION"    # ABSENT claim FALSE — match-key is actually present

ABSENT_MARKER = "ABSENT"

# Statuses that mean a tag is NOT cleanly cited — surfaced by the coordinator's
# health log, and the trigger Stage 2's graceful fallback will use (a claim with
# any of these is routed through full-package grounding instead of the cheap
# cited-scalar path). OK / ABSENT_OK are clean. OK_NONSCALAR is included: the
# path exists but a list/dict can't serve a scalar misquote check.
FLAG_STATUSES = frozenset(
    {OK_NONSCALAR, NOT_FOUND, MATCH_KEY_FLAG, UNKNOWN_COLLECTION, ABSENT_VIOLATION}
)


def _is_scalar(value) -> bool:
    """A leaf usable for a scalar misquote check: a number/string/bool, or a
    present-but-null leaf. A list or dict is NOT scalar (→ OK_NONSCALAR)."""
    return value is None or isinstance(value, (int, float, str, bool))

# ── Collection registry (live package shape) ─────────────────────────────────
# Top-level LIST sections addressable by a per-row match-field. (exercises and
# muscle_group_summary are lists, not name-keyed dicts — this index is required.)
_LIST_COLLECTIONS = {
    "exercises":            "name",
    "muscle_group_summary": "muscle_group",
    "goals":                "exercise_name",
}
# Top-level DICT sections are addressed directly (match-key "-").
# (Any other top-level dict in the package is also resolvable directly; the
# registry below is documentation of the expected ones.)
_DICT_SECTIONS = frozenset({
    "all_time_summary", "muscle_group_balance", "training_consistency",
    "day_of_week_patterns", "training_density", "exercise_lifecycle",
    "rankings", "bodyweight",
})


# ── Entity views (Stage 1.7): flat-address the dict-of-lists / nested-list ─────
# sections so ranking/superlative claims cite a SCALAR (not a ranked list, which
# resolved OK_NONSCALAR). Each builder inverts the section into
# {entity: {leaf: scalar}}; build_index keeps the original section dict as
# dict_row so the old "-" tags (e.g. muscle_group_balance|-|push_volume_lbs)
# still resolve exactly as before. ACCEPTED duplication: a ranking number is now
# citable both here and via the per-exercise leaf — one complete grounding check
# per claim beats one-path-per-fact (locked with the user).

def _rankings_view(section: dict) -> dict:
    """rankings = {ranking_name: [{exercise, value} | {exercise, volume_lbs,
    volume_kg}]} → {exercise: {ranking_name (or _lbs/_kg): scalar}}."""
    by_key: dict = {}
    for ranking_name, entries in (section or {}).items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            ex = e.get("exercise")
            if ex is None:
                continue
            d = by_key.setdefault(str(ex), {})
            if "value" in e and _is_scalar(e["value"]):
                d[ranking_name] = e["value"]
            if "volume_lbs" in e and _is_scalar(e["volume_lbs"]):
                d[f"{ranking_name}_lbs"] = e["volume_lbs"]
            if "volume_kg" in e and _is_scalar(e["volume_kg"]):
                d[f"{ranking_name}_kg"] = e["volume_kg"]
    return by_key


def _lifecycle_view(section: dict) -> dict:
    """exercise_lifecycle = {status: [{exercise_name, total_sessions, …}]}
    → {exercise_name: {<scalar field>: value}} (status lists collapse by name)."""
    by_key: dict = {}
    for entries in (section or {}).values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            ex = e.get("exercise_name")
            if ex is None:
                continue
            d = by_key.setdefault(str(ex), {})
            for k, v in e.items():
                if k != "exercise_name" and _is_scalar(v):
                    d[k] = v
    return by_key


def _mgb_distribution_view(section: dict) -> dict:
    """muscle_group_balance.distribution = [{muscle_group, pct_of_lbs_total, …}]
    → {muscle_group: {<scalar field>: value}}. (Top-level mgb scalars stay on
    dict_row, so muscle_group_balance|-|push_volume_lbs is unchanged.)"""
    by_key: dict = {}
    for e in (section or {}).get("distribution", []) or []:
        if not isinstance(e, dict):
            continue
        g = e.get("muscle_group")
        if g is None:
            continue
        d = by_key.setdefault(str(g), {})
        for k, v in e.items():
            if k != "muscle_group" and _is_scalar(v):
                d[k] = v
    return by_key


# collection → (entity label for the schema, builder). The section value must be
# a dict (rankings / exercise_lifecycle / muscle_group_balance all are).
_ENTITY_VIEW_BUILDERS = {
    "rankings":             ("exercise",      _rankings_view),
    "exercise_lifecycle":   ("exercise",      _lifecycle_view),
    "muscle_group_balance": ("muscle_group",  _mgb_distribution_view),
}


class CitationTag(NamedTuple):
    raw:               str
    collection:        str
    match_key:         str
    field_path:        str
    associated_number: Optional[str]   # nearest number preceding the tag, if any


# ── Tag patterns ─────────────────────────────────────────────────────────────
# Strict: a well-formed, resolvable tag (exactly three pipe-separated parts).
_TAG_RE = re.compile(r"\[\[([^|\]]+)\|([^|\]]+)\|([^\]]+)\]\]")
# Broad: ANY [[...]] bracket pair — strip() removes these even if malformed, so
# no tag residue can ever reach the user.
_ANY_BRACKET_RE = re.compile(r"\[\[[^\]]*\]\]")
# A number (optionally with thousands separators / decimal) just before the tag,
# possibly followed by a unit word — best-effort, for Stage-2 misquote checks.
_NUM_BEFORE_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*[A-Za-z%]*\s*$")


def build_index(package: dict) -> dict:
    """
    Build the addressing structure resolve_tag() walks.

    For each top-level key:
      - dict  → {"kind": "dict", "row": <section dict>}            (match-key "-")
      - list  → if the collection has a registered match-field:
                  {"kind": "list", "match_field": mf, "by_key": {str(row[mf]): row}}
                else:
                  {"kind": "unkeyed_list"}  (not addressable by a key → flagged)
    """
    index: dict = {}
    for key, val in (package or {}).items():
        if isinstance(val, dict):
            index[key] = {"kind": "dict", "row": val}
        elif isinstance(val, list):
            mf = _LIST_COLLECTIONS.get(key)
            if mf:
                by_key = {
                    str(r.get(mf)): r
                    for r in val
                    if isinstance(r, dict) and r.get(mf) is not None
                }
                index[key] = {"kind": "list", "match_field": mf, "by_key": by_key}
            else:
                index[key] = {"kind": "unkeyed_list"}

    # Stage 1.7: upgrade the dict-of-lists sections to entity views so their
    # ranked entries are flat-addressable by entity. dict_row keeps the original
    # section so the old "-" tags resolve exactly as before (backward-compatible).
    for coll, (label, builder) in _ENTITY_VIEW_BUILDERS.items():
        section = (package or {}).get(coll)
        if isinstance(section, dict):
            index[coll] = {
                "kind":         "entity_view",
                "entity_label": label,
                "by_key":       builder(section),
                "dict_row":     section,
            }
    return index


def parse_tags(draft_text: str) -> list:
    """Extract every well-formed [[collection|match-key|field-path]] tag (in
    order, robust to mid-sentence placement) with the number it trails."""
    tags: list = []
    for m in _TAG_RE.finditer(draft_text or ""):
        collection = m.group(1).strip()
        match_key  = m.group(2).strip()
        field_path = m.group(3).strip()
        before = draft_text[: m.start()]
        num_m  = _NUM_BEFORE_RE.search(before)
        number = num_m.group(1).replace(",", "") if num_m else None
        tags.append(CitationTag(m.group(0), collection, match_key, field_path, number))
    return tags


def _walk(row: Any, field_path: str):
    """Walk a dotted field-path into nested dicts. Returns (value, found_bool).
    STAGE-1 boundary: if the walk hits a list (can't be keyed here), it is a
    safe, non-silent miss (found=False → NOT_FOUND)."""
    cur = row
    for seg in field_path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None, False
    return cur, True


def resolve_tag(index: dict, collection: str, match_key: str, field_path: str):
    """Resolve one tag against build_index() output. Returns (status, value)."""
    entry = index.get(collection)
    if entry is None or entry.get("kind") == "unkeyed_list":
        return UNKNOWN_COLLECTION, None

    # ── Absence claim ──────────────────────────────────────────────────────
    if field_path == ABSENT_MARKER:
        if entry["kind"] not in ("list", "entity_view"):
            # ABSENT only meaningful for a membership collection.
            return MATCH_KEY_FLAG, None
        present = match_key in entry["by_key"]
        return (ABSENT_VIOLATION, match_key) if present else (ABSENT_OK, None)

    # ── Locate the row ─────────────────────────────────────────────────────
    if entry["kind"] == "dict":
        row = entry["row"]                       # match-key "-" → the section dict
    elif entry["kind"] == "entity_view":
        if match_key == "-":
            row = entry["dict_row"]              # old "-" behavior preserved
        else:
            row = entry["by_key"].get(match_key)  # NEW: flat per-entity scalars
            if row is None:
                return MATCH_KEY_FLAG, None
    else:
        row = entry["by_key"].get(match_key)
        if row is None:
            # Known list, key not in it → draft altered/invented a name. FLAG.
            return MATCH_KEY_FLAG, None

    # ── Walk the field-path to a leaf (value / n / confidence_label) ───────
    value, found = _walk(row, field_path)
    if not found:
        return NOT_FOUND, None
    # A scalar is usable for grounding's misquote check; a list/dict (e.g. a
    # ranked rankings.* list, or a parent object cited instead of its leaf) is
    # not — flag it OK_NONSCALAR so it can't pass as clean.
    return (OK, value) if _is_scalar(value) else (OK_NONSCALAR, value)


def strip_tags(draft_text: str) -> str:
    """
    Remove every [[...]] tag (well-formed OR malformed — no residue can reach the
    user), then tidy whitespace: collapse double spaces/tabs, drop spaces before
    punctuation and after opening brackets, trim line-trailing spaces.
    """
    out = _ANY_BRACKET_RE.sub("", draft_text or "")
    out = re.sub(r"[ \t]{2,}", " ", out)                # collapse runs of spaces
    out = re.sub(r"[ \t]+([.,;:!?)\]])", r"\1", out)    # space before punctuation
    out = re.sub(r"([(\[])[ \t]+", r"\1", out)          # space after open bracket
    out = re.sub(r"[ \t]+\n", "\n", out)                # trailing spaces per line
    return out.strip()


def extract_cited_values(draft_text: str, package: dict) -> list:
    """
    The small payload Stage 2 will feed to grounding (built + verified now).
    Parse every tag → resolve against the package → return one record per tag:
      {claim_number, tag, collection, match_key, field_path, status, value}
    """
    index = build_index(package)
    out: list = []
    for t in parse_tags(draft_text):
        status, value = resolve_tag(index, t.collection, t.match_key, t.field_path)
        out.append({
            "claim_number": t.associated_number,
            "tag":          t.raw,
            "collection":   t.collection,
            "match_key":    t.match_key,
            "field_path":   t.field_path,
            "status":       status,
            "value":        value,
        })
    return out


# Statuses that count as a clean, scalar-checkable citation. Anything else makes
# the WHOLE answer fall back to full-package grounding (Stage 2 binary split).
_CLEANLY_CITED = frozenset({OK, ABSENT_OK})


def build_grounding_context(cited: list, package: dict) -> dict:
    """
    Stage 2 split — PER-ANSWER, binary. Decides what grounding verifies against:

      - CHEAP  : iff `cited` is non-empty AND every record is cleanly cited
                 (status OK scalar / ABSENT_OK). Returns
                 {"mode": "cheap", "cited_values": [{claim_number, location, value}]}
                 — grounding checks each claim against its own cited scalar (a few
                 KB), not the ~415 KB package.
      - FULL   : if ANY record is not cleanly cited (OK_NONSCALAR / NOT_FOUND /
                 UNKNOWN_COLLECTION / MATCH_KEY_FLAG / ABSENT_VIOLATION), OR `cited`
                 is empty (a tag-less draft, or a resumed STRIPPED draft). Returns
                 {"mode": "full", "package": package} — today's EXACT grounding
                 input (whole package).

    The full-package fallback is the SAFETY path: it fires ~never (the live
    re-check was 7/7 answers clean) and can't miss a claim's support, so it is
    deliberately the complete check rather than an optimized subset.
    """
    if cited and all(c.get("status") in _CLEANLY_CITED for c in cited):
        cited_values = [
            {
                "claim_number": c.get("claim_number"),
                "location":     f"{c['collection']}|{c['match_key']}|{c['field_path']}",
                "value":        c.get("value"),
            }
            for c in cited
        ]
        return {"mode": "cheap", "cited_values": cited_values}
    return {"mode": "full", "package": package}


# ── Citable-field schema (generated FROM the package — never hand-written) ────
# The live re-check showed the draft model confabulates plausible-but-nonexistent
# field names (highest_volume, best_e1rm, pct_of_lbs_total) when told the tag
# FORMAT but not the real field SCHEMA. build_citable_schema derives the allowed
# (collection, field-path) leaves FROM the actual package, so it is always in
# sync by construction and per-scope-correct (broad-dropped fields just aren't
# listed). It is injected into the draft prompt; the model may cite only these.

def _scalar_leaf_paths(obj, prefix: str = "") -> set:
    """
    Dotted paths to SCALAR leaves under a dict. Descends nested dicts; SKIPS
    list-valued fields (not addressable by the flat 3-part tag) and skips None
    values (nothing citable). Returns a set of dotted path strings.
    """
    out: set = set()
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out |= _scalar_leaf_paths(v, path)
        elif isinstance(v, list):
            continue                      # list fields not flat-addressable
        elif v is None:
            continue                      # nothing to cite
        else:
            out.add(path)
    return out


def build_citable_schema(package: dict) -> str:
    """
    A compact, human-readable listing of the REAL citable (collection|match-key|
    field-path) leaves present in THIS package — leaf names only, no values
    (≈6 KB on a broad package, not the ~358 KB data). Injected into the draft
    prompt so the model can ONLY cite fields that actually exist.

    - List sections (exercises, muscle_group_summary, goals): union the scalar
      leaf paths across rows (one shared schema — rows share shape), labelled
      with the match-key field.
    - Top-level dict sections: scalar leaf paths, match-key "-".
    - Entity views (Stage 1.7): rankings / exercise_lifecycle / the per-group
      muscle_group_balance distribution are flat-addressable by entity, so their
      ranked scalars (highest_volume_lbs, most_stagnant, pct_of_lbs_total, …) are
      listed and citable as scalars.
    Defensive: missing/empty sections are skipped; an empty package yields a
    minimal string. Never raises.
    """
    pkg = package or {}
    lines: list = []

    # List sections — union leaf paths across rows.
    for coll, match_field in _LIST_COLLECTIONS.items():
        rows = pkg.get(coll)
        if not isinstance(rows, list) or not rows:
            continue
        leaves: set = set()
        for row in rows:
            leaves |= _scalar_leaf_paths(row)
        if leaves:
            lines.append(
                f"{coll} (match-key = the {match_field}): "
                + ", ".join(sorted(leaves))
            )

    # Top-level dict sections — addressed directly with match-key "-".
    for coll in sorted(_DICT_SECTIONS):
        section = pkg.get(coll)
        if not isinstance(section, dict):
            continue
        leaves = _scalar_leaf_paths(section)
        if leaves:
            lines.append(f"{coll} (match-key = -): " + ", ".join(sorted(leaves)))

    # Entity views (Stage 1.7) — flat per-entity scalar leaves for the
    # dict-of-lists / nested-list sections, so ranking/superlative claims cite a
    # scalar instead of a ranked list.
    for coll, (label, builder) in _ENTITY_VIEW_BUILDERS.items():
        section = pkg.get(coll)
        if not isinstance(section, dict):
            continue
        leaves = set()
        for row in builder(section).values():
            leaves |= _scalar_leaf_paths(row)
        if leaves:
            lines.append(
                f"{coll} (match-key = the {label}): " + ", ".join(sorted(leaves))
            )

    if not lines:
        return "(no citable fields available for this question)"

    guidance = (
        "GUIDANCE: cite ONLY a leaf listed above. For a NUMERIC/FACTUAL claim cite "
        "its value leaf (e.g. pr.weight). For a HEDGE/UNCERTAINTY claim cite the "
        "leaf that justifies it — a count or confidence leaf (e.g. "
        "training_frequency.session_count, or a *.comparison.confidence_label). "
        "For an ABSENCE claim use the ABSENT marker. List- and comment-valued "
        "fields are NOT listed and are NOT citable."
    )
    return "\n".join(lines) + "\n" + guidance
