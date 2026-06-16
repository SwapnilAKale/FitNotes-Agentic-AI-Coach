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
OK                 = "OK"                  # field-path resolved to a leaf value
NOT_FOUND          = "NOT_FOUND"           # row found, but field-path missing (bad path)
MATCH_KEY_FLAG     = "MATCH_KEY_FLAG"      # collection known, match-key not in it (name drift / invented)
UNKNOWN_COLLECTION = "UNKNOWN_COLLECTION"  # collection not addressable
ABSENT_OK          = "ABSENT_OK"           # ABSENT claim correct — match-key genuinely absent
ABSENT_VIOLATION   = "ABSENT_VIOLATION"    # ABSENT claim FALSE — match-key is actually present

ABSENT_MARKER = "ABSENT"

# Statuses that indicate a problem worth surfacing (used by the coordinator's
# Stage-1 health log). OK / ABSENT_OK are clean.
FLAG_STATUSES = frozenset(
    {NOT_FOUND, MATCH_KEY_FLAG, UNKNOWN_COLLECTION, ABSENT_VIOLATION}
)

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
        if entry["kind"] != "list":
            # ABSENT only meaningful for a membership collection (a list).
            return MATCH_KEY_FLAG, None
        present = match_key in entry["by_key"]
        return (ABSENT_VIOLATION, match_key) if present else (ABSENT_OK, None)

    # ── Locate the row ─────────────────────────────────────────────────────
    if entry["kind"] == "dict":
        row = entry["row"]                       # match-key "-" → the section dict
    else:
        row = entry["by_key"].get(match_key)
        if row is None:
            # Known list, key not in it → draft altered/invented a name. FLAG.
            return MATCH_KEY_FLAG, None

    # ── Walk the field-path to a leaf (value / n / confidence_label) ───────
    value, found = _walk(row, field_path)
    return (OK, value) if found else (NOT_FOUND, None)


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
