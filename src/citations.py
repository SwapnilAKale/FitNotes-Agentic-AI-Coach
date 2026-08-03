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
    "rankings", "bodyweight", "muscle_ontology_summary",
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


def _muscle_ontology_view(section: dict) -> dict:
    """muscle_ontology_summary.muscles = [{muscle, primary_sets, …}]
    → {muscle: {<scalar field>: value}}, so a per-muscle set count is a citable
    SCALAR. (Top-level scalars — unmapped_sets, unattributed_sets — stay on
    dict_row, so muscle_ontology_summary|-|unmapped_sets is unchanged.)

    Note what is NOT here, deliberately: there is no "lagging", "deficit", or
    "needs_work" leaf anywhere in the section, so a judgement claim has nothing
    to cite and the existing grounding stage rejects it. That is what enforces
    the no-verdict rule — not a phrase blocklist."""
    by_key: dict = {}
    for e in (section or {}).get("muscles", []) or []:
        if not isinstance(e, dict):
            continue
        name = e.get("muscle")
        if name is None:
            continue
        d = by_key.setdefault(str(name), {})
        for k, v in e.items():
            if k != "muscle" and _is_scalar(v):
                d[k] = v
    return by_key


# collection → (entity label for the schema, builder). The section value must be
# a dict (rankings / exercise_lifecycle / muscle_group_balance all are).
_ENTITY_VIEW_BUILDERS = {
    "rankings":                ("exercise",      _rankings_view),
    "exercise_lifecycle":      ("exercise",      _lifecycle_view),
    "muscle_group_balance":    ("muscle_group",  _mgb_distribution_view),
    "muscle_ontology_summary": ("muscle",        _muscle_ontology_view),
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
# B5: a DATE immediately before the tag (ISO or human month-name), anchored at the
# end. Checked BEFORE _NUM_BEFORE_RE so a "…2026-06-25 [[tag]]" date claim yields the
# full date, not the "-25" fragment _NUM_BEFORE_RE would grab.
_MONTHS_ALT = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
               r"|jul(?:y)?|aug(?:ust)?|sept?|sep(?:tember)?|oct(?:ober)?"
               r"|nov(?:ember)?|dec(?:ember)?)")
_DATE_BEFORE_RE = re.compile(
    r"(?i)("
    r"\d{4}-\d{2}-\d{2}"                                              # ISO
    r"|" + _MONTHS_ALT + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"      # June 25, 2026
    r"|\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS_ALT + r",?\s+\d{4}"      # 25 June 2026
    r")\s*$"
)


def _num_or_date_before(before: str):
    """The claim value immediately before a tag: a trailing DATE (full string) when
    present, else the trailing number (B5 — date-aware so a date claim isn't reduced
    to a stray numeric fragment). None when neither is present."""
    dm = _DATE_BEFORE_RE.search(before or "")
    if dm:
        return dm.group(1).strip()
    nm = _NUM_BEFORE_RE.search(before or "")
    return nm.group(1).replace(",", "") if nm else None


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
        number = _num_or_date_before(before)
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


# Statuses that count as a clean, scalar-checkable citation.
_CLEANLY_CITED = frozenset({OK, ABSENT_OK})
# B3: statuses that can use the CHEAP grounding path. OK_NONSCALAR is included — the
# cited leaf RESOLVED (it's a real list/dict, e.g. pain_analysis.pain_occurrences),
# and the cheap payload already carries its value, so grounding checks the claim
# against that small cited list instead of dragging in the ~400 KB package. Only
# genuinely-unresolvable statuses (NOT_FOUND / UNKNOWN_COLLECTION / MATCH_KEY_FLAG /
# ABSENT_VIOLATION) still force full — so a fabricated cite (B4) stays fully scrutinized.
_CHEAP_ELIGIBLE = frozenset({OK, ABSENT_OK, OK_NONSCALAR})


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

    BOTH modes also carry "display_sets" (the package's pre-formatted verbatim
    per-set lines, [] when absent): ground_check excises those lines from the
    draft BEFORE the grounding LLM sees it and reinserts them after, so the
    checker structurally cannot edit or delete a verbatim log line (it deleted
    a warmup line and "corrected" a true total-set count live). Cheap mode has
    no package, so the lines must ride the context.

    The full-package fallback is the SAFETY path: it fires ~never (the live
    re-check was 7/7 answers clean) and can't miss a claim's support, so it is
    deliberately the complete check rather than an optimized subset.
    """
    display_sets = (package or {}).get("display_sets") or []
    if cited and all(c.get("status") in _CHEAP_ELIGIBLE for c in cited):
        cited_values = [
            {
                "claim_number": c.get("claim_number"),
                "location":     f"{c['collection']}|{c['match_key']}|{c['field_path']}",
                "value":        c.get("value"),
            }
            for c in cited
        ]
        return {"mode": "cheap", "cited_values": cited_values,
                "display_sets": display_sets}
    return {"mode": "full", "package": package, "display_sets": display_sets}


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


# ── Recency guard (structural, pure — no LLM) ─────────────────────────────────
# The "most recent / latest / last session" date is a DETERMINISTIC fact the
# package already computed (progression.latest_session_date). The draft model is
# not allowed to override it, but a generative model writing prose can still bind
# that predicate to the wrong date — a salient pain/comment session, a stale one,
# or an invented one (Issue 3's wrong-predicate class). Prompt guidance lowers the
# odds but can't drive them to zero, so this guard is the GUARANTEE: every
# most-recent claim's date MUST equal that exercise's true latest; any mismatch is
# rewritten in place. It anchors on the PREDICATE, so only the date inside a
# most-recent clause is ever touched — an ordinary "pain on <date>" mention is
# never looked at. On ambiguous exercise scope it leaves the text unchanged (safe
# fallback = today's behaviour). Same house pattern as the display-line guard.

# "most recent | latest | last  [<up to 4 words, e.g. an exercise name>]  session|workout"
_RECENCY_PREDICATE_RE = re.compile(
    r"(?i)\b(?:most recent|latest|last)\b((?:\s+[A-Za-z][\w'&/-]*){0,4}?)\s+(?:session|workout)\b"
)
# Qualifiers that change "last X session" into something OTHER than the most
# recent session overall (a PR/heaviest/deload session, etc.). If one appears in
# the words between the keyword and "session", the claim is NOT a most-recent-
# overall claim → leave it alone.
_PREDICATE_DISALLOW = re.compile(
    r"(?i)\b(?:pr|prs|personal|record|best|heaviest|strongest|hardest|max|maximum"
    r"|heavy|light|failed|worst|deload)\b"
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# The claim's date must sit within this window after the predicate and not cross
# a sentence boundary (so we never reach into the next sentence's date).
_RECENCY_DATE_WINDOW = 90
_SENTENCE_BOUNDARY_RE = re.compile(r"[.;\n]")

# Human date forms the model actually writes in prose — ISO appears only inside
# the verbatim display block, never in the claim sentence, so an ISO-only matcher
# would never fire on real output. Month names are validated against a static map
# (no dateutil). Numeric slash forms (06/25/2026) are intentionally left alone:
# M/D vs D/M is ambiguous, so a no-op is the safe posture ([[fallback-path-safe-not-fast]]).
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]
_MONTH_NUM: dict = {}
for _i, _m in enumerate(_MONTH_NAMES, 1):
    _MONTH_NUM[_m.lower()] = _i
    _MONTH_NUM[_m[:3].lower()] = _i
_MONTH_NUM["sept"] = 9
# "June 25, 2026" / "Jun 25 2026" (month first); "25 June 2026" (day first).
_MFIRST_DATE_RE = re.compile(r"(?i)\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DFIRST_DATE_RE = re.compile(r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(\d{4})\b")


def _iso_ymd(y, m, d) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def _find_recency_date(window: str):
    """
    Earliest date token in `window`, ISO or human month-name form. Returns
    (start, end, iso_norm, raw, style) or None. Regex only LOCATES; the match is
    compared on `iso_norm`. `style` lets the correction re-render in the SAME shape
    the model used. Numeric slash dates are intentionally ignored.
    """
    cands = []
    m = _ISO_DATE_RE.search(window)
    if m:
        cands.append((m.start(), m.end(), m.group(0), m.group(0), {"kind": "iso"}))
    for m in _MFIRST_DATE_RE.finditer(window):
        num = _MONTH_NUM.get(m.group(1).lower())
        if not num:
            continue                                   # a 3-9 letter word that isn't a month
        full = _MONTH_NAMES[num - 1].lower()
        cands.append((m.start(), m.end(), _iso_ymd(m.group(3), num, m.group(2)),
                      m.group(0),
                      {"kind": "mfirst", "abbrev": m.group(1).lower() != full,
                       "comma": "," in m.group(0)}))
        break
    for m in _DFIRST_DATE_RE.finditer(window):
        num = _MONTH_NUM.get(m.group(2).lower())
        if not num:
            continue
        full = _MONTH_NAMES[num - 1].lower()
        cands.append((m.start(), m.end(), _iso_ymd(m.group(3), num, m.group(1)),
                      m.group(0),
                      {"kind": "dfirst", "abbrev": m.group(2).lower() != full}))
        break
    return min(cands, key=lambda c: c[0]) if cands else None


def _render_recency_date(correct_iso: str, style: dict) -> str:
    """Render `correct_iso` (YYYY-MM-DD) in the prose shape the model used, so the
    correction reads naturally (June 15, 2026 → June 25, 2026), not an ISO splice."""
    y, m, d = correct_iso.split("-")
    kind = style.get("kind")
    if kind == "iso":
        return correct_iso
    mon_full = _MONTH_NAMES[int(m) - 1]
    mon = mon_full[:3] if style.get("abbrev") else mon_full
    day = str(int(d))
    if kind == "dfirst":
        return f"{day} {mon} {int(y)}"
    comma = "," if style.get("comma") else ""
    return f"{mon} {day}{comma} {int(y)}"


def _norm_name(s: str) -> str:
    """Fold a name to compare-safe form: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def recency_truth(package: dict) -> dict:
    """
    The deterministic "most recent session" date the answer MUST use, per
    exercise plus a program-wide fallback:
      {"per_exercise": {exercise_name: latest_date}, "global": latest_date | None}
    latest_date = progression.latest_session_date (strength) / last_session_date
    (cardio), falling back to max(session dates). global = training_frequency.
    last_session_date, else the max across exercises. None-tolerant; never raises;
    omits exercises with no derivable date.
    """
    pkg = package or {}
    per: dict = {}
    for ex in pkg.get("exercises", []) or []:
        if not isinstance(ex, dict):
            continue
        name = ex.get("name")
        if not name:
            continue
        prog = ex.get("progression") or {}
        latest = prog.get("latest_session_date") or prog.get("last_session_date")
        if not latest:
            dates = [s.get("date") for s in (ex.get("sessions") or [])
                     if isinstance(s, dict) and s.get("date")]
            latest = max(dates) if dates else None
        if latest:
            per[str(name)] = latest
    tf = pkg.get("training_frequency") or {}
    global_latest = tf.get("last_session_date") or (max(per.values()) if per else None)
    return {"per_exercise": per, "global": global_latest}


def recency_guard(answer: str, package: dict) -> tuple:
    """
    Deterministic guarantee (pure, no LLM): every "most recent / latest / last
    <exercise?> session|workout" claim MUST carry that exercise's true latest
    session date. Any other date bound to such a predicate — a salient pain date,
    a stale date, or an invented one — is a wrong-predicate bind and is rewritten
    in place to the true latest; a flag is recorded. Only the date INSIDE the
    predicate's own clause is touched. On ambiguous exercise scope the claim is
    left unchanged (safe fallback). Returns (corrected_answer, flags).
    """
    if not answer or not answer.strip():
        return answer, []
    truth = recency_truth(package)
    per, global_latest = truth["per_exercise"], truth["global"]
    if not per and not global_latest:
        return answer, []

    norm_per    = {_norm_name(n): (n, d) for n, d in per.items()}
    single_name = next(iter(per)) if len(per) == 1 else None

    edits: list = []      # (date_start, date_end, correct_date)
    flags: list = []

    for pm in _RECENCY_PREDICATE_RE.finditer(answer):
        middle = pm.group(1) or ""

        # ── Locate the date inside the predicate's own clause ────────────────
        win_start = pm.end()
        win = answer[win_start: win_start + _RECENCY_DATE_WINDOW]
        b = _SENTENCE_BOUNDARY_RE.search(win)
        if b:
            win = win[: b.start()]
        hit = _find_recency_date(win)
        if hit is None:
            continue
        d_start, d_end, found_iso, found_raw, style = hit
        date_start = win_start + d_start
        date_end   = win_start + d_end

        # ── Scope the claim to ONE exercise's truth (package facts decide) ────
        correct = scope_name = None
        if middle.strip():
            # A qualified "last X session" (PR/heaviest/deload/…) is not a
            # most-recent-overall claim — never touch it.
            if _PREDICATE_DISALLOW.search(middle):
                continue
            mid_norm = _norm_name(middle)
            for nk, (nm, d) in norm_per.items():
                if nk and nk in mid_norm:          # a known exercise is named
                    correct, scope_name = d, nm
                    break
            if correct is None:
                # Non-empty qualifier that isn't a known exercise → can't
                # attribute safely → leave unchanged.
                continue
        else:
            # Bare "most recent session": one exercise in scope → its latest;
            # a purely program-level answer → the global latest; a multi-exercise
            # answer → the exercise named nearest BEFORE the predicate.
            if single_name is not None:
                correct, scope_name = per[single_name], single_name
            elif not per and global_latest:
                correct, scope_name = global_latest, "overall"
            else:
                head, best_pos = answer[: pm.start()], -1
                for nk, (nm, d) in norm_per.items():
                    for m in re.finditer(re.escape(nm), head, re.IGNORECASE):
                        if m.start() > best_pos:
                            best_pos, correct, scope_name = m.start(), d, nm
        if correct is None or found_iso == correct:
            continue                                # unattributable, or already right

        replacement = _render_recency_date(correct, style)
        edits.append((date_start, date_end, replacement))
        flags.append({
            "action":    "recency_corrected",
            "exercise":  scope_name,
            "original":  found_raw,
            "corrected": replacement,
            "reason": ("a most-recent/last-session claim carried a date that is "
                       "not the exercise's true latest session date"),
        })

    if not edits:
        return answer, []
    # Apply right-to-left so earlier spans keep their indices.
    out = answer
    for start, end, repl in sorted(edits, key=lambda e: -e[0]):
        out = out[:start] + repl + out[end:]
    return out, flags
