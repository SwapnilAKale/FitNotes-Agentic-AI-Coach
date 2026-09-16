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


def _exercise_muscle_map_view(section: dict) -> dict:
    """exercise_muscle_map = {exercise: {role: [muscle, …], in_store, ever_logged}}
    → {exercise: {role: "Muscle, Muscle"}}, so a "does X train Y" claim cites a
    SCALAR rather than resolving OK_NONSCALAR against a raw list.

    This is the section that grounds the claim the agent previously invented.
    The roles stay THREE SEPARATE LEAVES — there is deliberately no combined
    "muscles_worked" leaf, so a claim that a lift trains a muscle it merely
    holds has nothing to cite and the grounding stage rejects it. Same
    enforcement pattern as _muscle_ontology_view: the shape of the data is what
    makes the false claim uncitable.

    dict_row keeps the original section, so the raw lists stay addressable."""
    by_key: dict = {}
    for name, entry in (section or {}).items():
        if not isinstance(entry, dict):
            continue
        row: dict = {}
        for role in ("primary", "secondary", "limiting", "trains", "holds_only"):
            vals = entry.get(role)
            row[role] = ", ".join(vals) if isinstance(vals, list) else vals
        for flag in ("in_store", "ever_logged"):
            if flag in entry:
                row[flag] = entry[flag]
        by_key[str(name)] = row
    return by_key


# collection → (entity label for the schema, builder). The section value must be
# a dict (rankings / exercise_lifecycle / muscle_group_balance all are).
_ENTITY_VIEW_BUILDERS = {
    "rankings":                ("exercise",      _rankings_view),
    "exercise_lifecycle":      ("exercise",      _lifecycle_view),
    "muscle_group_balance":    ("muscle_group",  _mgb_distribution_view),
    "muscle_ontology_summary": ("muscle",        _muscle_ontology_view),
    "exercise_muscle_map":     ("exercise",      _exercise_muscle_map_view),
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


# ── A HELD MUSCLE IS NEVER A TRAINED MUSCLE ───────────────────────────────────
#
# Same reason recency_guard exists, and the same shape. Live, the draft cited
# `limiting` CORRECTLY — one tag, clean — and still wrote:
#
#   "Actually, your Lat Pulldowns do train your biceps; the biceps act as a
#    limiting muscle, meaning they hold and stabilize the load..."
#
# The citation gate proves WHICH FIELD the model read. It cannot police the verb
# in the sentence wrapped around it, and a truthful citation of `limiting` sits
# happily inside "do train". Prompt guidance lowers the rate; only this makes it
# an invariant.
#
# THE MAP DECIDES; THE REGEX ONLY FINDS THE CLAIM. exercise_muscle_map is the
# structural fact. Prose matching is the secondary claim-PRESENCE detector and
# nothing more — it never decides what is true, only where a sentence is making
# the assertion.

_TRAIN_VERB_RE = re.compile(
    r"\b(?:train(?:s|ed|ing)?|work(?:s|ed|ing)?|build(?:s|ing)?|"
    r"develop(?:s|ing|ment)?|grow(?:s|ing)?|hit(?:s|ting)?|"
    r"target(?:s|ing|ed)?|stimulat(?:e|es|ed|ing|ion)?|"
    r"engag(?:e|es|ed|ing)|activat(?:e|es|ed|ing|ion))\b", re.I)

# Any of these inside the clause means the sentence is not asserting the claim.
# The second group is DESCRIPTIVE, not negating: a clause that already explains
# the muscle is limiting/holding/stabilising is saying the right thing, and
# rewriting it would destroy a correct and more informative sentence.
_NEG_RE = re.compile(
    r"\b(?:not|n't|never|no|nor|without|rather than|instead of|"
    r"doesn|does not|don|do not|isn|is not|aren|are not|only holds?|"
    r"merely|barely|hardly|"
    r"limiting|stabilis\w*|stabiliz\w*|holds?\s+(?:and\s+\w+\s+)?the\s+load|"
    r"holding)\b", re.I)

# A "verb" straight after a determiner is a NOUN — "do the work", "your training".
# Live, the guard rewrote a fully CORRECT sentence because "do the work" at the
# end of it matched the training-verb pattern, destroying the useful half of the
# answer. Worse, the guard's own repair text ends "...do the work", so without
# this it was not even idempotent: its own output re-triggered it.
_DET_BEFORE_RE = re.compile(
    r"\b(?:the|a|an|your|my|its|his|her|their|this|that|all|some|any|no)\s+$",
    re.I)


# Passive voice: "the biceps are trained by ...". The active test below is
# position-based, so the passive needs naming explicitly or it slips through.
_PASSIVE_RE = re.compile(
    r"\b(?:are|is|was|were|get|gets|got)\s+(?:\w+\s+){0,2}"
    r"(?:trained|worked|built|developed|targeted|stimulated|hit)\b", re.I)


def _asserts_training(clause: str, muscle_pos: int) -> bool:
    """True only for a training verb ASSERTED OF THE MUSCLE at `muscle_pos`.

    The verb must PRECEDE the muscle — "trains your biceps" asserts it, while
    "the biceps get a little work" and "...while your lats do the work" do not.
    Position is what separates the verb sense from the noun sense reliably;
    a determiner check alone let "a little work" and "Lat Pulldown work"
    through.
    """
    for m in _TRAIN_VERB_RE.finditer(clause):
        if m.start() >= muscle_pos:
            continue                       # verb comes after the muscle
        if _DET_BEFORE_RE.search(clause[:m.start()]):
            continue                       # "do the work" — a noun
        return True
    return bool(_PASSIVE_RE.search(clause))

# Clauses within a sentence — the verb and the muscle must co-occur in ONE of
# them, so "trains your lats, while the biceps only hold" is not a violation.
_CLAUSE_SPLIT_RE = re.compile(
    r"[,;:]| — | -- |\bwhile\b|\bwhereas\b|\bthough\b|\balthough\b|\bbut\b")
# Sentences, with their offsets preserved so a repair can be spliced back.
_SENTENCE_SPAN_RE = re.compile(r"[^.!?\n]+[.!?]*\n?")


def limiting_truth(package: dict) -> dict:
    """Per exercise: which muscles it merely HOLDS, and which it actually trains.

    Straight off exercise_muscle_map, which is graph-sourced and
    window-independent — so this guard works even when the window carries no
    sets for the exercise, which is exactly when the model is most tempted to
    improvise.
    """
    out: dict = {}
    for name, entry in (package or {}).get("exercise_muscle_map", {}).items():
        if not isinstance(entry, dict):
            continue
        holds = [m for m in (entry.get("holds_only") or entry.get("limiting") or [])]
        if not holds:
            continue
        out[str(name)] = {
            "holds":   list(holds),
            "trains":  list(entry.get("trains") or []),
            "primary": list(entry.get("primary") or []),
        }
    return out


def _held_claim_repair(exercise: str, muscle: str, primary: list) -> str:
    """The replacement sentence, built ENTIRELY from the map.

    A whole sentence is replaced rather than the verb spliced: substitution at
    clause level is what mangles prose, while a generated sentence is always
    grammatical and always says exactly what the graph says.
    """
    low = muscle.lower()
    tail = (f" — the {low} hold the load while your "
            f"{_join_names(primary)} do the work." if primary
            else f" — the {low} hold the load without being trained by it.")
    return f"Your {exercise} does not train your {muscle}{tail}"


def _join_names(names: list) -> str:
    names = [str(n) for n in names if str(n).strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def limiting_claim_guard(answer: str, package: dict) -> tuple:
    """
    Deterministic guarantee (pure, no LLM): no sentence may assert that an
    exercise TRAINS a muscle the graph says it only HOLDS.

    A violating sentence is replaced with one generated from the map; anything
    ambiguous is left untouched and flagged, the same safe fallback
    recency_guard uses. Returns (corrected_answer, flags).
    """
    if not answer or not answer.strip():
        return answer, []
    truth = limiting_truth(package)
    if not truth:
        return answer, []

    norm_truth = {_norm_name(k): (k, v) for k, v in truth.items()}
    edits: list = []
    flags: list = []

    for sm in _SENTENCE_SPAN_RE.finditer(answer):
        sentence = sm.group(0)
        s_norm = _norm_name(sentence)

        # Which of the mapped exercises does this sentence name?
        named = [(orig, data) for key, (orig, data) in norm_truth.items()
                 if key and key in s_norm]
        if not named:
            continue

        hit = None
        for exercise, data in named:
            for muscle in data["holds"]:
                m_norm = _norm_name(muscle)
                for clause in _CLAUSE_SPLIT_RE.split(sentence):
                    c_low = clause.lower()
                    m_at = c_low.find(m_norm)
                    if m_at < 0:
                        continue
                    if not _asserts_training(clause, m_at):
                        continue
                    if _NEG_RE.search(clause):
                        continue          # already phrased correctly
                    hit = (exercise, muscle, data)
                    break
                if hit:
                    break
            if hit:
                break
        if not hit:
            continue

        exercise, muscle, data = hit
        repair = _held_claim_repair(exercise, muscle, data["primary"])
        # Preserve the original trailing whitespace/newline layout.
        trail = sentence[len(sentence.rstrip()):]
        edits.append((sm.start(), sm.end(), repair + trail))
        flags.append({
            "kind":      "limiting_claim",
            "exercise":  exercise,
            "muscle":    muscle,
            "original":  sentence.strip(),
            "corrected": repair,
            "reason": (f"the graph records {muscle} as HELD by {exercise}, not "
                       f"trained by it; a held muscle is never training volume"),
        })

    if not edits:
        return answer, []
    out = answer
    for start, end, repl in sorted(edits, key=lambda e: -e[0]):
        out = out[:start] + repl + out[end:]
    return out, flags


# ── A MUSCLE IS NOT TRAINED DIRECTLY TWO DAYS RUNNING ─────────────────────────
#
# Live, a one-week plan put Barbell Curl on Day 4 and Seated Machine Curl on
# Day 5 — direct biceps work on consecutive days. Every fact needed to catch
# that was already in the ontology, and nothing looked at it. The graph informed
# the answer but never CHECKED it, which is the same defect limiting_claim_guard
# was built for.
#
# THE GRAPH DECIDES; THE REGEX ONLY LOCATES. Day headings and exercise mentions
# are found by pattern, but what a lift trains — and therefore whether two days
# clash — comes from edges_by_exercise. Exercise names are a CLOSED vocabulary
# read from the store, never an open text match.
#
# SCOPED NARROWLY ON PURPOSE: primary work only, adjacent day numbers only, and
# only when the answer actually looks like a plan. High-frequency training is a
# legitimate choice; this catches the accidental clash, and every hit is flagged.

# An explicit "Day 3" / "| Day 3 |" — always a day.
# The leading class absorbs markdown chrome — "### Day 1", "- Day 2", "**Day 3**"
# are all headings the model actually produces.
_DAY_WORD_RE = re.compile(r"(?:^|\n|\|)[\s#*\-]*day\s*([1-9])\b", re.I)
# A table row opening with a bare number — a day ONLY when the table declares a
# Day column. A BARE NUMBER IS NOT A DAY ON ITS OWN: "1. Seated Machine Curl —
# 416 sets / 2. Cable Curl — 156 sets" is a ranked list, and reading it as a
# two-day plan flagged a clash in an answer containing no plan at all.
_DAY_CELL_RE = re.compile(r"(?:^|\n)\s*\|\s*(?:\*\*)?\s*([1-9])\s*(?:\*\*)?\s*\|")
_DAY_HEADER_RE = re.compile(r"\|\s*(?:\*\*)?\s*day\s*(?:\*\*)?\s*\|", re.I)

# WEEKDAY NAMES ARE DAYS TOO. Every plan in the live check used Mon/Tue/Thu/Fri
# rather than "Day N", so the guard scored zero on three plans that all violated.
# Mapping to 1-7 feeds the SAME adjacency test: Mon/Tue clash, Tue/Thu do not.
_WEEKDAYS = {"monday": 1, "mon": 1, "tuesday": 2, "tue": 2, "tues": 2,
             "wednesday": 3, "wed": 3, "thursday": 4, "thu": 4, "thur": 4,
             "thurs": 4, "friday": 5, "fri": 5, "saturday": 6, "sat": 6,
             "sunday": 7, "sun": 7}
_WEEKDAY_RE = re.compile(
    r"(?:^|\n|\|)[\s#*\-]*(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
    + r")\b", re.I)


def _plan_days(answer: str) -> list:
    """[(day_number, text_of_that_day)] — a plan's rows, in order.

    A day is the literal word "Day N", a weekday name, or a table row starting
    with a bare number IN A TABLE THAT DECLARES A DAY COLUMN. Requiring one of
    those is what separates a training plan from a numbered list.
    """
    text = answer or ""
    hits = [(int(m.group(1)), m.start(), m.end())
            for m in _DAY_WORD_RE.finditer(text)]
    hits += [(_WEEKDAYS[m.group(1).lower()], m.start(), m.end())
             for m in _WEEKDAY_RE.finditer(text)]
    if _DAY_HEADER_RE.search(text):
        hits += [(int(m.group(1)), m.start(), m.end())
                 for m in _DAY_CELL_RE.finditer(text)]
    hits.sort(key=lambda h: h[1])
    if len(hits) < 2:
        return []
    out = []
    for i, (day, _s, e) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(answer)
        out.append((day, answer[e:end]))
    return out


def plan_guard(answer: str, ontology: dict) -> tuple:
    """
    Deterministic check (pure, no LLM) over a training plan, at two strengths:

      VIOLATIONS — a muscle trained DIRECTLY on two consecutive days. Rebuild.
      CONCERNS   — a muscle that took SECONDARY work yesterday is targeted
                   directly today. Reported, not blocked: push/pull/legs on
                   back-to-back days is a legitimate structure, so the coach may
                   keep it provided it says why.

    Returns (violations, flags, concerns). It does NOT rewrite — a training plan
    cannot be regenerated in code, so the caller re-prompts once with the clash
    named and, failing that, states it plainly rather than shipping it silently.
    """
    if not answer or not ontology or not ontology.get("edges_by_exercise"):
        return [], [], []
    days = _plan_days(answer)
    if len(days) < 2:
        return [], [], []

    muscles = ontology.get("muscles", {})
    exercises = ontology.get("exercises", {})
    # Closed vocabulary: canonical names plus every logged alias, longest first
    # so "Seated Machine Curl" wins over "Machine Curl".
    vocab = {}
    for eid, ex in exercises.items():
        vocab.setdefault(_norm_name(ex["canonical_name"]), eid)
    for db_name, eid in (ontology.get("aliases") or {}).items():
        vocab.setdefault(_norm_name(db_name), eid)
    names = sorted(vocab, key=len, reverse=True)

    def worked(segment: str, role: str) -> dict:
        """{muscle_id: exercise_name} for muscles hit in this ROLE here.

        Keyed by ID, not name: Triceps Long Head and Triceps are different names
        for an overlapping claim on the same tissue, and only the ids carry the
        ancestry needed to see that.
        """
        seg = _norm_name(segment)
        found: dict = {}
        # LONGEST NAME WINS ITS SPAN. Plain substring matching bled badly:
        # "Incline Smith Machine Press" contains "Smith Machine Press", so an
        # incline day (Upper Chest) also counted as the flat lift (Mid Chest)
        # and produced a clash between two days that share no muscle at all.
        # Each matched span is consumed so a shorter name cannot re-match inside
        # a longer one already claimed.
        taken = []                       # [(start, end)] of consumed spans

        def free(start, end):
            return not any(s < end and start < e for s, e in taken)

        for nm in names:                 # already sorted longest-first
            if not nm:
                continue
            at = seg.find(nm)
            while at != -1:
                if free(at, at + len(nm)):
                    taken.append((at, at + len(nm)))
                    eid = vocab[nm]
                    for edge in ontology["edges_by_exercise"].get(eid, []):
                        if edge["role"] != role:
                            continue
                        if edge["muscle_id"] in muscles:
                            found.setdefault(edge["muscle_id"],
                                             exercises[eid]["canonical_name"])
                at = seg.find(nm, at + 1)
        return found

    ancestors = ontology.get("ancestors", {})

    def overlapping(a_ids, b_ids):
        """(a_id, b_id) pairs where one muscle CONTAINS the other.

        A clash is an ancestor relationship — Triceps Long Head sits inside
        Triceps — NOT a shared ancestor. Biceps and Triceps both live under Arms
        and are entirely different muscles; comparing via a common ancestor
        would flag every arm split ever written.
        """
        out = []
        for a in a_ids:
            for b in b_ids:
                if a == b or b in ancestors.get(a, ()) or a in ancestors.get(b, ()):
                    out.append((a, b))
        return out

    by_day = [(d, worked(text, "primary"), worked(text, "secondary"))
              for d, text in days]
    def label(mid):
        return muscles[mid]["name"]

    violations, concerns, flags = [], [], []
    for i in range(len(by_day) - 1):
        d1, p1, s1 = by_day[i]
        d2, p2, s2 = by_day[i + 1]
        if d2 != d1 + 1:
            continue                      # not adjacent days

        # HARD: trained directly two days running.
        seen = set()
        for a, b in overlapping(p1, p2):
            key = tuple(sorted((label(a), label(b))))
            if key in seen:
                continue
            seen.add(key)
            same = label(a) if a == b else f"{label(a)} / {label(b)}"
            detail = (f"{p1[a]} on day {d1} and {p2[b]} on day {d2} both train "
                      f"{same} directly")
            violations.append(detail)
            flags.append({
                "kind": "plan_consecutive_days", "severity": "violation",
                "muscle": same, "day_a": d1, "day_b": d2,
                "exercise_a": p1[a], "exercise_b": p2[b],
                "reason": detail + "; direct work needs a day between",
            })

        # SOFT: yesterday's ASSISTING work lands on today's target. This is the
        # arms-day-after-chest-and-back case — pressing loads the triceps as
        # `secondary`, which is exactly what that role was created to record.
        # Reported, never blocked: push/pull/legs on consecutive days is a
        # legitimate structure, so the coach may keep it WITH A REASON.
        seen_soft = set()
        for a, b in overlapping(s1, p2):
            key = tuple(sorted((label(a), label(b))))
            if key in seen or key in seen_soft:
                continue
            seen_soft.add(key)
            same = label(b)
            detail = (f"{s1[a]} on day {d1} already works {same} as a secondary "
                      f"muscle, and {p2[b]} targets it directly on day {d2}")
            concerns.append(detail)
            flags.append({
                "kind": "plan_secondary_interference", "severity": "concern",
                "muscle": same, "day_a": d1, "day_b": d2,
                "exercise_a": s1[a], "exercise_b": p2[b],
                "reason": detail + "; space them or say why it still works",
            })
    return violations, flags, concerns


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER GUARDS FOR THE LIVE-VERIFICATION DEFECTS (B1–B5)
# ══════════════════════════════════════════════════════════════════════════════
#
# The live check of 2026-08-06 got every number right and still shipped:
#   B1  "leg training frequency is low at 5.3 sessions per week"   (5.3 SETS)
#   B2  "25 sets for Chest, 28 for Back, 14 for Shoulders"         (invented)
#   B3  "Hamstring Curls Machine", "T Bar Barbell Row"             (near-miss names)
#   B4  a plan quoting 1.8 sets a week with no target to raise it to
#   B5  "90-day recovery trends"                                   (window label on physiology)
# Prompt rules for B1, B3 and B4 already existed and did not hold. The same
# lesson as recency_guard and limiting_claim_guard: guidance lowers the rate,
# only a deterministic check makes it an invariant. THE PACKAGE DECIDES; THE
# REGEX ONLY LOCATES.

SETS_PER_WEEK_TARGET = (10, 20)

# Sentences, DECIMAL-SAFE. _SENTENCE_SPAN_RE splits "1.8 sets" into "1." and
# "8 sets" — harmless for words, fatal for a guard that reads numbers.
_NUM_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]*\n?")
_BOUNDARY_RE = re.compile(r"(?<!\d)[.!?](?!\d)|\n")


def _muscle_rows(package: dict) -> list:
    mos = (package or {}).get("muscle_ontology_summary") or {}
    return [r for r in (mos.get("muscles") or [])
            if isinstance(r, dict) and r.get("muscle")]


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(a, b, tol: float = 0.05) -> bool:
    fa, fb = _num(a), _num(b)
    return fa is not None and fb is not None and abs(fa - fb) <= tol


def _fmt(value) -> str:
    v = _num(value)
    if v is None:
        return "?"
    return str(int(v)) if v == int(v) else f"{v:g}"


def _splice(answer: str, edits: list) -> str:
    """Apply (start, end, replacement) edits right-to-left, one per span."""
    unique = {(s, e): r for s, e, r in edits}
    out = answer
    for (start, end), repl in sorted(unique.items(), key=lambda kv: -kv[0][0]):
        out = out[:start] + repl + out[end:]
    return out


def _sentence_start(text: str, pos: int) -> int:
    start = 0
    for m in _BOUNDARY_RE.finditer(text, 0, pos):
        start = m.end()
    return start


# ── B1 · a set rate is never a session rate ───────────────────────────────────

_SESSIONS_PER_WEEK_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)(?P<noun>\s+(?:training\s+)?sessions?)"
    r"(?=\s+(?:per|a|each)\s+week\b)", re.I)
_FREQUENCY_WORD_RE = re.compile(r"\bfrequency\b", re.I)


def sets_as_sessions_guard(answer: str, package: dict) -> tuple:
    """
    "5.3 sessions per week" where 5.3 is a muscle's SETS per week and not the
    user's real session rate → "5.3 sets per week"; a "frequency" earlier in the
    same sentence becomes "volume". A number that IS the real session rate, or
    that matches no set rate, is left alone. Returns (answer, flags).
    """
    if not answer or not answer.strip():
        return answer, []
    per_week = [v for r in _muscle_rows(package)
                for v in (r.get("primary_sets_per_week"), r.get("secondary_sets_per_week"))
                if _num(v)]
    if not per_week:
        return answer, []
    real = ((package or {}).get("training_frequency") or {}).get("sessions_per_week")

    edits, flags = [], []
    for m in _SESSIONS_PER_WEEK_RE.finditer(answer):
        n = m.group("num")
        if real is not None and _close(n, real):
            continue                      # it IS the real session rate
        if not any(_close(n, v) for v in per_week):
            continue                      # no set rate either — not ours to judge
        singular = m.group("noun").strip().lower().endswith("session")
        noun = " set" if singular else " sets"
        edits.append((m.start("noun"), m.end("noun"), noun))
        s0 = _sentence_start(answer, m.start())
        for f in _FREQUENCY_WORD_RE.finditer(answer, s0, m.start()):
            word = f.group(0)
            edits.append((f.start(), f.end(), "Volume" if word[0].isupper() else "volume"))
        flags.append({
            "kind": "sets_as_sessions",
            "original": m.group(0).strip(),
            "corrected": f"{n}{noun}",
            "reason": f"{n} is a sets-per-week figure, not the user's session rate",
        })
    return (_splice(answer, edits), flags) if edits else (answer, [])


# ── B5 · a window label is not a physiological property ───────────────────────

_WINDOW_BODY_RE = re.compile(
    r"\b(?P<n>\d+)[-\s]day\s+"
    r"(?P<noun>recovery|fatigue|readiness|soreness|adaptation|growth)"
    r"(?P<tail>\s+(?:trends?|patterns?|data|signals?))?", re.I)
_AT_SENTENCE_START_RE = re.compile(r"(?:^|[.!?]\s*|\n\s*|[#*|\-]\s*)$")


def window_label_guard(answer: str, package: dict) -> tuple:
    """
    "90-day recovery trends" → "recovery trends over the last 90 days", when 90
    is the package's window. A window label on a count ("90-day total", "90-day
    window") is not touched. Returns (answer, flags).
    """
    if not answer or not answer.strip():
        return answer, []
    days = _num((package or {}).get("query_period_days"))
    if days is None:
        return answer, []

    edits, flags = [], []
    for m in _WINDOW_BODY_RE.finditer(answer):
        if int(m.group("n")) != int(days):
            continue
        repl = f"{m.group('noun')}{m.group('tail') or ''} over the last {m.group('n')} days"
        if _AT_SENTENCE_START_RE.search(answer[:m.start()]):
            repl = repl[0].upper() + repl[1:]
        edits.append((m.start(), m.end(), repl))
        flags.append({
            "kind": "window_label",
            "original": m.group(0),
            "corrected": repl,
            "reason": "the analysis window is a time range, not a property of recovery",
        })
    return (_splice(answer, edits), flags) if edits else (answer, [])


# ── B3 · exercise names are copied from the store, exactly ────────────────────

def _tolerant_name_re(name: str):
    """A pattern for a store name that tolerates what the live check produced:
    space for hyphen, and a plural added or dropped on any word. Single-word
    names return None — "your deadlifts" is ordinary prose, not a misspelling."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w]
    if len(words) < 2:
        return None
    parts = []
    for w in words:
        lw = w.lower()
        if lw.isdigit() or len(lw) <= 2:
            parts.append(re.escape(w))
            continue
        if lw.endswith(("ches", "shes", "xes", "sses")):
            base = w[:-2]
        elif lw.endswith("s") and not lw.endswith(("ss", "us", "is")):
            base = w[:-1]
        else:
            base = w
        parts.append(re.escape(base) + r"(?:e?s)?")
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\s\-]*".join(parts)
                      + r"(?![A-Za-z0-9])", re.I)


def exercise_name_guard(answer: str, package: dict, ontology: Optional[dict] = None) -> tuple:
    """
    A near-miss of a store exercise name is rewritten to the exact store name.
    The vocabulary is CLOSED: the package's exercises and suggestable_exercises,
    plus the graph's canonical names. Longest name wins its span, so "Incline
    Smith Machine Press" is never read as "Smith Machine Press". A span that
    already is a store name, in any case, is left alone. Returns (answer, flags).
    """
    if not answer or not answer.strip():
        return answer, []
    pkg = package or {}
    names = set()
    for ex in pkg.get("exercises") or []:
        if isinstance(ex, dict) and ex.get("name"):
            names.add(str(ex["name"]))
    for s in pkg.get("suggestable_exercises") or []:
        n = s.get("name") if isinstance(s, dict) else s
        if n:
            names.add(str(n))
    for ex in ((ontology or {}).get("exercises") or {}).values():
        if isinstance(ex, dict) and ex.get("canonical_name"):
            names.add(str(ex["canonical_name"]))
    if not names:
        return answer, []
    exact = {n.lower() for n in names}

    taken, edits, flags = [], [], []
    for name in sorted(names, key=len, reverse=True):
        pattern = _tolerant_name_re(name)
        if pattern is None:
            continue
        for m in pattern.finditer(answer):
            start, end = m.span()
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            found = m.group(0)
            if found.lower() in exact:
                continue                  # already a real store name
            edits.append((start, end, name))
            flags.append({
                "kind": "exercise_name",
                "original": found,
                "corrected": name,
                "reason": "exercise names are copied exactly from the store",
            })
    return (_splice(answer, edits), flags) if edits else (answer, [])


# ── B2 · a quoted set count must exist in the package ─────────────────────────

_CURRENT_CUE_RE = re.compile(
    r"\bcurrently\b|\bright now\b|"
    r"\byou(?:'re|\s+are)\s+(?:doing|performing|getting|hitting|averaging|training|logging)\b|"
    r"\byou\s+(?:perform|did|logged|performed|trained|got|completed|averaged)\b|"
    r"\b(?:over|in)\s+the\s+(?:last|past)\b", re.I)
_PLAN_CUE_RE = re.compile(
    r"\b(?:aim|target|add|adding|increase|increasing|raise|should|recommend\w*|"
    r"plan|goal|try|bring)\b", re.I)


def _set_count_truth(package: dict) -> dict:
    """{lowercase name: {label, values, facts}} for every muscle and category."""
    pkg = package or {}
    weeks = _num((pkg.get("muscle_ontology_summary") or {}).get("weeks_in_window"))
    truth: dict = {}

    def entry(label):
        return truth.setdefault(str(label).lower(),
                                {"label": str(label), "values": [], "facts": []})

    for r in _muscle_rows(pkg):
        e = entry(r["muscle"])
        for k in ("primary_sets", "secondary_sets", "limiting_sets",
                  "prior_primary_sets", "prior_secondary_sets",
                  "alltime_primary_sets", "alltime_secondary_sets"):
            v = _num(r.get(k))
            if v is not None:
                e["values"].append(v)
        for k in ("primary_sets_per_week", "secondary_sets_per_week"):
            v = _num(r.get(k))
            if v is not None:
                e["values"] += [v, float(round(v))]
        e["facts"].append(f"{_fmt(r.get('primary_sets'))} primary sets over the window "
                          f"({_fmt(r.get('primary_sets_per_week'))} a week)")
    for g in pkg.get("muscle_group_summary") or []:
        if not isinstance(g, dict) or not g.get("muscle_group"):
            continue
        total = _num(g.get("total_sets"))
        if total is None:
            continue
        e = entry(g["muscle_group"])
        e["values"].append(total)
        if weeks:
            per_week = round(total / weeks, 1)
            e["values"] += [per_week, float(round(per_week))]
            e["facts"].append(f"{_fmt(total)} sets over the window ({_fmt(per_week)} a week)")
        else:
            e["facts"].append(f"{_fmt(total)} sets over the window")
    return truth


def _plan_region_start(answer: str):
    """Where a multi-day plan begins, or None. Set counts inside a plan are
    prescriptions, not claims about what the user has done."""
    hits = ([m.start() for m in _DAY_WORD_RE.finditer(answer)]
            + [m.start() for m in _WEEKDAY_RE.finditer(answer)])
    return min(hits) if len(hits) >= 2 else None


def set_count_guard(answer: str, package: dict) -> tuple:
    """
    Deterministic check (pure, no LLM): a set count the answer CLAIMS for a
    muscle or category — "you currently perform 25 sets for Chest, 28 for
    Back" — must be one of that name's real figures (window, prior, all-time, or
    per-week, rounded or not). Prescriptions ("aim for 12 sets for Chest") and
    anything inside a plan are not claims and are skipped.

    Returns (violations, flags). It does NOT rewrite: code cannot know which of
    the real figures the sentence meant, so the caller re-prompts once, like
    plan_guard.
    """
    if not answer or not answer.strip():
        return [], []
    truth = _set_count_truth(package)
    if not truth:
        return [], []

    alt = "|".join(re.escape(t["label"])
                   for t in sorted(truth.values(), key=lambda t: -len(t["label"])))
    lead = re.compile(
        r"(?P<n>\d+(?:\.\d+)?)\s+(?:(?:direct|working|hard|total|primary|secondary)\s+)?"
        r"sets?\b(?:\s+(?:a|per|each)\s+week)?\s+(?:for|of|on|to)\s+(?:your\s+|the\s+)?"
        r"(?P<name>" + alt + r")\b", re.I)
    cont = re.compile(
        r"(?:,|\band\b)\s*(?:and\s+)?(?P<n>\d+(?:\.\d+)?)\s+(?:for|on)\s+"
        r"(?:your\s+|the\s+)?(?P<name>" + alt + r")\b", re.I)
    label_first = re.compile(
        r"(?P<name>" + alt + r")\s*[:\-–—]\s*(?P<n>\d+(?:\.\d+)?)\s+sets?\b", re.I)

    plan_from = _plan_region_start(answer)
    violations, flags, seen = [], [], set()
    for sm in _NUM_SENTENCE_RE.finditer(answer):
        if plan_from is not None and sm.start() >= plan_from:
            break
        sentence = sm.group(0)
        if not _CURRENT_CUE_RE.search(sentence) or _PLAN_CUE_RE.search(sentence):
            continue
        claims = [(m.group("n"), m.group("name")) for m in lead.finditer(sentence)]
        if claims:
            claims += [(m.group("n"), m.group("name")) for m in cont.finditer(sentence)]
        claims += [(m.group("n"), m.group("name")) for m in label_first.finditer(sentence)]
        for n, name in claims:
            t = truth[name.lower()]
            if any(_close(n, v) for v in t["values"]):
                continue
            key = (t["label"], n)
            if key in seen:
                continue
            seen.add(key)
            detail = f"{t['label']}: said {n}; your data has " + "; ".join(t["facts"])
            violations.append(detail)
            flags.append({"kind": "set_count", "name": t["label"], "claimed": n,
                          "original": sentence.strip(), "reason": detail})
    return violations, flags


# ── B4 · a sets-per-week figure in a plan comes with its target ───────────────

_TARGET_RANGE_RE = re.compile(
    rf"\b{SETS_PER_WEEK_TARGET[0]}\s*(?:-|–|—|to)\s*{SETS_PER_WEEK_TARGET[1]}\b")
_IMPROVE_CUE_RE = re.compile(
    r"\bbring(?:s|ing)?\s+(?:\w+\s+){0,3}up\b|\b(?:increase|raise|improve|boost)\w*\b|"
    r"\bmore\s+sets\b", re.I)
_PER_WEEK_FIGURE_RE = re.compile(
    r"(?P<n>\d+(?:\.\d+)?)\s+(?:(?:direct|primary|hard|working)\s+)?sets?\s+"
    r"(?:a|per|each)\s+week\b|(?P<n2>\d+(?:\.\d+)?)\s+sets?/(?:wk|week)\b", re.I)


def target_guard(answer: str, package: dict) -> tuple:
    """
    In a plan, or an answer about raising a muscle, the first time a muscle's
    real sets-per-week figure is quoted with no target anywhere in the answer,
    the target is stated right after it. Returns (answer, flags).
    """
    if not answer or not answer.strip() or _TARGET_RANGE_RE.search(answer):
        return answer, []
    if len(_plan_days(answer)) < 2 and not _IMPROVE_CUE_RE.search(answer):
        return answer, []
    rows = [(r["muscle"], r.get("primary_sets_per_week")) for r in _muscle_rows(package)]
    if not rows:
        return answer, []
    lo, hi = SETS_PER_WEEK_TARGET
    clause = f" (the usual target is {lo}–{hi} sets a week per muscle)"

    for sm in _NUM_SENTENCE_RE.finditer(answer):
        sentence = sm.group(0)
        for m in _PER_WEEK_FIGURE_RE.finditer(sentence):
            n = m.group("n") or m.group("n2")
            muscle = next((name for name, ppw in rows
                           if _close(n, ppw) and re.search(
                               r"\b" + re.escape(name.lower()) + r"\b", sentence.lower())),
                          None)
            if muscle is None:
                continue
            at = sm.start() + m.end()
            return (answer[:at] + clause + answer[at:],
                    [{"kind": "sets_per_week_target", "muscle": muscle,
                      "original": m.group(0), "corrected": m.group(0) + clause,
                      "reason": "a sets-per-week figure in a plan needs the target it "
                                "is measured against"}])
    return answer, []
