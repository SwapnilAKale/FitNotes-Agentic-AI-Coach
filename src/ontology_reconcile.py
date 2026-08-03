"""
src/ontology_reconcile.py
Stage 1 of new-exercise reconciliation: detect exercises the graph has never
seen, resolve the ones that need no judgment, queue the rest for approval.

Runs at UPLOAD time. Deterministic, offline, no LLM, no network — an upload must
never depend on a model or a working internet connection. The web-search
proposal (src/ontology_propose.py) runs later, when the user reviews the queue.

    detect_new(...)  -> ReconcileResult(auto_aliased, pending, already_known)
    load_pending()   -> [dict]           read ontology/pending_review.csv
    promote(...)     -> PromoteResult    write APPROVED rows into the store

WHAT MAY AUTO-RESOLVE — categorical evidence only:
  • the name normalises (case / spacing / punctuation) to one the store knows
  • the name is an exact word-permutation of exactly ONE canonical name,
    plural-tolerant — the same non-thresholded signal src/shared/resolver.py
    Tier 3 already uses
Two or more candidates never auto-picks, and fuzzy similarity is deliberately
NOT a tier: a genuine typo ("Dumbell Skullcrusher") goes to review, because
"close enough" is exactly how hundreds of sets get attributed to the wrong
muscle with nobody noticing.

R7 — muscles.csv IS NEVER WRITTEN HERE. Only ontology.WRITABLE_FILES are opened
for append, and promote() asserts it before touching anything.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from typing import NamedTuple, Optional

from src.ontology import (WRITABLE_FILES, MUSCLES_FILE, load_ontology,
                          resolve_muscle_names)
from src.shared.resolver import _expand_queries, _word_multiset

logger = logging.getLogger(__name__)

PENDING_FILE = "pending_review.csv"
PENDING_COLUMNS = (
    "db_exercise_name",   # exact FitNotes name — the alias key
    "logged_sets",        # how much this actually moves the numbers
    "fitnotes_category",  # hint only; may be blank or a custom category
    "decision",           # pending | alias | new | unsure   (filled by propose)
    "alias_of",           # canonical name, when decision == alias
    "muscles",            # "Lats:primary|Biceps:secondary"
    "evidence",           # one or two sentences from the proposer
    "sources",            # space-separated URLs
    "approved",           # y | n | (blank = undecided)  <- THE GATE
)

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ontology")


def _dir() -> str:
    return os.environ.get("ONTOLOGY_DIR", _DEFAULT_DIR)


def _pending_path() -> str:
    return os.path.join(_dir(), PENDING_FILE)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Case / spacing / punctuation-insensitive key. 'Cable Lat Pull With Ez-Bar'
    and 'cable lat pull with ezbar' collapse to the same string. Does NOT correct
    spelling — a misspelling must reach review, not be guessed at."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# ── The deterministic tier ────────────────────────────────────────────────────

class Match(NamedTuple):
    exercise_id: Optional[int]
    reason: str            # normalized-exact | word-permutation | ambiguous | none


def deterministic_match(name: str, ontology: dict) -> Match:
    """
    Resolve `name` to an existing ontology exercise using categorical evidence
    only, or report why it could not.

    Matches against BOTH canonical names and already-known aliases: a second
    user typing an alias the graph already learned should resolve immediately.
    """
    if not (name or "").strip():
        return Match(None, "none")

    # name -> exercise_id over canonical names + existing aliases.
    candidates: dict = {}
    for eid, ex in ontology.get("exercises", {}).items():
        candidates[ex["canonical_name"]] = eid
    for alias_lower, eid in ontology.get("aliases", {}).items():
        candidates.setdefault(alias_lower, eid)

    # ── Tier 1: normalised-exact ──────────────────────────────────────────────
    key = normalize(name)
    if key:
        hits = {eid for cand, eid in candidates.items() if normalize(cand) == key}
        if len(hits) == 1:
            return Match(hits.pop(), "normalized-exact")
        if len(hits) > 1:
            return Match(None, "ambiguous")

    # ── Tier 2: sole exact word-permutation (plural-tolerant) ─────────────────
    # "Skull Crusher Dumbbell" IS the user naming "Dumbbell Skull Crusher".
    # Word ORDER is the only thing ignored; a different word SET is a different
    # exercise, which is why "Machine Shrug Row" does not match "Machine Shrug".
    variants = {_word_multiset(v) for v in _expand_queries(name)}
    hits = {eid for cand, eid in candidates.items()
            if _word_multiset(cand) in variants}
    if len(hits) == 1:
        return Match(hits.pop(), "word-permutation")
    if len(hits) > 1:
        return Match(None, "ambiguous")

    return Match(None, "none")


# ── Detection ─────────────────────────────────────────────────────────────────

class ReconcileResult(NamedTuple):
    auto_aliased:  list   # [{db_exercise_name, canonical_name, reason}]
    pending:       list   # [{**PENDING_COLUMNS}]
    already_known: list   # names the store already had an exact alias for

    def summary(self) -> str:
        return (f"{len(self.already_known)} known, "
                f"{len(self.auto_aliased)} auto-aliased, "
                f"{len(self.pending)} pending review")


def detect_new(exercise_names,
               ontology:   Optional[dict] = None,
               set_counts: Optional[dict] = None,
               categories: Optional[dict] = None) -> ReconcileResult:
    """
    Classify each candidate name. PURE — writes nothing.

    set_counts / categories are optional context ({name: int} / {name: str});
    they only order and annotate the review queue, never the decision.
    """
    ont = ontology if ontology is not None else load_ontology()
    aliases = ont.get("aliases", {})
    auto, pending, known = [], [], []

    for name in exercise_names or []:
        name = (name or "").strip()
        if not name:
            continue
        if name.lower() in aliases:
            known.append(name)
            continue

        match = deterministic_match(name, ont)
        if match.exercise_id is not None:
            auto.append({
                "db_exercise_name": name,
                "exercise_id":      match.exercise_id,
                "canonical_name":   ont["exercises"][match.exercise_id]["canonical_name"],
                "reason":           match.reason,
            })
        else:
            pending.append({
                "db_exercise_name":  name,
                "logged_sets":       str((set_counts or {}).get(name, 0)),
                "fitnotes_category": (categories or {}).get(name, ""),
                "decision":          "pending",
                "alias_of":          "",
                "muscles":           "",
                "evidence":          "",
                "sources":           "",
                "approved":          "",
            })

    # Review what moves the numbers first.
    pending.sort(key=lambda r: (-int(r["logged_sets"] or 0), r["db_exercise_name"]))
    return ReconcileResult(auto, pending, known)


# ── Store writes ──────────────────────────────────────────────────────────────

def _assert_writable(filename: str) -> None:
    """R7 guard at the one place bytes are appended. muscles.csv is CLOSED —
    reaching it from an automated path is a bug, not a recoverable condition."""
    if filename == MUSCLES_FILE or filename not in WRITABLE_FILES:
        raise ValueError(
            f"refusing to write {filename!r}: the muscle set is closed and only "
            f"{', '.join(WRITABLE_FILES)} may be appended to")


def _append_rows(filename: str, columns: tuple, rows: list) -> None:
    _assert_writable(filename)
    if not rows:
        return
    path = os.path.join(_dir(), filename)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def _atomic_write(path: str, columns: tuple, rows: list) -> None:
    """tmp + os.replace with the Windows/OneDrive PermissionError retry the WAL
    and settings.py already use — the repo lives under OneDrive and the
    destination can be transiently locked."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def write_auto_aliases(auto_aliased: list) -> int:
    """Append the categorically-matched aliases. Only aliases.csv is touched —
    these name an exercise the store ALREADY has, so no exercise row and no edge
    is created. Returns how many were written."""
    rows = [{"db_exercise_name": a["db_exercise_name"],
             "exercise_id": a["exercise_id"]} for a in auto_aliased]
    _append_rows("aliases.csv", ("db_exercise_name", "exercise_id"), rows)
    for a in auto_aliased:
        logger.info("[ontology] auto-alias %r -> %r (%s)",
                    a["db_exercise_name"], a["canonical_name"], a["reason"])
    return len(rows)


# ── The pending queue ─────────────────────────────────────────────────────────

def load_pending() -> list:
    path = _pending_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return [r for r in csv.DictReader(fh)
                    if (r.get("db_exercise_name") or "").strip()]
    except (OSError, csv.Error) as exc:
        logger.error("[ontology] pending queue unreadable (%s) — treating as empty", exc)
        return []


def save_pending(rows: list) -> None:
    _atomic_write(_pending_path(), PENDING_COLUMNS, rows)


def merge_pending(new_rows: list) -> list:
    """Add newly-detected rows, keeping any existing row untouched — a proposal
    the user already reviewed must not be reset by a later upload."""
    existing = load_pending()
    seen = {(r.get("db_exercise_name") or "").lower() for r in existing}
    merged = existing + [r for r in new_rows
                         if (r["db_exercise_name"] or "").lower() not in seen]
    save_pending(merged)
    return merged


# ── Promote ───────────────────────────────────────────────────────────────────

class PromoteResult(NamedTuple):
    promoted: list
    skipped:  list        # [(name, why)]

    def summary(self) -> str:
        return f"{len(self.promoted)} promoted, {len(self.skipped)} skipped"


def _parse_muscles(spec: str) -> list:
    """'Lats:primary|Biceps:secondary' -> [(name, role)]."""
    out = []
    for chunk in (spec or "").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, role = chunk.rpartition(":")
        out.append((name.strip(), role.strip().lower()))
    return out


def promote(ontology: Optional[dict] = None) -> PromoteResult:
    """
    Write every pending row marked approved='y' into the store, then drop it from
    the queue. Rows not approved are left exactly as they are.

    R10 — this is the ONLY path from a proposal into the graph, and it runs only
    when the user asks for it. R7 — muscles.csv is never opened; an unknown
    muscle name skips the row rather than creating anything.
    """
    ont = ontology if ontology is not None else load_ontology(force=True)
    rows = load_pending()
    promoted, skipped, keep = [], [], []

    next_id = max(ont.get("exercises", {}) or {0: None}, default=0) + 1
    new_ex_rows, new_alias_rows, new_edge_rows = [], [], []

    for row in rows:
        name = (row.get("db_exercise_name") or "").strip()
        if (row.get("approved") or "").strip().lower() != "y":
            keep.append(row)
            continue

        decision = (row.get("decision") or "").strip().lower()
        source = (row.get("sources") or "").strip() or "user-approved"
        evidence = (row.get("evidence") or "").strip()
        provenance = f"{evidence} [{source}]" if evidence else source

        if decision == "alias":
            target = (row.get("alias_of") or "").strip()
            eid = next((i for i, e in ont["exercises"].items()
                        if e["canonical_name"].lower() == target.lower()), None)
            if eid is None:
                skipped.append((name, f"alias_of {target!r} is not a known exercise"))
                keep.append(row)
                continue
            new_alias_rows.append({"db_exercise_name": name, "exercise_id": eid})
            promoted.append({"name": name, "as": "alias", "target": target})
            continue

        if decision != "new":
            skipped.append((name, f"decision is {decision!r}, not 'alias' or 'new'"))
            keep.append(row)
            continue

        pairs = _parse_muscles(row.get("muscles", ""))
        ids, unknown = resolve_muscle_names(ont, [n for n, _r in pairs])
        if unknown:
            # R7: an unrecognised muscle is NEVER created to make the row fit.
            skipped.append((name, f"unknown muscle(s): {', '.join(unknown)}"))
            keep.append(row)
            continue
        roles = {n.lower(): r for n, r in pairs}
        if not any(roles.get(ont["muscles"][i]["name"].lower()) == "primary"
                   for i in ids):
            skipped.append((name, "no primary muscle"))
            keep.append(row)
            continue

        new_ex_rows.append({"id": next_id, "canonical_name": name,
                            "equipment": "", "movement_pattern": ""})
        new_alias_rows.append({"db_exercise_name": name, "exercise_id": next_id})
        for mid in ids:
            role = roles.get(ont["muscles"][mid]["name"].lower(), "secondary")
            new_edge_rows.append({"exercise_id": next_id, "muscle_id": mid,
                                  "role": role, "source": provenance})
        promoted.append({"name": name, "as": "new", "exercise_id": next_id,
                         "muscles": [ont["muscles"][i]["name"] for i in ids]})
        next_id += 1

    _append_rows("exercises.csv",
                 ("id", "canonical_name", "equipment", "movement_pattern"),
                 new_ex_rows)
    _append_rows("aliases.csv", ("db_exercise_name", "exercise_id"), new_alias_rows)
    _append_rows("exercise_muscle.csv",
                 ("exercise_id", "muscle_id", "role", "source"), new_edge_rows)

    save_pending(keep)
    if promoted:
        load_ontology(force=True)     # the store changed under the cache
    return PromoteResult(promoted, skipped)
