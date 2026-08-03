"""
src/ontology.py
Muscle ontology — the ONLY module that reads the `ontology/` store.

The store is reference DOMAIN KNOWLEDGE, not user data:

  ontology/muscles.csv          id, name, parent_id, size_class
  ontology/exercises.csv        id, canonical_name, equipment, movement_pattern
  ontology/exercise_muscle.csv  exercise_id, muscle_id, role, source
  ontology/aliases.csv          db_exercise_name, exercise_id

It deliberately lives OUTSIDE both `data/` and the `.fitnotes` file:
  • `data/` is gitignored wholesale (.gitignore), so a store there would not be
    versioned at all — and this must be reviewable in `git diff`.
  • the `.fitnotes` file is wiped and replaced on every backup upload (the reason
    src/wal.py exists), which would destroy an ontology stored inside it.

WHY CSV + PYTHON DICTS, not a second SQLite file with WITH RECURSIVE:
src/data_agent/process.py is documented pure — "no sqlite, no open(), no
os.environ". A plain dict crosses that boundary; a DB connection does not. With
~40 muscle nodes, subtree traversal is the `_descendants` loop below, and a
second SQLite file would add a build step and a second source of truth for no
gain. This is still a domain ontology with typed, transitive edges; the tree is
purely anatomical part-of and `size_class` is an ATTRIBUTE COLUMN, never a tree
tier (mixing "is part of" and "is the same size as" into one edge is how an
ontology rots — every later query would have to know which kind of parent it is
walking).

NEVER RAISES. A missing, unreadable, or structurally broken store logs loud and
yields an EMPTY ontology, which degrades the package's muscle section to {} via
_safe_compute. Reference data is not user data: a problem here must never
hard-stop an answer about the user's training.
"""

from __future__ import annotations

import csv
import logging
import os
import threading

logger = logging.getLogger(__name__)

_VALID_ROLES = frozenset({"primary", "secondary", "limiting"})

# The three roles differ by WHICH WAY FATIGUE FLOWS, which makes them
# distinguishable in practice rather than a matter of taste:
#
#   primary    the lift targets it.
#   secondary  the lift TRAINS it. Interference runs BOTH ways — delt work
#              before an incline press hurts the press, AND pressing hurts later
#              delt work. Counts as training volume.
#   limiting   the lift DEPENDS on it. Interference runs ONE way — grip work
#              before shrugs ruins the shrugs, but shrugs leave the grip fine.
#              The muscle holds the load without being trained by it, so it
#              NEVER counts as training volume.
#
# Crediting 266 sets of Smith Machine Shrugs toward grip training is false;
# deleting the edge loses a real scheduling fact. Hence three roles, not two.
#
# Precedence when one exercise reaches a muscle by several paths. A muscle
# genuinely trained by a lift is not demoted because another path merely leans
# on it.
ROLE_PRECEDENCE = ("primary", "secondary", "limiting")

# An exercise with this movement_pattern carries NO muscle edges on purpose —
# walking and cycling have no resistance-training muscle attribution worth
# counting. That is a different thing from a curation gap, and the summary
# reports the two separately (unattributed_exercises vs unmapped_exercises) so a
# missing alias can never hide behind "it's cardio".
CARDIO_PATTERN = "cardio"
_VALID_SIZE_CLASSES = frozenset({"large", "medium", "small"})

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ontology")

_lock = threading.Lock()
_cache: dict | None = None
_cache_dir: str | None = None


def _dir() -> str:
    return os.environ.get("ONTOLOGY_DIR", _DEFAULT_DIR)


def empty_ontology() -> dict:
    """The shape every consumer can rely on, with nothing in it."""
    return {
        "muscles": {}, "by_muscle_name": {}, "children": {},
        "descendants": {}, "ancestors": {}, "path": {},
        "exercises": {}, "edges": [], "edges_by_exercise": {},
        "aliases": {}, "reachable": frozenset(), "errors": [], "loaded": False,
    }


def _read_csv(directory: str, filename: str, required: tuple) -> list:
    """Read one store file into a list of dicts. Missing file / missing column is
    an error the caller turns into an empty ontology — never a partial load, which
    would silently under-count muscles."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{filename} is missing column(s): {', '.join(missing)}")
    # Drop comment rows (a '#' in the first column) so the curation drafts can
    # carry REVIEW markers without breaking the loader.
    first = required[0]
    return [r for r in rows if not str(r.get(first) or "").lstrip().startswith("#")]


def _int(value, label: str, errors: list):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        errors.append(f"{label}: {value!r} is not an integer")
        return None


def _descendants(root: int, children: dict, errors: list) -> frozenset:
    """A node plus every node beneath it. Iterative with a visited set, so a
    parent_id cycle is survived and reported rather than hanging the process."""
    seen: set = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, ()))
    return frozenset(seen)


def _build(directory: str) -> dict:
    o = empty_ontology()
    errors: list = o["errors"]

    # ── muscles ───────────────────────────────────────────────────────────────
    for row in _read_csv(directory, "muscles.csv",
                         ("id", "name", "parent_id", "size_class")):
        mid = _int(row["id"], "muscles.id", errors)
        name = (row["name"] or "").strip()
        if mid is None or not name:
            errors.append(f"muscles.csv: unusable row {row!r}")
            continue
        if mid in o["muscles"]:
            errors.append(f"muscles.csv: duplicate id {mid}")
            continue
        if name in o["by_muscle_name"]:
            # Names are the citation match-key, so they must be globally unique —
            # this is why the store says "Triceps Long Head", not "Long Head".
            errors.append(f"muscles.csv: duplicate name {name!r}")
            continue
        size = (row["size_class"] or "").strip().lower() or None
        if size is not None and size not in _VALID_SIZE_CLASSES:
            errors.append(f"muscles.csv: {name!r} has unknown size_class {size!r}")
            size = None
        o["muscles"][mid] = {
            "id": mid, "name": name,
            "parent_id": _int(row["parent_id"], "muscles.parent_id", errors),
            "size_class": size,
        }
        o["by_muscle_name"][name] = mid

    # Resolve parents, drop dangling links, then index children.
    children: dict = {mid: [] for mid in o["muscles"]}
    for mid, m in o["muscles"].items():
        pid = m["parent_id"]
        if pid is None:
            continue
        if pid not in o["muscles"]:
            errors.append(f"muscles.csv: {m['name']!r} has unknown parent_id {pid}")
            m["parent_id"] = None
            continue
        children[pid].append(mid)
    o["children"] = children

    for mid in o["muscles"]:
        o["descendants"][mid] = _descendants(mid, children, errors)

    # Ancestry: a node plus every node ABOVE it. This is what the rollup walks —
    # a set counted on Upper Traps must also reach Traps and Back. Bounded by the
    # muscle count so a cycle is reported, not looped on.
    for mid, m in o["muscles"].items():
        chain, node, guard = [], mid, 0
        while node is not None and guard <= len(o["muscles"]):
            chain.append(node)
            node = o["muscles"][node]["parent_id"]
            guard += 1
        if guard > len(o["muscles"]):
            errors.append(f"muscles.csv: parent_id cycle at {m['name']!r}")
            chain = [mid]
        o["ancestors"][mid] = frozenset(chain)
        o["path"][mid] = " > ".join(o["muscles"][n]["name"] for n in reversed(chain))

    # ── exercises ─────────────────────────────────────────────────────────────
    for row in _read_csv(directory, "exercises.csv",
                         ("id", "canonical_name", "equipment", "movement_pattern")):
        eid = _int(row["id"], "exercises.id", errors)
        name = (row["canonical_name"] or "").strip()
        if eid is None or not name:
            errors.append(f"exercises.csv: unusable row {row!r}")
            continue
        if eid in o["exercises"]:
            errors.append(f"exercises.csv: duplicate id {eid}")
            continue
        o["exercises"][eid] = {
            "id": eid, "canonical_name": name,
            "equipment": (row["equipment"] or "").strip() or None,
            "movement_pattern": (row["movement_pattern"] or "").strip() or None,
        }

    # ── edges ─────────────────────────────────────────────────────────────────
    seen_pairs: set = set()
    # An exercise cannot carry two roles for the SAME muscle — the duplicate-pair
    # check below already rejects that, so "limiting on a muscle the lift also
    # targets" is unrepresentable rather than merely discouraged.
    for row in _read_csv(directory, "exercise_muscle.csv",
                         ("exercise_id", "muscle_id", "role", "source")):
        eid = _int(row["exercise_id"], "exercise_muscle.exercise_id", errors)
        mid = _int(row["muscle_id"], "exercise_muscle.muscle_id", errors)
        role = (row["role"] or "").strip().lower()
        source = (row["source"] or "").strip()
        if eid is None or mid is None:
            continue
        if eid not in o["exercises"]:
            errors.append(f"exercise_muscle.csv: unknown exercise_id {eid}")
            continue
        if mid not in o["muscles"]:
            errors.append(f"exercise_muscle.csv: unknown muscle_id {mid}")
            continue
        if role not in _VALID_ROLES:
            errors.append(f"exercise_muscle.csv: bad role {role!r} on exercise {eid}")
            continue
        if not source:
            # Provenance is mandatory: sub-head attribution is genuinely contested
            # in the literature, so the agent's certainty must never exceed the
            # mapping's. An unsourced edge is not admissible.
            errors.append(f"exercise_muscle.csv: edge {eid}->{mid} has no source")
            continue
        if (eid, mid) in seen_pairs:
            errors.append(f"exercise_muscle.csv: duplicate edge {eid}->{mid}")
            continue
        seen_pairs.add((eid, mid))
        edge = {"exercise_id": eid, "muscle_id": mid, "role": role, "source": source}
        o["edges"].append(edge)
        o["edges_by_exercise"].setdefault(eid, []).append(edge)

    # ── aliases (the bridge from the user's DB names) ──────────────────────────
    for row in _read_csv(directory, "aliases.csv", ("db_exercise_name", "exercise_id")):
        db_name = (row["db_exercise_name"] or "").strip()
        eid = _int(row["exercise_id"], "aliases.exercise_id", errors)
        if not db_name or eid is None:
            errors.append(f"aliases.csv: unusable row {row!r}")
            continue
        if eid not in o["exercises"]:
            errors.append(f"aliases.csv: {db_name!r} points at unknown exercise_id {eid}")
            continue
        key = db_name.lower()
        if key in o["aliases"]:
            errors.append(f"aliases.csv: duplicate db_exercise_name {db_name!r}")
            continue
        o["aliases"][key] = eid

    # ── reachability ──────────────────────────────────────────────────────────
    # A muscle is REACHABLE when at least one exercise in the store maps to it or
    # to something beneath it. Only reachable muscles may be reported as
    # zero-coverage.
    #
    # This is load-bearing for honesty, not tidiness. Without it, a node the
    # ontology cannot distinguish produces a FALSE coverage claim: "Biceps Short
    # Head: 0 sets" while the user logs 416 sets of curls that obviously work it.
    # Rollup only carries counts UPWARD, so an unreachable leaf is permanently 0
    # no matter how the user trains. Zero sets on a reachable muscle is a fact
    # about training; zero sets on an unreachable one is an artifact of the
    # taxonomy, and the agent must never state the second as if it were the first.
    edged = {e["muscle_id"] for e in o["edges"]}
    o["reachable"] = frozenset(mid for mid, desc in o["descendants"].items()
                               if desc & edged)

    o["loaded"] = True
    return o


def load_ontology(force: bool = False) -> dict:
    """
    Load (and cache) the ontology store. Never raises.

    On any failure the return value is `empty_ontology()` — consumers see no
    muscles, no edges and no aliases, which makes every logged exercise land in
    the summary's `unmapped_exercises` list rather than producing wrong numbers.
    """
    global _cache, _cache_dir
    directory = _dir()
    with _lock:
        if not force and _cache is not None and _cache_dir == directory:
            return _cache
        try:
            ontology = _build(directory)
        except (OSError, ValueError, csv.Error) as exc:
            logger.error("[ontology] store at %s is unusable (%s) — "
                         "muscle analysis disabled for this run", directory, exc)
            ontology = empty_ontology()
        else:
            if ontology["errors"]:
                logger.warning("[ontology] %d problem(s) in %s:\n%s",
                               len(ontology["errors"]), directory,
                               "\n".join(f"  {e}" for e in ontology["errors"]))
            logger.info("[ontology] loaded %d muscles, %d exercises, %d edges, "
                        "%d aliases from %s",
                        len(ontology["muscles"]), len(ontology["exercises"]),
                        len(ontology["edges"]), len(ontology["aliases"]), directory)
        _cache, _cache_dir = ontology, directory
        return ontology


def clear_cache() -> None:
    """Drop the cached store (tests point ONTOLOGY_DIR at a fixture directory)."""
    global _cache, _cache_dir
    with _lock:
        _cache, _cache_dir = None, None


def subtree(ontology: dict, muscle_id: int) -> frozenset:
    """A muscle id plus every id beneath it. Empty for an unknown id."""
    return ontology.get("descendants", {}).get(muscle_id, frozenset())


def resolve_db_exercise(ontology: dict, db_exercise_name: str):
    """
    The bridge: an exact FitNotes exercise name -> ontology exercise id, or None.

    Deliberately an EXACT (case-insensitive) dict lookup, not fuzzy matching.
    src/shared/resolver.py resolves a USER'S TYPED QUERY against the user's
    exercise table — the opposite direction, and non-deterministic. Putting that
    in the hot path of a volume calculation would let a near-miss silently
    attribute 300 sets to the wrong muscle. Names that are not in aliases.csv
    return None and are reported as unmapped, never guessed.
    """
    if not db_exercise_name:
        return None
    return ontology.get("aliases", {}).get(db_exercise_name.strip().lower())


def is_unattributed(ontology: dict, exercise_id: int) -> bool:
    """True when this exercise deliberately carries no muscle edges (cardio)."""
    ex = ontology.get("exercises", {}).get(exercise_id)
    return bool(ex) and ex.get("movement_pattern") == CARDIO_PATTERN


# ── R7: the muscle set is CLOSED ──────────────────────────────────────────────
#
# Every human has the same muscles and the tree already covers them, so no user,
# no upload, and no model may ever add, remove or rename one. muscles.csv is the
# only file in the store that never grows.
#
# This is the invariant a growing UNIVERSAL graph loses most easily: a proposer
# that is merely confident would otherwise widen the taxonomy, and once a bogus
# node exists every later user inherits it. Three independent mechanisms hold it:
#   1. tests/test_ontology_frozen_muscles.py pins every row — an accidental edit
#      cannot pass the suite.
#   2. WRITABLE_FILES below; the promote path writes nothing else, and a test
#      asserts muscles.csv is byte-identical across a promote.
#   3. resolve_muscle_names() here, which REJECTS rather than creates.

MUSCLES_FILE = "muscles.csv"
# The only files any automated path may append to. muscles.csv is deliberately
# absent and must stay absent.
WRITABLE_FILES = ("exercises.csv", "aliases.csv", "exercise_muscle.csv")


def closed_muscle_names(ontology: dict) -> tuple:
    """Every muscle name in the store, sorted — the CLOSED vocabulary a proposal
    may draw from. Passed verbatim into the proposal prompt so the model is asked
    to choose from a list rather than to name anatomy freely."""
    return tuple(sorted(ontology.get("by_muscle_name", {})))


def resolve_muscle_names(ontology: dict, names) -> tuple:
    """
    Map proposed muscle names onto ids. Returns (resolved_ids, unknown_names).

    Rejects; never creates. A name outside the closed set comes back in
    `unknown` and the caller downgrades the whole proposal to 'unsure' — being
    confident is not a licence to widen the ontology.
    """
    by_name = ontology.get("by_muscle_name", {})
    lower = {n.lower(): mid for n, mid in by_name.items()}
    resolved, unknown = [], []
    for raw in names or []:
        key = str(raw or "").strip()
        mid = by_name.get(key) or lower.get(key.lower())
        if mid is None:
            unknown.append(key)
        elif mid not in resolved:
            resolved.append(mid)
    return tuple(resolved), tuple(unknown)


def top_level_region(ontology: dict, muscle_id: int):
    """The root of this muscle's branch ('Upper Traps' -> 'Back'), or None.

    R8: this is how an exercise in an UNKNOWN FitNotes category gets a display
    group — a new rowing variant rolls up to Back through its own muscles. No
    category is ever invented, because nothing here can create one."""
    m = ontology.get("muscles", {}).get(muscle_id)
    seen = set()
    while m is not None and m["id"] not in seen:
        seen.add(m["id"])
        if m["parent_id"] is None:
            return m["name"]
        m = ontology["muscles"].get(m["parent_id"])
    return None
