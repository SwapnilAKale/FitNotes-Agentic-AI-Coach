"""
src/ontology_propose.py
Stage 2 of new-exercise reconciliation: draft a mapping for a queued exercise so
the user has evidence in front of them when they approve or reject it.

Runs when the user REVIEWS the queue — never inside the /upload request. Upload
stays fast, offline-capable and free of model calls; a network failure or a
quota error can never break an upload or leave a half-written store.

    propose_one(row, ontology)  -> dict   (a pending row, enriched)
    propose_all(rows, ontology) -> list

THE PIPELINE — four stages, and the search is not optional:

  1. GRAPH PROPOSAL   Gemini, no tools. Sees the graph grouped by movement
                      pattern and maps the exercise from its pattern-mates.
  2. SEARCH           OUR CODE calls a search API with a fixed query. No model
                      decides whether it happens. No search, no proposal.
  3. READ             Gemini, no tools. Sees ONLY the search results and says
                      which muscles they name — not what it believes.
  4. COMPARE          Code. Stage 1's primary must agree with what the results
                      name; if not, the row is 'unsure' with both shown.

The search used to be a Gemini tool the model could choose to skip. In run 7 it
skipped it on three good answers and a check discarded all three. A search that
can be skipped is a check a user can neither see nor fix.

WHAT THIS MODULE CANNOT DO, by construction:
  • It cannot write to the store. It returns enriched rows; only
    ontology_reconcile.promote() writes, and only for approved='y' rows (R10).
  • It cannot widen the muscle set. Every proposed muscle is resolved against
    the CLOSED vocabulary, and one unknown name downgrades the whole proposal
    to 'unsure' (R7). Confidence is not a licence to invent anatomy.
  • It cannot invent a category. There is no category field in the output at
    all — a new rowing variant is described by its MUSCLES, which roll up to
    Back on their own (R8).

An 'unsure' decision is a success, not a failure: it means the evidence did not
settle the question, so the sets stay excluded until a human decides (R9).

A FAILED CALL IS NOT AN ANSWER. If the model or the search is never reached — no
API key, no network, quota — the row stays `pending` with a `NOT PROPOSED:` note.
'unsure' is reserved for a verdict about the movement.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from google import genai
from google.genai import types

from src import checkpoint as _ckpt
from src.ontology import (CARDIO_PATTERN, ROLE_PRECEDENCE, closed_muscle_names,
                          resolve_muscle_names, subtree)

logger = logging.getLogger(__name__)

# Neither Gemini call uses search any more, so the free tier's zero search
# grounding for Gemini 3 no longer applies — this is the app's own model, with
# 500 requests a day. ONTOLOGY_PROPOSE_MODEL overrides it without a code change.
PROPOSE_MODEL = "gemini-3.1-flash-lite"


def _model() -> str:
    return os.environ.get("ONTOLOGY_PROPOSE_MODEL") or PROPOSE_MODEL


# Per-minute 429 retry — the coordinator's own tuning (coordinator.py,
# PER_MINUTE_*), mirrored here rather than imported: importing the coordinator
# would pull the whole pipeline into a review script.
PER_MINUTE_WAIT_CAP = 70      # s — cap on one wait
PER_MINUTE_MAX_RETRIES = 2    # per call
PER_MINUTE_BUFFER = 2         # s — retry just after the window resets
PER_MINUTE_DEFAULT_WAIT = 55  # s — when the provider gives no retryDelay

# Transient 503 (model overload) retry — again the coordinator's tuning
# (coordinator.py, TRANSIENT_*). A 503 carries no retryDelay.
TRANSIENT_MAX_RETRIES = 2     # per call, counted apart from per-minute retries
TRANSIENT_BACKOFF = 5         # s

# More trained (primary + secondary) muscles than this needs a person. 69 of the
# graph's 74 exercises have four or fewer; the few above it are heavy compounds
# (deadlift, barbell rows) that were curated by hand. `limiting` muscles are
# held, not trained, and don't count.
MAX_TRAINED_MUSCLES = 4
_TRAINED_ROLES = ("primary", "secondary")

# The search, run by code. Tavily: 1,000 free searches a month, no card.
# Google's standalone search API is closed to new users and ends on 1 Jan 2027.
SEARCH_URL = "https://api.tavily.com/search"
SEARCH_KEY_VAR = "TAVILY_API_KEY"
SEARCH_MAX_RESULTS = 5
SEARCH_TIMEOUT = 20           # s
_READ_CONTENT_CAP = 1500      # chars of each result shown to the read stage

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


_SYSTEM = """\
You map a strength-training exercise name onto an existing muscle graph.

THE GRAPH IS YOUR EVIDENCE. The request lists every known exercise with its
checked muscles, grouped by movement pattern, and exercises in the same pattern
share their helper muscles. You do not search. Your answer is checked afterwards,
by code, against an independent web search; if the search names a different
primary muscle, a person decides.

HOW TO WORK

  1. Work out what the movement is.
  2. Choose its movement pattern from the patterns listed in the request.
  3. START FROM THE MUSCLES OF ITS PATTERN-MATES — the known exercises under
     that pattern. Take the primary from the closest match. Give only the
     secondaries the group shares, and a "limiting" muscle only if the group
     records one.
  4. Depart from the group only where the movement clearly works a different
     muscle. If you cannot tell what the movement is, answer "unsure".

You must decide ONE of three things:

  "alias"  — this is the SAME movement as an exercise already in the list of
             known exercises, just named differently. Give alias_of.
  "new"    — this is a genuinely different movement. Give its muscles.
  "unsure" — you cannot tell. THIS IS AN ACCEPTABLE ANSWER and is much better
             than a confident guess: an unsure item is shown to a human,
             whereas a wrong "alias" silently attributes hundreds of sets to
             the wrong muscle and nobody ever notices.

HARD RULES

1. The muscle list you are given is CLOSED and COMPLETE. Every human has the
   same muscles. You may ONLY use names from that list, spelled exactly as
   given. Never invent, rename, split or add a muscle. If the movement's target
   is not expressible in that list, answer "unsure".

2. Never propose a category, group or body part. Describe the movement by its
   MUSCLES only. A new type of row is described as lats/traps/rhomboids — the
   grouping is derived from that automatically.

3. Use the MOST SPECIFIC muscle the graph supports: a named head or region of a
   muscle, where the list has one, rather than its parent. NEVER use a whole
   body region — the request lists them. They are groupings of muscles, not
   muscles.

4. Every muscle gets exactly ONE role:
   "primary"   — a main mover; what the exercise is chosen to train.
   "secondary" — also trained, significantly. The test runs BOTH ways: tiring
                 this muscle first hurts the lift, AND the lift tires this
                 muscle for later work.
   "limiting"  — only HOLDS the load or the body position (grip on a shrug).
                 It runs ONE way: the lift depends on it but does not train
                 it. It never counts as training.
                 Do NOT record general bracing that nearly every free-weight
                 or standing lift needs — the core holding the torso still,
                 the legs standing. The graph records a held muscle only
                 where it genuinely caps the lift. If the pattern-mates have
                 no such muscle, give none.
   A "new" decision needs at least one "primary". Secondaries are rare: most
   exercises in the graph have one to three trained muscles in total.

5. Match the graph. A variation of a known movement gets its pattern-mates'
   muscles unless the movement clearly works a different primary.

6. "alias" only when a known exercise's listed muscles are ALSO right for this
   movement. A difference only in equipment or in body position does not make
   a different movement. Any other difference in the words usually does —
   "Machine Shrug" and "Machine Shrug Row" are not obviously the same thing;
   answer "unsure" if you cannot tell.

7. Do not write URLs. Sources are recorded from the independent search.

Return ONLY a JSON object, no markdown fences and no prose:

{"decision": "alias" | "new" | "unsure",
 "alias_of": "<exact canonical name>" or null,
 "movement_pattern": "<one pattern from the request>" (for "new") or null,
 "muscles": [{"muscle": "<exact name from the closed list>",
              "role": "primary" | "secondary" | "limiting"}],
 "evidence": "<one or two sentences: which pattern-mates you started from>"}
"""

_READ_SYSTEM = """\
You read web search results about ONE exercise and report which muscles THE
RESULTS say it works. Report only what the results state — not your own
knowledge, and not what you think is correct. If the results do not say, return
empty lists.

Use ONLY names from the closed muscle list, spelled exactly. Translate what the
results say onto the most specific name they support (for example
"gastrocnemius" is Calves). "primary" is what the results call the main or
target muscle; "secondary" is what they call assisting or secondary.

Return ONLY a JSON object, no markdown fences and no prose:

{"primary": ["<exact name>", ...], "secondary": ["<exact name>", ...]}
"""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_json(text: str):
    """Best-effort JSON extraction. A model that ignores the no-fences rule, or
    wraps the object in prose, must not cost the whole proposal."""
    cleaned = _FENCE_RE.sub("", text or "").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _snippet(text, n: int = 200) -> str:
    """A model reply, quoted and cut short — so a reply that was not JSON can
    still be read in the queue. Run 6 lost two rows to "model did not return a
    JSON object" with nothing to show what it said instead."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return "(empty reply)"
    return f'"{flat}"' if len(flat) <= n else f'"{flat[:n]}…"'


_MESSAGE_RE = re.compile(r"""['"]message['"]\s*:\s*['"]([^'"]+)['"]""")


def _short_error(exc) -> str:
    """'503 UNAVAILABLE: This model is currently experiencing high demand.' —
    not the provider's whole JSON body, which used to fill the evidence column.
    A plain message passes through untouched."""
    text = " ".join(str(exc).split())
    if "{" not in text:
        out = text
    else:
        head = text.split("{", 1)[0].strip().rstrip(".").strip()
        found = _MESSAGE_RE.search(text)
        out = f"{head}: {found.group(1)}" if found and head else (head or text)
    return out if len(out) <= 200 else out[:200] + "…"


def _unsure(row: dict, why: str) -> dict:
    """Downgrade to 'unsure', keeping the reason visible in the queue so the
    reviewer knows whether the model failed or the evidence genuinely did."""
    logger.info("[ontology] %r -> unsure (%s)", row.get("db_exercise_name"), why)
    return dict(row, decision="unsure", alias_of="", muscles="", movement_pattern="",
                sources="", evidence=(f"UNSURE: {why}" if why else "UNSURE"),
                approved="")


NOT_PROPOSED = "NOT PROPOSED:"


def _not_proposed(row: dict, why: str) -> dict:
    """The model or the search was never reached, so there is no verdict.

    The row stays `pending` — retried on the next run — with the reason visible.
    The first live run of --propose had no API key loaded and marked all ten
    queued exercises 'unsure', then reported success. That read as ten genuine
    judgements when not one question had been asked.
    """
    logger.warning("[ontology] %r -> not proposed (%s)", row.get("db_exercise_name"), why)
    return dict(row, decision="pending", alias_of="", muscles="", sources="",
                movement_pattern="", evidence=f"{NOT_PROPOSED} {why}", approved="")


# ── The graph ─────────────────────────────────────────────────────────────────

def regions(ontology: dict) -> list:
    """Whole body regions (the roots of the tree). Read from the store, never
    hardcoded. No edge in the curated graph attaches to one."""
    return sorted(m["name"] for m in ontology.get("muscles", {}).values()
                  if m.get("parent_id") is None)


def region_children(ontology: dict) -> dict:
    """{region: [the muscles directly inside it]}, read from parent_id. Run 5 wrote
    "Core" on an incline press; the model is shown what each region contains."""
    muscles = ontology.get("muscles", {})
    out = {m["name"]: [] for m in muscles.values() if m.get("parent_id") is None}
    for m in muscles.values():
        parent = muscles.get(m.get("parent_id"))
        if parent is not None and parent.get("parent_id") is None:
            out[parent["name"]].append(m["name"])
    return {r: sorted(kids) for r, kids in sorted(out.items())}


def patterns(ontology: dict) -> list:
    """Every movement pattern the graph uses, read from the store."""
    return sorted({(e.get("movement_pattern") or "").strip()
                   for e in ontology.get("exercises", {}).values()} - {""})


def _pattern_groups(ontology: dict) -> dict:
    """{pattern: [exercise ids, by name]}. Exercises with no pattern are grouped
    last under '(no pattern)'."""
    groups = {}
    for eid, ex in sorted(ontology.get("exercises", {}).items(),
                          key=lambda kv: kv[1]["canonical_name"].lower()):
        key = (ex.get("movement_pattern") or "").strip() or "(no pattern)"
        groups.setdefault(key, []).append(eid)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] == "(no pattern)", kv[0])))


def _group_trained_muscles(ontology: dict, pattern: str) -> set:
    """Names of the muscles any exercise in this pattern TRAINS."""
    muscles = ontology.get("muscles", {})
    by_ex = ontology.get("edges_by_exercise", {})
    return {muscles[e["muscle_id"]]["name"]
            for eid, ex in ontology.get("exercises", {}).items()
            if (ex.get("movement_pattern") or "").strip() == pattern
            for e in by_ex.get(eid, [])
            if e["role"] in _TRAINED_ROLES and e["muscle_id"] in muscles}


def _known_exercise_lines(ontology: dict) -> list:
    """The graph, grouped by movement pattern:

        [lateral raise]
        - Lateral Dumbbell Raise — Side Delts (primary)

    The first live runs saw only NAMES, so the model could neither match the
    graph nor tell whether an alias target trains the same muscles. Grouping
    by pattern puts the exercises it should start from side by side."""
    muscles = ontology.get("muscles", {})
    by_ex = ontology.get("edges_by_exercise", {})
    exercises = ontology.get("exercises", {})
    rank = {r: i for i, r in enumerate(ROLE_PRECEDENCE)}
    lines = []
    for pattern, ids in _pattern_groups(ontology).items():
        lines.append(f"[{pattern}]")
        for eid in ids:
            edges = sorted((e for e in by_ex.get(eid, []) if e["muscle_id"] in muscles),
                           key=lambda e: (rank.get(e["role"], 99),
                                          muscles[e["muscle_id"]]["name"]))
            parts = ", ".join(f"{muscles[e['muscle_id']]['name']} ({e['role']})"
                              for e in edges)
            lines.append(f"- {exercises[eid]['canonical_name']} — "
                         f"{parts or 'no muscles (cardio)'}")
    return lines


def _build_prompt(row: dict, ontology: dict) -> str:
    """Stage 1: the graph, and nothing from the web."""
    name = row.get("db_exercise_name")
    category = (row.get("fitnotes_category") or "").strip()
    # The category is a HINT and is labelled as one. It is frequently wrong (the
    # whole reason this ontology exists) and can be a user-invented string, so
    # the model is told not to trust it.
    hint = (f"\nThe user's app files it under the category {category!r}. Treat this "
            f"as a WEAK HINT ONLY — these categories are often wrong and may be "
            f"user-invented.\n" if category else "\n")
    return (
        f"Exercise name as logged: {name!r}\n"
        f"{hint}\n"
        f"Movement patterns in the graph (choose one):\n"
        # Cardio is never offered: a strength mapping that picked it is refused.
        f"{', '.join(p for p in patterns(ontology) if p != CARDIO_PATTERN)}\n\n"
        f"CLOSED muscle list (use these names EXACTLY, nothing else):\n"
        f"{', '.join(closed_muscle_names(ontology))}\n\n"
        f"Whole body regions — NEVER use one of these as a muscle; use a "
        f"muscle inside it:\n"
        + "".join(f"- {r} (use {', '.join(kids) or 'nothing'})\n"
                  for r, kids in region_children(ontology).items())
        + "\n"
        f"Exercises already in the graph, grouped by movement pattern, with their "
        f"checked muscles. START FROM THESE:\n"
        + "\n".join(_known_exercise_lines(ontology)) + "\n"
    )


# ── Stage 2: the search, run by code ──────────────────────────────────────────

class SearchFailed(Exception):
    """The search could not be run. The message is plain and never holds a key."""


def search_query(name: str) -> str:
    return f'"{name}" muscles worked'


def tavily_search(query: str, _urlopen=urllib.request.urlopen) -> list:
    """[{title, url, content}] for one query. Raises SearchFailed, never returns
    a silent empty list for a search that did not happen."""
    key = os.environ.get(SEARCH_KEY_VAR)
    if not key:
        raise SearchFailed(f"{SEARCH_KEY_VAR} is not set")
    body = json.dumps({"query": query, "search_depth": "basic",
                       "max_results": SEARCH_MAX_RESULTS,
                       "include_answer": False}).encode("utf-8")
    req = urllib.request.Request(
        SEARCH_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with _urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        why = {401: "the search API key was rejected",
               432: "this month's search credits are used up",
               433: "the search spending limit was reached"}.get(
                   exc.code, f"the search API returned HTTP {exc.code}")
        raise SearchFailed(why) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SearchFailed(f"the search API could not be reached ({type(exc).__name__})") from None
    return [{"title": str(r.get("title") or ""), "url": str(r.get("url") or ""),
             "content": str(r.get("content") or "")}
            for r in (data.get("results") or []) if isinstance(r, dict)
            and str(r.get("url") or "").startswith("http")]


# ── Stage 3: read the results ─────────────────────────────────────────────────

def _build_read_prompt(name: str, results: list, ontology: dict) -> str:
    """ONLY the results and the closed list — no graph, so the reader reports
    what the web says rather than agreeing with stage 1."""
    blocks = "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['content'][:_READ_CONTENT_CAP]}"
        for i, r in enumerate(results, 1))
    return (f"Exercise: {name!r}\n\n"
            f"CLOSED muscle list (use these names EXACTLY):\n"
            f"{', '.join(closed_muscle_names(ontology))}\n\n"
            f"SEARCH RESULTS:\n{blocks}\n")


def read_results(parsed, ontology: dict) -> dict:
    """{'primary': [ids], 'secondary': [ids]}. Names outside the closed set are
    dropped — the reader can no more widen the muscle set than the proposer.

    Whole regions are kept APART, in 'primary_regions'. "Shoulders" contains
    every delt, so beside a specific muscle it would agree with any shoulder
    answer at all (run 8's reader named Shoulders beside Front Delts). But when a
    region is ALL the results say — run 9's sites said only "chest" — it is
    still evidence, and compare() uses it then."""
    by_name = ontology.get("by_muscle_name", {})
    lower = {n.lower(): mid for n, mid in by_name.items()}
    whole = {by_name[r] for r in regions(ontology) if r in by_name}

    def ids(key, want_regions=False):
        out = []
        raw = parsed.get(key) if isinstance(parsed, dict) else None
        for name in raw if isinstance(raw, list) else []:
            mid = lower.get(str(name or "").strip().lower())
            if mid is not None and (mid in whole) == want_regions and mid not in out:
                out.append(mid)
        return out

    primary = ids("primary")
    return {"primary": primary,
            "secondary": [m for m in ids("secondary") if m not in primary],
            "primary_regions": ids("primary", want_regions=True)}


# ── Stage 4: compare ──────────────────────────────────────────────────────────

def _proposal_muscles(out: dict, ontology: dict) -> tuple:
    """(primary ids, every muscle id) of a graph answer — for an alias, the
    target's muscles from the graph."""
    if out["decision"] == "alias":
        eid = next((i for i, e in ontology.get("exercises", {}).items()
                    if e["canonical_name"] == out["alias_of"]), None)
        edges = [(e["muscle_id"], e["role"])
                 for e in ontology.get("edges_by_exercise", {}).get(eid, [])]
    else:
        by_name = ontology.get("by_muscle_name", {})
        edges = [(by_name[m], r)
                 for m, _, r in (p.rpartition(":") for p in out["muscles"].split("|"))
                 if m in by_name]
    return [m for m, r in edges if r == "primary"], [m for m, _r in edges]


def _same_or_nested(ontology: dict, a: int, b: int) -> bool:
    """The same muscle, or one inside the other (Triceps Lateral Head / Triceps)."""
    return a == b or b in subtree(ontology, a) or a in subtree(ontology, b)


def _site(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def compare(row: dict, parsed, out: dict, read: dict, results: list,
            ontology: dict) -> dict:
    """Hold the graph proposal to what the search results say. PURE.

    Two rules, both must hold (a head matches the muscle it belongs to):
      1. Every main muscle the graph answer claims is named by the search, as
         main or helper — the answer may not claim what the search never says.
      2. At least half of the search's main muscles are in the graph answer, in
         any role — the answer may not miss what the exercise mainly works.
    Run 8 aliased a seated cable row to a dumbbell row: one shared muscle (Lats)
    was enough to pass. Rule 2 catches it; the cable rows still pass.

    Secondaries from the search are shown, never added: the graph's pattern-mates
    decide those."""
    muscles = ontology.get("muscles", {})

    def names(ids):
        return ", ".join(muscles[i]["name"] for i in ids if i in muscles)

    answered = (isinstance(parsed, dict) and
                str(parsed.get("decision") or "").strip().lower() in ("alias", "new"))

    if out["decision"] in ("alias", "new"):
        claimed, answer = _proposal_muscles(out, ontology)
        # Region only: the results name no specific main muscle, just a region.
        # Every graph main muscle must then sit inside it. Known limit: "chest"
        # cannot tell upper from mid chest.
        regions_only = [] if read["primary"] else read.get("primary_regions", [])
        if regions_only:
            outside = [a for a in claimed
                       if not any(a in subtree(ontology, r) for r in regions_only)]
            if not claimed or outside:
                out = _unsure(row, f"the search only says {names(regions_only)}; "
                                   f"graph said {names(claimed) or 'nothing'}")
        else:
            said = read["primary"] + read["secondary"]
            unnamed = [a for a in claimed
                       if not any(_same_or_nested(ontology, a, b) for b in said)]
            missing = [b for b in read["primary"]
                       if not any(_same_or_nested(ontology, a, b) for a in answer)]
            if not read["primary"]:
                out = _unsure(row, "the search results did not name a primary muscle")
            elif unnamed:
                out = _unsure(row, f"graph calls {names(unnamed)} a main muscle; the "
                                   f"search never names it (search results say "
                                   f"{names(read['primary'])})")
            # MORE than half. Run 9's reader named four main muscles for a seated
            # cable row; a dumbbell row covered exactly two and passed.
            elif 2 * (len(read["primary"]) - len(missing)) <= len(read["primary"]):
                out = _unsure(row, f"the search's main muscles {names(missing)} are not "
                                   f"in the graph answer (graph said "
                                   f"{names(claimed) or 'nothing'})")

    notes = []
    if answered and out["decision"] == "unsure":
        # A rejected answer is still shown — in the evidence only. muscles and
        # alias_of stay empty, so an unsure row can never be promoted from it.
        notes.append(f"model proposed: {_attempt(parsed)}")
    elif out["decision"] == "alias":
        notes.append(f"graph: same muscles as {out['alias_of']}")
    elif out["decision"] == "new" and out.get("movement_pattern"):
        pattern = out["movement_pattern"]
        exercises = ontology.get("exercises", {})
        mates = [exercises[i]["canonical_name"]
                 for i in _pattern_groups(ontology).get(pattern, [])][:5]
        if mates:
            notes.append(f"graph: started from {pattern} ({', '.join(mates)})")
        # Where the answer left its group. A note, never a rejection: an uncommon
        # exercise can legitimately differ — but the reviewer sees it at a glance.
        group = _group_trained_muscles(ontology, pattern)
        trained = [m for m, _, r in (p.rpartition(":") for p in out["muscles"].split("|"))
                   if r in _TRAINED_ROLES]
        outside = [m for m in trained if m not in group]
        if group and outside:
            notes.append(f"not in other {pattern} exercises: {', '.join(outside)}")

    if read["primary"] or not read.get("primary_regions"):
        search = [f"search says primary: {names(read['primary']) or 'nothing'}"]
    else:
        search = [f"search says primary: (region only) {names(read['primary_regions'])}"]
    if read["secondary"]:
        search.append(f"also names: {names(read['secondary'])}")
    if read.get("unreadable"):
        search.append(f"reader reply was not JSON: {read['unreadable']}")
    sites = sorted({_site(r["url"]) for r in results} - {""})
    if sites:
        search.append("sites: " + ", ".join(sites))
    notes.append(" | ".join(search))

    if out["decision"] in ("alias", "new"):
        out = dict(out, sources=" ".join(r["url"] for r in results))
    out["evidence"] = f"{out['evidence']} [{' | '.join(notes)}]".strip()
    return out


def _attempt(parsed: dict) -> str:
    """The model's own answer, as written — unknown names and its pattern included."""
    if str(parsed.get("decision") or "").strip().lower() == "alias":
        return f"ALIAS {str(parsed.get('alias_of') or '').strip() or '?'}"
    pattern = str(parsed.get("movement_pattern") or "").strip()
    pairs = [f"{str(e.get('muscle') or '').strip()}:"
             f"{str(e.get('role') or '').strip().lower()}"
             for e in (parsed.get("muscles") or []) if isinstance(e, dict)]
    body = "|".join(pairs) or "no muscles"
    return f"({pattern}) {body}" if pattern else body


# ── Calling Gemini ────────────────────────────────────────────────────────────

def _generate(client, contents: str, system: str, sleep) -> str:
    """One Gemini call, no tools, with the per-minute 429 and 503 retries. Raises
    whatever is left once the retries are spent."""
    attempt = transient_attempt = 0
    while True:
        try:
            response = (client or _get_client()).models.generate_content(
                model=_model(),
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system,
                                                   temperature=0.0),
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:                 # key / network / quota / SDK
            # ONLY a per-minute limit is worth waiting for — it refills within a
            # minute. A daily limit will not clear by waiting, so it fails at once.
            if (_ckpt.is_rate_limit(exc) and _ckpt.is_per_minute_quota(exc)
                    and attempt < PER_MINUTE_MAX_RETRIES):
                wait = min(_ckpt.retry_delay_seconds(exc) or PER_MINUTE_DEFAULT_WAIT,
                           PER_MINUTE_WAIT_CAP) + PER_MINUTE_BUFFER
                logger.warning("[ontology] per-minute 429 — waiting %ds (retry %d/%d)",
                               wait, attempt + 1, PER_MINUTE_MAX_RETRIES)
                sleep(wait)
                attempt += 1
                continue
            # A 503 is Google's model being overloaded, not a limit on us.
            if (_ckpt.is_transient_server_error(exc)
                    and transient_attempt < TRANSIENT_MAX_RETRIES):
                wait = TRANSIENT_BACKOFF + PER_MINUTE_BUFFER
                logger.warning("[ontology] transient 503 — waiting %ds (retry %d/%d)",
                               wait, transient_attempt + 1, TRANSIENT_MAX_RETRIES)
                sleep(wait)
                transient_attempt += 1
                continue
            raise


# ── The pipeline ──────────────────────────────────────────────────────────────

def propose_one(row: dict, ontology: dict, client=None, sleep=time.sleep,
                search_fn=None) -> dict:
    """
    Enrich one pending row: graph proposal, code-run search, read, compare.

    Never raises and never writes. A call that fails leaves the row pending
    (NOT PROPOSED); an answer the evidence does not support comes back 'unsure'.
    """
    name = (row.get("db_exercise_name") or "").strip()
    if not name:
        return _unsure(row, "no exercise name")

    # 1. Graph proposal.
    try:
        text = _generate(client, _build_prompt(row, ontology), _SYSTEM, sleep)
    except Exception as exc:
        return _not_proposed(row, f"model call failed: {_short_error(exc)}")
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        return _unsure(row, f"model did not return a JSON object — it replied: "
                            f"{_snippet(text)}")
    decision = (str(parsed.get("decision") or "").strip().lower()
                if isinstance(parsed, dict) else "")
    if decision not in ("alias", "new"):
        # No mapping to check — no search is spent on it.
        return validate_proposal(row, parsed, ontology)

    # 2. The search. Run by code, every time; it cannot be skipped.
    try:
        results = (search_fn or tavily_search)(search_query(name))
    except Exception as exc:
        return _not_proposed(row, f"web search failed: {_short_error(exc)}")

    # 3. Read what the results say.
    read = {"primary": [], "secondary": [], "primary_regions": []}
    if results:
        try:
            read_text = _generate(client, _build_read_prompt(name, results, ontology),
                                  _READ_SYSTEM, sleep)
        except Exception as exc:
            return _not_proposed(row, f"model call failed: {_short_error(exc)}")
        read_parsed = _parse_json(read_text)
        read = read_results(read_parsed, ontology)
        if not isinstance(read_parsed, dict):
            read["unreadable"] = _snippet(read_text)

    # 4. Validate the proposal, then hold it to the results.
    out = validate_proposal(row, dict(parsed, sources=[r["url"] for r in results]),
                            ontology)
    return compare(row, parsed, out, read, results, ontology)


def validate_proposal(row: dict, parsed, ontology: dict) -> dict:
    """
    Turn a raw model object into a queue row, rejecting anything unsupported.
    PURE — no network, no writes — so every rule below is directly testable.
    """
    if not isinstance(parsed, dict):
        return _unsure(row, "model did not return a JSON object")

    decision = str(parsed.get("decision") or "").strip().lower()
    evidence = " ".join(str(parsed.get("evidence") or "").split())[:400]
    sources = [str(s).strip() for s in (parsed.get("sources") or [])
               if str(s).strip().startswith("http")]

    if decision not in ("alias", "new"):
        return _unsure(row, evidence or "model was not confident")
    if not sources:
        # An unsourced claim is indistinguishable from a guess.
        return _unsure(row, "no source URL — the web search returned no results")

    if decision == "alias":
        target = str(parsed.get("alias_of") or "").strip()
        canonical = {e["canonical_name"].lower(): e["canonical_name"]
                     for e in ontology.get("exercises", {}).values()}
        if target.lower() not in canonical:
            return _unsure(row, f"alias_of {target!r} is not a known exercise")
        return dict(row, decision="alias", alias_of=canonical[target.lower()],
                    muscles="", movement_pattern="", evidence=evidence,
                    sources=" ".join(sources), approved="")

    # decision == "new"
    entries = [e for e in (parsed.get("muscles") or []) if isinstance(e, dict)]
    names = [str(e.get("muscle") or "").strip() for e in entries]
    _ids, unknown = resolve_muscle_names(ontology, names)
    if unknown:
        # R7: the model does not get to widen the ontology by being confident.
        return _unsure(row, f"proposed muscle(s) outside the closed set: "
                            f"{', '.join(unknown)}")

    by_name = ontology.get("by_muscle_name", {})
    lower = {n.lower(): n for n in by_name}
    pairs, seen = [], set()
    for e in entries:
        exact = lower.get(str(e.get("muscle") or "").strip().lower())
        role = str(e.get("role") or "").strip().lower()
        if exact is None or role not in ROLE_PRECEDENCE or exact in seen:
            continue
        seen.add(exact)
        pairs.append((exact, role))

    # A region is a grouping, not a muscle. No curated edge uses one, and an edge
    # on "Chest" would sit beside Mid Chest, less specific than everything else.
    whole = sorted({m for m, _r in pairs} & set(regions(ontology)))
    if whole:
        return _unsure(row, f"{', '.join(whole)} is a whole body region, not a "
                            f"muscle — pick the specific muscle")
    if not any(r == "primary" for _m, r in pairs):
        return _unsure(row, "no primary muscle was given")
    trained = sum(1 for _m, r in pairs if r in _TRAINED_ROLES)
    if trained > MAX_TRAINED_MUSCLES:
        return _unsure(row, f"{trained} trained muscles — more than almost every "
                            f"exercise in the graph; needs a person to decide")
    pairs = [f"{m}:{r}" for m, r in pairs]

    # The pattern keeps a promoted exercise in its movement group — the group the
    # next uncommon exercise is mapped from. Cardio is refused: a cardio
    # exercise's muscle edges are ignored everywhere, so a strength mapping that
    # carried it would silently count for nothing.
    raw_pattern = str(parsed.get("movement_pattern") or "").strip()
    if raw_pattern.lower() == CARDIO_PATTERN:
        return _unsure(row, "a strength exercise with muscles cannot be 'cardio' — "
                            "cardio exercises carry no muscle volume")
    known = {p.lower(): p for p in patterns(ontology)}
    pattern = known.get(raw_pattern.lower(), "")
    if not pattern:
        shown = f"{raw_pattern!r} is" if raw_pattern else "no movement pattern was"
        evidence = f"{evidence} [{shown} not in the graph — a person sets it]".strip()

    return dict(row, decision="new", alias_of="", muscles="|".join(pairs),
                movement_pattern=pattern, evidence=evidence,
                sources=" ".join(sources), approved="")


def propose_all(rows: list, ontology: dict, client=None, only_pending=True,
                sleep=time.sleep, search_fn=None) -> list:
    """Propose for each queued row. A row the user already decided on
    (approved set) is left completely alone."""
    out = []
    for row in rows or []:
        if (row.get("approved") or "").strip():
            out.append(row)
            continue
        decision = (row.get("decision") or "").strip().lower()
        if only_pending and decision not in ("", "pending", "unsure"):
            out.append(row)
            continue
        out.append(propose_one(row, ontology, client=client, sleep=sleep,
                               search_fn=search_fn))
    return out
