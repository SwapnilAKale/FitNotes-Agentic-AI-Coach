"""
src/ontology_propose.py
Stage 2 of new-exercise reconciliation: draft a mapping for a queued exercise,
with a real web-search tool, so the user has evidence in front of them when they
approve or reject it.

Runs when the user REVIEWS the queue — never inside the /upload request. Upload
stays fast, offline-capable and free of model calls; a network failure or a
quota error can never break an upload or leave a half-written store.

    propose_one(row, ontology)  -> dict   (a pending row, enriched)
    propose_all(rows, ontology) -> list

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
"""

from __future__ import annotations

import json
import logging
import os
import re

from google import genai
from google.genai import types

from src.ontology import closed_muscle_names, resolve_muscle_names

logger = logging.getLogger(__name__)

# Same family as the rest of the project. Search grounding needs a live call, so
# this is the one place in the ontology path that touches the network.
PROPOSE_MODEL = "gemini-3.1-flash-lite"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


_SYSTEM = """\
You map a strength-training exercise name onto an existing muscle ontology.
Use the web search tool to find out what the movement actually is before
answering — exercise names vary between gyms, apps and countries, and two names
can describe the same movement or two different ones.

You must decide ONE of three things:

  "alias"  — this is the SAME movement as an exercise already in the list of
             known exercises, just named differently. Give alias_of.
  "new"    — this is a genuinely different movement. Give its muscles.
  "unsure" — the evidence does not settle it. THIS IS AN ACCEPTABLE ANSWER and
             is much better than a confident guess: an unsure item is shown to a
             human, whereas a wrong "alias" silently attributes hundreds of sets
             to the wrong muscle and nobody ever notices.

HARD RULES

1. The muscle list you are given is CLOSED and COMPLETE. Every human has the
   same muscles. You may ONLY use names from that list, spelled exactly as
   given. Never invent, rename, split or add a muscle. If the movement's target
   is not expressible in that list, answer "unsure".

2. Never propose a category, group or body part. Describe the movement by its
   MUSCLES only. A new type of row is described as lats/traps/rhomboids — the
   grouping is derived from that automatically.

3. A "new" decision needs at least one muscle with role "primary". Mark a muscle
   "secondary" only when it is a SIGNIFICANT part of the movement, not everything
   that is active. Two or three muscles is normal; eight is wrong.

4. A different word SET usually means a different movement. "Machine Shrug" and
   "Machine Shrug Row" are not obviously the same thing — search before deciding,
   and answer "unsure" if the sources disagree or are thin.

5. Cite what you actually found. sources must contain at least one real URL from
   your search. No sources means "unsure".

Return ONLY a JSON object, no markdown fences and no prose:

{"decision": "alias" | "new" | "unsure",
 "alias_of": "<exact canonical name>" or null,
 "muscles": [{"muscle": "<exact name from the closed list>",
              "role": "primary" | "secondary"}],
 "evidence": "<one or two sentences on what the sources said>",
 "sources": ["<url>", ...]}
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


def _unsure(row: dict, why: str) -> dict:
    """Downgrade to 'unsure', keeping the reason visible in the queue so the
    reviewer knows whether the model failed or the evidence genuinely did."""
    logger.info("[ontology] %r -> unsure (%s)", row.get("db_exercise_name"), why)
    return dict(row, decision="unsure", alias_of="", muscles="",
                evidence=(f"UNSURE: {why}" if why else "UNSURE"), approved="")


def _build_prompt(row: dict, ontology: dict) -> str:
    known = sorted(e["canonical_name"] for e in ontology.get("exercises", {}).values())
    category = (row.get("fitnotes_category") or "").strip()
    # The category is a HINT and is labelled as one. It is frequently wrong (the
    # whole reason this ontology exists) and can be a user-invented string, so
    # the model is told not to trust it.
    hint = (f"\nThe user's app files it under the category {category!r}. Treat this "
            f"as a WEAK HINT ONLY — these categories are often wrong and may be "
            f"user-invented.\n" if category else "\n")
    return (
        f"Exercise name as logged: {row.get('db_exercise_name')!r}\n"
        f"{hint}\n"
        f"CLOSED muscle list (use these names EXACTLY, nothing else):\n"
        f"{', '.join(closed_muscle_names(ontology))}\n\n"
        f"Exercises already known to the ontology (for an 'alias' decision):\n"
        f"{', '.join(known)}\n"
    )


def propose_one(row: dict, ontology: dict, client=None) -> dict:
    """
    Enrich one pending row with a searched, validated proposal.

    Never raises and never writes. Any failure — network, quota, malformed JSON,
    hallucinated muscle — comes back as 'unsure', which keeps the exercise's
    sets excluded and the row in the queue for a human.
    """
    name = (row.get("db_exercise_name") or "").strip()
    if not name:
        return _unsure(row, "no exercise name")

    try:
        response = (client or _get_client()).models.generate_content(
            model=PROPOSE_MODEL,
            contents=_build_prompt(row, ontology),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )
        text = getattr(response, "text", "") or ""
    except Exception as exc:                     # network / quota / SDK
        return _unsure(row, f"search or model call failed: {exc}")

    return validate_proposal(row, _parse_json(text), ontology)


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
        # Rule 5: an uncited claim is indistinguishable from a guess.
        return _unsure(row, "no source URL was cited")

    if decision == "alias":
        target = str(parsed.get("alias_of") or "").strip()
        canonical = {e["canonical_name"].lower(): e["canonical_name"]
                     for e in ontology.get("exercises", {}).values()}
        if target.lower() not in canonical:
            return _unsure(row, f"alias_of {target!r} is not a known exercise")
        return dict(row, decision="alias", alias_of=canonical[target.lower()],
                    muscles="", evidence=evidence,
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
        if exact is None or role not in ("primary", "secondary") or exact in seen:
            continue
        seen.add(exact)
        pairs.append(f"{exact}:{role}")

    if not any(p.endswith(":primary") for p in pairs):
        return _unsure(row, "no primary muscle was given")

    return dict(row, decision="new", alias_of="", muscles="|".join(pairs),
                evidence=evidence, sources=" ".join(sources), approved="")


def propose_all(rows: list, ontology: dict, client=None, only_pending=True) -> list:
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
        out.append(propose_one(row, ontology, client=client))
    return out
