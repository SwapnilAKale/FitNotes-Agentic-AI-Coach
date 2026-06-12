"""
scripts/diagnose_broad.py — read-only diagnostic for broad-scope package builds.

Verifies, independently of any agent report:
  1. The no-filter question builds as scope='broad' / ~397 KB via the SAME
     call _run_analytical makes (prepare_analysis_package, same arguments).
  2. G6 hard-stops a deliberately mislabeled focused/60-exercise package.
  3. Scope derivation across the four filter cases.
  4. The input size each Analysis Agent LLM call WOULD send (no LLM called).

Changes nothing. Calls no LLM. Starts no server.
Run: python scripts/diagnose_broad.py
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import (  # noqa: E402
    prepare_analysis_package,
    validate,
    DataAgentIntegrityError,
    _report_violations,
)
from src.analysis_agent import _build_user_message  # noqa: E402

QUESTION = "How has my overall training gone this past year?"

failures: list = []


def _size_kb(obj) -> float:
    return len(json.dumps(obj, default=str).encode()) / 1024


def _build(**kwargs) -> dict:
    # Exactly the call coordinator._run_analytical makes:
    #   prepare_analysis_package(query_period_days=..., exercise_names=...,
    #                            muscle_groups=..., include_phase2=True)
    return prepare_analysis_package(
        query_period_days=kwargs.get("query_period_days", 365),
        exercise_names=kwargs.get("exercise_names"),
        muscle_groups=kwargs.get("muscle_groups"),
        include_phase2=True,
    )


# ── 1. No-filter question (the production case) ───────────────────────────────

print("=" * 70)
print(f"QUESTION: {QUESTION!r}  (no exercise filter, no muscle-group filter)")
print("=" * 70)

pkg = _build()
scope = pkg.get("scope")
n_ex = len(pkg.get("exercises", []))
size_kb = _size_kb(pkg)
ok1 = scope == "broad" and size_kb < 450

print(f"  SCOPE          = {scope}")
print(f"  EXERCISES      = {n_ex}")
print(f"  SIZE_KB        = {size_kb:.1f}")
print(f"  EXPECTED       = broad / 60 / ~397 KB")
print(f"  PASS/FAIL      = {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    failures.append("no-filter scope/size")

# ── 2. G6 hard-stop proof ──────────────────────────────────────────────────────

print()
print("G6 hard-stop proof (broad 60-exercise package relabeled scope='focused'):")
mislabeled = dict(pkg)
mislabeled["scope"] = "focused"
g6_raised = False
try:
    # validate() collects violations; _report_violations() is the raising step
    # that prepare_analysis_package runs on the _run_analytical path.
    _report_violations(validate(mislabeled), "diagnose_broad")
except DataAgentIntegrityError:
    g6_raised = True

print(f"  G6_RAISED      = {g6_raised}")
print(f"  EXPECTED       = True")
print(f"  PASS/FAIL      = {'PASS' if g6_raised else 'FAIL'}")
if not g6_raised:
    failures.append("G6 hard-stop")

# ── 3. Scope matrix ────────────────────────────────────────────────────────────

print()
print("Scope matrix:")
matrix = [
    ("no filter",                            {},                                              "broad"),
    ("exercise_names=['Lat Pulldown']",      {"exercise_names": ["Lat Pulldown"]},            "focused"),
    ("exercise_names=['Notarealexercise']",  {"exercise_names": ["Notarealexercise"]},        "broad"),
    ("muscle_groups=['Back']",               {"muscle_groups": ["Back"]},                     "group"),
]
for label, kwargs, expected in matrix:
    p = _build(**kwargs)
    s = p.get("scope")
    kb = _size_kb(p)
    verdict = "PASS" if s == expected else "FAIL"
    print(f"  {label:40s} scope={s:<8s} size={kb:7.1f} KB  expect={expected:<8s} {verdict}")
    if verdict == "FAIL":
        failures.append(f"scope matrix: {label}")

# ── 4. Token-cost readout (measure inputs only — NO LLM call) ─────────────────

print()
print("Token-cost readout (input that WOULD be sent; no LLM called):")

# Draft call input: _build_user_message(package, question, research=None,
# memories=None, conversation_context=None) — package serialized compact inside.
draft_input = _build_user_message(pkg, QUESTION, None, None, None)
draft_kb = len(draft_input.encode()) / 1024

# Grounding call input (analysis_agent.ground_check): runs for ALL scopes,
# sending the draft plus the FULL compact package (no subset, no indent).
# Draft text itself is unknown without an LLM; use a representative
# ~2000-char placeholder — the package dwarfs it.
placeholder_draft = "x" * 2000
grounding_input = (
    f"[DRAFT ANSWER]\n{placeholder_draft}\n\n"
    f"[WORKOUT PACKAGE]\n{json.dumps(pkg, separators=(',', ':'))}"
)
grounding_kb = len(grounding_input.encode()) / 1024

total_kb = draft_kb + grounding_kb
est_tokens_k = total_kb * 1024 / 4 / 1000  # ~4 bytes/token for compact JSON
ceiling_ok = est_tokens_k < 250

print(f"  DRAFT_INPUT_KB     = {draft_kb:.1f}")
print(f"  GROUNDING_INPUT_KB = {grounding_kb:.1f}  (draft placeholder ~2 KB + full compact package)")
print(f"  TOTAL_PER_QUESTION_KB = {total_kb:.1f}")
print(f"  EST_TOKENS         = ~{est_tokens_k:.0f}k  "
      f"({'under' if ceiling_ok else 'OVER'} the 250k/min ceiling)")
print(f"  WORST_CASE (coverage-check retry re-runs draft+grounding): "
      f"{total_kb * 2:.1f} KB = ~{est_tokens_k * 2:.0f}k tokens across two rounds")
print(f"  NOTE: grounding runs for ALL scopes and re-sends the FULL compact "
      f"package (no subset) — every fact the draft used is findable, so the "
      f"REMOVE rule cannot strip a true claim for a missing source field.")
if not ceiling_ok:
    failures.append("token budget over 250k/min ceiling")

# ── Verdict ────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
else:
    print("ALL PASS")
