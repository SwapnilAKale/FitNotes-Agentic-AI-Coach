"""
scripts/probe_stage1_classify.py — read-only live probe of the Decomposition
Arc Stage-1 classifier schema (the emit-inert `requests` array).

Calls ONLY Coordinator._classify per message — no DB writes, no MCP tools, no
routing, no pipeline. One Gemini request per message (~10 per run against the
500/day free tier).

Per message it prints:
  1. the flat route + parameter fields (whole-message semantics — must match
     pre-Stage-1 behavior on single-part messages),
  2. the requests array: index / lane / intent_text / notable per-chunk params,
  3. a DIVERGENCE marker when the flat fields and the chunk array disagree
     (log-only in production; surfaced here for eyeballing),
  4. REQUESTS=None marker when the array was missing or dropped by the
     sanitizer (check stderr for the [coordinator] warning explaining why).

Run:  python scripts/probe_stage1_classify.py
Run TWICE (temp 0) — chunking must be two-run consistent to pass.
"""

import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True) or None)
except Exception:
    pass

os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.coordinator import Coordinator                  # noqa: E402

# (label, message, history_seed or None, expectation-notes)
PROBES = [
    ("1 single analytical (regression)",
     "How is my Lat Pulldown progressing?", None,
     "1 chunk mirroring flat; route analytical"),
    ("2 single write",
     "Log bench press 100x5 for today", None,
     "1 operational chunk (probe hits _classify directly; live traffic would "
     "hit the write-intent pre-guard first)"),
    ("3 analytical + write (the confirmed-drop case)",
     "Is my squat progressing and log bench 100 lbs for 5 reps today", None,
     "2 chunks: analytical squat + operational log; self-contained intent_texts"),
    ("4 analytical + research",
     "How's my chest volume this month, and what does science say about "
     "training frequency?", None,
     "2 chunks: analytical (Chest, ~30d) + operational research"),
    ("5 recall + analytical (history-seeded)",
     "what was that number again, and how's my deadlift trending?",
     [{"role": "user", "content": "What's my Lat Pulldown PR?"},
      {"role": "assistant",
       "content": "Your Lat Pulldown PR is 145 lbs, set on 2026-06-02."}],
     "2 chunks: recall (the 145 figure) + analytical deadlift"),
    ("6 out_of_scope + analytical",
     "Write me a poem about my bench press and show my bench PR", None,
     "2 chunks: out_of_scope poem + analytical PR"),
    ("7 single display (regression)",
     "Show me my last leg day", None,
     "1 chunk; display_intent true on flat AND chunk"),
    ("8 anti-split (ONE request, two exercises)",
     "Compare my squat and bench progress over the last 3 months", None,
     "EXACTLY 1 chunk with both exercise_names — must NOT split"),
    ("9 analytical + write (delete)",
     "Am I overtraining? Also delete my deadlift goal", None,
     "2 chunks: analytical + operational delete"),
    ("10 research + custom-SQL analytical",
     "Does creatine help, and how many days did I train last month "
     "excluding Sundays?", None,
     "2 chunks: operational research + analytical with needs_custom_sql true"),
]

_FLAT_KEYS = ["route", "display_intent", "exercise_names", "muscle_groups",
              "query_period_days", "rep_target", "cardio_lock",
              "needs_custom_sql", "custom_sql_intent"]


def _fmt_chunk(c: dict) -> str:
    extras = []
    for k in ("display_intent", "exercise_names", "muscle_groups",
              "query_period_days", "rep_target", "cardio_lock",
              "needs_custom_sql", "custom_sql_intent"):
        v = c.get(k)
        if v not in (None, False, 90):          # show only non-default values
            extras.append(f"{k}={v!r}")
    extra_s = ("  [" + ", ".join(extras) + "]") if extras else ""
    return f"    [{c['index']}] lane={c['lane']:<12} {c['intent_text']!r}{extra_s}"


def _divergence(params: dict) -> list:
    reqs = params.get("requests") or []
    if not reqs:
        return []
    out = []
    lanes = {c.get("lane") for c in reqs}
    if params.get("route") not in lanes:
        out.append(f"flat route {params.get('route')!r} not among lanes {sorted(lanes)}")
    for field in ("exercise_names", "muscle_groups"):
        flat = set(params.get(field) or [])
        union = set()
        for c in reqs:
            union.update(c.get(field) or [])
        if flat != union:
            out.append(f"flat {field} {sorted(flat)} != chunk union {sorted(union)}")
    return out


async def main() -> None:
    coord = Coordinator(agent_session=None)     # no operational agent needed
    for label, msg, history, expect in PROBES:
        coord._history = list(history) if history else []
        print("=" * 78)
        print(f"### {label}")
        print(f"MSG: {msg}")
        print(f"EXPECT: {expect}")
        try:
            params = await coord._classify(msg)
        except Exception as e:
            print(f"!! classify raised: {e}")
            continue
        if params.get("_parse_failed"):
            print("!! _parse_failed — classify degraded to the analytical default")
            continue
        flat = {k: params.get(k) for k in _FLAT_KEYS
                if params.get(k) not in (None, False)}
        print(f"FLAT: {json.dumps(flat, default=str)}")
        reqs = params.get("requests")
        if reqs is None:
            print("REQUESTS=None (missing or dropped — see [coordinator] warning)")
        else:
            print(f"REQUESTS ({len(reqs)} chunk{'s' if len(reqs) != 1 else ''}):")
            for c in reqs:
                print(_fmt_chunk(c))
        for d in _divergence(params):
            print(f"DIVERGENCE: {d}")
    print("=" * 78)
    print("Run complete. Run this script a SECOND time and compare: chunk "
          "counts, lanes, and intent_texts must be consistent run-to-run.")


if __name__ == "__main__":
    asyncio.run(main())
