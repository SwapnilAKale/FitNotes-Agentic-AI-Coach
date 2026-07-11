"""
scripts/diagnose_recency.py — read-only live diagnosis of Issue 3's RECENCY
sub-class (the "wrong-predicate" class).

Target (confirmed live): Sumo Squats.
  true latest session   = 2026-06-25  (no comment)
  salient session       = 2026-06-15  (pain-flagged "slight pain at the knee",
                                        the last-commented session — NOT the latest)

The documented failure: the draft narrates a "your most recent session" claim and
attaches the SALIENT (pain/comment) date 06-15 instead of the true latest 06-25.
progression.latest_session_date IS correct (= 06-25), so any error is bind-side.

This script forces a recency claim and, per run, prints:
  1. the raw PRE-STRIP tagged draft (what the model actually cited),
  2. every citation tag (field_path / status / value / extracted claim_number),
  3. the grounding-context mode + the exact [CITED VALUES] block grounding sees,
  4. whether the wrong date SURVIVES grounding into the cleaned answer.

It then classifies each run:
  X  = recency leaf cited but wrong/salient date in prose, misquote NOT caught
  Z  = salient date appears in prose with NO citation tag on it
  CLEAN = draft used the true latest date (no reproduction this run)

The analytical path is read-only (no DB writes; writes are the operational path).
This calls the live Gemini API but touches nothing in FitNotes.
Run: python scripts/diagnose_recency.py [runs]     (default 3 runs)
"""

import asyncio
import os
import sqlite3
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent import prepare_analysis_package        # noqa: E402
from src import citations as C                              # noqa: E402
from src import analysis_agent                              # noqa: E402

EXERCISE = "Sumo Squats"
QUESTION = f"When was my most recent {EXERCISE} session and how did it go?"
DB_PATH  = os.environ["FITNOTES_DB_PATH"]


def _date_variants(iso: str) -> list:
    """Human forms a model might write for a YYYY-MM-DD date."""
    d = datetime.strptime(iso, "%Y-%m-%d").date()
    month = d.strftime("%B")                     # June
    variants = {
        iso,                                     # 2026-06-15
        iso[5:],                                 # 06-15
        f"{month} {d.day}",                      # June 15
        f"{month} {d.day},",                     # June 15,
        f"{month} {d.day} {d.year}",             # June 15 2026
        f"{d.day} {month}",                      # 15 June
        f"{d.month}/{d.day}",                    # 6/15
    }
    return [v for v in variants if v]


def _first_hit(text: str, variants: list):
    """Return (matched_variant, index) of the earliest variant occurrence, else (None,-1)."""
    low = text.lower()
    best = (None, -1)
    for v in variants:
        i = low.find(v.lower())
        if i != -1 and (best[1] == -1 or i < best[1]):
            best = (v, i)
    return best


def _context(text: str, idx: int, span: int = 90) -> str:
    a = max(0, idx - span)
    b = min(len(text), idx + span)
    return text[a:b].replace("\n", " ")


def _ground_truth():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute(
        """SELECT MAX(tl.date) AS last_date
           FROM training_log tl JOIN exercise e ON tl.exercise_id=e._id
           WHERE e.name=?""", (EXERCISE,)).fetchone()
    con.close()
    return row["last_date"][:10]


async def _one_run(run_no: int, pkg: dict, true_date: str, salient_date: str) -> str:
    true_vars    = _date_variants(true_date)
    salient_vars = _date_variants(salient_date)

    print("\n" + "═" * 78)
    print(f"RUN {run_no}")
    print("═" * 78)

    draft_tagged = await analysis_agent.analyze(pkg, QUESTION)

    print("\n─── PRE-STRIP TAGGED DRAFT ───")
    print(draft_tagged)

    # ── Tags ──
    cited = C.extract_cited_values(draft_tagged, pkg)
    print("\n─── CITATION TAGS ───")
    if not cited:
        print("  (no tags emitted)")
    for c in cited:
        print(f"  {c['collection']}|{c['match_key']}|{c['field_path']}"
              f"  status={c['status']}  value={c['value']!r}"
              f"  claim_number={c['claim_number']!r}")

    recency_leaf_tags = [
        c for c in cited
        if c["field_path"] in ("progression.latest_session_date",
                               "progression.last_session_date",
                               "training_frequency.last_session_date")
    ]

    # ── Where does each date land in the DRAFT prose? ──
    sal_v, sal_i   = _first_hit(draft_tagged, salient_vars)
    true_v, true_i = _first_hit(draft_tagged, true_vars)
    print("\n─── DATE BINDING IN DRAFT ───")
    sal_desc  = f"FOUND {sal_v!r}" if sal_v else "absent"
    true_desc = f"FOUND {true_v!r}" if true_v else "absent"
    print(f"  salient {salient_date}: {sal_desc}"
          + (f"  ctx: …{_context(draft_tagged, sal_i)}…" if sal_v else ""))
    print(f"  true    {true_date}: {true_desc}"
          + (f"  ctx: …{_context(draft_tagged, true_i)}…" if true_v else ""))

    # Is the salient date tagged, or bare? Find the nearest tag AFTER the salient hit.
    salient_tagged = None
    if sal_v:
        after = draft_tagged[sal_i:]
        tags_after = C.parse_tags(after)
        salient_tagged = tags_after[0].raw if tags_after else None
        # only "immediately after" counts — same clause (within ~60 chars)
        if tags_after and after.find(tags_after[0].raw) > 60:
            salient_tagged = None
    print(f"  salient date carries a citation tag immediately after? "
          f"{salient_tagged if salient_tagged else 'NO'}")

    # ── Grounding context ──
    gctx = C.build_grounding_context(cited, pkg)
    print("\n─── GROUNDING CONTEXT ───")
    print(f"  mode = {gctx.get('mode')}")
    if gctx.get("mode") == "cheap":
        for cv in gctx["cited_values"]:
            print(f"    {cv['location']} = {cv['value']}   (answer stated: {cv['claim_number']})")

    # ── Grounding result: does the wrong date survive? ──
    cleaned, flagged = await analysis_agent.ground_check(draft_tagged and C.strip_tags(draft_tagged), gctx)
    a_sal_v, a_sal_i   = _first_hit(cleaned, salient_vars)
    a_true_v, a_true_i = _first_hit(cleaned, true_vars)
    print("\n─── AFTER GROUNDING (cleaned answer) ───")
    print(f"  flagged claims: {len(flagged)}")
    for f in flagged:
        print(f"    - {f.get('action')}: {f.get('original_claim')!r} ({f.get('reason')})")
    a_sal_desc  = f"YES {a_sal_v!r}" if a_sal_v else "no"
    a_true_desc = f"YES {a_true_v!r}" if a_true_v else "no"
    print(f"  salient {salient_date} in answer: {a_sal_desc}"
          + (f"  ctx: …{_context(cleaned, a_sal_i)}…" if a_sal_v else ""))
    print(f"  true    {true_date} in answer: {a_true_desc}")

    # ── B1 CHECK: grounding must not delete true set-count claims or edit
    # display lines (the live-reproduced grounding false-positive) ──
    display = pkg.get("display_sets") or []
    draft_stripped = C.strip_tags(draft_tagged)
    b1_ok = True
    print("\n─── B1 CHECK (grounding false-positive) ───")
    for line in display:
        if line in draft_stripped and line not in cleaned:
            print(f"  B1b VIOLATION: display line edited/removed by grounding: {line!r}")
            b1_ok = False
    if "4 sets" in draft_stripped and "4 sets" not in cleaned:
        print("  B1a VIOLATION: true '4 sets' claim removed by grounding")
        b1_ok = False
    set_flags = [f for f in flagged
                 if "set" in (str(f.get("original_claim")) + str(f.get("reason"))).lower()]
    for f in set_flags:
        print(f"  set-related flag: {f.get('action')}: {f.get('original_claim')!r} "
              f"({f.get('reason')})")
    print(f"  B1 verdict: {'PASS' if b1_ok else 'FAIL'}")

    # ── Classify ──
    if not sal_v:
        verdict = "CLEAN (salient date not in draft — no reproduction this run)"
    elif salient_tagged and recency_leaf_tags:
        verdict = "X (recency leaf cited, salient date in prose — check misquote-catch above)"
    elif salient_tagged:
        verdict = "X? (salient date tagged, but not to a recency leaf — inspect field_path)"
    else:
        verdict = "Z (salient date in prose with NO citation tag — uncheckable)"
    # Refine X by whether it survived grounding
    survived = bool(a_sal_v)
    print("\n─── RUN VERDICT ───")
    print(f"  {verdict}")
    print(f"  wrong (salient) date SURVIVED grounding: {survived}")
    return (f"RUN {run_no}: {verdict} | survived={survived} | "
            f"B1={'PASS' if b1_ok else 'FAIL'}")


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set — cannot run live diagnosis.")
        sys.exit(2)

    true_date = _ground_truth()
    print(f"GROUND TRUTH (direct read-only SQL): latest {EXERCISE} session = {true_date}")

    pkg = prepare_analysis_package(
        query_period_days=90, exercise_names=[EXERCISE], include_phase2=True)
    ex = next((e for e in pkg.get("exercises", []) if e.get("name") == EXERCISE), None)
    if ex is None:
        print(f"Package has no '{EXERCISE}' entry — aborting.")
        sys.exit(2)
    prog = ex.get("progression") or {}
    pkg_latest = prog.get("latest_session_date")
    salient = (ex.get("pain_analysis") or {}).get("pain_session_dates") or []
    salient_date = salient[0] if salient else "2026-06-15"

    print(f"PACKAGE progression.latest_session_date = {pkg_latest}")
    print(f"PACKAGE salient (pain) date             = {salient_date}")
    assert pkg_latest == true_date, (
        f"FIELD BUG (not this issue): package latest {pkg_latest} != SQL truth {true_date}")
    assert salient_date != true_date, "salient date == true date — no repro setup"
    print("Sanity OK: recency field is correct; any error is bind-side.\n")

    summary = []
    for i in range(1, runs + 1):
        summary.append(await _one_run(i, pkg, true_date, salient_date))

    print("\n" + "█" * 78)
    print("SUMMARY")
    print("█" * 78)
    for line in summary:
        print("  " + line)


if __name__ == "__main__":
    asyncio.run(main())
