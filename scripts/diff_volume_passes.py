"""
scripts/diff_volume_passes.py — real before/after deep-diff proofs for the
bar-weight/cross-unit passes. Read-only: no LLM, no server, no DB writes.

Modes:
  capture <out.json>   Build the FULL untrimmed collect() package on the
                       CURRENT code state and save it. Run this BEFORE
                       applying the Pass-2 edit to snapshot the baseline.
  diff <before.json>   Rebuild the package on the current code state and
                       deep-diff against the snapshot. Prints EVERY changed
                       field. PASS only if every change is an intended
                       per-unit volume field and no single-unit rollup's
                       volume value moved.
  fix2                 Retroactive Pass-1 Fix-2 proof: process_data from
                       HEAD's process.py vs HEAD+Fix-2-only, same fetched
                       bundle. PASS only if every changed path is inside
                       daily_workouts.

Package build: collect(query_period_days=365, aggregation_level="session",
include_phase2=True) — spans the 2025-12-26 Deadlift lbs->kg switch, keeps
sessions and all aggregation levels and daily_workouts (untrimmed).
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("FITNOTES_DB_PATH", "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

BUILD_KWARGS = dict(query_period_days=365, aggregation_level="session",
                    include_phase2=True)


def _build_current() -> dict:
    from src.data_agent import collect
    pkg = collect(**BUILD_KWARGS)
    # JSON round-trip so both sides have identical type normalization
    return json.loads(json.dumps(pkg, default=str))


# ── deep diff ──────────────────────────────────────────────────────────────────

ABSENT = "<ABSENT>"


def deep_diff(a, b, path=""):
    """Yield (path, before, after) for every leaf-level difference."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append((p, ABSENT, b[k]))
            elif k not in b:
                out.append((p, a[k], ABSENT))
            else:
                out += deep_diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.<len>", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += deep_diff(x, y, f"{path}[{i}]")
    else:
        if a != b:
            out.append((path, a, b))
    return out


# ── diff mode: Pass-2 per-unit volume proof ────────────────────────────────────

# Every change must match one of these — the intended per-unit volume fields.
_ALLOWED = [re.compile(p) for p in (
    r"^exercises\[\d+\]\.(weekly|monthly|yearly)_aggregations\[\d+\]\.total_volume(_lbs|_kg)?$",
    r"^exercises\[\d+\]\.volume_trend(_lbs|_kg)?$",
    r"^exercises\[\d+\]\.period_volume_(lbs|kg)$",
    r"^muscle_group_summary\[\d+\]\.(total_volume(_lbs|_kg)?|trend(_lbs|_kg)?)$",
    r"^muscle_group_summary\[\d+\]\.weekly_volumes\[\d+\]\.volume(_lbs|_kg)?$",
    r"^muscle_group_balance(\.|\[|$)",
    r"^rankings\.highest_volume(\.|\[|$)",
    r"^all_time_summary\.total_volume_raw_(typed(_lbs|_kg)?|note)$",
    r"^daily_workouts\[\d+\]\.total_volume(_lbs|_kg)?$",
    r"^daily_workouts\[\d+\]\.exercises\[\d+\]\.unit$",
    r"^training_density\.(avg_volume_per_session(_lbs|_kg)?|volume_trend(_lbs|_kg)?)$",
)]


def _allowed(path: str) -> bool:
    return any(rx.search(path) for rx in _ALLOWED)


def _canonicalize(pkg: dict) -> None:
    """
    Sort the order-only nondeterministic spots so cross-process hash
    randomization (PYTHONHASHSEED) cannot masquerade as a value change:
      - inter_exercise_correlation entry order (built from set iteration;
        sort ties were arbitrary before the deterministic tie-break fix)
      - categories_trained / session technique_variants (list(set) ordering)
    Values are untouched — only list order is normalized.
    """
    for ex in pkg.get("exercises", []):
        iec = ex.get("inter_exercise_correlation")
        if isinstance(iec, list):
            iec.sort(key=lambda e: e.get("preceding_exercise", ""))
        for s in ex.get("sessions", []) or []:
            tv = s.get("technique_variants")
            if isinstance(tv, list):
                tv.sort()
        tv = ex.get("technique_variants")
        if isinstance(tv, list) and all(isinstance(x, str) for x in tv):
            tv.sort()
    for day in pkg.get("daily_workouts", []) or []:
        ct = day.get("categories_trained")
        if isinstance(ct, list):
            ct.sort()


_FQM_RX = re.compile(
    r"^exercises\[(\d+)\]\.weekly_aggregations\[(\d+)\]\.form_quality_mode$")


def _is_form_mode_tie(before: dict, path: str, old_val, new_val) -> bool:
    """
    form_quality_mode = max(set(modes), key=count) used to break count-ties in
    hash-randomized set order. A change is a pure tie artifact iff BOTH the old
    and new value are argmax modes of the member sessions' form qualities —
    recomputed here from the baseline's own sessions.
    """
    m = _FQM_RX.match(path)
    if not m:
        return False
    ex   = before["exercises"][int(m.group(1))]
    week = ex["weekly_aggregations"][int(m.group(2))]
    by_date = {s["date"]: s.get("form_quality") for s in ex.get("sessions", [])}
    modes = [by_date[d] for d in week.get("session_dates", []) if d in by_date]
    if not modes or old_val not in modes or new_val not in modes:
        return False
    top = max(modes.count(v) for v in set(modes))
    return modes.count(old_val) == top and modes.count(new_val) == top


def _split_checks(before, after, path=""):
    """
    Walk both trees in parallel. Wherever `before` had a blended volume key and
    `after` has _lbs/_kg buckets, verify:
      kg == 0           -> lbs must equal the old blended value (value didn't move)
      lbs == 0, kg > 0  -> kg  must equal the old blended value
      both > 0          -> genuinely mixed: old blend must equal lbs + raw kg
    Returns (failures, mixed) — mixed entries are (path, old, lbs, kg).
    """
    PAIRS = (
        ("total_volume",            "total_volume_lbs",            "total_volume_kg"),
        ("volume",                  "volume_lbs",                  "volume_kg"),
        ("total_volume_raw_typed",  "total_volume_raw_typed_lbs",  "total_volume_raw_typed_kg"),
        ("avg_volume_per_session",  "avg_volume_per_session_lbs",  "avg_volume_per_session_kg"),
        ("push_volume",             "push_volume_lbs",             "push_volume_kg"),
        ("pull_volume",             "pull_volume_lbs",             "pull_volume_kg"),
    )
    failures, mixed = [], []
    if isinstance(before, dict) and isinstance(after, dict):
        for old_k, lbs_k, kg_k in PAIRS:
            if old_k in before and lbs_k in after and kg_k in after:
                old = before[old_k] or 0.0
                lbs = after[lbs_k] or 0.0
                kg  = after[kg_k]  or 0.0
                p   = f"{path}.{old_k}" if path else old_k
                tol = max(1.0, abs(old) * 1e-6) if old_k == "total_volume_raw_typed" else 0.25
                if kg == 0:
                    if abs(old - lbs) > tol:
                        failures.append(f"{p}: single-unit (lbs) value MOVED: {old} -> {lbs}")
                elif lbs == 0:
                    if abs(old - kg) > tol:
                        failures.append(f"{p}: single-unit (kg) value MOVED: {old} -> {kg}")
                else:
                    mixed.append((p, old, lbs, kg))
                    if abs(old - (lbs + kg)) > tol:
                        failures.append(
                            f"{p}: mixed rollup old blend {old} != lbs {lbs} + raw kg {kg} "
                            f"(= {round(lbs + kg, 1)})")
        for k in set(before) & set(after):
            f, m = _split_checks(before[k], after[k], f"{path}.{k}" if path else str(k))
            failures += f; mixed += m
    elif isinstance(before, list) and isinstance(after, list):
        for i, (x, y) in enumerate(zip(before, after)):
            f, m = _split_checks(x, y, f"{path}[{i}]")
            failures += f; mixed += m
    return failures, mixed


def _exercise_name_for(pkg: dict, path: str) -> str:
    m = re.match(r"^exercises\[(\d+)\]", path)
    if m:
        return pkg["exercises"][int(m.group(1))].get("name", "?")
    m = re.match(r"^daily_workouts\[(\d+)\]", path)
    if m:
        return f"(day {pkg['daily_workouts'][int(m.group(1))].get('date', '?')})"
    return ""


def run_diff(before_path: str) -> int:
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)
    after = _build_current()
    _canonicalize(before)
    _canonicalize(after)

    diffs = deep_diff(before, after)
    print(f"DEEP-DIFF: {len(diffs)} changed leaf field(s)\n")

    disallowed = []
    n_tie = 0
    for p, a, b in diffs:
        if _allowed(p):
            tag = "OK "
        elif _is_form_mode_tie(before, p, a, b):
            tag = "OK-TIE"   # recomputed: both values are argmax count-ties
            n_tie += 1
        else:
            tag = "FAIL"
            disallowed.append((p, a, b))
        name = _exercise_name_for(before, p)
        print(f"  [{tag}] {p}{'  <' + name + '>' if name else ''}")
        print(f"         before: {json.dumps(a, default=str)[:120]}")
        print(f"         after : {json.dumps(b, default=str)[:120]}")
    if n_tie:
        print(f"\n  ({n_tie} form_quality_mode count-tie artifact(s) verified by "
              f"recomputation from member sessions — order-of-set, not value)")

    failures, mixed = _split_checks(before, after)

    print("\n── Single-unit invariant + mixed-rollup reconciliation ──")
    if failures:
        for f_ in failures:
            print(f"  FAIL {f_}")
    else:
        print("  all single-unit volumes unchanged; all mixed blends reconcile to lbs + raw kg")

    # which EXERCISES genuinely have both buckets non-zero
    mixed_ex = sorted({
        before["exercises"][int(m.group(1))]["name"]
        for p, *_ in mixed
        for m in [re.match(r"^exercises\[(\d+)\]", p)] if m
    } | {
        ex["name"] for ex in after.get("exercises", [])
        if (ex.get("period_volume_lbs") or 0) > 0 and (ex.get("period_volume_kg") or 0) > 0
    })
    print(f"\nEXERCISES WITH BOTH BUCKETS NON-ZERO: {mixed_ex or 'none'}")

    # Deadlift breakdown
    print("\n── Deadlift old-blend vs new buckets ──")
    dl_b = next((e for e in before["exercises"] if e["name"] == "Deadlift"), None)
    dl_a = next((e for e in after["exercises"]  if e["name"] == "Deadlift"), None)
    if dl_b and dl_a:
        for level in ("yearly_aggregations", "monthly_aggregations"):
            for eb, ea in zip(dl_b.get(level, []), dl_a.get(level, [])):
                key = eb.get("year") or eb.get("month")
                old = eb.get("total_volume")
                lbs, kg = ea.get("total_volume_lbs"), ea.get("total_volume_kg")
                note = ""
                if lbs and kg:
                    note = (f"   OLD BLEND {old} == lbs {lbs} + RAW kg number {kg} "
                            f"(sum={round(lbs + kg, 1)}) — was adding kg onto lbs")
                print(f"  {level[:-13]:8s} {key}: old_blended={old}  ->  "
                      f"volume_lbs={lbs}  volume_kg={kg}{note}")
    else:
        print("  Deadlift not in window — cannot show breakdown")

    print()
    if disallowed or failures:
        print(f"FAIL — {len(disallowed)} non-volume change(s), {len(failures)} value failure(s)")
        return 1
    print("PASS — every changed field is an intended per-unit volume bucket")
    return 0


# ── fix2 mode: retroactive Pass-1 Fix-2 isolation proof ───────────────────────

_FIX2_OLD = """            working = [s for s in ex_sets if not s["is_warmup"]] or ex_sets
            max_w   = max(s["weight"] for s in working)
            vol     = sum((s["weight"] + bar_weight) * s["reps"] for s in ex_sets)
            e1rm    = max((_epley_1rm(s["weight"], s["reps"]) for s in working
                           if s["reps"] > 0), default=0.0)"""

_FIX2_NEW = """            working = [s for s in ex_sets if not s["is_warmup"]] or ex_sets
            max_w   = max(s["weight"] + bar_weight for s in working)
            vol     = sum((s["weight"] + bar_weight) * s["reps"] for s in ex_sets)
            e1rm    = max((_epley_1rm(s["weight"] + bar_weight, s["reps"]) for s in working
                           if s["reps"] > 0), default=0.0)"""


def _load_module_from_source(source: str, name: str):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                      encoding="utf-8")
    tmp.write(source); tmp.close()
    spec = importlib.util.spec_from_file_location(name, tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_fix2() -> int:
    from datetime import date, timedelta
    from src.data_agent.fetch import fetch_data, load_user_context

    head_src = subprocess.run(
        ["git", "show", "HEAD:src/data_agent/process.py"],
        capture_output=True, text=True, check=True).stdout
    if _FIX2_OLD not in head_src:
        print("FAIL — could not locate the pre-Fix-2 block in HEAD process.py")
        return 1
    fix2_src = head_src.replace(_FIX2_OLD, _FIX2_NEW, 1)

    mod_head = _load_module_from_source(head_src, "process_head")
    mod_fix2 = _load_module_from_source(fix2_src, "process_head_fix2")

    ctx     = load_user_context()
    today   = date.today()
    end_str = today.strftime("%Y-%m-%d")
    bundle  = fetch_data(end_str)
    start_str = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    args = (bundle, ctx, start_str, end_str, today, 365, None, None,
            "session", True)

    pkg_head = json.loads(json.dumps(mod_head.process_data(*args), default=str))
    pkg_fix2 = json.loads(json.dumps(mod_fix2.process_data(*args), default=str))

    diffs = deep_diff(pkg_head, pkg_fix2)
    inside  = [d for d in diffs if d[0].startswith("daily_workouts")]
    outside = [d for d in diffs if not d[0].startswith("daily_workouts")]

    print(f"RETRO FIX-2 DIFF (HEAD vs HEAD+Fix2 only): {len(diffs)} changed field(s)")
    print(f"  inside  daily_workouts: {len(inside)}")
    print(f"  outside daily_workouts: {len(outside)}")
    bad_inside = [p for p, _, _ in inside
                  if not re.search(r"\.(estimated_1rm|max_weight)$", p)]
    for p, a, b in outside:
        print(f"  FAIL (outside): {p}: {a} -> {b}")
    for p in bad_inside:
        print(f"  FAIL (inside, not e1rm/max_weight): {p}")
    sample = [d for d in inside if re.search(r"\.(estimated_1rm|max_weight)$", d[0])]
    for p, a, b in sample[:6]:
        print(f"  ok: {p}: {a} -> {b}")
    if len(sample) > 6:
        print(f"  ... and {len(sample) - 6} more e1rm/max_weight changes")

    print()
    if outside or bad_inside:
        print("FAIL — Fix-2 leaked outside daily_workouts e1rm/max_weight")
        return 1
    print("PASS — everything outside daily_workouts byte-identical; "
          "inside, only exercises[].estimated_1rm/max_weight changed")
    return 0


# ── entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        pkg = _build_current()
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(pkg, f, default=str)
        print(f"captured {len(json.dumps(pkg)) / 1024:.0f} KB package -> {sys.argv[2]}")
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "diff":
        sys.exit(run_diff(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == "fix2":
        sys.exit(run_fix2())
    print(__doc__)
    sys.exit(2)
