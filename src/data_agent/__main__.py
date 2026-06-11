"""
src/data_agent/__main__.py
CLI harness — run with:  python -m src.data_agent [period] [end_date] [exercises...]

Examples:
  python -m src.data_agent 90
  python -m src.data_agent all
  python -m src.data_agent 90 2026-05-28 "Lat Pulldown" "Deadlift"
"""

import sys
import json
from datetime import datetime

from . import collect, prepare_analysis_package


def _looks_like_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _print_summary(data: dict) -> None:
    print(f"\n  Period     : {data['query_start_date']} -> {data['query_end_date']}")
    print(f"  Aggregation: {data['aggregation_level']}")
    print(f"  Exercises  : {data['total_exercises_analyzed']}")
    cons = data["training_consistency"]
    if cons:
        print(f"  Consistency: {cons['distinct_training_days']} days  "
              f"{cons['sessions_per_week']}/week  {cons['weeks_missed']} weeks missed")
    bw = data["bodyweight"]
    if bw.get("current_kg"):
        print(f"  Bodyweight : {bw['current_kg']} kg  trend={bw['trend']}")
    ats = data.get("all_time_summary", {})
    if ats:
        print(f"  ALL-TIME   first={ats['first_training_date']}  "
              f"days={ats['total_training_days']}  sets={ats['total_sets']}  "
              f"streak={ats['longest_streak_days']}d  gap={ats['longest_gap_days']}d  "
              f"PRs={ats['total_prs_alltime']}")
    if data["goals"]:
        print(f"\n  GOALS")
        for g in data["goals"]:
            proj = g.get("projection") or {}
            print(f"    {g['exercise_name']}: {g['target_weight']} {g['unit']} "
                  f"x{g['target_reps']} by {g['target_date']}  "
                  f"on_track={proj.get('is_on_track')}  "
                  f"projected={proj.get('projected_achievement_date')}")
    print(f"\n  MUSCLE GROUP SUMMARY")
    print(f"  {'-'*68}")
    for mg in data["muscle_group_summary"]:
        rr = mg["rep_ranges"]
        print(f"  {mg['muscle_group']:12s}  ex={mg['exercise_count']:2d}  "
              f"sets={mg['total_sets']:4d}  vol={mg['total_volume']:10.0f}  "
              f"trend={mg['trend']:20s}  "
              f"S={rr['strength_pct']}% H={rr['hypertrophy_pct']}% E={rr['endurance_pct']}%")
    bal = data["muscle_group_balance"]
    if bal:
        print(f"\n  PUSH/PULL  push={bal['push_volume']:.0f}  "
              f"pull={bal['pull_volume']:.0f}  ratio={bal['push_pull_ratio']}  "
              f"dominant={bal['dominant_type']}")
    dow = data["day_of_week_patterns"]
    if dow:
        counts = {d["day"]: d["count"] for d in dow["distribution"]}
        print(f"  DAY OF WEEK  most={dow['most_common_day']}  skipped={dow['most_skipped_day']}")
        print("    " + "  ".join(f"{d[:3]}={counts.get(d,0)}" for d in
              ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
    ranks = data.get("rankings", {})
    if ranks.get("fastest_improving"):
        print(f"\n  FASTEST IMPROVING (top 5)")
        for r in ranks["fastest_improving"][:5]:
            print(f"    {r['exercise']:40s}  {r['value']:+.1f}%")
    if ranks.get("most_stagnant"):
        print(f"\n  MOST STAGNANT (top 5)")
        for r in [x for x in ranks["most_stagnant"] if x["value"] and x["value"] > 0][:5]:
            print(f"    {r['exercise']:40s}  {r['value']} days")
    lc = data.get("exercise_lifecycle", {})
    if lc.get("abandoned"):
        print(f"\n  ABANDONED EXERCISES ({len(lc['abandoned'])})")
        for e in lc["abandoned"][:5]:
            print(f"    {e['exercise_name']:40s}  last={e['last_date']}  ({e['days_since_last']}d ago)")
    if lc.get("substitutions"):
        print(f"\n  SUBSTITUTIONS DETECTED")
        for s in lc["substitutions"][:3]:
            print(f"    {s['stopped_exercise']} -> {s['started_exercise']}  ({s['category']})")


def _print_exercise(data: dict, exercise_name: str) -> None:
    ex = next((e for e in data["exercises"] if e["name"] == exercise_name), None)
    if ex is None:
        print(f"\n[NOT FOUND] '{exercise_name}'"); return
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  {ex['name']}  [{ex['category']}]  unit={ex['unit']}  "
          f"offset={ex['numeric_offset']}  bar={ex['bar_weight']} {ex['bar_weight_unit']}")
    print(sep)
    prog = ex["progression"]
    if prog:
        print(f"\n  PROGRESSION")
        pct_str = f"({prog['weight_change_pct']:+.1f}%)" if prog['weight_change_pct'] is not None else "(started from 0)"
        print(f"    {prog['first_session_date']} -> {prog['last_session_date']}  "
              f"{prog['display_weight_start']} -> {prog['display_weight_end']}  "
              f"{pct_str}  e1RM: {prog['e1rm_start']} -> {prog['e1rm_end']}")
        if prog.get("plateau_since"):
            print(f"    Plateau since {prog['plateau_since']} ({ex['plateau_days']} days)")
        if prog.get("regression_from_peak"):
            r = prog["regression_from_peak"]
            print(f"    REGRESSION: peak {r['peak_weight']} on {r['peak_date']} -> "
                  f"now {r['current_weight']} ({r['regression_pct']:.1f}% drop)")
        if prog.get("diminishing_returns"):
            dr = prog["diminishing_returns"]
            print(f"    Returns: {dr['pattern']}  "
                  f"{dr['early_rate_per_month']:+.2f} -> {dr['recent_rate_per_month']:+.2f} /mo")
    if ex.get("pr"):
        pr = ex["pr"]
        print(f"\n  PR (all-time)  {pr['weight']} {pr['unit']} x {pr['reps']}  "
              f"({pr['date']})  e1RM={pr['estimated_1rm']}")
    if ex.get("pr_period"):
        pp = ex["pr_period"]
        print(f"  PR (period)    {pp['weight']} {pp['unit']} x {pp['reps']}  "
              f"({pp['date']})  e1RM={pp['estimated_1rm']}")
    freq = ex["training_frequency"]
    if freq:
        gap_str = f"{freq['avg_days_between']}d" if freq.get('avg_days_between') is not None else "N/A"
        print(f"\n  FREQUENCY  {freq['sessions_per_week']}/week  "
              f"gap={gap_str}  since_last={freq['days_since_last']}d")
    rr = ex["rep_range_distribution"]
    print(f"  REP RANGES  S={rr['strength_pct']}%  H={rr['hypertrophy_pct']}%  "
          f"E={rr['endurance_pct']}%  dominant={rr['dominant_range']}")
    proj = ex.get("e1rm_projection", {})
    if proj:
        print(f"  e1RM PROJ   30d={proj['projected_30d']}  60d={proj['projected_60d']}  "
              f"90d={proj['projected_90d']}  conf={proj['projection_confidence']}")
    pa = ex["pain_analysis"]
    print(f"  PAIN        {pa['pain_session_count']} sessions  "
          f"{pa['failed_attempt_count']} failed attempts")
    if ex.get("technique_variants"):
        print(f"  TECHNIQUES  " +
              "  ".join(f"{tv['variant']}(n={tv['session_count']},e1RM={tv['avg_e1rm']})"
                        for tv in ex["technique_variants"]))
    rpb = ex.get("rest_performance_buckets") or {}
    if rpb.get("buckets"):
        bucket_str = "  ".join(
            f"{b['rest_range']}(n={b['n']},mean={b['mean_e1rm']},ci={b['ci_95']})"
            for b in rpb["buckets"])
        print(f"  REST EFFECT  {bucket_str}")
        if rpb.get("comparison"):
            c = rpb["comparison"]
            print(f"    → best={c['best_bucket']} vs {c['worst_bucket']}  "
                  f"diff={c['mean_diff_e1rm']}  d={c['cohen_d']}  "
                  f"ci_overlap={c['cis_overlap']}  [{c['confidence_label']}]")
    if ex.get("inter_exercise_correlation"):
        print(f"  INTER-EX (top 3)")
        for c in ex["inter_exercise_correlation"][:3]:
            print(f"    {c['preceding_exercise']:35s}  "
                  f"diff={c['mean_diff_e1rm']:+.1f}  d={c['cohen_d']}  "
                  f"n={c['n_preceded']}v{c['n_not_preceded']}  "
                  f"ci_overlap={c['cis_overlap']}  [{c['confidence_label']}]  {c['effect']}")
    lc = ex.get("learning_curve", {})
    if lc:
        print(f"  LEARNING    first={lc['first_ever_session']}  "
              f"sessions_to_PR={lc['sessions_to_first_pr']}  "
              f"total_alltime={lc['total_sessions_alltime']}")
    dur_prog = ex.get("duration_progression")
    if dur_prog and dur_prog.get("session_count", 0) >= 2:
        print(f"\n  DURATION PROGRESSION")
        print(f"    {dur_prog['first_session_date']} -> {dur_prog['last_session_date']}  "
              f"{dur_prog['duration_start_seconds']}s -> {dur_prog['duration_end_seconds']}s  "
              f"({dur_prog['duration_change_pct']:+.1f}%)  "
              f"peak={dur_prog['duration_peak_seconds']}s on {dur_prog['duration_peak_date']}")
    dist_prog = ex.get("distance_progression")
    if dist_prog:
        print(f"\n  DISTANCE PROGRESSION")
        print(f"    {dist_prog['first_session_date']} -> {dist_prog['last_session_date']}  "
              f"avg={dist_prog['avg_distance_km']}km/session  "
              f"peak={dist_prog['distance_peak_km']}km on {dist_prog['distance_peak_date']}  "
              f"total={dist_prog['total_distance_km']}km")
    agg = data["aggregation_level"]
    if agg == "session" and ex["sessions"]:
        print(f"\n  LAST 3 SESSIONS")
        for s in ex["sessions"][-3:]:
            dist_s = f"  total_dist={s['total_distance']}km" if s.get('total_distance', 0) > 0 else ""
            dur_s  = f"  total_dur={s['total_duration_seconds']}s" if s.get('total_duration_seconds', 0) > 0 else ""
            print(f"\n    {s['date']}  max={s['max_working_weight']} {s['unit']}  "
                  f"e1RM={s['estimated_1rm']}  vol={s['total_volume']}  "
                  f"sets={s['working_sets_count']}  form={s['form_quality']}"
                  f"{dist_s}{dur_s}")
            for i, st in enumerate(s["sets"]):
                tags = ("(W)" if st["is_warmup"] else "") + \
                       (f"[D{st['drop_group']}]" if st["drop_group"] else "") + \
                       ("[PAIN]" if st["is_pain_flag"] else "") + \
                       ("[FAIL]" if st["is_failed_attempt"] else "")
                cmt  = f"  [{st['comment']}]" if st["comment"] else ""
                dist_tag = f"  dist={st['distance']}km" if st.get('distance', 0) > 0 else ""
                dur_tag  = f"  dur={st['duration_seconds']}s" if st.get('duration_seconds', 0) > 0 else ""
                print(f"      Set {i+1}{tags}: "
                      f"{st['weight']} {s['unit']} x {st['reps']}  "
                      f"e1RM={st['estimated_1rm']}"
                      f"{dist_tag}{dur_tag}{cmt}")
    elif agg == "monthly" and ex["monthly_aggregations"]:
        print(f"\n  MONTHLY")
        for m in ex["monthly_aggregations"]:
            dist_str = f"  dist={m['total_distance']}km" if m.get('total_distance', 0) > 0 else ""
            dur_str  = f"  dur={m['total_duration_seconds']}s" if m.get('total_duration_seconds', 0) > 0 else ""
            print(f"    {m['month']}  max={m['max_working_weight']} {ex['unit']}  "
                  f"e1RM={m['peak_estimated_1rm']}  vol={m['total_volume']}  "
                  f"sessions={m['session_count']}{dist_str}{dur_str}")
    elif agg == "weekly" and ex["weekly_aggregations"]:
        print(f"\n  WEEKLY")
        for w in ex["weekly_aggregations"]:
            dist_str = f"  dist={w['total_distance']}km" if w.get('total_distance', 0) > 0 else ""
            dur_str  = f"  dur={w['total_duration_seconds']}s" if w.get('total_duration_seconds', 0) > 0 else ""
            print(f"    {w['week']}  max={w['max_working_weight']} {ex['unit']}  "
                  f"e1RM={w['peak_estimated_1rm']}  vol={w['total_volume']}  "
                  f"sessions={w['session_count']}{dist_str}{dur_str}")


if __name__ == "__main__":
    args = sys.argv[1:]
    period   = args[0] if len(args) > 0 else "90"
    end_arg  = args[1] if len(args) > 1 and _looks_like_date(args[1]) else None
    offset   = 1 if end_arg else 0
    ex_args  = [a for a in args[1 + offset:] if not _looks_like_date(a)]
    period_val = None if period.lower() == "all" else int(period)
    print(f"\n{'='*72}")
    print(f"  Data Agent — Complete  |  period={period}")
    print(f"{'='*72}")
    data = collect(query_period_days=period_val, end_date_str=end_arg)
    _print_summary(data)
    if not ex_args:
        print(f"\n  (no exercises specified — pass names after period to see detail)")
        print(f'  e.g. python -m src.data_agent 90 "" "Lat Pulldown" "Flat Dumbbell Bench Press"')
    else:
        for ex_name in ex_args:
            _print_exercise(data, ex_name)

    # ── Size comparison ──────────────────────────────────────────────────
    try:
        raw_kb = len(json.dumps(data).encode()) / 1024
        pkg    = prepare_analysis_package(
                     query_period_days=period_val, end_date_str=end_arg)
        pkg_kb = len(json.dumps(pkg).encode()) / 1024
        reduction = 100 * (1 - pkg_kb / raw_kb) if raw_kb > 0 else 0
        print(f"\n{'─'*72}")
        print(f"  PACKAGE SIZE")
        print(f"  collect() with sets:          {raw_kb:>8.1f} KB")
        print(f"  prepare_analysis_package():   {pkg_kb:>8.1f} KB")
        print(f"  Reduction:                    {reduction:>7.0f}%")
        print(f"{'─'*72}")
    except Exception as e:
        print(f"\n  [size comparison failed: {e}]")
