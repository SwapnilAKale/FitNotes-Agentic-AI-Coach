# Data Agent — Correctness Spec (Invariants + Golden Cases)

**Purpose.** The Data Agent is the one component that cannot be wrong. This spec
defines what "correct" *means* in checkable terms, so that every known silent
failure mode becomes a loud one and stays caught forever. It is written before
the fetch/process/validate refactor so the structure has something concrete to
satisfy.

Two kinds of guarantee live here:

- **Invariants** — deterministic post-conditions the package must satisfy on
  *any* input. The validator stage asserts these and fails loud. They catch the
  class of bug we hit (silent nulls, wrong units, dropped data) regardless of
  which exercise or period triggered it.
- **Golden cases** — specific known-correct outputs pinned against the real DB.
  They catch regressions in the actual numbers. Verified below against the
  2026-05-28 backup snapshot — **re-pin against the current DB before trusting
  them**, since comments are being edited.

---

## The three stages (target structure)

1. **Fetch** — touches the DB, returns typed raw rows. No interpretation. The
   only component that knows SQL exists.
2. **Process** — pure function: `(raw_rows, user_context) -> package`. No DB, no
   I/O. Everything testable by feeding synthetic rows.
3. **Validate** — independent invariant checks on the finished package. Does
   **not** recompute; asserts post-conditions. On failure: raise or flag, never
   emit a quietly-wrong package.

All three stay pure Python, zero LLM. Determinism is the whole point.

---

## Invariants

Each invariant lists the bug it would have caught, so the value is concrete.

### A. Units
- **A1.** Every exercise's `unit` is exactly `"kg"` or `"lbs"`. No other value, never null.
- **A2.** `unit == "kg"` only if the exercise is in `exercises_in_kg` (Deadlift only on/after 2025-12-26) **or** a comment-override applies to that set. Otherwise `"lbs"`.
- **A3.** No weight value appears anywhere in the package without a unit label travelling with it.
- **A4.** Within one exercise, all sessions share one unit, *except* Deadlift across its 2025-12-26 split and any set with a comment-override. A mixed-unit exercise that is neither of those is a validator failure. — *catches the Hand Gripper "Pounds" mislabel (M3): two lbs sets silently labeled kg.*

### B. Weights, PRs, progression
- **B1.** Plate weight is exactly `metric_weight * 2.2046 + offset`, rounded to 1 dp. Same input → same output, always.
- **B1a. (headline includes the bar)** The displayed/headline weight is `plate_weight + bar_weight`, with the bar in the exercise's own unit (kg for kg-native, lbs otherwise). Bar weight is included in **both** the headline weight **and** volume — never one without the other. A non-barbell exercise has bar = 0. — *decision: headline includes the bar (Deadlift PR shows 85 kg, not 65).* 
- **B2. (PR does not trust the app flag)** All-time and period PRs are computed from the full set history — **never** from `is_personal_record`. The PR is: highest headline weight, then the most reps at that weight, then the most recent date. The app's `is_personal_record` flag is "was-a-PR-at-the-time" and is not used to select the PR. — *catches flag dependence in `_compute_alltime_pr`. (The flag's last reader, the `_fetch_pr_history` fetch, has been removed — PR selection no longer fetches the flag at all.)*
- **B2a.** The PR carries the comment from its set. A PR without its comment (when one exists) is incomplete.
- **B3.** For any weight-based exercise: `pr.weight` (all-time) ≥ `pr_period.weight` ≥ every in-period session `max_working_weight`. A PR below a session max is a failure.
- **B4.** `pr.weight > 0` for any exercise classified weight-based; `pr is None` only for non-weight exercises.
- **B5.** Every `max_working_weight` equals an actual set's headline weight from that session — never a value not present in the session's sets.
- **B6.** No negative weight, reps, distance, or duration anywhere.
- **B7.** Cross-unit comparisons (all-time PR spanning Deadlift's lbs→kg switch) compare kg-normalized values, never raw display numbers. — *the `_to_kg` path; B3 must hold even across the unit switch.*
- **B8. (other PR-derived metrics also avoid the flag)** `is_pr_session`, `pr_velocity`, `pr_count` in aggregations, `pr_context`, and `learning_curve.sessions_to_first_pr` are all derived from `_pr_event_dates` (a deterministic running-max walk over set history) — not from `is_personal_record`. The refactor is complete: the flag is no longer fetched, stored, or read by any number-producing surface, so the earlier "known gap" is closed.

### C. Cardio (catches H1–H4)
- **C1.** If `is_cardio` and any in-period session has `distance > 0`, then `progression.distance_total_km` is non-null and `> 0`. — *catches H3: single-session distance showing null.*
- **C2.** If `is_cardio` and any session has `duration_seconds > 0`, then the duration progression fields are populated.
- **C3.** `all_time_sessions` is non-null whenever the exercise has ≥1 all-time session. — *catches H2: the `total_sessions_alltime` vs `total_alltime_sessions` key typo that makes this null for every cardio exercise.*
- **C4.** Cardio `sessions` list is non-empty whenever `total_sessions_period > 0`. — *catches H3: the weekly/monthly aggregation wiping `sessions` to `[]` for any >90-day cardio query.*
- **C5.** If the exercise has any comment in the period, the cardio structure carries it (a `comments` list and/or per-session `comment` field) and any pain-keyword comment sets a pain flag. — *catches H1: the `ex.clear()` rebuild destroying Treadmill's "kidney started paining."*
- **C6.** `pace` (once added) exists only on sessions with `distance > 0`. Cycling and Dead Hang (distance always 0) carry no pace field — no divide-by-zero, no fabricated pace. — *catches H4: pace must be pre-computed here, never derived by the Analysis Agent.*

### D. Comments & data integrity
- **D1.** A session's `comment_count` equals the number of its sets with a non-null comment. Comments come only from the `LEFT JOIN Comment` on `owner_id = training_log._id` — never approximate matching.
- **D2.** `has_pain_flag` is true iff at least one set comment matches the pain vocabulary.
- **D3.** Every comment token that affects unit / bar / warmup is either applied **or logged as unclassified**. An unrecognized bar/unit/warmup-like comment is never silently ignored. — *the comment-resolution pre-pass; this is the loud-failure rule for the token vocabulary.*
- **D4.** Every `full_comments` entry has a real date and text that matches a real row.

### E. Aggregation & temporal consistency
- **E1.** Every session date satisfies `query_start_date <= date <= query_end_date`. No row outside the window.
- **E2.** No session dated after the end anchor (no future data).
- **E3.** Weekly/monthly/yearly volume totals reconcile to the sum of their member session volumes within rounding. An aggregation that doesn't sum to its parts is a failure.
- **E4.** For the unfiltered package, `training_consistency.distinct_training_days` equals the count of distinct dates across all included exercises in the period.
- **E5.** ISO-week keys are year-boundary correct (the `_iso_week_key` rule) — a Dec-29 session in ISO week 1 maps to the next year, not W53.

### F. Statistical / thin-data
- **F1.** Every correlational output (`rest_performance_buckets`, `consecutive_day_effect`, `inter_exercise_correlation`, `dow_e1rm_pattern`, `bw_strength_correlation`) carries `n`, `ci_95`, `cohen_d`/`pearson`, `cis_overlap`, and `confidence_label`. No naked comparative claim without these.
- **F2.** `confidence_label` is one of `insufficient_data | weak | moderate | strong`.
- **F3.** `n` for a bucket equals the actual count of sessions in it.
- **F4.** CI is `None` when `n < 2`; Pearson CI is `None` when `n < 4`.

### G. Structural
- **G1.** Every exercise carries the required keys for its type (weight-based / cardio / duration-based). A weight-based exercise missing `progression`, or a cardio exercise missing the cardio block, is a failure.
- **G2.** No `None` where a value is required given that data exists for it.
- **G3.** The package serializes to JSON (no stray non-serializable objects).
- **G4.** Package size is reported; a package exceeding its per-scope soft ceiling is flagged. Ceilings: **BROAD ≤ 500 KB**, **GROUP ≤ 400 KB**, **FOCUSED ≤ 250 KB**. An unknown or missing scope uses the most restrictive (FOCUSED 250 KB). — *catches C2: the ~986 KB unfiltered package that 429s every broad question. With scope-aware trimming the BROAD 365d package is ≈ 400 KB.*
- **G5.** In a BROAD package, the following fields must be **absent** from every non-cardio exercise: `full_comments`, `inter_exercise_correlation`, `dow_e1rm_pattern`, `consecutive_day_effect`, `rest_performance_buckets`, `e1rm_history`, `pr_context`. A present key means `trim_package` leaked. — *soft flag; fires if scope profile was bypassed or trim logic regressed.*

---

## Golden cases

Verified against the **2026-05-28** backup. Re-pin against the current DB before
use. Each is `query(...)` → expected, with the invariant it also exercises.

| # | Query | Expected output | Also tests |
|---|-------|-----------------|------------|
| G-PR1 | Lat Pulldown all-time PR | **130.0 lbs × 9, 2026-05-18** (no bar; from full history, not the app flag), comment present (`"First 3 below the neck / Next 3 neck ups / Last 2 partials"`) | B1, B2, B2a, B5 |
| G-PR2 | Deadlift all-time PR, post-2025-12-26 | **85.0 kg × 5, 2026-04-20** (65 kg plates + 20 kg bar), labeled kg, comment present (`"Saw my back curl at the last rep..."`) | B1a, B2, B2a, B7 |
| G-PR3 | Lat Pulldown PR via flag vs full history | full-history PR (130×9) must **not** be overridden by the app flag, which also marks 115×12 and an old 60×15 | B2, B8 |
| G-MWE | Machine Wrist Extension top set display | plate recovered **+ 5 offset**, labeled kg, no bar (e.g. stored 9.072 → 25.0 kg) | A2, B1 |
| G-WALK | Walking all-time sessions | **78** | C3 |
| G-TREAD | Treadmill, 365-day window | session list non-empty; distance non-null for the 0.6 km session; **comment containing "kidney started paining" (2025-06-26) present** | C1, C4, C5 |
| G-CARDIO0 | Cycling / Dead Hang | duration progression present; **no pace field, no distance progression** (distance always 0) | C2, C6 |
| G-ALLTIME | All-time summary | first **2024-06-04**, last **2026-05-25**, **300** distinct training days | E1, E4 |
| G-BAR | Barbell Curl PR/volume on a date in each bar era | bar applied per the date range (22.05 / 27.56 / 33.07 lbs) and **included in both the headline weight and volume** | B1a |
| G-WARMUP | Flat Dumbbell Bench Press — Tier 1 warmup pre-pass | 2025-02-10 explicit-comment set (30×12 'Done first as a warmup') flagged; 2025-02-17 smooth 30/35/40 ramp NOT flagged (gap=5, not > 1.5×max_step=5) | D3 (Tier 1) |
| G-WARMUP-RAMP | Dumbbell Squats 2024-06-04 — smooth ramp negative case | sets 10/15/20 (equal +5 step): NO warmup flagged | D3 (negative) |

Notes:

- **G-BAR** and **G-WARMUP** encode design decisions, not just numbers: bar is in
  volume but not the headline weight (confirm that's intended), and a "warmup"
  comment seeds a per-exercise warmup profile rather than flagging only that one
  set.

---

---

## Scope profiles (C2 — package focusing)

`prepare_analysis_package` derives a **scope** from the query filters and applies
a trim profile before sending the package to the Analysis Agent. The scope is
stored as a top-level `"scope"` key so the validator and Analysis Agent can read it.

### Scope derivation

| Condition | Scope |
|-----------|-------|
| `exercise_names` provided and `len ≤ 3` | `"focused"` |
| `muscle_groups` provided (no short exercise list) | `"group"` |
| No filters | `"broad"` |

### One-aggregation-level rule (GROUP and BROAD)

Only one zoom level is retained per exercise; the other two are removed entirely.

| Window | Retained level |
|--------|---------------|
| < 180 days | `weekly_aggregations` |
| 180 – 730 days | `monthly_aggregations` |
| > 730 days or all-time | `yearly_aggregations` |

### Profiles

**FOCUSED** — unchanged from pre-scope behaviour. Full detail: all stat blocks,
all three aggregation levels, `full_comments` trimmed to 150, `sessions` kept.

**GROUP** — `full_comments` capped per exercise to *the 30 most recent entries
plus all pain-flagged entries* (deduped, chronological order). One aggregation
level. Deep-stat blocks (`inter_exercise_correlation`, etc.) retained.

**BROAD** — `full_comments` removed entirely. One aggregation level. The following
deep-stat blocks are removed from every non-cardio exercise:
`full_comments`, `inter_exercise_correlation`, `dow_e1rm_pattern`,
`consecutive_day_effect`, `rest_performance_buckets`, `e1rm_history`, `pr_context`.

Retained for every exercise in BROAD: `pr`, `pr_period`, `progression`,
`learning_curve`, `training_frequency`, `rep_range_distribution`,
`pain_analysis`, `comment_keyword_trends`, unit/bar metadata, and
the single aggregation level. Cardio blocks are untouched.
Top-level: `rankings`, `muscle_group_summary`, `exercise_lifecycle` retained.

> PR comment (`pr.comment`) is stored on the PR object at compute time and
> is **not** sourced from `full_comments` — it survives the BROAD drop intact.

---

## Failure policy

- A validator invariant failure **raises** (hard stop) for the integrity class
  (A, B, C, D, G) — a wrong number must never reach the Analysis Agent silently.
- The size/soft invariants (G4) and unclassified-comment logging (D3) **flag and
  log loud** rather than crash, so the pipeline degrades visibly instead of
  lying.
- Every golden case runs on every change to the process stage. A golden failure
  blocks the change.

The principle throughout: the Data Agent earns "never wrong" not by being clever
but by making every way it could be wrong **detectable** — an assertion that
fires or a test that goes red, never a quietly plausible number.
