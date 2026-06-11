# Comment-Resolution Pre-Pass — Token Table

**Purpose.** Define the recognized comment tokens, what each does, its scope, and
— most importantly — how to tell a *declaration* from *prose*. This is the design
the Data Agent's comment-resolution pre-pass implements. Grounded in an actual
scan of the 3,160 comments in the DB, not just `user_context` documentation.

## The core problem: declaration vs prose

A scan of real comments shows the danger is not vocabulary but context. For the
BAR effect, the intended declarations ("big black rod", "small black rod") sit
alongside prose that contains the same words:

- `"next time use black rod"` — a future reminder, **not** "I used it this set"
- `"because thumbless grip could no longer even hold the bar"` — narrative
- `"curling the rod"`, `"held the bar"`, `"grip closer to the middle of the rod"`

A substring/keyword match would read these as bar declarations and, under
carry-forward scope, silently corrupt the bar weight for every following set.
So matching discipline is per-effect, and **any token that is recognized as
affecting unit/bar/warmup but cannot be confidently classified is logged loud for
human review — never silently applied and never silently dropped.**

## Scope vocabulary

- **this-set** — affects only the set whose comment it is.
- **carry-forward** — sets a state that persists across sets/sessions until a
  later declaration changes it.
- **pattern-seed** — teaches a per-exercise predicate, then applies it to other
  sessions of that exercise.
- **within-session** — groups sets inside one session.

## The table

| Effect | Recognized tokens (from DB scan) | Scope | Matching rule | Applies to | Tier / status |
|--------|----------------------------------|-------|---------------|------------|---------------|
| **Unit override** | `Pounds` (×4), `kg`/`kgs` (×6) | carry-forward, anchored on curated | Curated `user_context` (kg-native list + date ranges) is the unit source of truth. A standalone unit token that **agrees** with the curated unit confirms it and carries forward (Deadlift "Kgs" on 12-26 → kg). A token that **disagrees** (a lone "Kg" in a run of lbs sets — the society-gym Flat DB Bench set) applies to **that set only** and is logged loud; it never carries forward. Never matched inside a numeric phrase ("50 pounds") or prose. Unit is high-stakes (drives the headline weight), so every disagreement is logged, never silently applied. Magnitude alone can't disambiguate the Deadlift switch (too light), which is why we anchor on curated, not raw numbers. | Deadlift (confirms the switch); any exercise (catches one-offs) | **Tier 1** |
| **Warmup seed** | `warm up` (×26), `warmup` (×8), `warmup for elbows` | pattern-seed | **At most ONE warmup per session**, always the first set (lowest set_id). The first set is the warmup if an explicit `warmup` comment is on it, **or** it is ≤ ~70% of the session's working-set weight **and** has ≥12 reps. The 70% is a *relative gap*, so it survives changing warmup weights and no-warmup days (close-together sets → no gap → nothing flagged). **Negative gate (semantic, not line-count):** a set carrying a genuine ROM/form hierarchy — multiple deliberate ROM terms with **no pain attribution** (e.g. "First 3 below the neck / Next 3 neck ups / Last 2 partials") — is never a warmup. But if any ROM term is attributed to pain ("last one partial since elbow pain", even on its own line), it is a pain note, NOT a form hierarchy: the warmup stands and pain sets `has_pain_flag`. Replaces the Layer-3 multi-flag behaviour. | the exercise the comment is on | **Tier 1 — clears G-WARMUP** |
| **Counterbalance** | `one support` (×12), `two support`/`2 support` (×3) | this-set | Match the support phrase (word or numeric); reduce that set's effective bar: one support −10 kg, two supports −20 kg. | Smith Machine exercises | **Tier 2 — M2** |
| **Bar selection** | `big black rod` (×8), `black barbell`/`small black barbell` (×6), `small black rod` (×4), `standard` | carry-forward | **Do NOT comment-derive.** Prose false-positives make carry-forward unsafe. Keep the curated date ranges in `user_context` as the source of truth. Use a recognized declaration only as a **cross-check**: if a clean bar declaration appears on a date *outside* its expected range, log loud for review. | barbell exercises | **Tier 3 — cross-check only, do not derive** |
| **Drop set** | `Nth set` (1st/2nd/3rd…), `Back to back` | within-session | already implemented (`_detect_drop_group`) | all | **Handled — no change** |
| **Technique variant** | `thumbless grip`, `wide grip`, `single hand`, … | this-set | already implemented (`_detect_technique_variants`) | all | **Handled — no change** |
| **Pain flag** | pain vocabulary + e.g. `kidney` | this-set | already implemented (`_is_pain_comment`) | all | **Handled — no change** |
| **ROM / form** | per-exercise hierarchies ("below the neck", "partial", "half", …); "numbers then ROM" (`2 3 partials`, `last 2 below the neck`) = only those reps | this-set / specific-reps | currently only keyword-counted; the per-exercise hierarchy + rep-number scoping is future work | per exercise | **Tier 4 — future** |
| **Rep quality** | `Last X assisted`, `All assisted`, `Supported`, `Negatives` | this-set / specific-reps | not yet computed | all | **Tier 4 — future** |

## Implementation order

1. **Tier 1 (now):** unit override + warmup seed. Clears the last comment-resolution
   golden xfail (G-WARMUP); G-HG is deleted (invalid premise). Both are low-risk;
   both run as a pre-pass over each exercise's sets *before* `_recover_typed_weight`
   (unit) and `_detect_warmup_flags` (warmup).
2. **Tier 2:** counterbalance support → reduces effective bar on flagged Smith
   sets (currently M2, volume-only and wrong).
3. **Tier 3:** bar cross-check — keep date ranges, only flag clean declarations
   that fall outside their expected range. No derivation.
4. **Tier 4:** ROM/form hierarchy and rep-quality — later, larger.

## Loud-logging rule (applies to every tier)

The pre-pass keeps a recognized vocabulary per effect. A comment that contains a
unit/bar/warmup/support-like token it cannot confidently classify as a
declaration is appended to an `unclassified_comment_tokens` log with the
exercise, date, and comment text. It is never silently applied and never silently
dropped. This is what turns "next time use black rod" from a silent corruption
into a visible review item.

## Resolved decisions

1. **Unit scope** — carry-forward anchored on curated: a unit token that agrees
   with the curated unit confirms + carries forward; one that disagrees is
   this-set + loud log; never a silent override. (Magnitude alone can't
   disambiguate the Deadlift switch — too light — so curated, not raw numbers, is
   the anchor.)
2. **Warmup count** — exactly one per session, always the first set.
3. **Warmup cutoff** — first set ≤ ~70% of the session's working-set weight AND
   ≥12 reps. The threshold is a *relative gap*, so it survives changing warmup
   weights and no-warmup days. Explicit `warmup` comment overrides.
4. **Form vs pain gate** — disqualify a warmup only on a genuine ROM/form
   hierarchy, judged by *semantics* (multiple deliberate ROM terms, no pain
   attribution), not by line count. Any pain attribution → pain note → warmup
   stands; pain stays in `has_pain_flag`.
5. **Hand Gripper / G-HG** — deleted. Hand Gripper is kg-native; the "Pounds"
   comments were a removed error. No unit exception, no golden case.
