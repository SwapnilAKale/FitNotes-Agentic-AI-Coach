# Muscle ontology

Reference **domain knowledge** — a hierarchical muscle taxonomy plus a
many-to-many exercise→muscle map. It is not user data and does not depend on any
particular person's training log.

Read only by [`src/ontology.py`](../src/ontology.py). Consumed by
`_compute_muscle_ontology_summary` in `src/data_agent/process.py`.

## Why it lives here

Not in `data/` — that directory is gitignored wholesale, and this must be
reviewable in `git diff`.

Not inside the `.fitnotes` file — that file is wiped and replaced on every
backup upload (the reason `src/wal.py` exists), which would destroy it.

## Why this exists at all

`exercise.category_id` in FitNotes is a single NOT NULL column: one category per
exercise. Multi-muscle reality cannot be expressed, and the result is already
wrong. In this user's own top-15 most-logged lifts:

- `Smith Machine Shrugs` (266 sets) is filed **Back** — it is upper traps.
- `Reverse Cable Curls` (296 sets) is filed **Forearms** — it is elbow flexion
  (brachioradialis / brachialis), not wrist work.

So "how much back work did I do" is wrong today, silently.

## Files

| File | Columns |
|---|---|
| `muscles.csv` | `id, name, parent_id, size_class` |
| `exercises.csv` | `id, canonical_name, equipment, movement_pattern` |
| `exercise_muscle.csv` | `exercise_id, muscle_id, role, source` |
| `aliases.csv` | `db_exercise_name, exercise_id` |

### `muscles.csv` — CLOSED. Never grows.

Every human has the same muscles and this tree already covers them, so **no
user, no upload and no model may ever add, remove or rename one**. The other
three files grow as new exercises are reconciled; this one does not.

Three independent mechanisms hold that:

1. `tests/test_ontology_frozen_muscles.py` pins every row. An accidental or
   automated edit fails the suite. Changing the taxonomy on purpose means
   updating that snapshot in the same commit — deliberately, and reviewed.
2. `ontology.WRITABLE_FILES` excludes it, and `ontology_reconcile._assert_writable`
   raises before any automated path can open it.
3. `ontology.resolve_muscle_names` **rejects** an unknown name rather than
   creating it, so a confident proposal cannot widen the ontology.

`parent_id` is **purely anatomical part-of** and nothing else. `size_class`
(`large` / `medium` / `small`, blank allowed) is an **attribute column, never a
tree tier** — putting size into the parent chain would make `parent_id` mean two
incompatible things ("is part of" and "is the same size as"), and every later
query would have to know which kind of parent it was walking.

Muscle **names must be globally unique**: the name is the citation match-key the
Analysis Agent quotes. That is why the store says `Triceps Long Head` and
`Biceps Long Head`, never a bare `Long Head`.

Depth is arbitrary — deepening the tree later needs no migration. Currently:
regions (Chest, Back, Shoulders, Arms, Legs, Core) → muscles → heads, with heads
only where both well-established and separately trainable (triceps heads, delt
heads, trap regions). Individual quad heads are deliberately **not** modelled:
they are not meaningfully trainable in isolation, so the node would add curation
cost and enable no decision.

### `exercise_muscle.csv`

`role` is `primary`, `secondary` or `limiting`. **Never blend them into one
weighted number** — a fractional weight would park rear delts, erectors and
forearms permanently at the bottom of every ranking as an artifact of the
constant. The package reports all three as separate columns.

### Choosing between `secondary` and `limiting`

The test is **which way the interference runs**:

| | direction | meaning |
|---|---|---|
| `secondary` | **both ways** | The lift trains it. Delt work before an incline press hurts the press, *and* pressing hurts later delt work. **Counts as training volume.** |
| `limiting` | **one way** | The lift depends on it. Grip work before shrugs ruins the shrugs, but shrugs leave the grip fine. The muscle holds the load without being trained by it. **Never counts as training volume, and never counts as coverage.** |

Grip on Smith Machine Shrugs is the case that forced this. Marking it
`secondary` credits 266 sets of shrugs toward grip training, which is false.
Deleting the edge loses a real scheduling fact — grip genuinely caps how much
you can shrug. `limiting` keeps the fact and drops the false volume.

A `limiting` muscle with no primary or secondary sets is still **untouched**: it
was held, not trained, and coverage says so.

Precedence when one exercise reaches a muscle by several paths:
**`primary` > `secondary` > `limiting`.** A muscle genuinely trained is not
demoted because another path merely leans on it. One `(exercise, muscle)` pair
carries exactly one role — the duplicate-pair rule makes anything else
unrepresentable.

Attach each edge at **the most specific node the evidence supports**. Rollup is
automatic: a muscle's reported count is its own edges plus everything beneath it,
and an exercise mapped to both a parent and its child is counted once. So
`Dumbbell Skull Crusher → Triceps Long Head` already gives `Triceps` the count —
do **not** add a redundant parent edge.

Secondaries are curated **sparingly**: only where a muscle is a *significant*
part of the lift, not everything that twitches. The justification is scheduling
("I deadlifted yesterday, is a leg day today wrong?") and gains reality (an
upper/lower split with no direct arm work still grows arms) — not volume
book-keeping.

`source` is **mandatory on every edge**. Sub-head attribution is genuinely
contested in the literature: EMG activation studies are noisy and do not cleanly
predict hypertrophy. The agent's certainty must never exceed the mapping's, so
every edge records where it came from. An unsourced edge is rejected by the
loader.

### `aliases.csv`

The bridge from the user's exact FitNotes exercise names to canonical ontology
exercises. Lookup is an **exact, case-insensitive dict get** — deliberately not
fuzzy. `src/shared/resolver.py` matches a user's *typed query* against the
exercise table, which is the opposite direction and non-deterministic; a
near-miss there would silently attribute hundreds of sets to the wrong muscle.

A logged exercise absent from this file is reported in the package's
`unmapped_exercises` and its sets are **visibly excluded** from muscle maths,
never silently dropped.

Exercises whose `movement_pattern` is `cardio` carry no muscle edges on purpose;
they appear as `unattributed_exercises`, which is a different thing from a
curation gap.

## New exercises after an upload

When a backup arrives with an exercise the graph has never seen, `/upload` and
`/upload/confirm` both run `src/ontology_reconcile.detect_new`. It is
deterministic and offline — no model, no network — so an upload can never fail
or hang on it.

| Evidence | What happens |
|---|---|
| Name normalises (case / spacing / punctuation) to one the store knows | **auto-alias**, logged in the upload response |
| Exact word-permutation of exactly one known name, plural-tolerant | **auto-alias**, logged |
| Two or more candidates, or none | **queued** in `pending_review.csv` |

Fuzzy similarity is deliberately **not** a tier. A real typo
(`Dumbell Skullcrusher`) goes to review, because "close enough" is how hundreds
of sets get attributed to the wrong muscle with nobody noticing. `Machine Shrug
Row` does not fold into `Machine Shrug` for the same reason — a different word
set is a different movement until proven otherwise.

A queued exercise's sets are **excluded from every muscle number** and appear in
the package as `pending_review_exercises`, which the agent must disclose.

```
python scripts/review_pending.py            # see the queue
python scripts/review_pending.py --propose  # web-searched draft for each
<edit pending_review.csv: set approved to y or n>
python scripts/review_pending.py --promote  # write the y rows into the graph
```

`--propose` is the only step that uses the network, and it writes nothing to the
store — it only fills in decision / muscles / evidence / sources so there is
something to judge. Anything it cannot settle comes back `unsure`, which is a
success: the item waits for a human instead of being guessed at. `--promote` is
the only step that writes, and it can never touch `muscles.csv`.

Because the graph is universal, each new exercise is reviewed **once, ever** —
every later user inherits the mapping. That is also why an unreviewed guess is
unacceptable: a bad edge would propagate to everyone.

## Editing

1. `python scripts/draft_ontology_edges.py --draft` regenerates
   `exercises.draft.csv` / `aliases.draft.csv` from the exercises actually
   logged. Review, then drop the `.draft`.
2. Author `exercise_muscle.csv` by hand — primaries first, then secondaries.
3. `python scripts/draft_ontology_edges.py` reports what is still uncovered.
4. `pytest tests/test_ontology_load.py` enforces store integrity.

A row whose first column starts with `#` is ignored, so review markers are safe
to leave in a draft.

A missing or broken store never raises: it degrades to an empty ontology, the
package's muscle section becomes `{}`, and every logged exercise lands in
`unmapped_exercises`. Reference data must never hard-stop an answer about the
user's own training.

## What the agent may say

- **Coverage is a fact.** "No sets touched the erectors in the last 90 days" is a
  structural statement about the data and may be stated flatly — *with the window
  named*.
- **Relative volume is numbers.** Report counts and gaps.
- **Never a verdict.** Not "lagging", "weak", "neglected", or "you should train
  X more". There is no judgment field in this store to cite, which is what
  actually enforces it.
