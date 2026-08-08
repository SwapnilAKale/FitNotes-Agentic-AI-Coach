# Routing rules — what decides a lane

**Read this before adding a routing rule anywhere.**

There are four lanes (`analytical`, `operational`, `recall`, `out_of_scope`) plus
a per-chunk `decomposed` dispatch. Six things can decide which one a turn gets,
and they are strictly ordered. Historically that order existed only as statement
order inside `_node_entry_boundary` and had to be re-derived by every reader —
which is how the same rule ended up written in two languages that disagreed.

## The ladder

Each rung decides, or defers to the next. Nothing skips upward.

| # | Decider | Where | Beats everything below |
|---|---|---|---|
| 0 | **Pending-flow consumption** — a follow-up answer or a decomposition resume | `Coordinator.route`, `_consume_pending_*` | the turn never reaches the graph fresh |
| 1 | **Pre-seeded params** — a resume already carrying its verdict | `_node_entry_boundary` (first branch) | every guard below is skipped, deliberately |
| 2 | **`/log` boundary** — a typed `/log` prefix, or a live `/log` carry | `_node_entry_boundary` | trusted user signal → operational, **no classify call** |
| 3 | **Filler short-circuit** — bare greeting / thanks / empty | `_filler_reply` | canned reply, straight to END, no pipeline |
| 4 | **Write-intent regex** — see *Two strengths* below | `_write_intent_form` | does **not** decide alone; arms a hint and spends the classify call |
| 5 | **The classifier** | `_CLASSIFY_SYSTEM` → `_classify` | the normal decider for everything that reaches it |
| 6 | **Distrust override** | `_node_classify` | can overrule rung 5 — but only under the two conditions below |

Then the graph's conditional edges dispatch the resulting verdict
(`_route_after_entry`, `_route_after_classify` in `src/graph/coordinator_graph.py`).

## Two strengths of write signal

Rung 4 does not produce one verdict, it produces one of two:

- **`imperative`** — an explicit write verb (`log`, `record`, `save`, `add`,
  `delete`, `remove`, `update`, `change`, `correct`, or `set a goal`). The user
  named the action, so this **binds**: it overrules a successful classification.
- **`narration`** — `"I did … today/yesterday"` with no write verb. A **hint**
  only. It spends the classify call and stands in when that call *fails*, but a
  successful classification overrules it.

Question phrasing beats both (`_is_question`): if the message reads as a
question, no write signal fires at all and the classifier decides.

Why the split: `"I did chest and triceps today"` (a session to record) and
`"I did shrugs today and my grip gave out"` (a complaint to explain) are the
same shape. No regex separates them; the classifier can. So the classifier
decides, and the regex only guarantees a write is never lost to an *error*.

## When the override fires

```
regex fired?  ──no──→  classifier decides. Done.
     │yes
     ▼
classify FAILED?  ──yes──→  regex verdict wins  (any strength; an errored
     │no                     call is no evidence, and a write must not be
     ▼                       lost to one)
strength == imperative
  AND not decomposable?  ──yes──→  regex verdict wins (write safety)
     │no
     ▼
classifier decides.
```

"Decomposable" means `_is_mixed_lane_multi` — 2+ requests spanning 2+ lanes.
A genuine mixed turn (`"log bench 100x5 and how is my back going"`) always
splits; the override exists to stop a write leaking analytical, never to stop
decomposition.

## Invariants

1. **No analytical → operational edge exists.** An analytical failure retries or
   clean-fails; it never switches lanes. Post-strip, operational has no read
   tools and would fabricate an answer. Enforced by graph topology, not by a
   check — see the docstring in `coordinator_graph.py`.
2. **A `/log` boundary never spends a classify call.** It is a trusted signal.
3. **Uncertainty always resolves analytical** — one default, stated once in the
   prompt under *WHEN YOU ARE UNSURE*, covering every pair.
4. **One question test** (`_is_question`), one quantity test (`_QUANTITY_RE`),
   one mixed-lane test (`_is_mixed_lane_multi`). Each has exactly one definition
   and multiple callers. There were previously two, two, and three.

## Adding a rule

Before writing one, find which rung owns the decision:

- *Wrong lane for a phrasing the classifier should recognise* → rung 5, the
  prompt. Do **not** add a regex gate; that is what created the drift.
- *Something must be true regardless of what the model says* → rung 4 or 6, and
  say explicitly whether it binds over a successful classification.
- *The turn should not reach the pipeline at all* → rung 2 or 3.

Pure gates are pinned in `tests/test_routing_gates_golden.py`; override reach is
pinned in `tests/test_write_override_reach.py`. Both are input→output tables —
add the new case in both directions (fires / does not fire).

**Do not state a rule in the prompt that is also enforced in code.** If code
enforces it, the prompt should describe the *outcome*, not re-specify the test.
That duplication is what this document exists to prevent.
