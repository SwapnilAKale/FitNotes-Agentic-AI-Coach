# System-wide audit — brief for a fresh session

**Status:** requested 2026-09-11, NOT started. Deliberately deferred: running it
inside the write-safety live check would have blocked that check, and the
session was long enough that its context was degrading.

**How to start it:** open a NEW session and hand it this file. It needs a clean
context window — that is the whole reason it was deferred rather than done.

## Why this exists

The write-safety arc (2026-08-29 → 2026-09-11) was supposed to be a live check
of one feature. It turned into nine phases that found, in order: writes with no
confirmation panel, a refusal the agent reported as a success, a verifier that
could only ever say yes, a confirmed delete that never executed, a phantom row
that sat in real training data for eleven days, and a body-weight tool with no
update or delete at all.

Every one of those was found by **exercising the system**, not by reading it.
None were found by tests, because the tests asserted what the code did rather
than what the system owed the user. The pattern is consistent enough to be worth
acting on: **the gaps are in the places nobody has walked through end to end.**

So the ask is not "review the code". It is: **go through every single detail of
the system and check it does what it claims**, with the same standard the live
check used — verify against the database and the logs, never against the reply
text.

## The standard to hold

From this arc, the rules that actually caught things:

1. **Check both sinks.** A write is verified when the stored rows say so AND the
   text shown to the user says so. Tool-call log lines prove an attempt, not an
   outcome.
2. **Verify the invariant, not the instance.** Every bug here was one member of
   a class. "Which tools are executes" was written out by hand in four places
   and no two agreed. Sweep the class.
3. **Absence is not evidence.** `verify_set_deleted` returned "confirmed
   deleted" for a row that never existed. Any check that can only return success
   is not a check.
4. **A structural fact beats prose.** Gates should read what the code did, not
   what the model said about it.
5. **Ask what the failure costs the user.** "Nothing was saved" when something
   was invites a duplicate. "It's deleted" when it isn't means they stop
   looking. The direction matters more than the frequency.

## Known, already recorded — do not re-derive

The full list lives in the agent memory file `future-work-register.md`. In
summary, open at the time of writing:

- **Bodyweight has no CRUD** — log only, no update or delete. Insert-only, so
  logging body fat creates a *second* row for the same date. `body_fat = 0.0` is
  a sentinel for "not measured" that reads as a real 0%. Its confirm panel shows
  a raw JSON blob because it has no staged slot for the renderer to read. One
  fix (give it the staged/execute shape) closes all four.
- **A staged write is journalled twice** when the turn re-runs across the
  confirm boundary. Currently defended at replay, not prevented at source. This
  is the shape that produced phantom row 15771.
- **Four tools write with no confirmation and no claim-gate coverage** —
  the three `*_exercise_quirk` tools (`data/user_context.json`) and
  `delete_user_article` (Chroma). The last irreversibly destroys a user-uploaded
  article with no prompt.
- **Four data-coupled test failures** that break on every database upload —
  cardio ×2 (assume Cycling has no distance), the analytical package size budget
  (531 KB against a hardcoded 500 KB, and growing), and 8 logged exercises with
  no ontology alias (muscle-based analysis silently omits them).

## Where to look first

Ranked by where this arc found things, not by size:

1. **The analytical lane.** The entire write path has now been walked end to
   end; the read/analysis path has not been walked at all in this arc. It is
   larger, and `analysis_agent.py` (~657 lines of prompt) is the one piece the
   routing cleanup never reached — Stage 3 is still open.
2. **Anything with "verify" in its name.** One of the four verifiers was
   structurally unable to fail. The others were checked and are sound, but the
   pattern deserves a sweep wherever a check can return success by default.
3. **Every tool the agent can call, against what it claims to do.** The audit
   that found the ungated quirk/article tools took one command and surfaced four
   real gaps. Do it exhaustively: list `list_tools()`, and for each, confirm what
   it writes, whether it is gated, whether it is tracked by the claim gate, and
   whether it has the CRUD the user expects.
4. **The upload/replay path.** Phases 8 and 9 of the live check (WAL toggle,
   the id-collision scenario, restore) were never run — see the runbook at
   `~/.claude/plans/live-check-runbook.md`.
5. **Frontend ↔ backend agreement.** The confirm panel was showing raw JSON for
   most write types for the entire life of the feature, and nobody noticed until
   a screenshot. Assume other surfaces have the same gap between what the system
   knows and what it shows.

## Ground rules for whoever runs this

- **The user drives anything live.** Claude runs the server (`python server.py
  --debug`, never uvicorn) and watches the logs; the user sends the prompts.
- **Report before cleaning.** If a check leaves data behind, report it and wait
  for a go-ahead before touching the database. Back up first, delete by exact
  id, show before/after.
- **Test writes must be purged from the WAL too**, not just the database — a
  test row that stays in the journal replays into real data on the next upload.
  That is precisely how row 15771 survived.
- **Report faithfully.** If something is unverified, say unverified. Several
  findings in this arc came from a wrong first diagnosis being corrected, and
  the corrections were more valuable than the original claims.
