CONTEXT
Make the workout-write flow structurally correct by moving the deterministic steps (execute + verify) off the LLM and onto the server. Root causes from the read-only diagnosis (cite-confirmed):
- (ii) LOAD-BEARING: execute is agent-driven — /confirm flips `allow_execute` and re-prompts the agent to call `execute_staged_workout` (server.py:624,640) instead of the server calling it deterministically (the way it already calls `discard_staged_writes` at server.py:547,633). Because execute shares the agent's tool path, the agent can call it mid-staging turn, which ends the turn early (run A: staged 1 of 7 then stopped).
- (i) LOAD-BEARING: `verify_workout_logged` is exposed to and encouraged for the staging agent, and it reads `training_log` not the slot (combined_server.py:1650-1672), so mid-stage it is STRUCTURALLY GUARANTEED to return `sets_found: 0`. The agent reads that false 0 as "write failed" and RE-STAGES; the slot appends (combined_server.py:1438); the exercise doubles.
- (iii)/(iv) cosmetic steering: `next_step` on every log_workout result (combined_server.py:1466) and the per-item prompt guidance (agent.py:94,111-116,139,160-164) bias the model toward a stage→execute→verify rhythm. Once execute+verify leave the agent's toolset these can't be acted on, but reword them so the prompt stops advertising a flow that no longer exists.

TARGET DESIGN (three phases, three owners):
1. STAGE (agent): the /chat turn — resolve names, emit N log_workout calls (each appends to the slot), then STOP. The agent's write-adjacent toolset is log_workout ONLY. It has NO execute tool and NO verify tool. It structurally cannot write, verify, or interleave.
2. EXECUTE+VERIFY (server, deterministic, ATOMIC): on /confirm(confirmed=true) the SERVER calls execute_staged_workout via session.call_tool (mirroring discard). Execute inserts the whole batch, reads back training_log for that date, compares written-count vs staged-count, and COMMITS only on match; on mismatch it ROLLS BACK (the dirty state never persists — no separate delete) and returns a loud failure. Verify lives INSIDE the execute transaction so no half-verified state is observable.
3. The agent is not in the execute/verify loop at all. The success/failure message comes from the server's execute+verify result.

This makes the locked invariant STRUCTURAL: the only code path to a DB write is /confirm → server → execute. No agent call, reload, or misfire can reach a write.

FILES IN SCOPE
- mcp_servers/combined_server.py — execute+verify atomicity; reword next_step.
- server.py — /confirm calls execute deterministically via session.call_tool; unexpose handling.
- src/agent.py — remove execute_staged_workout + verify_workout_logged from the operational toolset; reword the stage/execute/verify prompt block to stage-all-then-stop.
- tests/test_write_path_comments_cardio.py — new tests.
- README.md, lessons.md, personal_lessons.md.
Do NOT touch: the slot append/discard lifecycle (working — leave combined_server.py:1438 and the discard tool as-is); Fix 3 batch transaction shape except to fold verify into it; Fix 4 panel; the WAL path except where execute already appends; sibling staged flows (goal/set edits) — this stage is workout-write only; the upload-warning comparison (separate bug, next).

CHANGE 1 — combined_server.py: execute becomes atomic execute+verify
Re-read the current `_execute_staged_workout_sync` structure before editing (it opens conn, loops staged workouts→sets inserting, one commit, pops slot, WAL-appends per workout).

Restructure so read-back verify happens BEFORE commit, in the same transaction:
- Insert the whole batch (all staged workouts/sets/comments). Capture each inserted training_log rowid into an id-set (use lastrowid after each INSERT).
- Before commit: read back training_log WHERE _id IN (<id-set>) — do NOT match by date alone (would sweep pre-existing rows). Compute written-count = len(rows returned). staged-count = total sets across the batch.
- If written-count == staged-count → COMMIT, pop the slot, WAL-append per workout, return {"success": true, "sets_written": N, "exercises_written": M, "verified": true, "message": "..."}.
- If mismatch → ROLLBACK (conn.rollback(), no commit), retain the slot (batch stays inspectable, DB untouched), do NOT WAL-append, return {"success": false, "verified": false, "sets_expected": N, "sets_found": K, "message": "Write verification failed — batch rolled back, no changes made. <detail>"}.
- ONE connection/transaction: insert + read-back + commit/rollback are atomic. No separate delete, no second connection.
- Slot pop happens ONLY on the COMMIT path. On rollback the slot is RETAINED (batch stays staged so the user can retry confirm, and the DB is provably untouched).

CHANGE 2 — combined_server.py: reword next_step (~:1466)
The log_workout staged result currently returns "next_step": "Call execute_staged_workout to complete the write."
Change to: "next_step": "Staged. Continue staging remaining exercises with log_workout. Do not call execute — the server commits the full batch when the user confirms."

CHANGE 3 — server.py: /confirm drives execute deterministically
- On confirmed=true: instead of setting allow_execute and calling session.answer("Yes, confirmed, please execute") (:624,640), call `result = await session.call_tool("execute_staged_workout", {})` directly (mirror the discard call pattern at :547/:633). Parse the result and return it as the confirmation response to the frontend. The agent is NOT re-prompted.
- On confirmed=false: keep the existing discard-on-cancel (already correct, do not touch).
- Before removing the allow_execute re-prompt: check whether allow_execute / the EXECUTE_TOOLS gate / session.answer path is used by ANY sibling staged flow (goal/set edits). If yes: scope this change to workout-execute only and leave sibling paths untouched. If no: remove the allow_execute re-prompt block entirely. State what was found and what was removed or kept.

CHANGE 4 — src/agent.py: unexpose execute + verify; reword prompt
- Filter `execute_staged_workout` and `verify_workout_logged` out of the operational agent's tool list (the list_tools()-derived set at :337-348). The handlers in combined_server.py are KEPT (called server-side via session.call_tool); only the agent's visible schema changes. Unexpose, do not delete.
- Reword the prompt block at :94, :111-116, :139, :160-164: replace per-item "call execute after log_workout / verify after execute" with: "To log a workout: call log_workout once per exercise (all in the same turn) to stage the full day. Then STOP — do not call execute or verify. The server commits the batch automatically when the user confirms." Remove VERIFY-group instructions for the workout staging path. Leave sibling goal/set guidance untouched if those flows are unchanged.
- Verify the agent schema no longer contains execute_staged_workout or verify_workout_logged (confirm in the report).

OUT OF SCOPE — do NOT touch: slot append + discard lifecycle; Fix 4 panel; WAL design beyond existing per-workout append on commit; sibling goal/set staged flows (unless Change 3 confirms allow_execute is unused — state what was decided); upload-warning comparison; any read/analytical path. No general delete tool — rollback is the execute transaction undoing itself.

TESTS (tests/test_write_path_comments_cardio.py; temp SQLite / direct calls; no server, no Gemini)
1. execute+verify COMMIT path: stage a valid batch → call _execute_staged_workout_sync → assert verified:true, sets_written matches staged-count, rows present in DB by id-set, slot popped.
2. execute+verify ROLLBACK path: stage a batch, then monkeypatch the read-back so written-count < staged-count → assert success:false, verified:false, ZERO rows with the staged ids in DB (rolled back), slot RETAINED. Non-vacuity: a control clean batch commits in the same test, proving rollback is what suppressed the write.
3. tool-unexposure: build the operational agent's tool list and assert execute_staged_workout and verify_workout_logged are absent, while their handlers exist in combined_server's dispatch map.
4. atomicity: assert no committed-but-unverified state reachable (verify-before-commit guaranteed by id-set read-back inside transaction).
5. next_step reword: assert log_workout staged result no longer contains "Call execute_staged_workout".

Run pytest tests/ -v. State the baseline count before this change; expect baseline + new, all green.

State plainly: pytest proves the execute+verify+rollback machinery and tool-unexposure. It does NOT prove the agent now stages-all-then-stops and that /confirm→server-execute writes correctly live. That is the mandatory two-run live check (server restarted between runs), owed on quota reset:
- Run the 7-exercise day prompt; confirm the agent emits 7 log_workout calls, NO execute/verify calls in the /chat turn, then stops.
- Press Confirm → server executes → sqlite3 trace: exactly 7 exercises / 15 sets / 4 comments; Deadlift kg round-trip (100/110/120); Treadmill+Swimming unit=3 with distance+duration; Cycling unit=2 duration-only; each comment owner_id == its set _id.
- Both runs identical. This is the bar; do not mark complete on pytest alone.

DOCS
- README.md: targeted line — workout writes are committed server-side on explicit confirmation, with atomic write-and-verify (rolled back on any mismatch); the agent stages, the server commits.
- lessons.md: two entries:
  (a) Generic — a deterministic step (commit, verify) owned by an LLM is non-deterministic by construction; move it to the layer that can do it deterministically and remove the tool from the model's reach (remove the option, don't instruct). Verify-before-commit in one transaction beats write-then-check-then-delete: the dirty state never persists.
  (b) NEW — reversible-unknown testing rule: when a scenario's output is unknown AND reversible, live-testing it is worth doing even at risk of breaking, to learn the agent's real behavior under unpredictable/broken input; irreversible actions are excluded.
- personal_lessons.md: full detail — execute+verify folded into one transaction (combined_server.py refs), commit-on-match/rollback-loud-on-mismatch, slot-pop only on commit, slot retained on rollback, next_step reworded (:1466), /confirm now server-driven via session.call_tool (server.py refs) mirroring discard, execute+verify unexposed from operational toolset (agent.py:337-348) with prompt block reworded (agent.py:94/111-116/139/160-164), the diagnosis's (i)+(ii) load-bearing split, and the doubled-Deadlift run that motivated it. Pull all line refs from the actual code, not memory.

REPORT
- The restructured execute+verify: how written-count is compared to staged-count, how inserted rows are identified (id-set from lastrowid, not date), and the commit-vs-rollback branch (file:line).
- Slot-pop placement (commit-path only) and slot retained on rollback (confirm).
- /confirm change: confirm it now calls execute via session.call_tool directly; what was found re: allow_execute and whether sibling flows still need it; what was removed or kept.
- Tool unexposure: confirm execute + verify absent from the operational agent schema, handlers retained in dispatch.
- next_step: confirm new wording.
- pytest baseline count + new tests added + result.
- Anything found that contradicts the diagnosis (esp. any sibling flow that breaks if allow_execute path is removed — scope around it).
No other changes.
