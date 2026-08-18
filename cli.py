#!/usr/bin/env python3
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stdio_utf8 import force_utf8_stdio
force_utf8_stdio()

from src.agent import AgentSession
from src.coordinator import Coordinator, MSG_VERIFY_RESTATE, format_verify_fail_message

# Rate-limit errors re-raised by Coordinator — mirror _is_rate_limit from coordinator.py
try:
    from google.api_core.exceptions import ResourceExhausted as _GeminiResourceExhausted
except ImportError:
    class _GeminiResourceExhausted(Exception): pass  # type: ignore[misc]

try:
    from google.genai import errors as _genai_errors
    _GenaiClientError = _genai_errors.ClientError
except (ImportError, AttributeError):
    _GenaiClientError = None


def _is_rate_limit(exc: Exception) -> bool:
    """True if exc is a Gemini/google-api 429 / ResourceExhausted error."""
    if isinstance(exc, _GeminiResourceExhausted):
        return True
    if _GenaiClientError is not None and isinstance(exc, _GenaiClientError):
        return getattr(exc, "code", None) == 429
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
memory_only = "--memory-only" in sys.argv

BANNER = """
╔═══════════════════════════════════════╗
║         FitNotes Personal Coach       ║
║   Your AI-powered strength companion  ║
╚═══════════════════════════════════════╝
Type your question or 'exit' to quit.
"""


async def _finalize_staged_workout(session, coordinator, result, question) -> str:
    """
    Stage-2 verify + caller-driven commit for a gate-approved staged batch —
    the CLI's single execute seam, mirroring server.py's panel-branch verify
    and /confirm workout execute (same-seam rule: web and CLI verify
    identically). Verify FIRST: ONE LLM diff of the assembled /log-flow turns
    against the staged slot + its deterministic rendering. FAIL discards the
    slot immediately and asks the user to re-state (execute skipped; discard
    BEFORE record so the FAIL verdict survives the whole-dict clear). ERROR
    fails OPEN — the interactive gate already showed the write, and stage 3
    sees a not-verified signal. Returns the line to print.
    """
    # Slot FIRST (same seam as server.py's ghost-panel guard): the gate flag
    # was armed at tool-CALL time, so only the slot proves staging happened.
    # PROVEN empty (successful read, empty list) ⇒ the call was refused (unit
    # guard / clarification) — nothing to verify or execute; the agent's ask
    # was already printed, so return no extra line. Read failure is UNKNOWN ⇒
    # keep today's path (execute on an empty slot is a no-op error).
    slot_raw = ""
    try:
        slot_raw = await session.call_tool("read_staged_workout_slot", {})
        if json.loads(slot_raw).get("staged_workouts") == []:
            print("[cli] ghost confirm suppressed — log_workout was called "
                  "but staged nothing", file=sys.stderr)
            return ""
    except Exception as exc:
        print(f"[cli] staged-slot read failed: {exc}", file=sys.stderr)
    verdict = {"verdict": "ERROR", "reason": "verify unavailable"}
    try:
        if not slot_raw:
            # Slot read failed above — non-verifiable, never diff a good
            # batch against an empty string and FAIL it.
            raise RuntimeError("slot read failed — verify skipped")
        fmt = json.loads(await session.call_tool(
            "format_staged_workout_for_confirmation", {}))
        # verify_log_staging skips the LLM (verdict ERROR) when preview is
        # missing — a name-blind diff could spuriously FAIL a good batch.
        # No [question] fallback: a missing flow thread must skip-verify
        # (ERROR, fail-open), never diff the slot against the bare reply
        # text and FAIL a good batch. Same seam as server.py's panel verify.
        verdict = await coordinator.verify_log_staging(
            result.get("log_flow_turns"),
            slot_raw, fmt.get("preview"))
    except Exception as exc:
        print(f"[cli] staging verify errored: {exc}", file=sys.stderr)
        verdict = {"verdict": "ERROR", "reason": str(exc)}
    if verdict.get("verdict") == "FAIL":
        try:
            await session.call_tool("discard_staged_writes", {})
            await session.call_tool("record_workout_verify", verdict)
        except Exception as exc:
            print(f"[cli] staging verify FAIL cleanup failed: {exc}", file=sys.stderr)
        print(f"[cli] staging verify FAIL: {verdict.get('reason')}", file=sys.stderr)
        preview = fmt.get("preview")
        return format_verify_fail_message(preview) if preview else MSG_VERIFY_RESTATE
    try:
        await session.call_tool("record_workout_verify", verdict)
    except Exception as exc:
        print(f"[cli] record_workout_verify failed: {exc}", file=sys.stderr)
    outcome = json.loads(await session.call_tool("execute_staged_workout", {}))
    if outcome.get("success"):
        session._staged_active = False
        # The execute happened outside the agent's turn, so nothing else updates
        # its history — without this it still believes the batch is pending.
        session.note_host_write(outcome.get("message", "Workout saved and verified."))
        return f"✅ {outcome.get('message', 'Workout saved and verified.')}"
    return f"❌ {outcome.get('message') or outcome.get('error') or 'Workout write failed — nothing was saved.'}"


async def _confirm_restored_workout(session, coordinator, result) -> str:
    """
    CLI arm of the write-path restore (same seam as the server's panel
    arming): the coordinator restored a checkpointed staged batch and
    signalled it via result["restore_staged"]. The CLI has no panel, so the
    restored batch passes its interactive gate: print the slot-rendered
    preview, ask Confirm? (yes/no), yes → execute, no → discard + clear the
    checkpoint.

    Three-way verify handling mirrors the server: a checkpointed PASS verdict
    skips the LLM (already verified — case 1); no verdict (case 2) runs the
    stage-2 verify NOW on the restored slot with the restored flow turns as
    Input A — FAIL discards + clears + re-states WITHOUT prompting; ERROR
    falls THROUGH to the yes/no gate exactly like PASS (fail-open means
    "don't block", never "skip the gate" — execute can only fire from the
    yes branch). Returns the line to print.
    """
    from src import checkpoint as _ckpt

    slot_raw = ""
    preview = None
    try:
        slot_raw = await session.call_tool("read_staged_workout_slot", {})
        fmt = json.loads(await session.call_tool(
            "format_staged_workout_for_confirmation", {}))
        preview = fmt.get("preview")
    except Exception as exc:
        print(f"[cli] restored-slot preview failed: {exc}", file=sys.stderr)

    restored = result.get("restored_verify")
    if isinstance(restored, dict) and restored.get("verdict") == "PASS":
        verdict = restored              # case 1: verified pre-interruption
    else:
        # Case 2: the 429 hit between staging and verify — verify NOW, on the
        # restored slot, with the restored flow turns as Input A.
        try:
            verdict = await coordinator.verify_log_staging(
                result.get("log_flow_turns") or [], slot_raw, preview)
        except Exception as exc:
            print(f"[cli] staging verify errored: {exc}", file=sys.stderr)
            verdict = {"verdict": "ERROR", "reason": str(exc)}
        if verdict.get("verdict") == "FAIL":
            try:
                await session.call_tool("discard_staged_writes", {})
                await session.call_tool("record_workout_verify", verdict)
            except Exception as exc:
                print(f"[cli] restore verify FAIL cleanup failed: {exc}",
                      file=sys.stderr)
            _ckpt.clear_staged_checkpoint()
            print(f"[cli] restore verify FAIL: {verdict.get('reason')}",
                  file=sys.stderr)
            return format_verify_fail_message(preview) if preview else MSG_VERIFY_RESTATE
        try:
            await session.call_tool("record_workout_verify", verdict)
        except Exception as exc:
            print(f"[cli] record_workout_verify failed: {exc}", file=sys.stderr)

    # PASS or ERROR (fail-open): the human gate decides — never auto-execute.
    print(f"\n{'='*60}")
    print("⚠️  RESTORED STAGED WORKOUT — awaiting your confirmation")
    print(f"{'='*60}")
    print(preview or "(preview unavailable — the staged batch is shown above)")
    print()
    while True:
        reply = input("Confirm? (yes/no): ").strip().lower()
        if reply in {"yes", "y"}:
            outcome = json.loads(
                await session.call_tool("execute_staged_workout", {}))
            if outcome.get("success"):
                session._staged_active = False
                _ckpt.clear_staged_checkpoint()
                # See _confirm_staged_workout: the agent is not otherwise told.
                session.note_host_write(
                    outcome.get("message", "Workout saved and verified."))
                return f"✅ {outcome.get('message', 'Workout saved and verified.')}"
            return f"❌ {outcome.get('message') or outcome.get('error') or 'Workout write failed — nothing was saved.'}"
        if reply in {"no", "n", "cancel"}:
            try:
                await session.call_tool("discard_staged_writes", {})
            except Exception as exc:
                print(f"[cli] discard on cancel failed: {exc}", file=sys.stderr)
            _ckpt.clear_staged_checkpoint()
            session._staged_active = False
            return "❌ Cancelled. Nothing was saved."
        print("Please type 'yes' or 'no'.")


async def main() -> None:
    session = AgentSession(DB_PATH, memory_only=memory_only)

    # Fix 5: the agent only STAGES workouts — it has no execute tool. When the
    # gate below approves a log_workout staging, this flag arms the post-turn
    # code to drive execute_staged_workout via session.call_tool (caller-driven,
    # same pattern as the web server's /confirm — never the agent).
    turn_state = {"staged_workout": False}

    async def confirmation_handler(tool_name: str, arguments: dict) -> bool:
        """Called before any write tool executes. Returns True to proceed, False to cancel."""
        print(f"\n{'='*60}")
        print(f"⚠️  WRITE ACTION REQUESTED: {tool_name}")
        print(f"{'='*60}")

        if tool_name in {
            "execute_staged_workout", "execute_staged_goal",
            "execute_staged_goal_update", "execute_staged_goal_delete",
            "execute_staged_set_update", "execute_staged_set_delete",
        }:
            print("The agent wants to EXECUTE the staged write to your database.")
            print("This will permanently modify your FitNotes data.")
        else:
            args_display = json.dumps(arguments, indent=2)
            print(args_display)
            for key in ["date", "target_date", "current_target_date"]:
                if key in arguments and arguments[key]:
                    print(f"\n📅 Date: {arguments[key]}")
                    break

        print()
        while True:
            response = input("Confirm? (yes/no): ").strip().lower()
            if response in {"yes", "y"}:
                print("✅ Confirmed. Proceeding with write.")
                if tool_name == "log_workout":
                    turn_state["staged_workout"] = True
                return True
            elif response in {"no", "n", "cancel"}:
                print("❌ Cancelled. No changes made.")
                if tool_name == "log_workout" and turn_state["staged_workout"]:
                    # A partial batch may already be staged this turn — drop it so
                    # it can't linger into a later execute.
                    await session.call_tool("discard_staged_writes", {})
                    turn_state["staged_workout"] = False
                return False
            else:
                print("Please type 'yes' or 'no'.")

    session.confirmation_handler = confirmation_handler

    try:
        await session.initialize()
        print(BANNER)

        if not memory_only and not os.path.exists(DB_PATH):
            print(f"WARNING: Database not found at {DB_PATH}")
            print("Drop your FitNotes_Backup.fitnotes file into the data/ folder first.")
            print()

        coordinator = Coordinator(session) if not memory_only else None

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break
            try:
                if memory_only:
                    result = await session.answer(question)
                    if result["error"] and result["error"] != "max_iterations_reached":
                        print(f"\n[Error] {result['error']}\n")
                    else:
                        print(f"\n{result['answer']}\n")
                else:
                    result = await coordinator.route(question)
                    if result.get("route"):
                        print(f"\x1b[2m[route: {result['route']}]\x1b[0m")
                    # #19: a decomposed turn about to finalize a staged workout
                    # must print the WRITE-EXCLUDED merge — the write chunk's
                    # part is staging-time text that the "✅ logged" line below
                    # supersedes. Same seam as the web panel stash. Non-decomposed
                    # and non-staged turns print result['answer'] as before.
                    display_answer = result["answer"]
                    if turn_state["staged_workout"] and result.get("decomposed"):
                        display_answer = result.get("decomposed_nonwrite_answer") or ""
                    if display_answer:
                        print(f"\n{display_answer}\n")
                    # Fix 5: caller-driven commit. Every staged exercise was
                    # already approved through the gate above, so the CLI (never
                    # the agent) runs the stage-2 verify then the atomic
                    # execute+verify for the batch (see _finalize_staged_workout).
                    if turn_state["staged_workout"]:
                        turn_state["staged_workout"] = False
                        line = await _finalize_staged_workout(
                            session, coordinator, result, question)
                        # "" = ghost suppressed (nothing was staged) — the
                        # agent's ask above is the turn's whole output.
                        if line:
                            print(f"{line}\n")
                    elif result.get("restore_staged"):
                        # Resume of a quota-interrupted /log turn: the
                        # coordinator restored the checkpointed batch — pass
                        # it through the interactive gate (never the agent).
                        line = await _confirm_restored_workout(
                            session, coordinator, result)
                        print(f"{line}\n")
            except Exception as e:
                error_str = str(e)
                if turn_state["staged_workout"]:
                    # The turn died between staging and execute — drop the batch
                    # so it can't be committed by a later, unrelated confirm.
                    turn_state["staged_workout"] = False
                    try:
                        await session.call_tool("discard_staged_writes", {})
                    except Exception:
                        pass
                if _is_rate_limit(e):
                    wait_msg = ""
                    if "Please try again in" in error_str:
                        wait_msg = error_str.split("Please try again in")[1].split(".")[0].strip()
                    # Checkpoint status message (QuotaInterrupted) — typing
                    # 'continue' resumes the saved question via the Coordinator.
                    cp_msg = getattr(e, "user_message", None)
                    if cp_msg:
                        print(f"\n[{cp_msg}]")
                    print(f"\n[Rate limit reached. Reset in {wait_msg if wait_msg else 'some time'}. Type 'exit' to quit or wait and try again.]\n")
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    print("\n[Gemini is under high demand right now. Wait a few minutes and try again.]\n")
                else:
                    print(f"\n[Error: {e}]\n")
                continue
            except asyncio.CancelledError:
                print("\nRequest cancelled. Goodbye.")
                break
    finally:
        await session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
