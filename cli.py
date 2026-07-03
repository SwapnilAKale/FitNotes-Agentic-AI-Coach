#!/usr/bin/env python3
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import AgentSession
from src.coordinator import Coordinator

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
                    print(f"\n{result['answer']}\n")
                    # Fix 5: caller-driven commit. Every staged exercise was
                    # already approved through the gate above, so the CLI (never
                    # the agent) runs the atomic execute+verify for the batch.
                    if turn_state["staged_workout"]:
                        turn_state["staged_workout"] = False
                        outcome = json.loads(
                            await session.call_tool("execute_staged_workout", {}))
                        if outcome.get("success"):
                            session._staged_active = False
                            print(f"✅ {outcome.get('message', 'Workout saved and verified.')}\n")
                        else:
                            print(f"❌ {outcome.get('message') or outcome.get('error') or 'Workout write failed — nothing was saved.'}\n")
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
