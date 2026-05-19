#!/usr/bin/env python3
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from groq import RateLimitError

from src.agent import AgentSession

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
                return True
            elif response in {"no", "n", "cancel"}:
                print("❌ Cancelled. No changes made.")
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
                result = await session.answer(question)
                if result["error"] and result["error"] != "max_iterations_reached":
                    print(f"\n[Error] {result['error']}\n")
                else:
                    print(f"\n{result['answer']}\n")
            except RateLimitError as e:
                wait_msg = ""
                if "Please try again in" in str(e):
                    wait_msg = str(e).split("Please try again in")[1].split(".")[0].strip()
                print(f"\n[Rate limit reached. Reset in {wait_msg if wait_msg else 'some time'}. Type 'exit' to quit or wait and try again.]\n")
                continue
            except Exception as e:
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
