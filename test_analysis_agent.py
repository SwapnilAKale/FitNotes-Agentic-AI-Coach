"""
Temporary test for coordinator.py — delete after verification.
Run: python test_analysis_agent.py
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from src.coordinator import Coordinator


async def test():
    # Test with no AgentSession (operational path will return stub message)
    coordinator = Coordinator(agent_session=None)

    tests = [
        "How has my Walking been lately?",
               "How is my Lat Pullddown progressing?"
    ]

    for question in tests:
        print(f"\nQ: {question}")
        result = await coordinator.route(question)
        print(f"   route    = {result['route']}")
        print(f"   flagged  = {len(result['flagged_claims'])} claim(s)")
        for f in result['flagged_claims']:
            print(f"  [{f['action']}] {f['original_claim'][:80]}")
            print(f"    → {f['reason']}")
        print(f"   answer:\n{result['answer']}")


if __name__ == "__main__":
    asyncio.run(test())
