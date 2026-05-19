"""
Standalone memory test. Runs without MCP server.
Tests: add, list, format, Gemini conversation with memory injection, auto-extract, forget.
Uses a separate test memory file so real memory.json is not affected.
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point memory to a test file
import src.memory as memory_module
memory_module.MEMORY_PATH = memory_module.Path("data/memory_test.json")

from src.memory import (
    add_fact, get_all_facts, delete_fact,
    format_memory_for_prompt, load_memory, save_memory
)
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

MODEL = "gemini-2.5-flash-lite"

# Reset test memory file
save_memory({"facts": [], "last_updated": None, "summary": None})
print("✓ Reset test memory file\n")

# Test 1: Add facts
print("=== TEST 1: Adding facts ===")
r1 = add_fact("user_fact", "User is 22 years old", "user_stated", "high")
r2 = add_fact("preference", "User prefers answers in lbs not kg", "user_stated", "high")
r3 = add_fact("training_pattern", "User trains 4 days per week", "user_stated", "high")
r4 = add_fact("injury", "User has mild right shoulder discomfort", "user_stated", "medium")
r4_dup = add_fact("user_fact", "User is 22 years old", "user_stated", "high")
print(f"Add fact 1: {r1['status']}")
print(f"Add fact 2: {r2['status']}")
print(f"Add fact 3: {r3['status']}")
print(f"Add fact 4: {r4['status']}")
print(f"Add duplicate: {r4_dup['status']} (should be 'duplicate')")

# Test 2: List facts
print("\n=== TEST 2: List all facts ===")
facts = get_all_facts()
print(f"Total facts stored: {len(facts)}")
for f in facts:
    print(f"  [{f['id']}] [{f['category']}] {f['content']}")

# Test 3: Format for prompt
print("\n=== TEST 3: Memory prompt injection ===")
formatted = format_memory_for_prompt()
print(formatted)

# Test 4: Gemini conversation with memory injection
print("\n=== TEST 4: Gemini conversation with memory ===")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

system_with_memory = f"""You are a fitness coach assistant.
{formatted}

Answer questions using what you know about the user from memory."""

response = client.models.generate_content(
    model=MODEL,
    contents=[
        types.Content(role="user", parts=[types.Part(text=system_with_memory)]),
        types.Content(role="model", parts=[types.Part(text="Understood.")]),
        types.Content(role="user", parts=[types.Part(text="What do you know about me?")])
    ],
    config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=300,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)
print("Agent response:")
print(response.candidates[0].content.parts[0].text)

# Test 5: Forget a fact
print("\n=== TEST 5: Forget a fact ===")
fact_id = facts[0]["id"]
result = delete_fact(fact_id)
print(f"Delete fact {fact_id}: {result['status']}")
remaining = get_all_facts()
print(f"Facts remaining: {len(remaining)}")

# Test 6: Cap enforcement
print("\n=== TEST 6: Cap enforcement (add 28 more facts) ===")
for i in range(28):
    add_fact("user_fact", f"Test fact number {i+5}", "agent_inferred", "low")
final_memory = load_memory()
print(f"Active facts: {len(final_memory['facts'])}")
print(f"Summary exists: {bool(final_memory.get('summary'))}")
if final_memory.get('summary'):
    print(f"Summary (first 100 chars): {final_memory['summary'][:100]}")

# Cleanup
try:
    os.remove("data/memory_test.json")
    print("\n✓ Test memory file cleaned up")
except Exception:
    pass

print("\n=== ALL TESTS COMPLETE ===")
