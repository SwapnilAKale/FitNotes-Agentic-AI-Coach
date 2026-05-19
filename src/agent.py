import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import date as _date
from pathlib import Path

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.memory import add_fact, format_memory_for_prompt
from src.schema_prompt import build_user_context_prompt, load_user_context

MODEL = "gemini-2.5-flash-lite"

SERVERS_DIR = Path(__file__).parent.parent / "mcp_servers"

SYSTEM_PROMPT = f"""Today's date is {_date.today().strftime('%Y-%m-%d')}.

DATE RESOLUTION RULE: When the user mentions a date without a year (e.g. 'May 17', 'December 25'):
1. First assume the current year ({_date.today().year}).
2. If that date is in the future (hasn't happened yet this year), use the previous year ({_date.today().year - 1}).
3. If the resulting date exists in the database, proceed.
4. If the resulting date does NOT exist in the database for that exercise, try the other year.
5. If neither year has data, tell the user no records were found and ask them to clarify.

Example: Today is 2026-01-05. User says "December 25".
December 25 2026 is in the future → try December 25 2025 first.
If no data on 2025-12-25, try 2025-12-25 → if still nothing, ask user to clarify.

Always call get_exercise_sessions to verify the date exists before staging any write.

You are a personal fitness coach assistant with tools for querying workout history and fitness research.

REASONING: Write a brief "Thought:" before each tool call explaining why, and after results explaining what you learned. Stop calling tools when you have enough data.

TOOL RULES:
- resolve_exercise_name — for any specific exercise name the user mentions; NOT for muscle groups (use get_weekly_volume or query_workout_data directly for those).
- get_exercise_history / get_personal_record / get_weekly_volume / query_workout_data / run_read_only_sql — personal workout data.
- search_fitness_knowledge — training science and principles.
- read_exercise_comments — form, technique, equipment, drop sets (not just weight/reps). Summarise in ≤3 paragraphs: starting form → progression → current state.
- get_exercise_sessions — list session dates/stats to help identify a specific session for update/delete.
- remember_fact: when user tells you something personal, states a preference, mentions an injury, or clarifies a convention. Store it immediately.
- recall_memories: at the start of personal questions to check stored context.
- forget_fact: when user says something is no longer true.
- Call at least one tool before answering. Never fabricate data. Report weights exactly as returned.
- For complex multi-step questions, write a short PLAN before calling tools.

WRITE ACTIONS:
- Always resolve_exercise_name first.
- Before update_workout_set or delete_workout_set, call get_exercise_sessions to get the exact set details including weight AND reps for each set. Use the exact reps from the session data as old_reps — never guess.
- log_workout: ask for the date if not given — never assume today.
- When a staging tool returns staged: true, call the matching execute tool
  immediately in the same response turn. The CLI has already handled confirmation.
  Do not add any text asking the user if they want to proceed.
  Do not repeat what is about to be written. Just call execute.
- After every execute_, call the matching verify_ tool and report the result:
  verify_workout_logged / verify_goal_set / verify_set_updated / verify_set_deleted
- For deletes: verify_set_deleted returning verified: true = success (item gone = correct).
- If "cancelled": true is returned, acknowledge and stop — do not retry.
- Final answer after any write MUST state one of:
  "✅ [data] has been saved to your database."
  "❌ [data] was NOT saved — write was cancelled."
  "❌ [data] was NOT saved — an error occurred."

DATE DISAMBIGUATION:
When an update/delete tool returns needs_clarification: true (date missing), present:
"I need to identify the session. Choose an option:
1. Give me an approximate date — I'll show records within 7 days
2. Show me your last 10 sessions (newest first)
3. Give me a date range"
Then call get_exercise_sessions with mode="approximate" / "recent" / "range" accordingly.
Present date + max weight + total reps per session. Ask the user to confirm before proceeding.
For goals: list all found goals with target_date, weight, reps, start_date and ask which to use.\
"""


class AgentSession:
    def __init__(self, db_path: str, memory_only: bool = False):
        self.db_path = db_path
        self._memory_only = memory_only
        self._session: ClientSession | None = None
        self._client: genai.Client | None = None
        self._all_tools: list[dict] | None = None
        self._exit_stack = AsyncExitStack()
        self._initialized = False
        self._conversation_history: list = []  # list of exchanges; each exchange = list of message dicts
        self._context_messages: list[dict] = []  # pinned user-context pair, prepended to every call
        self.confirmation_handler: callable | None = None
        self._staged_active: bool = False
        self._effective_system_prompt: str = SYSTEM_PROMPT

    async def initialize(self) -> None:
        if self._memory_only:
            self._all_tools = self._build_memory_only_tools()
            tool_names = [t["function"]["name"] for t in self._all_tools]

            memory_context = format_memory_for_prompt()
            self._effective_system_prompt = SYSTEM_PROMPT
            if memory_context:
                self._effective_system_prompt = SYSTEM_PROMPT + "\n\n" + memory_context

            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self._initialized = True
            return

        abs_db_path = os.path.abspath(self.db_path)
        abs_chroma_path = os.path.abspath(
            os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
        )
        server_env = {
            **os.environ,
            "FITNOTES_DB_PATH": abs_db_path,
            "CHROMA_DB_PATH": abs_chroma_path,
            # Force UTF-8 I/O in subprocesses so non-ASCII print() calls
            # (e.g. the → arrow in src/rag.py) don't raise UnicodeEncodeError
            # when the parent terminal uses CP1252.
            "PYTHONIOENCODING": "utf-8",
        }

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVERS_DIR / "combined_server.py")],
            env=server_env,
        )
        streams = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(*streams)
        )
        await self._session.initialize()

        tools = (await self._session.list_tools()).tools

        self._all_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

        tool_names = [t["function"]["name"] for t in self._all_tools]

        user_context_path = os.path.join(os.path.dirname(abs_db_path), "user_context.json")
        ctx = load_user_context(user_context_path)
        ctx_text = build_user_context_prompt(ctx)
        if ctx_text:
            self._context_messages = [
                {"role": "user", "content": ctx_text},
                {"role": "assistant", "content": "Understood. I'll apply these data interpretation rules to all queries."},
            ]

        memory_context = format_memory_for_prompt()
        self._effective_system_prompt = SYSTEM_PROMPT
        if memory_context:
            self._effective_system_prompt = SYSTEM_PROMPT + "\n\n" + memory_context

        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._initialized = True

    # ------------------------------------------------------------------ #
    #  Schema conversion helpers                                           #
    # ------------------------------------------------------------------ #

    def _json_schema_to_gemini(self, schema: dict) -> types.Schema:
        """Recursively convert a JSON Schema dict to types.Schema."""
        type_map = {
            "string": types.Type.STRING,
            "number": types.Type.NUMBER,
            "integer": types.Type.INTEGER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
        }
        schema_type = type_map.get(schema.get("type", "string"), types.Type.STRING)
        kwargs: dict = {"type": schema_type}

        if "description" in schema:
            kwargs["description"] = schema["description"]
        if "enum" in schema:
            kwargs["enum"] = [str(v) for v in schema["enum"]]
        if schema_type == types.Type.OBJECT:
            props = schema.get("properties", {})
            if props:
                kwargs["properties"] = {
                    k: self._json_schema_to_gemini(v) for k, v in props.items()
                }
            if "required" in schema:
                kwargs["required"] = schema["required"]
        if schema_type == types.Type.ARRAY and "items" in schema:
            kwargs["items"] = self._json_schema_to_gemini(schema["items"])

        return types.Schema(**kwargs)

    def _build_gemini_tools(self, openai_tools: list[dict]) -> list[types.Tool]:
        """Convert a list of OpenAI-format tool dicts to a single Gemini types.Tool."""
        declarations = [
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=self._json_schema_to_gemini(t["function"]["parameters"])
                if t["function"].get("parameters")
                else None,
            )
            for t in openai_tools
        ]
        return [types.Tool(function_declarations=declarations)]

    # ------------------------------------------------------------------ #
    #  Message format conversion                                           #
    # ------------------------------------------------------------------ #

    def _convert_messages_to_contents(self, messages: list[dict]) -> list[types.Content]:
        """Convert OpenAI message dicts to Gemini Content objects.

        Rules:
        - system  → prepended to the text of the first user message
        - user    → Content(role="user", parts=[text])
        - assistant (text only)       → Content(role="model", parts=[text])
        - assistant (with tool_calls) → Content(role="model", parts=[text?, fn_calls…])
        - tool    → grouped with adjacent tool messages into one Content(role="user",
                    parts=[function_response…])
        """
        # Build lookup: tool_call_id → function_name (needed for function_response)
        tool_call_lookup: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    tool_call_lookup[tc["id"]] = tc["function"]["name"]

        system_content = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"), None
        )

        contents: list[types.Content] = []
        first_user_seen = False
        i = 0
        non_system = [m for m in messages if m.get("role") != "system"]

        while i < len(non_system):
            msg = non_system[i]
            role = msg.get("role")

            if role == "user":
                text = msg.get("content") or ""
                if not first_user_seen and system_content:
                    text = f"{system_content}\n\n{text}" if text else system_content
                    first_user_seen = True
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                ))
                i += 1

            elif role == "assistant":
                parts: list[types.Part] = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                for tc in msg.get("tool_calls") or []:
                    raw_args = tc["function"]["arguments"]
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    parts.append(types.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=fn_args,
                    ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                i += 1

            elif role == "tool":
                # Collect all consecutive tool messages into one user Content
                tool_parts: list[types.Part] = []
                while i < len(non_system) and non_system[i].get("role") == "tool":
                    m = non_system[i]
                    tc_id = m.get("tool_call_id", "")
                    fn_name = tool_call_lookup.get(tc_id, tc_id)
                    content_str = m.get("content", "")
                    try:
                        response_data = json.loads(content_str)
                    except (json.JSONDecodeError, TypeError):
                        response_data = {"result": content_str}
                    tool_parts.append(types.Part.from_function_response(
                        name=fn_name,
                        response=response_data,
                    ))
                    i += 1
                contents.append(types.Content(role="user", parts=tool_parts))

            else:
                i += 1

        return contents

    # ------------------------------------------------------------------ #
    #  Tool execution                                                      #
    # ------------------------------------------------------------------ #

    def _build_memory_only_tools(self) -> list[dict]:
        """Return OpenAI-format tool dicts for the 3 memory tools (no MCP needed)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember_fact",
                    "description": (
                        "Store a fact about the user for future sessions. Call when the user "
                        "tells you something personal, states a preference, mentions an injury, "
                        "reveals a training pattern, or clarifies a data convention not in user_context.json."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["user_fact", "preference", "training_pattern", "injury", "convention"],
                                "description": "category of the fact",
                            },
                            "content": {
                                "type": "string",
                                "description": "the fact as a clear concise statement",
                            },
                            "source": {
                                "type": "string",
                                "enum": ["user_stated", "agent_inferred"],
                                "description": "how the fact was learned",
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "confidence level",
                            },
                        },
                        "required": ["category", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_memories",
                    "description": (
                        "Retrieve all facts stored about the user from previous sessions. "
                        "Call at the start of personal questions to check if relevant context exists."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "forget_fact",
                    "description": (
                        "Delete a stored memory by its ID. Use when the user says something "
                        "is outdated or asks you to forget it. First call recall_memories to get the ID."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_id": {
                                "type": "string",
                                "description": "the 8-character ID of the fact to delete",
                            },
                        },
                        "required": ["fact_id"],
                    },
                },
            },
        ]

    async def _call_memory_tool(self, tool_name: str, arguments: dict) -> str:
        """Call src/memory.py functions directly without MCP."""
        from src.memory import add_fact as _add_fact, get_all_facts, delete_fact
        if tool_name == "remember_fact":
            result = _add_fact(
                category=arguments["category"],
                content=arguments["content"],
                source=arguments.get("source", "user_stated"),
                confidence=arguments.get("confidence", "high"),
            )
            return json.dumps(result)
        elif tool_name == "recall_memories":
            facts = get_all_facts()
            if not facts:
                return json.dumps({"message": "No memories stored yet.", "facts": []})
            return json.dumps({"count": len(facts), "facts": facts})
        elif tool_name == "forget_fact":
            result = delete_fact(arguments["fact_id"])
            return json.dumps(result)
        return json.dumps({"error": f"Unknown memory tool: {tool_name}"})

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self._memory_only:
            return await self._call_memory_tool(tool_name, arguments)
        result = await self._session.call_tool(tool_name, arguments)
        parts = [item.text for item in result.content if hasattr(item, "text")]
        return "\n".join(parts)

    def _save_exchange(self, messages: list[dict], new_exchange_start: int) -> None:
        self._conversation_history.append(messages[new_exchange_start:])
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    # ------------------------------------------------------------------ #
    #  Main answer loop                                                    #
    # ------------------------------------------------------------------ #

    async def answer(self, question: str) -> dict:
        history_messages: list[dict] = [
            msg for exchange in self._conversation_history for msg in exchange
        ]
        messages: list[dict] = [
            *self._context_messages,
            *history_messages,
            {"role": "user", "content": question},
        ]
        new_exchange_start = len(self._context_messages) + len(history_messages)
        max_iterations = 7
        tool_calls_made = 0

        _execute_tool_names = {"execute_staged_workout", "execute_staged_goal"}
        _base_openai = [t for t in self._all_tools if t["function"]["name"] not in _execute_tool_names]
        _base_gemini_tools = self._build_gemini_tools(_base_openai)
        _write_gemini_tools = self._build_gemini_tools(self._all_tools)

        WRITE_TOOLS = {
            "log_workout", "set_goal", "log_bodyweight",
            "execute_staged_workout", "execute_staged_goal",
            "update_goal", "execute_staged_goal_update",
            "delete_goal", "execute_staged_goal_delete",
            "update_workout_set", "execute_staged_set_update",
            "delete_workout_set", "execute_staged_set_delete",
        }

        for iteration in range(max_iterations):
            active_gemini_tools = _write_gemini_tools if self._staged_active else _base_gemini_tools

            # Force at least one tool call on the first iteration
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ) if iteration == 0 else None

            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=MODEL,
                    contents=self._convert_messages_to_contents(messages),
                    config=types.GenerateContentConfig(
                        system_instruction=self._effective_system_prompt,
                        tools=active_gemini_tools,
                        tool_config=tool_config,
                        temperature=0.3,
                        max_output_tokens=4000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Error] API call failed: {e}", flush=True)
                return {
                    "question": question,
                    "answer": "I encountered an error processing that request. The query may have been too complex. Try rephrasing.",
                    "tool_calls_made": tool_calls_made,
                    "error": str(e),
                }

            # Parse response
            candidate = response.candidates[0]
            raw_parts = candidate.content.parts if (candidate and candidate.content) else []
            text_parts = [p for p in raw_parts if getattr(p, "text", None)]
            fc_parts = [p for p in raw_parts if getattr(p, "function_call", None)]

            combined_text = "\n".join(p.text for p in text_parts) or None

            # Store assistant turn in OpenAI-format dict for history
            msg_dict: dict = {"role": "assistant", "content": combined_text}
            if fc_parts:
                msg_dict["tool_calls"] = [
                    {
                        "id": f"call_{iteration}_{i}",
                        "type": "function",
                        "function": {
                            "name": p.function_call.name,
                            "arguments": json.dumps(dict(p.function_call.args)),
                        },
                    }
                    for i, p in enumerate(fc_parts)
                ]
            messages.append(msg_dict)


            if not fc_parts:
                # No function calls — final answer
                final_answer = combined_text or ""
                if tool_calls_made >= 2:
                    final_answer = await self._reflect(question, final_answer)
                self._save_exchange(messages, new_exchange_start)
                return {
                    "question": question,
                    "answer": final_answer,
                    "tool_calls_made": tool_calls_made,
                    "error": None,
                }

            # Execute tool calls
            write_cancelled = False
            for tc_dict, fc_part in zip(msg_dict["tool_calls"], fc_parts):
                tool_name = fc_part.function_call.name
                arguments = dict(fc_part.function_call.args)
                tool_call_id = tc_dict["id"]

                print("[thinking...]", flush=True)

                if tool_name in WRITE_TOOLS and self.confirmation_handler:
                    approved = await self.confirmation_handler(tool_name, arguments)
                    if not approved:
                        result = json.dumps({
                            "cancelled": True,
                            "message": "Write action cancelled by user. No changes were made.",
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result,
                        })
                        write_cancelled = True
                        continue

                try:
                    result = await self.call_tool(tool_name, arguments)
                except Exception as exc:
                    result = f"Tool error: {exc}"

                if len(result) > 800:
                    result = result[:800] + "... [truncated]"

                try:
                    parsed = json.loads(result)
                    if tool_name in {"log_workout", "set_goal"} and "staged_key" in parsed:
                        self._staged_active = True
                    elif tool_name in _execute_tool_names:
                        self._staged_active = False
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

            tool_calls_made += len(fc_parts)

            if write_cancelled:
                messages.append({
                    "role": "user",
                    "content": "The write action was cancelled. Do not retry it.",
                })
                self._save_exchange(messages, new_exchange_start)
                return {
                    "question": question,
                    "answer": combined_text or "Write action cancelled. No changes were made.",
                    "tool_calls_made": tool_calls_made,
                    "error": None,
                }

        self._save_exchange(messages, new_exchange_start)
        return {
            "question": question,
            "answer": "Reached maximum tool calls without a final answer.",
            "tool_calls_made": tool_calls_made,
            "error": "max_iterations_reached",
        }

    # ------------------------------------------------------------------ #
    #  Reflection                                                          #
    # ------------------------------------------------------------------ #

    async def _reflect(self, question: str, answer: str) -> str:
        """Ask the model to briefly review its own answer. Falls back to original on error."""
        prompt = (
            "Review the answer below for these specific problems:\n"
            "1. Does it answer what was actually asked?\n"
            "2. Are weight values consistent with what the tools returned?\n"
            "3. Does it claim research facts not found in the retrieved documents?\n\n"
            "If there is a genuine problem, rewrite the answer to fix it.\n"
            "If the answer is correct, return it exactly as-is.\n\n"
            "CRITICAL: Return ONLY the answer text. No thoughts, no reasoning, no \"Thought:\" prefixes, "
            "no explanations of what you reviewed. Just the answer the user should see.\n\n"
            f"Answer to review:\n{answer}"
        )
        reflection_contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=MODEL,
                contents=reflection_contents,
                config=types.GenerateContentConfig(
                    system_instruction=f"Original question: {question}",
                    temperature=0.1,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_parts = response.candidates[0].content.parts if (response.candidates and response.candidates[0].content) else []
            text_parts = [p for p in raw_parts if getattr(p, "text", None)]
            return "\n".join(p.text for p in text_parts) or answer
        except Exception:
            return answer

    async def _auto_extract_memories(self) -> None:
        """Scan conversation and extract new facts worth storing."""
        if len(self._conversation_history) < 4:
            return

        conversation_text = ""
        for exchange in self._conversation_history[-8:]:
            for m in exchange:
                role = "User" if m.get("role") == "user" else "Agent"
                content = m.get("content") or ""
                if content and len(content) > 5:
                    conversation_text += f"{role}: {content[:300]}\n"

        if not conversation_text.strip():
            return

        prompt = (
            "Read this conversation and identify NEW facts worth remembering for future sessions.\n"
            "Only extract things the user explicitly stated or clearly implied.\n"
            "Do not extract things already in the workout database (weights, dates, exercise names).\n"
            "Do not extract things already obvious from context.\n\n"
            "Categories: user_fact, preference, training_pattern, injury, convention\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            "Return ONLY a JSON array. If nothing is worth storing, return [].\n"
            'Example: [{"category": "user_fact", "content": "User is 22 years old", "confidence": "high"}]'
        )

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_parts = response.candidates[0].content.parts if (response.candidates and response.candidates[0].content) else []
            text = "\n".join(p.text for p in raw_parts if getattr(p, "text", None)).strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            facts = json.loads(text.strip())
            for fact in facts:
                if fact.get("content"):
                    result = add_fact(
                        category=fact.get("category", "user_fact"),
                        content=fact["content"],
                        source="agent_inferred",
                        confidence=fact.get("confidence", "medium"),
                    )
                    if result.get("status") == "saved":
                        print(f"[Memory] Saved: {fact['content'][:80]}", flush=True)
        except Exception:
            pass  # Never block shutdown on memory failure

    async def close(self) -> None:
        try:
            await self._auto_extract_memories()
        except Exception:
            pass
        await self._exit_stack.aclose()
