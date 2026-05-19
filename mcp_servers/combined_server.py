import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.environ.get("FITNOTES_DB_PATH", "./data/FitNotes_Backup.fitnotes")
ALLOWED_TABLES = {"training_log", "exercise", "Category", "Comment"}

server = Server("fitnotes-coach")

_kb = None
_staged_writes: dict = {}


def _get_kb():
    global _kb
    if _kb is None:
        chroma_path = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
        from src.rag import FitnessKnowledgeBase
        _kb = FitnessKnowledgeBase(chroma_path=chroma_path)
    return _kb


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_workout_data",
            description=(
                "Answer a natural-language question about the user's personal workout history "
                "using the FitNotes database. Use for questions about PRs, exercise history, volume, "
                "workout dates, sets, reps, progress over time. Returns SQL used, rows, and a natural-language answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "the natural-language question about workout data",
                    }
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="get_personal_record",
            description=(
                "Get the personal record (PR) for a specific exercise. Returns the heaviest "
                "weight ever lifted, reps, and date. Handles unit conversion and bar weight automatically."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name as stored in FitNotes",
                    }
                },
                "required": ["exercise_name"],
            },
        ),
        types.Tool(
            name="get_exercise_history",
            description=(
                "Get recent training history for a specific exercise. "
                "Shows sets, weights, reps, and dates for the last N days."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exercise name",
                    },
                    "days": {
                        "description": "number of days to look back (default 30)",
                    },
                },
                "required": ["exercise_name"],
            },
        ),
        types.Tool(
            name="get_weekly_volume",
            description=(
                "Get total training volume (sets and load) broken down by muscle group "
                "for the last N days. Useful for assessing training balance and overtraining risk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "description": "number of days to look back (default 30)",
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="run_read_only_sql",
            description=(
                "Run a custom read-only SQL query against the FitNotes database. "
                "Use only for questions that the other tools cannot answer. "
                "RESTRICTIONS: SELECT only. Max 100 rows. "
                "Allowed tables: training_log, exercise, Category, Comment. Timeout: 5 seconds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "a SELECT query only",
                    }
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="search_fitness_knowledge",
            description=(
                "Search the fitness knowledge base for research-backed information about "
                "training principles, hypertrophy, recovery, nutrition, and exercise science. "
                "Uses hybrid retrieval (BM25 + semantic) and reranking for accuracy. "
                "Returns titles, sources, and relevant text from PubMed abstracts and Wikipedia."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the fitness science question or topic to search for",
                    },
                    "n_results": {
                        "description": "number of results to return, max 10 (default 5)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="resolve_exercise_name",
            description=(
                "Resolves a colloquial or partial exercise name to the exact name stored "
                "in the FitNotes database. Use this BEFORE calling any other database tool when the "
                "user's exercise name might not match exactly. Examples: 'hammer curl' → 'Dumbbell Hammer Curl', "
                "'skull crusher' → 'dumbbell skull crusher', 'seated machine curl' → 'Seated Machine Curl (Kg)'. "
                "If multiple candidates are returned, present them to the user and ask for clarification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_term": {
                        "type": "string",
                        "description": "the colloquial or partial exercise name the user provided",
                    }
                },
                "required": ["user_term"],
            },
        ),
        types.Tool(
            name="read_exercise_comments",
            description=(
                "Read the user's personal notes and comments for a specific exercise. "
                "Comments contain per-set form quality, ROM notes, drop set markers, equipment changes, "
                "unit declarations, and training observations. Use this to understand form progression, "
                "identify drop sets, check equipment used on a specific date, or answer questions about "
                "training quality rather than just weight/reps."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name (use resolve_exercise_name first if unsure)",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "start date YYYY-MM-DD (optional)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "end date YYYY-MM-DD (optional)",
                    },
                    "limit": {
                        "description": "max comments to return, default 15, max 15 (optional)",
                    },
                },
                "required": ["exercise_name"],
            },
        ),
        types.Tool(
            name="log_workout",
            description=(
                "Log one or more sets for an exercise to the training log. "
                "IMPORTANT: This tool stages the write and requires user confirmation before executing. "
                "Always call resolve_exercise_name first to verify the exact exercise name. "
                "Use today's date unless the user specifies otherwise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name from resolve_exercise_name",
                    },
                    "date": {
                        "type": "string",
                        "description": "date in YYYY-MM-DD format (default: today)",
                    },
                    "sets": {
                        "type": "array",
                        "description": "list of sets to log",
                        "items": {
                            "type": "object",
                            "properties": {
                                "weight": {"type": "number"},
                                "unit": {"type": "string", "enum": ["lbs", "kg"]},
                                "reps": {"type": "integer"},
                            },
                            "required": ["weight", "unit", "reps"],
                        },
                    },
                },
                "required": ["exercise_name", "sets"],
            },
        ),
        types.Tool(
            name="execute_staged_workout",
            description=(
                "Executes the previously staged workout log after user confirmation. "
                "Only call this after the user has explicitly confirmed the staged write."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="set_goal",
            description=(
                "Set a training goal for a specific exercise. "
                "Stages the write and requires user confirmation. "
                "Always call resolve_exercise_name first to verify the exact exercise name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name from resolve_exercise_name",
                    },
                    "target_weight": {"type": "number"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                    "target_reps": {"type": "integer"},
                    "target_date": {
                        "type": "string",
                        "description": "target date in YYYY-MM-DD format",
                    },
                    "title": {
                        "type": "string",
                        "description": "short description of the goal (optional)",
                    },
                },
                "required": ["exercise_name", "target_weight", "unit", "target_reps", "target_date"],
            },
        ),
        types.Tool(
            name="execute_staged_goal",
            description=(
                "Executes the previously staged goal after user confirmation. "
                "Only call this after the user has explicitly confirmed the staged write."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="log_bodyweight",
            description=(
                "Log today's body weight and optional body fat percentage. "
                "This tool writes directly to the database after CLI confirmation — no separate execute call needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "body_weight": {
                        "type": "number",
                        "description": "body weight value in the specified unit",
                    },
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                    "body_fat_percent": {
                        "type": "number",
                        "description": "body fat percentage (optional)",
                    },
                },
                "required": ["body_weight", "unit"],
            },
        ),
        types.Tool(
            name="verify_workout_logged",
            description=(
                "Verifies that a workout was actually saved to the database by checking "
                "for the specific sets on a specific date. Call this immediately after "
                "execute_staged_workout to confirm the write succeeded. "
                "Only checks the exact date and exercise — does not scan all data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name",
                    },
                    "date": {
                        "type": "string",
                        "description": "date in YYYY-MM-DD format",
                    },
                    "expected_sets": {
                        "type": "integer",
                        "description": "number of sets that should have been written",
                    },
                },
                "required": ["exercise_name", "date", "expected_sets"],
            },
        ),
        types.Tool(
            name="verify_goal_set",
            description=(
                "Verifies that a goal was actually saved to the database. "
                "Call immediately after execute_staged_goal. "
                "Only checks the specific exercise and target date."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name",
                    },
                    "target_date": {
                        "type": "string",
                        "description": "target date in YYYY-MM-DD format",
                    },
                },
                "required": ["exercise_name", "target_date"],
            },
        ),
        types.Tool(
            name="update_goal",
            description=(
                "Update an existing goal's target weight, reps, or date. "
                "Stages the update and requires confirmation. Shows before → after. "
                "If current_target_date is omitted, looks up all goals and auto-selects if only one exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "current_target_date": {
                        "type": "string",
                        "description": "identifies which goal to update (YYYY-MM-DD); omit if unknown — tool will look it up",
                    },
                    "new_target_weight": {"type": "number"},
                    "new_target_reps": {"type": "integer"},
                    "new_target_date": {"type": "string"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                },
                "required": ["exercise_name"],
            },
        ),
        types.Tool(
            name="execute_staged_goal_update",
            description="Executes the staged goal update. Only call after user confirms.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="delete_goal",
            description=(
                "Permanently deletes a goal from the database. "
                "DESTRUCTIVE — cannot be undone. Requires extra confirmation. "
                "If target_date is omitted, looks up all goals and auto-selects if only one exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "target_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD; omit if unknown — tool will look it up",
                    },
                },
                "required": ["exercise_name"],
            },
        ),
        types.Tool(
            name="execute_staged_goal_delete",
            description="Permanently deletes the staged goal. Only call after user confirms.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="update_workout_set",
            description=(
                "Correct a logged set — fix wrong weight or rep count. "
                "Stages the update and requires confirmation. Shows before → after. "
                "If date is omitted, returns 3 disambiguation options to help identify the session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD; omit if unknown — tool will return disambiguation options"},
                    "old_weight": {
                        "type": "number",
                        "description": "the incorrect weight as originally typed",
                    },
                    "old_reps": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                    "new_weight": {"type": "number"},
                    "new_reps": {"type": "integer"},
                },
                "required": ["exercise_name", "old_weight", "old_reps", "unit"],
            },
        ),
        types.Tool(
            name="execute_staged_set_update",
            description="Executes the staged workout set update. Only call after user confirms.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="delete_workout_set",
            description=(
                "Permanently removes a specific set from training history. "
                "DESTRUCTIVE — corrupts historical records if misused. Extra confirmation required. "
                "If date is omitted, returns 3 disambiguation options to help identify the session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD; omit if unknown — tool will return disambiguation options"},
                    "weight": {
                        "type": "number",
                        "description": "as originally typed",
                    },
                    "reps": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                },
                "required": ["exercise_name", "weight", "reps", "unit"],
            },
        ),
        types.Tool(
            name="execute_staged_set_delete",
            description="Permanently deletes the staged workout set. Only call after user confirms.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="verify_set_updated",
            description=(
                "Verifies a workout set was correctly updated by checking the new values "
                "exist on the given date. Call after execute_staged_set_update."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "date": {"type": "string"},
                    "expected_weight": {
                        "type": "number",
                        "description": "new weight as typed",
                    },
                    "expected_reps": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                },
                "required": ["exercise_name", "date", "expected_weight", "expected_reps", "unit"],
            },
        ),
        types.Tool(
            name="verify_set_deleted",
            description=(
                "Verifies a workout set was deleted by confirming it no longer exists. "
                "Call after execute_staged_set_delete. verified: true means deletion succeeded."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string"},
                    "date": {"type": "string"},
                    "weight": {
                        "type": "number",
                        "description": "as originally typed",
                    },
                    "reps": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["lbs", "kg"]},
                },
                "required": ["exercise_name", "date", "weight", "reps", "unit"],
            },
        ),
        types.Tool(
            name="get_exercise_sessions",
            description=(
                "Get a list of dates and basic stats for an exercise, used for disambiguation "
                "when the user is not sure which session to update or delete. "
                "Supports three modes: recent (last N sessions), approximate (within ±7 days "
                "of a given date), and range (between two dates). "
                "Returns session summaries only — call update/delete with the specific date once identified."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "exercise_name": {
                        "type": "string",
                        "description": "exact exercise name",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["recent", "approximate", "range"],
                        "description": "recent: last N sessions; approximate: within ±7 days of a date; range: between two dates",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "number of sessions to return (default 10, max 20); used for mode 'recent'",
                    },
                    "approximate_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD centre date; used for mode 'approximate'",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "YYYY-MM-DD start of range; used for mode 'range'",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "YYYY-MM-DD end of range; used for mode 'range'",
                    },
                },
                "required": ["exercise_name", "mode"],
            },
        ),
        types.Tool(
            name="remember_fact",
            description=(
                "Store a fact about the user for future sessions. Call when the user "
                "tells you something personal, states a preference, mentions an injury, reveals a "
                "training pattern, or clarifies a data convention not in user_context.json."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["user_fact", "preference", "training_pattern", "injury", "convention"],
                        "description": "category of the fact",
                    },
                    "content": {
                        "type": "string",
                        "description": "the fact to remember, written as a clear statement",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["user_stated", "agent_inferred"],
                        "description": "how the fact was learned (default: user_stated)",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "confidence level (default: high)",
                    },
                },
                "required": ["category", "content"],
            },
        ),
        types.Tool(
            name="recall_memories",
            description=(
                "Retrieve all facts stored about the user from previous sessions. "
                "Call at the start of personal questions to check if relevant context exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="forget_fact",
            description=(
                "Delete a stored memory by its ID. Use when the user says something "
                "is outdated or asks you to forget it. First call recall_memories to get the ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": "the ID of the fact to delete",
                    },
                },
                "required": ["fact_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "query_workout_data":
        result = await _query_workout_data(arguments["question"])
    elif name == "get_personal_record":
        result = await _get_personal_record(arguments["exercise_name"])
    elif name == "get_exercise_history":
        result = await _get_exercise_history(
            arguments["exercise_name"],
            int(arguments.get("days", 30)),
        )
    elif name == "get_weekly_volume":
        result = await _get_weekly_volume(int(arguments.get("days", 30)))
    elif name == "run_read_only_sql":
        result = await _run_read_only_sql(arguments["sql"])
    elif name == "search_fitness_knowledge":
        query = arguments["query"]
        n_results = min(int(arguments.get("n_results", 5)), 10)
        result = await _search_fitness_knowledge(query, n_results)
    elif name == "resolve_exercise_name":
        result = await _resolve_exercise_name(arguments["user_term"])
    elif name == "read_exercise_comments":
        result = await _read_exercise_comments(
            arguments["exercise_name"],
            arguments.get("date_from"),
            arguments.get("date_to"),
            int(arguments.get("limit", 15)),  # default 15, hard cap 15
        )
    elif name == "log_workout":
        result = await _log_workout(arguments)
    elif name == "execute_staged_workout":
        result = await _execute_staged_workout()
    elif name == "set_goal":
        result = await _set_goal(arguments)
    elif name == "execute_staged_goal":
        result = await _execute_staged_goal()
    elif name == "log_bodyweight":
        result = await _log_bodyweight(arguments)
    elif name == "verify_workout_logged":
        result = await _verify_workout_logged(
            arguments["exercise_name"],
            arguments["date"],
            int(arguments["expected_sets"]),
        )
    elif name == "verify_goal_set":
        result = await _verify_goal_set(
            arguments["exercise_name"],
            arguments["target_date"],
        )
    elif name == "update_goal":
        result = await _update_goal(arguments)
    elif name == "execute_staged_goal_update":
        result = await _execute_staged_goal_update()
    elif name == "delete_goal":
        result = await _delete_goal(arguments)
    elif name == "execute_staged_goal_delete":
        result = await _execute_staged_goal_delete()
    elif name == "update_workout_set":
        result = await _update_workout_set(arguments)
    elif name == "execute_staged_set_update":
        result = await _execute_staged_set_update()
    elif name == "delete_workout_set":
        result = await _delete_workout_set(arguments)
    elif name == "execute_staged_set_delete":
        result = await _execute_staged_set_delete()
    elif name == "verify_set_updated":
        result = await _verify_set_updated(
            arguments["exercise_name"],
            arguments["date"],
            float(arguments["expected_weight"]),
            int(arguments["expected_reps"]),
            arguments["unit"],
        )
    elif name == "verify_set_deleted":
        result = await _verify_set_deleted(
            arguments["exercise_name"],
            arguments["date"],
            float(arguments["weight"]),
            int(arguments["reps"]),
            arguments["unit"],
        )
    elif name == "get_exercise_sessions":
        result = await _get_exercise_sessions(arguments)
    elif name == "remember_fact":
        result = _remember_fact(arguments)
    elif name == "recall_memories":
        result = _recall_memories()
    elif name == "forget_fact":
        result = _forget_fact(arguments)
    else:
        result = json.dumps({"error": f"Unknown tool: {name}"})

    return [types.TextContent(type="text", text=result)]


def _query_workout_data_sync(question: str) -> str:
    from src.text_to_sql import answer_question

    result = answer_question(question, DB_PATH)
    return json.dumps(
        {
            "sql": result.get("sql", ""),
            "answer": result.get("answer", ""),
            "rows_returned": len(result.get("rows", [])),
            "error": result.get("error"),
        }
    )


async def _query_workout_data(question: str) -> str:
    return await asyncio.to_thread(_query_workout_data_sync, question)


def _get_personal_record_sync(exercise_name: str) -> str:
    from src.text_to_sql import answer_question

    question = f"What is my PR for {exercise_name}?"
    result = answer_question(question, DB_PATH)
    rows = result.get("rows", [])
    out = {
        "exercise": exercise_name,
        "answer": result.get("answer", ""),
        "sql": result.get("sql", ""),
        "rows_returned": len(rows),
        "error": result.get("error"),
    }
    if rows and rows[0].get("reps") == 1:
        out["single_rep_warning"] = True
        out["warning_message"] = "This PR is a single-rep set. Check comments — it may be a failed attempt or form break rather than a true max."
    return json.dumps(out)


async def _get_personal_record(exercise_name: str) -> str:
    return await asyncio.to_thread(_get_personal_record_sync, exercise_name)


def _get_exercise_history_sync(exercise_name: str, days: int = 30) -> str:
    from src.db import get_connection, run_query

    safe_name = exercise_name.replace("'", "''")
    sql = f"""
        SELECT tl.date,
               tl.metric_weight * 2.2046 AS typed_value,
               tl.reps
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        WHERE e.name = '{safe_name}'
          AND tl.date >= date('now', '-{days} days')
        ORDER BY tl.date DESC, tl.metric_weight DESC
        LIMIT 100;
    """
    try:
        conn = get_connection(DB_PATH)
        rows = run_query(conn, sql)
        return json.dumps(
            {
                "exercise": exercise_name,
                "days": days,
                "note": "typed_value is metric_weight * 2.2046 — recovers the original logged number",
                "rows": rows,
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def _get_exercise_history(exercise_name: str, days: int = 30) -> str:
    return await asyncio.to_thread(_get_exercise_history_sync, exercise_name, days)


def _get_weekly_volume_sync(days: int = 30) -> str:
    from src.db import get_connection, run_query

    sql = f"""
        SELECT c.name AS muscle_group,
               COUNT(*) AS total_sets,
               ROUND(SUM(tl.metric_weight * 2.2046 * tl.reps), 1) AS total_volume
        FROM training_log tl
        JOIN exercise e ON tl.exercise_id = e._id
        JOIN Category c ON e.category_id = c._id
        WHERE tl.date >= date('now', '-{days} days')
        GROUP BY c.name
        ORDER BY total_sets DESC;
    """
    try:
        conn = get_connection(DB_PATH)
        rows = run_query(conn, sql)
        return json.dumps(
            {
                "days": days,
                "note": "total_volume uses typed_value (metric_weight * 2.2046) x reps.",
                "volume_by_muscle_group": rows,
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def _get_weekly_volume(days: int = 30) -> str:
    return await asyncio.to_thread(_get_weekly_volume_sync, days)


async def _run_read_only_sql(sql: str) -> str:
    normalized = sql.lstrip().upper()
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return json.dumps({"error": "Rejected: SQL must start with SELECT or WITH."})

    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)
    referenced = {t for t in table_refs}
    disallowed = referenced - ALLOWED_TABLES
    if disallowed:
        return json.dumps({"error": f"Rejected: references disallowed tables: {sorted(disallowed)}"})

    from src.db import get_connection, run_query

    try:
        conn = get_connection(DB_PATH)
        rows = await asyncio.to_thread(run_query, conn, sql, 100, 5)
        return json.dumps({"rows": rows, "rows_returned": len(rows)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _search_fitness_knowledge_sync(query: str, n_results: int = 5) -> str:
    from src.answer import _documents_are_relevant

    kb = _get_kb()
    results = kb.retrieve(query, n_results)

    if not results:
        return json.dumps(
            {"found": False, "message": "No relevant research found for this query."}
        )

    if not _documents_are_relevant(query, results):
        return json.dumps(
            {"found": False, "message": "No relevant research found for this query."}
        )

    docs = [
        {
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "year": doc.get("year", ""),
            "url": doc.get("url", ""),
            "text": doc.get("text", ""),
        }
        for doc in results
    ]
    return json.dumps({"found": True, "documents": docs})


async def _search_fitness_knowledge(query: str, n_results: int = 5) -> str:
    return await asyncio.to_thread(_search_fitness_knowledge_sync, query, n_results)


def _resolve_exercise_name_sync(user_term: str) -> str:
    from src.db import get_connection

    conn = get_connection(DB_PATH)

    # Exact match first
    cursor = conn.execute(
        "SELECT name FROM exercise WHERE LOWER(name) = LOWER(?)", (user_term,)
    )
    row = cursor.fetchone()
    if row:
        return json.dumps({
            "exact_match": True,
            "resolved_name": row["name"],
            "candidates": [],
            "message": "Exact match found.",
        })

    # Partial match
    cursor = conn.execute(
        "SELECT name FROM exercise WHERE LOWER(name) LIKE LOWER(?) ORDER BY name LIMIT 8",
        (f"%{user_term}%",),
    )
    candidates = [r["name"] for r in cursor.fetchall()]
    if candidates:
        return json.dumps({
            "exact_match": False,
            "resolved_name": None,
            "candidates": candidates,
            "message": (
                f"No exact match. Found {len(candidates)} possible exercise(s). "
                "Present these to the user and ask which one they mean before calling any data tool."
            ),
        })

    # Word-by-word fallback
    words = user_term.split()
    if words:
        placeholders = " OR ".join(["LOWER(name) LIKE LOWER(?)"] * len(words))
        params = tuple(f"%{w}%" for w in words)
        cursor = conn.execute(
            f"SELECT DISTINCT name FROM exercise WHERE {placeholders} ORDER BY name LIMIT 8",
            params,
        )
        candidates = [r["name"] for r in cursor.fetchall()]

    if candidates:
        return json.dumps({
            "exact_match": False,
            "resolved_name": None,
            "candidates": candidates,
            "message": (
                f"No exact or partial match. Found {len(candidates)} exercise(s) matching "
                "individual words. Present these to the user and ask which one they mean."
            ),
        })

    return json.dumps({
        "exact_match": False,
        "resolved_name": None,
        "candidates": [],
        "message": f"No exercises found matching '{user_term}'. Try a different term.",
    })


async def _resolve_exercise_name(user_term: str) -> str:
    return await asyncio.to_thread(_resolve_exercise_name_sync, user_term)


def _get_interpretation_note(exercise_name: str) -> str:
    ctx_path = os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "user_context.json")
    try:
        with open(ctx_path, encoding="utf-8") as f:
            ctx = json.load(f)
        h = ctx.get("form_quality_hierarchies", {}).get(exercise_name)
        if h:
            hierarchy = h.get("hierarchy_best_to_worst")
            if hierarchy:
                terms = [item.split(" — ")[0].split(" / ")[0].strip() for item in hierarchy]
                chain = " > ".join(terms)
                return (
                    f"{h.get('type', 'Form quality hierarchy')}. "
                    f"Best to worst: {chain}. "
                    "Form errors are explicitly stated — absence of form notes means form was acceptable."
                )
            return (
                f"{h.get('type', 'Form quality tracking')}. "
                "Form errors are explicitly stated — absence of form notes means form was acceptable."
            )
    except Exception:
        pass
    return "Comments describe set quality, equipment, and form observations."


def _read_exercise_comments_sync(
    exercise_name: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 15,
) -> str:
    from src.db import get_connection

    limit = min(int(limit), 15)
    conn = get_connection(DB_PATH)

    conditions = ["e.name = ?"]
    params: list = [exercise_name]

    if date_from:
        conditions.append("c.date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("c.date <= ?")
        params.append(date_to)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT c.date, tl.metric_weight * 2.2046 AS typed_value, tl.reps, c.comment
        FROM Comment c
        JOIN training_log tl ON tl._id = c.owner_id
        JOIN exercise e ON e._id = tl.exercise_id
        WHERE {where}
        ORDER BY c.date ASC
        LIMIT ?
    """
    params.append(limit)

    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            return json.dumps({
                "found": False,
                "message": f"No comments recorded for {exercise_name} in this date range.",
            })
        comments = [
            {
                "date": row["date"],
                "typed_value": row["typed_value"],
                "reps": row["reps"],
                "comment": row["comment"],
            }
            for row in rows
        ]
        return json.dumps({
            "found": True,
            "exercise": exercise_name,
            "count": len(comments),
            "comments": comments,
            "interpretation_note": _get_interpretation_note(exercise_name),
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def _read_exercise_comments(
    exercise_name: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 15,
) -> str:
    return await asyncio.to_thread(
        _read_exercise_comments_sync, exercise_name, date_from, date_to, limit
    )


def _log_workout_sync(arguments: dict) -> str:
    import datetime
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    date_str = arguments.get("date") or ""
    sets = arguments["sets"]

    if not date_str.strip():
        return json.dumps({
            "error": True,
            "needs_clarification": True,
            "message": "Date is required to log a workout. Please ask the user: 'What date was this workout? (format: YYYY-MM-DD or say today/yesterday)'",
        })

    workout_date = datetime.date.fromisoformat(date_str)
    if workout_date > datetime.date.today():
        return json.dumps({
            "error": True,
            "message": f"Cannot log a workout for {date_str} — that date is in the future.",
        })

    conn = get_connection(DB_PATH)
    cursor = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,))
    row = cursor.fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found. Use resolve_exercise_name first."})

    exercise_id = row["_id"]

    cursor = conn.execute(
        "SELECT MAX(metric_weight) AS max_w FROM training_log WHERE exercise_id = ?",
        (exercise_id,),
    )
    pr_row = cursor.fetchone()
    current_pr_metric = pr_row["max_w"] if pr_row and pr_row["max_w"] is not None else 0.0

    staged_sets = []
    new_prs = 0

    for s in sets:
        weight = float(s["weight"])
        unit = s["unit"]
        reps = int(s["reps"])
        metric_weight = weight / 2.2046
        is_pr = metric_weight > current_pr_metric
        if is_pr:
            current_pr_metric = metric_weight
            new_prs += 1
        staged_sets.append({
            "metric_weight": metric_weight,
            "reps": reps,
            "is_personal_record": 1 if is_pr else 0,
        })

    _staged_writes["workout"] = {
        "exercise_id": exercise_id,
        "date": date_str,
        "sets": staged_sets,
    }

    pr_note = f" ({new_prs} new PR{'s' if new_prs != 1 else ''})" if new_prs else ""
    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "workout",
        "summary": f"{len(sets)} sets of {exercise_name} on {date_str}{pr_note}",
        "next_step": "Call execute_staged_workout to complete the write.",
    })


async def _log_workout(arguments: dict) -> str:
    return await asyncio.to_thread(_log_workout_sync, arguments)


def _execute_staged_workout_sync() -> str:
    from src.db import get_write_connection

    staged = _staged_writes.get("workout")
    if not staged:
        return json.dumps({"error": "No staged workout found. Call log_workout first."})

    conn = get_write_connection(DB_PATH)
    sets_written = 0
    try:
        for s in staged["sets"]:
            conn.execute(
                """INSERT INTO training_log
                   (exercise_id, date, metric_weight, reps, unit, is_personal_record, is_complete)
                   VALUES (?, ?, ?, ?, 0, ?, 1)""",
                (staged["exercise_id"], staged["date"], s["metric_weight"], s["reps"], s["is_personal_record"]),
            )
            sets_written += 1
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to write workout: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("workout", None)
    return json.dumps({"success": True, "sets_written": sets_written, "message": "Workout logged successfully."})


async def _execute_staged_workout() -> str:
    return await asyncio.to_thread(_execute_staged_workout_sync)


def _set_goal_sync(arguments: dict) -> str:
    import datetime
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    target_weight = float(arguments["target_weight"])
    unit = arguments["unit"]
    target_reps = int(arguments["target_reps"])
    target_date = arguments["target_date"]
    title = arguments.get("title", f"Reach {target_weight} {unit} on {exercise_name}")

    conn = get_connection(DB_PATH)
    cursor = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,))
    row = cursor.fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found. Use resolve_exercise_name first."})

    exercise_id = row["_id"]
    metric_weight = target_weight / 2.2046

    _staged_writes["goal"] = {
        "exercise_id": exercise_id,
        "metric_weight": metric_weight,
        "reps": target_reps,
        "title": title,
        "target_date": target_date,
        "start_date": datetime.date.today().isoformat(),
    }

    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "goal",
        "summary": f"{exercise_name} — {target_weight} {unit} × {target_reps} reps by {target_date}",
        "next_step": "Call execute_staged_goal to complete the write.",
    })


async def _set_goal(arguments: dict) -> str:
    return await asyncio.to_thread(_set_goal_sync, arguments)


def _execute_staged_goal_sync() -> str:
    from src.db import get_write_connection

    staged = _staged_writes.get("goal")
    if not staged:
        return json.dumps({"error": "No staged goal found. Call set_goal first."})

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO Goal
               (type_id, exercise_id, metric_weight, reps, unit, title, target_date,
                sort_order, distance, duration_seconds, start_date)
               VALUES (1, ?, ?, ?, 0, ?, ?, 0, 0, 0, ?)""",
            (
                staged["exercise_id"],
                staged["metric_weight"],
                staged["reps"],
                staged["title"],
                staged["target_date"],
                staged["start_date"],
            ),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to write goal: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("goal", None)
    return json.dumps({"success": True, "message": "Goal saved successfully."})


async def _execute_staged_goal() -> str:
    return await asyncio.to_thread(_execute_staged_goal_sync)


def _log_bodyweight_sync(arguments: dict) -> str:
    import datetime
    from src.db import get_write_connection

    body_weight = float(arguments["body_weight"])
    unit = arguments["unit"]
    body_fat_percent = arguments.get("body_fat_percent")

    today = datetime.date.today().isoformat()
    body_weight_metric = body_weight / 2.2046

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO BodyWeight (date, body_weight_metric, body_fat) VALUES (?, ?, ?)",
            (today, body_weight_metric, body_fat_percent),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to log body weight: {exc}"})
    finally:
        conn.close()

    return json.dumps({"success": True, "message": f"Body weight logged: {body_weight} {unit} on {today}."})


async def _log_bodyweight(arguments: dict) -> str:
    return await asyncio.to_thread(_log_bodyweight_sync, arguments)


def _verify_workout_logged_sync(exercise_name: str, date: str, expected_sets: int) -> str:
    from src.db import get_connection

    conn = get_connection(DB_PATH)
    sql = """
        SELECT tl.metric_weight * 2.2046 AS typed_value, tl.reps, tl.is_personal_record
        FROM training_log tl
        JOIN exercise e ON e._id = tl.exercise_id
        WHERE e.name = :exercise_name AND tl.date = :date
        ORDER BY tl._id ASC
    """
    try:
        rows = conn.execute(sql, {"exercise_name": exercise_name, "date": date}).fetchall()
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    sets = [
        {
            "weight_as_typed": round(row["typed_value"], 2),
            "reps": row["reps"],
            "is_pr": bool(row["is_personal_record"]),
        }
        for row in rows
    ]
    sets_found = len(sets)
    verified = sets_found == expected_sets
    message = (
        f"✅ {sets_found} sets verified in database."
        if verified
        else f"❌ Expected {expected_sets} sets, found {sets_found}. Write may have failed."
    )
    return json.dumps({
        "verified": verified,
        "exercise": exercise_name,
        "date": date,
        "sets_found": sets_found,
        "sets_expected": expected_sets,
        "sets": sets,
        "message": message,
    })


async def _verify_workout_logged(exercise_name: str, date: str, expected_sets: int) -> str:
    return await asyncio.to_thread(_verify_workout_logged_sync, exercise_name, date, expected_sets)


def _verify_goal_set_sync(exercise_name: str, target_date: str) -> str:
    from src.db import get_connection

    conn = get_connection(DB_PATH)
    sql = """
        SELECT g.metric_weight * 2.2046 AS target_typed, g.reps, g.target_date, g.start_date, g.title
        FROM Goal g
        JOIN exercise e ON e._id = g.exercise_id
        WHERE e.name = :exercise_name AND g.target_date = :target_date
    """
    try:
        row = conn.execute(sql, {"exercise_name": exercise_name, "target_date": target_date}).fetchone()
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if row:
        return json.dumps({
            "verified": True,
            "exercise": exercise_name,
            "target_weight_as_typed": round(row["target_typed"], 2),
            "target_reps": row["reps"],
            "target_date": row["target_date"],
            "message": "✅ Goal verified in database.",
        })
    return json.dumps({
        "verified": False,
        "exercise": exercise_name,
        "target_date": target_date,
        "message": "❌ Goal not found in database. Write may have failed.",
    })


async def _verify_goal_set(exercise_name: str, target_date: str) -> str:
    return await asyncio.to_thread(_verify_goal_set_sync, exercise_name, target_date)


def _update_goal_sync(arguments: dict) -> str:
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    current_target_date = arguments.get("current_target_date") or ""
    new_target_weight = arguments.get("new_target_weight")
    new_target_reps = arguments.get("new_target_reps")
    new_target_date = arguments.get("new_target_date")
    unit = arguments.get("unit", "lbs")

    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    if not current_target_date.strip():
        goals = conn.execute(
            """SELECT g.metric_weight * 2.2046 AS typed_weight, g.reps, g.target_date, g.start_date
               FROM Goal g WHERE g.exercise_id = ? ORDER BY g.target_date""",
            (exercise_id,),
        ).fetchall()
        if len(goals) == 0:
            return json.dumps({"error": True, "message": f"No goals found for {exercise_name}."})
        if len(goals) == 1:
            current_target_date = goals[0]["target_date"]
        else:
            return json.dumps({
                "needs_clarification": True,
                "message": f"Multiple goals found for {exercise_name}. Please specify which one:",
                "goals": [
                    {
                        "target_date": g["target_date"],
                        "typed_weight": round(g["typed_weight"], 2),
                        "reps": g["reps"],
                        "start_date": g["start_date"],
                    }
                    for g in goals
                ],
            })

    goal = conn.execute(
        "SELECT * FROM Goal WHERE exercise_id = ? AND target_date = ?",
        (exercise_id, current_target_date),
    ).fetchone()
    if not goal:
        return json.dumps({"error": f"No goal found for '{exercise_name}' with target date {current_target_date}."})

    current_typed = round(goal["metric_weight"] * 2.2046, 2)
    current_reps = goal["reps"]
    current_date = goal["target_date"]

    new_metric_weight = (float(new_target_weight) / 2.2046) if new_target_weight is not None else None
    final_metric = new_metric_weight if new_metric_weight is not None else goal["metric_weight"]
    final_reps = int(new_target_reps) if new_target_reps is not None else current_reps
    final_date = new_target_date or current_date

    _staged_writes["update_goal"] = {
        "goal_id": goal["_id"],
        "new_metric_weight": final_metric,
        "new_reps": final_reps,
        "new_target_date": final_date,
    }
    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "update_goal",
        "summary": f"Update {exercise_name} goal (target date: {final_date})",
        "next_step": "Call execute_staged_goal_update to complete the write.",
    })


async def _update_goal(arguments: dict) -> str:
    return await asyncio.to_thread(_update_goal_sync, arguments)


def _execute_staged_goal_update_sync() -> str:
    from src.db import get_write_connection

    staged = _staged_writes.get("update_goal")
    if not staged:
        return json.dumps({"error": "No staged goal update found. Call update_goal first."})

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute(
            "UPDATE Goal SET metric_weight = ?, reps = ?, target_date = ? WHERE _id = ?",
            (staged["new_metric_weight"], staged["new_reps"], staged["new_target_date"], staged["goal_id"]),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to update goal: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("update_goal", None)
    return json.dumps({"success": True, "message": "Goal updated successfully."})


async def _execute_staged_goal_update() -> str:
    return await asyncio.to_thread(_execute_staged_goal_update_sync)


def _delete_goal_sync(arguments: dict) -> str:
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    target_date = arguments.get("target_date") or ""

    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    if not target_date.strip():
        goals = conn.execute(
            """SELECT g.metric_weight * 2.2046 AS typed_weight, g.reps, g.target_date, g.start_date
               FROM Goal g WHERE g.exercise_id = ? ORDER BY g.target_date""",
            (exercise_id,),
        ).fetchall()
        if len(goals) == 0:
            return json.dumps({"error": True, "message": f"No goals found for {exercise_name}."})
        if len(goals) == 1:
            target_date = goals[0]["target_date"]
        else:
            return json.dumps({
                "needs_clarification": True,
                "message": f"Multiple goals found for {exercise_name}. Please specify which one:",
                "goals": [
                    {
                        "target_date": g["target_date"],
                        "typed_weight": round(g["typed_weight"], 2),
                        "reps": g["reps"],
                        "start_date": g["start_date"],
                    }
                    for g in goals
                ],
            })

    goal = conn.execute(
        "SELECT * FROM Goal WHERE exercise_id = ? AND target_date = ?",
        (exercise_id, target_date),
    ).fetchone()
    if not goal:
        return json.dumps({"error": f"No goal found for '{exercise_name}' with target date {target_date}."})

    _staged_writes["delete_goal"] = {
        "goal_id": goal["_id"],
        "exercise_name": exercise_name,
        "target_date": target_date,
    }
    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "delete_goal",
        "summary": f"Delete {exercise_name} goal — {round(goal['metric_weight'] * 2.2046, 2)} lbs × {goal['reps']} reps by {target_date}",
        "next_step": "Call execute_staged_goal_delete to complete the write.",
    })


async def _delete_goal(arguments: dict) -> str:
    return await asyncio.to_thread(_delete_goal_sync, arguments)


def _execute_staged_goal_delete_sync() -> str:
    from src.db import get_write_connection

    staged = _staged_writes.get("delete_goal")
    if not staged:
        return json.dumps({"error": "No staged goal deletion found. Call delete_goal first."})

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute("DELETE FROM Goal WHERE _id = ?", (staged["goal_id"],))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to delete goal: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("delete_goal", None)
    return json.dumps({"success": True, "message": "Goal deleted successfully."})


async def _execute_staged_goal_delete() -> str:
    return await asyncio.to_thread(_execute_staged_goal_delete_sync)


def _update_workout_set_sync(arguments: dict) -> str:
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    date = arguments.get("date") or ""
    if not date.strip():
        return json.dumps({
            "needs_clarification": True,
            "message": "Date required to identify the specific set. Choose an option:",
            "options": {
                "1": "Give an approximate date — I'll show records within 7 days of it",
                "2": "Show me the last 10 sessions for this exercise (newest first)",
                "3": "Give a date range — I'll show all sessions within it",
            },
        })
    old_weight = float(arguments["old_weight"])
    old_reps = int(arguments["old_reps"])
    unit = arguments["unit"]
    new_weight = arguments.get("new_weight")
    new_reps = arguments.get("new_reps")

    old_stored = old_weight / 2.2046

    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    matches = conn.execute(
        """SELECT tl._id, tl.metric_weight * 2.2046 AS typed_value, tl.reps, tl.is_personal_record
           FROM training_log tl
           WHERE tl.exercise_id = ? AND tl.date = ? AND tl.reps = ?
             AND ABS(tl.metric_weight - ?) < 0.01
           ORDER BY tl._id ASC""",
        (exercise_id, date, old_reps, old_stored),
    ).fetchall()

    if not matches:
        return json.dumps({"error": f"No set found for '{exercise_name}' on {date}: {old_weight} {unit} × {old_reps} reps."})
    if len(matches) > 1:
        return json.dumps({
            "needs_clarification": True,
            "message": f"Multiple identical sets found ({old_weight} {unit} × {old_reps} reps) on {date}. Cannot determine which to update.",
        })

    target = matches[0]
    current_typed = round(target["typed_value"], 2)
    current_reps = target["reps"]

    eff_new_weight = float(new_weight) if new_weight is not None else None
    eff_new_reps = int(new_reps) if new_reps is not None else None
    final_metric = (eff_new_weight / 2.2046) if eff_new_weight is not None else (old_weight / 2.2046)
    final_reps = eff_new_reps if eff_new_reps is not None else current_reps

    _staged_writes["update_set"] = {
        "set_id": target["_id"],
        "exercise_id": exercise_id,
        "date": date,
        "new_metric_weight": final_metric,
        "new_reps": final_reps,
        "new_typed_weight": eff_new_weight or current_typed,
        "unit": unit,
        "exercise_name": exercise_name,
    }
    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "update_set",
        "summary": f"Update {exercise_name} on {date}",
        "next_step": "Call execute_staged_set_update to complete the write.",
    })


async def _update_workout_set(arguments: dict) -> str:
    return await asyncio.to_thread(_update_workout_set_sync, arguments)


def _execute_staged_set_update_sync() -> str:
    from src.db import get_connection, get_write_connection

    staged = _staged_writes.get("update_set")
    if not staged:
        return json.dumps({"error": "No staged set update found. Call update_workout_set first."})

    # Recheck PR: is the new weight a new all-time best for this exercise?
    read_conn = get_connection(DB_PATH)
    pr_row = read_conn.execute(
        "SELECT MAX(metric_weight) AS max_w FROM training_log WHERE exercise_id = ? AND _id != ?",
        (staged["exercise_id"], staged["set_id"]),
    ).fetchone()
    other_max = pr_row["max_w"] if pr_row and pr_row["max_w"] is not None else 0.0
    is_pr = 1 if staged["new_metric_weight"] > other_max else 0

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute(
            "UPDATE training_log SET metric_weight = ?, reps = ?, is_personal_record = ? WHERE _id = ?",
            (staged["new_metric_weight"], staged["new_reps"], is_pr, staged["set_id"]),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to update set: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("update_set", None)
    return json.dumps({"success": True, "message": "Set updated successfully.", "is_personal_record": bool(is_pr)})


async def _execute_staged_set_update() -> str:
    return await asyncio.to_thread(_execute_staged_set_update_sync)


def _delete_workout_set_sync(arguments: dict) -> str:
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    date = arguments.get("date") or ""
    if not date.strip():
        return json.dumps({
            "needs_clarification": True,
            "message": "Date required to identify the specific set. Choose an option:",
            "options": {
                "1": "Give an approximate date — I'll show records within 7 days of it",
                "2": "Show me the last 10 sessions for this exercise (newest first)",
                "3": "Give a date range — I'll show all sessions within it",
            },
        })
    weight = float(arguments["weight"])
    reps = int(arguments["reps"])
    unit = arguments["unit"]

    stored_weight = weight / 2.2046

    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    matches = conn.execute(
        """SELECT tl._id, tl.metric_weight * 2.2046 AS typed_value, tl.reps, tl.is_personal_record
           FROM training_log tl
           WHERE tl.exercise_id = ? AND tl.date = ? AND tl.reps = ?
             AND ABS(tl.metric_weight - ?) < 0.01
           ORDER BY tl._id ASC""",
        (exercise_id, date, reps, stored_weight),
    ).fetchall()

    if not matches:
        return json.dumps({"error": f"No set found for '{exercise_name}' on {date}: {weight} {unit} × {reps} reps."})
    if len(matches) > 1:
        return json.dumps({
            "needs_clarification": True,
            "message": f"Multiple identical sets found ({weight} {unit} × {reps} reps) on {date}. Cannot determine which to delete.",
        })

    target = matches[0]
    typed_w = round(target["typed_value"], 2)

    _staged_writes["delete_set"] = {
        "set_id": target["_id"],
        "exercise_name": exercise_name,
        "date": date,
        "weight": typed_w,
        "reps": reps,
        "unit": unit,
        "stored_weight": stored_weight,
        "exercise_id": exercise_id,
    }
    return json.dumps({
        "staged": True,
        "requires_confirmation": True,
        "staged_key": "delete_set",
        "summary": f"Delete {exercise_name} on {date}: {typed_w} {unit} × {reps} reps",
        "next_step": "Call execute_staged_set_delete to complete the write.",
    })


async def _delete_workout_set(arguments: dict) -> str:
    return await asyncio.to_thread(_delete_workout_set_sync, arguments)


def _execute_staged_set_delete_sync() -> str:
    from src.db import get_write_connection

    staged = _staged_writes.get("delete_set")
    if not staged:
        return json.dumps({"error": "No staged set deletion found. Call delete_workout_set first."})

    conn = get_write_connection(DB_PATH)
    try:
        conn.execute("DELETE FROM training_log WHERE _id = ?", (staged["set_id"],))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return json.dumps({"error": f"Failed to delete set: {exc}"})
    finally:
        conn.close()

    _staged_writes.pop("delete_set", None)
    return json.dumps({"success": True, "message": "Set deleted successfully."})


async def _execute_staged_set_delete() -> str:
    return await asyncio.to_thread(_execute_staged_set_delete_sync)


def _verify_set_updated_sync(exercise_name: str, date: str, expected_weight: float, expected_reps: int, unit: str) -> str:
    from src.db import get_connection

    stored_weight = expected_weight / 2.2046
    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    match = conn.execute(
        """SELECT tl.metric_weight * 2.2046 AS typed_value, tl.reps
           FROM training_log tl
           WHERE tl.exercise_id = ? AND tl.date = ? AND tl.reps = ?
             AND ABS(tl.metric_weight - ?) < 0.01""",
        (exercise_id, date, expected_reps, stored_weight),
    ).fetchone()

    if match:
        return json.dumps({
            "verified": True,
            "exercise": exercise_name,
            "date": date,
            "found_weight": round(match["typed_value"], 2),
            "found_reps": match["reps"],
            "message": f"✅ Updated set verified: {expected_weight} {unit} × {expected_reps} reps found in database.",
        })
    return json.dumps({
        "verified": False,
        "exercise": exercise_name,
        "date": date,
        "message": f"❌ Updated set not found: {expected_weight} {unit} × {expected_reps} reps not in database. Update may have failed.",
    })


async def _verify_set_updated(exercise_name: str, date: str, expected_weight: float, expected_reps: int, unit: str) -> str:
    return await asyncio.to_thread(_verify_set_updated_sync, exercise_name, date, expected_weight, expected_reps, unit)


def _verify_set_deleted_sync(exercise_name: str, date: str, weight: float, reps: int, unit: str) -> str:
    from src.db import get_connection

    stored_weight = weight / 2.2046
    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    match = conn.execute(
        """SELECT tl._id FROM training_log tl
           WHERE tl.exercise_id = ? AND tl.date = ? AND tl.reps = ?
             AND ABS(tl.metric_weight - ?) < 0.01""",
        (exercise_id, date, reps, stored_weight),
    ).fetchone()

    if not match:
        return json.dumps({"verified": True, "message": "✅ Set confirmed deleted — no longer in database."})
    return json.dumps({"verified": False, "message": "❌ Set still exists — delete may have failed."})


async def _verify_set_deleted(exercise_name: str, date: str, weight: float, reps: int, unit: str) -> str:
    return await asyncio.to_thread(_verify_set_deleted_sync, exercise_name, date, weight, reps, unit)


def _get_exercise_sessions_sync(arguments: dict) -> str:
    from src.db import get_connection

    exercise_name = arguments["exercise_name"]
    mode = arguments["mode"]

    conn = get_connection(DB_PATH)
    row = conn.execute("SELECT _id FROM exercise WHERE name = ?", (exercise_name,)).fetchone()
    if not row:
        return json.dumps({"error": f"Exercise '{exercise_name}' not found."})
    exercise_id = row["_id"]

    base_select = """
        SELECT tl._id, tl.date, tl.metric_weight * 2.2046 AS typed_weight, tl.reps
        FROM training_log tl
        WHERE tl.exercise_id = :exercise_id
    """

    try:
        if mode == "recent":
            limit = min(int(arguments.get("limit", 10)), 20)
            raw_rows = conn.execute(
                base_select + " ORDER BY tl.date DESC, tl._id ASC",
                {"exercise_id": exercise_id},
            ).fetchall()
        elif mode == "approximate":
            approximate_date = arguments.get("approximate_date", "")
            raw_rows = conn.execute(
                base_select + """
                  AND tl.date BETWEEN date(:approximate_date, '-7 days')
                                  AND date(:approximate_date, '+7 days')
                ORDER BY tl.date DESC, tl._id ASC""",
                {"exercise_id": exercise_id, "approximate_date": approximate_date},
            ).fetchall()
        elif mode == "range":
            date_from = arguments.get("date_from", "")
            date_to = arguments.get("date_to", "")
            raw_rows = conn.execute(
                base_select + " AND tl.date BETWEEN :date_from AND :date_to ORDER BY tl.date DESC, tl._id ASC",
                {"exercise_id": exercise_id, "date_from": date_from, "date_to": date_to},
            ).fetchall()
        else:
            return json.dumps({"error": f"Unknown mode '{mode}'. Use 'recent', 'approximate', or 'range'."})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    # Group individual rows by date (SQL ORDER BY date DESC preserves recency order)
    sessions_map: dict = {}
    for r in raw_rows:
        d = r["date"]
        if d not in sessions_map:
            sessions_map[d] = []
        sessions_map[d].append({"weight": round(r["typed_weight"], 2), "reps": r["reps"]})

    sessions = [
        {
            "date": date,
            "sets": sets,
            "max_weight": max(s["weight"] for s in sets),
            "total_sets": len(sets),
        }
        for date, sets in sessions_map.items()
    ]

    if mode == "recent":
        sessions = sessions[:limit]

    return json.dumps({
        "exercise": exercise_name,
        "mode": mode,
        "sessions": sessions,
        "count": len(sessions),
    })


async def _get_exercise_sessions(arguments: dict) -> str:
    return await asyncio.to_thread(_get_exercise_sessions_sync, arguments)


def _remember_fact(arguments: dict) -> str:
    from src.memory import add_fact
    result = add_fact(
        category=arguments["category"],
        content=arguments["content"],
        source=arguments.get("source", "user_stated"),
        confidence=arguments.get("confidence", "high"),
    )
    return json.dumps(result)


def _recall_memories() -> str:
    from src.memory import get_all_facts
    facts = get_all_facts()
    if not facts:
        return json.dumps({"message": "No memories stored yet.", "facts": []})
    return json.dumps({"count": len(facts), "facts": facts})


def _forget_fact(arguments: dict) -> str:
    from src.memory import delete_fact
    result = delete_fact(arguments["fact_id"])
    return json.dumps(result)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        # Redirect stdout to stderr so print() diagnostics don't corrupt the MCP
        # protocol stream (stdio_server has already captured sys.stdout.buffer above)
        sys.stdout = sys.stderr
        # Pre-load the knowledge base in the main thread before serving requests.
        # SentenceTransformer initialises OpenMP/PyTorch, which deadlocks when done
        # from a thread-pool thread (asyncio.to_thread) on Windows. Loading here
        # runs in the asyncio main thread where OpenMP is safe, and subsequent
        # retrieve() calls (via asyncio.to_thread) reuse the already-initialised
        # models without triggering another OpenMP init.
        _get_kb()._load()
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
