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

server = Server("fitnotes-db")


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
    else:
        result = json.dumps({"error": f"Unknown tool: {name}"})

    return [types.TextContent(type="text", text=result)]


async def _query_workout_data(question: str) -> str:
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


async def _get_personal_record(exercise_name: str) -> str:
    from src.text_to_sql import answer_question

    question = f"What is my PR for {exercise_name}?"
    result = answer_question(question, DB_PATH)
    return json.dumps(
        {
            "exercise": exercise_name,
            "answer": result.get("answer", ""),
            "sql": result.get("sql", ""),
            "rows_returned": len(result.get("rows", [])),
            "error": result.get("error"),
        }
    )


async def _get_exercise_history(exercise_name: str, days: int = 30) -> str:
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
                "note": "typed_value is metric_weight * 2.2046 — recovers the original logged number (lbs for lbs-native exercises, kg for kg-native exercises)",
                "rows": rows,
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def _get_weekly_volume(days: int = 30) -> str:
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
                "note": "total_volume uses typed_value (metric_weight * 2.2046) × reps. Units are lbs for lbs-native exercises.",
                "volume_by_muscle_group": rows,
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


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
        rows = run_query(conn, sql, row_limit=100, timeout_seconds=5)
        return json.dumps({"rows": rows, "rows_returned": len(rows)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def main():
    async with stdio_server() as (read_stream, write_stream):
        # Redirect stdout to stderr so print() diagnostics don't corrupt the MCP
        # protocol stream (stdio_server has already captured sys.stdout.buffer above)
        sys.stdout = sys.stderr
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
