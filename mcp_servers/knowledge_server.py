import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv

load_dotenv()

server = Server("fitness-knowledge")

_kb = None


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
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_fitness_knowledge":
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )
        ]

    query = arguments["query"]
    n_results = min(int(arguments.get("n_results", 5)), 10)

    result = await _search_fitness_knowledge(query, n_results)
    return [types.TextContent(type="text", text=result)]


async def _search_fitness_knowledge(query: str, n_results: int = 5) -> str:
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
