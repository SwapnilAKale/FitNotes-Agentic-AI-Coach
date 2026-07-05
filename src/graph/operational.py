"""
Operational subgraph — the agent ReAct loop as a StateGraph.

Replaces the hand-written `for iteration in range(12)` loop that lived in
AgentSession.answer(). Topology ONLY: every node body is a method on
AgentSession (src/agent.py) reached through runtime context, so instance
seams (`session._run_collect`, `session.confirmation_handler`) and module
seams keep resolving exactly where they always did.

    START → prepare → agent_step ──(no tool calls)──→ finalize_answer → END
                        │  ▲
              (tool calls)  └──────────(else)──────────┐
                        ▼                              │
                    exec_tools ──(write_cancelled)──→ finalize_cancelled → END
                        └────────(iteration ≥ 12)───→ finalize_max_iter  → END

The max-iteration guard is an iteration counter in state + conditional edge
(NOT recursion_limit): hitting it must return the exact canned answer dict
with error="max_iterations_reached", never raise GraphRecursionError.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.graph.persistence import get_saver
from src.graph.state import GraphRunContext, OperationalState

# The hand-built loop's `max_iterations = 12`, now a topology constant.
MAX_ITERATIONS = 12


def _prepare(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return runtime.context.session._op_prepare(state, runtime.context.cache)


async def _agent_step(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.session._op_agent_step(state, runtime.context.cache)


async def _exec_tools(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.session._op_exec_tools(state, runtime.context.cache)


async def _finalize_answer(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.session._op_finalize_answer(state)


def _finalize_cancelled(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return runtime.context.session._op_finalize_cancelled(state)


def _finalize_max_iter(state: OperationalState, runtime: Runtime[GraphRunContext]):
    return runtime.context.session._op_finalize_max_iter(state)


def _route_after_agent_step(state: OperationalState) -> str:
    return "exec_tools" if state.get("has_tool_calls") else "finalize_answer"


def _route_after_exec_tools(state: OperationalState) -> str:
    if state.get("write_cancelled"):
        return "finalize_cancelled"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize_max_iter"
    return "agent_step"


def _build() -> StateGraph:
    g = StateGraph(OperationalState, context_schema=GraphRunContext)
    g.add_node("prepare", _prepare)
    g.add_node("agent_step", _agent_step)
    g.add_node("exec_tools", _exec_tools)
    g.add_node("finalize_answer", _finalize_answer)
    g.add_node("finalize_cancelled", _finalize_cancelled)
    g.add_node("finalize_max_iter", _finalize_max_iter)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "agent_step")
    g.add_conditional_edges(
        "agent_step", _route_after_agent_step,
        {"exec_tools": "exec_tools", "finalize_answer": "finalize_answer"},
    )
    g.add_conditional_edges(
        "exec_tools", _route_after_exec_tools,
        {
            "finalize_cancelled": "finalize_cancelled",
            "finalize_max_iter": "finalize_max_iter",
            "agent_step": "agent_step",
        },
    )
    g.add_edge("finalize_answer", END)
    g.add_edge("finalize_cancelled", END)
    g.add_edge("finalize_max_iter", END)
    return g


# Compiled once per live saver; the graph is instance-independent (nodes get
# their AgentSession from runtime context), so one compilation serves every
# session. Rebuilt only when the saver changes (tests repoint the sqlite path).
_compiled: tuple | None = None


def get_operational_graph():
    global _compiled
    saver = get_saver()
    if _compiled is None or _compiled[0] is not saver:
        _compiled = (saver, _build().compile(checkpointer=saver))
    return _compiled[1]
