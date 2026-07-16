"""
Parent coordinator graph — classify-and-dispatch as a router topology.
Replaces the body of Coordinator._route_fresh.

    START → entry_boundary ──(filler)──────────────────────→ END
                │  │
                │  └─(write-intent / /log boundary)─→ dispatch_operational
                ▼                                          │
             classify ──(parse-fail / out_of_scope)→ END   │
                │  │                                       │
                │  └─(operational)─→ dispatch_operational ─┤
                ▼                                          ▼
        dispatch_analytical ─────────────────────────→ finalize → END

The deterministic write-intent pre-guard and /log boundary are evaluated in
entry_boundary and routed by a conditional edge BEFORE the classify node —
a real write short-circuits to operational without spending the classify
LLM call, exactly as the hand-rolled statement order did, now as structure.

INVARIANT (topology): there is NO edge from dispatch_analytical to
dispatch_operational. The analytical clean-fail handling lives inside the
dispatch_analytical node body; an exception can abort the run but can never
switch lanes — an exception is not a routing signal.

Terminal short-circuits (filler / unparseable / out_of_scope) route
straight to END without passing finalize, so they never touch conversation
history (the hand-built behavior).

Node bodies are Coordinator._node_* methods (src/coordinator.py) resolved
through runtime context at call time, keeping every instance-method and
module-global monkeypatch seam live.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.graph.persistence import get_saver
from src.graph.state import CoordinatorState, GraphRunContext


def _entry_boundary(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return runtime.context.coordinator._node_entry_boundary(state)


async def _classify_node(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._node_classify(state)


async def _dispatch_analytical(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._node_dispatch_analytical(state)


async def _dispatch_operational(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._node_dispatch_operational(state)


async def _dispatch_recall(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._node_recall_dispatch(state)


async def _dispatch_decomposed(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._node_dispatch_decomposed(state)


def _finalize(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return runtime.context.coordinator._node_finalize_turn(state)


def _route_after_entry(state: CoordinatorState) -> str:
    if state.get("result") is not None:          # filler short-circuit
        return END
    params = state.get("params")
    if params is not None:                       # pre-guard or decomposition resume
        # A disambiguation-resume carries the FULL turn (requests intact). If it
        # is still a mixed-lane multi-chunk turn, it must re-enter the decomposed
        # dispatch — dispatching by the flat `route` alone would silently
        # collapse it to one lane and drop the sibling chunk(s). Same test as
        # _route_after_classify.
        reqs = params.get("requests") or []
        if len(reqs) >= 2 and len({c.get("lane") for c in reqs}) >= 2:
            return "dispatch_decomposed"
        # Dispatch by the params' own route: classify never runs. The /log and
        # write-intent synthetic dicts all carry "operational" (unchanged
        # behavior); a pre-seeded analytical resume dispatches analytical.
        if params.get("route") == "analytical":
            return "dispatch_analytical"
        return "dispatch_operational"
    return "classify"


def _route_after_classify(state: CoordinatorState) -> str:
    params = state.get("params") or {}
    if params.get("_parse_failed"):
        return "unparseable"
    # Stage 3: a multi-chunk message with MIXED lanes executes per-chunk and
    # merges in index order. All-analytical multi-chunk deliberately stays a
    # single run (live-proven good, and one pipeline is cheaper than two).
    reqs = params.get("requests") or []
    if len(reqs) >= 2 and len({c.get("lane") for c in reqs}) >= 2:
        return "dispatch_decomposed"
    route = params.get("route", "analytical")
    if route == "out_of_scope":
        return "out_of_scope"
    if route == "recall":
        return "dispatch_recall"
    if route == "analytical":
        return "dispatch_analytical"
    return "dispatch_operational"


def _unparseable(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return {"result": runtime.context.coordinator._unparseable_response()}


def _out_of_scope(state: CoordinatorState, runtime: Runtime[GraphRunContext]):
    return {"result": runtime.context.coordinator._out_of_scope_response()}


def _build() -> StateGraph:
    g = StateGraph(CoordinatorState, context_schema=GraphRunContext)
    g.add_node("entry_boundary", _entry_boundary)
    g.add_node("classify", _classify_node)
    g.add_node("unparseable", _unparseable)
    g.add_node("out_of_scope", _out_of_scope)
    g.add_node("dispatch_analytical", _dispatch_analytical)
    g.add_node("dispatch_operational", _dispatch_operational)
    g.add_node("dispatch_recall", _dispatch_recall)
    g.add_node("dispatch_decomposed", _dispatch_decomposed)
    g.add_node("finalize", _finalize)
    g.add_edge(START, "entry_boundary")
    g.add_conditional_edges(
        "entry_boundary", _route_after_entry,
        {END: END, "dispatch_operational": "dispatch_operational",
         "dispatch_analytical": "dispatch_analytical",
         # A disambiguation-resume carries the full mixed-lane turn and must be
         # able to re-enter the decomposed dispatch (not collapse to one lane).
         "dispatch_decomposed": "dispatch_decomposed", "classify": "classify"},
    )
    g.add_conditional_edges(
        "classify", _route_after_classify,
        {
            "unparseable": "unparseable",
            "out_of_scope": "out_of_scope",
            "dispatch_analytical": "dispatch_analytical",
            "dispatch_operational": "dispatch_operational",
            "dispatch_recall": "dispatch_recall",
            "dispatch_decomposed": "dispatch_decomposed",
        },
    )
    g.add_edge("unparseable", END)
    g.add_edge("out_of_scope", END)
    # NO analytical → operational edge exists (see module docstring).
    g.add_edge("dispatch_analytical", "finalize")
    g.add_edge("dispatch_operational", "finalize")
    g.add_edge("dispatch_recall", "finalize")
    g.add_edge("dispatch_decomposed", "finalize")
    g.add_edge("finalize", END)
    return g


_compiled: tuple | None = None


def get_coordinator_graph():
    global _compiled
    saver = get_saver()
    if _compiled is None or _compiled[0] is not saver:
        _compiled = (saver, _build().compile(checkpointer=saver))
    return _compiled[1]
