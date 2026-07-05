"""
Analytical subgraph — the Data Agent → Analysis Agent pipeline as a linear
node chain. Replaces the hand-sequenced body of Coordinator._run_analytical.

    START → resolve_scope ──(disambiguation)──→ END
                 │
                 ▼
          build_package → draft → ground → coverage → display_fidelity → END

One node per stage; plain edges between them. The Data Agent's
fetch/process/validate internals stay a SINGLE pure-Python call inside
build_package (their A–G/G6 invariants are enforced inside
prepare_analysis_package — never decomposed into graph nodes). The coverage
retry-once re-runs analyze+ground INSIDE the coverage node, exactly as the
hand-built code did, keeping the chain linear.

The ~366KB package and every package-derived artifact live in the
non-persisted RunCache (context), never in state: persisted checkpoints
carry only params + the verbatim draft/answer strings — a resume re-enters
at START and build_package rebuilds the package free from params.

Node bodies are Coordinator._stage_* methods (src/coordinator.py) so every
existing monkeypatch seam (module-global prepare_analysis_package,
analysis_agent.analyze/ground_check, instance _coverage_check,
_call_with_per_minute_retry) keeps resolving where it always did.

There is NO edge from any node here to the operational lane: an exception
aborts the run and propagates to the dispatcher's clean-fail handler —
an exception is never a routing signal.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.graph.persistence import get_saver
from src.graph.state import AnalyticalState, GraphRunContext


async def _resolve_scope(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_resolve_scope(state)


async def _build_package(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_build_package(state, runtime.context.cache)


async def _draft(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_draft(state, runtime.context.cache)


async def _ground(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_ground(state, runtime.context.cache)


async def _coverage(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_coverage(state, runtime.context.cache)


async def _display_fidelity(state: AnalyticalState, runtime: Runtime[GraphRunContext]):
    return await runtime.context.coordinator._stage_display_fidelity(state, runtime.context.cache)


def _route_after_resolve(state: AnalyticalState) -> str:
    # Disambiguation early-exit: the question ends with the candidate prompt,
    # bypassing the pipeline AND record_external_exchange (as the hand-built
    # early return did).
    return END if state.get("early_answer") is not None else "build_package"


def _build() -> StateGraph:
    g = StateGraph(AnalyticalState, context_schema=GraphRunContext)
    g.add_node("resolve_scope", _resolve_scope)
    g.add_node("build_package", _build_package)
    g.add_node("draft", _draft)
    g.add_node("ground", _ground)
    g.add_node("coverage", _coverage)
    g.add_node("display_fidelity", _display_fidelity)
    g.add_edge(START, "resolve_scope")
    g.add_conditional_edges(
        "resolve_scope", _route_after_resolve,
        {END: END, "build_package": "build_package"},
    )
    g.add_edge("build_package", "draft")
    g.add_edge("draft", "ground")
    g.add_edge("ground", "coverage")
    g.add_edge("coverage", "display_fidelity")
    g.add_edge("display_fidelity", END)
    return g


_compiled: tuple | None = None


def get_analytical_graph():
    global _compiled
    saver = get_saver()
    if _compiled is None or _compiled[0] is not saver:
        _compiled = (saver, _build().compile(checkpointer=saver))
    return _compiled[1]
