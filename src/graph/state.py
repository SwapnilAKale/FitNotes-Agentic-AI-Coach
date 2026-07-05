"""
Graph state schemas + the non-persisted run context.

HARD CONSTRAINT (checkpointer seam): the ~366KB analytical package must NEVER
enter persisted graph state. State schemas below carry only small JSON-safe
values; the package (and every package-derived artifact: grounding context,
cited values, live google.genai Content objects with thought_signature) rides
in RunCache, which is delivered per-invocation through LangGraph's
context_schema. Runtime context is injected from config at invoke time and is
never serialized by the checkpointer — excluding the package by construction,
not by trimming.

On resume the cache starts empty and the package rebuilds free from the saved
params (pure Python, re-validated — every A–G/G6 invariant re-applies), exactly
as the hand-built checkpoint documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict


class CoordinatorState(TypedDict, total=False):
    """Parent graph: replaces the body of Coordinator._route_fresh."""

    question: str            # mutated by entry_boundary (/log prefix strip, tail peel)
    log_carry: bool          # snapshot from route(); consumed by entry_boundary
    flow_turns: list         # snapshot from route(); extended on a carry turn
    log_boundary: bool
    trailing_note: bool
    fallback_write: bool
    params: Optional[dict]   # classify output (or deterministic write params)
    route: str               # "analytical" | "operational" | terminal short-circuits
    answer: str
    flagged_claims: list
    error: Optional[str]
    result: Optional[dict]   # the contracted route() return dict (terminal nodes)


class AnalyticalState(TypedDict, total=False):
    """Analytical subgraph: replaces the body of Coordinator._run_analytical.

    The package itself is NOT here — see RunCache. `draft`/`answer` are the
    same small strings the hand-built checkpoint already persisted verbatim.
    """

    question: str
    params: dict
    scoped_question: str
    exercise_names: Optional[list]
    muscle_groups: Optional[list]
    unresolved_names: Optional[list]
    effective_names: Optional[list]
    draft: Optional[str]
    answer: str
    flagged: list
    resume_completed_stage: Optional[str]   # from the checkpoint slot
    resume_draft: Optional[str]
    early_answer: Optional[str]             # disambiguation short-circuit


class OperationalState(TypedDict, total=False):
    """Operational subgraph: replaces the AgentSession.answer() loop body.

    `messages` holds OpenAI-style dicts only (JSON-safe); the live
    google.genai contents (thought_signature) live in RunCache.
    """

    question: str
    messages: list
    new_exchange_start: int
    iteration: int
    tool_calls_made: int
    execute_attempted: bool
    write_cancelled: bool
    has_tool_calls: bool
    last_text: str            # combined_text of the last agent step
    result: Optional[dict]    # the contracted answer() return dict


class RunCache:
    """Mutable per-invocation carrier for everything the checkpointer must
    never see. Created fresh by each facade call; empty after a resume.
    """

    def __init__(self) -> None:
        # Analytical lane
        self.pkg: Optional[dict] = None            # the analytical package
        self.gctx: Optional[dict] = None           # grounding context
        self.memories: Any = None
        self.research: Any = None                  # always None on this lane
        self.custom_query: Optional[dict] = None
        self.conversation_context: Optional[list] = None
        # Operational lane
        self.gemini_contents: Optional[list] = None   # live types.Content
        self.effective_prompt: Optional[str] = None

    async def ensure_package(self, coordinator, params: dict) -> dict:
        """Rebuild-on-resume: return the package, building it from params via
        the coordinator's package stage if the cache is empty (fresh process
        resume). The build path goes through the same module seam
        (coordinator-module ``prepare_analysis_package``) as a fresh run.
        """
        if self.pkg is None:
            self.pkg = await coordinator._build_package_from_params(params)
        return self.pkg


@dataclass
class GraphRunContext:
    """context_schema payload — injected at ainvoke() time, never checkpointed."""

    coordinator: Any = None    # Coordinator instance (parent + analytical graphs)
    session: Any = None        # AgentSession instance (operational graph)
    cache: RunCache = field(default_factory=RunCache)
