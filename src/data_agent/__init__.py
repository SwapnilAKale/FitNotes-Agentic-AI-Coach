"""
src/data_agent/__init__.py
Public API — thin facades over fetch / process / validate.

    from src.data_agent import collect, prepare_analysis_package, query
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from .fetch    import fetch_data, load_user_context, query, sanitize_sql
from .process  import (
    process_data, trim_package, _get_aggregation_level,
    match_muscle_group, MUSCLE_GROUP_NAMES,
)
from .validate import validate, Violation  # noqa: F401  (Violation re-exported)

__all__ = [
    "collect", "prepare_analysis_package", "query", "sanitize_sql",
    "DataAgentIntegrityError", "match_muscle_group", "MUSCLE_GROUP_NAMES",
]

_log = logging.getLogger(__name__)


class DataAgentIntegrityError(Exception):
    """
    Raised when the built package fails one or more INTEGRITY invariant checks.
    A wrong package must never reach the Analysis Agent, so the pipeline
    hard-stops here instead of proceeding with bad data.

    Attributes:
        violations  list[Violation] — every failing integrity check
    """
    def __init__(self, violations: list):
        self.violations = violations
        ids  = ", ".join(v.invariant_id for v in violations)
        msgs = "; ".join(f"{v.invariant_id}: {v.message}" for v in violations)
        super().__init__(f"data integrity check failed [{ids}]: {msgs}")


def _derive_scope_from_package(
    package:        dict,
    exercise_names: Optional[list],
    muscle_groups:  Optional[list],
) -> tuple:
    """
    Derive scope from EFFECTIVE package contents, not classifier intent.

    Returns (scope: str, unresolved_names: list).

    Rules:
      - exercise_names filter was applied AND effective exercises <= 3 -> focused
      - muscle_groups filter was applied AND matched at least one exercise -> group
      - otherwise (no filter, or any filter that matched nothing) -> broad

    A filter that resolves to zero exercises logs loud and falls back to broad
    so the broad trim runs — an empty-filter package must never receive
    scope='focused' and skip trimming.
    """
    exercises   = package.get("exercises", [])
    n_effective = len(exercises)
    unresolved: list = []

    if exercise_names:
        effective_lower = {ex["name"].lower() for ex in exercises}
        unresolved = [n for n in exercise_names if n.lower() not in effective_lower]

        if n_effective == 0:
            _log.warning(
                "[data_agent] exercise_names filter resolved to 0 exercises in package. "
                "Unresolved: %s. Filter list was: %s. Falling back to BROAD scope.",
                ", ".join(repr(n) for n in unresolved),
                exercise_names,
            )
            return "broad", unresolved

        if n_effective <= 3:
            return "focused", unresolved

        # >3 effective exercises despite an exercise_names filter — fall through

    if muscle_groups:
        if n_effective > 0:
            return "group", unresolved
        _log.warning(
            "[data_agent] muscle_groups filter %r matched 0 exercises in package. "
            "Falling back to BROAD scope.",
            muscle_groups,
        )
        return "broad", unresolved

    return "broad", unresolved


def _display_scope(
    exercise_names: Optional[list],
    muscle_groups:  Optional[list],
    unresolved:     Optional[list],
) -> list:
    """
    Display targets for the package's `display_sets` (pure — no SQL, no LLM):
    ONE per resolved scope present — each resolved exercise AND each muscle group.

    No XOR, no precedence, no dedup. The Analysis Agent picks display-vs-analyze
    from the QUESTION (analytical questions ignore display_sets), so building a
    block for every present scope is always safe. This deliberately supersedes the
    old strict XOR, which dropped display entirely when a single-exercise question
    also carried an inferred parent-group tag (e.g. Lat Pulldown + Back).

    Returns [("exercise", name), …, ("category", group), …]; [] → no display_sets.
    """
    ex = exercise_names or []
    mg = muscle_groups or []
    resolved_ex = [n for n in ex if n not in (unresolved or [])]
    targets = [("exercise", n) for n in resolved_ex]
    targets += [("category", g) for g in mg]
    return targets


def _report_violations(violations: list, source: str) -> None:
    """
    Log soft violations (G4, G5, D3) as warnings.
    Raise DataAgentIntegrityError if any INTEGRITY violations are present —
    a wrong package must never reach the Analysis Agent.
    """
    integrity: list = []
    for v in violations:
        if v.severity == "integrity":
            _log.warning("[validate] INTEGRITY %s — %s", v.invariant_id, v.message)
            integrity.append(v)
        else:
            _log.warning("[validate] soft %s — %s", v.invariant_id, v.message)
    if integrity:
        raise DataAgentIntegrityError(integrity)


def collect(
    query_period_days: Optional[int]  = 90,
    end_date_str:      Optional[str]  = None,
    start_date_str:    Optional[str]  = None,
    muscle_groups:     Optional[list] = None,
    exercise_names:    Optional[list] = None,
    aggregation_level: Optional[str]  = None,
    include_phase2:    bool            = False,
    reps_floor:        Optional[int]  = None,
) -> dict:
    """
    Complete workout data collection. Single source of truth at any time scale.

    Args:
        query_period_days : Days to look back. None = all-time.
        end_date_str      : End date YYYY-MM-DD. Defaults to today.
        start_date_str    : Direct start date override (takes precedence).
        muscle_groups     : Filter to specific muscle groups e.g. ["Back", "Biceps"].
        exercise_names    : Filter to specific exercises.
        aggregation_level : "session" | "weekly" | "monthly". Auto if None.
        include_phase2    : Fetch full comment history for triggered exercises.
        reps_floor        : 6b — rep target for a rep-floor strength PR (None = none).
    """
    ctx      = load_user_context()
    today    = date.today()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else today
    end_str  = end_date.strftime("%Y-%m-%d")

    bundle = fetch_data(end_str)

    if start_date_str:
        start_str = start_date_str
    elif query_period_days is None:
        start_str = bundle["training_dates"][0] if bundle["training_dates"] else end_str
    else:
        start_str = (end_date - timedelta(days=query_period_days)).strftime("%Y-%m-%d")

    agg_level = aggregation_level or _get_aggregation_level(query_period_days)

    package = process_data(
        bundle, ctx, start_str, end_str, today,
        query_period_days, muscle_groups, exercise_names, agg_level, include_phase2,
        reps_floor=reps_floor,
    )
    _report_violations(validate(package), "collect")
    return package


def prepare_analysis_package(
    query_period_days: Optional[int]  = 90,
    end_date_str:      Optional[str]  = None,
    start_date_str:    Optional[str]  = None,
    muscle_groups:     Optional[list] = None,
    exercise_names:    Optional[list] = None,
    aggregation_level: Optional[str]  = None,
    include_phase2:    bool            = True,
    reps_floor:        Optional[int]  = None,
    cardio_lock:       Optional[dict] = None,
) -> dict:
    """
    Wrapper over collect() for the analytical pipeline.
    Strips and trims raw data to hit 100-300 KB before sending to the
    Analysis Agent. All pre-computed analytics are preserved. Only
    raw series and full enumerations are trimmed.

    6b parameterized PRs (both default to None = current behavior):
      reps_floor  : rep target → adds pr_repfloor to strength blocks (all-time basis).
      cardio_lock : {"field","value"} (value already unit-normalized) → adds
                    pr_cardio_locked to cardio blocks. The static pr/pr_period are
                    unchanged either way; a null value = "no qualifying set".
    """
    package = collect(
        query_period_days=query_period_days,
        end_date_str=end_date_str,
        start_date_str=start_date_str,
        muscle_groups=muscle_groups,
        exercise_names=exercise_names,
        aggregation_level=aggregation_level,
        include_phase2=include_phase2,
        reps_floor=reps_floor,
    )
    # Scope is derived from what actually survived filtering, not from
    # classifier intent. A filter that matched nothing must build as BROAD.
    scope, unresolved = _derive_scope_from_package(package, exercise_names, muscle_groups)
    trimmed = trim_package(package, scope=scope, cardio_lock=cardio_lock)
    if unresolved:
        trimmed["unresolved_exercise_names"] = unresolved

    # Approach (b): session-display is a NORMAL package field. Build a block for
    # EVERY resolved scope present (each exercise, each muscle group) and flatten
    # them into one display_sets list. The Analysis Agent decides display-vs-analyze
    # from the QUESTION — there is no routing flag, and superset display data is
    # safe (analytical questions ignore it).
    targets = _display_scope(exercise_names, muscle_groups, unresolved)
    if targets:
        from . import session_display  # local import avoids any import cycle
        flat: list = []
        for kind, target in targets:
            flat.extend(session_display.build_display_sets(kind, target))
        if flat:
            trimmed["display_sets"] = flat

    _report_violations(validate(trimmed), "prepare_analysis_package")
    return trimmed
