"""
src/data_agent/__init__.py
Public API — thin facades over fetch / process / validate.

    from src.data_agent import collect, prepare_analysis_package, query
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from .fetch    import fetch_data, load_user_context, query, sanitize_sql
from .process  import process_data, trim_package, _get_aggregation_level
from .validate import validate, Violation  # noqa: F401  (Violation re-exported)

__all__ = [
    "collect", "prepare_analysis_package", "query", "sanitize_sql",
    "DataAgentIntegrityError",
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


def _derive_scope(exercise_names: Optional[list],
                  muscle_groups:  Optional[list]) -> str:
    """Derive trim profile scope from query filters."""
    if exercise_names and len(exercise_names) <= 3:
        return "focused"
    if muscle_groups:
        return "group"
    return "broad"


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
) -> dict:
    """
    Wrapper over collect() for the analytical pipeline.
    Strips and trims raw data to hit 100-300 KB before sending to the
    Analysis Agent. All pre-computed analytics are preserved. Only
    raw series and full enumerations are trimmed.
    """
    package = collect(
        query_period_days=query_period_days,
        end_date_str=end_date_str,
        start_date_str=start_date_str,
        muscle_groups=muscle_groups,
        exercise_names=exercise_names,
        aggregation_level=aggregation_level,
        include_phase2=include_phase2,
    )
    scope   = _derive_scope(exercise_names, muscle_groups)
    trimmed = trim_package(package, scope=scope)
    _report_violations(validate(trimmed), "prepare_analysis_package")
    return trimmed
