"""
Step C, Part 1: the analytical custom-SQL lane refuses weight/volume aggregates.

Custom SQL is for counts / dates / gaps / streaks / patterns. Weight and volume
have authoritative package fields (muscle_group_summary, pr/pr_period,
progression, e1rm_*). A raw SUM(metric_weight*reps) blends kg-native and lbs
rows, so the executor REFUSES weight/volume aggregates before running them.
Per-row metric_weight SELECT (no aggregate) is still allowed (existing caveat).

No Gemini — SQL strings are fed to the guard directly.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("FITNOTES_DB_PATH",  "data/FitNotes_Backup.fitnotes")
os.environ.setdefault("USER_CONTEXT_PATH", "data/user_context.json")

from src.data_agent.fetch import _weight_aggregate_reason, query  # noqa: E402


# ── Detector: weight/volume aggregates are refused ───────────────────────────

@pytest.mark.parametrize("sql", [
    "SELECT SUM(metric_weight*reps) FROM training_log",
    "SELECT SUM(metric_weight * reps) FROM training_log",
    "SELECT AVG(metric_weight) FROM training_log",
    "SELECT SUM(metric_weight * 2.2046 * reps) AS volume FROM training_log",
    "select  sum( metric_weight )  from training_log",          # whitespace/casing
    "SELECT MAX(metric_weight) FROM training_log",
    "SELECT MIN(metric_weight) FROM training_log",
    "SELECT TOTAL(metric_weight * reps) FROM training_log",
    # aliased WITH AS: vol is bound to a metric_weight expression then aggregated
    "SELECT SUM(vol) FROM (SELECT metric_weight * reps AS vol FROM training_log) t",
    # #4 fix — alias WITHOUT AS bypass: the projection still names metric_weight,
    # so the invariant rule (metric_weight + blend aggregate) catches it.
    "SELECT SUM(v) FROM (SELECT metric_weight v FROM training_log) t",
    "SELECT AVG(w) FROM (SELECT metric_weight w FROM training_log) t",
    "SELECT MAX(x) FROM (SELECT metric_weight * reps x FROM training_log) t",
    # #4 fix — GROUP_CONCAT serializes raw mixed-unit kg+lbs values (string blend)
    "SELECT GROUP_CONCAT(metric_weight) FROM training_log",
    "SELECT group_concat(metric_weight, ',') FROM training_log",
])
def test_weight_volume_aggregates_refused(sql):
    assert _weight_aggregate_reason(sql) is not None


# ── Detector: counts / dates / gaps / per-row select are allowed ─────────────

@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) AS n, date FROM training_log GROUP BY date",
    "SELECT MAX(date) FROM training_log",
    "SELECT date FROM training_log ORDER BY date",                # streak/gap input
    "SELECT COUNT(*) FROM training_log WHERE metric_weight > 0",  # weight in filter, COUNT exempt
    "SELECT date, SUM(reps) FROM training_log GROUP BY date",     # reps aggregate, no metric_weight
    "SELECT metric_weight, reps FROM training_log LIMIT 5",       # per-row select (allowed)
    # COUNT alongside a per-row metric_weight column — granularity decision:
    # COUNT is exempt (counts rows, never blends weights), so "how many sets +
    # their weights" still works. Only SUM/AVG/MIN/MAX/TOTAL/GROUP_CONCAT refuse.
    "SELECT COUNT(*) AS n, metric_weight FROM training_log GROUP BY metric_weight",
])
def test_non_weight_queries_allowed(sql):
    assert _weight_aggregate_reason(sql) is None


# ── query() returns a structured refusal, never executes ─────────────────────

def test_query_refuses_weight_aggregate_structured():
    out = query("SELECT SUM(metric_weight * reps) FROM training_log")
    assert out["refused"] is True
    assert "muscle_group_summary" in out["reason"]
    assert out["rows"] == [] and out["row_count"] == 0
    assert out["warning"].startswith("REFUSED:")


def test_query_refuses_alias_without_as_bypass():
    # #4 regression: this previously executed and returned a blended kg+lbs SUM
    # (~155k). It must now refuse before running — no rows leak.
    out = query("SELECT SUM(v) FROM (SELECT metric_weight v FROM training_log) t")
    assert out["refused"] is True
    assert out["rows"] == [] and out["row_count"] == 0
    assert out["warning"].startswith("REFUSED:")


def test_query_refuses_group_concat_weight():
    out = query("SELECT GROUP_CONCAT(metric_weight) FROM training_log")
    assert out["refused"] is True
    assert out["rows"] == [] and out["row_count"] == 0


def test_query_executes_counts_normally():
    out = query("SELECT COUNT(*) AS n FROM training_log")
    assert not out.get("refused")
    assert out["row_count"] == 1
    assert out["rows"][0]["n"] > 0


def test_query_allows_per_row_metric_weight_with_caveat():
    out = query("SELECT metric_weight, reps FROM training_log LIMIT 3")
    assert not out.get("refused")
    assert out["row_count"] == 3
    # per-row select keeps the existing plates-only typed_weight caveat
    assert "typed_weight" in out["columns"]
    assert "PARTIAL CONVERSION APPLIED" in out["warning"]


# ── Custom-SQL prompt steers away from weight aggregation ────────────────────

def test_custom_sql_prompt_forbids_weight_aggregation(monkeypatch):
    """The coordinator's custom-SQL prompt must tell the model NOT to aggregate
    weight / compute volume. (The shared _SQL_SYSTEM is left untouched because
    the operational text-to-SQL pipeline legitimately answers weight questions.)"""
    import asyncio
    from types import SimpleNamespace
    from src import coordinator as coordinator_mod
    from src.coordinator import Coordinator

    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)

    seen = {}

    def fake_generate_sql(prompt_question, schema):
        seen["prompt"] = prompt_question
        return "SELECT COUNT(*) FROM training_log"

    def fake_query(sql):
        return {"rows": [{"n": 1}], "columns": ["n"], "row_count": 1, "warning": ""}

    import src.llm as llm_mod
    import src.data_agent as da_mod
    monkeypatch.setattr(llm_mod, "generate_sql", fake_generate_sql)
    monkeypatch.setattr(da_mod, "query", fake_query)

    asyncio.run(c._generate_custom_sql("how many gaps", "session gaps"))

    p = seen["prompt"].lower()
    assert "counts" in p and "dates" in p and "gaps" in p
    assert "metric_weight" in p
    assert "volume" in p


# ── Refusal degrades cleanly: _generate_custom_sql returns None ──────────────

def test_refusal_degrades_to_none(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from src import coordinator as coordinator_mod
    from src.coordinator import Coordinator

    monkeypatch.setattr(coordinator_mod.genai, "Client",
                        lambda api_key=None: SimpleNamespace())
    c = Coordinator(agent_session=None)

    import src.llm as llm_mod
    import src.data_agent as da_mod
    monkeypatch.setattr(llm_mod, "generate_sql",
                        lambda pq, s: "SELECT SUM(metric_weight*reps) FROM training_log")
    # Route through the REAL query() so the guard fires and returns a refusal.
    monkeypatch.setattr(da_mod, "query", query)

    result = asyncio.run(c._generate_custom_sql("total volume", "total volume by group"))
    assert result is None        # clean fallback to package, no crash, no rows
