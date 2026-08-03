"""
Upload-time reconciliation must behave identically on BOTH paths that replace
the DB file: /upload and /upload/confirm (the warned-then-confirmed path).

/upload/confirm previously computed no new-exercise diff at all, so a warned
upload silently skipped the notice AND the ontology check. A fix at a shared
seam has to land on every interface, so this file pins that both endpoints
compute the diff and run the same helper.
"""

import inspect

import pytest

import server
from src import ontology as ont_mod


ENDPOINTS = ("upload_db", "upload_confirm")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_computes_the_new_exercise_diff(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "_get_exercise_names(DB_PATH)" in src, endpoint
    assert "new_exercises" in src, endpoint


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_runs_ontology_reconciliation(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "_reconcile_new_exercises(new_exercises)" in src, endpoint


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_passes_new_exercises_to_reload(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "reload_db(new_exercises=new_exercises)" in src, endpoint


# ── The helper itself ─────────────────────────────────────────────────────────

def test_no_new_exercises_is_a_no_op():
    assert server._reconcile_new_exercises([]) == {
        "auto_aliased": [], "pending": [], "already_known": []}


def test_known_exercise_is_reported_as_known(monkeypatch, tmp_path):
    ont_mod.clear_cache()
    result = server._reconcile_new_exercises(["Deadlift"])
    assert result["already_known"] == ["Deadlift"]
    assert result["auto_aliased"] == [] and result["pending"] == []


def test_reconcile_failure_never_breaks_an_upload(monkeypatch):
    """An upload must not fail because reference-data bookkeeping did."""
    import src.ontology_reconcile as rec
    monkeypatch.setattr(rec, "detect_new",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = server._reconcile_new_exercises(["Whatever"])
    assert "boom" in result["error"]
    assert result["auto_aliased"] == [] and result["pending"] == []


def test_exercise_context_never_raises_on_a_bad_path():
    counts, cats = server._exercise_context("no/such/file.fitnotes")
    assert counts == {} and cats == {}


def test_exercise_context_reads_counts_and_categories():
    counts, cats = server._exercise_context(server.DB_PATH)
    assert counts.get("Deadlift", 0) > 0
    assert cats.get("Deadlift") == "Back"
