"""
Upload-time reconciliation must behave identically on BOTH paths that replace
the DB file: /upload and /upload/confirm (the warned-then-confirmed path).

/upload/confirm previously computed no new-exercise diff at all, so a warned
upload silently skipped the notice AND the ontology check. A fix at a shared
seam has to land on every interface, so this file pins that both endpoints
compute the diff and run the same helper.
"""

import inspect
import shutil
from pathlib import Path

import pytest

import server
from src import ontology as ont_mod


ENDPOINTS = ("upload_db", "upload_confirm")

_REAL_STORE = Path(__file__).resolve().parent.parent / "ontology"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """A WRITABLE COPY of the real store.

    _reconcile_new_exercises writes: it appends auto-aliases and merges rows into
    the review queue. Pointed at the real ontology/ these tests would edit the
    user's actual queue — an earlier version of this file did exactly that and
    left a junk row in it. The graph must be real (so resolution behaves), the
    writes must not be.
    """
    dest = tmp_path / "ontology"
    shutil.copytree(_REAL_STORE, dest)
    monkeypatch.setenv("ONTOLOGY_DIR", str(dest))
    ont_mod.clear_cache()
    yield dest
    ont_mod.clear_cache()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_computes_the_new_exercise_diff(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "_get_exercise_names(DB_PATH)" in src, endpoint
    assert "new_exercises" in src, endpoint


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_runs_ontology_reconciliation(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "_reconcile_new_exercises(new_exercises)" in src, endpoint


def test_the_seam_and_the_aliasing_test_share_one_definition():
    """They held separate opinions about what needs mapping, and that is the
    whole reason a logged exercise could sit unmapped AND unqueued. One
    function, read by both."""
    from src import ontology_reconcile as rec
    assert callable(rec.logged_exercise_names)
    assert "logged_exercise_names" in inspect.getsource(
        server._reconcile_new_exercises)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_passes_new_exercises_to_reload(endpoint):
    src = inspect.getsource(getattr(server, endpoint))
    assert "reload_db(new_exercises=new_exercises)" in src, endpoint


# ── The helper itself ─────────────────────────────────────────────────────────

# ── What earns a mapping: being TRAINED ───────────────────────────────────────
#
# Reconciliation used to take its candidates from the argument — the diff of the
# exercise TABLE between uploads. That proxy is wrong in both directions:
#
#   under-queues: a FitNotes stock exercise has been in the table since install,
#                 so training it for the first time triggered nothing and its
#                 sets went uncounted. `Incline Barbell Bench Press` reached the
#                 graph only after 18 sets and a manual sweep.
#   over-queues:  a first upload has no prior table to diff, so every defined
#                 exercise looks new — 66 of them never trained, all queued for
#                 a review the user never asked for.
#
# The candidate set is now logged_exercise_names(), the SAME definition
# test_every_logged_exercise_is_aliased reads, so the seam and the test cannot
# drift apart again — which is exactly how this survived a whole arc.

def test_reconciliation_reads_logged_exercises_not_its_argument(isolated_store):
    """The root fix. The argument no longer decides anything."""
    from_nothing = server._reconcile_new_exercises([])
    from_junk = server._reconcile_new_exercises(["Not A Real Exercise At All"])
    assert from_nothing == from_junk
    assert from_nothing["already_known"], \
        "an empty argument must still reconcile everything logged"


def test_a_trained_stock_exercise_is_seen(isolated_store):
    """`Deadlift` is a FitNotes default that has been in the table forever — the
    table diff would never surface it. Being TRAINED is what counts."""
    result = server._reconcile_new_exercises()
    assert "Deadlift" in result["already_known"]


def test_a_defined_but_untrained_exercise_is_never_queued(isolated_store):
    """THE NEGATIVE THAT MATTERS — the 66-item first-upload queue must not come
    back. Nothing without logged sets may appear anywhere in the result."""
    from src import ontology_reconcile as rec
    trained = {n.lower() for n in rec.logged_exercise_names()}
    result = server._reconcile_new_exercises()
    for bucket in ("pending", "already_known"):
        for name in result[bucket]:
            assert name.lower() in trained, \
                f"{name!r} has no logged sets and must not be reconciled"


def test_dismissed_names_are_not_reconciled(isolated_store):
    """A name the user called not-an-exercise never returns to the queue."""
    from src import ontology_reconcile as rec
    dismissed = rec.not_exercise_names()
    if not dismissed:
        pytest.skip("nothing dismissed in this environment")
    result = server._reconcile_new_exercises()
    seen = {n.lower() for n in result["pending"] + result["already_known"]}
    assert not (seen & dismissed)


def test_reconcile_failure_never_breaks_an_upload(monkeypatch, isolated_store):
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
