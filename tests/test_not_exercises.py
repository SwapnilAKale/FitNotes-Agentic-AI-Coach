"""
"Not an exercise" — ONE list, read system-wide.

WHY THIS EXISTS. Some people log journal markers as exercises ("Morning",
"Society") to note when or where they trained. Those are not training and must
stay out of every volume, session and muscle number.

That used to be `EXCLUDED_CATEGORY_IDS = (10, 11, 12)` in fetch.py — ONE user's
category ids, hardcoded, and re-copied into eight other places including five
test files. Another person's database numbers its categories differently, so the
skip silently stopped working for them: the same single-user assumption that let
newly-trained stock exercises slip past reconciliation entirely.

There is no automatic test available. "Morning" logs reps=1, which is
indistinguishable from a real bodyweight set. A human says so once, and the
answer is remembered by name.

PER-USER BY NATURE. A journal habit is not a fact about anatomy, so the list is
gitignored like pending_review.csv, while exercises.csv / aliases.csv /
exercise_muscle.csv are the universal graph everyone shares.
"""

import os
import subprocess
from pathlib import Path

import pytest

from src import ontology_reconcile as rec

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_DIR", str(tmp_path))
    return tmp_path


# ── The list itself ───────────────────────────────────────────────────────────

def test_absent_list_means_nothing_is_dismissed(store_dir):
    """Degrades OPEN, never closed: a missing list must show the user too much
    rather than silently hide their training."""
    assert rec.not_exercise_names() == frozenset()


def test_dismissing_is_idempotent(store_dir):
    assert rec.dismiss_as_not_exercise(["Morning"]) == 1
    assert rec.dismiss_as_not_exercise(["Morning"]) == 0
    assert rec.dismiss_as_not_exercise(["morning"]) == 0, "match is case-insensitive"
    assert "morning" in rec.not_exercise_names()


def test_dismissing_preserves_earlier_entries(store_dir):
    rec.dismiss_as_not_exercise(["Morning"])
    rec.dismiss_as_not_exercise(["Society"])
    assert rec.not_exercise_names() == frozenset({"morning", "society"})


def test_a_damaged_list_does_not_raise(store_dir):
    (store_dir / rec.NOT_EXERCISES_FILE).write_text("this is not,csv\x00\n",
                                                    encoding="utf-8")
    assert isinstance(rec.not_exercise_names(), frozenset)


# ── It reaches the rest of the system ─────────────────────────────────────────

def test_dismissed_names_leave_the_logged_set(store_dir):
    """logged_exercise_names is what reconciliation and the aliasing test both
    read, so a dismissal has to remove the name there."""
    before = rec.logged_exercise_names()
    if "Morning" not in before:
        pytest.skip("no journal markers in this database")
    rec.dismiss_as_not_exercise(["Morning"])
    assert "Morning" not in rec.logged_exercise_names()


def test_the_sql_clause_is_parameterised(store_dir):
    """Dismissed names are USER DATA and must never be interpolated into SQL."""
    from src.data_agent.fetch import excluded_names_clause
    rec.dismiss_as_not_exercise(["Robert'); DROP TABLE training_log;--"])
    clause, params = excluded_names_clause("e")
    assert "DROP TABLE" not in clause.upper(), "the name reached the SQL text"
    assert "?" in clause
    # Names are lowercased for a case-insensitive compare; the payload must
    # travel as a BOUND PARAMETER, never as SQL.
    assert any("drop table" in p.lower() for p in params)


def test_an_empty_list_produces_no_clause(store_dir):
    """No dismissals must mean no WHERE fragment at all — not a clause that
    accidentally filters everything."""
    from src.data_agent.fetch import excluded_names_clause
    assert excluded_names_clause("e") == ("", [])


# ── The promote path ──────────────────────────────────────────────────────────

def test_not_exercise_decision_leaves_the_queue_for_good(store_dir, monkeypatch):
    """Asked once, ever. A skipped row would sit in the queue forever."""
    from src import ontology as ont_mod
    rec.save_pending([{"db_exercise_name": "Morning", "decision": "not_exercise",
                       "approved": "y"}])
    ont_mod.clear_cache()
    result = rec.promote(ont_mod.load_ontology(force=True))
    assert [p["as"] for p in result.promoted] == ["not_exercise"]
    assert rec.load_pending() == []
    assert "morning" in rec.not_exercise_names()


def test_a_dismissal_never_touches_the_shared_graph(store_dir):
    """The graph is universal; a journal habit is not. Nothing may be written to
    exercises.csv / aliases.csv / exercise_muscle.csv."""
    from src import ontology as ont_mod
    rec.save_pending([{"db_exercise_name": "Society", "decision": "not_exercise",
                       "approved": "y"}])
    ont_mod.clear_cache()
    rec.promote(ont_mod.load_ontology(force=True))
    for graph_file in ("exercises.csv", "aliases.csv", "exercise_muscle.csv"):
        assert not (store_dir / graph_file).exists(), graph_file


# ── No module may re-hardcode the old rule ────────────────────────────────────

def test_no_source_file_hardcodes_the_category_ids():
    """Nine copies collapsed to one. A new copy is how this broke the first
    time — the ids are one user's database, not a universal fact."""
    needle = "category_id" + " NOT IN (10"      # split so this file is not a hit
    offenders = []
    for folder in ("src", "tests", "scripts"):
        for path in (_ROOT / folder).rglob("*.py"):
            # review_pending.py owns the ONE-TIME migration of the old rule into
            # the new list — that is its whole purpose, not a stray copy.
            if path.name in ("review_pending.py", Path(__file__).name):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if needle in line:
                    offenders.append(f"{path.relative_to(_ROOT)}: {line.strip()}")
    assert not offenders, "re-hardcoded category exclusion:\n" + "\n".join(offenders)


def test_the_list_is_user_local_not_committed():
    """It must never enter the shared repo — one person's journal habit is not
    reference data for everyone."""
    rel = f"ontology/{rec.NOT_EXERCISES_FILE}"
    out = subprocess.run(["git", "check-ignore", rel], cwd=_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"{rel} is NOT gitignored"
