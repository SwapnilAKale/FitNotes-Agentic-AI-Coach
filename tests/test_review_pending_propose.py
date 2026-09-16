"""
scripts/review_pending.py --propose, driven end to end.

WHY THIS EXISTS. The first real run of --propose returned 'unsure' for all ten
queued exercises. The cause was not the model: GEMINI_API_KEY lives in .env,
server.py calls load_dotenv(), and this script — the only entry point that needs
the key — never did. Every row failed individually, was written back as
'unsure', and the run exited 0.

These tests pin the three things that would have made that impossible to miss:
the script loads .env, it refuses to run without a key, and a partial failure
exits non-zero while keeping whatever did succeed.

ISOLATION. Every test runs against a writable COPY of ontology/ and patches
load_dotenv to a no-op. The real queue must never be written by a test (an
earlier seam test did exactly that), and the real .env must not be allowed to
put the key back into an environment a test deliberately emptied.
"""

import csv
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "review_pending.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("review_pending_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    """(script module, path to the isolated queue)."""
    from src import ontology as ont_mod
    dest = tmp_path / "ontology"
    shutil.copytree(_ROOT / "ontology", dest)
    monkeypatch.setenv("ONTOLOGY_DIR", str(dest))
    ont_mod.clear_cache()
    # The copy carries whatever the REAL queue holds today. A queue that has
    # already been proposed is (correctly) skipped by --propose, which made these
    # tests pass or fail depending on the user's own review state. Undecide every
    # row in the copy so the test controls its input.
    from src import ontology_reconcile as rec
    rec.save_pending([dict(r, decision="pending", alias_of="", muscles="",
                           evidence="", sources="", approved="")
                      for r in rec.load_pending()])
    mod = _load_script()
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["review_pending.py", "--propose"])
    yield mod, dest / "pending_review.csv"
    ont_mod.clear_cache()


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _rows(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ── The key ───────────────────────────────────────────────────────────────────

def test_script_loads_dotenv(monkeypatch, tmp_path):
    """The entry point must load .env before it can propose anything."""
    from src import ontology as ont_mod
    dest = tmp_path / "ontology"
    shutil.copytree(_ROOT / "ontology", dest)
    monkeypatch.setenv("ONTOLOGY_DIR", str(dest))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ont_mod.clear_cache()
    mod = _load_script()
    called = []
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", ["review_pending.py"])
    mod.main()
    assert called, "main() never loaded .env"


def test_propose_refuses_to_run_without_a_key(env, monkeypatch, capsys):
    """Refuse BEFORE spending anything: exit 2, queue byte-identical, no call."""
    mod, queue = env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import src.ontology_propose as prop
    calls = []
    monkeypatch.setattr(prop, "propose_one",
                        lambda row, *a, **k: calls.append(row) or row)
    before = _md5(queue)

    assert mod.main() == 2
    assert _md5(queue) == before, "a keyless run must not touch the queue"
    assert calls == [], "a keyless run must not attempt a single proposal"
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY" in out
    assert "sk-" not in out and "AIza" not in out   # never echo a key value


def test_propose_refuses_without_a_search_key(env, monkeypatch, capsys):
    """The search is run by code. Without its key, refuse before any Gemini call
    is spent — never fall back to proposing without a search."""
    mod, queue = env
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-for-test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    import src.ontology_propose as prop
    calls = []
    monkeypatch.setattr(prop, "propose_one",
                        lambda row, *a, **k: calls.append(row) or row)
    before = _md5(queue)

    assert mod.main() == 2
    assert _md5(queue) == before
    assert calls == []
    out = capsys.readouterr().out
    assert "TAVILY_API_KEY" in out
    assert "dummy-for-test" not in out


# ── Partial failure ───────────────────────────────────────────────────────────

def test_partial_failure_exits_nonzero_and_keeps_successes(env, monkeypatch, capsys):
    """The run that did not fully work must say so in its exit code — and must
    not throw away the proposals that did come back."""
    mod, queue = env
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-for-test")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-for-test")
    import src.ontology_propose as prop

    names = [r["db_exercise_name"] for r in _rows(queue)]
    assert len(names) >= 2, "fixture queue needs at least two rows"
    good = names[0]

    def fake(row, ontology, client=None, sleep=None, **_):
        if row["db_exercise_name"] == good:
            return dict(row, decision="new", muscles="Upper Chest:primary",
                        evidence="found it", sources="https://example.org/x",
                        approved="")
        return prop._not_proposed(row, "search or model call failed: quota")
    monkeypatch.setattr(prop, "propose_one", fake)

    assert mod.main() == 1
    after = {r["db_exercise_name"]: r for r in _rows(queue)}
    assert after[good]["decision"] == "new"
    assert after[good]["muscles"] == "Upper Chest:primary"
    for n in names[1:]:
        assert after[n]["decision"] == "pending"
        assert after[n]["evidence"].startswith(prop.NOT_PROPOSED)
        assert after[n]["approved"] == ""
    assert f"{len(names) - 1} of {len(names)} could NOT be proposed" in capsys.readouterr().out


def test_a_clean_run_exits_zero(env, monkeypatch):
    """The negative: when every row is actually proposed, success is success."""
    mod, queue = env
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-for-test")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-for-test")
    import src.ontology_propose as prop
    monkeypatch.setattr(prop, "propose_one",
                        lambda row, ontology, client=None, sleep=None, **_: dict(
                            row, decision="unsure", evidence="UNSURE: sources disagreed",
                            approved=""))
    assert mod.main() == 0
