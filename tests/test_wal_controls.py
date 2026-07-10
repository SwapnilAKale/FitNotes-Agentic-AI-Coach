"""
User-facing WAL controls: the persistent replay on/off setting gating
upload-time replay (server._maybe_replay_wal), the /settings endpoints,
and the archive-then-empty wipe (wal.wipe + /wal-wipe). All WAL/settings
paths are conftest-isolated to tmp_path.
"""

import asyncio
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import server as srv                           # noqa: E402
from src import settings, wal                  # noqa: E402


def _body(resp):
    return json.loads(resp.body)


# ── settings store ─────────────────────────────────────────────────────────────

def test_setting_default_true_when_file_missing():
    assert not os.path.exists(os.environ["SETTINGS_PATH"])
    assert settings.get_setting("wal_replay_enabled", True) is True


def test_setting_round_trip_persists_to_file():
    settings.set_setting("wal_replay_enabled", False)
    assert settings.get_setting("wal_replay_enabled", True) is False
    with open(os.environ["SETTINGS_PATH"], encoding="utf-8") as f:
        assert json.load(f) == {"wal_replay_enabled": False}
    settings.set_setting("wal_replay_enabled", True)
    assert settings.get_setting("wal_replay_enabled", False) is True


def test_setting_corrupt_file_falls_back_to_default():
    with open(os.environ["SETTINGS_PATH"], "w", encoding="utf-8") as f:
        f.write("{not json")
    assert settings.get_setting("wal_replay_enabled", True) is True


# ── replay gate ────────────────────────────────────────────────────────────────

def test_replay_runs_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(wal, "replay_writes",
                        lambda db: calls.append(db) or {"replayed": 2, "conflicts": 0, "errors": []})
    settings.set_setting("wal_replay_enabled", True)
    result = asyncio.run(srv._maybe_replay_wal())
    assert calls == [srv.DB_PATH]
    assert result["replayed"] == 2
    assert "skipped" not in result


def test_replay_skipped_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(wal, "replay_writes", lambda db: calls.append(db))
    settings.set_setting("wal_replay_enabled", False)
    result = asyncio.run(srv._maybe_replay_wal())
    assert calls == []                                   # negative: never invoked
    assert result == {"replayed": 0, "conflicts": 0, "skipped": True}


def test_skip_leaves_entries_pending_for_a_later_upload():
    wal.append_write("execute_staged_workout", {"exercise_id": 1, "date": "2026-07-01",
                                                "sets": [{"metric_weight": 10, "reps": 5}]})
    settings.set_setting("wal_replay_enabled", False)
    asyncio.run(srv._maybe_replay_wal())
    records = wal.get_records()
    assert len(records) == 1
    assert records[0]["status"] == "pending"             # untouched, replayable later


# ── wipe ───────────────────────────────────────────────────────────────────────

def test_wipe_archives_then_empties():
    id1 = wal.append_write("execute_staged_workout", {"exercise_id": 1, "sets": []})
    id2 = wal.append_write("execute_staged_workout", {"exercise_id": 2, "sets": []})
    result = wal.wipe()
    assert result["wiped"] == 2
    assert wal.get_records() == []
    with open(result["archive"], encoding="utf-8") as f:
        archived = json.load(f)
    assert [r["id"] for r in archived] == [id1, id2]     # exact prior records
    assert os.path.dirname(result["archive"]) == os.path.dirname(wal.WAL_PATH)


def test_wipe_on_empty_is_noop_without_archive():
    result = wal.wipe()
    assert result == {"wiped": 0, "archive": None}


# ── endpoints ──────────────────────────────────────────────────────────────────

def test_get_settings_reports_state_and_pending():
    wal.append_write("execute_staged_workout", {"exercise_id": 1, "sets": []})
    body = _body(asyncio.run(srv.get_settings()))
    assert body["wal_replay_enabled"] is True
    assert body["wal_pending"] == 1


def test_post_settings_persists_and_reflects():
    body = _body(asyncio.run(srv.post_settings(srv.SettingsRequest(wal_replay_enabled=False))))
    assert body == {"wal_replay_enabled": False}
    assert settings.get_setting("wal_replay_enabled", True) is False
    assert _body(asyncio.run(srv.get_settings()))["wal_replay_enabled"] is False


def test_wal_wipe_endpoint_empties_journal():
    wal.append_write("execute_staged_workout", {"exercise_id": 1, "sets": []})
    body = _body(asyncio.run(srv.wal_wipe()))
    assert body["wiped"] == 1
    assert wal.get_records() == []
