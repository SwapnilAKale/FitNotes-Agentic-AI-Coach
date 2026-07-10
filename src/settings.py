"""
src/settings.py — persistent user preferences.

A single JSON object in data/settings.json (override with SETTINGS_PATH).
Currently holds:
  wal_replay_enabled (bool, default True) — whether saved chat-logged
  workouts are applied onto a freshly uploaded FitNotes backup. When False,
  /upload skips wal.replay_writes and pending journal entries stay pending.

Writes are atomic (tmp + os.replace) with the same Windows/OneDrive
PermissionError retry the WAL uses — the repo lives under OneDrive and the
destination can be transiently locked.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "wal_replay_enabled": True,
}

_lock = threading.Lock()


def _path() -> str:
    return os.environ.get("SETTINGS_PATH", "data/settings.json")


def _load() -> dict:
    path = _path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.error("[settings] %s is unreadable (%s) — using defaults", path, exc)
        return {}


def _save(data: dict) -> None:
    path = _path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def get_setting(key: str, default=None):
    """Read one setting; falls back to the declared default, then `default`."""
    with _lock:
        data = _load()
    if key in data:
        return data[key]
    return _DEFAULTS.get(key, default)


def set_setting(key: str, value) -> None:
    with _lock:
        data = _load()
        data[key] = value
        _save(data)
    logger.info("[settings] %s = %r", key, value)
