#!/usr/bin/env python3
"""
Container start-up: make sure the data volume holds what the app needs.

Run by deploy/entrypoint.sh before `python server.py`. Idempotent and quick.

  * Always: if the volume has no Chroma index yet, copy in the RAG index that
    the image build made from corpus/raw/ (SEED_CHROMA_DIR). Without it the
    coach cannot cite research.
  * DEMO_MODE=1 and (no workout DB in the volume, or DEMO_RESET_ON_START=1):
    empty the volume and seed it with a freshly generated SYNTHETIC database
    (scripts/make_demo_db.py) plus default state files.
  * Otherwise (your own deployment, no DB yet): leave it alone and say how to
    add one — upload your FitNotes backup from the UI's attach menu.

Never touches anything outside APP_DATA_DIR.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


DATA = Path(os.environ.get("APP_DATA_DIR", "data"))
DB = Path(os.environ.get("FITNOTES_DB_PATH", str(DATA / "FitNotes_Backup.fitnotes")))
CHROMA = Path(os.environ.get("CHROMA_DB_PATH", str(DATA / "chroma_db")))
SEED_CHROMA = Path(os.environ.get("SEED_CHROMA_DIR", "/app/seed/chroma_db"))

# Default state for a brand-new demo user — the same shapes the app creates.
DEFAULT_STATE = {
    # src/data_agent/fetch.py opens this unconditionally; empty overrides mean
    # "every exercise is typed in lbs, no bar-weight adjustments, no quirks".
    "user_context.json": {
        "unit_overrides": {"exercises_in_kg": []},
        "bar_weights_not_included": {
            "smith_machine": {"exercises": [], "bar_weight_lbs": 44.09},
            "exercise_bar_history": {},
        },
        "exercise_quirks": [],
    },
    "memory.json": {"facts": [], "last_updated": None, "summary": None,
                    "demographics": {}},
    "agent_writes.json": [],
}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _copy_seed_chroma() -> bool:
    if not (SEED_CHROMA / "chroma.sqlite3").exists():
        print(f"[prepare] no seed RAG index at {SEED_CHROMA} — skipping")
        return False
    CHROMA.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SEED_CHROMA, CHROMA, dirs_exist_ok=True)
    print(f"[prepare] RAG index copied into {CHROMA}")
    return True


def _empty_data_dir() -> None:
    for child in DATA.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def seed_demo() -> None:
    from scripts.make_demo_db import generate

    _empty_data_dir()
    info = generate(str(DB), weeks=int(os.environ.get("DEMO_WEEKS", "12")),
                    seed=int(os.environ.get("DEMO_SEED", "42")))
    print(f"[prepare] demo DB: {info['sessions']} sessions, {info['rows']} rows, "
          f"{info['start']} → {info['end']}")
    for name, value in DEFAULT_STATE.items():
        _write_json(DATA / name, value)

    # Mark every generated row as "the app's" — the same watermark an upload
    # records (server._record_app_data_watermark). The agent may then add and
    # edit its OWN entries, but a visitor cannot get it to delete or rewrite
    # the synthetic history that everyone else sees.
    conn = sqlite3.connect(DB)
    try:
        marks = {t: int(conn.execute(f"SELECT COALESCE(MAX(_id), 0) FROM [{t}]").fetchone()[0])
                 for t in ("training_log", "Goal")}
    finally:
        conn.close()
    _write_json(DATA / "settings.json",
                {"wal_replay_enabled": True, "app_data_watermark": marks})
    (DATA / "user_articles").mkdir(exist_ok=True)
    _copy_seed_chroma()


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    demo = _flag("DEMO_MODE")
    reset = _flag("DEMO_RESET_ON_START")

    if demo and (reset or not DB.exists()):
        why = "DEMO_RESET_ON_START=1" if DB.exists() else "no database in the volume"
        print(f"[prepare] DEMO_MODE: seeding synthetic demo data ({why})")
        seed_demo()
        return 0

    if not (CHROMA / "chroma.sqlite3").exists():
        _copy_seed_chroma()
    if not DB.exists():
        print(f"[prepare] no workout database at {DB}. Open the app and upload your "
              "FitNotes backup (attach menu → database), or set DEMO_MODE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
