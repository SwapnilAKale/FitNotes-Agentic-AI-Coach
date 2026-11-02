#!/usr/bin/env python3
"""
Make a consistent copy of the visitor-analytics database (src/analytics.py).

Runs INSIDE the app container; scripts/export_analytics.py calls it over SSH:

    docker compose exec -T app python scripts/analytics_snapshot.py /tmp/snap.sqlite

It uses SQLite's backup API, never a raw file copy: the live database is in
WAL mode and is being written to, so copying the file could catch it half-way.
The copy is a single self-contained file (no -wal/-shm), integrity-checked.
Prints one line of JSON: path, bytes, sha256, integrity, row counts.

Exit codes: 0 ok, 2 no analytics database yet, 1 anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analytics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", help="where to write the snapshot (e.g. /tmp/snap.sqlite)")
    ap.add_argument("--source", default=None,
                    help="analytics database (default: ANALYTICS_DB_PATH or "
                         "data/analytics/analytics.sqlite)")
    args = ap.parse_args(argv)
    try:
        info = analytics.snapshot(args.dest, args.source)
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(info))
    return 0 if info["integrity"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
