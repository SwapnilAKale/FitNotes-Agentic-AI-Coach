#!/usr/bin/env python3
"""
Copy the public demo's visitor analytics to this PC, over SSH only.

Run it on your own computer (Windows, macOS or Linux), from the repo folder:

    python scripts/export_analytics.py

What it does, in ONE ssh connection (so a passphrase-protected key asks once):

  1. Inside the `app` container, makes a consistent snapshot of
     data/analytics/analytics.sqlite with SQLite's backup API
     (scripts/analytics_snapshot.py), not a raw copy of the live WAL database.
  2. Streams that snapshot back through the ssh connection.
  3. Deletes the snapshot from the container.

Then, on this PC, it checks the copy's SHA-256 against the one the container
reported, runs SQLite's integrity check, and saves it as
analytics_export/analytics-YYYYMMDD-HHMMSS.sqlite plus analytics_export/latest.sqlite
(the folder is gitignored). Point your private dashboard at latest.sqlite.

Every command is built as an argument list for subprocess, never one string
for a shell on this PC — Windows PowerShell 5.1 strips inner double quotes
from native-command arguments, which breaks quoted ssh commands typed there.

Defaults match the live demo; override any of them:
  --key      ~/.ssh/oracle_fitnotes   (C:\\Users\\<you>\\.ssh\\oracle_fitnotes)
  --host     ubuntu@130.210.14.28
  --repo     ~/FitNotes-Agentic-AI-Coach   (on the VM)
  --service  app                      (Docker Compose service)
  --out      analytics_export         (relative to this repo)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_KEY = str(Path.home() / ".ssh" / "oracle_fitnotes")
DEFAULT_HOST = "ubuntu@130.210.14.28"
DEFAULT_REPO = "~/FitNotes-Agentic-AI-Coach"
DEFAULT_SERVICE = "app"
DEFAULT_OUT = str(REPO_ROOT / "analytics_export")

# Runs inside the container as `sh -c CONTAINER_SCRIPT _ <tmp path>`.
# The snapshot's JSON goes to stderr, the snapshot bytes to stdout, and the
# temporary file is removed whatever happens. Exit status: the snapshot's if it
# failed, else cat's.
CONTAINER_SCRIPT = (
    "python scripts/analytics_snapshot.py $1 >&2; s=$?; "
    "if [ $s -ne 0 ]; then rm -f $1; exit $s; fi; "
    "cat $1; s=$?; rm -f $1; exit $s"
)
# No double quotes anywhere in what is sent (the temp path never has spaces):
# nothing in it depends on how Windows passes quotes to ssh.exe.


def _remote_cd(repo: str) -> str:
    """`cd <repo>` for the VM's shell, keeping a leading ~ expandable."""
    if repo == "~":
        return "cd ~"
    if repo.startswith("~/"):
        return "cd ~/" + shlex.quote(repo[2:])
    return "cd " + shlex.quote(repo)


def remote_command(repo: str, service: str, tmp_path: str) -> str:
    """The one command line the VM's shell runs."""
    return (f"{_remote_cd(repo)} && docker compose exec -T {shlex.quote(service)} "
            f"sh -c {shlex.quote(CONTAINER_SCRIPT)} _ {shlex.quote(tmp_path)}")


def ssh_command(key: str, host: str, remote: str) -> list[str]:
    return ["ssh", "-i", key, "-o", "ConnectTimeout=20", host, remote]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_info(stderr: str) -> dict:
    """The snapshot script's JSON line (the last JSON object on stderr; the
    rest can be ssh / docker noise)."""
    for line in reversed((stderr or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def export(key: str = DEFAULT_KEY, host: str = DEFAULT_HOST,
           repo: str = DEFAULT_REPO, service: str = DEFAULT_SERVICE,
           out: str = DEFAULT_OUT, run=subprocess.run,
           now: datetime.datetime | None = None) -> Path:
    """Fetch one snapshot into `out`. Returns the saved file. Raises
    RuntimeError with a readable message on any failure."""
    now = now or datetime.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"analytics-{stamp}.sqlite"
    part = out_dir / f"analytics-{stamp}.sqlite.part"
    tmp_remote = f"/tmp/fitnotes-analytics-{stamp}.sqlite"

    cmd = ssh_command(key, host, remote_command(repo, service, tmp_remote))
    print(f"[export] connecting to {host} …")
    try:
        with open(part, "wb") as fh:
            proc = run(cmd, stdout=fh, stderr=subprocess.PIPE)
    except FileNotFoundError:
        part.unlink(missing_ok=True)
        raise RuntimeError("`ssh` was not found. On Windows 10/11 it is the built-in "
                           "OpenSSH client (Settings → Apps → Optional features).")
    stderr = proc.stderr.decode("utf-8", "replace") if isinstance(
        proc.stderr, (bytes, bytearray)) else (proc.stderr or "")
    info = _snapshot_info(stderr)

    if proc.returncode != 0:
        part.unlink(missing_ok=True)
        why = info.get("error") or stderr.strip()[-800:] or f"exit {proc.returncode}"
        if proc.returncode == 2 and info.get("error"):
            why += " (nothing recorded yet, or analytics is off on the server)"
        raise RuntimeError(f"export failed: {why}")

    local_sha = _sha256(part)
    if not info.get("sha256"):
        part.unlink(missing_ok=True)
        raise RuntimeError("export failed: the server did not report a checksum")
    if local_sha != info["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"export failed: checksum mismatch (server "
                           f"{info['sha256'][:12]}…, local {local_sha[:12]}…)")

    conn = sqlite3.connect(part)
    try:
        check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("visits", "turns", "events")}
    finally:
        conn.close()
    if check != "ok":
        part.unlink(missing_ok=True)
        raise RuntimeError(f"export failed: integrity check said {check!r}")

    os.replace(part, final)
    shutil.copyfile(final, out_dir / "latest.sqlite")
    print(f"[export] saved {final} ({final.stat().st_size:,} bytes, sha256 "
          f"{local_sha[:12]}…)")
    print(f"[export] rows: {counts['visits']} visits, {counts['turns']} turns, "
          f"{counts['events']} events")
    print(f"[export] latest copy: {out_dir / 'latest.sqlite'}")
    return final


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Copy the demo's visitor analytics to this PC over SSH.")
    ap.add_argument("--key", default=DEFAULT_KEY, help=f"SSH private key (default {DEFAULT_KEY})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"user@host (default {DEFAULT_HOST})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"repo folder on the VM (default {DEFAULT_REPO})")
    ap.add_argument("--service", default=DEFAULT_SERVICE,
                    help=f"Docker Compose service (default {DEFAULT_SERVICE})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"local folder (default {DEFAULT_OUT})")
    args = ap.parse_args(argv)
    if not Path(args.key).is_file():
        print(f"[export] SSH key not found: {args.key} (pass --key)", file=sys.stderr)
        return 1
    try:
        export(args.key, args.host, args.repo, args.service, args.out)
    except RuntimeError as exc:
        print(f"[export] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
