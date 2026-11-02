#!/usr/bin/env python3
"""
Download DB-IP's free "IP to City Lite" database (MMDB) for the visitor
analytics' city / country lookup (src/analytics.py).

    python deploy/fetch_geoip.py /app/seed/geo/dbip-city-lite.mmdb [YYYY-MM]

Run by the Dockerfile at build time. DB-IP publishes one file a month at
https://download.db-ip.com/free/dbip-city-lite-YYYY-MM.mmdb.gz; this tries the
month given (GEOIP_MONTH build arg), then the current and the previous month.
The file is checked by looking up a public address before it replaces anything.

Never fails the build: on any error it prints a warning and exits 0, and the
analytics simply record no location.

Licence: CC BY 4.0. Attribution: "IP Geolocation by DB-IP" (https://db-ip.com).
"""

from __future__ import annotations

import datetime
import gzip
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://download.db-ip.com/free/dbip-city-lite-{month}.mmdb.gz"
ATTRIBUTION = "IP Geolocation by DB-IP (https://db-ip.com), CC BY 4.0"


def candidate_months(explicit: str | None = None,
                     today: datetime.date | None = None) -> list[str]:
    today = today or datetime.date.today()
    first = today.replace(day=1)
    prev = (first - datetime.timedelta(days=1)).replace(day=1)
    months = [first.strftime("%Y-%m"), prev.strftime("%Y-%m")]
    if explicit and explicit.strip():
        months.insert(0, explicit.strip())
    seen: list[str] = []
    for m in months:
        if m not in seen:
            seen.append(m)
    return seen


def _valid(path: str) -> bool:
    try:
        import maxminddb
        with maxminddb.open_database(path) as reader:
            rec = reader.get("8.8.8.8")
        return bool(rec and rec.get("country"))
    except Exception as exc:
        print(f"[geoip] downloaded file is not usable: {exc}")
        return False


def fetch(dest: str, month: str | None = None, opener=urllib.request.urlopen) -> bool:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for m in candidate_months(month):
        url = URL.format(month=m)
        print(f"[geoip] trying {url}")
        fd, tmp = tempfile.mkstemp(dir=dest_path.parent, suffix=".mmdb")
        os.close(fd)
        try:
            with opener(url, timeout=120) as resp, open(tmp, "wb") as out:
                with gzip.GzipFile(fileobj=resp) as gz:
                    shutil.copyfileobj(gz, out, 1 << 20)
            if not _valid(tmp):
                continue
            os.replace(tmp, dest_path)
            (dest_path.parent / "SOURCE.txt").write_text(
                f"{ATTRIBUTION}\nfile: dbip-city-lite-{m}.mmdb\n", encoding="utf-8")
            print(f"[geoip] saved {dest_path} ({dest_path.stat().st_size:,} bytes, {m})")
            return True
        except Exception as exc:
            print(f"[geoip] {m}: {exc}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    print("[geoip] WARNING: no IP location database; visitor analytics will "
          "record no city/country. Rebuild later to try again.")
    return False


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    fetch(argv[0], argv[1] if len(argv) > 1 else os.environ.get("GEOIP_MONTH"))
    return 0                                  # never fail the build


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
