"""
Download the speech models dictation needs (about 480 MB) and check each file.

    python scripts/fetch_speech_models.py [folder]           download what is missing or damaged
    python scripts/fetch_speech_models.py [folder] --check   only check; exits 1 if anything is wrong

The folder defaults to SPEECH_MODELS, else models/speech/. Already have the
models (e.g. from Confab)? Point SPEECH_MODELS at that folder instead.

Every file listed in src/dictation/models.json is accepted only if its size and
SHA-256 match. Downloads resume from a ".part" file; an archive is checked
before it is unpacked, and only the listed files are moved into place.
Ported from confab-stt's scripts/fetch-models.cjs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "src" / "dictation" / "models.json"


def mb(n: int) -> str:
    return f"{n / 1048576:.0f} MB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check_piece(root: Path, piece: dict) -> tuple[list, list]:
    missing, bad = [], []
    for f in piece["files"]:
        path = root / piece["path"] / f["path"]
        if not path.is_file():
            missing.append(f["path"])
        elif path.stat().st_size != f["size"] or sha256_file(path) != f["sha256"]:
            bad.append(f["path"])
    return missing, bad


def download(url: str, dest: Path, size: int, sha256: str, label: str, attempts: int = 4) -> None:
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        try:
            if have < size:
                req = urllib.request.Request(url, headers={"User-Agent": "fitnotes-coach",
                                                           **({"Range": f"bytes={have}-"} if have else {})})
                with urllib.request.urlopen(req, timeout=60) as res:
                    if res.status == 200 and have:
                        have = 0               # no resume support: start again
                    last = 0.0
                    with open(part, "ab" if have else "wb") as out:
                        done = have
                        while chunk := res.read(1048576):
                            out.write(chunk)
                            done += len(chunk)
                            if time.monotonic() - last > 2:
                                last = time.monotonic()
                                print(f"  {label}: {mb(done)} of {mb(size)}", flush=True)
            got = part.stat().st_size
            if got != size:
                raise OSError(f"the download stopped at {mb(got)} of {mb(size)}")
            if sha256_file(part) != sha256:
                part.unlink()
                raise ValueError(f"{label} did not match its SHA-256 checksum, so it was deleted.")
            part.replace(dest)
            return
        except Exception as exc:
            if attempt == attempts or (isinstance(exc, ValueError) and attempt >= 2):
                raise SystemExit(f"Could not download {label} from {url}: {exc}")
            time.sleep(attempt)


def fetch_piece(root: Path, piece: dict, need: list, downloads: Path) -> None:
    spec, target = piece["download"], root / piece["path"]
    extra, archive = spec.get("extra", {}), spec.get("archive")
    from_archive = [n for n in need if n not in extra and archive]
    work = downloads / piece["path"]
    for name in (n for n in need if n not in from_archive):
        f = next(x for x in piece["files"] if x["path"] == name)
        source = extra.get(name)
        if not source:
            raise SystemExit(f"{piece['name']}: no download source for {name}.")
        if source.startswith("repo:"):
            # A Windows checkout may turn a text file's line endings into CRLF;
            # the manifest hashes the LF original.
            data = (REPO / source[5:]).read_bytes()
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                data = data.replace(b"\r\n", b"\n")
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise SystemExit(f"{piece['name']}: {source[5:]} does not match its checksum.")
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes(data)
            continue
        download(source, work / name, f["size"], f["sha256"], f"{piece['name']}: {name}")
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).replace(target / name)
    if from_archive:
        path = downloads / Path(archive["url"]).name
        download(archive["url"], path, archive["size"], archive["sha256"], piece["name"])
        staging = downloads / ("unpack-" + piece["path"])
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  Unpacking {piece['name']} (this can take a minute or two)...", flush=True)
        with tarfile.open(path) as tar:
            tar.extractall(staging, filter="data")
        for name in from_archive:
            src = staging / archive.get("folder", "") / name
            if not src.is_file():
                raise SystemExit(f"{piece['name']}: {name} is not in the downloaded archive.")
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            src.replace(target / name)
        shutil.rmtree(staging, ignore_errors=True)
        path.unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    root = Path(args[0] if args else os.environ.get("SPEECH_MODELS") or REPO / "models" / "speech").resolve()
    pieces = json.loads(MANIFEST.read_text(encoding="utf-8"))["pieces"]
    if "--check" in argv:
        ok = True
        for piece in pieces:
            missing, bad = check_piece(root, piece)
            ok &= not (missing or bad)
            print(("ok       " if not (missing or bad) else "MISSING  ") + piece["path"]
                  + ("" if not (missing or bad) else f" ({', '.join(missing + bad)})"))
        return 0 if ok else 1
    downloads = root / ".downloads"
    for piece in pieces:
        print(f"Checking {piece['name']}...", flush=True)
        missing, bad = check_piece(root, piece)
        if not (missing or bad):
            print(f"  Present and verified ({mb(piece['totalBytes'])}).")
            continue
        print("  Not downloaded yet. Downloading..." if len(missing) == len(piece["files"])
              else f"  {len(missing) + len(bad)} file(s) missing or damaged. Downloading...", flush=True)
        fetch_piece(root, piece, missing + bad, downloads)
        missing, bad = check_piece(root, piece)
        if missing or bad:
            raise SystemExit(f"{piece['name']} failed its SHA-256 check after download: {', '.join((missing + bad)[:5])}")
        print("  Downloaded and verified.")
    shutil.rmtree(downloads, ignore_errors=True)
    print(f"Speech models ready in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
