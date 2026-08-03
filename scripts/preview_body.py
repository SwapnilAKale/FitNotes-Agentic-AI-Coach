#!/usr/bin/env python3
"""
scripts/preview_body.py
Render frontend/body/figure.json as an ASCII silhouette — front, side and back.

This exists because three shipped defects in a row (a spinner-hang, a canvas
blown to 2x width, a stick figure) were all BLIND failures rather than thinking
failures. Geometry cannot be verified by reading it. This is the loop that was
missing: look at the figure in the terminal, fix the numbers, look again, and
only then put it in a browser.

The side view matters most — it is the one that catches a figure built from
rotationally-symmetric parts, where the chest, glutes and feet have no projection
and the whole thing reads as a column.

    python scripts/preview_body.py                 # all three views
    python scripts/preview_body.py --view front
    python scripts/preview_body.py --muscles       # shade muscle volumes
    python scripts/preview_body.py --part torso    # isolate one part

Pure geometry. Nothing in this file knows what FitNotes is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE = os.path.join(_ROOT, "frontend", "body", "figure.json")

# ── the geometry kernel ───────────────────────────────────────────────────────
# A part is a chain of rings: {c:[x,y,z], rx, rz}. Consecutive rings loft into a
# tube. The SAME primitive builds the base form and every muscle belly, so the
# viewer and this previewer never diverge — there is one shape language.


def load_figure(path: str = FIGURE) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mirrored(part: dict) -> dict:
    """A part authored on one side, flipped in x. Symmetry is structural rather
    than typed twice, so the two halves cannot drift apart."""
    out = json.loads(json.dumps(part))
    out["name"] = part["name"] + " (L)"
    for r in out["rings"]:
        r["c"][0] = -r["c"][0]
    return out


def expand(figure: dict) -> list:
    """Every part that actually exists, mirrors included."""
    parts = []
    for p in figure["parts"]:
        parts.append(p)
        if p.get("mirror"):
            parts.append(mirrored(p))
    return parts


def _lerp(a, b, t):
    return a + (b - a) * t


def sample_rings(part: dict, per_segment: int = 14) -> list:
    """Interpolate the chain so the silhouette is smooth rather than faceted.
    Returns [(cx, cy, cz, rx, rz)]."""
    rings = part["rings"]
    out = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for s in range(per_segment):
            t = s / per_segment
            # smoothstep gives the bellies a rounded profile instead of cones
            ts = t * t * (3 - 2 * t)
            out.append((
                _lerp(a["c"][0], b["c"][0], t),
                _lerp(a["c"][1], b["c"][1], t),
                _lerp(a["c"][2], b["c"][2], t),
                _lerp(a["rx"], b["rx"], ts),
                _lerp(a["rz"], b["rz"], ts),
            ))
    last = rings[-1]
    out.append((last["c"][0], last["c"][1], last["c"][2], last["rx"], last["rz"]))
    return out


# ── rasteriser ────────────────────────────────────────────────────────────────

VIEWS = {
    # name: (horizontal axis index, flip)
    "front": (0, 1),    # x across
    "side":  (2, 1),    # z across  (facing right)
    "back":  (0, -1),   # x across, mirrored
}


def rasterise(parts: list, view: str, w: int, h: int,
              shade_muscles: bool) -> list:
    axis, flip = VIEWS[view]
    grid = [[" "] * w for _ in range(h)]

    # Fixed world window so all three views share a scale and can be compared.
    y0, y1 = -0.06, 1.06
    # Terminal cells are about twice as tall as they are wide. Derive the
    # horizontal span from that instead of hard-coding it, or every judgement
    # about proportion is made against a stretched image.
    CHAR_ASPECT = 0.5
    span = (w / h) * (y1 - y0) * CHAR_ASPECT
    cx = 0.0

    def to_col(u):
        return int(round((u * flip - cx + span / 2) / span * (w - 1)))

    def to_row(y):
        return int(round((1 - (y - y0) / (y1 - y0)) * (h - 1)))

    ordered = sorted(parts, key=lambda p: 0 if p.get("layer") == "base" else 1)
    for part in ordered:
        is_muscle = part.get("layer") == "muscle"
        ch = ("#" if is_muscle else ".") if shade_muscles else "#"
        if is_muscle and not shade_muscles:
            ch = "#"
        for (px, py, pz, rx, rz) in sample_rings(part):
            u = px if axis == 0 else pz
            r = rx if axis == 0 else rz
            row = to_row(py)
            if row < 0 or row >= h:
                continue
            c0, c1 = to_col(u - r), to_col(u + r)
            if c1 < c0:
                c0, c1 = c1, c0
            for col in range(max(c0, 0), min(c1, w - 1) + 1):
                if grid[row][col] == " " or (is_muscle and shade_muscles):
                    grid[row][col] = ch
    return grid


def render(figure: dict, view: str, w: int, h: int, shade: bool,
           only: str | None) -> str:
    parts = expand(figure)
    if only:
        parts = [p for p in parts if only.lower() in p["name"].lower()]
        if not parts:
            return f"  (no part matching {only!r})"
    grid = rasterise(parts, view, w, h, shade)
    return "\n".join("  " + "".join(r).rstrip() for r in grid)


def landmarks(h: int) -> dict:
    """Rows at which the canonical 8-head landmarks should appear, for eyeballing
    proportion against the drawing."""
    y0, y1 = -0.06, 1.06
    out = {}
    for name, y in [("crown", 1.000), ("chin", 0.875), ("shoulder", 0.820),
                    ("nipple", 0.750), ("navel", 0.625), ("crotch", 0.500),
                    ("knee", 0.250), ("sole", 0.000)]:
        out[int(round((1 - (y - y0) / (y1 - y0)) * (h - 1)))] = name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["front", "side", "back", "all"], default="all")
    ap.add_argument("--muscles", action="store_true",
                    help="shade muscle volumes distinctly from the base form")
    ap.add_argument("--part", help="isolate parts whose name contains this")
    ap.add_argument("--width", type=int, default=46)
    ap.add_argument("--height", type=int, default=46)
    args = ap.parse_args()

    if not os.path.exists(FIGURE):
        print(f"!! {os.path.relpath(FIGURE, _ROOT)} does not exist yet.")
        return 1
    figure = load_figure()

    views = ["front", "side", "back"] if args.view == "all" else [args.view]
    marks = landmarks(args.height)

    blocks = []
    for v in views:
        body = render(figure, v, args.width, args.height, args.muscles,
                      args.part).splitlines()
        titled = [f"  {v.upper():^{args.width}}"] + body
        blocks.append(titled)

    # side-by-side, with landmark rulers down the left
    print()
    height = max(len(b) for b in blocks)
    for i in range(height):
        label = marks.get(i - 1, "")
        line = f"{label:>9} |" if label else " " * 9 + " |"
        for b in blocks:
            line += (b[i] if i < len(b) else "").ljust(args.width + 4)
        print(line)
    print()
    n_base = sum(1 for p in expand(figure) if p.get("layer") == "base")
    n_mus = sum(1 for p in expand(figure) if p.get("layer") == "muscle")
    print(f"  {n_base} base parts, {n_mus} muscle volumes "
          f"({len(figure['parts'])} authored, mirrors expanded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
