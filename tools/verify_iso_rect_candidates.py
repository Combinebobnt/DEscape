#!/usr/bin/env python3
"""Verifies descape.iso_geometry.tiles_in_screen_rect() -- v2.6 completion
plan's Track B, Phase B-A. tiles_in_screen_rect()
replaces an O(w*h) full-grid scan (still used directly today by
render.refresh_region_iso, which this script leaves untouched) with an
O(candidates) analytic s=x+y / d=y-x range solve -- this script is the
proof those two produce the EXACT SAME ordered result, not just the same
set: order matters for correct depth compositing, so a set-equality check
alone would miss a real bug.

Checks:
  1. Exact-order equivalence: for every real example file, >=200 random
     screen rects (plus explicit edge-straddling, fully-outside, and 1-px
     cases) -- tiles_in_screen_rect()'s output must be np.array_equal to a
     brute-force full-grid scan using the SAME underlying
     tile_screen_bounds_swept() formula, in the SAME depth_order.
  2. Degenerate/out-of-range rects: a rect entirely outside the canvas (in
     each of the four directions) returns an empty (0, 2) array, not a
     crash or a spurious candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry
from descape.render import tile_pixels_for_map
from descape.scenario_io import load_map_and_units

RNG_SEED = 20260805
N_RANDOM_RECTS_PER_FILE = 220


def full_scan_candidates(x0: int, y0: int, x1: int, y1: int, w: int, h: int, proj) -> np.ndarray:
    """Reference implementation: the O(w*h) full-grid scan
    render.refresh_region_iso itself uses for its terrain_intersects mask,
    built from the same tile_screen_bounds_swept() primitive
    tiles_in_screen_rect() uses -- so this isolates the enumeration
    STRATEGY as the only thing under test, not the underlying geometry
    formula (which both share)."""
    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    tsx, tsy, tsx_hi, tsy_hi = iso_geometry.tile_screen_bounds_swept(xs, ys, proj)
    mask = (tsx < x1) & (tsx_hi > x0) & (tsy < y1) & (tsy_hi > y0)
    order = iso_geometry.depth_order(w, h)
    return order[mask[order[:, 1], order[:, 0]]]


def make_rects(rng: np.random.Generator, canvas_w: int, canvas_h: int, n: int) -> list[tuple[int, int, int, int]]:
    rects = []
    # Explicit edge cases -- these are the ones a purely-random draw could
    # plausibly never hit, and are exactly the shapes the plan calls out.
    rects.append((0, 0, canvas_w, canvas_h))  # whole canvas
    rects.append((-500, -500, canvas_w + 500, canvas_h + 500))  # canvas fully inside rect
    rects.append((-1000, -1000, -900, -900))  # fully outside, negative
    rects.append((canvas_w + 900, canvas_h + 900, canvas_w + 1000, canvas_h + 1000))  # fully outside, positive
    rects.append((-50, -50, 1, 1))  # straddles the top-left canvas edge
    rects.append((canvas_w - 1, canvas_h - 1, canvas_w + 50, canvas_h + 50))  # straddles bottom-right
    for _ in range(8):
        x0 = int(rng.integers(0, max(1, canvas_w)))
        y0 = int(rng.integers(0, max(1, canvas_h)))
        rects.append((x0, y0, x0 + 1, y0 + 1))  # 1-px rects at random canvas positions

    # Bulk random rects, spanning well outside the canvas on both ends so
    # partially/fully-outside rects are common, not just the explicit cases.
    margin = max(200, canvas_w // 4, canvas_h // 4)
    for _ in range(n - len(rects)):
        x0 = int(rng.integers(-margin, canvas_w + margin))
        y0 = int(rng.integers(-margin, canvas_h + margin))
        rw = int(rng.integers(1, max(2, canvas_w // 3)))
        rh = int(rng.integers(1, max(2, canvas_h // 3)))
        rects.append((x0, y0, x0 + rw, y0 + rh))
    return rects


def check_real_files(files: list[Path]) -> tuple[bool, str]:
    rng = np.random.default_rng(RNG_SEED)
    failures = []
    lines = []
    total_rects = 0
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        tile_px = tile_pixels_for_map(w, h)
        proj = iso_geometry.canvas_size_and_origin(
            w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
        )

        rects = make_rects(rng, proj.canvas_w, proj.canvas_h, N_RANDOM_RECTS_PER_FILE)
        bad = 0
        for x0, y0, x1, y1 in rects:
            expected = full_scan_candidates(x0, y0, x1, y1, w, h, proj)
            got = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)
            if not np.array_equal(expected, got):
                bad += 1
                if bad <= 3:
                    lines.append(
                        f"  MISMATCH {path.name} rect=({x0},{y0},{x1},{y1}): "
                        f"full_scan={expected.shape[0]} candidates, tiles_in_screen_rect={got.shape[0]}"
                    )

        total_rects += len(rects)
        status = "PASS" if bad == 0 else "FAIL"
        if bad:
            failures.append(path.name)
        lines.append(f"{status}  {path.name:36s} {w}x{h} tile_px={tile_px:2d}  {len(rects)} rects, {bad} mismatched")

    print("\n".join(lines))
    if failures:
        return False, f"{len(failures)}/{len(files)} files had mismatching rects: {failures}"
    return True, f"OK ({len(files)} files, {total_rects} rects total, every one exactly order-matched the full scan)"


def check_out_of_range_returns_empty() -> tuple[bool, str]:
    w, h, tile_px = 20, 20, 32
    proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION)
    problems = []
    far_outside = [
        (-10000, -10000, -9000, -9000),
        (proj.canvas_w + 9000, 0, proj.canvas_w + 10000, 100),
        (0, proj.canvas_h + 9000, 100, proj.canvas_h + 10000),
        (-10000, proj.canvas_h + 9000, -9000, proj.canvas_h + 10000),
    ]
    for x0, y0, x1, y1 in far_outside:
        got = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)
        if got.shape != (0, 2):
            problems.append(f"rect=({x0},{y0},{x1},{y1}): expected empty (0,2), got shape {got.shape}")
    if problems:
        return False, "; ".join(problems)
    return True, "OK (4 far-outside rects on a 20x20 map, all returned empty)"


def main() -> None:
    scenario_dir = ROOT / "examples"
    files = sorted(scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {scenario_dir}")
        sys.exit(1)

    checks = [
        ("Exact-order equivalence vs full scan (real files)", lambda: check_real_files(files)),
        ("Out-of-range rects return empty", check_out_of_range_returns_empty),
    ]

    failures = 0
    for name, fn in checks:
        print(f"\n=== {name} ===")
        ok, detail = fn()
        print(f"{'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            failures += 1

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
