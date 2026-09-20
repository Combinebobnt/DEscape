#!/usr/bin/env python3
"""Times a Sloped pick plane's SUB-RECT rewrite against a whole-plane
rebuild. This is the measurement that fixes
render_cache.PICK_PLANE_PATCH_MAX_FRACTION, Batch B step B8's drop-vs-patch
threshold. Informational only, matching tools/bench_sloped_patch.py's
convention: always runs, never pass/fail.

Why a separate tool rather than another section in bench_sloped_patch.py:
that file's section 3 is the stable before/after reference for the hover
cost itself and must keep timing exactly what it times today. This one asks
a different question (how a sub-rect id composite scales against the full
plane) and adds an end-to-end leg through a real SlopedChunkCache.

Section 1 is the constant's own evidence: composite_ids_rect_sloped over
sub-rects of a chunk_px-squared plane at each AREA_FRACTIONS share of its
area, in two shapes. A centred SQUARE is the plan's own 25/50/75/100% ask,
widened so the curve is fitted rather than guessed from four samples; a
full-width BAND is the other shape a real drag bbox cuts out of a plane, and
it costs more per pixel because it spans every candidate column. Cost per
pixel is not flat either way: the tiles straddling the sub-rect's own border
pay a full candidate walk and _clipped_paint for the fraction of themselves
that lands inside it, so a small rect carries proportionally more border.
Read the ratio column, not the ms. Each sub-rect is paired with a full-rect
timing taken right beside it and compared against that neighbour rather than
against one reference at the top, because this machine drifts 15-20% across
a single run, which is wider than the differences the threshold turns on.

Section 2 is what the threshold actually buys: a real cache, a real
one-tile elevation edit inside the cursor's own chunk, and the cost of
patch() plus the pick_tile() that a drag's very next mouseMoveEvent pays.
Run it before and after B8.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape.elevation_tools import set_tile_elevation
from descape.render import (
    composite_ids_rect_sloped,
    dirty_screen_bbox_sloped,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import DEFAULT_CHUNK_PX, SlopedChunkCache
from descape.scenario_io import load_map_and_units

# Area fractions of one full plane to time; 1.0 is the full-plane rebuild
# every other row is measured against (see the module docstring).
AREA_FRACTIONS = (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)

REPEATS = 8


def _time_ms(fn, repeats: int = REPEATS) -> float:
    """Mean wall-clock milliseconds per call, one untimed warm-up first.
    Same contract as bench_sloped_patch._time_ms: the lru_cached geometry
    these gather through must not be charged to the first timed call."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000 / repeats


def _sub_rect(
    px0: int, py0: int, plane_w: int, plane_h: int, fraction: float, band: bool = False
) -> tuple[int, int, int, int]:
    """A centred sub-rect of the plane covering `fraction` of its area.
    `band` makes it full-width instead of square, which is the other shape a
    real drag bbox cuts out of a plane and carries more border per pixel."""
    if band:
        w = plane_w
        h = max(1, round(plane_h * fraction))
    else:
        side = math.sqrt(fraction)
        w, h = max(1, round(plane_w * side)), max(1, round(plane_h * side))
    x0 = px0 + (plane_w - w) // 2
    y0 = py0 + (plane_h - h) // 2
    return x0, y0, x0 + w, y0 + h


def _plane_rect(proj, chunk_px: int) -> tuple[int, int, int, int]:
    """The chunk-grid plane nearest the canvas centre, clipped to the canvas
    the same way SlopedChunkCache._pick_plane clips it."""
    cx = (proj.canvas_w // 2) // chunk_px
    cy = (proj.canvas_h // 2) // chunk_px
    x0, y0 = cx * chunk_px, cy * chunk_px
    return x0, y0, min(proj.canvas_w, x0 + chunk_px), min(proj.canvas_h, y0 + chunk_px)


def _section_1(scenario, corner_rise, proj, tile_px: int, chunk_px: int) -> list[str]:
    px0, py0, px1, py1 = _plane_rect(proj, chunk_px)
    plane_w, plane_h = px1 - px0, py1 - py0
    def _ids(x0, y0, x1, y1):
        return lambda: composite_ids_rect_sloped(scenario, x0, y0, x1, y1, corner_rise, proj, tile_px)

    # Paired with a full-rect timing beside it, not one reference at the
    # top, so this machine's within-run drift cancels (module docstring).
    lines = [
        f"  (1) sub-rect id composite vs a full {plane_w}x{plane_h} plane rebuild",
        f"    {'area':>6}  {'shape':>6}  {'rect':>12}  {'sub':>9}  {'full':>9}  {'vs full':>8}",
    ]
    for band in (False, True):
        for fraction in AREA_FRACTIONS:
            if band and fraction == 1.0:
                continue
            x0, y0, x1, y1 = _sub_rect(px0, py0, plane_w, plane_h, fraction, band=band)
            ms = _time_ms(_ids(x0, y0, x1, y1))
            full_ms = _time_ms(_ids(px0, py0, px1, py1))
            share = ms / full_ms if full_ms else float("nan")
            lines.append(
                f"    {fraction:5.0%}  {'band' if band else 'square':>6}  "
                f"{f'{x1 - x0}x{y1 - y0}':>12}  {ms:7.3f}ms  {full_ms:7.3f}ms  {share:7.2%}"
            )
    return lines


def _section_2(path: Path, chunk_px: int) -> list[str]:
    """One real edit through the real cache: warm the cursor's plane, edit a
    tile inside that plane's own chunk, then time patch() and the pick_tile()
    the next mouseMoveEvent runs."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px)

    px0, py0, px1, py1 = _plane_rect(proj, chunk_px)
    sx, sy = (px0 + px1) // 2, (py0 + py1) // 2
    tile = cache.pick_tile(sx, sy)
    if tile is None:
        return ["  (2) skipped: the canvas centre is off-map on this file"]

    patch_ms: list[float] = []
    pick_ms: list[float] = []
    for i in range(6):
        ex, ey = tile
        before_grid = [int(t.elevation) for t in mm.terrain]
        set_tile_elevation(mm, ex, ey, int(mm.get_tile(ex, ey).elevation) + (1 if i % 2 == 0 else -1))
        dirty = [idx for idx, t in enumerate(mm.terrain) if int(t.elevation) != before_grid[idx]]
        elevation_changed: set = set()
        bbox = dirty_screen_bbox_sloped(
            scenario, dirty, elevations, proj, with_units=True, elevation_changed=elevation_changed
        )
        if bbox is None:
            continue
        cache.pick_tile(sx, sy)  # warm, as a drag leaves it
        t0 = time.perf_counter()
        cache.patch(bbox, elevation_changed=elevation_changed)
        t1 = time.perf_counter()
        cache.pick_tile(sx, sy)
        t2 = time.perf_counter()
        if i:  # first pass is the warm-up
            patch_ms.append((t1 - t0) * 1000)
            pick_ms.append((t2 - t1) * 1000)

    if not patch_ms:
        return ["  (2) skipped: no in-range edit produced a bbox"]
    mean_patch = sum(patch_ms) / len(patch_ms)
    mean_pick = sum(pick_ms) / len(pick_ms)
    return [
        "  (2) one real elevation edit under the cursor, through the cache",
        f"    {'patch()':>22}  {mean_patch:7.3f}ms",
        f"    {'next pick_tile()':>22}  {mean_pick:7.3f}ms",
        f"    {'total':>22}  {mean_patch + mean_pick:7.3f}ms",
    ]


def bench(path: Path, chunk_px: int) -> str:
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    _elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    lines = [
        (
            f"  {path.name} ({mm.map_width}x{mm.map_height}, tile_px={tile_px}, "
            f"canvas {proj.canvas_w}x{proj.canvas_h}, chunk_px={chunk_px})"
        ),
        "",
    ]
    lines += _section_1(scenario, corner_rise, proj, tile_px, chunk_px)
    lines += [""]
    lines += _section_2(path, chunk_px)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    parser.add_argument("--chunk-px", type=int, default=DEFAULT_CHUNK_PX)
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        raise SystemExit(f"No .aoe2scenario files found in {args.scenario_dir}")
    largest = max(files, key=lambda p: load_map_and_units(p).map_manager.map_width)
    print(bench(largest, args.chunk_px))


if __name__ == "__main__":
    main()
