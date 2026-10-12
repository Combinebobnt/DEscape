#!/usr/bin/env python3
"""Times descape.render.composite_rect_sloped against composite_rect_iso at
edit-sized rects and at a cold full chunk -- Track C4's "Gate 2" measurement,
the per-tile sloped composite cost this project flags as
"unestimated -- a Phase 6 deliverable to measure, not assume". Informational
only, matching this project's own bench_chunk_px.py/bench_iso_backend.py
convention -- always runs, never pass/fail.

The number this exists to inform is whether Sloped can be live-edited at all,
or whether it needs an auto-switch to Stepped mid-stroke while a stroke is in
flight (the C4 plan's Step 5).

WHAT THIS CAN AND CANNOT HONESTLY MEASURE. This is the compositor CONSTANT,
not end-to-end patch latency: the rects here are SYNTHETIC, sized like a
one-tile ring around a propagated elevation edit, never derived from a real
edit.

That distinction used to mean end-to-end was UNMEASURABLE -- corner_rise was
never rebuilt and dirty_screen_bbox_sloped did not exist. C4's Step 3 shipped
both (2026-08-21), so the real path can now be timed, and was: on a blank
480x480 at tile_px=32, one elevation click propagating to 9 tiles costs 25ms
sloped against 11ms stepped, of which 4.6ms is the whole-array corner_rise
rebuild. Extending this tool to measure that directly is a real follow-up;
until then the numbers below are still only the constant.

Expect the sloped constant to be WORSE than it was before the seam resample,
not better: sloped_quad_indices' non-equal-corner branch now allocates all
five arrays fresh where its predecessor aliased three of diamond_indices',
a measured 7.5x per-entry payload (61 KB/entry at tile_px=64, 246 KB at 128).

Do NOT use tools/bench_iso_memory.py for the per-entry payload question --
it reports whole-process peak RSS per render stage only, never a per-entry
cost. Measure the arrays directly, as sloped_quad_indices' own docstring did.

Since TASK-143's perf gate every composite here goes through a level-warm
SlopedChunkCache / IsoChunkCache (_composite_rect(), sprites off, unit marks
on), not a bare composite_rect_*() call. The bare call has no unit pack and no
terrain mirrors, so on the native backend it took the per-tile path and, with
blending on, re-read both whole-map mirrors per call: neither is what the app
pays. Figures from before that change are not comparable.

--blend on|off sets View > Layers > Terrain Blending on both caches (the gate
compares the two, interleaved). `on` needs the game install's blendomatic.
`keyed` is each rect's candidate tiles carrying a blend key, counted outside
the timers; an `on` run whose rects hold none measured no blending and exits
non-zero.

Section (4) is the pan / margin-warm cost of the chunk under the canvas
centre: prepare_chunk_job() is what a margin-warm tick pays on the GUI thread
(candidates, blend keys and tables), run() the nogil kernel on a worker.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import composite_backend, iso_geometry, render
from descape.render import (
    composite_ids_rect_sloped,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import DEFAULT_CHUNK_PX, IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units
from descape.view_layers import DEFAULT_LAYERS

# Screen-rect sides to time, in TILES. A single elevation click propagates
# through MapManager._elevation_tile_recursion and each changed tile reshapes
# up to its own 3x3 neighborhood, so 3 and 5 bracket a realistic one-click
# dirty rect; 1 is the floor, not a realistic edit.
EDIT_RECT_TILES = (1, 3, 5)

# Repeats per timing, so a single rect's cost clears timer noise. The rects
# here are small enough that one call is well under a millisecond.
REPEATS = 20


def _time_ms(fn, repeats: int = REPEATS) -> float:
    """Mean wall-clock milliseconds per call. Runs fn once first, untimed:
    both compositors gather through lru_cached geometry (diamond_indices,
    sloped_quad_indices), so a cold first call would charge this rect for
    cache fills every later call gets free -- which is the steady-state cost
    an interactive stroke actually pays."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000 / repeats


def _centered_rect(proj, tile_px: int, n_tiles: int) -> tuple[int, int, int, int]:
    """A screen-space square of side n_tiles * tile_px at the canvas center.
    Square in SCREEN pixels rather than a projected tile block: the rect a
    patch path passes down is a screen bbox, and squaring it keeps the three
    sizes comparable to each other and to the Stepped run at the same rect."""
    side = n_tiles * tile_px
    cx, cy = proj.canvas_w // 2, proj.canvas_h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    return x0, y0, min(proj.canvas_w, x0 + side), min(proj.canvas_h, y0 + side)


def _warm_cache(cache):
    """Level 0 and its unit pack built, as the app has them before any patch."""
    cache._level(0)
    job = cache.pack_warm_job(0)
    if job is not None:
        for _ in job.gen:
            pass
    return cache


def _keyed(cache, rect: tuple[int, int, int, int]) -> int:
    """How many of the rect's candidate tiles carry a blend key (0 with blending off)."""
    if not render.blend_effective(cache.layers):
        return 0
    mm = cache.scenario.map_manager
    tiles = iso_geometry.tiles_in_screen_rect(*rect, mm.map_width, mm.map_height, cache.proj)
    if not tiles.shape[0]:
        return 0
    keys = render._blend_keys(cache.scenario, tiles[:, 0], tiles[:, 1], cache._terrain_ids, cache._layer_ids)
    return sum(k is not None for k in keys)


def bench_sloped_patch(path: Path, blend: bool = True) -> tuple[str, int]:
    """The report, and how many blend-keyed candidates its Sloped rects held."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    layers = replace(DEFAULT_LAYERS, terrain_blend=blend)

    elevations_s, corner_rise, proj_s = sloped_elevations_and_proj(scenario)
    elevations_i, proj_i = elevations_and_proj(scenario)
    sloped = _warm_cache(
        SlopedChunkCache(scenario, elevations_s, corner_rise, proj_s, tile_px, sprites=False, layers=layers)
    )
    stepped = _warm_cache(IsoChunkCache(scenario, elevations_i, proj_i, tile_px, sprites=False, layers=layers))
    keyed_total = 0

    lines = [
        (
            f"  {path.name} ({mm.map_width}x{mm.map_height}, tile_px={tile_px}, "
            f"canvas {proj_s.canvas_w}x{proj_s.canvas_h}), terrain blending {'on' if blend else 'off'}"
        ),
        "",
        "  (1) edit-sized rects -- synthetic, NOT from a real edit (see module docstring)",
        f"    {'rect':>14}  {'sloped':>10}  {'stepped':>10}  {'ratio':>7}  {'keyed':>6}",
    ]

    for n in EDIT_RECT_TILES:
        x0, y0, x1, y1 = _centered_rect(proj_s, tile_px, n)
        sloped_ms = _time_ms(lambda: sloped._composite_rect(0, x0, y0, x1, y1))
        stepped_ms = _time_ms(lambda: stepped._composite_rect(0, x0, y0, x1, y1))
        ratio = sloped_ms / stepped_ms if stepped_ms else float("nan")
        keyed = _keyed(sloped, (x0, y0, x1, y1))
        keyed_total += keyed
        lines.append(
            f"    {f'{n}t {x1 - x0}x{y1 - y0}':>14}  {sloped_ms:8.3f}ms  {stepped_ms:8.3f}ms  {ratio:6.2f}x  {keyed:6d}"
        )

    lines += ["", f"  (2) cold full chunk at chunk_px={DEFAULT_CHUNK_PX} -- pan-into-unseen-territory proxy"]
    x0, y0 = proj_s.canvas_w // 2, proj_s.canvas_h // 2
    x1, y1 = min(proj_s.canvas_w, x0 + DEFAULT_CHUNK_PX), min(proj_s.canvas_h, y0 + DEFAULT_CHUNK_PX)
    chunk_sloped = _time_ms(lambda: sloped._composite_rect(0, x0, y0, x1, y1), repeats=3)
    chunk_stepped = _time_ms(lambda: stepped._composite_rect(0, x0, y0, x1, y1), repeats=3)
    ratio = chunk_sloped / chunk_stepped if chunk_stepped else float("nan")
    keyed = _keyed(sloped, (x0, y0, x1, y1))
    keyed_total += keyed
    lines.append(
        f"    {f'{x1 - x0}x{y1 - y0}':>14}  {chunk_sloped:8.3f}ms  {chunk_stepped:8.3f}ms  {ratio:6.2f}x  {keyed:6d}"
    )

    # (3) The hover cost: SlopedChunkCache._pick_plane's ID-only recomposite,
    # paid once per chunk boundary the cursor crosses. Same candidate walk
    # with the texture crop and _slope_shade skipped, so it should come in
    # materially under (2) -- measured rather than assumed, since that is the
    # premise the whole "build the plane lazily, don't cache it alongside RGB"
    # decision rests on.
    lines += ["", f"  (3) ID-only pick plane at chunk_px={DEFAULT_CHUNK_PX} -- per-chunk-crossing hover cost"]
    ids_ms = _time_ms(
        lambda: composite_ids_rect_sloped(scenario, x0, y0, x1, y1, corner_rise, proj_s, tile_px), repeats=3
    )
    share = ids_ms / chunk_sloped if chunk_sloped else float("nan")
    lines.append(
        f"    {f'{x1 - x0}x{y1 - y0}':>14}  {ids_ms:8.3f}ms  "
        f"({share:.0%} of the same rect's colour composite)"
    )

    lines += [
        "",
        f"  (4) margin-warm chunk job at chunk_px={DEFAULT_CHUNK_PX} -- the chunk under the canvas centre",
        f"    {'style':>14}  {'prepare (GUI)':>14}  {'kernel (worker)':>16}  {'keyed':>6}",
    ]
    for name, cache in (("sloped", sloped), ("stepped", stepped)):
        # The same canvas point at both styles; nothing here inserts a chunk, so it is never cached.
        cx, cy = cache.proj.canvas_w // 2 // cache.chunk_px, cache.proj.canvas_h // 2 // cache.chunk_px
        job = cache.prepare_chunk_job(0, cx, cy)
        if job is None:
            lines.append(f"    {name:>14}  no worker path (numpy backend, or a rect the kernel declines)")
            continue
        prepare_ms = _time_ms(lambda: cache.prepare_chunk_job(0, cx, cy), repeats=5)
        kernel_ms = _time_ms(job.run, repeats=5)
        keyed = _keyed(cache, cache._chunk_rect(0, cx, cy))
        keyed_total += keyed if name == "sloped" else 0
        lines.append(f"    {name:>14}  {prepare_ms:12.3f}ms  {kernel_ms:14.3f}ms  {keyed:6d}")
    return "\n".join(lines), keyed_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    parser.add_argument("--file", help="A file name in scenario_dir (default: its widest map)")
    parser.add_argument(
        "--blend", choices=("on", "off"), default="on",
        help="View > Layers > Terrain Blending on both caches (on needs the game install)",
    )
    args = parser.parse_args()
    blend = args.blend == "on"
    if blend and not render.blend_effective():
        raise SystemExit("--blend on: no blendomatic readable (no AoE2:DE install configured). Pass --blend off.")

    if args.file:
        path = args.scenario_dir / args.file
    else:
        files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
        if not files:
            raise SystemExit(f"No .aoe2scenario files found in {args.scenario_dir}")
        path = max(files, key=lambda p: load_map_and_units(p).map_manager.map_width)

    print(composite_backend.describe())
    text, keyed = bench_sloped_patch(path, blend)
    print(text)
    if blend and keyed == 0:
        raise SystemExit("--blend on, but no measured rect holds a blend-keyed tile: nothing blended was timed")


if __name__ == "__main__":
    main()
