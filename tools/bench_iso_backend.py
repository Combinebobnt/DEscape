#!/usr/bin/env python3
"""Phase 0 rendering-backend benchmark, for the "real isometric Z-height
terrain rendering" plan.

Times four things on the same tile data, for each dataset:
  1. the existing flat renderer (descape.render.render_terrain) -- today's
     shipped baseline, unchanged.
  2. the Phase 0 numpy prototype iso-blit kernel
     (native/iso_render/iso_bench_numpy.py) -- same crop+shade cost as (1),
     blitted at an isometric screen position instead of a flat one.
  3. the Phase 0 Cython prototype of the same kernel
     (native/iso_render/iso_kernel.pyx, must be built first -- see
     native/iso_render/README.md).
  4. per-touched-tile incremental cost: one tile's composite into an
     already-allocated, already-warm canvas -- the shape of Phase 4's
     bounded-region redraw, not a full-map render -- for both backends.
     This is the number the plan's Phase 0 recommendation names explicitly
     (the ~100ms/touched-tile target that decision #6's edit-disable scope
     cut is conditioned on), and a full-map µs/tile average does not answer
     it: that average amortizes canvas allocation and gets sequential-write
     cache behavior a single incremental composite into a warm canvas does
     not get.

Datasets: the two real example files the plan names (2_Joan_coop_2,
C2_ElCid_coop_4) plus a synthetic 480x480 "Ludicrous" worst case, matching
the v1.6 section's own synthetic-scenario benchmark methodology
(no real example file is anywhere near AoE2:DE's largest map size).

Also sanity-checks that the numpy and Cython kernels produce byte-identical
output on every dataset -- if the Cython port has a bug, the timing
numbers below aren't trustworthy either.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "native" / "iso_render"))

import numpy as np

from descape import asset_source
from descape.render import render_terrain, tile_pixels_for_map
from descape.scenario_io import load_map_and_units

from iso_bench_numpy import iso_canvas_size_and_origin, render_terrain_iso_numpy, render_tile_iso

try:
    from iso_kernel import render_terrain_iso_cython, render_tile_iso_single
except ImportError as e:
    print(
        "ERROR: iso_kernel extension not built. Run from native/iso_render/:\n"
        "  ../../.venv/bin/python setup.py build_ext --inplace\n"
        f"(import error: {e})",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass
class SyntheticTile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0  # grass -- see terrain_texture_map.json


def synthetic_ludicrous_tiles(size: int = 480) -> list[SyntheticTile]:
    """A 480x480 tile set with the same elevation range as the plan's other
    synthetic-worst-case benchmarks (0..6), not a flat map -- elevation
    shading cost is part of what's being measured."""
    tiles = []
    for y in range(size):
        for x in range(size):
            tiles.append(SyntheticTile(x=x, y=y, elevation=(x + y) % 7))
    return tiles


def time_it(fn, *args, repeats: int = 1, **kwargs):
    best = None
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best, result


def warm_texture_cache(tiles) -> None:
    """Primes asset_source.get_terrain_texture_array's lru_cache for every
    terrain_id actually present, so the first *timed* call doesn't pay for
    disk reads + Pillow decode/resize of a real .dds -- the 5-11us/tile flat
    number is a steady-state figure, not a cold-cache one, so this keeps
    the comparison apples-to-apples."""
    for tid in {t.terrain_id for t in tiles}:
        asset_source.get_terrain_texture_array(tid)


def bench_incremental(tiles, w: int, h: int, tile_px: int, reps: int = 2000) -> None:
    """Per-touched-tile incremental cost: one representative tile,
    recomposited `reps` times into an already-allocated, already-warm
    canvas -- the shape of a single Phase 4 dirty-tile redraw, not a
    full-map render. Both backends get their own canvas so neither
    benefits from the other having already touched (and cache-warmed) the
    same memory region first."""
    max_elev = max(t.elevation for t in tiles)
    elev_step = tile_px // 4
    canvas_h, canvas_w, origin_x, origin_y = iso_canvas_size_and_origin(w, h, tile_px, max_elev, elev_step)

    tile = tiles[len(tiles) // 2]  # an arbitrary, representative interior tile
    warm_texture_cache([tile])

    img_numpy = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(reps):
        render_tile_iso(img_numpy, tile, tile_px, elev_step, origin_x, origin_y)
    numpy_t = (time.perf_counter() - t0) / reps

    img_cython = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(reps):
        render_tile_iso_single(img_cython, tile, tile_px, elev_step, origin_x, origin_y)
    cython_t = (time.perf_counter() - t0) / reps

    same = np.array_equal(img_numpy, img_cython)
    print(f"  Per-touched-tile incremental (1 tile into a warm canvas, {reps:,} reps, target <100ms):")
    print(f"    numpy:  {numpy_t*1e3:7.4f} ms/tile")
    print(f"    cython: {cython_t*1e3:7.4f} ms/tile")
    print(f"    numpy/cython output byte-identical: {same}")


def bench_dataset(name: str, tiles, w: int, h: int, scenario=None, repeats: int = 3) -> None:
    tile_px = tile_pixels_for_map(w, h)
    n = len(tiles)
    print(f"\n=== {name} ({w}x{h}, {n:,} tiles, tile_px={tile_px}) ===")
    warm_texture_cache(tiles)

    if scenario is not None:
        flat_t, _ = time_it(render_terrain, scenario, repeats=repeats)
        print(f"  Flat  (existing, real render_terrain): {flat_t*1e3:8.1f} ms  ({flat_t/n*1e6:6.2f} us/tile)")
    else:
        flat_t = None
        print("  Flat  (existing, real render_terrain): skipped (no LoadedScenario for synthetic dataset)")

    numpy_t, numpy_img = time_it(render_terrain_iso_numpy, tiles, w, h, tile_px, repeats=repeats)
    print(f"  Iso numpy  (Phase 0 prototype):         {numpy_t*1e3:8.1f} ms  ({numpy_t/n*1e6:6.2f} us/tile)")

    cython_t, cython_img = time_it(render_terrain_iso_cython, tiles, w, h, tile_px, repeats=repeats)
    print(f"  Iso cython (Phase 0 prototype):         {cython_t*1e3:8.1f} ms  ({cython_t/n*1e6:6.2f} us/tile)")

    if flat_t is not None:
        print(f"  Iso numpy / flat ratio:  {numpy_t/flat_t:5.2f}x")
        print(f"  Iso cython / flat ratio: {cython_t/flat_t:5.2f}x")
    print(f"  Cython speedup over numpy: {numpy_t/cython_t:5.2f}x")

    same = np.array_equal(numpy_img, cython_img)
    print(f"  numpy/cython output byte-identical: {same}")
    if not same:
        print("  !!! MISMATCH -- Cython port has a correctness bug, timing numbers below are not trustworthy")

    bench_incremental(tiles, w, h, tile_px)


def main() -> None:
    for fname in (
        "2_Joan_coop_2_v0_15.aoe2scenario",
        "C2_ElCid_coop_4_v0_13.aoe2scenario",
        # 220x220, not the 480x480 "Ludicrous" size it was fetched for,
        # but still the largest real file in this set, and the most
        # terrain/unit-diverse (~13k units, elevation 0..4, heavy
        # forest/water mix), so kept as a third real data point rather
        # than discarded.
        "F7_3_York (865).aoe2scenario",
        # The real thing: confirmed 480x480 via dump_scenario.py's "Map
        # size" line -- exercises the tile_px=32 large-map path for real,
        # not synthetically. Compare directly against the synthetic
        # 480x480 dataset below (same dimensions, same tile_px) to see how
        # much the synthetic stand-in's uniform terrain/elevation pattern
        # differs from a real file's actual mix.
        "F7_2_Dos Pilas (648).aoe2scenario",
    ):
        path = ROOT / "examples" / fname
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        bench_dataset(fname, list(mm.terrain), mm.map_width, mm.map_height, scenario=scenario)

    tiles = synthetic_ludicrous_tiles(480)
    bench_dataset("synthetic 480x480 Ludicrous worst case", tiles, 480, 480, scenario=None)


if __name__ == "__main__":
    main()
