"""Verifies descape.render.SlopedChunkCache -- Phase 6 (Sloped)'s
counterpart to IsoChunkCache/FlatChunkCache, docs/PLAN_V2_6.md's Track C3.
Mirrors tests/test_flat_chunks.py's load-bearing checks for the Sloped
compositor: composite_rect_sloped() must be provably indistinguishable,
pixel-for-pixel, from render_terrain_sloped() regardless of chunk request
order or what's already cached -- the same bar tools/verify_iso_chunks.py
holds IsoChunkCache to.

Checks:
  1. Stitched chunks (render_rect over the WHOLE canvas) == a fresh full
     render_terrain_sloped().
  2. Chunk request order doesn't matter (row-major vs shuffled-reversed).
  3. patch() after a real elevation edit (elevation_tools.set_tile_elevation,
     the real single-tile edit path with propagation) re-establishes
     byte-identity against a fresh full render -- the meaningful edit for
     Sloped, unlike Flat's terrain-paint-only scripted ops.
  4. Eviction round-trips byte-identically.
  5. invalidate_region() round-trips byte-identically.
  6. composite_rect_sloped() mutates nothing it reads (corner_rise array
     stays bit-identical).

A single small scenario (BLANK_TEMPLATE_PATH) with a hand-built sloped
elevation pattern, not the full example corpus: this class's own single-
mip-level scope (see its docstring) doesn't need per-file coverage the way
byte-identity across many real terrain/unit layouts does -- that bar is
already covered by tests/test_sloped_render.py and tests/
test_sloped_geometry.py's partition checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape.elevation_tools import set_tile_elevation
from descape.render import SlopedChunkCache, render_terrain_sloped, sloped_elevations_and_proj, tile_pixels_for_map
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

RNG_SEED = 20260813


def _load_sloped_scenario():
    """A small square scenario with a real (non-flat, non-degenerate) ramp:
    climbs by one elevation level across the map's width -- gentle enough
    to stay within this project's own +-1-elevation-between-neighbors
    invariant (see iso_geometry.ELEV_STEP_DIVISOR's own comment)."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    return scenario


def _make_cache(scenario, **kwargs) -> SlopedChunkCache:
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, **kwargs)


def test_stitched_chunks_match_full_render():
    scenario = _load_sloped_scenario()
    cache = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_terrain_sloped(scenario, with_units=True)
    assert stitched.shape == (canvas_h, canvas_w, 3)
    # SlopedChunkCache's canvas_dims() is tight (no skirt-headroom padding,
    # see its own docstring); render_terrain_sloped's own canvas is padded
    # to match Stepped's shape. Compare only the region the cache actually
    # covers.
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_chunk_order_independence():
    scenario = _load_sloped_scenario()
    cache_a = _make_cache(scenario)
    cache_b = _make_cache(scenario)
    canvas_w, canvas_h = cache_a.canvas_dims()
    n_cx = (canvas_w + cache_a.chunk_px - 1) // cache_a.chunk_px
    n_cy = (canvas_h + cache_a.chunk_px - 1) // cache_a.chunk_px

    keys = [(cx, cy) for cy in range(n_cy) for cx in range(n_cx)]
    rng = np.random.default_rng(RNG_SEED)
    keys_shuffled = keys[:]
    rng.shuffle(keys_shuffled)

    for cx, cy in keys:
        cache_a.get_chunk(0, cx, cy)
    for cx, cy in reversed(keys_shuffled):
        cache_b.get_chunk(0, cx, cy)

    rect_a = cache_a.render_rect(0, 0, canvas_w, canvas_h)
    rect_b = cache_b.render_rect(0, 0, canvas_w, canvas_h)
    assert np.array_equal(rect_a, rect_b)


def test_patch_after_elevation_edit_matches_full_render():
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    cache = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    # Real single-tile edit path, well away from the map edge so its
    # propagation has real neighbors on every side.
    ex, ey = mm.map_width // 4, mm.map_height // 4
    before = np.array([[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)])
    set_tile_elevation(mm, ex, ey, 3)
    after = np.array([[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)])
    changed_ys, changed_xs = np.nonzero(before != after)
    assert changed_xs.size > 0, "set_tile_elevation produced no change -- test setup is broken"

    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    cache.elevations = elevations
    cache.corner_rise = corner_rise

    # Conservative dirty bbox: the swept bbox (any elevation) of every
    # changed tile PLUS a one-tile ring, unioned -- generous on purpose
    # (this test verifies chunk-cache plumbing, not the tight bbox Track
    # C4's own dirty_screen_bbox_sloped will derive).
    from descape import iso_geometry as ig

    x0 = y0 = x1 = y1 = None
    for cx, cy in zip(changed_xs.tolist(), changed_ys.tolist()):
        for nx in range(max(0, cx - 1), min(mm.map_width, cx + 2)):
            for ny in range(max(0, cy - 1), min(mm.map_height, cy + 2)):
                tx0, ty0, tx1, ty1 = ig.tile_screen_bounds_swept(nx, ny, proj)
                x0 = tx0 if x0 is None else min(x0, tx0)
                y0 = ty0 if y0 is None else min(y0, ty0)
                x1 = tx1 if x1 is None else max(x1, tx1)
                y1 = ty1 if y1 is None else max(y1, ty1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas_w, x1), min(canvas_h, y1)

    cache.patch((x0, y0, x1, y1))
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_terrain_sloped(scenario, with_units=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_eviction_round_trips():
    scenario = _load_sloped_scenario()
    cache = _make_cache(scenario, max_chunks=2)
    canvas_w, canvas_h = cache.canvas_dims()
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy < 3:
        pytest.skip(f"only {n_cx * n_cy} chunks at chunk_px={cache.chunk_px}, need >= 3 to exercise eviction")

    first = cache.get_chunk(0, 0, 0).copy()
    touched = 0
    for cy in range(n_cy):
        for cx in range(n_cx):
            if (cx, cy) == (0, 0):
                continue
            cache.get_chunk(0, cx, cy)
            touched += 1
            if touched >= 3:
                break
        if touched >= 3:
            break
    assert (0, 0, 0) not in cache._cache, "chunk (0,0,0) should have been evicted by now"

    refetched = cache.get_chunk(0, 0, 0)
    assert np.array_equal(first, refetched)


def test_invalidate_region_round_trips():
    scenario = _load_sloped_scenario()
    cache = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()
    chunk = cache.get_chunk(0, 0, 0).copy()
    assert (0, 0, 0) in cache._cache

    cache.invalidate_region((0, 0, min(canvas_w, cache.chunk_px), min(canvas_h, cache.chunk_px)))
    assert (0, 0, 0) not in cache._cache

    refetched = cache.get_chunk(0, 0, 0)
    assert np.array_equal(chunk, refetched)


def test_composite_rect_sloped_mutates_nothing_it_reads():
    scenario = _load_sloped_scenario()
    cache = _make_cache(scenario)
    corner_rise_before = cache.corner_rise.copy()
    elevations_before = cache.elevations.copy()
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)
    assert np.array_equal(corner_rise_before, cache.corner_rise)
    assert np.array_equal(elevations_before, cache.elevations)
