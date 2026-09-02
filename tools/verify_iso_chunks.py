#!/usr/bin/env python3
"""Verifies descape.render.composite_rect_iso() and IsoChunkCache -- v2.6
completion plan's Track B, Phase B-B. This is the
load-bearing check for Phase B-B: a chunk cache is only useful if it's
provably indistinguishable, pixel-for-pixel, from the existing full-canvas
compositor -- no "seams are acceptable near chunk boundaries" concession,
because render_terrain_iso() remains a fully independent implementation
(see composite_rect_iso()'s own docstring for why that duplication is kept
on purpose).

Checks, per real example file:
  1. Stitched chunks == full render: IsoChunkCache.render_rect() over the
     WHOLE canvas must be np.array_equal to an independent
     render_terrain_iso() call.
  2. Arbitrary non-chunk-aligned rects == full render crop: check 1 alone
     only ever calls render_rect() at x0=y0=0 with chunk-aligned bounds,
     where every per-chunk copy offset in its stitching loop is trivially
     zero or a round chunk_px multiple -- this instead throws ~30 random
     rects per file at it (non-zero origin, size not a multiple of
     chunk_px, straddling chunk seams), the shape any real future viewport
     rect will actually have.
  3. Chunk request order doesn't matter: two fresh caches over the same
     scenario/elevations/proj, chunks fetched in different orders, must
     assemble to byte-identical pixels regardless.
  4. patch() after each of verify_iso_incremental.py's own scripted edit
     ops (terrain paint, elevation raise/lower, a big jump, then their
     undos) re-establishes byte-identity against a fresh full render --
     the same bar refresh_region_iso() itself has to clear.
  5. Eviction round-trips byte-identically: forcing a chunk out of a
     small-max_chunks cache and re-fetching it reproduces the exact same
     pixels it had before eviction.
  6. invalidate_region() round-trips byte-identically: evicting a region
     from a fully-warmed cache and re-fetching it reproduces a fresh full
     render -- exercises invalidate_region() directly (distinct from
     check 5's LRU-driven eviction).
  7. composite_rect_iso() never mutates elevations -- called directly,
     elevations must be bit-identical before and after.

CHUNK_PX benching (256/512/1024 on the largest real map, to pick
DEFAULT_CHUNK_PX) used to live here, informational and not pass/fail --
moved to tools/bench_chunk_px.py (the pytest migration plan's Ordering
step 4), since it was never a check_ prefixed pass/fail check to begin
with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation
from descape.render import (
    composite_rect_iso,
    dirty_screen_bbox_iso,
    render_terrain_iso,
    render_terrain_iso_with_proj,
    tile_pixels_for_map,
)
from descape.render_cache import IsoChunkCache
from descape.scenario_io import load_map_and_units

RNG_SEED = 20260805


def _editable(scenario) -> bool:
    return scenario.terrain_write_supported and scenario.map_is_square


def _pick_different_terrain(current: int) -> int:
    return 15 if current == 2 else 2


def _scripted_ops(mm):
    """Same fixed op sequence verify_iso_incremental.py's own _scripted_ops
    uses -- duplicated here (each tools/verify_*.py script is self-
    contained, per project convention) rather than imported, so this
    script's checks are the same bar refresh_region_iso() itself has to
    clear."""
    w, h = mm.map_width, mm.map_height
    cx, cy = w // 2, h // 2
    paint_a = (min(2, w - 1), 0)
    paint_b = (0, min(2, h - 1))

    def op_paint():
        for x, y in (paint_a, paint_b):
            tile = mm.get_tile(x, y)
            tile.terrain_id = _pick_different_terrain(tile.terrain_id)
            tile.layer = -1

    def op_raise():
        tile = mm.get_tile(cx, cy)
        # A plain min(MAX_ELEVATION, elevation + 1) is a silent no-op when the
        # tile is already at the ceiling, which produces no dirty tiles and
        # silently skips this op instead of exercising it -- raise if below
        # the ceiling, otherwise lower, so the op always produces a real change.
        target = tile.elevation + 1 if tile.elevation < iso_geometry.MAX_ELEVATION else max(iso_geometry.MIN_ELEVATION, tile.elevation - 1)
        set_tile_elevation(mm, cx, cy, target)

    def op_lower():
        tile = mm.get_tile(cx, cy)
        set_tile_elevation(mm, cx, cy, max(iso_geometry.MIN_ELEVATION, tile.elevation - 1))

    def op_big_jump():
        tx, ty = min(cx + 3, w - 1), min(cy + 3, h - 1)
        tile = mm.get_tile(tx, ty)
        target = iso_geometry.MAX_ELEVATION if tile.elevation < iso_geometry.MAX_ELEVATION else iso_geometry.MIN_ELEVATION
        set_tile_elevation(mm, tx, ty, target)

    return [("terrain paint", op_paint), ("elevation raise", op_raise), ("elevation lower", op_lower), ("big jump", op_big_jump)]


def check_stitched_matches_full(files: list[Path]) -> tuple[bool, str]:
    problems = []
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        _img, elevations, proj = render_terrain_iso_with_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_terrain_iso(scenario)
        if stitched.shape != full.shape:
            problems.append(f"{path.name}: shape mismatch, stitched {stitched.shape} vs full {full.shape}")
            continue
        if not np.array_equal(stitched, full):
            diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
            problems.append(f"{path.name}: {diff} pixels differ from a full render")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(files)} files, chunked assembly byte-identical to a full render)"


def check_order_independence(files: list[Path]) -> tuple[bool, str]:
    rng = np.random.default_rng(RNG_SEED)
    problems = []
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        _img, elevations_a, proj = render_terrain_iso_with_proj(scenario)
        elevations_b = elevations_a.copy()

        cache_a = IsoChunkCache(scenario, elevations_a, proj, tile_px)
        cache_b = IsoChunkCache(scenario, elevations_b, proj, tile_px)
        canvas_w, canvas_h = cache_a.canvas_dims()
        n_cx = (canvas_w + cache_a.chunk_px - 1) // cache_a.chunk_px
        n_cy = (canvas_h + cache_a.chunk_px - 1) // cache_a.chunk_px

        keys = [(cx, cy) for cy in range(n_cy) for cx in range(n_cx)]
        keys_shuffled = keys[:]
        rng.shuffle(keys_shuffled)

        for cx, cy in keys:  # forward, row-major
            cache_a.get_chunk(0, cx, cy)
        for cx, cy in reversed(keys_shuffled):  # shuffled AND reversed
            cache_b.get_chunk(0, cx, cy)

        rect_a = cache_a.render_rect(0, 0, canvas_w, canvas_h)
        rect_b = cache_b.render_rect(0, 0, canvas_w, canvas_h)
        if not np.array_equal(rect_a, rect_b):
            diff = int(np.count_nonzero(np.any(rect_a != rect_b, axis=2)))
            problems.append(f"{path.name}: {diff} pixels differ between fetch orders")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(files)} files, row-major vs shuffled-reversed chunk fetch orders produced identical pixels)"


def check_patch_after_scripted_ops(files: list[Path]) -> tuple[bool, str]:
    problems = []
    n_checked = 0
    for path in files:
        scenario = load_map_and_units(path)
        if not _editable(scenario):
            continue
        n_checked += 1
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        _img, elevations, proj = render_terrain_iso_with_proj(scenario)

        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()
        cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk so patch() has something to patch

        hist = EditHistory()
        ops = _scripted_ops(mm)

        for label, apply_fn in ops:
            hist.begin_stroke(mm.terrain)
            apply_fn()
            dirty = hist.commit_stroke(label, mm.terrain)
            if not dirty:
                problems.append(f"{path.name} [{label}]: produced no dirty tiles -- expected a real change")
                continue
            bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
            if bbox is None:
                problems.append(f"{path.name} [{label}]: dirty_screen_bbox_iso declined unexpectedly")
                continue
            cache.patch(bbox)
            stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
            full = render_terrain_iso(scenario)
            if not np.array_equal(stitched, full):
                diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
                problems.append(f"{path.name} [{label}]: patch() result differs from a full re-composite at {diff} px")

        for _ in ops:
            dirty = hist.undo(mm.terrain)
            if not dirty:
                continue
            bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
            if bbox is None:
                problems.append(f"{path.name} [undo]: dirty_screen_bbox_iso declined unexpectedly")
                continue
            cache.patch(bbox)

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_terrain_iso(scenario)
        if not np.array_equal(stitched, full):
            diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
            problems.append(f"{path.name} [undo-to-start]: patch() result differs from a full re-composite at {diff} px")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({n_checked} editable files, patch() after every scripted op + full undo matched a fresh full render)"


def check_eviction_round_trips(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    _img, elevations, proj = render_terrain_iso_with_proj(scenario)

    cache = IsoChunkCache(scenario, elevations, proj, tile_px, max_chunks=2)
    canvas_w, canvas_h = cache.canvas_dims()
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy < 3:
        return True, f"skipped ({path.name} has only {n_cx * n_cy} chunks at chunk_px={cache.chunk_px}, need >= 3)"

    first_key = (0, 0, 0)
    first = cache.get_chunk(*first_key).copy()
    # Touch enough OTHER chunks to force (0,0,0) out of a max_chunks=2 cache.
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
    if first_key in cache._cache:
        return False, f"{path.name}: chunk (0,0,0) was still cached after touching {touched} others at max_chunks=2"

    refetched = cache.get_chunk(*first_key)
    if not np.array_equal(first, refetched):
        diff = int(np.count_nonzero(np.any(first != refetched, axis=2)))
        return False, f"{path.name}: re-fetched evicted chunk differs from its original at {diff} pixels"
    return True, f"OK ({path.name}, chunk (0,0,0) evicted after {touched} others then re-fetched byte-identical)"


def check_arbitrary_rects_match_full(files: list[Path]) -> tuple[bool, str]:
    """render_rect()'s own stitching arithmetic is only exercised at
    x0=y0=0, chunk-aligned bounds by the "stitched == full render" check
    above -- with a zero origin and chunk-aligned edges, every
    per-chunk offset in render_rect()'s copy loop is trivially zero or a
    round chunk_px multiple, so a transposed or off-by-one slice index
    there could still pass that check. Real (future) viewport rects have
    neither property: non-zero origin, not a multiple of chunk_px, and
    straddling chunk seams. Compares render_rect() against the
    corresponding crop of an independent full render for random such
    rects."""
    rng = np.random.default_rng(RNG_SEED)
    problems = []
    total_rects = 0
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        _img, elevations, proj = render_terrain_iso_with_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()
        full = render_terrain_iso(scenario)

        for _ in range(30):
            x0 = int(rng.integers(0, canvas_w))
            y0 = int(rng.integers(0, canvas_h))
            rw = int(rng.integers(1, max(2, canvas_w // 3)))
            rh = int(rng.integers(1, max(2, canvas_h // 3)))
            x1, y1 = min(canvas_w, x0 + rw), min(canvas_h, y0 + rh)
            if x1 <= x0 or y1 <= y0:
                continue
            total_rects += 1
            rect = cache.render_rect(x0, y0, x1, y1)
            expected = full[y0:y1, x0:x1]
            if not np.array_equal(rect, expected):
                diff = int(np.count_nonzero(np.any(rect != expected, axis=2)))
                problems.append(f"{path.name} rect=({x0},{y0},{x1},{y1}): {diff} pixels differ")

    if problems:
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        return False, shown + more
    return True, f"OK ({len(files)} files, {total_rects} random non-chunk-aligned rects matched a full render crop)"


def check_invalidate_region_round_trips(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    _img, elevations, proj = render_terrain_iso_with_proj(scenario)
    # max_chunks generously large -- default (256) is smaller than some real
    # files' full chunk count at chunk_px=512 (e.g. 200x200 -> 25x13=325
    # chunks), which would silently LRU-evict chunk (0,0,0) during the warm
    # below before this check ever calls invalidate_region(), making the
    # "evicted nothing" guard below fire for the wrong reason.
    cache = IsoChunkCache(scenario, elevations, proj, tile_px, max_chunks=100000)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    bbox = (0, 0, min(cache.chunk_px, canvas_w), min(cache.chunk_px, canvas_h))
    keys_before = set(cache._cache)
    cache.invalidate_region(bbox)
    evicted = keys_before - set(cache._cache)
    if not evicted:
        return False, f"{path.name}: invalidate_region({bbox}) evicted nothing (test doesn't actually exercise it)"

    refetched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_terrain_iso(scenario)
    if not np.array_equal(refetched, full):
        diff = int(np.count_nonzero(np.any(refetched != full, axis=2)))
        return False, f"{path.name}: after invalidate_region + re-fetch, {diff} pixels differ from a full render"
    return True, f"OK ({path.name}, invalidate_region evicted {len(evicted)} chunk(s), re-fetch matched a full render)"


def check_composite_rect_iso_elevations_untouched(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    _img, elevations, proj = render_terrain_iso_with_proj(scenario)
    before = elevations.copy()

    from descape.render import _building_bboxes_iso, _units_by_tile

    units_by_tile = _units_by_tile(scenario)
    building_bboxes = _building_bboxes_iso(units_by_tile, mm.map_width, mm.map_height, proj, elevations)
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h + (proj.max_elev - proj.min_elev) * proj.elev_step
    composite_rect_iso(
        scenario, 0, 0, min(1024, canvas_w), min(1024, canvas_h), elevations, proj, tile_px, units_by_tile, building_bboxes
    )

    if not np.array_equal(before, elevations):
        diff = int(np.count_nonzero(before != elevations))
        return False, f"{path.name}: elevations changed at {diff} entries after composite_rect_iso()"
    return True, f"OK ({path.name}, elevations bit-identical before/after composite_rect_iso())"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {args.scenario_dir}")
        sys.exit(1)

    checks = [
        ("Stitched chunks == full render", lambda: check_stitched_matches_full(files)),
        ("Arbitrary non-chunk-aligned rects == full render crop", lambda: check_arbitrary_rects_match_full(files)),
        ("Chunk request order doesn't matter", lambda: check_order_independence(files)),
        ("patch() after scripted ops matches full render", lambda: check_patch_after_scripted_ops(files)),
        ("Eviction round-trips byte-identically", lambda: check_eviction_round_trips(files)),
        ("invalidate_region round-trips byte-identically", lambda: check_invalidate_region_round_trips(files)),
        ("composite_rect_iso() leaves elevations untouched", lambda: check_composite_rect_iso_elevations_untouched(files)),
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
