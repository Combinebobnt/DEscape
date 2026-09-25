#!/usr/bin/env python3
"""Per-stroke-step draw latency, phase-split by cause, across real example
files. Informational only, matching this project's own bench_iso_backend.py/
bench_iso_memory.py convention -- always runs, never pass/fail.

Graduated from a scratch script that established the one thing that makes
the measurement correct: patch() only recomposites RESIDENT chunks, so an
empty cache makes it a no-op and times nothing. Render the whole canvas at
mip 0 first -- exactly what being on screen does before any edit.

Each step is broken into phases so a per-stroke stutter can be attributed to
a cause instead of guessed at:
  stroke_scan      EditHistory.stroke_dirty_indices
  bbox             dirty_screen_bbox_iso/_sloped (Flat: a bare tile-rect
                    union, no elevation term, so this phase is ~0 there)
  refresh_sources  cache._refresh_source_caches(), fed the elevation-changed
                    subset of this step's own edit (empty for a pure
                    terrain-paint step, which is what this bench's own
                    strokes are) -- so this reflects whatever that method
                    actually decides to rebuild, not a fixed cost. For
                    Sloped this ALSO includes what Stepped keeps in
                    level_rebuild below -- SlopedChunkCache has no per-level
                    laziness (see its own docstring), so its corner_rise/
                    building_bboxes/sprites rebuild together whenever ANY
                    elevation changed, every patch(). That is not a bug in
                    this bench; it is the fact being measured.
  level_rebuild    cache._level(mip)'s lazy building_bboxes/sprite rebuild.
                    Stepped only -- folded into refresh_sources for Sloped,
                    always 0 for Flat (no building_bboxes concept there).
  bystander_scan   composite_rect_iso()/composite_rect_sloped()'s own
                    candidates-vs-building_bboxes merge, timed by calling
                    render._bystander_candidates(), the real shared helper,
                    with the real chunk-bucketed grid, so this figure tracks
                    whatever that helper actually costs rather than a copy
                    of it that can drift. Always 0 for Flat.
  composite        the residual: the real _composite_rect() call's own
                    time, minus the bystander figure above.

--edit elevation swaps the terrain-paint stroke for Elevate's +1 raise
(elevation_tools.set_tiles_elevation, propagation included), Stepped/Sloped
only, square maps only. It adds a diff-floor column: the y-extent of the
pixels that actually changed inside the bbox, read as the cache's unpatched
chunks before patch() and its patched chunks after. bbox-w/bbox-h are
reported separately because only y carries the elevation sweep.

ms/step is felt latency, the number to optimize. ms/dirty-tile divides by
the count that matters for the cost model: stroke_dirty_indices dedupes the
stroke's own footprint, so a one-tile cursor advance at brush 9 adds only
the leading edge, not the whole brush's tile count -- the cost is
sublinear-per-tile even where ms/step looks flat. Both are reported so
either question is answerable from the same run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import composite_backend, iso_geometry, render
from descape.beach_edges import apply_beach_ring
from descape.brush import BRUSH_SHAPE_CIRCLE, brush_tiles
from descape.edit_history import EditHistory, tile_state
from descape.elevation_tools import set_tiles_elevation
from descape.render import (
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units

FILES = [
    "0_June_Event_Scenario.aoe2scenario",
    "2_Joan_coop_1_v0_13.aoe2scenario",
    "blank_map.aoe2scenario",
]
STROKE_LEN = 15
BRUSH_SIZES = (1, 9)
TERRAIN_IDS = [0, 5, 10, 2]
# WATER_DEEP -- what --beach-width paints, so the ring has something to ring.
BEACH_WATER_ID = 22


def _ms(seconds: float) -> float:
    return seconds * 1000


def _bystander_scan_ms(candidates, building_bboxes, grid, x0, y0, x1, y1, w) -> float:
    """Times render._bystander_candidates() itself: the real helper both
    composites call, with the real chunk-bucketed grid, not a replicate.

    It used to inline a copy of the block, back when the two composites
    duplicated it and there was nothing to call. A copy is now wrong rather
    than merely redundant: the real path's cost went to near zero with the
    grid while a copy still walks every entry, so the composite column below
    (this figure subtracted from the real _composite_rect time) would
    under-report by whatever the copy cost."""
    t0 = time.perf_counter()
    if building_bboxes:
        render._bystander_candidates(candidates, building_bboxes, grid, x0, y0, x1, y1, w)
    return _ms(time.perf_counter() - t0)


def _make_cache(style: str, scenario, sprites: bool):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites)
        cache.set_sprites_enabled(sprites)
        return cache, elevations, proj, tile_px
    if style == "sloped":
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites)
        cache.set_sprites_enabled(sprites)
        return cache, elevations, proj, tile_px
    if style == "flat":
        cache = FlatChunkCache(scenario, tile_px)
        return cache, None, None, tile_px
    raise ValueError(style)


def _patch_phased(cache, style: str, bbox, elevation_changed: set | None = None) -> dict[str, float]:
    """Reimplements _ChunkCacheBase.patch()'s own loop, timed phase by
    phase -- see this module's docstring for what each phase means. Never
    changes what gets drawn: same chunk lookup, same _composite_rect() call,
    same in-place chunk write patch() itself does.

    elevation_changed is threaded straight into _refresh_source_caches(),
    same as the real patch() does -- omitting this would always measure the
    pre-optimization wholesale rebuild (None), regardless of what actually
    changed. A terrain-paint-only stroke step (this bench's own strokes)
    passes an EMPTY set, not None -- None means "unknown, assume the
    worst", which a real caller that already knows nothing elevation-related
    changed should never pass."""
    phases = {"refresh_sources": 0.0, "level_rebuild": 0.0, "bystander_scan": 0.0, "composite": 0.0}

    t0 = time.perf_counter()
    cache._refresh_source_caches(elevation_changed)
    phases["refresh_sources"] = _ms(time.perf_counter() - t0)

    px0, py0, px1, py1 = bbox
    if px1 <= px0 or py1 <= py0:
        return phases

    for mip in sorted({key[0] for key in cache._cache}):
        lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, bbox)
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        cx0, cy0 = lx0 // cache.chunk_px, ly0 // cache.chunk_px
        cx1, cy1 = (lx1 - 1) // cache.chunk_px, (ly1 - 1) // cache.chunk_px
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                chunk = cache._cache.get((mip, cx, cy))
                if chunk is None:
                    continue
                chunk_x0, chunk_y0 = cx * cache.chunk_px, cy * cache.chunk_px
                chunk_x1, chunk_y1 = chunk_x0 + chunk.shape[1], chunk_y0 + chunk.shape[0]
                ix0, iy0 = max(lx0, chunk_x0), max(ly0, chunk_y0)
                ix1, iy1 = min(lx1, chunk_x1), min(ly1, chunk_y1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue

                if style == "stepped":
                    t0 = time.perf_counter()
                    lvl = cache._level(mip)
                    phases["level_rebuild"] += _ms(time.perf_counter() - t0)
                    proj, building_bboxes, grid = lvl.proj, lvl.building_bboxes, lvl.bystander_grid
                elif style == "sloped":
                    proj, building_bboxes, grid = cache.proj, cache.building_bboxes, cache.bystander_grid
                else:
                    proj = building_bboxes = grid = None

                if proj is not None:
                    map_w = cache.scenario.map_manager.map_width
                    candidates = iso_geometry.tiles_in_screen_rect(
                        ix0, iy0, ix1, iy1, map_w,
                        cache.scenario.map_manager.map_height, proj,
                    )
                    phases["bystander_scan"] += _bystander_scan_ms(
                        candidates, building_bboxes, grid, ix0, iy0, ix1, iy1, map_w
                    )

                t0 = time.perf_counter()
                patched = cache._composite_rect(mip, ix0, iy0, ix1, iy1)
                phases["composite"] += _ms(time.perf_counter() - t0)
                chunk[iy0 - chunk_y0 : iy1 - chunk_y0, ix0 - chunk_x0 : ix1 - chunk_x0] = patched
    return phases


def _diff_y_extent(before, after) -> int:
    """Row span of the pixels that differ between two same-shaped crops, 0 if none."""
    rows = np.flatnonzero((before != after).any(axis=2).any(axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def _run_stroke(
    path: Path, style: str, brush_size: int, sprites: bool, edit: str = "paint", beach_width: int = 0
) -> str:
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    if edit == "elevation" and (style == "flat" or mm.map_width != mm.map_height):
        return f"  {path.name:32s} style={style:7s} skipped (elevation mode needs Stepped/Sloped and a square map)"
    cache, elevations, proj, tile_px = _make_cache(style, scenario, sprites)

    # Realize the chunks covering the whole canvas at mip 0 first -- see
    # this module's docstring for why an unprimed cache times nothing.
    canvas_w, canvas_h = cache.canvas_dims(0)
    cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    hist = EditHistory()
    hist.begin_stroke(mm.terrain)
    stroke_seen_state: dict[int, object] = {}
    stroke_painted: set[tuple[int, int]] = set()
    cx, cy = mm.map_width // 2, mm.map_height // 2

    scan_ms, bbox_ms, dirty_counts, bbox_areas = [], [], [], []
    bbox_ws, bbox_hs, floor_hs = [], [], []
    phase_totals = {"refresh_sources": [], "level_rebuild": [], "bystander_scan": [], "composite": []}

    for step in range(STROKE_LEN):
        x, y = cx + step, cy
        if not (0 <= x < mm.map_width):
            break
        footprint = [
            t for t in brush_tiles(x, y, brush_size, BRUSH_SHAPE_CIRCLE, mm.map_width, mm.map_height)
            if t not in stroke_painted
        ]
        if not footprint:
            continue
        if edit == "elevation":
            set_tiles_elevation(
                mm, [(tx, ty, min(iso_geometry.MAX_ELEVATION, mm.get_tile(tx, ty).elevation + 1)) for tx, ty in footprint]
            )
        else:
            # beach_width > 0 paints a single WATER terrain rather than
            # cycling TERRAIN_IDS: the shoreline only exists around water,
            # and a cycling paint would lay a ring one step and none the
            # next, averaging two different features together.
            paint_id = BEACH_WATER_ID if beach_width else TERRAIN_IDS[step % len(TERRAIN_IDS)]
            for tx, ty in footprint:
                tile = mm.get_tile(tx, ty)
                tile.terrain_id = paint_id
                tile.layer = -1
            if beach_width:
                apply_beach_ring(mm, footprint, paint_id, None, beach_width)
        # The CORE only, never the ring -- see the water/beach plan's hazard
        # 1, and ViewerWindow.on_edit_stroke_tile, which this mirrors.
        stroke_painted.update(footprint)

        t0 = time.perf_counter()
        all_dirty = hist.stroke_dirty_indices(mm.terrain)
        scan_ms.append(_ms(time.perf_counter() - t0))

        new_dirty = {i for i in all_dirty if tile_state(mm.terrain[i]) != stroke_seen_state.get(i)}
        for i in all_dirty:
            stroke_seen_state[i] = tile_state(mm.terrain[i])
        dirty_counts.append(len(new_dirty))

        if style == "flat":
            t0 = time.perf_counter()
            coords = [(mm.terrain[i].x, mm.terrain[i].y) for i in new_dirty]
            rects = [(px * tile_px, py * tile_px, (px + 1) * tile_px, (py + 1) * tile_px) for px, py in coords]
            bbox_ms.append(_ms(time.perf_counter() - t0))
            bbox_areas.append(sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects))
            phases = dict.fromkeys(phase_totals, 0.0)
            for rect in rects:
                step_phases = _patch_phased(cache, style, rect)
                for k in phase_totals:
                    phases[k] += step_phases[k]
        else:
            bbox_fn = dirty_screen_bbox_iso if style == "stepped" else dirty_screen_bbox_sloped
            elevation_changed: set = set()
            t0 = time.perf_counter()
            bbox = bbox_fn(
                scenario, new_dirty, elevations, proj, with_units=True, with_sprites=cache.sprites_enabled,
                elevation_changed=elevation_changed,
            )
            bbox_ms.append(_ms(time.perf_counter() - t0))
            if bbox is None:
                bbox_areas.append(0)
                phases = dict.fromkeys(phase_totals, 0.0)
            else:
                bbox_areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                bbox_ws.append(bbox[2] - bbox[0])
                bbox_hs.append(bbox[3] - bbox[1])
                # Unpatched resident chunks still hold the pre-edit pixels.
                before = cache.render_rect(*bbox, mip=0).copy() if edit == "elevation" else None
                phases = _patch_phased(cache, style, bbox, elevation_changed)
                if before is not None:
                    floor_hs.append(_diff_y_extent(before, cache.render_rect(*bbox, mip=0)))

        for k, totals in phase_totals.items():
            totals.append(phases[k])

    n = len(scan_ms)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    total_patch = [
        phase_totals["refresh_sources"][i] + phase_totals["level_rebuild"][i]
        + phase_totals["bystander_scan"][i] + phase_totals["composite"][i]
        for i in range(n)
    ]
    mean_dirty = mean(dirty_counts)
    mean_step = mean(scan_ms) + mean(bbox_ms) + mean(total_patch)
    ms_per_tile = mean_step / mean_dirty if mean_dirty else 0.0

    return (
        f"  {path.name:32s} style={style:7s} brush={brush_size} sprites={sprites!s:5s} "
        f"ms/step={mean_step:7.2f} ms/dirty-tile={ms_per_tile:6.3f} dirty-tiles={mean_dirty:5.1f} "
        f"bbox-px={mean(bbox_areas):9.0f} bbox-w={mean(bbox_ws):6.0f} bbox-h={mean(bbox_hs):6.0f} "
        + (f"diff-floor-h={mean(floor_hs):6.0f} " if edit == "elevation" else "")
        + "| "
        f"scan={mean(scan_ms):5.2f} bbox={mean(bbox_ms):5.2f} "
        f"refresh_sources={mean(phase_totals['refresh_sources']):6.2f} "
        f"level_rebuild={mean(phase_totals['level_rebuild']):6.2f} "
        f"bystander_scan={mean(phase_totals['bystander_scan']):5.2f} "
        f"composite={mean(phase_totals['composite']):6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    parser.add_argument("--styles", default="stepped,sloped,flat", help="Comma-separated: stepped,sloped,flat")
    parser.add_argument("--edit", choices=("paint", "elevation"), default="paint", help="Stroke kind")
    parser.add_argument(
        "--beach-width",
        type=int,
        default=0,
        help="Auto-beach ring width (0 = plain Draw, the baseline to compare against)",
    )
    parser.add_argument("--files", help="Comma-separated file names, overriding the built-in list")
    args = parser.parse_args()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    names = [n.strip() for n in args.files.split(",")] if args.files else FILES
    print(composite_backend.describe())
    for name in names:
        path = args.scenario_dir / name
        if not path.exists():
            print(f"  {name:32s} skipped (not found in {args.scenario_dir})")
            continue
        for brush_size in BRUSH_SIZES:
            for sprites in (False, True):
                for style in styles:
                    print(
                        _run_stroke(path, style, brush_size, sprites, args.edit, args.beach_width),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
