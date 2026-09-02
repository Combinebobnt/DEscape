#!/usr/bin/env python3
"""Sprite rendering cost -- the measurement gate P3-g3 ships behind.

Informational only, matching this project's bench_iso_backend.py /
bench_incremental_latency.py convention: always runs, never pass/fail. Needs a
real AoE2:DE install configured, since sprite pixels are read from it at
runtime and nothing is bundled; with no install every unit falls back to a
coloured mark and every number below is zero by construction.

Four things get measured, because they fail in different ways:

- **Distinct sprite keys per file.** Decode cost scales with distinct
  (graphic, frame, player) combinations actually used, NOT with the unit count
  -- most GAIA clutter and most buildings of one type share a handful of
  graphics. This is also what sizes the scaled LRU: a capacity below a real
  file's distinct count means every rebuild re-decodes everything, which turns
  a cache into pure overhead.
- **Layer build, cold and warm.** sprite_draws_by_anchor() is the
  per-mip-level lazy rebuild, so it is paid once per edit per resident level,
  right alongside the ~15-20ms units_by_tile/building_bboxes rebuild that
  already exists there. Cold is the first paint after opening a file; warm is
  every edit after that.
- **Worst-case single-frame resolution.** Reported separately and on purpose,
  not folded into the warm average: sld_decoder holds no cache by design, and
  the longest measured delta chain in the real install is 313 links, so one
  wanted frame can pay for hundreds of predecessor decodes. An average hides
  exactly the case that would show up as a visible stall.
- **Composite cost per chunk, with and against without.** The sprite branch in
  _paint_tile_and_units_iso is the part that runs on every chunk fetch, so it
  is what a pan or a zoom pays.

Peak resident bytes of both caches are reported alongside, since the capacity
decision is a memory/latency trade rather than a pure latency one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, iso_geometry, render, settings, unit_sprites
from descape.scenario_io import load_map_and_units

CHUNK_PX = 512


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _cache_bytes() -> tuple[int, int]:
    native = sum(
        sum(a.nbytes for a in value[:2] if a is not None)
        for value in unit_sprites._native_cache.values()
    )
    # _MISS entries are a sentinel, not an array, and carry no pixels.
    scaled = sum(
        d.rgba.nbytes for d in unit_sprites._scaled_cache.values() if d is not unit_sprites._MISS
    )
    return native, scaled


def _projection(scenario):
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = render.tile_pixels_for_map(w, h)
    _, elevations = render._terrain_grid_and_elevations(scenario)
    proj = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION,
        elev_step_pct=settings.get_elev_step_pct(),
    )
    return elevations, proj, tile_px


def _distinct_keys(scenario, proj) -> tuple[int, int, int]:
    """(unit count, distinct (file_name, frame) pairs, units that resolve)."""
    graphics = unit_sprites.graphic_map()
    seen = set()
    total = resolved = 0
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            total += 1
            entry = graphics.get(unit.unit_const)
            if entry is None:
                continue
            rotation = 0.0 if player_id == 0 else float(unit.rotation)
            index = unit_sprites.angle_index(rotation, max(1, entry["angle_count"]))
            seen.add((entry["file_name"], index * max(1, entry["frame_count"])))
            resolved += 1
    return total, len(seen), resolved


def _worst_frame_resolution(scenario, proj) -> tuple[str, float]:
    """The slowest single cold sprite_for() over this file's distinct keys.

    Cold means the caches are dropped between attempts, so each timing pays
    the whole delta chain behind that frame -- which is the point.
    """
    graphics = unit_sprites.graphic_map()
    worst = ("", 0.0)
    seen = set()
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            entry = graphics.get(unit.unit_const)
            if entry is None or entry["file_name"] in seen:
                continue
            seen.add(entry["file_name"])
            rotation = 0.0 if player_id == 0 else float(unit.rotation)
            unit_sprites.clear_caches()
            start = time.perf_counter()
            got = unit_sprites.sprite_for(unit.unit_const, rotation, player_id, proj.half_w)
            elapsed = time.perf_counter() - start
            if got is not None and elapsed > worst[1]:
                worst = (entry["file_name"], elapsed)
    return worst


def bench_file(path: Path) -> str:
    scenario = load_map_and_units(path)
    elevations, proj, tile_px = _projection(scenario)
    lines = [f"  {path.name}"]

    total, distinct, resolved = _distinct_keys(scenario, proj)
    pct = 100 * resolved / total if total else 0.0
    lines.append(
        f"    units {total} | resolve to a sprite {resolved} ({pct:.1f}%) | "
        f"distinct (file, frame) {distinct}"
    )

    unit_sprites.clear_caches()
    start = time.perf_counter()
    layer = render.sprite_draws_by_anchor(scenario, proj, elevations)
    cold = time.perf_counter() - start
    warm = min(
        _time(lambda: render.sprite_draws_by_anchor(scenario, proj, elevations)) for _ in range(3)
    )
    native_b, scaled_b = _cache_bytes()
    lines.append(
        f"    layer build: cold {_ms(cold)} | warm {_ms(warm)} | "
        f"sprites drawn {len(layer.skip_ids)} at {len(layer.by_anchor)} anchors"
    )
    lines.append(
        f"    cache resident: native {native_b / 1e6:.1f}MB in "
        f"{len(unit_sprites._native_cache)} entries | scaled {scaled_b / 1e6:.1f}MB in "
        f"{len(unit_sprites._scaled_cache)} entries"
        + ("  <-- AT CAPACITY, so it is thrashing" if len(unit_sprites._scaled_cache)
           >= unit_sprites._scaled_cache.capacity else "")
    )

    name, worst = _worst_frame_resolution(scenario, proj)
    lines.append(f"    worst single cold frame: {_ms(worst)} ({name})")

    unit_sprites.clear_caches()
    layer = render.sprite_draws_by_anchor(scenario, proj, elevations)
    units_by_tile = render._units_by_tile(scenario)
    mm = scenario.map_manager
    bboxes = render._building_bboxes_iso(units_by_tile, mm.map_width, mm.map_height, proj, elevations)
    merged = render.merge_sprite_bboxes(bboxes, layer)
    canvas_w, canvas_h = render._canvas_pixel_dims(proj)
    rects = _sample_rects(canvas_w, canvas_h)

    def composite(bbs, sprites):
        for x0, y0 in rects:
            render.composite_rect_iso(
                scenario, x0, y0, min(x0 + CHUNK_PX, canvas_w), min(y0 + CHUNK_PX, canvas_h),
                elevations, proj, tile_px, units_by_tile, bbs, sprites=sprites,
            )

    without = min(_time(lambda: composite(bboxes, None)) for _ in range(3)) / len(rects)
    with_ = min(_time(lambda: composite(merged, layer)) for _ in range(3)) / len(rects)
    ratio = with_ / without if without else float("inf")
    lines.append(
        f"    composite per {CHUNK_PX}px chunk: marks {_ms(without)} | "
        f"sprites {_ms(with_)} ({ratio:.2f}x) over {len(rects)} chunks"
    )
    return "\n".join(lines)


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _sample_rects(canvas_w: int, canvas_h: int) -> list[tuple[int, int]]:
    """A band of chunks across the canvas middle, where a real map's units are
    densest -- a corner chunk on a big map is usually empty water or edge, and
    timing those would flatter the sprite path rather than test it."""
    ys = [canvas_h // 2 - CHUNK_PX, canvas_h // 2]
    xs = list(range(CHUNK_PX, min(canvas_w, 5 * CHUNK_PX), CHUNK_PX))
    return [(x, max(0, y)) for y in ys for x in xs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scenarios", type=Path, nargs="*",
        help="Specific .aoe2scenario files (default: the biggest few in examples/)",
    )
    parser.add_argument("--install", type=Path, help="AoE2:DE install root override")
    args = parser.parse_args()

    if args.install:
        asset_source.set_install_path_override(args.install)
    if asset_source.get_install_path() is None:
        raise SystemExit(
            "No AoE2:DE install configured -- every unit would fall back to a "
            "coloured mark and every number here would be zero. Pass --install."
        )

    files = args.scenarios
    if not files:
        candidates = sorted((ROOT / "examples").glob("*.aoe2scenario"))
        files = sorted(candidates, key=lambda p: p.stat().st_size)[-3:]
    if not files:
        raise SystemExit("No scenarios to measure")

    print(f"Sprite bench -- install {asset_source.get_install_path()}")
    print(
        f"caches: native capacity {unit_sprites.NATIVE_CACHE_SIZE}, "
        f"scaled capacity {unit_sprites.SCALED_CACHE_SIZE}"
    )
    worsts = []
    for path in files:
        print(bench_file(path))
        worsts.append(path)
    print(f"\nMeasured {len(worsts)} file(s). See this module's docstring for what each line means.")


if __name__ == "__main__":
    main()
