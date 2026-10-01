#!/usr/bin/env python3
"""Sprite rendering cost -- the measurement gate P3-g3 ships behind.

Informational only, matching this project's bench_iso_backend.py /
bench_incremental_latency.py convention: always runs, never pass/fail. Needs a
real AoE2:DE install configured, since sprite pixels are read from it at
runtime and nothing is bundled; with no install every unit falls back to a
coloured mark and every number below is zero by construction.

Five things get measured, because they fail in different ways:

- **Distinct sprite keys per file.** Decode cost scales with distinct
  (graphic, frame, player) combinations actually used, NOT with the unit count
  -- most GAIA clutter and most buildings of one type share a handful of
  graphics. This is also what sizes SCALED_CACHE_BYTES: a byte budget below a
  real file's resident-level working set means every rebuild re-decodes
  everything, which turns a cache into pure overhead.
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
- **Flat's icon layer, per mip level** (P3-g7). Flat's own arm, because it adds
  an axis Stepped does not have: an icon is FOOTPRINT-sized, so a level's whole
  working set quadruples with each step up the ladder, and ICON_CACHE_BYTES has
  to exceed the biggest level's or that level thrashes.

Peak resident bytes of every cache are reported alongside, since the capacity
decision is a memory/latency trade rather than a pure latency one.

`--cold-levels` replaces all of the above with the cold-first-paint row (see
_bench_cold_levels): one process builds Stepped level A cold, then its two
finer neighbours B and C, then A again, each through a fresh IsoChunkCache's
_level() with the process-wide sprite caches kept, as a reopen or a zoom does.
Each build reports its time, how many .sld files it walked and how many
whole-file .sld reads it made (a walk plus any decode whose bytes were no
longer cached). A Sloped row
then builds level A cold and times Sloped's eager construction after it, the
Stepped-to-Sloped switch.

`--ladder` replaces all of the above with the zoom-in row (see bench_ladder):
one IsoChunkCache builds every Stepped level coarsest to finest through
_level(), as a load followed by zooming in does, then a fresh IsoChunkCache
builds them again with the process-wide sprite caches kept (pass `rewarm`,
the floor a level still costs when a paint reaches it before its warm).
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, iso_geometry, render, settings, unit_sprites
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units

CHUNK_PX = 512


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _cache_bytes() -> tuple[int, int]:
    # _MISS entries are a sentinel, not an array, and carry no pixels. It lives
    # in BOTH LRUs as of P3-g7 (it moved down into the native one, whose key is
    # scale-independent) -- so both sums have to skip it, not just the scaled.
    native = sum(
        sum(a.nbytes for a in value[:2] if a is not None)
        for value in unit_sprites._native_cache.values()
        if value is not unit_sprites._MISS
    )
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
    scaled_cache = unit_sprites._scaled_cache
    lines.append(
        f"    cache resident: native {native_b / 1e6:.1f}MB in "
        f"{len(unit_sprites._native_cache)} entries | scaled {scaled_b / 1e6:.1f}MB in "
        f"{len(scaled_cache)} entries (budget {scaled_cache.capacity_bytes / 1e6:.0f}MB)"
        + ("  <-- AT CAPACITY, so it is thrashing"
           if scaled_cache._bytes >= scaled_cache.capacity_bytes else "")
    )

    name, worst = _worst_frame_resolution(scenario, proj)
    lines.append(f"    worst single cold frame: {_ms(worst)} ({name})")

    lines.extend(_bench_stepped_sprites(scenario, proj, elevations))
    lines.extend(_bench_flat_icons(scenario, tile_px))

    unit_sprites.clear_caches()
    layer = render.sprite_draws_by_anchor(scenario, proj, elevations)
    units_by_tile = render._units_by_tile(scenario)
    mm = scenario.map_manager
    bboxes = render._building_bboxes_iso(scenario, mm.map_width, mm.map_height, proj, elevations)
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


def _bench_stepped_sprites(scenario, proj, elevations) -> list[str]:
    """Stepped's scaled-sprite arm (Item 24), an adjacent-level thrashing probe.

    Mirrors _bench_flat_icons's structure, but the axis that matters for
    _scaled_cache is ADJACENT mip levels sharing one byte budget -- the real
    level warm keeps the current level +/-1 resident (level_warm.neighbour_mips)
    -- not repeated same-level rebuilds. So each pair builds level A cold,
    builds its neighbour B, then rebuilds A: a rebuild near A's own cold cost
    means B evicted it. mip_projections_for(), not the raw candidate dict,
    because Stepped's elev_step term is not exact at every candidate tile_px
    the way Flat's icons are.
    """
    mm = scenario.map_manager
    projs = iso_geometry.mip_projections_for(
        mm.map_width, mm.map_height, proj, settings.get_elev_step_pct()
    )
    levels = sorted(projs)
    lines = ["    stepped sprites, adjacent-level thrashing (Item 24):"]
    for level_a, level_b in pairwise(levels):
        proj_a, proj_b = projs[level_a], projs[level_b]
        unit_sprites.clear_caches()
        cold = _time(lambda: render.sprite_draws_by_anchor(scenario, proj_a, elevations))
        _time(lambda: render.sprite_draws_by_anchor(scenario, proj_b, elevations))
        rebuild = _time(lambda: render.sprite_draws_by_anchor(scenario, proj_a, elevations))
        ratio = rebuild / cold if cold else 0.0
        flag = "  <-- THRASHING" if rebuild >= 0.5 * cold else ""
        lines.append(
            f"      mip {level_a:>2}/{level_b:>2} tile_px {proj_a.tile_px:>3}/{proj_b.tile_px:>3}: "
            f"cold {_ms(cold)} | rebuild after neighbour {_ms(rebuild)} ({ratio:.2f}x){flag}"
        )
    scaled_cache = unit_sprites._scaled_cache
    lines.append(
        f"      scaled cache: {scaled_cache._bytes / 1e6:.1f}MB in {len(scaled_cache)} entries "
        f"(budget {scaled_cache.capacity_bytes / 1e6:.0f}MB)"
    )
    return lines


def _bench_flat_icons(scenario, tile_px: int) -> list[str]:
    """Flat's icon-layer arm (P3-g7), per mip level.

    Per LEVEL rather than once, because that is the axis Flat adds: an icon
    entry is FOOTPRINT-sized, so its bytes quadruple with each step up the
    ladder and a budget that fits one level can thrash on the next. The whole
    point of ICON_CACHE_BYTES is to exceed the biggest level's working set --
    a warm rebuild that matches its own cold one is the tell that it does not,
    the same failure NATIVE_CACHE_BYTES/SCALED_CACHE_BYTES's comment records.

    Flat multiplies the cold build by RESIDENT level count, which Stepped
    already does too, so that is accepted by precedent rather than new -- but
    it is worth a number rather than being left undiscussed.
    """
    unit_sprites.clear_caches()
    unit_sprites._icon_cache.clear()
    lines = ["    flat icons (P3-g7):"]
    for level, px in sorted(iso_geometry.mip_tile_px_candidates(tile_px).items()):
        cold = _time(lambda: render._flat_icon_layer(scenario, px))
        warm = min(_time(lambda: render._flat_icon_layer(scenario, px)) for _ in range(3))
        icons, rows = render._flat_icon_layer(scenario, px)
        flag = "  <-- WARM MATCHES COLD, so the icon cache is thrashing" if warm > 0.5 * cold else ""
        lines.append(
            f"      mip {level:>2} tile_px {px:>3}: cold {_ms(cold)} | warm {_ms(warm)} | "
            f"{len(icons)}/{rows} icons{flag}"
        )
    cache = unit_sprites._icon_cache
    sizes = sorted(v.rgba.nbytes for v in cache.values() if isinstance(v, unit_sprites.SpriteDraw))
    lines.append(
        f"      icon cache: {cache._bytes / 1e6:.1f}MB in {len(cache)} entries "
        f"(budget {cache.capacity_bytes / 1e6:.0f}MB)"
        + (f" | entry bytes min {sizes[0]} median {sizes[len(sizes) // 2]} max {sizes[-1]}" if sizes else "")
    )
    return lines


class _WalkCounter:
    """Counts .sld walks by swapping unit_sprites.load_sld, the one name every
    sprite path walks a file through, with or without a per-file index cache,
    and whole-file .sld reads by wrapping Path.read_bytes, which every walk and
    decode reads through with or without a bytes cache.

    A call that finds no file still counts: it costs the open attempt, and on
    the pre-index code it recurred on every native miss just like a walk."""

    def __init__(self):
        self.paths: list[str] = []
        self.reads = 0
        self._real = None
        self._real_read = None

    def __enter__(self):
        self._real = real = unit_sprites.load_sld
        self._real_read = real_read = Path.read_bytes

        def counted(*args, **kwargs):
            self.paths.append(str(args[0]))
            return real(*args, **kwargs)

        def counted_read(path):
            if path.suffix == ".sld":
                self.reads += 1
            return real_read(path)

        unit_sprites.load_sld = counted
        Path.read_bytes = counted_read
        return self

    def __exit__(self, *_exc):
        unit_sprites.load_sld = self._real
        Path.read_bytes = self._real_read

    def take(self) -> tuple[int, int, int]:
        """(walks, distinct files, whole-file reads) since the last take()."""
        walks, files, reads = len(self.paths), len(set(self.paths)), self.reads
        self.paths.clear()
        self.reads = 0
        return walks, files, reads


def _cold_level_order(levels: list[int], level_a: int | None) -> list[int]:
    """[A, B, C, A]: A defaults to the coarsest level, where a whole-map first
    paint lands; B and C are the next two finer, or the nearest two others."""
    a = levels[0] if level_a is None else level_a
    if a not in levels:
        raise SystemExit(f"--level {a} is not one of this file's mip levels {levels}")
    others = sorted((lvl for lvl in levels if lvl != a), key=lambda lvl: (abs(lvl - a), -lvl))
    return [a, *others[:2], a]


def bench_cold_levels(path: Path, level_a: int | None) -> str:
    """The cold-first-paint oracle for the SLD index work. Every build uses a
    fresh IsoChunkCache so its own level state is cold while unit_sprites'
    caches stay as the previous build left them; only _level() is timed."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = render.elevations_and_proj(scenario)

    def stepped_cache() -> IsoChunkCache:
        return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)

    levels = stepped_cache().mip_levels()
    order = _cold_level_order(levels, level_a)
    lines = [f"  {path.name}", f"    stepped cold levels {order} (mip levels {levels}):"]
    labels = ["cold"] + ["neighbour"] * (len(order) - 2) + ["revisit"]

    unit_sprites.clear_caches()
    with _WalkCounter() as counter:
        for mip, label in zip(order, labels, strict=True):
            cache = stepped_cache()
            elapsed = _time(lambda cache=cache, mip=mip: cache._level(mip))
            walks, files, reads = counter.take()
            lines.append(
                f"      mip {mip:>2} tile_px {cache.mip_tile_px(mip):>3} {label:<9}: {_ms(elapsed):>10} | "
                f".sld walks {walks} over {files} files, {reads} reads | "
                f"native cache {len(unit_sprites._native_cache)} entries"
            )
        lines.extend(_index_line())

        sloped_elevations, corner_rise, sloped_proj = render.sloped_elevations_and_proj(scenario)
        unit_sprites.clear_caches()
        cache = stepped_cache()
        stepped = _time(lambda: cache._level(order[0]))
        s_walks, s_files, s_reads = counter.take()
        sloped = _time(
            lambda: SlopedChunkCache(scenario, sloped_elevations, corner_rise, sloped_proj, tile_px, sprites=True)
        )
        walks, files, reads = counter.take()
        lines.append(
            f"    sloped after stepped mip {order[0]} cold ({_ms(stepped)}, {s_walks} walks over {s_files} files, "
            f"{s_reads} reads): construction {_ms(sloped)} | .sld walks {walks} over {files} files, {reads} reads"
        )
    return "\n".join(lines)


def _mb(nbytes: int) -> str:
    return f"{nbytes / 1e6:.1f}MB"


def _ladder_caches() -> str:
    native_b, scaled_b = _cache_bytes()
    parts = [
        f"native {len(unit_sprites._native_cache)} / {_mb(native_b)}",
        f"scaled {len(unit_sprites._scaled_cache)} / {_mb(scaled_b)}",
    ]
    held = getattr(unit_sprites, "_sld_bytes_cache", None)
    if held is not None:
        parts.append(f"sld bytes {len(held)} / {_mb(held._bytes)}")
    return " | ".join(parts)


def bench_ladder(path: Path) -> str:
    """The zoom-in oracle: every Stepped level coarsest to finest on ONE cache
    (pass `first`: the load's level, then each zoom step's first visit), then
    the same order on a fresh cache with unit_sprites' caches kept (pass
    `rewarm`). One `ladder` line per build; only _level() is timed."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = render.elevations_and_proj(scenario)

    def stepped_cache() -> IsoChunkCache:
        return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)

    levels = stepped_cache().mip_levels()
    lines = [f"  {path.name}", f"    stepped ladder (mip levels {levels}):"]
    unit_sprites.clear_caches()
    with _WalkCounter() as counter:
        for label in ("first", "rewarm"):
            cache = stepped_cache()
            for mip in levels:
                elapsed = _time(lambda cache=cache, mip=mip: cache._level(mip))
                walks, files, reads = counter.take()
                lines.append(
                    f"      ladder {path.stem} {label:<6} mip {mip:>2} tile_px {cache.mip_tile_px(mip):>3}: "
                    f"{_ms(elapsed):>10} | .sld walks {walks} over {files} files, {reads} reads | "
                    f"{_ladder_caches()}"
                )
    return "\n".join(lines)


def _index_line() -> list[str]:
    # Feature-detected so this mode also runs on code without the SLD index.
    index = getattr(unit_sprites, "_sld_index_cache", None)
    if index is None:
        return ["      sld index: not present in this build"]
    lines = [f"      sld index: {len(index)} files resident (cap {index.capacity})"]
    held = getattr(unit_sprites, "_sld_bytes_cache", None)
    if held is not None:
        lines.append(
            f"      sld bytes: {held._bytes / 1e6:.1f}MB in {len(held)} files (budget {held.capacity_bytes / 1e6:.0f}MB)"
        )
    return lines


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
    parser.add_argument(
        "--cold-levels", action="store_true",
        help="Only the cold-first-paint row: Stepped level A cold, two neighbours, A again, then Sloped",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Level A for --cold-levels (default: the coarsest mip level)",
    )
    parser.add_argument(
        "--ladder", action="store_true",
        help="Only the zoom-in row: every Stepped level coarsest to finest on one cache, then again on a fresh one",
    )
    args = parser.parse_args()
    if args.ladder and args.cold_levels:
        parser.error("--ladder and --cold-levels are separate modes")

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
        f"caches: native budget {unit_sprites.NATIVE_CACHE_BYTES / 1e6:.0f}MB, "
        f"scaled budget {unit_sprites.SCALED_CACHE_BYTES / 1e6:.0f}MB, "
        f"icon budget {unit_sprites.ICON_CACHE_BYTES / 1e6:.0f}MB"
    )
    worsts = []
    for path in files:
        if args.ladder:
            print(bench_ladder(path))
        elif args.cold_levels:
            print(bench_cold_levels(path, args.level))
        else:
            print(bench_file(path))
        worsts.append(path)
    print(f"\nMeasured {len(worsts)} file(s). See this module's docstring for what each line means.")


if __name__ == "__main__":
    main()
