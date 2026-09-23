#!/usr/bin/env python3
"""Chunk-preload cost -- the measurement gate the chunk-preload design
decided synchronous-vs-background-vs-stop on (it chose neither: the per-level
cost is additive across mips, so the warm ended up sliced across idle event-
loop ticks instead -- see descape/level_warm.py).

Informational only, matching this project's bench_unit_sprites.py/
bench_iso_backend.py convention: always runs, never pass/fail. Needs a real
AoE2:DE install (sprite pixels are read from it at runtime, nothing is
bundled); with no install every unit falls back to a coloured mark and every
number below understates the real cold cost.

Stepped and Flat get the full level-build/chunk-grid breakdown below. Sloped
gets a narrower row (added for the 2026-09-07 margin-warm plan's Step A3.6):
it has exactly one mip level (SlopedChunkCache._init_mip_levels({0: tile_px}),
see its own docstring), so there is no not-yet-visited LEVEL to warm and the
level-build columns don't apply -- but a chunk warm (descape.margin_warm) is
a different question: Sloped's single-level ladder means mip_for_scale's
clamp binds, so a zoomed-in viewport still covers only a fraction of the
mip-0 canvas, and the margin ring around it is non-empty. Sloped's own
_refresh_source_caches() already builds everything (units_by_tile,
corner_rise, building_bboxes, sprites) eagerly at construction -- see that
method's own docstring -- so a freshly-constructed cache is already in the
"after level-warm" state the other two styles reach via an explicit warm
call; the Sloped row measures only the per-chunk compositing cost on top of
that.

Three things measured per real corpus file, per style, per mip level:

- **Level build, cold and warm.** IsoChunkCache._level(mip) / FlatChunkCache.
  _level_icons(mip) -- the per-mip sprite-resolve-and-.sld-decode rebuild the
  plan's fact 3 says is the real cost, not the chunk grid.
- **Resident bytes of a fully-warmed chunk grid at that level**, against
  today's max_bytes (one level-0 canvas of RGB) -- what multiple holding a
  second level's chunks on top would need.
- **First paint, cold vs. after a level-only warm.** A cold cache pays level
  build + chunk composite together; a cache that only ran the level build
  first should pay just the composite on its first real chunk fetch, if
  fact 3 is right that the layer build dominates.

`--per-chunk` replaces all of the above with a per-chunk distribution: one
timed get_chunk() per chunk of the mip-0 grid, per style, over `--repeats`
fresh caches (see _bench_per_chunk). That is the margin-warm tick cost the
single-rect columns above only approximate (an unaligned 512px rect can
touch up to four chunks).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, render, unit_sprites
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units

SAMPLE_PX = 512


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _stepped_cache(scenario) -> IsoChunkCache:
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    _img, elevations, proj = render.render_terrain_iso_with_proj(scenario)
    return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)


def _flat_cache(scenario) -> FlatChunkCache:
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    return FlatChunkCache(scenario, tile_px, sprites=True)


def _sloped_cache(scenario) -> SlopedChunkCache:
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=True)


def _iso_level_warm(cache: IsoChunkCache, mip: int) -> None:
    cache._level(mip)


def _flat_level_warm(cache: FlatChunkCache, mip: int) -> None:
    # Builds unit_draws internally too (see _level_icons' own docstring), so
    # this alone is Flat's counterpart to IsoChunkCache._level(mip).
    cache._level_icons(mip)


def _sample_rect(canvas_w: int, canvas_h: int, x_offset: int = 0) -> tuple[int, int, int, int]:
    """One chunk-sized rect near the map centre, where units are usually
    densest -- a corner is often empty water/edge and would flatter the
    sprite path rather than exercise it (same reasoning as bench_unit_
    sprites.py's own _sample_rects). `x_offset` shifts the rect sideways to
    sample a second, differently-located chunk (Step B1's reconciliation
    check) without wandering into a corner."""
    x0 = max(0, min(canvas_w - SAMPLE_PX, canvas_w // 2 - SAMPLE_PX // 2 + x_offset))
    y0 = max(0, canvas_h // 2 - SAMPLE_PX // 2)
    return x0, y0, min(x0 + SAMPLE_PX, canvas_w), min(y0 + SAMPLE_PX, canvas_h)


def _bench_style(name: str, make_cache, level_warm, scenario) -> list[str]:
    lines = [f"    {name}:"]
    probe = make_cache(scenario)
    canvas_w0, canvas_h0 = probe.canvas_dims(0)
    max_bytes = canvas_w0 * canvas_h0 * 3
    levels = probe.mip_levels()
    del probe

    for mip in levels:
        # Cold level build: a fresh cache, nothing warmed yet.
        unit_sprites.clear_caches()
        cache = make_cache(scenario)
        cold = _time(lambda: level_warm(cache, mip))
        warm = min(_time(lambda: level_warm(cache, mip)) for _ in range(3))

        # Resident bytes AND wall time of a fully-warmed chunk grid at this
        # level, timed on a level-warm cache (layer already built, so this
        # isolates chunk-grid compositing from the layer-build cost above) --
        # this is the number a full-level eager preload would actually cost.
        canvas_w, canvas_h = cache.canvas_dims(mip)
        before = cache._cache_bytes
        cache.render_rect(0, 0, canvas_w, canvas_h, mip=mip)
        chunk_bytes = cache._cache_bytes - before
        multiple = chunk_bytes / max_bytes if max_bytes else float("inf")

        unit_sprites.clear_caches()
        grid_cache = make_cache(scenario)
        level_warm(grid_cache, mip)
        gw, gh = grid_cache.canvas_dims(mip)
        full_grid_warm = _time(lambda: grid_cache.render_rect(0, 0, gw, gh, mip=mip))

        # First paint, cold (fresh cache, nothing warmed) vs. after a
        # level-only warm (fresh cache, level build done, no chunk fetched).
        rect = _sample_rect(canvas_w, canvas_h)
        rect2 = _sample_rect(canvas_w, canvas_h, x_offset=SAMPLE_PX * 3)
        unit_sprites.clear_caches()
        cold_cache = make_cache(scenario)
        first_paint_cold = _time(lambda: cold_cache.render_rect(*rect, mip=mip))

        unit_sprites.clear_caches()
        warm_layer_cache = make_cache(scenario)
        level_warm(warm_layer_cache, mip)
        first_paint_after_warm = _time(lambda: warm_layer_cache.render_rect(*rect, mip=mip))
        # Second, differently-located chunk fetched on the SAME level-warm
        # cache right after the first (Step B1). If this is cheap, whatever
        # made the first chunk expensive was a one-time setup step that ran
        # during that first fetch, not a per-chunk cost; if it's ALSO
        # expensive, the cost is genuinely per-chunk.
        first_paint_after_warm2 = _time(lambda: warm_layer_cache.render_rect(*rect2, mip=mip))

        lines.append(
            f"      mip {mip:>2}: level build cold {_ms(cold)} | warm {_ms(warm)} | "
            f"first paint cold {_ms(first_paint_cold)} | after level-warm {_ms(first_paint_after_warm)} | "
            f"2nd chunk same cache {_ms(first_paint_after_warm2)} | "
            f"warm chunk grid {chunk_bytes / 1e6:.1f}MB ({multiple:.2f}x today's max_bytes) | "
            f"full grid warm (post level-warm) {_ms(full_grid_warm)}"
        )
    return lines


def _bench_sloped(scenario) -> list[str]:
    """Sloped's own row (Step A3.6) -- narrower than _bench_style()'s: one
    mip level, no separate level-build step (_refresh_source_caches()
    already ran everything eagerly at construction, see _sloped_cache's own
    docstring), so only the two chunk-compositing columns this module's
    docstring explains apply."""
    lines = ["    sloped:"]
    probe = _sloped_cache(scenario)
    canvas_w, canvas_h = probe.canvas_dims(0)
    del probe

    rect = _sample_rect(canvas_w, canvas_h)
    rect2 = _sample_rect(canvas_w, canvas_h, x_offset=SAMPLE_PX * 3)
    unit_sprites.clear_caches()
    cache = _sloped_cache(scenario)
    first_chunk = _time(lambda: cache.render_rect(*rect, mip=0))
    second_chunk = _time(lambda: cache.render_rect(*rect2, mip=0))

    lines.append(f"      mip  0: after level-warm {_ms(first_chunk)} | 2nd chunk same cache {_ms(second_chunk)}")
    return lines


def _sloped_level_warm(cache: SlopedChunkCache, mip: int) -> None:
    # Construction already built everything (see _bench_sloped's docstring).
    pass


_PER_CHUNK_STYLES = {
    "stepped": (_stepped_cache, _iso_level_warm),
    "flat": (_flat_cache, _flat_level_warm),
    "sloped": (_sloped_cache, _sloped_level_warm),
}


def _pct(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _dist(label: str, samples: list[float]) -> str:
    if not samples:
        return f"{label}: n=0"
    return (
        f"{label}: n={len(samples)} median {_ms(_pct(samples, 0.5))} | "
        f"p90 {_ms(_pct(samples, 0.9))} | max {_ms(max(samples))}"
    )


def _bench_per_chunk(scenario, style: str, repeats: int) -> list[str]:
    """--per-chunk mode: one timed get_chunk(0, cx, cy) per chunk of the
    whole mip-0 grid, on a level-warm cache -- exactly MarginWarmer.tick()'s
    unit of work. Mip 0 because it is Sloped's only level, so it is the one
    level all three styles can be compared on.

    Run 1 starts from cleared sprite caches (cold); later runs build a fresh
    chunk cache but keep the process-wide sprite caches (warm), which is the
    in-app margin-warm case once the viewport has painted. "centre" is the
    middle half of each axis, which sits inside the iso diamond, so it
    excludes the empty canvas corners that dilute the "all" figures."""
    make_cache, level_warm = _PER_CHUNK_STYLES[style]
    cold: dict[str, list[float]] = {"all": [], "centre": []}
    warm: dict[str, list[float]] = {"all": [], "centre": []}
    lines = [f"    {style} (mip 0, {repeats} run(s)):"]
    for run in range(repeats):
        if run == 0:
            unit_sprites.clear_caches()
        start = time.perf_counter()
        cache = make_cache(scenario)
        level_warm(cache, 0)
        setup = time.perf_counter() - start
        canvas_w, canvas_h = cache.canvas_dims(0)
        nx, ny = -(-canvas_w // cache.chunk_px), -(-canvas_h // cache.chunk_px)
        into = cold if run == 0 else warm
        for cy in range(ny):
            for cx in range(nx):
                t = _time(lambda c=cache, cx=cx, cy=cy: c.get_chunk(0, cx, cy))
                into["all"].append(t)
                if nx // 4 <= cx < nx - nx // 4 and ny // 4 <= cy < ny - ny // 4:
                    into["centre"].append(t)
        lines.append(
            f"      run {run + 1}: construct + level warm {_ms(setup)} (load-time, not per-tick) | "
            f"grid {nx}x{ny}"
        )
        del cache
    lines.extend(
        "      " + _dist(f"{label} {region}", bucket[region])
        for label, bucket in (("cold", cold), ("warm", warm))
        for region in ("centre", "all")
        if bucket[region]
    )
    return lines


def bench_file_per_chunk(path: Path, styles: list[str], repeats: int) -> str:
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    lines = [f"  {path.name} ({mm.map_width}x{mm.map_height})"]
    for style in styles:
        lines.extend(_bench_per_chunk(scenario, style, repeats))
    return "\n".join(lines)


def bench_file(path: Path) -> str:
    scenario = load_map_and_units(path)
    lines = [f"  {path.name}"]
    lines.extend(_bench_style("stepped", _stepped_cache, _iso_level_warm, scenario))
    lines.extend(_bench_style("flat", _flat_cache, _flat_level_warm, scenario))
    lines.extend(_bench_sloped(scenario))
    return "\n".join(lines)


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
        "--per-chunk", action="store_true",
        help="Time get_chunk() over the whole mip-0 grid per style instead (margin-warm tick cost)",
    )
    parser.add_argument(
        "--styles", default="stepped,flat,sloped",
        help="Comma-separated styles for --per-chunk (default: all three)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Fresh caches per style for --per-chunk")
    args = parser.parse_args()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    unknown = [s for s in styles if s not in _PER_CHUNK_STYLES]
    if unknown:
        parser.error(f"unknown style(s): {', '.join(unknown)}")

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

    print(f"Chunk-preload bench -- install {asset_source.get_install_path()}")
    for path in files:
        print(bench_file_per_chunk(path, styles, args.repeats) if args.per_chunk else bench_file(path))
    print(f"\nMeasured {len(files)} file(s). See this module's docstring for what each line means.")


if __name__ == "__main__":
    main()
