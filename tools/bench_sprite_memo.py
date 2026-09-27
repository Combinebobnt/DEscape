#!/usr/bin/env python3
"""Cost of a Stepped/Sloped sprite-layer rebuild, split into what a per-unit
sprite-contribution memo can and can't save. Informational only, never
pass/fail, matching bench_unit_sprites.py's convention. Needs a real AoE2:DE
install; with none every unit falls back to a mark and the rows mean little.

Graduated from the 2026-09-25 scratch probe. Per file and style (Stepped at
mip 0 and one other level, Sloped), against the real caches' own levels, with
the art caches warmed by one untimed walk first:

  full          sprite_draws_by_anchor(), today's whole walk
  overrides     render._wall_variant_rotation_overrides_uncached(), paid by
                every walk (the memoized wrapper hides it after the first)
  resolve       _resolve_unit_sprite() for every unit, nothing else
  merge         re-merging already-resolved contributions into a SpriteLayer
  key           building a memo key per unit (the bench's own replica of the
                memo's key, so the row exists on a checkout without one)
  memo hit      the real memo walk with every unit a hit (lane only)
  memo 1 miss   the same after moving one unit (lane only)
  units_by_tile / building_bboxes / bystander_grid / merge_bboxes
                the other passes a wholesale rebuild pays
  memo MB       tracemalloc'd size of one level's memo (the real one when the
                checkout has it, else the bench's replica)

Each timing is the median of --repeat runs.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from testkit import settings_isolation

FILES = ["old-allies-final-v2.aoe2scenario", "2_Joan_coop_1_v0_13.aoe2scenario"]


def _median_ms(fn, repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def _units(scenario):
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            yield player_id, i, unit


def _replica_key(scenario, proj, heights, sloped: bool, overrides, player_id, i, unit):
    from descape import render

    ux, uy = int(unit.x), int(unit.y)
    try:
        if sloped:
            rise = (heights[uy][ux], heights[uy][ux + 1], heights[uy + 1][ux], heights[uy + 1][ux + 1])
        else:
            rise = heights[uy][ux]
    except IndexError:
        rise = None
    return (
        unit.x, unit.y, unit.unit_const,
        overrides.get((player_id, i), render.stored_rotation(player_id, unit)),
        getattr(unit, "reference_id", None), player_id,
        scenario.team_indices[player_id], scenario.player_colors[player_id], rise,
    )


def _merge(contributions):
    """sprite_draws_by_anchor_sliced()'s own accumulation, over a prebuilt list."""
    from descape import render

    by_anchor: dict = {}
    bboxes: dict = {}
    skip_ids: set = set()
    farm_by_tile: dict = {}
    for contribution in contributions:
        if contribution is None:
            continue
        if contribution.farm_tiles:
            farm_by_tile.update(contribution.farm_tiles)
        for key, piece_draws in contribution.by_anchor.items():
            by_anchor.setdefault(key, []).extend(piece_draws)
            own = contribution.bboxes[key]
            bbox = bboxes.get(key)
            bboxes[key] = own if bbox is None else (
                min(bbox[0], own[0]), min(bbox[1], own[1]), max(bbox[2], own[2]), max(bbox[3], own[3]),
            )
        skip_ids.add(contribution.skip_id)
    return render.SpriteLayer(by_anchor=by_anchor, bboxes=bboxes, skip_ids=frozenset(skip_ids), farm_by_tile=farm_by_tile)


def _bench_level(scenario, cache, proj, elevations, corner_rise, repeat: int) -> list[str]:
    from descape import render

    mm = scenario.map_manager
    sloped = corner_rise is not None
    layers = cache.layers
    kwargs = {
        "corner_rise": corner_rise, "with_farms": layers.farm_overlay,
        "tree_scale": layers.tree_scale, "hero_glow": layers.hero_glow,
    }
    uf = cache.unit_filter

    def full():
        return render.sprite_draws_by_anchor(scenario, proj, elevations, uf, **kwargs)

    full()  # warm the art caches
    overrides = render.wall_variant_rotation_overrides(scenario)
    units = list(_units(scenario))

    def resolve_all():
        return [
            render._resolve_unit_sprite(
                scenario, proj, elevations, uf, corner_rise, overrides, layers.farm_overlay,
                pid, i, u, layers.tree_scale, layers.hero_glow,
            )
            for pid, i, u in units
        ]

    contributions = resolve_all()
    heights_src = corner_rise if sloped else elevations

    def keys():
        heights = heights_src.tolist()
        return [_replica_key(scenario, proj, heights, sloped, overrides, pid, i, u) for pid, i, u in units]

    rows = {
        "full": _median_ms(full, repeat),
        "overrides": _median_ms(lambda: render._wall_variant_rotation_overrides_uncached(scenario), repeat),
        "resolve": _median_ms(resolve_all, repeat),
        "merge": _median_ms(lambda: _merge(contributions), repeat),
        "key": _median_ms(keys, repeat),
    }

    memo_cls = getattr(render, "SpriteMemo", None)
    if memo_cls is not None:
        _layer, memo = render._drain(render.sprite_draws_by_anchor_sliced(
            scenario, proj, elevations, uf, memo=memo_cls(), **kwargs
        ))

        def memo_walk():
            return render._drain(render.sprite_draws_by_anchor_sliced(
                scenario, proj, elevations, uf, memo=memo, **kwargs
            ))

        rows["memo hit"] = _median_ms(memo_walk, repeat)
        victim = next(u for _pid, _i, u in units if 1 <= u.x < mm.map_width - 2)
        victim.x += 1
        try:
            rows["memo 1 miss"] = _median_ms(memo_walk, repeat)
        finally:
            victim.x -= 1

    headroom = render._unit_rise_headroom_px(corner_rise, elevations, proj) if sloped else 0
    bboxes = render._building_bboxes_iso(scenario, mm.map_width, mm.map_height, proj, elevations, headroom, unit_filter=uf)
    layer = full()
    merged = render.merge_sprite_bboxes(bboxes, layer)
    rows["units_by_tile"] = _median_ms(lambda: render._units_by_tile(scenario, uf), repeat)
    rows["building_bboxes"] = _median_ms(
        lambda: render._building_bboxes_iso(scenario, mm.map_width, mm.map_height, proj, elevations, headroom, unit_filter=uf),
        repeat,
    )
    rows["merge_bboxes"] = _median_ms(lambda: render.merge_sprite_bboxes(bboxes, layer), repeat)
    rows["bystander_grid"] = _median_ms(lambda: render.build_bystander_grid(merged, cache.chunk_px), repeat)

    gc.collect()
    tracemalloc.start()
    if memo_cls is not None:
        before = tracemalloc.get_traced_memory()[0]
        held = render._drain(render.sprite_draws_by_anchor_sliced(
            scenario, proj, elevations, uf, memo=memo_cls(), **kwargs
        ))
        # The layer is built alongside the memo; measure the memo alone.
        layer_bytes_probe = held[0]
        del layer_bytes_probe
        held = held[1]
        gc.collect()
        memo_bytes = tracemalloc.get_traced_memory()[0] - before
        memo_label = "memo MB"
    else:
        before = tracemalloc.get_traced_memory()[0]
        heights = heights_src.tolist()
        held = {
            id(u): (u, _replica_key(scenario, proj, heights, sloped, overrides, pid, i, u), c)
            for (pid, i, u), c in zip(units, resolve_all(), strict=True)
        }
        gc.collect()
        memo_bytes = tracemalloc.get_traced_memory()[0] - before
        memo_label = "memo MB (replica)"
    tracemalloc.stop()
    del held

    line = " ".join(f"{k} {v:.1f}" for k, v in rows.items())
    return [f"      {line} | {memo_label} {memo_bytes / 1e6:.1f}"]


def bench_file(path: Path, styles, repeat: int) -> None:
    from descape import render
    from descape.render_cache import IsoChunkCache, SlopedChunkCache
    from descape.scenario_io import load_map_and_units

    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    print(f"  {path.name} units {sum(len(u) for u in scenario.unit_manager.units)}", flush=True)
    if "stepped" in styles:
        elevations, proj = render.elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
        levels = cache.mip_levels()
        other = next((m for m in levels if m != 0 and m > 0), next(m for m in levels if m != 0))
        for mip in (0, other):
            lvl = cache._levels[mip]
            print(f"    stepped mip {mip} (tile_px {lvl.tile_px})", flush=True)
            for line in _bench_level(scenario, cache, lvl.proj, elevations, None, repeat):
                print(line, flush=True)
    if "sloped" in styles:
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=True)
        print("    sloped", flush=True)
        for line in _bench_level(scenario, cache, cache.proj, cache.elevations, cache.corner_rise, repeat):
            print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, nargs="?", default=ROOT / "examples")
    parser.add_argument("--files", help="Comma-separated file names, overriding the built-in list")
    parser.add_argument("--styles", default="stepped,sloped")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    names = [n.strip() for n in args.files.split(",")] if args.files else FILES
    styles = {s.strip() for s in args.styles.split(",") if s.strip()}
    settings_isolation.pin_install_path()
    with tempfile.TemporaryDirectory(prefix="bench_sprite_memo_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        for name in names:
            path = args.scenario_dir / name
            if not path.exists():
                print(f"  {name} skipped (not found in {args.scenario_dir})")
                continue
            bench_file(path, styles, args.repeat)


if __name__ == "__main__":
    main()
