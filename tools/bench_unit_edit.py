#!/usr/bin/env python3
"""Times ViewerWindow._after_unit_mutation()'s current whole-canvas sequence
for one `set_position` unit move, per style -- a unit-editing performance
batch's decision gate: whether a per-edit splice of the chunk caches' unit
sources is worth building, or whether the recomposite cost dominates enough
that a bbox-scoped repaint alone already captures the win.

**No ViewerWindow**: drives UnitEditModel and each chunk cache directly,
reproducing _after_unit_mutation()'s sequence (cache.invalidate_units() ->
cache.invalidate_region(whole canvas) -> map_view repaint -> unit_pick.
build_index()) rather than importing viewer.py, whose ViewerWindow needs a
real QApplication and settings isolation (see tools/bench_fill_latency.py
for the testkit pattern that keeps it off the user's real config.yaml).

Phases, timed separately so D4's splice can be judged against each one
rather than the total:

  - _capture x2 (UnitEditModel.begin_unit_edit()/commit_unit_edit()'s own
    before/after snapshot, unit_model.py:788)
  - invalidate_units() -- the wholesale per-style source rebuild
  - canvas eviction (invalidate_region over the whole canvas) and the
    recomposite of just the chunks a real viewport would actually
    re-request (render_rect over a viewport-sized rect, not the whole
    canvas -- the viewport is what MapView actually asks for after a
    repaint, per _after_unit_mutation's own map_view.invalidate_region()
    call feeding a real paint event, not a full-canvas render_rect)
  - _rebuild_unit_index() (unit_pick.build_index(), viewer.py:1805)
  - build_bystander_grid() (Sloped only -- IsoChunkCache/FlatChunkCache
    have no bystander grid)

Two further sections, both with real sprites on (the configured AoE2:DE
install) and mips 0 and 1 resident before each edit:

  - Flat's lazy per-level rebuild. FlatChunkCache.invalidate_units() only
    drops unit_draws and the icon layers; the rebuild lands on the next
    composite, which the phases above never time. Reported per resident
    level (draws, icons), plus the first viewport recomposite after the
    invalidate. The draws+icons sum is the Flat row splice's decision gate.
  - A Convert-shaped batch: reassign CONVERT_K units inside one
    begin/commit_unit_edit, per style, then invalidate_units() either
    wholesale (changed=None) or with one UnitSplice per unit, plus the
    lazy source rebuild each style owes on its resident levels.

Informational only, matching tools/bench_pick_plane_patch.py's convention:
always runs, never pass/fail. Read the ratio each phase takes of the total,
not the absolute ms -- this machine drifts 15-20% across a run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import render, unit_pick, unit_sprites
from descape.edit_history import EditHistory
from descape.render import elevations_and_proj, sloped_elevations_and_proj, tile_pixels_for_map
from descape.render_cache import DEFAULT_CHUNK_PX, FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice
from descape.scenario_io import load_map_and_units
from descape.unit_model import UnitEditModel

CORPUS_FILE = "F7_2_Dos Pilas (648).aoe2scenario"
REPEATS = 5
# The sprite sections pay an untimed wholesale restore per repeat, so fewer.
SPRITE_REPEATS = 3
CONVERT_K = 20
RESIDENT_MIPS = (0, 1)
# A 1920x1080 viewport is what MapView actually re-requests after a repaint
# (_after_unit_mutation's map_view.invalidate_region() feeds a real paint
# event next, never a full-canvas render_rect) -- see the module docstring.
VIEWPORT_W, VIEWPORT_H = 1920, 1080


def _time_ms(fn, repeats: int = REPEATS) -> float:
    """Mean wall-clock milliseconds per call, one untimed warm-up first."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000 / repeats


def _viewport_rect(canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
    w, h = min(VIEWPORT_W, canvas_w), min(VIEWPORT_H, canvas_h)
    x0, y0 = (canvas_w - w) // 2, (canvas_h - h) // 2
    return x0, y0, x0 + w, y0 + h


def _pick_movable_unit(scenario):
    """The first unit any player owns -- GAIA (player 0) can carry ids that
    the model refuses to relocate for no such reason, but any concrete unit
    works for a bare set_position timing, so just take the first one found
    scanning players 1..8 first (real buildings/units), falling back to
    GAIA."""
    manager = scenario.unit_manager
    for player in [*range(1, len(manager.units)), 0]:
        units = manager.units[player]
        if units:
            return player, units[0]
    raise RuntimeError("corpus file has no units at all")


def _run_phases(cache, scenario, model: UnitEditModel, player: int, unit, viewport) -> list[str]:
    lines: list[str] = []
    history = EditHistory()
    orig_x, orig_y, orig_z = unit.x, unit.y, unit.z

    def _capture_pair():
        model.begin_unit_edit([player])
        model.set_position(unit, orig_x + 1, orig_y, orig_z)
        model.commit_unit_edit("bench move", history, push=False)
        model.begin_unit_edit([player])
        model.set_position(unit, orig_x, orig_y, orig_z)
        model.commit_unit_edit("bench move back", history, push=False)

    lines.append(f"    {'_capture x2':>28}  {_time_ms(_capture_pair):7.3f}ms")

    # Warm the viewport before timing eviction+recomposite, mirroring a real
    # session that has already painted once.
    cache.render_rect(*viewport)

    canvas_w, canvas_h = cache.canvas_dims(0)

    def _move_and_settle():
        unit.x, unit.y = orig_x + 1, orig_y
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)
        unit_pick.build_index(scenario, cache.unit_filter)
        unit.x, unit.y = orig_x, orig_y
        # Undo left the cache mid-edit-state for the next repeat's warm-up.
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    def _invalidate_units_only():
        unit.x, unit.y = orig_x + 1, orig_y
        cache.invalidate_units()
        unit.x, unit.y = orig_x, orig_y
        cache.invalidate_units()

    lines.append(f"    {'invalidate_units()':>28}  {_time_ms(_invalidate_units_only):7.3f}ms")

    def _evict_and_recomposite():
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    lines.append(f"    {'evict + recomposite visible':>28}  {_time_ms(_evict_and_recomposite):7.3f}ms")

    def _rebuild_index():
        unit_pick.build_index(scenario, cache.unit_filter)

    lines.append(f"    {'_rebuild_unit_index()':>28}  {_time_ms(_rebuild_index):7.3f}ms")

    if cache.style == "sloped":
        def _bystander():
            render.build_bystander_grid(cache.building_bboxes, cache.chunk_px)

        lines.append(f"    {'build_bystander_grid()':>28}  {_time_ms(_bystander):7.3f}ms")

    lines.append(f"    {'--- whole sequence, once ---':>28}")
    lines.append(f"    {'set_position + full sequence':>28}  {_time_ms(_move_and_settle):7.3f}ms")
    return lines


def _make_cache(style: str, scenario, chunk_px: int, sprites: bool = False):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        return IsoChunkCache(scenario, elevations, proj, tile_px, chunk_px=chunk_px, sprites=sprites)
    if style == "sloped":
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px, sprites=sprites)
    return FlatChunkCache(scenario, tile_px, chunk_px=chunk_px, sprites=sprites)


def _resident_mips(cache) -> tuple[int, ...]:
    return tuple(m for m in RESIDENT_MIPS if m in cache.mip_levels())


def _warm_resident(cache) -> None:
    """Paints a viewport at every resident mip, so each level's sources exist."""
    for mip in _resident_mips(cache):
        cache.render_rect(*_viewport_rect(*cache.canvas_dims(mip)), mip=mip)


def _settle(cache) -> None:
    """Forces the lazy source rebuild each style owes after invalidate_units(),
    without compositing. Sloped rebuilds eagerly inside the call, so nothing."""
    if cache.style == "stepped":
        for mip in _resident_mips(cache):
            cache._level(mip)
    elif cache.style == "flat":
        for mip in _resident_mips(cache):
            cache._level_unit_draws(mip)
            cache._level_icons(mip)


def _footprint(scenario, unit) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    mm = scenario.map_manager
    tiles = render.unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ()
    return (int(unit.x), int(unit.y)), tuple(tiles)


def _flat_rebuild_section(path: Path, chunk_px: int) -> list[str]:
    """Flat, sprites on: what a wholesale invalidate_units() costs once the
    next paint pays for it, per resident level."""
    scenario = load_map_and_units(path)
    cache = _make_cache("flat", scenario, chunk_px, sprites=True)
    _warm_resident(cache)
    _player, unit = _pick_movable_unit(scenario)
    orig_x, orig_y = unit.x, unit.y
    canvas_w, canvas_h = cache.canvas_dims(0)
    viewport = _viewport_rect(canvas_w, canvas_h)
    mips = _resident_mips(cache)
    totals: dict[str, float] = {}

    def _add(key: str, t0: float) -> float:
        now = time.perf_counter()
        totals[key] = totals.get(key, 0.0) + (now - t0) * 1000
        return now

    for repeat in range(SPRITE_REPEATS + 1):
        # Repeat 0 is the untimed warm-up; the edit alternates so each is real.
        unit.x = orig_x + 1 if repeat % 2 == 0 else orig_x
        t = time.perf_counter()
        cache.invalidate_units()
        t = _add("invalidate_units()", t)
        for mip in mips:
            cache._level_unit_draws(mip)
            t = _add(f"mip {mip} draws", t)
            cache._level_icons(mip)
            t = _add(f"mip {mip} icons", t)
        if repeat == 0:
            totals.clear()
    unit.x, unit.y = orig_x, orig_y

    def _first_recomposite():
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    rows = cache.unit_draws[0].shape[0] if cache.unit_draws is not None else 0
    icons = len(cache._level_icons(0) or {})
    lines = [f"  [flat, sprites on] {rows} rows, {icons} icons at mip 0, resident mips {list(mips)}"]
    for key, total in totals.items():
        lines.append(f"    {key:>28}  {total / SPRITE_REPEATS:7.3f}ms")
    # invalidate_units() rebuilds mip 0's draws eagerly, so it counts too.
    gate = sum(totals.values()) / SPRITE_REPEATS
    lines.append(f"    {'GATE draws+icons, resident':>28}  {gate:7.3f}ms  (Step 2 builds if >= ~33ms)")
    lines.append(
        f"    {'first viewport recomposite':>28}  {_time_ms(_first_recomposite, SPRITE_REPEATS):7.3f}ms"
        "  (invalidate_units + evict + mip 0 viewport)"
    )
    return lines


def _convert_batch(scenario, k: int) -> tuple[int, int, list]:
    """k units of the most populous non-GAIA player, each alone on its tiles
    and not a wall/connector, so the splice path is actually exercised, plus
    a destination player that isn't the source."""
    manager = scenario.unit_manager
    mm = scenario.map_manager
    source = max(range(1, len(manager.units)), key=lambda p: len(manager.units[p]))
    destination = next(p for p in range(1, len(manager.units)) if p != source)
    occupancy: dict[tuple[int, int], int] = {}
    for units in manager.units:
        for u in units:
            for tile in render.unit_occupied_tiles(u, mm.map_width, mm.map_height) or ():
                occupancy[tile] = occupancy.get(tile, 0) + 1
    excluded = unit_sprites.wall_connector_consts()
    batch = []
    for u in manager.units[source]:
        if u.unit_const in excluded or unit_sprites.rotation_variant_eligible(u.unit_const):
            continue
        tiles = render.unit_occupied_tiles(u, mm.map_width, mm.map_height)
        if not tiles or any(occupancy[t] != 1 for t in tiles):
            continue
        batch.append(u)
        if len(batch) == k:
            break
    return source, destination, batch


def _convert_section(path: Path, style: str, chunk_px: int) -> list[str]:
    scenario = load_map_and_units(path)
    cache = _make_cache(style, scenario, chunk_px, sprites=True)
    _warm_resident(cache)
    model = UnitEditModel(scenario)
    history = EditHistory()
    source, destination, batch = _convert_batch(scenario, CONVERT_K)
    dest_list = scenario.unit_manager.units[destination]
    totals: dict[str, float] = {}

    def _run(use_splice: bool) -> None:
        t0 = time.perf_counter()
        model.begin_unit_edit([source, destination])
        splices = []
        for u in batch:
            own, tiles = _footprint(scenario, u)
            model.reassign(u, destination)
            splices.append(
                UnitSplice(destination, len(dest_list) - 1, u, own, own, tiles, tiles, old_player_id=source)
            )
        model.commit_unit_edit("bench convert", history)
        t1 = time.perf_counter()
        cache.invalidate_units(splices if use_splice else None)
        t2 = time.perf_counter()
        _settle(cache)
        t3 = time.perf_counter()
        mode = "splice" if use_splice else "wholesale"
        for key, value in (
            ("model: begin + reassign + commit", t1 - t0),
            (f"{mode}: invalidate_units()", t2 - t1),
            (f"{mode}: lazy source rebuild", t3 - t2),
            (f"{mode}: TOTAL cache", t3 - t1),
        ):
            totals[key] = totals.get(key, 0.0) + value * 1000
        # Untimed restore: exact list order back, sources rebuilt from it.
        history.undo(scenario.map_manager.terrain, units=model)
        cache.invalidate_units()
        _settle(cache)

    _run(False)  # warm-up
    totals.clear()
    for use_splice in (False, True):
        for _ in range(SPRITE_REPEATS):
            _run(use_splice)
    lines = [(
        f"  [{style}, sprites on] Convert batch: {len(batch)} units, player {source} -> {destination}, "
        f"resident mips {list(_resident_mips(cache))}"
    )]
    model_key = "model: begin + reassign + commit"
    for key, total in totals.items():
        count = 2 * SPRITE_REPEATS if key == model_key else SPRITE_REPEATS
        lines.append(f"    {key:>36}  {total / count:8.3f}ms")
    return lines


def bench(path: Path, chunk_px: int = DEFAULT_CHUNK_PX) -> str:
    lines = [f"  {path.name}"]

    for style in ("stepped", "sloped", "flat"):
        scenario = load_map_and_units(path)
        model = UnitEditModel(scenario)
        player, unit = _pick_movable_unit(scenario)
        cache = _make_cache(style, scenario, chunk_px)

        canvas_w, canvas_h = cache.canvas_dims(0)
        viewport = _viewport_rect(canvas_w, canvas_h)
        lines.append(
            f"  [{style}] canvas {canvas_w}x{canvas_h}, unit player={player} "
            f"const={unit.unit_const} at ({unit.x:.1f}, {unit.y:.1f})"
        )
        lines += _run_phases(cache, scenario, model, player, unit, viewport)
        lines.append("")

    lines += _flat_rebuild_section(path, chunk_px)
    lines.append("")
    for style in ("stepped", "sloped", "flat"):
        lines += _convert_section(path, style, chunk_px)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", type=Path, nargs="?", default=ROOT / "examples" / CORPUS_FILE, help="A .aoe2scenario file"
    )
    parser.add_argument("--chunk-px", type=int, default=DEFAULT_CHUNK_PX)
    args = parser.parse_args()

    if not args.scenario.exists():
        print(f"no such file: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    print(bench(args.scenario, args.chunk_px))


if __name__ == "__main__":
    main()
