"""Corpus-tier probe of the Stepped/Sloped Convert splice (2026-09-27 plan,
Step B): random Convert-drag batches through UnitEditModel.reassign() on real
files with real sprites, each spliced, then compared for exact equality with
a fresh cache. Modelled on the W4 Flat row-splice probe.

Each round reassigns every unit of one player inside a brush-5 x 8-tile window
(one real flush's worth), so walls, gates and shared tiles arrive the way a
drag delivers them. Checked per round: units_by_tile (bucket order included),
each resident level's sprite layer and building_bboxes, and the native unit
pack's rows where the native backend is live. A round may fall back to the
wholesale path only when its shared-tile component is over the cap.

Draws are compared by content, not identity: the scaled-sprite cache is
byte-capped, so a fresh cache on a big map can get an equal but new object.

Needs AOE2DE_INSTALL_PATH, since conftest hides the configured install.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from test_bystander_grid_patch import grid_state

from descape import asset_source, composite_backend, render, render_cache
from descape.edit_history import EditHistory
from descape.native_composite import ROW_SPRITE
from descape.render import elevations_and_proj, sloped_elevations_and_proj, tile_pixels_for_map
from descape.render_cache import IsoChunkCache, SlopedChunkCache, UnitSplice
from descape.scenario_io import load_map_and_units
from descape.unit_filter import UnitFilter
from descape.unit_model import UnitEditModel

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FILES = (
    "8tp3w9j.aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
    "old-allies-final-v2.aoe2scenario",
    "2_Joan_coop_1_v0_13.aoe2scenario",
)
ROUNDS = 8
STEPPED_MIPS = (0, 1)
WINDOW_W, WINDOW_H = 5, 8
VIEW_PX = 1024


def _make_cache(style: str, scenario, unit_filter: UnitFilter):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True, unit_filter=unit_filter)
    else:
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=True, unit_filter=unit_filter)
    for mip in _mips(cache):
        cache.render_rect(0, 0, VIEW_PX, VIEW_PX, mip=mip)
    return cache


def _mips(cache) -> tuple[int, ...]:
    return STEPPED_MIPS if isinstance(cache, IsoChunkCache) else (0,)


def _draw_key(keys: dict, rgba: np.ndarray, hx: int, hy: int) -> tuple:
    k = keys.get(id(rgba))
    if k is None:
        k = keys[id(rgba)] = (rgba.shape, hash(rgba.tobytes()))
    return (*k, hx, hy)


def _level_state(cache, mip: int, spliced: bool, keys: dict):
    if isinstance(cache, IsoChunkCache):
        lvl = cache._levels[mip] if spliced else cache._level(mip)
        assert not spliced or lvl.gen == cache._source_gen, f"mip {mip} went stale"
        building_bboxes, sprites, grid = lvl.building_bboxes, lvl.sprites, lvl.bystander_grid
    else:
        building_bboxes, sprites, grid = cache.building_bboxes, cache.sprites, cache.bystander_grid
    by_anchor = {
        k: [(_draw_key(keys, d.rgba, d.hotspot_x, d.hotspot_y), px, py) for d, px, py in v]
        for k, v in sprites.by_anchor.items()
    }
    return (
        dict(building_bboxes), by_anchor, dict(sprites.bboxes), sprites.skip_ids, dict(sprites.farm_by_tile),
        grid_state(grid),
    )


def _pack_state(pack, keys: dict):
    """Every tile's rows, derived in full, with sprite slots resolved to content."""
    pack.ensure(np.arange(pack.w * pack.h))
    out = {}
    for t, (start, n) in enumerate(pack.index.reshape(-1, 2).tolist()):
        if not n:
            continue
        out[t] = [
            (kind, a, b, _draw_key(keys, pack._rgbas[c][0], 0, 0) if kind == ROW_SPRITE else c)
            for kind, a, b, c in pack.rows[start : start + n].tolist()
        ]
    return out, pack.farm_tid.tobytes(), pack.farm_mask.tobytes(), pack.farm_rgb.tobytes()


def _assert_matches_fresh(cache, style: str, scenario, unit_filter: UnitFilter, tag: str) -> None:
    keys: dict = {}
    got = {m: _level_state(cache, m, True, keys) for m in _mips(cache)}
    fresh = _make_cache(style, scenario, unit_filter)
    assert cache.units_by_tile.keys() == fresh.units_by_tile.keys(), f"{tag}: units_by_tile keys"
    for tile, bucket in cache.units_by_tile.items():
        want = [(id(u), c) for u, c in fresh.units_by_tile[tile]]
        assert [(id(u), c) for u, c in bucket] == want, f"{tag}: units_by_tile[{tile}] order"
    names = ("building_bboxes", "by_anchor", "bboxes", "skip_ids", "farm_by_tile", "bystander_grid")
    for mip in _mips(cache):
        want = _level_state(fresh, mip, False, keys)
        for name, a, b in zip(names, got[mip], want, strict=True):
            assert a == b, f"{tag}: mip {mip} {name}"
        if composite_backend.native is None:
            continue
        pack = cache._unit_pack_of(mip, create=False)
        assert pack is not None, f"{tag}: mip {mip} lost its unit pack instead of refreshing it"
        assert _pack_state(pack, keys) == _pack_state(fresh._unit_pack_of(mip, create=True), keys), (
            f"{tag}: mip {mip} unit pack"
        )


def _footprint(scenario, unit):
    mm = scenario.map_manager
    tiles = render.unit_occupied_tiles(unit, mm.map_width, mm.map_height)
    return (int(unit.x), int(unit.y)), None if tiles is None else tuple(tiles)


def _over_cap(scenario, cache, splices) -> bool:
    """Whether a refused batch was refused for the component cap alone."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 10**9)
        component, _why = render_cache._reassign_component_splices(
            scenario, cache.units_by_tile, cache.unit_filter, splices
        )
    return component is not None and len(component) > render_cache._COMPONENT_SPLICE_MAX_UNITS


@pytest.mark.corpus
@pytest.mark.parametrize("hide", [False, True], ids=["all-shown", "one-hidden"])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("name", FILES)
def test_random_convert_batches_splice_exactly_like_a_fresh_cache(name, style, hide, monkeypatch):
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not in examples/")
    scenario = load_map_and_units(path)
    units = scenario.unit_manager.units
    rng = random.Random(f"{name}-{style}-{hide}")
    players = [p for p in range(1, len(units)) if units[p]]
    hidden = rng.choice(players) if hide else None
    unit_filter = UnitFilter(players=frozenset(p for p in range(len(units)) if p != hidden)) if hide else UnitFilter()
    cache = _make_cache(style, scenario, unit_filter)
    model = UnitEditModel(scenario)
    history = EditHistory()
    wholesale = []
    real_units_by_tile = render._units_by_tile
    monkeypatch.setattr(render, "_units_by_tile", lambda *a, **k: (wholesale.append(1), real_units_by_tile(*a, **k))[1])
    stats = {"spliced": 0, "component": 0, "over_cap": 0}
    shared_seeded = 0

    for r in range(ROUNDS):
        source = rng.choice([p for p in range(len(units)) if units[p]])
        destination = rng.choice([p for p in range(len(units)) if p != source])
        # Every other round seeds its window on a unit sharing a tile, when the player has one.
        shared = [
            u for u in units[source]
            if any(len(cache.units_by_tile.get(t, ())) > 1 for t in _footprint(scenario, u)[1] or ())
        ] if r % 2 == 0 else []
        shared_seeded += bool(shared)
        seed = rng.choice(shared or units[source])
        x0, y0 = int(seed.x) - WINDOW_W // 2, int(seed.y) - WINDOW_H // 2
        batch = [
            u for u in units[source]
            if x0 <= int(u.x) < x0 + WINDOW_W and y0 <= int(u.y) < y0 + WINDOW_H and _footprint(scenario, u)[1]
        ]
        model.begin_unit_edit([source, destination])
        splices = []
        for u in batch:
            own, tiles = _footprint(scenario, u)
            model.reassign(u, destination)
            splices.append(UnitSplice(
                destination, len(units[destination]) - 1, u, own, own, tiles, tiles, old_player_id=source
            ))
        model.commit_unit_edit(f"probe convert {r}", history)
        tag = f"round {r}: {len(splices)} units {source}->{destination}"

        eligible = render_cache._batch_splice_eligible(cache.units_by_tile, splices)
        plan = render_cache._splice_plan(scenario, cache.units_by_tile, cache.unit_filter, splices)
        if plan is None:
            assert _over_cap(scenario, cache, splices), f"{tag}: fell back below the component cap"
        wholesale.clear()
        cache.invalidate_units(splices)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        if plan is None:
            stats["over_cap"] += 1
            for mip in _mips(cache):
                cache.render_rect(0, 0, VIEW_PX, VIEW_PX, mip=mip)
        else:
            assert not wholesale, f"{tag}: the splice path rebuilt units_by_tile wholesale"
            stats["spliced" if eligible else "component"] += 1
        _assert_matches_fresh(cache, style, scenario, unit_filter, tag)

    if shared_seeded:
        assert stats["component"] or stats["over_cap"], f"no shared-seeded round left the batch path: {stats}"
