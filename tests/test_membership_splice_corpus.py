"""Corpus-tier probe of the Stepped/Sloped membership component splice
(unit-edit component splice plan, Step 1): random Draw-shaped and group-Move
batches through UnitEditModel on real files with real sprites, each spliced,
then compared for exact equality with a fresh cache, the Convert corpus
probe's checks (test_convert_splice_corpus._assert_matches_fresh) plus the
pixels of the batch's neighbourhood.

A Draw round removes the GAIA trees in a brush-5 window seeded on a tile two
units share, and rolls new trees onto window tiles, some of them holding
other units. A Move round shifts one player's units in a window by a small
delta, as _move_units() does. A wall Move round does the same around a wall
with an orthogonal connector neighbour, so the batch holds a moved wall and
its reshaped neighbours join the component (group-move-wall plan, Step 2).
C2_ElCid_coop_1 is radian-encoded, the only kind of file where a neighbour
re-derives at all; the other three are integer-only. A round may fall back to
the wholesale path only for an over-cap component.

Needs AOE2DE_INSTALL_PATH, since conftest hides the configured install.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from test_convert_splice_corpus import EXAMPLES, _assert_matches_fresh, _footprint, _make_cache, _mips

from descape import asset_source, iso_geometry, render, render_cache, terrain_units, unit_sprites
from descape.edit_history import EditHistory
from descape.render_cache import UnitSplice
from descape.scenario_io import load_map_and_units
from descape.unit_filter import UnitFilter
from descape.unit_model import UnitEditModel

FILES = (
    "F7_2_Dos Pilas (648).aoe2scenario",
    "old-allies-final-v2.aoe2scenario",
    "2_Joan_coop_1_v0_13.aoe2scenario",
    "C2_ElCid_coop_1_v0_16.aoe2scenario",
)
ROUNDS = 9
KINDS = ("draw", "move", "wall-move")
BRUSH = 5
PIXEL_PAD_TILES = 12
PIXEL_PAD_UP = 1024
TREES = sorted(terrain_units.TREE_CONSTS)


def _shared_seed(rng, scenario, units_by_tile):
    """A tile holding two or more visible units, as (x, y)."""
    shared = sorted(t for t, bucket in units_by_tile.items() if len(bucket) > 1)
    return rng.choice(shared)


def _window(scenario, cx: int, cy: int) -> list[tuple[int, int]]:
    mm = scenario.map_manager
    half = BRUSH // 2
    return [
        (x, y)
        for x in range(cx - half, cx + half + 1)
        for y in range(cy - half, cy + half + 1)
        if 0 <= x < mm.map_width and 0 <= y < mm.map_height
    ]


def _draw_round(rng, scenario, model, units_by_tile) -> list[UnitSplice]:
    """_apply_terrain_unit_plan()'s shape: remove_many, then add_many appended."""
    tiles = _window(scenario, *_shared_seed(rng, scenario, units_by_tile))
    window = set(tiles)
    gaia = scenario.unit_manager.units[0]
    removes = [u for u in gaia if u.unit_const in terrain_units.TREE_CONSTS and (int(u.x), int(u.y)) in window]
    position = {id(u): i for i, u in enumerate(gaia)}
    splices = []
    for unit in removes:
        own, occ = _footprint(scenario, unit)
        splices.append(UnitSplice(0, position[id(unit)], unit, own, None, occ or (), ()))
    adds = [
        terrain_units.UnitAddSpec(x + 0.5, y + 0.5, rng.choice(TREES), 0.0, 0)
        for x, y in rng.sample(tiles, min(len(tiles), 8))
    ]
    model.remove_many(removes)
    added = model.add_many(0, adds)
    base = len(gaia) - len(added)
    for offset, unit in enumerate(added):
        own, occ = _footprint(scenario, unit)
        splices.append(UnitSplice(0, base + offset, unit, None, own, (), occ or ()))
    return splices


def _wall_seed(rng, scenario, units_by_tile):
    """A non-GAIA rotation-variant wall with an orthogonal connector
    neighbour, as its (x, y); a shared tile if the file has none."""
    mm = scenario.map_manager
    units = scenario.unit_manager.units
    connectors = unit_sprites.wall_connector_consts()
    tiles = {
        t for per in units for u in per if u.unit_const in connectors
        for t in render.unit_occupied_tiles(u, mm.map_width, mm.map_height) or ()
    }
    seeds = sorted(
        (int(u.x), int(u.y)) for per in units[1:] for u in per
        if unit_sprites.rotation_variant_eligible(u.unit_const) and unit_sprites.neighbour_mask(int(u.x), int(u.y), tiles)
    )
    return rng.choice(seeds) if seeds else _shared_seed(rng, scenario, units_by_tile)


def _move_round(rng, scenario, model, units_by_tile, seed=_shared_seed) -> list[UnitSplice]:
    """_move_units()' shape: a player's units in a window, one delta, fields only."""
    units = scenario.unit_manager.units
    cx, cy = seed(rng, scenario, units_by_tile)
    window = set(_window(scenario, cx, cy))
    candidates = [
        (p, i, u) for p in range(1, len(units)) for i, u in enumerate(units[p])
        if (int(u.x), int(u.y)) in window and _footprint(scenario, u)[1]
    ]
    if not candidates:
        return []
    walls = [p for p, _i, u in candidates if seed is _wall_seed and unit_sprites.rotation_variant_eligible(u.unit_const)]
    player = walls[0] if walls else rng.choice(candidates)[0]
    dx, dy = rng.choice([(1, 0), (0, 1), (-1, 1), (2, 2), (-2, 1)])
    mm = scenario.map_manager
    splices = []
    for p, i, unit in candidates:
        if p != player or not (0 <= unit.x + dx < mm.map_width and 0 <= unit.y + dy < mm.map_height):
            continue
        old_own, old_tiles = _footprint(scenario, unit)
        model.set_position(unit, unit.x + dx, unit.y + dy, unit.z)
        new_own, new_tiles = _footprint(scenario, unit)
        splices.append(UnitSplice(p, i, unit, old_own, new_own, old_tiles or (), new_tiles or ()))
    return splices


def _assert_pixels_match(cache, style: str, scenario, unit_filter, splices, tag: str) -> None:
    """mip 0 around the batch against a fresh cache: the tiles' corners at
    elevation 0, padded generously for sprite reach and elevation."""
    tiles = render_cache._splice_tiles(splices)
    corners = [
        iso_geometry.tile_screen_origin(x + dx, y + dy, 0, cache.proj)
        for x, y in tiles
        for dx in (-PIXEL_PAD_TILES, PIXEL_PAD_TILES)
        for dy in (-PIXEL_PAD_TILES, PIXEL_PAD_TILES)
    ]
    w, h = cache.canvas_dims(0)
    px0 = max(0, min(c[0] for c in corners))
    py0 = max(0, min(c[1] for c in corners) - PIXEL_PAD_UP)
    px1 = min(w, max(c[0] for c in corners) + cache.proj.tile_px)
    py1 = min(h, max(c[1] for c in corners) + cache.proj.tile_px)
    fresh = _make_cache(style, scenario, unit_filter)
    got = cache.render_rect(px0, py0, px1, py1, mip=0)
    want = fresh.render_rect(px0, py0, px1, py1, mip=0)
    assert np.array_equal(got, want), f"{tag}: mip 0 pixels around the batch"


@pytest.mark.corpus
@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("name", FILES)
def test_random_draw_and_move_batches_splice_exactly_like_a_fresh_cache(name, style, monkeypatch):
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not in examples/")
    scenario = load_map_and_units(path)
    rng = random.Random(f"membership-{name}-{style}")
    unit_filter = UnitFilter()
    cache = _make_cache(style, scenario, unit_filter)
    model = UnitEditModel(scenario)
    history = EditHistory()
    wholesale = []
    real_units_by_tile = render._units_by_tile
    monkeypatch.setattr(render, "_units_by_tile", lambda *a, **k: (wholesale.append(1), real_units_by_tile(*a, **k))[1])
    stats = {"batch": 0, "component": 0, "wholesale": 0, "wall": 0, "reshaped": 0}

    for r in range(ROUNDS):
        kind = KINDS[r % len(KINDS)]
        shapes_before = dict(render.wall_variant_rotation_overrides(scenario))
        model.begin_unit_edit(list(range(len(scenario.unit_manager.units))), fields_only=kind != "draw")
        if kind == "draw":
            splices = _draw_round(rng, scenario, model, cache.units_by_tile)
        else:
            seed = _wall_seed if kind == "wall-move" else _shared_seed
            splices = _move_round(rng, scenario, model, cache.units_by_tile, seed)
        model.commit_unit_edit(f"probe {kind} {r}", history)
        if not splices:
            continue
        tag = f"round {r} {kind}: {len(splices)} splices"

        plan, reason = render_cache._splice_plan_or_refusal(scenario, cache.units_by_tile, unit_filter, splices)
        if plan is None:
            assert reason == "cap" or reason.endswith("/component:cap"), f"{tag}: fell back for {reason}"
        elif any(unit_sprites.rotation_variant_eligible(s.unit.unit_const) for s in splices):
            stats["wall"] += 1
            moved = {(s.player_id, s.index) for s in splices}
            shapes = render.wall_variant_rotation_overrides(scenario)
            stats["reshaped"] += any(
                shapes.get(k) != shapes_before.get(k) for k in {*shapes, *shapes_before} if k not in moved
            )
        wholesale.clear()
        cache.invalidate_units(splices)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        if plan is None:
            stats["wholesale"] += 1
            for mip in _mips(cache):
                cache.render_rect(0, 0, 1024, 1024, mip=mip)
        else:
            assert not wholesale, f"{tag}: the splice path rebuilt units_by_tile wholesale"
            stats["batch" if plan is splices else "component"] += 1
        _assert_matches_fresh(cache, style, scenario, unit_filter, tag)
        if plan is not None and plan is not splices:
            _assert_pixels_match(cache, style, scenario, unit_filter, splices, tag)

    assert stats["component"], f"no round took the component path: {stats}"
    assert stats["wall"], f"no round spliced a moved wall: {stats}"
    if name.startswith("C2_ElCid"):
        assert stats["reshaped"], f"no wall round reshaped a stationary wall in the radian file: {stats}"
