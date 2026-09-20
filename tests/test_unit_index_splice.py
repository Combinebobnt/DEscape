"""Batch D's D4: patch_index_for_move()/patch_index_for_add() (unit_pick.py)
update a UnitIndex without a full build_index() walk -- the "order-shifting
rule": a move patches by_tile and the moved entry in place, an add appends,
and a remove or reassign still needs a full build_index() (out of this
file's scope, since order shifts under those two).

Oracle for both: build_index() on the scenario AFTER the edit, called fresh,
must describe the exact same index a patch would have produced starting from
the BEFORE index -- same entries (by value), same by_key, same by_tile
membership per tile (as sets: intra-tile LIST order is not part of either
function's contract, see unit_pick.pick_unit's own per-style key, which
resolves ties through `order`/depth_order rather than raw list position)."""

from __future__ import annotations

from descape import render
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_filter import UnitFilter
from descape.unit_pick import build_index, patch_index_for_add, patch_index_for_move, unit_key
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

MAP_W = MAP_H = 20
BUILDING_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == 3 and sy == 3)
PLAIN_CONST = 999999


def _grid() -> list[SyntheticTile]:
    return [SyntheticTile(x, y, elevation=0) for y in range(MAP_H) for x in range(MAP_W)]


def _scenario(units_by_player):
    return FakeScenario(MAP_W, MAP_H, _grid(), units_by_player)


def _by_tile_as_sets(index):
    return {tile: frozenset(orders) for tile, orders in index.by_tile.items()}


def _assert_indexes_equivalent(patched, fresh) -> None:
    assert len(patched.entries) == len(fresh.entries)
    for a, b in zip(patched.entries, fresh.entries, strict=True):
        assert (a.player_id, a.unit, a.own_x, a.own_y, a.order) == (b.player_id, b.unit, b.own_x, b.own_y, b.order)
    assert set(patched.by_key) == set(fresh.by_key)
    for key in fresh.by_key:
        a, b = patched.by_key[key], fresh.by_key[key]
        assert (a.player_id, a.unit, a.own_x, a.own_y, a.order) == (b.player_id, b.unit, b.own_x, b.own_y, b.order)
    assert _by_tile_as_sets(patched) == _by_tile_as_sets(fresh)


def test_patch_index_for_move_matches_a_fresh_build():
    mover = SyntheticUnit(5.0, 5.0, BUILDING_CONST, reference_id=1)
    bystander = SyntheticUnit(2.0, 2.0, PLAIN_CONST, reference_id=2)
    scenario = _scenario([[], [mover, bystander]])
    index = build_index(scenario)

    mm = scenario.map_manager
    old_bounds = render.unit_tile_bounds(mover, mm.map_width, mm.map_height)
    mover.x, mover.y = 12.0, 12.0
    patch_index_for_move(scenario, index, 1, mover, old_bounds)

    fresh = build_index(scenario)
    _assert_indexes_equivalent(index, fresh)


def test_patch_index_for_move_leaves_an_unrelated_neighbour_untouched():
    """A regression this file exists to catch: patch_index_for_move must
    not disturb another unit's by_tile membership or `order`, even one
    sitting right next to the mover's OLD footprint."""
    mover = SyntheticUnit(5.0, 5.0, BUILDING_CONST, reference_id=1)  # 3x3, own tile (5, 5)
    neighbour = SyntheticUnit(6.0, 5.0, PLAIN_CONST, reference_id=2)  # adjacent tile
    scenario = _scenario([[], [mover, neighbour]])
    index = build_index(scenario)
    neighbour_key = unit_key(1, neighbour)
    neighbour_order = index.by_key[neighbour_key].order
    neighbour_tile = (int(neighbour.x), int(neighbour.y))
    assert neighbour_order in index.by_tile[neighbour_tile]

    mm = scenario.map_manager
    old_bounds = render.unit_tile_bounds(mover, mm.map_width, mm.map_height)
    mover.x, mover.y = 15.0, 15.0
    patch_index_for_move(scenario, index, 1, mover, old_bounds)

    assert index.by_key[neighbour_key].order == neighbour_order
    assert index.by_tile[neighbour_tile] == [neighbour_order]


def test_patch_index_for_add_matches_a_fresh_build():
    existing = SyntheticUnit(2.0, 2.0, PLAIN_CONST, reference_id=1)
    scenario = _scenario([[], [existing]])
    index = build_index(scenario)

    new_unit = SyntheticUnit(10.0, 10.0, BUILDING_CONST, reference_id=2)
    scenario.unit_manager.units[1].append(new_unit)
    patch_index_for_add(scenario, index, 1, new_unit)

    fresh = build_index(scenario)
    _assert_indexes_equivalent(index, fresh)


def test_patch_index_for_add_is_a_noop_when_the_filter_hides_it():
    scenario = _scenario([[], []])
    hidden = SyntheticUnit(10.0, 10.0, BUILDING_CONST, reference_id=1)
    scenario.unit_manager.units[1].append(hidden)
    index = build_index(scenario, unit_filter=UnitFilter(players=frozenset({2})))

    patch_index_for_add(scenario, index, 1, hidden, unit_filter=UnitFilter(players=frozenset({2})))

    assert index.entries == []
    assert index.by_tile == {}


def test_patch_index_for_move_is_a_noop_for_an_unindexed_unit():
    """A unit the filter already hid has no by_key entry -- patch_index_for_
    move must leave the (empty) index untouched rather than raising, the
    caller's cue to fall back to a full rebuild instead."""
    scenario = _scenario([[], []])
    unit = SyntheticUnit(5.0, 5.0, BUILDING_CONST, reference_id=1)
    scenario.unit_manager.units[1].append(unit)
    index = build_index(scenario, unit_filter=UnitFilter(players=frozenset({2})))
    assert index.entries == []

    mm = scenario.map_manager
    old_bounds = render.unit_tile_bounds(unit, mm.map_width, mm.map_height)
    unit.x, unit.y = 12.0, 12.0
    patch_index_for_move(scenario, index, 1, unit, old_bounds)

    assert index.entries == []
    assert index.by_tile == {}
