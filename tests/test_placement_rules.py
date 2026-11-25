"""descape/placement_rules.py (GH #176/#177, GH #174): the terrain an object
may stand on and the objects it may not overlap. Qt-free.

Synthetic grids from testkit.fakes with real consts, so the committed
object_catalog.json rows are what is under test. The corpus-marked test at
the bottom re-measures both rules against real files: neither is exact, and
the day one of them starts refusing a tenth of what authors really placed is
the day it stopped matching the game.
"""

from __future__ import annotations

import random

import pytest

from descape import asset_source, game_dataset, object_catalog, placement_rules
from descape.terrain_palette import BUILDING_TILE_SPANS, DOODAD_TILE_SPANS, FOUNDATION_TERRAIN
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

import conftest

VILLAGER = 83
ARCHER = 4
FISHING_SHIP = 13
HOUSE = 70  # 2x2
TOWN_CENTER = 109  # 4x4, obstruction 0 in the .dat
FARM = 50  # 3x3, class 49
SHORE_FISH = 69
STONE_WALL = 117
GATE_STRAIGHT = 64  # 4x1
GATE_DIAGONAL = 659  # 4x4 bounds, six real tiles
GATE_RUN_Y = 88  # GTBA2, 1x4
GATE_RUN_Y_CORNER = 95  # GTBC2, its 1x1 corner piece
RUGS = 711  # eye candy: obstruction 0 with a 0.5 clearance
COLOSSUS = 263  # 8x8, the widest footprint in the catalog
HAWK = 96  # obstruction 5 with zero clearance

GRASS = 0
WATER_SHALLOW = 1
BEACH = 2
WATER_DEEP = 22
BEACH_WET = 107
SIZE = 24


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """Messages are asserted by name, and AOE2DE_INSTALL_PATH would swap in the install's."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _scenario(units_by_player=(), terrain=None, fill: int = GRASS) -> FakeScenario:
    """A SIZE x SIZE map of `fill`, with `terrain` as {(x, y): terrain_id} overrides."""
    terrain = terrain or {}
    tiles = [SyntheticTile(x, y, 0, terrain.get((x, y), fill)) for y in range(SIZE) for x in range(SIZE)]
    return FakeScenario(SIZE, SIZE, tiles, [list(units) for units in units_by_player] or [[]])


def _unit(const: int, x: float, y: float, garrisoned_in_id: int = -1) -> SyntheticUnit:
    return SyntheticUnit(x, y, const, garrisoned_in_id=garrisoned_in_id)


# --- terrain --------------------------------------------------------------------


def test_a_villager_is_refused_on_deep_water_and_allowed_on_grass() -> None:
    scenario = _scenario(terrain={(5, 5): WATER_DEEP})
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 5.5, 5.5) == "Villager Male can't stand on Water Deep"
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 6.5, 5.5) is None


def test_a_fishing_ship_is_the_reverse() -> None:
    scenario = _scenario(terrain={(5, 5): WATER_DEEP})
    assert placement_rules.terrain_refusal(scenario, FISHING_SHIP, 5.5, 5.5) is None
    assert placement_rules.terrain_refusal(scenario, FISHING_SHIP, 6.5, 5.5) == "Fishing Ship can't stand on Grass 1"


def test_a_house_is_refused_when_any_footprint_tile_is_beach() -> None:
    """The House at (10, 10) covers tiles 9..10 on both axes; one of the four is Beach."""
    scenario = _scenario(terrain={(9, 10): BEACH})
    assert placement_rules.terrain_refusal(scenario, HOUSE, 10, 10) == "House can't stand on Beach"
    assert placement_rules.terrain_refusal(scenario, HOUSE, 12, 10) is None
    # A villager walks on that same Beach tile.
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 9.5, 10.5) is None


@pytest.mark.parametrize("shore", [BEACH, BEACH_WET])
def test_shore_fish_need_a_beach_beside_them(shore: int) -> None:
    """GH #174. Refused in open water, allowed with a shore in the 8-tile
    ring, a diagonal included. BEACH_WET is not one of the .dat's two literal
    side-terrain ids: it counts through the Beach restriction row."""
    open_water = _scenario(fill=WATER_SHALLOW)
    assert placement_rules.terrain_refusal(open_water, SHORE_FISH, 5.5, 5.5) == "Shore Fish needs a beach tile beside it"
    beside = _scenario(fill=WATER_SHALLOW, terrain={(6, 6): shore})
    assert placement_rules.terrain_refusal(beside, SHORE_FISH, 5.5, 5.5) is None
    two_away = _scenario(fill=WATER_SHALLOW, terrain={(7, 5): shore})
    assert placement_rules.terrain_refusal(two_away, SHORE_FISH, 5.5, 5.5) is not None


def test_shore_fish_still_cannot_stand_on_the_beach_itself() -> None:
    """The footprint rule comes first: the fish's own tile has to be water."""
    scenario = _scenario(fill=WATER_SHALLOW, terrain={(5, 5): BEACH})
    assert placement_rules.terrain_refusal(scenario, SHORE_FISH, 5.5, 5.5) == "Shore Fish can't stand on Beach"


def test_grass_beside_a_shore_fish_is_not_a_shore() -> None:
    scenario = _scenario(fill=WATER_SHALLOW, terrain={(6, 5): GRASS})
    assert placement_rules.terrain_refusal(scenario, SHORE_FISH, 5.5, 5.5) is not None


def test_unknown_data_never_refuses() -> None:
    """A const or a terrain id the catalog lacks passes, on its own tile and in a shore ring."""
    unknown_terrain = 9999
    assert unknown_terrain not in object_catalog.dat_terrains()
    scenario = _scenario(terrain={(5, 5): unknown_terrain, (8, 8): WATER_DEEP})
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 5.5, 5.5) is None
    assert placement_rules.terrain_refusal(scenario, 999999, 8.5, 8.5) is None
    assert placement_rules.refusal(scenario, 999999, 8.5, 8.5) is None
    ring = _scenario(fill=WATER_SHALLOW, terrain={(6, 5): unknown_terrain})
    assert placement_rules.terrain_refusal(ring, SHORE_FISH, 5.5, 5.5) is None


def test_an_off_map_point_is_not_the_terrain_rules_to_refuse() -> None:
    scenario = _scenario(fill=WATER_DEEP)
    assert placement_rules.terrain_refusal(scenario, VILLAGER, SIZE + 3.5, 5.5) is None


# --- laid terrain (GH #173: Map Analysis reads a bridge's deck) ------------------

BRIDGE_MIDDLE = 606  # Bridge A--Middle, 2x3, foundation terrain 28 (Water 2D, Bridge)
BRIDGE_TILES = {(9, 9), (10, 9), (9, 10), (10, 10), (9, 11), (10, 11)}  # at (10, 10.5)


def test_a_villager_on_a_bridge_over_shallow_water_passes_only_with_laid_terrain() -> None:
    assert FOUNDATION_TERRAIN[BRIDGE_MIDDLE] == 28
    scenario = _scenario([[_unit(BRIDGE_MIDDLE, 10, 10.5)]], fill=WATER_SHALLOW)
    laid = placement_rules.laid_terrain(scenario)
    assert laid == {tile: frozenset({28}) for tile in BRIDGE_TILES}
    refused = "Villager Male can't stand on Water Shallow"
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 9.5, 10.5) == refused
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 9.5, 10.5, laid) is None
    # Off the deck, the water still refuses.
    assert placement_rules.terrain_refusal(scenario, VILLAGER, 11.5, 10.5, laid) == refused
    # TASK-349: Place Unit and Move (refusal()) read the deck too, and only the deck.
    assert placement_rules.refusal(scenario, VILLAGER, 9.5, 10.5) is None
    assert placement_rules.refusal(scenario, VILLAGER, 11.5, 10.5) == refused


def test_a_garrisoned_bridge_and_a_const_without_a_foundation_lay_nothing() -> None:
    assert STONE_WALL not in FOUNDATION_TERRAIN
    garrisoned = _scenario([[_unit(BRIDGE_MIDDLE, 10, 10.5, garrisoned_in_id=7)]], fill=WATER_SHALLOW)
    assert placement_rules.laid_terrain(garrisoned) == {}
    walls = _scenario([[_unit(STONE_WALL, 9.5, 10.5), _unit(VILLAGER, 9.5, 10.5)]], fill=WATER_SHALLOW)
    assert placement_rules.laid_terrain(walls) == {}


def test_the_laid_terrain_memo_rebuilds_on_a_dataset_switch(monkeypatch) -> None:
    """A data mod refills FOUNDATION_TERRAIN in place with no unit edit; the generation bump must drop the memo."""
    scenario = _scenario([[_unit(BRIDGE_MIDDLE, 10, 10.5)]], fill=WATER_SHALLOW)
    scenario.unit_gen = 0
    assert placement_rules._memoized_laid_terrain(scenario) == {tile: frozenset({28}) for tile in BRIDGE_TILES}
    monkeypatch.setitem(FOUNDATION_TERRAIN, BRIDGE_MIDDLE, BEACH)
    assert placement_rules._memoized_laid_terrain(scenario)[(9, 9)] == frozenset({28}), "no switch, no rebuild"
    monkeypatch.setattr(game_dataset, "_generation", game_dataset.generation() + 1)
    assert placement_rules._memoized_laid_terrain(scenario) == {tile: frozenset({BEACH}) for tile in BRIDGE_TILES}


def test_a_moved_bridge_lays_its_deck_where_it_lands_and_a_group_without_one_keeps_the_memo() -> None:
    """TASK-483: moved_laid_terrain() for a group move; None means no moved unit lays a foundation."""
    bridge, villager = _unit(BRIDGE_MIDDLE, 10, 10.5), _unit(VILLAGER, 20.5, 20.5)
    scenario = _scenario([[bridge, villager]], fill=WATER_SHALLOW)
    moved = placement_rules.moved_laid_terrain(scenario, {id(bridge): (14, 10.5), id(villager): (24.5, 20.5)})
    assert moved == {(x + 4, y): frozenset({28}) for x, y in BRIDGE_TILES}
    assert placement_rules.moved_laid_terrain(scenario, {id(villager): (24.5, 20.5)}) is None
    assert placement_rules.move_refusal(scenario, villager, 13.5, 10.5) is not None
    assert placement_rules.move_refusal(scenario, villager, 13.5, 10.5, laid=moved) is None
    assert placement_rules.move_refused(scenario, villager, 9.5, 10.5, laid=moved)


# --- objects --------------------------------------------------------------------


def _blocked(units, const: int, x: float, y: float, exclude=()) -> bool:
    return placement_rules.blocker(_scenario([units]), const, x, y, exclude) is not None


def test_two_villagers_collide_on_one_point_and_not_half_a_tile_apart() -> None:
    """Clearance boxes, not tile occupancy: both points are inside tile (5, 5)."""
    standing = [_unit(VILLAGER, 5.25, 5.5)]
    assert _blocked(standing, VILLAGER, 5.25, 5.5)
    assert _blocked(standing, VILLAGER, 5.5, 5.5)  # 0.25 apart, two 0.2 half-widths
    assert not _blocked(standing, VILLAGER, 5.75, 5.5)


def test_a_villager_on_a_house_collides() -> None:
    house = [_unit(HOUSE, 10, 10)]
    assert _blocked(house, VILLAGER, 10.5, 10.5)
    assert _blocked(house, VILLAGER, 9.5, 9.5)
    assert not _blocked(house, VILLAGER, 11.5, 10.5)


def test_two_houses_sharing_an_edge_do_not_collide() -> None:
    house = [_unit(HOUSE, 10, 10)]
    assert not _blocked(house, HOUSE, 12, 10)
    assert not _blocked(house, HOUSE, 10, 8)
    assert _blocked(house, HOUSE, 11, 10)
    # Coordinates are float32 in the file: an edge 0.0005 inside the other is still shared.
    assert not _blocked(house, HOUSE, 11.9995, 10)
    assert _blocked(house, HOUSE, 11.99, 10)


def test_a_town_center_is_solid_though_the_dat_says_obstruction_0() -> None:
    """The BUILDING_TILE_SPANS arm of boxes(): a House on a Town Center and a
    Town Center on a House both collide."""
    assert "obstruction" not in object_catalog.dat_objects()[TOWN_CENTER]
    assert _blocked([_unit(TOWN_CENTER, 10, 10)], HOUSE, 11, 11)
    assert _blocked([_unit(HOUSE, 11, 11)], TOWN_CENTER, 10, 10)
    assert not _blocked([_unit(TOWN_CENTER, 10, 10)], HOUSE, 13, 10)


def test_a_farm_is_passable_both_ways() -> None:
    """Class 49 is on the passable allowlist: a unit stands on a Farm, and a
    Farm goes down over a unit (stacking is TASK-244's ground)."""
    assert placement_rules.boxes(FARM, 10.5, 10.5) == ()
    assert not _blocked([_unit(FARM, 10.5, 10.5)], VILLAGER, 10.5, 10.5)
    assert not _blocked([_unit(VILLAGER, 10.5, 10.5)], FARM, 10.5, 10.5)


def test_the_solid_obstruction_0_buildings_are_the_town_centers_and_the_wonder_shell() -> None:
    """The allowlist is a judgment over the 156 obstruction-0 buildings. This
    pins what it leaves solid, so a .dat regen that adds one fails here and
    gets a decision instead of silently blocking or passing."""
    objects = object_catalog.dat_objects()
    solid = {
        const
        for const in BUILDING_TILE_SPANS
        if "obstruction" not in objects[const] and placement_rules.boxes(const, 10.5, 10.5)
    }
    assert solid == {71, 109, 141, 142, 182, 482, 483, 612, 613, 615, 616, 619, 620, 2275, 2276, 2277}
    assert {objects[const]["class"] for const in solid} == {3}


def test_a_diagonal_gate_blocks_on_its_six_tiles_only() -> None:
    """A wall in the 4x4 bounding box's empty corner does not collide; one on
    a real tile does. Both directions, since the gate is the mover on a click."""
    gate_boxes = placement_rules.boxes(GATE_DIAGONAL, 10, 10)
    assert len(gate_boxes) == 6
    on_tiles = {(int(x0), int(y0)) for x0, y0, _x1, _y1 in gate_boxes}
    corners = {(8, 11), (11, 8)} - on_tiles
    assert corners, "this gate's empty corners moved; pick two from its bounding box"
    for tx, ty in corners:
        wall = [_unit(STONE_WALL, tx + 0.5, ty + 0.5)]
        assert not _blocked(wall, GATE_DIAGONAL, 10, 10)
        assert not _blocked([_unit(GATE_DIAGONAL, 10, 10)], STONE_WALL, tx + 0.5, ty + 0.5)
    for tx, ty in on_tiles:
        wall = [_unit(STONE_WALL, tx + 0.5, ty + 0.5)]
        assert _blocked(wall, GATE_DIAGONAL, 10, 10)
        assert _blocked([_unit(GATE_DIAGONAL, 10, 10)], STONE_WALL, tx + 0.5, ty + 0.5)


MOUNTAIN = 744  # MNTN3: a 6x6 footprint (TASK-134), clearance 2.0


def test_a_mountain_blocks_with_its_4x4_clearance_box_inside_its_6x6_footprint() -> None:
    """TASK-134 decision 4: the object rule keeps the game's clearance box,
    so it answers exactly as before the footprint grew."""
    assert DOODAD_TILE_SPANS[MOUNTAIN] == (6, 6)
    assert placement_rules.boxes(MOUNTAIN, 20.0, 20.0) == ((18.0, 18.0, 22.0, 22.0),)
    assert _blocked([_unit(MOUNTAIN, 20.0, 20.0)], VILLAGER, 21.5, 21.5)
    assert not _blocked([_unit(MOUNTAIN, 20.0, 20.0)], VILLAGER, 22.5, 22.5)  # footprint, outside the box


def test_a_mountains_terrain_rule_reads_all_36_footprint_tiles() -> None:
    footprint = [(x, y) for y in range(17, 23) for x in range(17, 23)]
    for corner in (footprint[0], footprint[-1]):
        scenario = _scenario(terrain={corner: WATER_DEEP})
        assert placement_rules.terrain_refusal(scenario, MOUNTAIN, 20.0, 20.0) is not None, corner
    outside = _scenario(terrain={(16, 16): WATER_DEEP, (23, 23): WATER_DEEP})
    assert placement_rules.terrain_refusal(outside, MOUNTAIN, 20.0, 20.0) is None


def test_a_straight_gate_is_one_box_of_its_span() -> None:
    assert placement_rules.boxes(GATE_STRAIGHT, 10, 10.5) == ((8.0, 10.0, 12.0, 11.0),)


def test_a_garrisoned_unit_never_blocks() -> None:
    assert _blocked([_unit(VILLAGER, 5.5, 5.5)], VILLAGER, 5.5, 5.5)
    assert not _blocked([_unit(VILLAGER, 5.5, 5.5, garrisoned_in_id=7)], VILLAGER, 5.5, 5.5)


def test_passable_eye_candy_and_zero_clearance_objects_never_block() -> None:
    """Obstruction decides, not clearance: RUGS has a 0.5 clearance and
    obstruction 0. A hawk obstructs but has no clearance box at all."""
    objects = object_catalog.dat_objects()
    assert objects[RUGS]["clearance"] == [0.5, 0.5] and "obstruction" not in objects[RUGS]
    assert objects[HAWK]["obstruction"] and "clearance" not in objects[HAWK]
    for const in (RUGS, HAWK, SHORE_FISH):
        assert placement_rules.boxes(const, 5.5, 5.5) == ()
        assert not _blocked([_unit(const, 5.5, 5.5)], VILLAGER, 5.5, 5.5)
        assert not _blocked([_unit(VILLAGER, 5.5, 5.5)], const, 5.5, 5.5)


def test_exclude_is_by_identity() -> None:
    first, second = _unit(VILLAGER, 5.5, 5.5), _unit(VILLAGER, 5.5, 5.5)
    scenario = _scenario([[first, second]])
    assert placement_rules.blocker(scenario, VILLAGER, 5.5, 5.5) == (0, first)
    # Equal dataclasses, so only an identity test tells them apart.
    assert first == second
    found = placement_rules.blocker(scenario, VILLAGER, 5.5, 5.5, exclude=[first])
    assert found is not None and found[1] is second
    assert placement_rules.blocker(scenario, VILLAGER, 5.5, 5.5, exclude=[first, second]) is None


def test_blocker_names_the_owner_and_the_first_unit_in_list_order() -> None:
    gaia_house, player_house = _unit(HOUSE, 10, 10), _unit(HOUSE, 11, 10)
    scenario = _scenario([[gaia_house], [], [player_house]])
    assert placement_rules.blocker(scenario, VILLAGER, 10.5, 10.5) == (0, gaia_house)
    found = placement_rules.blocker(scenario, VILLAGER, 11.5, 10.5)
    assert found is not None and found[0] == 2 and found[1] is player_house


def test_the_prefilter_reaches_the_widest_footprint() -> None:
    """blocker() skips far units before any catalog lookup. Its bound has to
    cover an 8x8 monument, as the standing unit and as the mover."""
    assert BUILDING_TILE_SPANS[COLOSSUS] == (8, 8)
    monument = [_unit(COLOSSUS, 12, 12)]
    assert _blocked(monument, VILLAGER, 15.5, 12.5)
    assert _blocked(monument, VILLAGER, 8.5, 8.5)
    assert not _blocked(monument, VILLAGER, 16.5, 12.5)
    assert _blocked([_unit(VILLAGER, 15.5, 12.5)], COLOSSUS, 12, 12)
    assert _blocked([_unit(COLOSSUS, 4, 12)], COLOSSUS, 11, 12)
    assert not _blocked([_unit(COLOSSUS, 4, 12)], COLOSSUS, 12, 12)


def test_the_grid_answers_exactly_as_the_scan_does() -> None:
    """BlockerGrid is the group move's index. Same answer, same unit, for
    every mover shape, with build-time and per-query exclusions."""
    rng = random.Random(245)
    consts = [VILLAGER, ARCHER, HOUSE, TOWN_CENTER, FARM, STONE_WALL, GATE_STRAIGHT, GATE_DIAGONAL, RUGS, COLOSSUS]
    half_steps = [n / 2 for n in range(4, 2 * SIZE - 4)]
    units_by_player = [
        [
            _unit(rng.choice(consts), rng.choice(half_steps), rng.choice(half_steps), rng.choice([-1, -1, -1, 5]))
            for _ in range(14)
        ]
        for _player in range(3)
    ]
    scenario = _scenario(units_by_player)
    everyone = [unit for units in units_by_player for unit in units]
    excluded = rng.sample(everyone, 12)
    grid = placement_rules.BlockerGrid(scenario, exclude=excluded)
    blocked = 0
    for _ in range(600):
        const = rng.choice(consts)
        x = rng.uniform(2, SIZE - 2) if rng.random() < 0.5 else rng.choice(half_steps)
        y = rng.uniform(2, SIZE - 2) if rng.random() < 0.5 else rng.choice(half_steps)
        skip = rng.sample(everyone, 2)
        expected = placement_rules.blocker(scenario, const, x, y, exclude=excluded + skip)
        found = grid.blocker(const, x, y, exclude=skip)
        if expected is None:
            assert found is None, (const, x, y)
        else:
            assert found is not None and found[0] == expected[0] and found[1] is expected[1], (const, x, y)
            blocked += 1
    assert 60 < blocked < 540, "the sample stopped exercising both answers"


def test_a_unit_the_grid_adds_blocks_a_later_query() -> None:
    """GH #170's brush grows the grid as it places, so the next candidate sees the last."""
    scenario = _scenario([[], []])
    grid = placement_rules.BlockerGrid(scenario)
    assert grid.blocker(VILLAGER, 10.5, 10.5) is None
    house = _unit(HOUSE, 10, 10)
    scenario.unit_manager.units[1].append(house)
    grid.add(1, house)
    assert grid.blocker(VILLAGER, 10.5, 10.5) == (1, house)
    assert grid.blocker(VILLAGER, 12.5, 10.5) is None


def test_an_added_gaia_unit_is_reported_before_a_player_one_unit_as_the_scan_does() -> None:
    """Order is (player, list index), not build order: a GAIA unit added after
    the build still comes first, exactly as a fresh blocker() scan finds it."""
    player_house = _unit(HOUSE, 11, 10)
    scenario = _scenario([[_unit(ARCHER, 3.5, 3.5)], [player_house]])
    grid = placement_rules.BlockerGrid(scenario)
    gaia_house = _unit(HOUSE, 10, 10)
    scenario.unit_manager.units[0].append(gaia_house)
    grid.add(0, gaia_house)
    expected = placement_rules.blocker(scenario, VILLAGER, 10.5, 10.5)
    assert expected == (0, gaia_house)
    found = grid.blocker(VILLAGER, 10.5, 10.5)
    assert found is not None and found[0] == 0 and found[1] is gaia_house
    fresh = placement_rules.BlockerGrid(scenario).blocker(VILLAGER, 10.5, 10.5)
    assert fresh is not None and fresh[1] is gaia_house


def test_a_garrisoned_unit_the_grid_adds_never_blocks() -> None:
    scenario = _scenario([[], []])
    grid = placement_rules.BlockerGrid(scenario)
    occupant = _unit(VILLAGER, 10.5, 10.5, garrisoned_in_id=4)
    scenario.unit_manager.units[1].append(occupant)
    grid.add(1, occupant)
    assert grid.blocker(VILLAGER, 10.5, 10.5) is None


# --- refused(): refusal() without the message ---------------------------------------


def _refused_cases():
    """A map with water, a beach strip, a bridge over shallow water and some blockers."""
    terrain = {(x, y): WATER_DEEP for x in range(6) for y in range(SIZE)}
    terrain.update({(6, y): BEACH for y in range(SIZE)})
    terrain.update({(x, y): WATER_SHALLOW for x in range(14, 18) for y in range(8, 14)})
    house, archer = _unit(HOUSE, 10, 14), _unit(ARCHER, 8.5, 4.5)
    bridge = _unit(BRIDGE_MIDDLE, 16, 10.5)
    scenario = _scenario([[archer], [house, bridge]], terrain=terrain)
    consts = [VILLAGER, ARCHER, FISHING_SHIP, HOUSE, TOWN_CENTER, FARM, SHORE_FISH, STONE_WALL, RUGS, 999999]
    points = [(x / 2, y / 2) for x in range(0, 2 * SIZE + 2, 3) for y in range(0, 2 * SIZE + 2, 3)]
    return scenario, consts, points, house


def test_refused_answers_exactly_as_refusal_does() -> None:
    """TASK-327's predicate: over a point grid x consts, with exclude and with a grid."""
    scenario, consts, points, house = _refused_cases()
    grid = placement_rules.BlockerGrid(scenario)
    grid_without = placement_rules.BlockerGrid(scenario, exclude=[house])
    answers = set()
    for const in consts:
        for x, y in points:
            for exclude, used_grid in (((), None), ((house,), None), ((), grid), ((), grid_without)):
                expected = placement_rules.refusal(scenario, const, x, y, exclude=exclude, grid=used_grid) is not None
                assert placement_rules.refused(scenario, const, x, y, exclude=exclude, grid=used_grid) == expected, (
                    const, x, y, exclude, used_grid,
                )
                answers.add(expected)
    assert answers == {True, False}


def test_refused_reads_a_caller_held_laid_map_in_place_of_the_memo() -> None:
    """The brush holds one laid_terrain() for its stroke: a bridge it knows
    lifts the water rule, an empty map does not, and the answer with the
    real map is the memo's."""
    scenario, _consts, _points, _house = _refused_cases()
    laid = placement_rules.laid_terrain(scenario)
    on_bridge = (16.5, 10.5)
    assert placement_rules.refused(scenario, VILLAGER, *on_bridge) is False
    assert placement_rules.refused(scenario, VILLAGER, *on_bridge, laid=laid) is False
    assert placement_rules.refused(scenario, VILLAGER, *on_bridge, laid={}) is True


def test_move_refused_skips_a_garrisoned_unit_as_move_refusal_does() -> None:
    transport = _unit(HOUSE, 12, 14)
    scenario = _scenario([[], [transport]], terrain={(5, 5): WATER_DEEP})
    walker, occupant = _unit(VILLAGER, 3.5, 3.5), _unit(VILLAGER, 12, 14, garrisoned_in_id=9)
    for unit in (walker, occupant):
        for x, y in ((5.5, 5.5), (12.5, 14.5), (8.5, 8.5)):
            expected = placement_rules.move_refusal(scenario, unit, x, y) is not None
            assert placement_rules.move_refused(scenario, unit, x, y) == expected
    assert placement_rules.move_refused(scenario, walker, 5.5, 5.5)
    assert not placement_rules.move_refused(scenario, occupant, 5.5, 5.5)


def test_a_gates_own_corner_pieces_never_block_its_move() -> None:
    """TASK-340: real files keep a GTBA2's corner pieces (GTBC2) as separate
    solid units on its end tiles, so a move shorter than the gate refused on
    the piece left behind. Another owner's piece, or one off the gate's own
    tiles, still blocks; the grid path answers the same."""
    gate = _unit(GATE_RUN_Y, 10.5, 10)
    own = [_unit(GATE_RUN_Y_CORNER, 10.5, 8.5), _unit(GATE_RUN_Y_CORNER, 10.5, 11.5)]
    assert placement_rules.gate_companions(_scenario([[], [gate, *own]]), gate) == own
    for blockers, expected in (
        ([[], [gate, *own]], None),
        ([[_unit(GATE_RUN_Y_CORNER, 10.5, 11.5)], [gate, own[0]]], "blocked by"),
        ([[], [gate, own[0], _unit(GATE_RUN_Y_CORNER, 10.5, 12.5)]], "blocked by"),
    ):
        scenario = _scenario(blockers)
        grid = placement_rules.BlockerGrid(scenario, exclude=[gate])
        for kwargs in ({"exclude": [gate]}, {"grid": grid}):
            reason = placement_rules.move_refusal(scenario, gate, 10.5, 11, **kwargs)
            assert (reason is None) if expected is None else reason.startswith(expected), (blockers, reason)
            assert placement_rules.move_refused(scenario, gate, 10.5, 11, **kwargs) == (reason is not None)


def test_companions_by_gate_finds_each_moving_gates_pieces_per_owner() -> None:
    """TASK-531: both ends' pieces, stacked copies included, per gate across
    two owners. Another owner's piece, a piece off the gate's boxes and a
    non-corner const are not a gate's; a move with no A/B gate scans nothing."""
    gate1, gate2 = _unit(GATE_RUN_Y, 10.5, 10), _unit(GATE_RUN_Y, 5.5, 18)
    ends1 = [_unit(GATE_RUN_Y_CORNER, 10.5, 8.5), _unit(GATE_RUN_Y_CORNER, 10.5, 8.5), _unit(GATE_RUN_Y_CORNER, 10.5, 11.5)]
    ends2 = [_unit(GATE_RUN_Y_CORNER, 5.5, 16.5), _unit(GATE_RUN_Y_CORNER, 5.5, 19.5)]
    off_box, not_corner = _unit(GATE_RUN_Y_CORNER, 10.5, 12.5), _unit(VILLAGER, 10.5, 9.5)
    theirs = _unit(GATE_RUN_Y_CORNER, 10.5, 9.5)
    scenario = _scenario([[theirs], [gate1, *ends1, off_box, not_corner], [ends2[0], gate2, ends2[1]]])

    def ids(found):
        return {gate: [(player_id, id(p)) for player_id, p in rows] for gate, rows in found.items()}

    found = placement_rules.companions_by_gate(scenario, [(1, gate1), (2, gate2), (1, not_corner)])

    assert ids(found) == {id(gate1): [(1, id(p)) for p in ends1], id(gate2): [(2, id(p)) for p in ends2]}
    # The same match as TASK-340's per-gate rule.
    assert ids(placement_rules.companions_by_gate(scenario, [(1, gate1)])) == {
        id(gate1): [(1, id(p)) for p in placement_rules.gate_companions(scenario, gate1)]
    }
    assert placement_rules.companions_by_gate(_scenario([[], [gate1]]), [(1, gate1)]) == {}

    class _NoScan:
        @property
        def unit_manager(self):
            raise AssertionError("scanned with no gate moving")

    assert placement_rules.companions_by_gate(_NoScan(), [(1, not_corner), (1, ends1[0])]) == {}


def test_end_tiles_are_each_orientations_run_extremes() -> None:
    """TASK-532: the stone family at low corner (10, 10), low end first along A +x, C +x+y, B +y, D +x-y."""
    from descape import gate_orientation
    from descape.render_units import span_anchor
    from descape.terrain_palette import tile_span

    def at_corner(const):
        return placement_rules.end_tiles(const, *span_anchor(10, 10, *tile_span(const, (1, 1))))

    assert gate_orientation.orientation_siblings(64) == (64, 659, 88, 667)
    assert at_corner(64) == ((10, 10), (13, 10))
    assert at_corner(659) == ((10, 10), (13, 13))
    assert at_corner(88) == ((10, 10), (10, 13))
    assert at_corner(667) == ((10, 13), (13, 10))
    with pytest.raises(ValueError):
        placement_rules.end_tiles(VILLAGER, 10.5, 10.5)


def test_companion_landing_round_trips_and_keeps_each_pieces_offset() -> None:
    """TASK-532: four single steps and two quarter turns bring both ends' pieces,
    a stacked copy's sub-tile offset included, back to their first point and
    const, every step landing on an end tile of the gate's new footprint."""
    from descape import gate_orientation
    from descape.unit_model import reanchored

    gate = _unit(GATE_RUN_Y, 10.5, 10)
    pieces = [_unit(GATE_RUN_Y_CORNER, 10.5, 8.5), _unit(GATE_RUN_Y_CORNER, 10.25, 8.75), _unit(GATE_RUN_Y_CORNER, 10.5, 11.5)]
    first = [(p.unit_const, p.x, p.y) for p in pieces]
    for steps, count in ((1, 4), (2, 2), (-1, 4)):
        for _ in range(count):
            new_const = gate_orientation.cycle_const(gate.unit_const, steps)
            new_x, new_y = reanchored(gate, new_const)
            landed = [placement_rules.companion_landing(gate, new_const, new_x, new_y, p) for p in pieces]
            ends = placement_rules.end_tiles(new_const, new_x, new_y)
            for p, (const, x, y) in zip(pieces, landed, strict=True):
                assert const == gate_orientation.corner_piece(new_const)
                assert (int(x), int(y)) in ends
                p.unit_const, p.x, p.y = const, x, y
            gate.unit_const, gate.x, gate.y = new_const, new_x, new_y
            assert (pieces[0].x - int(pieces[0].x), pieces[1].x - int(pieces[1].x)) == (0.5, 0.25)
        assert [(p.unit_const, p.x, p.y) for p in pieces] == first, steps
    # B 88 to D 667, one step: the low end goes to the diagonal's (low x, high y) tip.
    assert placement_rules.companion_landing(gate, 667, *reanchored(gate, 667), pieces[0]) == (671, 10.5, 11.5)
    assert placement_rules.companion_landing(gate, 667, *reanchored(gate, 667), pieces[2]) == (671, 13.5, 8.5)


def test_companion_landing_takes_the_nearer_end_and_refuses_a_foreign_sibling() -> None:
    gate = _unit(GATE_RUN_Y, 10.5, 10)
    mid_low, mid_high = _unit(GATE_RUN_Y_CORNER, 10.5, 9.25), _unit(GATE_RUN_Y_CORNER, 10.5, 10.75)
    assert placement_rules.companion_landing(gate, 64, 10, 8.5, mid_low) == (81, 8.5, 8.25)
    assert placement_rules.companion_landing(gate, 64, 10, 8.5, mid_high) == (81, 11.5, 8.75)
    with pytest.raises(ValueError):
        placement_rules.companion_landing(gate, 64, 10, 8.5, _unit(VILLAGER, 10.5, 8.5))


def test_refused_builds_no_message(monkeypatch) -> None:
    """Pins "no message" without a timing: the two name lookups raise."""
    scenario, consts, points, _house = _refused_cases()
    expected = {
        (const, x, y): placement_rules.refusal(scenario, const, x, y) is not None
        for const in consts
        for x, y in points[::7]
    }

    def boom(*_args, **_kwargs):
        raise AssertionError("refused() built a message")

    monkeypatch.setattr(object_catalog, "display_name", boom)
    monkeypatch.setattr(object_catalog, "resolve_name", boom)
    for (const, x, y), answer in expected.items():
        assert placement_rules.refused(scenario, const, x, y) == answer
    assert any(expected.values()) and not all(expected.values())


# --- refusal(): both rules, as a message --------------------------------------------


def test_refusal_checks_terrain_first_and_then_names_the_blocker() -> None:
    house = _unit(HOUSE, 12, 14)
    archer = _unit(ARCHER, 5.5, 5.5)
    scenario = _scenario([[], [archer], [house]], terrain={(12, 14): WATER_DEEP, (5, 5): WATER_DEEP})
    # (5.5, 5.5) is both on water and inside the Archer, which lays no foundation.
    assert placement_rules.refusal(scenario, VILLAGER, 5.5, 5.5) == "Villager Male can't stand on Water Deep"
    # TASK-349: the House's own foundation covers the water under it, so only the House refuses there.
    assert placement_rules.refusal(scenario, VILLAGER, 12.5, 14.5) == "blocked by House (Player 2) at (12, 14)"
    assert placement_rules.refusal(scenario, VILLAGER, 11.5, 13.5) == "blocked by House (Player 2) at (12, 14)"
    assert placement_rules.refusal(scenario, VILLAGER, 11.5, 13.5, exclude=[house]) is None
    assert placement_rules.refusal(scenario, VILLAGER, 9.5, 13.5) is None


def test_refusal_names_a_gaia_blocker_and_an_off_grid_position() -> None:
    scenario = _scenario([[_unit(ARCHER, 5.25, 6.75)]])
    assert placement_rules.refusal(scenario, VILLAGER, 5.3, 6.7) == "blocked by Archer (GAIA) at (5.25, 6.75)"


def test_refusal_reads_a_grid_in_place_of_the_scan() -> None:
    house = _unit(HOUSE, 12, 14)
    scenario = _scenario([[], [house]])
    assert placement_rules.refusal(scenario, VILLAGER, 11.5, 13.5, grid=placement_rules.BlockerGrid(scenario)) == (
        "blocked by House (Player 1) at (12, 14)"
    )
    without = placement_rules.BlockerGrid(scenario, exclude=[house])
    assert placement_rules.refusal(scenario, VILLAGER, 11.5, 13.5, grid=without) is None


def test_a_garrisoned_mover_skips_both_rules() -> None:
    """With Show Garrisoned on an occupant can be dragged; it stands wherever
    its host does, on water and inside the host's own box."""
    transport = _unit(HOUSE, 12, 14)
    scenario = _scenario([[], [transport]], terrain={(5, 5): WATER_DEEP})
    walker, occupant = _unit(VILLAGER, 3.5, 3.5), _unit(VILLAGER, 12, 14, garrisoned_in_id=9)
    assert placement_rules.move_refusal(scenario, walker, 5.5, 5.5) is not None
    assert placement_rules.move_refusal(scenario, walker, 12.5, 14.5) is not None
    assert placement_rules.move_refusal(scenario, occupant, 5.5, 5.5) is None
    assert placement_rules.move_refusal(scenario, occupant, 12.5, 14.5) is None
    assert placement_rules.move_refusal(scenario, walker, 8.5, 8.5) is None


# --- re-measured against real files -----------------------------------------------

_FISH_CLASSES = frozenset({5, 31, 33})
_KIND_BY_TYPE = {80: "buildings", 70: "creatables", 10: "type-10"}
_SCOPES = ("subset", "full")
# Percent, (corpus subset, --corpus-full), pinned just above the 2026-10-12 measurement:
# buildings 9.97 / 6.19 (wall and gate joins, off-parity houses), creatables 0.18 / 0.42, type-10 0.77 / 1.14.
_OVERLAP_CEILING = {"buildings": (11.0, 7.0), "creatables": (0.5, 0.6), "type-10": (1.0, 1.4)}
# Measured: buildings 0.88 / 0.38, creatables 0.87 / 0.20, type-10 0.00 / 0.04, fish 0.00 / 1.63.
_TERRAIN_CEILING = {"buildings": 1.0, "creatables": 1.0, "type-10": 1.0, "fish": 3.0}
# Shore fish with a shore in their ring: 99.5% / 95.0% (967 of 1018).
_SHORE_FLOOR = 93.0


def _kind(entry: dict) -> str | None:
    if entry["class"] in _FISH_CLASSES:
        return "fish"
    return _KIND_BY_TYPE.get(entry["type"])


@pytest.mark.corpus
def test_real_placements_mostly_satisfy_both_rules(corpus_files, request) -> None:
    """Neither rule is the game's exact one, so each is re-measured rather
    than trusted: what authors really placed must overwhelmingly pass. The
    violators are author overrides (a House on Beach, fish on farm terrain)
    and the joins the editor allows (walls into gates, off-parity houses).

    A ceiling, never an equality: a new examples/ file may move a rate, and
    only a jump means a rule broke."""
    from collections import Counter

    from descape import render_units, scenario_io

    conftest.require_reference_corpus(request.config)
    scope = _SCOPES.index("full" if request.config.getoption("--corpus-full") else "subset")
    objects = object_catalog.dat_objects()
    placed, on_wrong_terrain = Counter(), Counter()
    solid, overlapping = Counter(), Counter()
    shore_fish = beside_shore = 0
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        mm = loaded.map_manager
        grid = placement_rules.BlockerGrid(loaded)
        for units in loaded.unit_manager.units:
            for unit in units:
                entry = objects.get(unit.unit_const)
                kind = None if entry is None else _kind(entry)
                tiles = render_units.occupied_tiles_for(unit.unit_const, unit.x, unit.y, mm.map_width, mm.map_height)
                if kind is None or tiles is None:
                    continue
                placed[kind] += 1
                on_wrong_terrain[kind] += placement_rules._footprint_refusal(loaded, unit.unit_const, tiles) is not None
                if unit.unit_const == SHORE_FISH:
                    shore_fish += 1
                    beside_shore += placement_rules._shore_refusal(loaded, unit.unit_const, tiles) is None
                if unit.garrisoned_in_id != -1 or not placement_rules.boxes(unit.unit_const, unit.x, unit.y):
                    continue
                solid[kind] += 1
                overlapping[kind] += grid.blocker(unit.unit_const, unit.x, unit.y, exclude=[unit]) is not None

    def percent(part: int, whole: int) -> float:
        return 100 * part / whole

    measured = {
        "terrain": {kind: round(percent(on_wrong_terrain[kind], placed[kind]), 2) for kind in _TERRAIN_CEILING},
        "overlap": {kind: round(percent(overlapping[kind], solid[kind]), 2) for kind in _OVERLAP_CEILING},
        "shore": round(percent(beside_shore, shore_fish), 1),
    }
    # Enough of each kind that a rate means something (the corpus subset holds 1,147 creatables).
    assert all(placed[kind] >= 1000 for kind in _TERRAIN_CEILING), dict(placed)
    assert shore_fish >= 500
    for kind, ceiling in _TERRAIN_CEILING.items():
        assert measured["terrain"][kind] < ceiling, measured
    for kind, ceilings in _OVERLAP_CEILING.items():
        assert measured["overlap"][kind] < ceilings[scope], measured
    assert measured["shore"] >= _SHORE_FLOOR, measured


# --- TerrainGaps (TASK-328: wall runs follow the terrain rule) --------------------

SEA_WALL = 788  # row 21: refuses land
MAP_REVEALER = 0  # no restriction row


def _terrain_row(ids) -> FakeScenario:
    tiles = [SyntheticTile(x, 0, 0, terrain_id) for x, terrain_id in enumerate(ids)]
    return FakeScenario(len(tiles), 1, tiles, [[]])


def test_terrain_gaps_agree_with_terrain_refusal_for_every_wall_const_on_every_terrain() -> None:
    from descape import unit_sprites

    ids = sorted(object_catalog.dat_terrains())
    scenario = _terrain_row(ids)
    walls = unit_sprites.wall_family_consts()
    assert len(walls) >= 8, walls
    for const in walls:
        gaps = placement_rules.TerrainGaps(scenario, const)
        for tx, terrain_id in enumerate(ids):
            refused = placement_rules.terrain_refusal(scenario, const, tx + 0.5, 0.5) is not None
            assert ((tx, 0) in gaps) == refused, (const, terrain_id)


def test_terrain_gaps_hold_water_for_a_stone_wall_and_land_for_a_sea_wall() -> None:
    scenario = _terrain_row([GRASS, WATER_SHALLOW])
    stone = placement_rules.TerrainGaps(scenario, STONE_WALL)
    sea = placement_rules.TerrainGaps(scenario, SEA_WALL)
    assert (1, 0) in stone and (0, 0) not in stone
    assert (0, 0) in sea and (1, 0) not in sea
    assert stone and sea


def test_an_off_map_tile_is_never_a_terrain_gap() -> None:
    gaps = placement_rules.TerrainGaps(_terrain_row([WATER_DEEP, WATER_DEEP]), STONE_WALL)
    assert (0, 0) in gaps
    for tile in [(-1, 0), (2, 0), (0, -1), (0, 1)]:
        assert tile not in gaps, tile


def test_a_const_with_no_restriction_row_has_no_terrain_gaps() -> None:
    scenario = _terrain_row([WATER_DEEP, GRASS])
    for const in (MAP_REVEALER, 999_999):
        gaps = placement_rules.TerrainGaps(scenario, const)
        assert not gaps
        assert (0, 0) not in gaps and (1, 0) not in gaps


# Percent of wall-family placements on terrain their own row refuses: measured 0.10% (8 of 8,369), 2026-10-29.
_WALL_TERRAIN_CEILING = 0.5


@pytest.mark.corpus
def test_real_wall_placements_mostly_stand_where_terrain_gaps_allow(corpus_files, request) -> None:
    """A wall run gaps where TerrainGaps says (TASK-328), so what authors
    really placed must overwhelmingly pass it. A ceiling, never an equality."""
    from descape import scenario_io, unit_sprites

    conftest.require_reference_corpus(request.config)
    walls = frozenset(unit_sprites.wall_family_consts())
    placed = refused = 0
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        gaps = {const: placement_rules.TerrainGaps(loaded, const) for const in walls}
        for units in loaded.unit_manager.units:
            for unit in units:
                if unit.unit_const not in walls:
                    continue
                placed += 1
                refused += (int(unit.x), int(unit.y)) in gaps[unit.unit_const]
    assert placed >= 1000, placed
    assert 100 * refused / placed < _WALL_TERRAIN_CEILING, (refused, placed)
