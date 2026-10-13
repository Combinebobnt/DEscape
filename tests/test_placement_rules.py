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

from descape import asset_source, object_catalog, placement_rules
from descape.terrain_palette import BUILDING_TILE_SPANS
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

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


# --- refusal(): both rules, as a message --------------------------------------------


def test_refusal_checks_terrain_first_and_then_names_the_blocker() -> None:
    house = _unit(HOUSE, 12, 14)
    scenario = _scenario([[], [], [house]], terrain={(12, 14): WATER_DEEP})
    # (12.5, 14.5) is both on water and inside the House.
    assert placement_rules.refusal(scenario, VILLAGER, 12.5, 14.5) == "Villager Male can't stand on Water Deep"
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
_SCOPES = ("quick", "full")
# Percent, (quick subset, --corpus-full), pinned just above the 2026-10-12 measurement:
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

    from descape import render, scenario_io

    scope = _SCOPES.index("full" if request.config.getoption("--corpus-full") else "quick")
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
                tiles = render.occupied_tiles_for(unit.unit_const, unit.x, unit.y, mm.map_width, mm.map_height)
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
    # Enough of each kind that a rate means something (the quick subset holds 1,147 creatables).
    assert all(placed[kind] >= 1000 for kind in _TERRAIN_CEILING), dict(placed)
    assert shore_fish >= 500
    for kind, ceiling in _TERRAIN_CEILING.items():
        assert measured["terrain"][kind] < ceiling, measured
    for kind, ceilings in _OVERLAP_CEILING.items():
        assert measured["overlap"][kind] < ceilings[scope], measured
    assert measured["shore"] >= _SHORE_FLOOR, measured
