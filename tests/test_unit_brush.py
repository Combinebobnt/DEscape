"""descape/unit_brush.py (GH #170): the Unit Brush's planner, seeded, on a
synthetic map with a plain list-appending add. Qt-free, install-free."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from descape import asset_source, brush, placement_rules, unit_kind, unit_rotation, unit_sprites, unit_variant
from descape.place_rotation import PlaceRotation, rotation_for
from descape.render import occupied_tiles_for
from descape.unit_brush import BrushStroke, Spec, hover_anchor, is_refused_const, paint_dab
from testkit.fakes import FakeScenario, SyntheticTile

OAK = 349  # cyclable, 1x1, a full-tile box
PINE = 350
DEER = 65  # GAIA-only, ANGLE
VILLAGER = 83  # mobile, ANGLE
HOUSE = 70  # 2x2, INERT
FARM = 50  # 3x3, passable
BARRACKS = 12  # 3x3
FORTRESS = 33  # 4x4
MOUNTAIN = 744  # 6x6 footprint, 4x4 clearance
STONE_WALL = 117
STONE_GATE = 64
BRIDGE_MIDDLE = 606  # lays foundation 28 over water
WIDE = 999_001  # synthetic: a 1x1 const with a 1.6-wide box
GRASS, WATER_SHALLOW, WATER_DEEP = 0, 1, 22
SIZE = 30
P1, GAIA = 1, 0


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """Messages are asserted by name, and AOE2DE_INSTALL_PATH would swap in the install's."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    garrisoned_in_id: int = -1


def _scenario(terrain=None, units_by_player=None, size=SIZE) -> FakeScenario:
    terrain = terrain or {}
    tiles = [SyntheticTile(x, y, 0, terrain.get((x, y), GRASS)) for y in range(size) for x in range(size)]
    units = units_by_player or [[] for _ in range(9)]
    return FakeScenario(size, size, tiles, units)


def _adder(scenario):
    def add(spec: Spec):
        unit = _Unit(spec.x, spec.y, spec.unit_const, spec.rotation)
        scenario.unit_manager.units[spec.player].append(unit)
        return unit

    return add


def _stroke(consts, owner=P1, density=1.0, rules=True, randomize=True, seed=7, jitter=None, pending=None):
    if isinstance(consts, int):
        consts = (consts,)
    return BrushStroke(
        consts=tuple(consts), chosen_owner=owner, density=density, rules=rules, randomize=randomize,
        pending=pending or PlaceRotation(), rng=random.Random(seed), jitter=jitter,
    )


def _square(x0: int, y0: int, size: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(y0, y0 + size) for x in range(x0, x0 + size)]


def _footprints(scenario, placed) -> list[list[tuple[int, int]]]:
    mm = scenario.map_manager
    return [occupied_tiles_for(u.unit_const, u.x, u.y, mm.map_width, mm.map_height) for _p, u in placed]


def _no_shared_tile(scenario, placed) -> bool:
    seen = set()
    for tiles in _footprints(scenario, placed):
        if seen & set(tiles):
            return False
        seen |= set(tiles)
    return True


# --- density and re-crossing ---------------------------------------------------------


def test_density_100_fills_every_passable_tile_once() -> None:
    scenario = _scenario()
    stroke = _stroke(OAK)
    placed = paint_dab(scenario, stroke, _square(5, 5, 5), _adder(scenario))
    assert sorted((int(u.x), int(u.y)) for _p, u in placed) == sorted(_square(5, 5, 5))
    assert all((u.x % 1, u.y % 1) == (0.5, 0.5) for _p, u in placed)


def test_density_50_fills_about_half_and_re_crossing_adds_nothing() -> None:
    scenario = _scenario()
    stroke = _stroke(OAK, density=0.5)
    add = _adder(scenario)
    tiles = _square(2, 2, 20)
    placed = paint_dab(scenario, stroke, tiles, add)
    assert 150 < len(placed) < 250, len(placed)
    assert paint_dab(scenario, stroke, tiles, add) == []
    assert paint_dab(scenario, stroke, _square(2, 2, 10), add) == []
    assert sum(len(units) for units in scenario.unit_manager.units) == len(placed)


# --- the stroke never stacks on itself --------------------------------------------------


@pytest.mark.parametrize("rules", [True, False])
def test_a_single_size_5_dab_of_houses_never_overlaps_itself(rules) -> None:
    scenario = _scenario()
    stroke = _stroke(HOUSE, rules=rules)
    placed = paint_dab(scenario, stroke, brush.brush_tiles(12, 12, 5, "square", SIZE, SIZE), _adder(scenario))
    assert len(placed) >= 4
    assert _no_shared_tile(scenario, placed)


@pytest.mark.parametrize("rules", [True, False])
def test_a_multi_dab_house_stroke_never_overlaps_itself(rules) -> None:
    scenario = _scenario()
    stroke = _stroke(HOUSE, rules=rules)
    add = _adder(scenario)
    for cx in range(6, 22):
        paint_dab(scenario, stroke, brush.brush_tiles(cx, 12, 3, "circle", SIZE, SIZE), add)
    assert len(stroke.placed) >= 8
    assert _no_shared_tile(scenario, stroke.placed)


def test_shapeless_consts_never_stack_within_a_stroke() -> None:
    """Fish and birds have no box, so only the stroke's own tiles keep them apart."""
    shore_fish = 69
    assert not placement_rules.boxes(shore_fish, 5.5, 5.5)
    scenario = _scenario(terrain=dict.fromkeys(_square(0, 0, SIZE), WATER_SHALLOW))
    stroke = _stroke(shore_fish, rules=False, jitter=0.4)
    add = _adder(scenario)
    for _ in range(3):
        paint_dab(scenario, stroke, _square(4, 4, 6), add)
    assert len(stroke.placed) == 36
    assert len({(int(u.x), int(u.y)) for _p, u in stroke.placed}) == 36


# --- placement rules ---------------------------------------------------------------------


def test_rules_on_skip_water_for_a_land_unit_and_tiles_under_an_obstruction() -> None:
    water = dict.fromkeys(_square(5, 5, 3), WATER_DEEP)
    blocker = _Unit(12, 6, HOUSE)  # tiles 11..12 x 5..6
    units = [[] for _ in range(9)]
    units[2].append(blocker)
    scenario = _scenario(terrain=water, units_by_player=units)
    stroke = _stroke(VILLAGER)
    placed = paint_dab(scenario, stroke, _square(4, 4, 10), _adder(scenario))
    tiles = {(int(u.x), int(u.y)) for _p, u in placed}
    assert not tiles & set(water)
    assert not tiles & {(11, 5), (12, 5), (11, 6), (12, 6)}
    assert len(tiles) == 100 - 9 - 4
    assert stroke.refused == 13
    assert stroke.refusal_reason == "Villager Male can't stand on Water Deep"


def test_the_refusal_reason_is_the_one_at_refusal_time() -> None:
    """A bridge the stroke lays later over the refused water tile would read
    as allowed at release, leaving "N refused" with no reason."""
    water = dict.fromkeys(_square(8, 8, 8), WATER_SHALLOW)
    scenario = _scenario(terrain=water)
    stroke = _stroke(VILLAGER)
    paint_dab(scenario, stroke, [(10, 10)], _adder(scenario))
    assert stroke.refused == 1
    scenario.unit_manager.units[0].append(_Unit(11, 10.5, BRIDGE_MIDDLE))
    assert placement_rules.refusal(scenario, VILLAGER, 10.5, 10.5) is None, "vacuous: the bridge does not cover it"
    assert stroke.refusal_reason == "Villager Male can't stand on Water Shallow"


def test_rules_off_places_them() -> None:
    water = dict.fromkeys(_square(5, 5, 3), WATER_DEEP)
    units = [[] for _ in range(9)]
    units[2].append(_Unit(12, 6, HOUSE))
    scenario = _scenario(terrain=water, units_by_player=units)
    stroke = _stroke(VILLAGER, rules=False)
    placed = paint_dab(scenario, stroke, _square(4, 4, 10), _adder(scenario))
    assert len(placed) == 100
    assert stroke.refused == 0 and stroke.refusal_reason is None and stroke.grid is None


def test_a_wide_box_never_overlaps_another_from_the_same_dab(monkeypatch) -> None:
    """The grid learns each placed unit before the next candidate: a box
    wider than its tile reaches the neighbours, which the tile check alone
    would let through."""
    objects = dict(placement_rules._objects())
    objects[WIDE] = {"class": 0, "clearance": [0.8, 0.8], "obstruction": 2}
    monkeypatch.setattr(placement_rules, "_objects", lambda: objects)
    scenario = _scenario()
    stroke = _stroke(WIDE)
    placed = paint_dab(scenario, stroke, _square(5, 5, 5), _adder(scenario))
    assert 4 <= len(placed) < 25
    for i, (_p, a) in enumerate(placed):
        for _q, b in placed[i + 1:]:
            assert not placement_rules._overlap(
                placement_rules.boxes(WIDE, a.x, a.y), placement_rules.boxes(WIDE, b.x, b.y)
            ), (a, b)
    off = _scenario()
    assert len(paint_dab(off, _stroke(WIDE, rules=False), _square(5, 5, 5), _adder(off))) == 25


def test_a_mobile_stroke_reads_a_bridge_and_builds_the_laid_map_once(monkeypatch) -> None:
    """A villager crosses onto a bridge's deck over water, not onto open
    water; the stroke holds one laid_terrain() for every candidate, where
    the per-edit memo would rebuild after each add (unit_gen)."""
    water = dict.fromkeys(_square(8, 8, 8), WATER_SHALLOW)
    units = [[] for _ in range(9)]
    units[0].append(_Unit(11, 10.5, BRIDGE_MIDDLE))  # tiles 10..11 x 9..11
    scenario = _scenario(terrain=water, units_by_player=units)
    calls = []
    real = placement_rules.laid_terrain
    monkeypatch.setattr(placement_rules, "laid_terrain", lambda s: calls.append(1) or real(s))
    stroke = _stroke(VILLAGER)
    add = _adder(scenario)
    placed = paint_dab(scenario, stroke, _square(9, 8, 4), add)
    paint_dab(scenario, stroke, _square(12, 12, 3), add)
    on_water = {(int(u.x), int(u.y)) for _p, u in placed} & set(water)
    assert on_water == {(10, 9), (11, 9), (10, 10), (11, 10), (10, 11), (11, 11)}
    assert stroke.refused > 6
    assert len(calls) == 1
    assert _stroke(OAK).laid is None and placement_rules.refused(scenario, VILLAGER, 13.5, 13.5)


# --- map edge, anchors, jitter ---------------------------------------------------------------


def test_a_footprint_crossing_the_map_edge_is_skipped() -> None:
    scenario = _scenario()
    stroke = _stroke(FARM, rules=False)
    placed = paint_dab(scenario, stroke, [(0, 10), (SIZE - 1, 10), (10, 10)], _adder(scenario))
    assert [(u.x, u.y) for _p, u in placed] == [(10.5, 10.5)]


def test_tile_span_anchors_a_mountain_on_the_game_grid() -> None:
    scenario = _scenario()
    stroke = _stroke(MOUNTAIN, rules=False)
    (_p, unit), = paint_dab(scenario, stroke, [(14, 14)], _adder(scenario))
    assert (unit.x, unit.y) == (15.0, 15.0)
    tiles = occupied_tiles_for(MOUNTAIN, unit.x, unit.y, SIZE, SIZE)
    assert len(tiles) == 36 and (14, 14) in tiles


def test_jitter_keeps_each_unit_on_its_rolled_tile() -> None:
    scenario = _scenario()
    stroke = _stroke(OAK, rules=False, jitter=0.9)
    tiles = _square(3, 3, 12)
    placed = paint_dab(scenario, stroke, tiles, _adder(scenario))
    assert sorted((int(u.x), int(u.y)) for _p, u in placed) == sorted(tiles)
    assert len({(u.x % 1, u.y % 1) for _p, u in placed}) > 100


def test_jitter_leaves_a_houses_two_wide_axes_on_the_grid() -> None:
    scenario = _scenario()
    stroke = _stroke(HOUSE, rules=False, jitter=0.4)
    placed = paint_dab(scenario, stroke, _square(6, 6, 6), _adder(scenario))
    assert placed and all(u.x == int(u.x) and u.y == int(u.y) for _p, u in placed)


def test_hover_anchor_is_where_a_dab_stands_the_unit_and_none_exactly_where_it_places_nothing() -> None:
    """The oracle behind the hover ghost: every case runs a real dab, with the
    rules and density out of the way so the map edge is the only skip left."""
    size = 120
    scenario = _scenario(size=size)
    edges = (0, 1, 2, size - 3, size - 2, size - 1)
    tiles = [(40, 40)] + [(x, y) for x in edges for y in edges]
    wrong = []
    outcomes = {}
    for const in (OAK, HOUSE, BARRACKS, FORTRESS):
        for tile in tiles:
            stroke = _stroke(const, rules=False, density=1.0, jitter=0.0)
            placed = paint_dab(scenario, stroke, [tile], lambda spec: _Unit(spec.x, spec.y, spec.unit_const))
            painted = (placed[0][1].x, placed[0][1].y) if placed else None
            anchor = hover_anchor(const, tile, size, size)
            outcomes[const, tile] = anchor
            if anchor != painted:
                wrong.append((const, tile, anchor, painted))
    assert not wrong, wrong
    assert outcomes[BARRACKS, (0, 0)] is None and outcomes[BARRACKS, (size - 1, size - 1)] is None
    assert outcomes[HOUSE, (0, 0)] is not None and outcomes[FORTRESS, (size - 1, size - 1)] is not None


# --- objects, owners, rotation ------------------------------------------------------------


def test_a_random_pick_yields_both_consts() -> None:
    scenario = _scenario()
    stroke = _stroke((OAK, PINE))
    placed = paint_dab(scenario, stroke, _square(5, 5, 6), _adder(scenario))
    assert {u.unit_const for _p, u in placed} == {OAK, PINE}


def test_a_gaia_only_const_goes_to_gaia() -> None:
    assert unit_kind.placement_owner(DEER, P1) == GAIA
    scenario = _scenario()
    stroke = _stroke((DEER, VILLAGER), owner=P1)
    assert stroke.owners() == [GAIA, P1]
    placed = paint_dab(scenario, stroke, _square(5, 5, 6), _adder(scenario))
    assert {p for p, u in placed if u.unit_const == DEER} == {GAIA}
    assert {p for p, u in placed if u.unit_const == VILLAGER} == {P1}
    assert stroke.gaia_only == sum(1 for _p, u in placed if u.unit_const == DEER) > 0


def test_random_rotations_vary_and_stay_on_rotation_for_values() -> None:
    scenario = _scenario()
    stroke = _stroke((OAK, VILLAGER))
    placed = paint_dab(scenario, stroke, _square(3, 3, 10), _adder(scenario))
    oak = [u.rotation for _p, u in placed if u.unit_const == OAK]
    villager = [u.rotation for _p, u in placed if u.unit_const == VILLAGER]
    assert len(set(oak)) > 5 and len(set(villager)) > 5
    count = unit_variant.variant_count_for(OAK)
    assert all(r == int(r) and 0 <= r < count for r in oak)
    angles = unit_rotation.angle_count_for(VILLAGER)
    assert all(
        r == pytest.approx(unit_rotation.facing_to_rotation(unit_rotation.rotation_to_facing(r, angles), angles))
        for r in villager
    )


def test_unticked_random_gives_every_unit_the_pending_rotation() -> None:
    scenario = _scenario()
    pending = PlaceRotation(facing=1.2, variant=5, variant_const=OAK)
    stroke = _stroke((OAK, VILLAGER), randomize=False, pending=pending)
    placed = paint_dab(scenario, stroke, _square(3, 3, 6), _adder(scenario))
    for player, unit in placed:
        assert unit.rotation == rotation_for(pending, unit.unit_const, player)
    assert {u.rotation for _p, u in placed if u.unit_const == OAK} == {5.0}


def test_a_seed_reproduces_the_stroke() -> None:
    runs = []
    for _ in range(2):
        scenario = _scenario()
        stroke = _stroke((OAK, PINE), density=0.4, jitter=0.3, seed=99)
        placed = paint_dab(scenario, stroke, _square(2, 2, 15), _adder(scenario))
        runs.append([(p, u.unit_const, u.x, u.y, u.rotation) for p, u in placed])
    assert runs[0] == runs[1]


# --- refused consts ----------------------------------------------------------------------


def test_walls_gates_and_cliffs_refuse_and_name_their_tool() -> None:
    cliff = min(unit_kind.cliff_consts())
    assert "Wall" in is_refused_const(STONE_WALL)
    assert all(is_refused_const(c) for c in unit_sprites.wall_family_consts())
    assert "Place Unit" in is_refused_const(STONE_GATE)
    assert "Cliff tool" in is_refused_const(cliff)
    for const in (OAK, DEER, VILLAGER, HOUSE, FARM, MOUNTAIN):
        assert is_refused_const(const) is None
