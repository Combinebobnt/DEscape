"""descape/wall_run.py and UnitEditModel.set_wall_variant() -- the Wall Run
tool's planner and its write path (2026-09-19 wall-runs plan), headless.

Synthetic units in tests/test_wall_connectivity.py's style, not corpus ones:
examples/ is gitignored, so a default-tier test cannot depend on it. The
planner itself is pure, so most of what's here needs nothing but a tile set;
the two model tests use the real UnitEditModel against the same units_120x120
fixture the viewer tests load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from descape import unit_sprites, wall_run

# Stone Wall. A real const, so rotation_variant_eligible() and
# wall_connector_consts() both accept it without monkeypatching -- the
# planner reads those, and pinning the real answer is the point.
WALL = 117
# A real stone gate const: a connector, but NOT rotation_variant_eligible,
# so it should count as a neighbour and never be rewritten.
GATE = 64
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
# A Stone Wall in that fixture, storing index 2 in the RADIAN convention.
_REF_WALL = 102


@dataclass
class FakeUnit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


def _wall_at(tx: int, ty: int, rotation: float = 0.0, const: int = WALL) -> FakeUnit:
    return FakeUnit(x=tx + 0.5, y=ty + 0.5, unit_const=const, rotation=rotation)


def _existing(units) -> tuple[set[tuple[int, int]], list[wall_run.ExistingWall]]:
    """wall_scene()'s output shape, built by hand: every unit is 1x1 here,
    so its occupied tile set is its own anchor tile."""
    tiles = {(int(u.x), int(u.y)) for u in units}
    walls = [
        wall_run.ExistingWall(unit=u, player_id=1, tx=int(u.x), ty=int(u.y))
        for u in units
        if unit_sprites.rotation_variant_eligible(u.unit_const)
    ]
    return tiles, walls


def _plan(tiles, existing=()):
    existing_tiles, existing_walls = _existing(existing)
    return wall_run.plan_wall_run(
        tiles, unit_const=WALL, existing_tiles=existing_tiles, existing_walls=existing_walls
    )


# --- the path's own shapes --------------------------------------------


def test_a_straight_run_resolves_to_the_run_index_throughout():
    """0 along x for the interior, 2 at the two ends (one neighbour each) --
    exactly the 4-bit table, since a straight run is orthogonal everywhere."""
    plan = _plan([(x, 10) for x in range(5, 11)])
    assert [n.variant for n in plan.nodes] == [2, 0, 0, 0, 0, 2]
    assert [(n.x, n.y) for n in plan.nodes] == [(x + 0.5, 10.5) for x in range(5, 11)]


def test_a_run_along_y_resolves_to_index_1():
    plan = _plan([(10, y) for y in range(5, 11)])
    assert [n.variant for n in plan.nodes][1:-1] == [1, 1, 1, 1]


def test_a_bent_runs_corner_resolves_to_2():
    """An L: down the y axis, then along x. The corner has one NORTH and one
    EAST neighbour, which is the 4-bit table's tower case."""
    tiles = [(10, y) for y in range(5, 11)] + [(x, 10) for x in range(11, 15)]
    plan = _plan(tiles)
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert by_tile[(10, 10)] == 2
    assert by_tile[(10, 7)] == 1
    assert by_tile[(13, 10)] == 0


def test_a_diagonal_run_resolves_by_its_sense():
    """Indices 3 and 4, which no code in this tree could derive before. A
    NE-SW run (x and y moving opposite ways) is 4; NW-SE is 3."""
    ne_sw = _plan([(10 + i, 20 - i) for i in range(6)])
    assert [n.variant for n in ne_sw.nodes] == [2, 4, 4, 4, 4, 2]
    nw_se = _plan([(10 + i, 20 + i) for i in range(6)])
    assert [n.variant for n in nw_se.nodes] == [2, 3, 3, 3, 3, 2]


def test_a_single_click_places_one_wall_at_the_isolated_fallback():
    """2, not add()'s own rotation default of 0: 24 of 24 isolated corpus
    walls store 2, and the game re-derives either value anyway."""
    plan = _plan([(7, 7)])
    assert [n.variant for n in plan.nodes] == [wall_run.ISOLATED_VARIANT] == [2]


def test_every_node_writes_the_literal_index_and_frame_zero():
    plan = _plan([(x, 10) for x in range(5, 9)])
    assert [n.rotation for n in plan.nodes] == [float(n.variant) for n in plan.nodes]
    assert all(n.initial_animation_frame == 0 for n in plan.nodes)
    assert all(n.unit_const == WALL for n in plan.nodes)


# --- existing walls ----------------------------------------------------


def test_a_tile_already_holding_a_wall_is_skipped_and_still_a_neighbour():
    """Skipped, not stacked -- and the skipped tile's occupant still shapes
    what lands beside it, which is what makes drawing along an existing wall
    reshape rather than double it."""
    existing = [_wall_at(7, 10)]
    plan = _plan([(x, 10) for x in range(5, 11)], existing)
    assert plan.skipped == 1
    assert [(int(n.x), int(n.y)) for n in plan.nodes] == [
        (5, 10), (6, 10), (8, 10), (9, 10), (10, 10)
    ]
    # (6, 10) still reads (7, 10) as its EAST neighbour, so it is a run
    # piece rather than a run end.
    assert {(int(n.x), int(n.y)): n.variant for n in plan.nodes}[(6, 10)] == 0


def test_a_gate_counts_as_a_neighbour_and_is_never_rewritten():
    existing = [_wall_at(7, 10, const=GATE)]
    plan = _plan([(x, 10) for x in range(5, 11)], existing)
    assert plan.skipped == 1
    assert plan.rewrites == []
    assert {(int(n.x), int(n.y)): n.variant for n in plan.nodes}[(6, 10)] == 0


def test_a_run_meeting_an_existing_wall_rewrites_exactly_that_junction():
    """A T: an existing y-axis run that a new x-axis run meets at (10, 10).
    Only the junction's own shape changed, so only it is rewritten."""
    existing = [_wall_at(10, y, rotation=1.0) for y in range(8, 13)]
    plan = _plan([(x, 10) for x in range(5, 10)], existing)
    assert len(plan.rewrites) == 1
    player_id, unit, variant = plan.rewrites[0]
    assert (int(unit.x), int(unit.y)) == (10, 10)
    assert player_id == 1
    assert variant == 2  # a junction, not a run


def test_an_unchanged_neighbour_is_not_rewritten_to_the_value_it_already_has():
    """Otherwise its owner is pulled into the touched-player set and
    snapshotted for nothing, and the undo record misreports the edit.

    Extends the same y run rather than crossing it. (10, 12) was the run's
    end and is now its interior, so its derived index becomes the 1 it
    already stores -- a real change in mask, and still no rewrite."""
    existing = [_wall_at(10, y, rotation=1.0) for y in range(8, 13)]
    plan = _plan([(10, y) for y in range(13, 16)], existing)
    assert plan.rewrites == []


def test_a_rewrite_compares_against_the_decoded_index_not_the_raw_float():
    """The same no-op extension, but in a radian-encoded file, where index 1
    is stored as 1.256637. A raw-float compare would see 1.256637 != 1 and
    rewrite every one of these walls to a shape it already had."""
    radian_1 = 1 * 2 * 3.141592653589793 / 5
    existing = [_wall_at(10, y, rotation=radian_1) for y in range(8, 13)]
    plan = _plan([(10, y) for y in range(13, 16)], existing)
    assert plan.rewrites == []


def test_a_distant_wall_is_left_alone():
    existing = [_wall_at(30, 30, rotation=4.0)]
    plan = _plan([(x, 10) for x in range(5, 11)], existing)
    assert plan.rewrites == []


def test_touched_players_is_the_owner_plus_every_rewritten_walls_owner():
    existing = [_wall_at(10, y, rotation=1.0) for y in range(8, 13)]
    plan = _plan([(x, 10) for x in range(5, 10)], existing)
    assert wall_run.touched_players(3, plan) == [1, 3]
    assert wall_run.touched_players(1, plan) == [1]


# --- Wall Rectangle's ring (2026-09-21 wall enclosure plan) -------------


def _ring(x0, y0, x1, y1, size=60):
    from descape import shape_tools

    return shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, size, size)


def test_a_ring_has_towers_at_corners_and_runs_along_its_edges():
    plan = _plan(_ring(5, 10, 13, 14))
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert len(by_tile) == 2 * 9 + 2 * 3
    for corner in ((5, 10), (13, 10), (5, 14), (13, 14)):
        assert by_tile[corner] == 2
    for x in range(6, 13):
        assert by_tile[(x, 10)] == by_tile[(x, 14)] == 0
    for y in range(11, 14):
        assert by_tile[(5, y)] == by_tile[(13, y)] == 1


def test_a_one_wide_ring_is_a_straight_run():
    from descape import shape_tools

    ring = _plan(_ring(5, 10, 13, 10))
    run = _plan(shape_tools.wall_path_tiles(5, 10, 13, 10, 60, 60))
    as_map = lambda plan: {(n.x, n.y): n.variant for n in plan.nodes}  # noqa: E731
    assert as_map(ring) == as_map(run)


def test_a_one_by_one_ring_is_one_isolated_piece():
    plan = _plan(_ring(7, 7, 7, 7))
    assert [n.variant for n in plan.nodes] == [wall_run.ISOLATED_VARIANT]


def test_a_ring_sharing_an_edge_with_existing_walls_skips_and_rewrites_them():
    """An existing 3-piece y run lying on the ring's left side: its three
    tiles are skipped, and its two ends become run interior (2 -> 1), a
    reshape that rides in the plan. Its middle piece already stores 1."""
    existing = [_wall_at(5, 11, 2.0), _wall_at(5, 12, 1.0), _wall_at(5, 13, 2.0)]
    plan = _plan(_ring(5, 10, 13, 14), existing)
    assert plan.skipped == 3
    assert len(plan.nodes) == 24 - 3
    assert [(unit, variant) for _p, unit, variant in plan.rewrites] == [
        (existing[0], 1),
        (existing[2], 1),
    ]
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert by_tile[(5, 10)] == by_tile[(5, 14)] == 2


def test_a_ring_clipped_at_the_map_edge_is_an_open_three_sided_run():
    plan = _plan(_ring(-3, 10, 5, 14))
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert min(x for x, _y in by_tile) == 0
    # The open ends at x=0 have one neighbour each, so they are run ends.
    assert by_tile[(0, 10)] == by_tile[(0, 14)] == 2
    assert by_tile[(2, 10)] == 0
    assert by_tile[(5, 12)] == 1
    assert (0, 12) not in by_tile


def test_a_path_wholly_on_existing_walls_plans_nothing_new():
    existing = [_wall_at(x, 10) for x in range(5, 11)]
    plan = _plan([(x, 10) for x in range(5, 11)], existing)
    assert plan.nodes == []
    assert plan.skipped == 6


# --- set_wall_variant()'s guard ---------------------------------------
#
# Against the real UnitEditModel and the same units_120x120 fixture
# test_unit_model.py uses. Reference 102 is a Stone Wall storing the RADIAN
# encoding of index 2 (2.513274), which is what makes the round-trip below
# assert something: the write must land the literal 4.0, not another radian.


def _open():
    from descape.scenario_io import load_map_and_units
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded)


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _add(model, const: int):
    from descape.edit_history import EditHistory

    model.begin_unit_edit([1])
    unit = model.add(1, const, 20.5, 20.5)
    model.commit_unit_edit("probe", EditHistory())
    return unit


def test_set_wall_variant_writes_the_literal_index():
    from descape.edit_history import EditHistory

    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    assert wall.rotation != 4.0

    model.begin_unit_edit([1], fields_only=True)
    model.set_wall_variant(wall, 4)
    model.commit_unit_edit("Place wall run", EditHistory())

    assert wall.rotation == 4.0
    assert unit_sprites.is_literal_variant_index(wall.rotation, 5)
    # Untouched, per the measurement that all 8193 corpus walls store 0.
    assert wall.initial_animation_frame == 0


def test_set_wall_variant_undoes_back_to_the_stored_radian():
    from descape.edit_history import EditHistory

    loaded, model = _open()
    history = EditHistory()
    wall = _unit(loaded, _REF_WALL)
    before = wall.rotation

    model.begin_unit_edit([1], fields_only=True)
    model.set_wall_variant(wall, 3)
    model.commit_unit_edit("Place wall run", history)

    history.undo([], None, None, model)
    assert _unit(loaded, _REF_WALL).rotation == before


def test_set_wall_variant_refuses_a_gate():
    _loaded, model = _open()
    gate = _add(model, GATE)
    with pytest.raises(ValueError, match="not one of the wall consts"):
        model.set_wall_variant(gate, 2)


def test_set_wall_variant_refuses_a_non_wall_const():
    loaded, model = _open()
    archer = _unit(loaded, 201)
    with pytest.raises(ValueError, match="not one of the wall consts"):
        model.set_wall_variant(archer, 2)


@pytest.mark.parametrize("index", [-1, 5, 99])
def test_set_wall_variant_refuses_an_out_of_range_index(index):
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    with pytest.raises(ValueError, match=r"must be 0\.\.4"):
        model.set_wall_variant(wall, index)


def test_apply_wall_plan_adds_and_rewrites_in_one_pass():
    """The write half end to end, through the real model: new pieces via
    add_many(), the junction via set_wall_variant()."""
    from descape.edit_history import EditHistory
    from descape.render import unit_tile_bounds

    loaded, model = _open()
    mm = loaded.map_manager
    wall = _unit(loaded, _REF_WALL)
    bounds = unit_tile_bounds(wall, mm.map_width, mm.map_height)
    wx, wy = bounds[0], bounds[2]

    existing_tiles, existing_walls = wall_run.wall_scene(loaded, mm.map_width, mm.map_height)
    assert (wx, wy) in existing_tiles
    # Both sides of the existing wall, so its mask goes from 0 (isolated,
    # reading its stored 2) to WEST|EAST -- a real shape change, and so a
    # real rewrite. Flanking it on one side alone would derive 2 again and
    # correctly plan nothing.
    tiles = [(wx - 2, wy), (wx - 1, wy), (wx + 1, wy), (wx + 2, wy)]
    plan = wall_run.plan_wall_run(
        tiles, unit_const=WALL, existing_tiles=existing_tiles, existing_walls=existing_walls
    )
    assert len(plan.nodes) == 4
    assert len(plan.rewrites) == 1

    before = sum(len(u) for u in loaded.unit_manager.units)
    players = wall_run.touched_players(1, plan)
    model.begin_unit_edit(players)
    wall_run.apply_wall_plan(model, 1, plan)
    model.commit_unit_edit("Place 4 walls", EditHistory())

    assert sum(len(u) for u in loaded.unit_manager.units) == before + 4
    # A run along x now, written as the literal index, replacing the radian
    # 2.513274 the fixture ships.
    assert wall.rotation == 0.0
    placed = loaded.unit_manager.units[1][-4:]
    assert all(u.unit_const == WALL for u in placed)
    assert all(u.initial_animation_frame == 0 for u in placed)


# --- cross-path consistency -------------------------------------------
#
# The assertion that catches the read side and the write side drifting
# apart. Shaped like test_wall_connectivity.py's own synthetic harness.


@dataclass
class _Tile:
    x: int
    y: int
    elevation: int = 0
    terrain_id: int = 0
    layer: int = -1


class _MapManager:
    def __init__(self, size: int):
        self.map_width = self.map_height = size
        self.terrain = [_Tile(x, y) for y in range(size) for x in range(size)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, size, units_by_player):
        from descape.terrain_palette import PLAYER_COLORS

        self.map_manager = _MapManager(size)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def test_a_committed_run_agrees_with_the_render_sides_own_derivation():
    """Every wall this tool writes must read back through
    render.wall_variant_rotation_overrides() as the value it already stores.

    A lone radian-encoded wall is planted far away on purpose: without it
    the file classifies as integer-only, render.py skips it outright, and
    the assertion would pass vacuously no matter what the write side did.
    """
    from descape import render

    run = [(x, 10) for x in range(5, 15)] + [(14, y) for y in range(11, 18)]
    plan = _plan(run)
    placed = [
        FakeUnit(x=n.x, y=n.y, unit_const=n.unit_const, rotation=n.rotation)
        for n in plan.nodes
    ]
    radian_probe = _wall_at(40, 40, rotation=2 * 3.141592653589793 / 5)
    scenario = _Scenario(60, [[], [*placed, radian_probe], [], [], [], [], [], [], []])

    overrides = render.wall_variant_rotation_overrides(scenario)
    assert overrides, "the radian probe should have forced the override path to run"
    for (player_id, index), derived in overrides.items():
        unit = scenario.unit_manager.units[player_id][index]
        if unit is radian_probe:
            continue
        assert derived == unit.rotation, (unit.x, unit.y, derived, unit.rotation)
