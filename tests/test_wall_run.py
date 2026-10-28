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


def _plan(tiles, existing=(), blocked=None):
    existing_tiles, existing_walls = _existing(existing)
    # Only pass the keyword when a test asks for it, so the toggle-off path
    # stays the call every pre-GH #124 test made.
    extra = {} if blocked is None else {"blocked_tiles": frozenset(blocked)}
    return wall_run.plan_wall_run(
        tiles, unit_const=WALL, existing_tiles=existing_tiles, existing_walls=existing_walls, **extra
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


# --- a gate placed over walls (GH #159) ---------------------------------


def _gate_plan(gate_tiles, existing=()):
    existing_tiles, existing_walls = _existing(existing)
    return wall_run.plan_gate_over_walls(gate_tiles, existing_tiles=existing_tiles, existing_walls=existing_walls)


def _x_run(x0: int, x1: int, y: int) -> list[FakeUnit]:
    """A run along x storing the indices plan_wall_run() itself would write."""
    return [_wall_at(x, y, rotation=0.0 if x0 < x < x1 else 2.0) for x in range(x0, x1 + 1)]


def _stone_gate_tiles(tx: int, ty: int) -> list[tuple[int, int]]:
    """Stone Gate 64's footprint for a click on (tx, ty): 4 along x, tx-1..tx+2."""
    from descape import render

    tiles = render.occupied_tiles_for(GATE, tx + 0.5, ty + 0.5, 60, 60)
    assert tiles == [(tx + dx, ty) for dx in range(-1, 3)]
    return tiles


def test_a_gate_on_a_straight_run_removes_exactly_the_covered_walls():
    run = _x_run(5, 14, 10)
    plan = _gate_plan(_stone_gate_tiles(9, 10), run)
    assert [id(u) for _p, u in plan.removals] == [id(u) for u in run[3:7]]
    # (7, 10) and (12, 10) still have a connector either side: the gate.
    assert plan.rewrites == []
    assert plan


def test_a_gate_across_a_runs_end_reshapes_the_wall_it_newly_touches():
    """The x run's last two pieces go under the gate; its far end lands on
    (12, 10), right above a y run's tower end, which becomes a run along y."""
    x_run = _x_run(5, 10, 10)
    y_run = [_wall_at(12, y, rotation=1.0 if 11 < y < 14 else 2.0) for y in range(11, 15)]
    plan = _gate_plan(_stone_gate_tiles(10, 10), x_run + y_run)
    assert sorted(int(u.x) for _p, u in plan.removals) == [9, 10]
    assert [(int(u.x), int(u.y), v) for _p, u, v in plan.rewrites] == [(12, 11, 1)]


def test_a_gate_over_no_walls_plans_nothing():
    plan = _gate_plan(_stone_gate_tiles(30, 30), _x_run(5, 14, 10))
    assert plan.removals == [] and plan.rewrites == []
    assert not plan


def test_stacked_walls_on_one_tile_are_all_removed():
    stacked = [_wall_at(9, 10), _wall_at(9, 10), _wall_at(10, 10)]
    plan = _gate_plan(_stone_gate_tiles(9, 10), stacked)
    assert [id(u) for _p, u in plan.removals] == [id(u) for u in stacked]


def test_a_gate_already_under_the_footprint_is_left_alone():
    # Through wall_scene(), which is what keeps a gate out of the removal candidates.
    old_gate = FakeUnit(x=10.0, y=10.5, unit_const=GATE)
    scenario = _Scenario(60, [[], [old_gate, _wall_at(8, 10)], [], [], [], [], [], [], []])
    existing_tiles, existing_walls = wall_run.wall_scene(scenario, 60, 60)
    plan = wall_run.plan_gate_over_walls(
        _stone_gate_tiles(9, 10), existing_tiles=existing_tiles, existing_walls=existing_walls
    )
    assert [int(u.x) for _p, u in plan.removals] == [8]
    assert all(u is not old_gate for _p, u in plan.removals)


def test_walls_of_two_owners_are_both_removed_and_both_owners_touched():
    scenario = _Scenario(60, [[_wall_at(8, 10)], [], [_wall_at(9, 10)], [], [], [], [], [], []])
    existing_tiles, existing_walls = wall_run.wall_scene(scenario, 60, 60)
    plan = wall_run.plan_gate_over_walls(
        _stone_gate_tiles(9, 10), existing_tiles=existing_tiles, existing_walls=existing_walls
    )
    assert sorted(p for p, _u in plan.removals) == [0, 2]
    assert wall_run.touched_players(3, plan) == [0, 2, 3]


def test_touched_players_carries_a_gate_plans_rewritten_owner():
    x_run = _x_run(5, 10, 10)
    y_run = [_wall_at(12, y, rotation=2.0) for y in (11, 12)]
    plan = _gate_plan(_stone_gate_tiles(10, 10), x_run + y_run)
    assert plan.rewrites
    # _existing() owns everything as player 1.
    assert wall_run.touched_players(4, plan) == [1, 4]


def test_a_diagonal_gate_keeps_the_walls_on_its_bounding_box_corners():
    from descape import render

    x0, x1, y0, y1 = render.unit_tile_bounds(FakeUnit(x=40.0, y=40.0, unit_const=_DIAGONAL_GATE), 60, 60)
    footprint = render.occupied_tiles_for(_DIAGONAL_GATE, 40.0, 40.0, 60, 60)
    box = [_wall_at(x, y) for y in range(y0, y1) for x in range(x0, x1)]
    plan = _gate_plan(footprint, box)
    removed = {(int(u.x), int(u.y)) for _p, u in plan.removals}
    assert removed == set(footprint)
    assert len(removed) == 6
    assert (x0 + 3, y0) not in removed and (x0, y0 + 3) not in removed


def test_apply_gate_plan_removes_adds_and_leaves_the_gate_last():
    from descape.edit_history import EditHistory
    from descape.render import unit_tile_bounds

    loaded, model = _open()
    mm = loaded.map_manager
    wall = _unit(loaded, _REF_WALL)
    bounds = unit_tile_bounds(wall, mm.map_width, mm.map_height)
    wx, wy = bounds[0], bounds[2]
    existing_tiles, existing_walls = wall_run.wall_scene(loaded, mm.map_width, mm.map_height)
    plan = wall_run.plan_gate_over_walls(
        _stone_gate_tiles(wx, wy), existing_tiles=existing_tiles, existing_walls=existing_walls
    )
    assert [u for _p, u in plan.removals] == [wall]

    before = sum(len(u) for u in loaded.unit_manager.units)
    model.begin_unit_edit(wall_run.touched_players(1, plan))
    gate = wall_run.apply_gate_plan(model, 1, GATE, wx + 0.5, wy + 0.5, plan)
    model.commit_unit_edit("Place unit", EditHistory())

    assert sum(len(u) for u in loaded.unit_manager.units) == before
    assert loaded.unit_manager.units[1][-1] is gate
    assert gate.unit_const == GATE
    assert all(u.reference_id != _REF_WALL for u in loaded.unit_manager.get_all_units())


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


# --- Wall Circle's ellipse (GH #125) -------------------------------------


def _ellipse(x0, y0, x1, y1, size=60):
    from descape import shape_tools

    return shape_tools.ellipse_perimeter_tiles(x0, y0, x1, y1, size, size)


def _expected_shape(tile, ring: set) -> int:
    """The shape a ring tile's two ring neighbours call for, read off their
    directions: an axis stretch is a run (0 along x, 1 along y), a diagonal
    stretch a diagonal run (4 for NE-SW, 3 for NW-SE, north being -y), and a
    tile between an axis and a diagonal step a tower (2)."""
    x, y = tile
    steps = {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (x + dx, y + dy) in ring}
    assert len(steps) == 2, (tile, steps)
    shapes = {
        frozenset({(-1, 0), (1, 0)}): 0,
        frozenset({(0, -1), (0, 1)}): 1,
        frozenset({(1, -1), (-1, 1)}): 4,
        frozenset({(-1, -1), (1, 1)}): 3,
    }
    return shapes.get(frozenset(steps), 2)


@pytest.mark.parametrize("box", [(5, 10, 19, 20), (5, 10, 20, 21), (10, 5, 20, 29), (3, 3, 33, 33)])
def test_an_ellipse_ring_gives_runs_diagonals_and_towers_where_its_steps_say(box):
    ring = _ellipse(*box)
    plan = _plan(ring)
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert set(by_tile) == set(ring)
    assert by_tile == {tile: _expected_shape(tile, set(ring)) for tile in ring}
    assert set(by_tile.values()) == {0, 1, 2, 3, 4}, "every shape should be exercised"


def test_an_ellipse_sharing_a_tile_with_an_existing_run_skips_it_and_rewrites_the_junction():
    """An existing x run (3..7, 15) crosses the ellipse's left apex (5, 15):
    that tile is skipped, and its piece, now a crossing, is rewritten 0 -> 2
    in the same plan. The run's other pieces keep their shapes."""
    ring = _ellipse(5, 10, 19, 20)
    assert {(5, 14), (5, 15), (5, 16)} <= set(ring) and (4, 15) not in ring and (6, 15) not in ring
    existing = [_wall_at(x, 15, 2.0 if x in (3, 7) else 0.0) for x in range(3, 8)]
    plan = _plan(ring, existing)
    assert plan.skipped == 1
    assert len(plan.nodes) == len(ring) - 1
    assert [(unit, variant) for _p, unit, variant in plan.rewrites] == [(existing[2], 2)]
    by_tile = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert by_tile[(5, 14)] == by_tile[(5, 16)] == 2


def test_a_path_wholly_on_existing_walls_plans_nothing_new():
    existing = [_wall_at(x, 10) for x in range(5, 11)]
    plan = _plan([(x, 10) for x in range(5, 11)], existing)
    assert plan.nodes == []
    assert plan.skipped == 6


# --- occupied tiles (GH #124: Walls skip occupied tiles) ----------------


def _variants(plan) -> dict[tuple[int, int], int]:
    return {(int(n.x), int(n.y)): n.variant for n in plan.nodes}


def test_a_blocked_tile_mid_run_is_a_gap_with_ends_either_side():
    """A blocked tile is not a neighbour: (6, 10) and (8, 10) each see one
    path neighbour, so both are run ends (2), not a straight run (0)."""
    plan = _plan([(x, 10) for x in range(5, 11)], blocked={(7, 10)})
    assert _variants(plan) == {(5, 10): 2, (6, 10): 2, (8, 10): 2, (9, 10): 0, (10, 10): 2}
    assert plan.blocked == 1
    assert plan.skipped == 0


def test_a_connector_still_counts_as_a_neighbour_with_blocked_tiles_passed():
    plan = _plan([(x, 10) for x in range(5, 11)], [_wall_at(7, 10)], blocked={(9, 10)})
    assert _variants(plan)[(6, 10)] == 0
    assert _variants(plan)[(8, 10)] == 2
    assert (9, 10) not in _variants(plan)


def test_blocked_and_skipped_count_independently_and_a_wall_tile_counts_as_skipped():
    """A tile holding both a wall and another object follows the wall rule."""
    plan = _plan([(x, 10) for x in range(5, 11)], [_wall_at(7, 10)], blocked={(7, 10), (9, 10)})
    assert plan.skipped == 1
    assert plan.blocked == 1
    assert len(plan.nodes) == 4


def test_a_path_wholly_blocked_plans_nothing():
    tiles = [(x, 10) for x in range(5, 11)]
    plan = _plan(tiles, [_wall_at(5, 11)], blocked=set(tiles))
    assert plan.nodes == []
    assert plan.rewrites == []
    assert plan.blocked == 6
    assert plan.skipped == 0


def test_an_existing_wall_beside_a_blocked_gap_is_not_rewritten_as_if_connected():
    """The existing wall at (8, 10) stores 2 and sees only (9, 10) across the
    tree at (7, 10). Feeding the tree into the neighbour mask would read W+E and
    rewrite it to 0."""
    wall = _wall_at(8, 10, rotation=2.0)
    plan = _plan([(x, 10) for x in range(5, 11)], [wall], blocked={(7, 10)})
    assert plan.rewrites == []
    assert _variants(plan)[(6, 10)] == 2
    assert _variants(plan)[(9, 10)] == 0


def test_without_blocked_tiles_the_plan_counts_none():
    plan = _plan([(x, 10) for x in range(5, 11)])
    assert plan.blocked == 0
    assert len(plan.nodes) == 6


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


def test_a_gapped_run_agrees_with_the_render_sides_own_derivation():
    """GH #124's cross-path check: render only reads connectors, so a gap at
    a blocked tile must read as a gap on both sides."""
    from descape import render

    run = [(x, 10) for x in range(5, 15)] + [(14, y) for y in range(11, 18)]
    blocked = {(8, 10), (11, 10), (14, 13)}
    plan = _plan(run, blocked=blocked)
    assert plan.blocked == 3
    placed = [
        FakeUnit(x=n.x, y=n.y, unit_const=n.unit_const, rotation=n.rotation)
        for n in plan.nodes
    ]
    radian_probe = _wall_at(40, 40, rotation=2 * 3.141592653589793 / 5)
    scenario = _Scenario(60, [[], [*placed, radian_probe], [], [], [], [], [], [], []])

    overrides = render.wall_variant_rotation_overrides(scenario)
    assert overrides
    for (player_id, index), derived in overrides.items():
        unit = scenario.unit_manager.units[player_id][index]
        if unit is radian_probe:
            continue
        assert derived == unit.rotation, (unit.x, unit.y, derived, unit.rotation)


# Real consts, so blocked_tiles() reads the committed catalog and footprints.
_TREE = 349
_GOLD_MINE = 66
_INVISIBLE_BLOCKER = 1776
_HOUSE = 70
_PASSABLE_EYE_CANDY = 143  # RUBL1, obstruction 0
_OBSTACLE = 623  # ROCKX, obstruction 2 (GH #149)
_GRASS = 1358
_DIAGONAL_GATE = 659


def test_blocked_tiles_covers_every_footprint_except_passable_eye_candy_and_connectors():
    from descape import render, unit_kind

    assert _PASSABLE_EYE_CANDY in unit_kind.passable_eye_candy_consts()
    assert _OBSTACLE in unit_kind.obstacle_consts()
    assert _INVISIBLE_BLOCKER not in unit_kind.eye_candy_consts()
    house = FakeUnit(x=21.0, y=21.0, unit_const=_HOUSE)
    diagonal_gate = FakeUnit(x=40.0, y=40.0, unit_const=_DIAGONAL_GATE)
    gaia = [
        _wall_at(10, 10, const=_TREE),
        _wall_at(12, 10, const=_GOLD_MINE),
        _wall_at(14, 10, const=_INVISIBLE_BLOCKER),
        _wall_at(16, 10, const=_PASSABLE_EYE_CANDY),
        _wall_at(18, 10, const=_OBSTACLE),
    ]
    player = [house, _wall_at(30, 30), _wall_at(34, 30, const=GATE), diagonal_gate]
    scenario = _Scenario(60, [gaia, player, [], [], [], [], [], [], []])

    house_tiles = set(render.unit_occupied_tiles(house, 60, 60))
    assert len(house_tiles) == 4
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    assert blocked == {(10, 10), (12, 10), (14, 10), (18, 10)} | house_tiles
    # The diagonal gate's bounding rect has empty corners; none of it blocks.
    x0, x1, y0, y1 = render.unit_tile_bounds(diagonal_gate, 60, 60)
    assert not blocked & {(x, y) for x in range(x0, x1) for y in range(y0, y1)}


def test_a_diagonal_footprint_blocks_its_tiles_but_not_its_empty_corners(monkeypatch):
    """The test above only proves connectors are skipped, and every real
    sparse-footprint const is one. With 659 treated as a plain object, its
    six offsets block, (0,0) and (3,3) included, and the empty corners don't."""
    from descape import render

    connectors = unit_sprites.wall_connector_consts()
    assert _DIAGONAL_GATE in connectors
    monkeypatch.setattr(unit_sprites, "wall_connector_consts", lambda: connectors - {_DIAGONAL_GATE})
    gate = FakeUnit(x=40.0, y=40.0, unit_const=_DIAGONAL_GATE)
    scenario = _Scenario(60, [[], [gate], [], [], [], [], [], [], []])

    x0, _x1, y0, _y1 = render.unit_tile_bounds(gate, 60, 60)
    offsets = {(1, 1), (1, 2), (2, 1), (2, 2), (0, 0), (3, 3)}
    assert set(render.unit_occupied_tiles(gate, 60, 60)) == {(x0 + dx, y0 + dy) for dx, dy in offsets}
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    assert blocked == {(x0 + dx, y0 + dy) for dx, dy in offsets}
    assert not blocked & {(x0 + 3, y0), (x0, y0 + 3)}


_BLOCKER_3X1 = 2424


def test_a_wall_run_across_a_3x1_blocker_leaves_a_3_tile_gap():
    """GH #121 x GH #124: the blocker's real span blocks, not only its centre tile."""
    gaia = [_wall_at(10, 10, const=_BLOCKER_3X1)]
    scenario = _Scenario(60, [gaia, [], [], [], [], [], [], [], []])
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    assert blocked == {(9, 10), (10, 10), (11, 10)}
    plan = _plan([(x, 10) for x in range(5, 16)], blocked=blocked)
    assert plan.blocked == 3
    assert {int(node.x) for node in plan.nodes} == {5, 6, 7, 8, 12, 13, 14, 15}


def test_an_obstacle_blocks_its_tile_and_grass_does_not():
    """GH #149: solid decor is occupied, passable decor is not."""
    gaia = [_wall_at(10, 10, const=_OBSTACLE), _wall_at(12, 10, const=_GRASS)]
    scenario = _Scenario(60, [gaia, [], [], [], [], [], [], [], []])
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    assert (10, 10) in blocked
    assert (12, 10) not in blocked


def test_a_wall_run_through_a_rock_leaves_that_tile_empty():
    gaia = [_wall_at(10, 10, const=_OBSTACLE)]
    scenario = _Scenario(60, [gaia, [], [], [], [], [], [], [], []])
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    plan = _plan([(x, 10) for x in range(5, 16)], blocked=blocked)
    assert plan.blocked == 1
    assert {int(node.x) for node in plan.nodes} == set(range(5, 16)) - {10}


def test_a_mountain_blocks_its_whole_footprint():
    """TASK-209, closed by TASK-134: a 6x6 mountain on a whole coordinate
    blocks all 36 tiles of its footprint, not just its anchor tile."""
    mountain = FakeUnit(x=20.0, y=20.0, unit_const=744, rotation=0.0)  # MNTN3
    scenario = _Scenario(60, [[mountain], [], [], [], [], [], [], [], []])
    blocked = wall_run.blocked_tiles(scenario, 60, 60)
    assert blocked == {(x, y) for x in range(17, 23) for y in range(17, 23)}
    assert len(blocked) == 36


# --- a moved gate (GH #194: Moving a gate updates walls) -----------------

_PALISADE = 72
_VILLAGER = 83
_GATE_Y = 88  # Stone Gate, 4 along y
_GATE_NE_SW = 667  # Stone Gate, the other diagonal from 659


def _tile(unit) -> tuple[int, int]:
    return int(unit.x), int(unit.y)


def _gate_tiles(const: int, tx: int, ty: int) -> list[tuple[int, int]]:
    """`const`'s sparse footprint with its bounding box's low corner on (tx, ty)."""
    from descape import render

    span = render.tile_span(const, render.NON_BUILDING_SPAN)
    return render.occupied_tiles_for(const, *render.span_anchor(tx, ty, *span), 60, 60)


def _gated_run(x0: int, x1: int, y: int, gate_x: int) -> list[FakeUnit]:
    """_x_run() with a Stone Gate standing on gate_x..gate_x+3: the walls it
    replaced are gone, and the ones beside it already store the run index."""
    return [u for u in _x_run(x0, x1, y) if not gate_x <= int(u.x) < gate_x + 4]


def _move_plan(moves, walls=(), blocked=(), removable=None):
    """plan_gate_moves() for [(old_tiles, new_tiles), ...] over `walls`, the
    stationary ones: FakeUnits (owned by player 1), or ExistingWalls as is."""
    walls = list(walls)
    if walls and not isinstance(walls[0], wall_run.ExistingWall):
        static_tiles, static_walls = _existing(walls)
    else:
        static_tiles, static_walls = {(w.tx, w.ty) for w in walls}, walls
    return wall_run.plan_gate_moves(
        [wall_run.GateMove(frozenset(old), frozenset(new)) for old, new in moves],
        static_tiles=static_tiles,
        static_walls=static_walls,
        moved_tiles={tile for _old, new in moves for tile in new},
        blocked_tiles=frozenset(blocked),
        removable=removable,
    )


def _fills(plan) -> dict[tuple[int, int], int]:
    return {(int(node.x), int(node.y)): node.variant for _p, node in plan.fills}


def _after_move(walls, plan) -> list[FakeUnit]:
    """The stationary walls once `plan` is applied, for planning the next move."""
    gone = {id(u) for _p, u in plan.removals}
    kept = [u for u in walls if id(u) not in gone]
    for _p, unit, variant in plan.rewrites:
        unit.rotation = float(variant)
    return kept + [
        FakeUnit(x=node.x, y=node.y, unit_const=node.unit_const, rotation=node.rotation) for _p, node in plan.fills
    ]


@pytest.mark.parametrize(
    ("const", "line", "step"),
    [
        (GATE, [(10, 10), (11, 10), (12, 10), (13, 10)], (1, 0)),
        (_GATE_Y, [(10, 10), (10, 11), (10, 12), (10, 13)], (0, 1)),
        (_DIAGONAL_GATE, [(10, 10), (11, 11), (12, 12), (13, 13)], (1, 1)),
        (_GATE_NE_SW, [(10, 13), (11, 12), (12, 11), (13, 10)], (1, -1)),
    ],
)
def test_gate_run_line_is_the_line_through_each_stone_orientation(const, line, step):
    tiles = _gate_tiles(const, 10, 10)
    assert len(tiles) == (4 if step[0] * step[1] == 0 else 6)
    assert wall_run.gate_run_line(tiles) == (line, step)


def test_gate_run_line_has_no_answer_for_one_tile_or_a_lineless_shape():
    assert wall_run.gate_run_line([(7, 7)]) is None
    assert wall_run.gate_run_line([]) is None
    # A 2x3 block has no diagonal, and a full square has two.
    assert wall_run.gate_run_line([(x, y) for x in range(2) for y in range(3)]) is None
    assert wall_run.gate_run_line([(x, y) for x in range(2) for y in range(2)]) is None


def test_a_gate_moved_five_along_a_run_swaps_four_walls_for_four_fills():
    """The GH #194 screenshots: the walls under the new footprint go and the
    four tiles the gate left get the run's own wall, as run interior."""
    walls = _gated_run(5, 20, 10, gate_x=8)
    old, new = _gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 13, 10)
    plan = _move_plan([(old, new)], walls)

    assert sorted(_tile(u) for _p, u in plan.removals) == sorted(new)
    assert [(p, n.unit_const, n.variant, n.initial_animation_frame) for p, n in plan.fills] == [(1, WALL, 0, 0)] * 4
    assert sorted(_fills(plan)) == sorted(old)
    assert [(n.x, n.y) for _p, n in plan.fills] == [(tx + 0.5, ty + 0.5) for tx, ty in old]
    assert plan.rewrites == []
    assert plan.blocked == 0
    assert plan
    assert not {_tile(u) for u in _after_move(walls, plan)} & set(new)


def test_a_gate_moved_one_along_a_run_swaps_one_wall_for_one_fill():
    """The three tiles both footprints share are neither removed nor filled."""
    walls = _gated_run(5, 20, 10, gate_x=8)
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 9, 10))], walls)
    assert [_tile(u) for _p, u in plan.removals] == [(12, 10)]
    assert _fills(plan) == {(8, 10): 0}
    assert plan.rewrites == []


def test_a_gate_moved_one_back_along_a_run_fills_from_its_high_end():
    """The low end's wall is the one the gate lands on, so it is no source."""
    walls = _gated_run(5, 20, 10, gate_x=8)
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 7, 10))], walls)
    assert [_tile(u) for _p, u in plan.removals] == [(7, 10)]
    assert _fills(plan) == {(11, 10): 0}


def test_a_free_standing_gate_leaves_nothing_behind():
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 13, 10))], _x_run(30, 40, 30))
    assert (plan.removals, plan.fills, plan.rewrites, plan.blocked) == ([], [], [], 0)
    assert not plan


def test_a_wall_only_at_the_high_end_is_still_the_source():
    """(8, 10) has nothing west of it once the gate is gone, so it is a run end."""
    walls = [_wall_at(x, 10, rotation=2.0 if x == 15 else 0.0, const=_PALISADE) for x in range(12, 16)]
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 30, 30))], walls)
    assert {n.unit_const for _p, n in plan.fills} == {_PALISADE}
    assert _fills(plan) == {(8, 10): 2, (9, 10): 0, (10, 10): 0, (11, 10): 0}
    assert plan.removals == [] and plan.rewrites == []


def test_two_different_ends_fill_with_the_low_ends_wall():
    low = wall_run.ExistingWall(unit=_wall_at(7, 10, rotation=2.0, const=_PALISADE), player_id=2, tx=7, ty=10)
    high = wall_run.ExistingWall(unit=_wall_at(12, 10, rotation=2.0), player_id=1, tx=12, ty=10)
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 30, 30))], [high, low])
    assert [(p, n.unit_const) for p, n in plan.fills] == [(2, _PALISADE)] * 4
    assert wall_run.gate_move_players(plan) == {2}


def test_a_gate_stepped_sideways_off_its_run_twice_fills_only_once():
    """The trail case, and why the source must be collinear: after the first
    step the filled row lies beside the gate, not past either end of it."""
    walls = _gated_run(5, 20, 10, gate_x=8)
    first = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 8, 11))], walls)
    assert sorted(_fills(first)) == [(x, 10) for x in range(8, 12)]
    assert first.removals == []

    walls = _after_move(walls, first)
    assert sorted(_tile(u) for u in walls) == [(x, 10) for x in range(5, 21)]
    second = _move_plan([(_gate_tiles(GATE, 8, 11), _gate_tiles(GATE, 8, 12))], walls)
    assert second.fills == []
    assert second.removals == []
    # The run the gate stood against is a plain run again.
    assert {_tile(u): v for _p, u, v in second.rewrites} == {(x, 10): 0 for x in range(8, 12)}


def test_a_diagonal_gate_fills_its_four_line_tiles_not_its_six_tile_footprint():
    old = _gate_tiles(_DIAGONAL_GATE, 10, 10)
    assert len(old) == 6
    walls = [_wall_at(i, i, rotation=2.0 if i in (5, 20) else 3.0) for i in range(5, 21) if not 10 <= i <= 13]
    plan = _move_plan([(old, _gate_tiles(_DIAGONAL_GATE, 30, 40))], walls)
    assert _fills(plan) == {(i, i): 3 for i in range(10, 14)}
    assert not {(11, 12), (12, 11)} & set(_fills(plan))
    assert plan.rewrites == []


def test_a_blocked_line_tile_is_left_open_with_run_ends_either_side():
    walls = _gated_run(5, 20, 10, gate_x=8)
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 13, 10))], walls, blocked={(9, 10)})
    assert _fills(plan) == {(8, 10): 2, (10, 10): 2, (11, 10): 0}
    assert plan.blocked == 1
    assert len(plan.removals) == 4


def test_a_wholly_blocked_line_is_a_falsy_plan_that_still_counts():
    """The lone wall keeps its shape (a tower either way), so nothing at all changes."""
    old = _gate_tiles(GATE, 8, 10)
    plan = _move_plan([(old, _gate_tiles(GATE, 30, 30))], [_wall_at(7, 10, rotation=2.0)], blocked=set(old))
    assert not plan
    assert plan.blocked == 4
    assert plan.fills == []


def test_a_gate_moved_outward_from_a_runs_end_keeps_the_run_attached():
    """The intended axial case: each fill is the next move's source, so the
    run grows behind a gate carried along its own line."""
    walls = [_wall_at(x, 10, rotation=2.0 if x == 5 else 0.0) for x in range(5, 10)]
    first = _move_plan([(_gate_tiles(GATE, 10, 10), _gate_tiles(GATE, 14, 10))], walls)
    assert _fills(first) == {(x, 10): 0 for x in range(10, 14)}

    walls = _after_move(walls, first)
    second = _move_plan([(_gate_tiles(GATE, 14, 10), _gate_tiles(GATE, 18, 10))], walls)
    assert _fills(second) == {(x, 10): 0 for x in range(14, 18)}
    assert sorted(_tile(u) for u in _after_move(walls, second)) == [(x, 10) for x in range(5, 18)]
    assert first.rewrites == [] and second.rewrites == []


def test_a_run_meeting_the_gates_flank_is_reshaped_and_nothing_fills():
    """A y run ending under the gate's last tile: no wall continues the gate's
    own line, so nothing fills, and the touching wall becomes a run end."""
    walls = [_wall_at(11, y, rotation=2.0 if y == 15 else 1.0) for y in range(11, 16)]
    plan = _move_plan([(_gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 30, 30))], walls)
    assert plan.fills == [] and plan.removals == []
    assert [(_tile(u), v) for _p, u, v in plan.rewrites] == [((11, 11), 2)]
    assert plan


def test_wall_scene_and_blocked_tiles_leave_out_exactly_the_excluded_units():
    gate = FakeUnit(x=10.0, y=10.5, unit_const=GATE)
    wall, other_wall = _wall_at(7, 10), _wall_at(20, 20)
    villager, tree = _wall_at(9, 12, const=_VILLAGER), _wall_at(30, 30, const=_TREE)
    scenario = _Scenario(60, [[tree], [gate, wall, other_wall, villager], [], [], [], [], [], [], []])
    gate_tiles = set(_gate_tiles(GATE, 8, 10))

    tiles, walls = wall_run.wall_scene(scenario, 60, 60)
    assert tiles == gate_tiles | {(7, 10), (20, 20)}
    assert [w.unit for w in walls] == [wall, other_wall]
    assert wall_run.blocked_tiles(scenario, 60, 60) == {(9, 12), (30, 30)}

    tiles, walls = wall_run.wall_scene(scenario, 60, 60, exclude_ids=frozenset({id(gate), id(wall)}))
    assert tiles == {(20, 20)}
    assert [w.unit for w in walls] == [other_wall]
    assert wall_run.blocked_tiles(scenario, 60, 60, exclude_ids=frozenset({id(villager)})) == {(30, 30)}


def _gate_scenario(extra=()):
    """A Stone Gate on (8..11, 10) inside the x run 5..20, owned by player 1."""
    gate = FakeUnit(x=10.0, y=10.5, unit_const=GATE)
    walls = _gated_run(5, 20, 10, gate_x=8)
    return gate, walls, _Scenario(60, [[], [gate, *walls, *extra], [], [], [], [], [], [], []])


def _shifted(units, dx: float, dy: float):
    return [(u, u.x + dx, u.y + dy) for u in units]


def test_scenario_gate_moves_plans_the_screenshot_case_off_a_scenario():
    """The gate's own old tiles must not count as connectors: wall_scene()
    without the mover excluded would find nothing to fill."""
    gate, _walls, scenario = _gate_scenario()
    plan = wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate], 5, 0))
    assert sorted(_tile(u) for _p, u in plan.removals) == [(x, 10) for x in range(13, 17)]
    assert _fills(plan) == {(x, 10): 0 for x in range(8, 12)}
    assert plan.rewrites == []


def test_scenario_gate_moves_leaves_walls_that_move_with_the_gate_alone():
    """Not removed (one starts under the new footprint), not a source (the
    low end's is a Palisade), not rewritten (it stores a stale shape)."""
    low_end = _wall_at(7, 10, rotation=4.0, const=_PALISADE)
    under_new = _wall_at(10, 13)
    static_under_new = _wall_at(9, 13, rotation=2.0)
    gate, walls, scenario = _gate_scenario([under_new, static_under_new])
    walls[2] = low_end
    scenario.unit_manager.units[1][3] = low_end
    assert _tile(scenario.unit_manager.units[1][3]) == (7, 10)

    plan = wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate, low_end, under_new], 0, 3))

    assert [u for _p, u in plan.removals] == [static_under_new]
    assert {n.unit_const for _p, n in plan.fills} == {WALL}
    # (8, 10) lost its west neighbour along with the gate, so it is a run end.
    assert _fills(plan) == {(8, 10): 2, (9, 10): 0, (10, 10): 0, (11, 10): 0}
    # Nor is the run the Palisade left re-derived: only a gate's own sites are.
    assert plan.rewrites == []


def test_scenario_gate_moves_blocks_a_mover_at_its_new_tile_not_its_old_one():
    arriving = _wall_at(9, 7, const=_VILLAGER)
    leaving = _wall_at(10, 10, const=_VILLAGER)
    gate, _walls, scenario = _gate_scenario([arriving, leaving])
    plan = wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate, arriving, leaving], 0, 3))
    assert sorted(_fills(plan)) == [(8, 10), (10, 10), (11, 10)]
    assert plan.blocked == 1


def test_scenario_gate_moves_blocks_on_whatever_stays_in_the_gateway():
    """Always, whatever "Walls skip occupied tiles" says: no wall lands on a guard."""
    guard = _wall_at(9, 10, const=_VILLAGER)
    gate, _walls, scenario = _gate_scenario([guard])
    plan = wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate], 0, 3))
    assert sorted(_fills(plan)) == [(8, 10), (10, 10), (11, 10)]
    assert plan.blocked == 1


def test_scenario_gate_moves_plans_nothing_while_the_footprint_stays_put():
    """An arrow nudge is 0.1 tile: no gate changes tiles, so the walls beside
    it are not re-derived and the move stays a plain field edit."""
    gate, walls, scenario = _gate_scenario()
    walls[3].rotation = 4.0  # a stale stored shape right beside the gate
    assert wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate], 0.1, 0)) is None
    assert wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([walls[0]], 5, 5)) is None
    assert wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate], 1, 0)) is not None


def test_scenario_gate_moves_keeps_a_wall_the_model_would_not_remove():
    gate, walls, scenario = _gate_scenario()
    kept = next(u for u in walls if _tile(u) == (14, 10))
    plan = wall_run.scenario_gate_moves(scenario, 60, 60, _shifted([gate], 5, 0), removable=lambda u: u is not kept)
    removed = [u for _p, u in plan.removals]
    assert kept not in removed
    assert sorted(_tile(u) for u in removed) == [(13, 10), (15, 10), (16, 10)]
    assert len(plan.fills) == 4


def test_gate_move_players_is_every_owner_the_plan_touches():
    node = wall_run.WallNode(unit_const=WALL, variant=0, x=8.5, y=10.5)
    plan = wall_run.GateMovePlan(
        removals=[(0, _wall_at(13, 10))], fills=[(2, node)], rewrites=[(5, _wall_at(12, 10), 2)], blocked=3
    )
    assert wall_run.gate_move_players(plan) == {0, 2, 5}
    assert wall_run.gate_move_players(wall_run.GateMovePlan(removals=[], fills=[], rewrites=[])) == set()


def test_apply_gate_move_plan_adds_the_fills_then_rewrites():
    """Through the real model: each fill lands in its own owner's list with a
    literal index and frame 0, and the rewrite replaces the fixture wall's radian 2."""
    from descape.edit_history import EditHistory

    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    owner = next(p for p, units in enumerate(loaded.unit_manager.units) if any(u is wall for u in units))
    plan = wall_run.GateMovePlan(
        removals=[],
        fills=[
            (1, wall_run.WallNode(unit_const=WALL, variant=0, x=30.5, y=30.5)),
            (2, wall_run.WallNode(unit_const=_PALISADE, variant=2, x=31.5, y=30.5)),
            (1, wall_run.WallNode(unit_const=WALL, variant=3, x=32.5, y=30.5)),
        ],
        rewrites=[(owner, wall, 4)],
    )
    before = [len(u) for u in loaded.unit_manager.units]

    history = EditHistory()
    model.begin_unit_edit(sorted(wall_run.gate_move_players(plan)))
    added = wall_run.apply_gate_move_plan(model, plan)
    model.commit_unit_edit("Move unit", history)

    units = loaded.unit_manager.units
    assert [len(u) for u in units] == [n + {1: 2, 2: 1}.get(p, 0) for p, n in enumerate(before)]
    assert [(u.unit_const, u.x, u.rotation) for u in units[1][-2:]] == [(WALL, 30.5, 0.0), (WALL, 32.5, 3.0)]
    assert (units[2][-1].unit_const, units[2][-1].rotation) == (_PALISADE, 2.0)
    assert all(u.initial_animation_frame == 0 for u in added) and len(added) == 3
    assert wall.rotation == 4.0

    history.undo([], None, None, model)
    assert [len(u) for u in loaded.unit_manager.units] == before
    assert _unit(loaded, _REF_WALL).rotation != 4.0


def test_a_wall_a_moved_gate_leaves_with_no_neighbour_becomes_a_tower():
    """GH #266's tower rule in the gate part too: between two gates the wall
    is run interior, and once both leave with their lines kept open it
    stands alone, so it takes 2 rather than keeping its run index."""
    gate_a, gate_b = _gate_tiles(GATE, 8, 10), _gate_tiles(GATE, 3, 10)
    wall = _wall_at(7, 10, rotation=0.0)
    away = [(gate_a, _gate_tiles(GATE, 30, 30)), (gate_b, _gate_tiles(GATE, 30, 40))]
    plan = _move_plan(away, [wall], blocked=set(gate_a) | set(gate_b))
    assert plan.fills == []
    assert plan.rewrites == [(1, wall, 2)]


# --- a moved wall (GH #266) -----------------------------------------------
#
# Each fixture wall stores the index its neighbours derive, unless a test
# says it is stale on purpose, so no test passes on a mis-shaped fixture.

_RADIAN_1 = 2 * 3.141592653589793 / 5  # index 1 in the radian convention


def _walls_scenario(*walls, gaia=()):
    """`walls` owned by player 1, and `gaia` by GAIA, on a 60x60 map."""
    return _Scenario(60, [list(gaia), list(walls), [], [], [], [], [], [], []])


def _plan_moves(scenario, moves, gate_walls: bool = False):
    return wall_run.scenario_move_plan(scenario, 60, 60, moves, gate_walls=gate_walls)


def _by_tile(entries) -> dict[tuple[int, int], int]:
    """(player_id, unit, variant) entries keyed by each unit's tile before the move."""
    return {_tile(u): v for _p, u, v in entries}


def test_a_mid_run_wall_moved_off_its_run_is_a_tower_and_so_are_its_old_neighbours():
    """The GH #266 screenshot: the moved piece, and the run ends either side of the gap it left."""
    run = _x_run(5, 15, 10)
    wall = run[5]
    assert (_tile(wall), wall.rotation) == ((10, 10), 0.0)
    plan = _plan_moves(_walls_scenario(*run), [(wall, wall.x, wall.y + 3)])
    assert plan.moved == [(1, wall, 2)]
    assert _by_tile(plan.rewrites) == {(9, 10): 2, (11, 10): 2}
    assert (plan.removals, plan.fills, plan.blocked, plan.radian_flip) == ([], [], 0, False)
    assert plan and not plan.wholesale


def test_a_lone_tower_moved_into_a_runs_gap_joins_it():
    run = [u for u in _x_run(5, 15, 10) if _tile(u) != (10, 10)]
    for unit in run:
        if _tile(unit) in ((9, 10), (11, 10)):
            unit.rotation = 2.0  # the run ends either side of the gap
    tower = _wall_at(30, 30, rotation=2.0)
    plan = _plan_moves(_walls_scenario(*run, tower), [(tower, 10.5, 10.5)])
    assert plan.moved == [(1, tower, 0)]
    assert _by_tile(plan.rewrites) == {(9, 10): 0, (11, 10): 0}


def test_a_stale_lone_wall_moved_into_the_open_again_becomes_a_tower():
    """Re-derived whether or not its mask changed, so moving a wall an earlier move left stale fixes it."""
    stale = _wall_at(30, 30, rotation=0.0)
    plan = _plan_moves(_walls_scenario(stale), [(stale, 40.5, 40.5)])
    assert plan.moved == [(1, stale, 2)]
    assert plan.rewrites == []


def test_a_whole_consistent_run_moved_together_changes_no_wall():
    run = _x_run(5, 15, 10)
    plan = _plan_moves(_walls_scenario(*run), _shifted(run, 0, 5))
    assert (plan.moved, plan.rewrites) == ([], [])
    assert not plan


def test_a_whole_run_moved_together_rewrites_only_its_stale_wall():
    """The cost of always re-deriving a carried wall, pinned: a structure moved whole fixes any stale wall in it."""
    run = _x_run(5, 15, 10)
    run[3].rotation = 4.0
    plan = _plan_moves(_walls_scenario(*run), _shifted(run, 0, 5))
    assert plan.moved == [(1, run[3], 0)]
    assert plan.rewrites == []


def test_a_wall_moved_off_a_stack_reshapes_nothing_beside_the_tile_it_left():
    """The trigger rule: (10, 10) still holds a wall afterwards, so no
    neighbour's mask changed, and the neighbour storing radian noise (index 1
    where it derives 2) is left as it is."""
    staying, leaving = _wall_at(10, 10, rotation=0.0), _wall_at(10, 10, rotation=0.0)
    noisy, end = _wall_at(9, 10, rotation=_RADIAN_1), _wall_at(11, 10, rotation=2.0)
    plan = _plan_moves(_walls_scenario(noisy, staying, leaving, end), [(leaving, 10.5, 15.5)])
    assert plan.moved == [(1, leaving, 2)]
    assert plan.rewrites == []


def test_a_wall_landing_diagonal_to_a_stationary_wall_reshapes_it():
    """The 8-bit mask: (20, 20) ends a NW-SE diagonal at (19, 19), and the landing wall continues it."""
    tail, end, lone = _wall_at(19, 19, rotation=2.0), _wall_at(20, 20, rotation=2.0), _wall_at(40, 40, rotation=2.0)
    plan = _plan_moves(_walls_scenario(tail, end, lone), [(lone, 21.5, 21.5)])
    assert plan.rewrites == [(1, end, 3)]
    assert plan.moved == []


def test_a_runs_middle_left_by_both_its_ends_becomes_a_tower():
    """`isolated`: no neighbour derives no shape, which _neighbour_rewrites() skips without it."""
    west, middle, east = _x_run(5, 7, 10)
    plan = _plan_moves(_walls_scenario(west, middle, east), [(west, 5.5, 20.5), (east, 7.5, 20.5)])
    assert plan.rewrites == [(1, middle, 2)]
    assert plan.moved == []
    alone = [wall_run.ExistingWall(unit=middle, player_id=1, tx=6, ty=10)]
    assert wall_run._neighbour_rewrites(alone, {(6, 10)}, set(), set()) == []
    assert wall_run._neighbour_rewrites(alone, {(6, 10)}, set(), set(), isolated=2) == [(1, middle, 2)]


def test_a_group_nudge_re_derives_a_neighbour_that_stays_in_its_tile():
    """The same 0.1 carries one wall over a tile edge and not its neighbour, which counts as staying."""
    run = _x_run(9, 12, 10)
    crossing, staying = run[1], run[2]
    crossing.y = 10.95
    plan = _plan_moves(_walls_scenario(*run), _shifted([crossing, staying], 0, 0.1))
    assert plan.moved == [(1, crossing, 2)]
    assert plan.rewrites == [(1, staying, 2)]


def test_a_gate_moved_with_the_box_off_reshapes_nothing_beside_its_old_site():
    """The walls either side of the gate's old site keep their run index. A
    moved wall still counts the co-moved gate on its new tiles: landing
    between it and a lone wall, it is run interior."""
    lone, beyond = _wall_at(30, 30, rotation=2.0), _wall_at(13, 13, rotation=2.0)
    gate, _walls, scenario = _gate_scenario([lone, beyond])
    plan = _plan_moves(scenario, [(gate, gate.x, gate.y + 3), (lone, 12.5, 13.5)])
    assert plan.moved == [(1, lone, 0)]
    assert plan.rewrites == []
    assert (plan.removals, plan.fills) == ([], [])
    assert not plan.wholesale


def test_a_gate_and_a_wall_moved_with_the_box_on_name_each_rewritten_wall_once():
    """GH #194's removals and fills as before. (12, 10) loses its east
    neighbour, the moved wall, so both parts rewrite it; the merge keeps one."""
    gate, walls, scenario = _gate_scenario()
    carried = next(u for u in walls if _tile(u) == (13, 10))
    moves = _shifted([gate, carried], 0, 3)
    assert [_tile(u) for _p, u, _v in wall_run.scenario_gate_moves(scenario, 60, 60, moves).rewrites] == [(12, 10)]

    plan = _plan_moves(scenario, moves, gate_walls=True)

    assert plan.removals == []
    assert _fills(plan) == {(x, 10): 0 for x in range(8, 12)}
    assert [(_tile(u), v) for _p, u, v in plan.rewrites] == [((12, 10), 2), ((14, 10), 2)]
    assert plan.moved == [(1, carried, 2)]
    assert plan.wholesale


def test_radian_flip_is_set_only_when_the_plan_overwrites_the_last_radian_wall_value():
    def plan_for(rotation=_RADIAN_1, gaia=()):
        wall = _wall_at(10, 10, rotation=rotation)
        return _plan_moves(_walls_scenario(wall, gaia=gaia), [(wall, 20.5, 20.5)])

    flipping = plan_for()
    assert flipping.moved and flipping.radian_flip and flipping.wholesale
    # A GAIA wall keeps its radian index 2 (a tower already), so the file stays radian.
    kept = plan_for(gaia=(_wall_at(40, 40, rotation=2 * _RADIAN_1),))
    assert kept.moved and not kept.radian_flip and not kept.wholesale
    integer = plan_for(rotation=1.0)
    assert integer.moved and not integer.radian_flip


def test_scenario_move_plan_answers_none_without_a_scene_walk(monkeypatch):
    """A villager, a 0.1 wall nudge inside its tile, and a gate with the box off."""
    villager = _wall_at(40, 40, const=_VILLAGER)
    gate, walls, scenario = _gate_scenario([villager])
    wall = walls[3]
    assert _tile(wall) == (12, 10)

    def walk(*_args, **_kwargs):
        raise AssertionError("planned through a scene walk")

    monkeypatch.setattr(wall_run, "wall_scene", walk)
    assert _plan_moves(scenario, _shifted([villager], 5, 0), gate_walls=True) is None
    assert _plan_moves(scenario, _shifted([wall], 0.1, 0), gate_walls=True) is None
    assert _plan_moves(scenario, _shifted([gate], 5, 0), gate_walls=False) is None
    with pytest.raises(AssertionError, match="scene walk"):
        _plan_moves(scenario, _shifted([wall], 1, 0))


def test_move_plan_players_adds_each_carried_walls_owner():
    plan = wall_run.MovePlan(
        removals=[(0, _wall_at(13, 10))], fills=[], rewrites=[(5, _wall_at(12, 10), 2)], moved=[(3, _wall_at(9, 9), 2)]
    )
    assert wall_run.move_plan_players(plan) == {0, 3, 5}


# --- set_cliff_frame()'s guard ----------------------------------------
#
# set_wall_variant()'s sibling for the Cliff drag's join rewrites (GH #44),
# against the same model and fixture. A cliff is added first: the fixture
# has none.

_CLIFF = 264  # Cliff (Default) 01
_MARBLE_CLIFF = 2178


def _add_cliff(model, const: int = _CLIFF, frame: int = 19):
    from descape.edit_history import EditHistory

    model.begin_unit_edit([0])
    cliff = model.add(0, const, 30.5, 30.5, rotation=float(frame), initial_animation_frame=frame)
    model.commit_unit_edit("probe", EditHistory())
    return cliff


def test_set_cliff_frame_writes_rotation_and_animation_frame_together():
    from descape.edit_history import EditHistory

    _loaded, model = _open()
    cliff = _add_cliff(model)
    before = (cliff.unit_const, cliff.x, cliff.y, cliff.z, cliff.reference_id)

    model.begin_unit_edit([0], fields_only=True)
    model.set_cliff_frame(cliff, 2)
    model.commit_unit_edit("Place cliffs", EditHistory())

    assert (cliff.rotation, cliff.initial_animation_frame) == (2.0, 2)
    assert type(cliff.initial_animation_frame) is int
    assert (cliff.unit_const, cliff.x, cliff.y, cliff.z, cliff.reference_id) == before


def test_set_cliff_frame_undoes_back_to_the_stored_frame():
    from descape.edit_history import EditHistory

    _loaded, model = _open()
    history = EditHistory()
    cliff = _add_cliff(model, frame=19)

    model.begin_unit_edit([0], fields_only=True)
    model.set_cliff_frame(cliff, 3)
    model.commit_unit_edit("Place cliffs", history)
    assert (cliff.rotation, cliff.initial_animation_frame) == (3.0, 3)

    history.undo([], None, None, model)
    assert (cliff.rotation, cliff.initial_animation_frame) == (19.0, 19)


def test_set_cliff_frame_refuses_a_non_cliff_const():
    loaded, model = _open()
    for unit in (_unit(loaded, _REF_WALL), _unit(loaded, 201)):
        before = (unit.rotation, unit.initial_animation_frame)
        with pytest.raises(ValueError, match="not a cliff const"):
            model.set_cliff_frame(unit, 2)
        assert (unit.rotation, unit.initial_animation_frame) == before


@pytest.mark.parametrize("frame", [-1, 24, 25, 99])
def test_set_cliff_frame_refuses_an_out_of_range_frame(frame):
    """24 too: with no install the frame count falls back to the .dat's 25,
    and no cliff family has a frame 24."""
    _loaded, model = _open()
    cliff = _add_cliff(model)
    with pytest.raises(ValueError, match=r"must be 0\.\.23"):
        model.set_cliff_frame(cliff, frame)
    assert (cliff.rotation, cliff.initial_animation_frame) == (19.0, 19)


def test_set_cliff_frame_is_bounded_by_the_const_s_own_frame_count(monkeypatch):
    """Marble has 23 frames, so its last legal frame is 22."""
    from descape import cliff_catalog
    from descape.edit_history import EditHistory

    monkeypatch.setattr(cliff_catalog, "frame_count", lambda const: 23 if const == _MARBLE_CLIFF else 24)
    _loaded, model = _open()
    marble = _add_cliff(model, _MARBLE_CLIFF, frame=1)
    with pytest.raises(ValueError, match=r"must be 0\.\.22"):
        model.set_cliff_frame(marble, 23)
    model.begin_unit_edit([0], fields_only=True)
    model.set_cliff_frame(marble, 22)
    model.commit_unit_edit("Place cliffs", EditHistory())
    assert marble.rotation == 22.0
