"""descape/unit_shape.py (GH #365): Unit Line and Unit Shape's latch,
box-local lattice and candidate plan. Qt-free, install-free."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from descape import asset_source
from descape.place_rotation import PlaceRotation, rotation_for
from descape.render_units import _span_start, occupied_tiles_for
from descape.shape_tools import RIGHT_TRIANGLE, TRIANGLE, rect_perimeter_tiles
from descape.terrain_palette import tile_span
from descape.unit_shape import (
    ELLIPSE,
    LINE,
    RECTANGLE,
    UNITS,
    WALL,
    cell_block,
    latch_for,
    lattice_cells,
    place_cells,
    plan_candidates,
    refusal,
)
from testkit.fakes import FakeScenario, SyntheticTile

VILLAGER = 83  # 1x1
HOUSE = 70  # 2x2
FARM = 50  # 3x3
CASTLE = 82  # 4x4
STONE_WALL = 117
PALISADE = 72
STONE_GATE = 64
CLIFF = 264
BIG = 1000


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _latch(consts, kind=RECTANGLE, *, filled=False, spacing=0, randomize=False, rules=True, owner=1):
    return latch_for(
        consts, owner, kind, filled=filled, spacing=spacing, randomize=randomize, rules=rules, pending=PlaceRotation()
    )


def _footprints(latch, cells, rng=None):
    """Every candidate's footprint tiles, as render reads them back."""
    out = []
    for const, x, y in plan_candidates(latch, cells, rng or random.Random(1)):
        w, h = tile_span(const, (1, 1))
        x0, y0 = _span_start(x, w), _span_start(y, h)
        out.append({(x0 + i, y0 + j) for i in range(w) for j in range(h)})
    return out


QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
PICKS = ((VILLAGER,), (HOUSE,), (FARM,), (CASTLE,), (VILLAGER, CASTLE))


def test_every_footprint_lies_inside_the_dragged_box_in_all_four_quadrants():
    bad = []
    a = (500, 500)
    for pick in PICKS:
        for kind in (LINE, RECTANGLE, ELLIPSE, TRIANGLE, RIGHT_TRIANGLE):
            for filled in (False, True):
                for spacing in (0, 1, 3):
                    latch = _latch(pick, kind, filled=filled, spacing=spacing)
                    for sx, sy in QUADRANTS:
                        for bw in range(latch.W, latch.W + 13, 3):
                            for bh in range(latch.H, latch.H + 13, 4):
                                e = (a[0] + sx * (bw - 1), a[1] + sy * (bh - 1))
                                box_x = range(min(a[0], e[0]), max(a[0], e[0]) + 1)
                                box_y = range(min(a[1], e[1]), max(a[1], e[1]) + 1)
                                cells = lattice_cells(latch, a, e, False, BIG, BIG)
                                for fp in _footprints(latch, cells):
                                    if any(x not in box_x or y not in box_y for x, y in fp):
                                        bad.append((pick, kind, filled, spacing, (sx, sy), bw, bh))
                                        break
    assert bad == []


def test_footprints_never_overlap_and_the_anchor_cell_holds_the_anchor():
    for pick in PICKS:
        latch = _latch(pick, RECTANGLE, filled=True)
        for sx, sy in QUADRANTS:
            a = (50, 50)
            cells = lattice_cells(latch, a, (a[0] + sx * 13, a[1] + sy * 9), False, BIG, BIG)
            # Local (0, 0) rasterizes first and is the anchor's cell.
            assert a in cell_block(latch, cells[0], BIG, BIG)
            fps = _footprints(latch, cells)
            assert sum(len(fp) for fp in fps) == len(set().union(*fps))


def test_a_box_shorter_than_the_span_still_places_one_cell_on_the_anchor_corner():
    latch = _latch((CASTLE,), RECTANGLE)
    assert lattice_cells(latch, (20, 20), (21, 20), False, BIG, BIG) == [(20, 20)]
    # Dragged up-left, the cell extends from the anchor in the drag direction.
    assert lattice_cells(latch, (20, 20), (19, 18), False, BIG, BIG) == [(17, 17)]
    # A zero-length axis counts as +1.
    assert lattice_cells(latch, (20, 20), (20, 20), False, BIG, BIG) == [(20, 20)]


def test_spacing_leaves_exact_gaps_between_cells():
    latch = _latch((HOUSE,), LINE, spacing=1)
    cells = lattice_cells(latch, (10, 10), (20, 10), False, BIG, BIG)
    # Cells of 2 plus 1 gap: 11 tiles hold 4 cells (2+1+2+1+2+1+2).
    assert cells == [(10, 10), (13, 10), (16, 10), (19, 10)]
    back = lattice_cells(latch, (20, 10), (10, 10), False, BIG, BIG)
    assert back == [(19, 10), (16, 10), (13, 10), (10, 10)]


def test_a_spacing_zero_filled_farm_rectangle_tiles_edge_to_edge():
    latch = _latch((FARM,), RECTANGLE, filled=True)
    cells = lattice_cells(latch, (0, 0), (8, 5), False, BIG, BIG)
    assert sorted(cells) == [(x, y) for x in (0, 3, 6) for y in (0, 3)]


def test_centring_matches_the_footprint_read_back():
    # A 1x1 in a 4x4 cell sits at (W - 1) // 2 in, and a 2x2 at (W - 2) // 2.
    latch = _latch((VILLAGER, CASTLE, HOUSE))
    for const, x, y in plan_candidates(latch, [(10, 20)] * 30, random.Random(3)):
        w, h = tile_span(const, (1, 1))
        assert (_span_start(x, w), _span_start(y, h)) == (10 + (4 - w) // 2, 20 + (4 - h) // 2)
        footprint = occupied_tiles_for(const, x, y, BIG, BIG)
        assert footprint and all(10 <= tx < 14 and 20 <= ty < 24 for tx, ty in footprint)


def test_a_single_const_pick_draws_nothing_from_the_rng():
    rng = random.Random(9)
    plan_candidates(_latch((HOUSE,)), [(0, 0), (2, 0)], rng)
    assert rng.random() == random.Random(9).random()


def test_triangle_orientation_follows_the_drag_direction():
    latch = _latch((VILLAGER,), RIGHT_TRIANGLE)
    for sx, sy in QUADRANTS:
        a = (50, 50)
        e = (a[0] + sx * 6, a[1] + sy * 6)
        cells = set(lattice_cells(latch, a, e, False, BIG, BIG))
        # The right angle is at the anchor; the far corner is empty.
        assert {(a[0] + sx * i, a[1]) for i in range(7)} <= cells
        assert {(a[0], a[1] + sy * i) for i in range(7)} <= cells
        assert e not in cells
    iso = _latch((VILLAGER,), TRIANGLE)
    cells = set(lattice_cells(iso, (50, 50), (42, 46), False, BIG, BIG))
    assert {(x, 50) for x in range(42, 51)} <= cells  # the base on the anchor's row
    assert [c for c in cells if c[1] == 46] == [(46, 46)]  # the apex, mid-way along


def test_shift_snaps_the_cell_delta_not_the_tile_delta():
    latch = _latch((HOUSE,), RECTANGLE, spacing=1)
    # 3 x 2 cells unsnapped; Shift squares to 3 x 3 cells, growing the box downward.
    cells = lattice_cells(latch, (0, 0), (7, 4), True, BIG, BIG)
    assert sorted({c[1] for c in cells}) == [0, 3, 6]
    assert sorted({c[0] for c in cells}) == [0, 3, 6]


def test_off_map_cells_drop_and_partly_off_cells_stay():
    latch = _latch((FARM,), RECTANGLE, filled=True)
    cells = lattice_cells(latch, (4, 4), (-8, 4), False, 30, 30)
    # Cells at x 2, -1, -4, -7: -1 is partly off, -4 and -7 wholly.
    assert cells == [(2, 4), (-1, 4)]
    assert cell_block(latch, (-1, 4), 30, 30) == [(x, y) for y in (4, 5, 6) for x in (0, 1)]


def test_a_wall_pick_takes_the_wall_branch_with_tile_cells():
    latch = _latch((STONE_WALL,), RECTANGLE, filled=True, spacing=4)
    assert (latch.branch, latch.W, latch.H, latch.spacing, latch.filled) == (WALL, 1, 1, 0, False)
    cells = lattice_cells(latch, (12, 9), (4, 3), False, BIG, BIG)
    assert sorted(cells) == sorted(rect_perimeter_tiles(4, 3, 12, 9, BIG, BIG))


def test_a_wall_line_is_place_units_run():
    from descape.shape_tools import snap_wall_delta, wall_path_tiles

    latch = _latch((STONE_WALL,), LINE)
    assert lattice_cells(latch, (5, 5), (12, 8), False, BIG, BIG) == wall_path_tiles(5, 5, 12, 8, BIG, BIG)
    dx, dy = snap_wall_delta(7, 3)
    assert lattice_cells(latch, (5, 5), (12, 8), True, BIG, BIG) == wall_path_tiles(5, 5, 5 + dx, 5 + dy, BIG, BIG)


@pytest.mark.parametrize(
    ("kind", "corners", "legs"),
    [
        # Base (10..20, 10), apex (15, 15): a 3-tower cluster at each acute base corner, one at the apex.
        (TRIANGLE, {(10, 10), (11, 10), (11, 11), (19, 10), (20, 10), (19, 11), (15, 15)}, {3, 4}),
        # Right angle at (10, 10), one tower; a 3-tower cluster at each acute corner.
        (RIGHT_TRIANGLE, {(10, 10), (19, 10), (20, 10), (19, 11), (10, 19), (11, 19), (10, 20)}, {4}),
    ],
)
def test_shift_triangle_walls_put_towers_at_the_corners_and_diagonal_walls_on_the_legs(kind, corners, legs):
    """The AoE2:DE check's expectation (TASK-608 Owed 1): what plan_wall_run()
    writes on a Shift triangle's outline. Towers (variant 2) stand exactly on
    the corner tiles, the axis-aligned edges are straight runs (0 along x,
    1 along y) and the 45-degree legs are diagonal walls (3 or 4), never a
    tower chain."""
    from descape import wall_run

    latch = _latch((STONE_WALL,), kind)
    cells = lattice_cells(latch, (10, 10), (20, 15), True, BIG, BIG)
    plan = wall_run.plan_wall_run(cells, unit_const=STONE_WALL, existing_tiles=set(), existing_walls=[])
    variants = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    assert set(variants) == set(cells)
    assert {tile for tile, v in variants.items() if v == 2} == corners
    for (x, y), v in variants.items():
        if (x, y) in corners:
            continue
        if y == 10:
            assert v == 0, (x, y)
        elif x == 10 and kind == RIGHT_TRIANGLE:
            assert v == 1, (x, y)
        else:
            assert v in legs, (x, y, v)


@pytest.mark.parametrize(
    ("kind", "end", "base_towers"),
    [
        # Two corners plus one junction where each leg leaves the base.
        (TRIANGLE, (30, 13), 4),
        # The right-angle end has no junction: its leg is the vertical side.
        (RIGHT_TRIANGLE, (24, 14), 3),
    ],
)
def test_a_flat_triangle_wall_without_shift_keeps_its_base_whole_and_one_tile_thick(kind, end, base_towers):
    """TASK-609: a shallow leg used to run beside the base as a tower band and
    cost the base its corners. Now the base row is whole, its only towers
    are its corners and the junctions, and the row beside it holds just the
    two tiles where the sides leave it."""
    from descape import wall_run

    latch = _latch((STONE_WALL,), kind)
    cells = lattice_cells(latch, (10, 10), end, False, BIG, BIG)
    plan = wall_run.plan_wall_run(cells, unit_const=STONE_WALL, existing_tiles=set(), existing_walls=[])
    variants = {(int(n.x), int(n.y)): n.variant for n in plan.nodes}
    base = {(x, 10) for x in range(10, end[0] + 1)}
    assert base <= set(variants)
    assert {(10, 10), (end[0], 10)} <= {tile for tile in base if variants[tile] == 2}
    assert sum(variants[tile] == 2 for tile in base) == base_towers
    assert len([tile for tile in variants if tile[1] == 11]) == 2


def test_a_unit_pick_takes_the_largest_span_per_axis():
    latch = _latch((VILLAGER, CASTLE, VILLAGER), spacing=99)
    assert (latch.branch, latch.consts, latch.W, latch.H, latch.spacing) == (UNITS, (VILLAGER, CASTLE), 4, 4, 20)


@pytest.mark.parametrize(
    ("pick", "reason"),
    [
        ((), "pick an object"),
        ((STONE_GATE,), "gates"),
        ((HOUSE, STONE_GATE), "gates"),
        ((CLIFF,), "cliffs"),
        ((STONE_WALL, HOUSE), "walls alone"),
        ((STONE_WALL, PALISADE), "one wall"),
    ],
)
def test_refused_picks_name_their_reason(pick, reason):
    assert reason in refusal(pick)
    with pytest.raises(ValueError):
        _latch(pick)


def test_accepted_picks_have_no_refusal():
    assert refusal((STONE_WALL,)) is None
    assert refusal((STONE_WALL, STONE_WALL)) is None
    assert refusal((HOUSE, VILLAGER)) is None


# --- placing the unit branch (unit_brush.place_candidate()) ------------------------------

OAK = 349  # cyclable, 1x1
DEER = 65  # GAIA-only, ANGLE
WATER_DEEP, GRASS = 22, 0
SIZE = 30


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    garrisoned_in_id: int = -1


def _scenario(terrain=None) -> FakeScenario:
    terrain = terrain or {}
    tiles = [SyntheticTile(x, y, 0, terrain.get((x, y), GRASS)) for y in range(SIZE) for x in range(SIZE)]
    return FakeScenario(SIZE, SIZE, tiles, [[] for _ in range(9)])


def _adder(scenario):
    def add(spec):
        unit = _Unit(spec.x, spec.y, spec.unit_const, spec.rotation)
        scenario.unit_manager.units[spec.player].append(unit)
        return unit

    return add


def _place(scenario, latch, anchor, end, seed=5):
    cells = lattice_cells(latch, anchor, end, False, SIZE, SIZE)
    return place_cells(scenario, latch, cells, _adder(scenario), random.Random(seed))


def test_rules_refuse_with_one_reason_and_free_placement_lifts_them():
    water = {(x, y): WATER_DEEP for x in range(5, 8) for y in range(5, 8)}
    scenario = _scenario(water)
    stroke = _place(scenario, _latch((VILLAGER,), RECTANGLE, filled=True), (4, 4), (9, 9))
    assert len(stroke.placed) == 36 - 9 and stroke.refused == 9
    assert stroke.refusal_reason == "Villager Male can't stand on Water Deep"
    free = _scenario(water)
    stroke = _place(free, _latch((VILLAGER,), RECTANGLE, filled=True, rules=False), (4, 4), (9, 9))
    assert len(stroke.placed) == 36 and stroke.refused == 0 and stroke.refusal_reason is None


def test_rules_see_units_placed_earlier_in_the_same_shape():
    scenario = _scenario()
    scenario.unit_manager.units[2].append(_Unit(6, 6, HOUSE))  # tiles 5..6 x 5..6
    stroke = _place(scenario, _latch((HOUSE,), RECTANGLE, filled=True), (1, 1), (10, 10))
    assert stroke.refused == 1  # only the cell at (5, 5), which the House fills
    tiles = [occupied_tiles_for(u.unit_const, u.x, u.y, SIZE, SIZE) for _p, u in stroke.placed]
    flat = [t for fp in tiles for t in fp]
    assert len(flat) == len(set(flat))


def test_rotation_is_random_with_random_ticked_else_the_pending_one():
    scenario = _scenario()
    stroke = _place(scenario, _latch((OAK,), RECTANGLE, filled=True, randomize=True), (0, 0), (9, 9))
    assert len({u.rotation for _p, u in stroke.placed}) > 3
    pending = PlaceRotation(facing=1.2, variant=5, variant_const=OAK)
    latch = latch_for((OAK, VILLAGER), 1, RECTANGLE, filled=True, spacing=0, randomize=False, rules=True,
                      pending=pending)
    stroke = _place(_scenario(), latch, (0, 0), (9, 9))
    for player, unit in stroke.placed:
        assert unit.rotation == rotation_for(pending, unit.unit_const, player)


def test_a_gaia_only_const_goes_to_gaia_and_off_map_cells_drop():
    scenario = _scenario()
    stroke = _place(scenario, _latch((DEER, VILLAGER), LINE), (25, 3), (40, 3))
    assert {p for p, u in stroke.placed if u.unit_const == DEER} == {0}
    assert {p for p, u in stroke.placed if u.unit_const == VILLAGER} == {1}
    assert stroke.gaia_only == sum(1 for _p, u in stroke.placed if u.unit_const == DEER)
    assert sorted(int(u.x) for _p, u in stroke.placed) == [25, 26, 27, 28, 29]


def test_a_footprint_running_off_the_map_is_dropped():
    scenario = _scenario()
    stroke = _place(scenario, _latch((FARM,), LINE), (25, 3), (31, 3))
    # Cells at x 25 and 28: the second runs off a 30-wide map.
    assert [(u.x, u.y) for _p, u in stroke.placed] == [(26.5, 4.5)]


def test_place_cells_refuses_the_wall_branch():
    with pytest.raises(ValueError):
        place_cells(_scenario(), _latch((STONE_WALL,)), [(1, 1)], _adder(_scenario()), random.Random(1))
