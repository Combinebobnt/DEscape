"""region_transform: a pasted region turned or flipped as a block (GH #197).
Pure: RegionBlocks built by hand, no Qt, no scenario.

The claims: tiles land where TRANSFORMS on the padded square sends them, a
unit's footprint image is the transform of its footprint, every unit field
follows the mirror rules (facing through facing_image(), walls through
wall_index_image() in an integer file only, everything else verbatim), the
four direction names mean what the screen shows, and the doubled pivot brings
four quarter turns back to the first anchor.
"""

from __future__ import annotations

import itertools
import math

import pytest

from descape import bridge_piece_frames, gate_orientation, iso_geometry, render_units, unit_rotation
from descape.mirror_tools import AXIS_SWAPPING, TRANSFORMS, compose, facing_image, invert
from descape.region_clipboard import RegionBlock, RegionUnit
from descape.region_transform import (
    FLIP_WEST_NORTH,
    FLIP_WEST_SOUTH,
    ROTATE_CCW,
    ROTATE_CW,
    WRITE_ANGLE,
    WRITE_WALL,
    FirstUnit,
    anchor_for,
    image_on_map,
    lands_on_map,
    pivot2_of,
    plan_in_place,
    transform_block,
)
from descape.terrain_palette import tile_span

ELEMENTS = list(TRANSFORMS)
ARCHER = 4  # 1x1, ANGLE (16 facings)
HOUSE = 70  # 2x2, INERT
CASTLE = 82  # 4x4, INERT
GATE_NE = 64  # (4, 1)
GATE_DIAGONAL = 659  # 4x4, sparse diagonal footprint
OAK = 349  # 1x1, VARIANT
WALL = 117  # 1x1, five wall shapes
CLIFF = 264  # 3x3 cliff
BRIDGE = 605  # (2, 3), an `a` bridge's top piece
MOLE = 2421  # (3, 1), no orientation sibling
WOODEN_2X2 = 911  # (2, 2), a wooden bridge with no `b` art
_RADIAN_3 = 3 * 2 * math.pi / 5


def _unit(const: int, dx: float, dy: float, rotation: float = 0.0, garrison_slot: int = -1, player: int = 1):
    return RegionUnit(
        player=player,
        unit_const=const,
        dx=dx,
        dy=dy,
        z=0.5,
        rotation=rotation,
        status=2,
        initial_animation_frame=3,
        caption_string_id=-1,
        caption_string="cap",
        garrison_slot=garrison_slot,
        capture_flag=1,
    )


def _block(w: int = 5, h: int = 3, units=()) -> RegionBlock:
    """A w x h block with a distinct value per tile in every layer."""
    count = w * h
    return RegionBlock(
        width=w,
        height=h,
        terrain_ids=tuple(range(count)),
        elevations=tuple(100 + i for i in range(count)),
        layers=tuple(200 + i for i in range(count)),
        units=tuple(units),
    )


def _low_corner(element: str, w: int, h: int) -> tuple[int, int]:
    n = max(w, h)
    fn = TRANSFORMS[element]
    corners = [fn(x, y, n) for x in range(w) for y in range(h)]
    return min(c[0] for c in corners), min(c[1] for c in corners)


@pytest.mark.parametrize("element", ELEMENTS)
def test_every_tile_lands_where_the_padded_square_sends_it(element):
    block = _block()
    result = transform_block(block, element, False).block
    swapped = element in AXIS_SWAPPING
    assert (result.width, result.height) == ((3, 5) if swapped else (5, 3))
    low_x, low_y = _low_corner(element, 5, 3)
    for y in range(3):
        for x in range(5):
            ix, iy = TRANSFORMS[element](x, y, 5)
            dst = (iy - low_y) * result.width + (ix - low_x)
            src = y * 5 + x
            assert result.terrain_ids[dst] == block.terrain_ids[src]
            assert result.elevations[dst] == block.elevations[src]
            assert result.layers[dst] == block.layers[src]


def test_the_identity_returns_the_same_block():
    block = _block(units=[_unit(ARCHER, 1.5, 1.5)])
    result = transform_block(block, "id", False)
    assert result.block is block
    assert result.kept_orientation == 0


@pytest.mark.parametrize("element", ["x", "rot60", "a6"])
def test_anything_but_a_d4_element_is_refused(element):
    # AGENTS.md names transform_block() as the region turn's D4-only guard.
    with pytest.raises(ValueError):
        transform_block(_block(units=[_unit(ARCHER, 1.5, 1.5)]), element, False)


@pytest.mark.parametrize(("first", "second"), list(itertools.product(ELEMENTS, ELEMENTS)))
def test_tile_arrays_obey_the_group_law(first, second):
    block = _block()
    twice = transform_block(transform_block(block, first, False).block, second, False).block
    once = transform_block(block, compose(second, first), False).block
    assert (twice.width, twice.height) == (once.width, once.height)
    assert twice.terrain_ids == once.terrain_ids
    assert twice.elevations == once.elevations


# Units whose footprints sit wholly inside a 9 x 6 block (odd width, even height).
_FOOTPRINT_UNITS = {
    "mid-tile archer": _unit(ARCHER, 2.3, 1.7),
    "tile-boundary archer": _unit(ARCHER, 6.0, 4.0),
    "2x2 house": _unit(HOUSE, *render_units.span_anchor(1, 3, 2, 2)),
    "4x4 castle": _unit(CASTLE, *render_units.span_anchor(4, 1, 4, 4)),
    "NE gate": _unit(GATE_NE, *render_units.span_anchor(1, 0, 4, 1)),
    "diagonal gate": _unit(GATE_DIAGONAL, *render_units.span_anchor(5, 2, 4, 4)),
}


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("name", list(_FOOTPRINT_UNITS))
def test_a_units_footprint_image_is_the_transform_of_its_footprint(element, name):
    w, h = 9, 6
    unit = _FOOTPRINT_UNITS[name]
    source_tiles = render_units.occupied_tiles_for(unit.unit_const, unit.dx, unit.dy, w, h)
    assert source_tiles, "the source footprint must lie inside the block"
    (image,) = transform_block(_block(w, h, [unit]), element, False).block.units
    n = max(w, h)
    low_x, low_y = _low_corner(element, w, h)
    expected = {
        (ix - low_x, iy - low_y) for ix, iy in (TRANSFORMS[element](x, y, n) for x, y in source_tiles)
    }
    new_w, new_h = (h, w) if element in AXIS_SWAPPING else (w, h)
    got = render_units.occupied_tiles_for(image.unit_const, image.dx, image.dy, new_w, new_h)
    assert set(got) == expected


@pytest.mark.parametrize("element", ELEMENTS)
def test_unit_fields_follow_the_mirror_rules(element):
    units = [
        _unit(ARCHER, 1.5, 1.5, rotation=1.25),
        _unit(OAK, 2.5, 1.5, rotation=37.0),
        _unit(HOUSE, *render_units.span_anchor(3, 0, 2, 2), rotation=7.0),
    ]
    archer, oak, house = transform_block(_block(units=units), element, False).block.units
    assert archer.rotation == facing_image(1.25, element, unit_rotation.angle_count_for(ARCHER))
    assert oak.rotation == 37.0
    assert house.rotation == 7.0
    for before, after in zip(units, (archer, oak, house), strict=True):
        assert (after.player, after.z, after.status, after.initial_animation_frame) == (1, 0.5, 2, 3)
        assert (after.caption_string_id, after.caption_string, after.capture_flag) == (-1, "cap", 1)
        assert after.garrison_slot == before.garrison_slot


@pytest.mark.parametrize(
    ("element", "expected"),
    [("r", (1.0, 0.0, 2.0, 4.0, 3.0)), ("mx", (0.0, 1.0, 2.0, 4.0, 3.0)), ("a", (1.0, 0.0, 2.0, 3.0, 4.0))],
)
def test_wall_shapes_permute_in_an_integer_file_only(element, expected):
    walls = [_unit(WALL, 0.5 + i, 0.5, rotation=float(i)) for i in range(5)]
    turned = transform_block(_block(units=walls), element, False).block.units
    assert tuple(u.rotation for u in turned) == expected
    radian = [_unit(WALL, 0.5, 0.5, rotation=_RADIAN_3)]
    (kept,) = transform_block(_block(units=radian), element, True).block.units
    assert kept.rotation == _RADIAN_3


def test_unit_order_and_garrison_slots_survive_and_an_occupant_stays_on_its_holder():
    w, h = 9, 6
    for element in ELEMENTS:
        units = [
            _unit(ARCHER, 7.5, 4.5),
            _unit(CASTLE, *render_units.span_anchor(2, 1, 4, 4)),
            _unit(ARCHER, 4.0, 3.0, garrison_slot=1),  # stored on its holder's point
            _unit(OAK, 0.5, 0.5),
        ]
        turned = transform_block(_block(w, h, units), element, False).block.units
        assert [u.unit_const for u in turned] == [ARCHER, CASTLE, ARCHER, OAK], element
        assert [u.garrison_slot for u in turned] == [-1, -1, 1, -1], element
        assert (turned[2].dx, turned[2].dy) == (turned[1].dx, turned[1].dy), element


def test_kept_orientation_counts_cliffs_always_and_unsquare_spans_under_a_swap():
    units = [
        _unit(CLIFF, *render_units.span_anchor(0, 0, 3, 3)),
        _unit(BRIDGE, *render_units.span_anchor(5, 0, 2, 3)),
        _unit(GATE_NE, *render_units.span_anchor(0, 4, 4, 1)),
        _unit(ARCHER, 8.5, 5.5),
        _unit(MOLE, *render_units.span_anchor(4, 3, 3, 1)),
        _unit(DE_END, *render_units.span_anchor(8, 0, 1, 2)),
    ]
    block = _block(9, 6, units)
    assert transform_block(block, "r", False).kept_orientation == 2  # the cliff and the mole, not the bridges
    assert transform_block(block, "mx", False).kept_orientation == 1


DE_END, DE_MIDDLE = 1993, 1996  # cd end and middle, (1, 2); an end's frame is its run sense (TASK-705)


@pytest.mark.parametrize("element", [e for e in TRANSFORMS if e != "id"])
def test_a_de_bridge_run_keeps_its_ends_facing_its_middles(element):
    """A run along y: 1993 frame 1 (inward +y) at the low end, two 1996
    middles, 1993 frame 0 (inward -y) at the high end."""
    units = [
        _unit(DE_END, *render_units.span_anchor(0, 0, 1, 2), rotation=1.0),
        _unit(DE_MIDDLE, *render_units.span_anchor(0, 2, 1, 2), rotation=0.0),
        _unit(DE_MIDDLE, *render_units.span_anchor(0, 4, 1, 2), rotation=1.0),
        _unit(DE_END, *render_units.span_anchor(0, 6, 1, 2), rotation=0.0),
    ]
    result = transform_block(_block(1, 8, units), element, False)
    assert result.kept_orientation == 0
    low, mid_a, mid_b, high = result.block.units
    middle = ((mid_a.dx + mid_b.dx) / 2, (mid_a.dy + mid_b.dy) / 2)
    for source, image in ((units[1], mid_a), (units[2], mid_b)):
        row = bridge_piece_frames.frames(image.unit_const)
        assert row[int(image.rotation)] == bridge_piece_frames.frames(DE_MIDDLE)[int(source.rotation)]
    for end in (low, high):
        _kind, (vx, vy) = bridge_piece_frames.frames(end.unit_const)[int(end.rotation)]
        to_middle = (middle[0] - end.dx, middle[1] - end.dy)
        assert vx * to_middle[0] + vy * to_middle[1] > 0, (element, end)


def test_a_2x2_wooden_bridge_is_kept_under_an_axis_swap_only():
    block = _block(4, 4, [_unit(WOODEN_2X2, *render_units.span_anchor(0, 0, 2, 2)), _unit(904, 3.5, 3.5)])
    assert transform_block(block, "r", False).kept_orientation == 1  # 911, not the 1x1 904
    assert transform_block(block, "r2", False).kept_orientation == 0


def test_rotate_cw_turns_a_facing_the_way_rotate_step_does():
    """rotate_step(+1) is clockwise on screen (AoE2 rotation grows clockwise),
    so ROTATE_CW must add a quarter turn to a facing, not take one away."""
    assert facing_image(0.0, ROTATE_CW) == pytest.approx(math.pi / 2)
    assert unit_rotation.rotate_step(0.0, 4, +1) == pytest.approx(math.pi / 2)
    assert facing_image(0.0, ROTATE_CCW) == pytest.approx(3 * math.pi / 2)
    assert compose(ROTATE_CW, ROTATE_CCW) == "id"


def test_the_flips_swap_the_compass_corners_they_are_named_for():
    """Through tile_screen_origin() itself, on a square: West is the leftmost
    corner, North the topmost, South the bottommost, East the rightmost.
    FLIP_WEST_NORTH swaps West with North (and South with East),
    FLIP_WEST_SOUTH West with South (and North with East). In top-down Flat
    coordinates they are (n-1-x, y) and (x, n-1-y). Neither swaps the tile
    axes, so a flip keeps a selection's rectangle (TASK-613)."""
    assert FLIP_WEST_NORTH not in AXIS_SWAPPING and FLIP_WEST_SOUTH not in AXIS_SWAPPING
    n = 9
    proj = iso_geometry.canvas_size_and_origin(n, n, 32, 0, 0)
    corners = [(x, y) for x in (0, n - 1) for y in (0, n - 1)]
    screen = {c: iso_geometry.tile_screen_origin(*c, 0, proj) for c in corners}
    west = min(corners, key=lambda c: screen[c][0])
    east = max(corners, key=lambda c: screen[c][0])
    north = min(corners, key=lambda c: screen[c][1])
    south = max(corners, key=lambda c: screen[c][1])
    wn, ws = TRANSFORMS[FLIP_WEST_NORTH], TRANSFORMS[FLIP_WEST_SOUTH]
    assert (wn(*west, n), wn(*north, n), wn(*south, n), wn(*east, n)) == (north, west, east, south)
    assert (ws(*west, n), ws(*south, n), ws(*north, n), ws(*east, n)) == (south, west, east, north)
    for x, y in [(0, 0), (2, 7), (8, 3), (5, 5), (1, 0)]:
        assert wn(x, y, n) == (n - 1 - x, y), (x, y)
        assert ws(x, y, n) == (x, n - 1 - y), (x, y)


def test_a_flip_keeps_a_non_square_rect_and_mirrors_inside_it_where_a_turn_does_not():
    """The user's symptom (TASK-613): a screen-space flip moved a non-square
    selection to where a quarter turn puts it. An asymmetric L of archers in
    a 2 x 3 rect mirrors inside the same rect; a 2 x 3 span keeps no
    orientation it would have lost under an axis swap."""
    rect = (40, 30, 42, 33)
    spots = [(40.5, 30.5), (40.5, 31.5), (40.5, 32.5), (41.5, 32.5)]
    firsts = [FirstUnit(i, ARCHER, x, y, 0.0) for i, (x, y) in enumerate(spots)]
    expected = {
        FLIP_WEST_NORTH: [(41.5, 30.5), (41.5, 31.5), (41.5, 32.5), (40.5, 32.5)],
        FLIP_WEST_SOUTH: [(40.5, 32.5), (40.5, 31.5), (40.5, 30.5), (41.5, 30.5)],
    }
    turned = plan_in_place(firsts, ROTATE_CW, rect, False)
    assert turned.rect != rect
    for element, want in expected.items():
        plan = plan_in_place(firsts, element, rect, False)
        assert plan.rect == rect, element
        assert [(image.x, image.y) for image in plan.images] == want, element
        screen = plan_in_place(firsts, {FLIP_WEST_NORTH: "a", FLIP_WEST_SOUTH: "d"}[element], rect, False)
        assert screen.rect == turned.rect, "the old screen flip landed where a quarter turn does"
        bridge = [FirstUnit("bridge", BRIDGE, *render_units.span_anchor(40, 30, 2, 3), 0.0)]
        assert plan_in_place(bridge, element, rect, False).kept_orientation == 0, element


def test_four_turns_of_an_odd_parity_block_return_the_first_anchor_even_with_moves_between():
    block = _block(5, 2)
    anchor = (40, 17)
    pivot = pivot2_of(anchor, block)
    element = "id"
    for _ in range(4):
        element = compose(ROTATE_CW, element)
        turned = transform_block(block, element, False).block
        anchor = anchor_for(pivot, turned)
    assert element == "id"
    assert anchor == (40, 17)

    pivot = pivot2_of((40, 17), block)
    for dx, dy in [(3, -1), (0, 0), (-2, 5), (1, 1)]:
        element = compose(ROTATE_CCW, element)
        pivot = (pivot[0] + 2 * dx, pivot[1] + 2 * dy)
        anchor = anchor_for(pivot, transform_block(block, element, False).block)
    assert element == "id"
    assert anchor == (40 + 2, 17 + 5)


def test_lands_on_map_is_any_tile_on_the_map():
    block = _block(20, 2)
    assert lands_on_map(block, (119, 60), 120, 120)
    assert not lands_on_map(block, (128, 51), 120, 120)
    assert not lands_on_map(block, (-20, 0), 120, 120)
    assert lands_on_map(block, (-19, -1), 120, 120)


# --- plan_in_place(): the same images for units that already exist (TASK-256) ---

BARRACKS = 12  # 3x3, INERT
VILLAGER = 83  # 1x1, ANGLE
_F32_41_3 = 41.29999923706055  # an off-grid float32 position
_RECT = (40, 30, 52, 42)  # 12 x 12, so a re-plan about the image rect is the same rect
_ODD_RECT = (40, 30, 51, 32)  # 11 x 2: odd width - height, the pivot sits on a half tile


def _gate_firsts() -> list[FirstUnit]:
    """One gate of each orientation group, each at a different orientation."""
    firsts = []
    for i, siblings in enumerate(sorted(gate_orientation.groups().values())):
        const = siblings[i % 4]
        span = tile_span(const, render_units.NON_BUILDING_SPAN)
        tile = (40 + i % 8, 30 + (i // 8) * 4)
        firsts.append(FirstUnit(("gate", i), const, *render_units.span_anchor(*tile, *span), 7.0))
    return firsts


def _firsts() -> list[FirstUnit]:
    """Every case the plan names: an off-grid facing, 2x2 and 3x3 buildings on
    both parities, a gate of each family, a wall at each index, a cliff, a tree,
    and an occupant on its holder's point (index 5 holds index 6)."""
    firsts = [
        FirstUnit("off-grid archer", ARCHER, _F32_41_3, 31.70000076293945, 1.25),
        FirstUnit("boundary archer", ARCHER, 46.0, 35.0, 0.39269909262657166),
        FirstUnit("house", HOUSE, *render_units.span_anchor(42, 32, 2, 2), 7.0),
        FirstUnit("off-parity house", HOUSE, 44.5, 37.5, 0.0),
        FirstUnit("off-parity barracks", BARRACKS, 48.0, 33.0, 7.0),
        FirstUnit("barracks", BARRACKS, *render_units.span_anchor(47, 38, 3, 3), 0.0),
        FirstUnit("occupant", VILLAGER, *render_units.span_anchor(47, 38, 3, 3), 2.0, holder=5),
        FirstUnit("cliff", CLIFF, *render_units.span_anchor(40, 39, 3, 3), 11.0),
        FirstUnit("oak", OAK, 51.5, 41.5, 37.0),
    ]
    firsts += [FirstUnit(("wall", i), WALL, 40.5 + i, 41.5, float(i)) for i in range(5)]
    return firsts + _gate_firsts()


def _applied(firsts, plan) -> list[FirstUnit]:
    """`firsts` with `plan` written over them, the way the viewer's setters would."""
    if not plan.images:
        return list(firsts)
    by_key = {image.key: image for image in plan.images}
    out = []
    for first in firsts:
        image = by_key[first.key]
        rotation = first.rotation if image.rotation_write is None else float(image.rotation_write[1])
        out.append(FirstUnit(first.key, image.unit_const, image.x, image.y, rotation, first.holder))
    return out


def _lift(firsts, rect, steps):
    """The lift's rule: every step plans from the first state and the element composed so far."""
    element = "id"
    state = list(firsts)
    for step in steps:
        element = compose(step, element)
        state = _applied(firsts, plan_in_place(firsts, element, rect, False))
    return element, state


def _fields(units) -> list[tuple]:
    return [(u.key, u.unit_const, u.x, u.y, u.rotation) for u in units]


def test_plan_in_place_plans_nothing_for_the_identity():
    plan = plan_in_place(_firsts(), "id", _RECT, False)
    assert (plan.images, plan.rect, plan.block, plan.kept_orientation) == ((), _RECT, None, 0)
    with pytest.raises(ValueError):
        plan_in_place(_firsts(), "rot60", _RECT, False)


@pytest.mark.parametrize("rect", [_RECT, _ODD_RECT], ids=["square", "odd"])
@pytest.mark.parametrize("element", ELEMENTS)
def test_a_lift_that_composes_back_to_the_identity_returns_every_first_field_exactly(element, rect):
    """Each element then its inverse, and four quarter turns, from one first
    state. Re-planning from each result instead drifts an off-grid facing by
    an ulp and an odd rect's anchor by a tile, so this is the rule that keeps
    a lift exact."""
    firsts = _firsts()
    for steps in ([element, invert(element)], [ROTATE_CW] * 4, [ROTATE_CW, FLIP_WEST_NORTH, ROTATE_CW, FLIP_WEST_NORTH]):
        final, state = _lift(firsts, rect, steps)
        assert final == "id", steps
        assert _fields(state) == _fields(firsts), steps


@pytest.mark.parametrize("element", ELEMENTS)
def test_on_a_square_rect_the_inverse_of_a_plan_returns_each_const_and_position(element):
    """Planning the inverse from the result about the same rect is the
    lattice's own round trip: every unit comes back to its const and tile,
    within an ulp of its float position and facing (the drift the lift's
    first-state rule above avoids). An off-parity building is left out: its
    image is re-anchored on parity, as a pasted turn's is."""
    firsts = [u for u in _firsts() if u.key != "occupant" and not str(u.key).startswith("off-parity")]
    once = _applied(firsts, plan_in_place(firsts, element, _RECT, False))
    back = _applied(once, plan_in_place(once, invert(element), _RECT, False))
    for before, after in zip(firsts, back, strict=True):
        assert (after.unit_const, int(after.x), int(after.y)) == (before.unit_const, int(before.x), int(before.y))
        assert (after.x, after.y, after.rotation) == pytest.approx((before.x, before.y, before.rotation), abs=1e-9)


@pytest.mark.parametrize("rect", [_RECT, _ODD_RECT], ids=["square", "odd"])
@pytest.mark.parametrize("element", ELEMENTS[1:])
def test_plan_in_place_puts_each_unit_where_the_pasted_region_turn_does(element, rect):
    """The same images as transform_block() of the rect copied as a block and
    re-pasted about its centre, so an in-place flip agrees with GH #197's."""
    firsts = _firsts()
    tx0, ty0, tx1, ty1 = rect
    w, h = tx1 - tx0, ty1 - ty0
    block = RegionBlock(
        w, h, tuple(range(w * h)), tuple(range(w * h)), tuple(range(w * h)),
        tuple(_unit(u.unit_const, u.x - tx0, u.y - ty0, u.rotation, u.holder) for u in firsts),
    )
    turned = transform_block(block, element, False).block
    ax, ay = anchor_for(pivot2_of((tx0, ty0), block), turned)
    plan = plan_in_place(firsts, element, rect, False, tiles=block)
    assert plan.rect == (ax, ay, ax + turned.width, ay + turned.height)
    assert plan.block.terrain_ids == turned.terrain_ids
    assert [(i.unit_const, i.x, i.y) for i in plan.images] == [(u.unit_const, ax + u.dx, ay + u.dy) for u in turned.units]
    for first, image, pasted in zip(firsts, plan.images, turned.units, strict=True):
        if image.rotation_write is None:
            assert pasted.rotation == first.rotation, first.key
        else:
            assert float(image.rotation_write[1]) == pasted.rotation, first.key


@pytest.mark.parametrize(
    ("element", "expected"),
    [("r", (1, 0, None, 4, 3)), ("mx", (None, None, None, 4, 3)), ("a", (1, 0, None, None, None))],
)
def test_a_walls_index_is_written_permuted_in_an_integer_file_and_left_alone_in_a_radian_one(element, expected):
    """Only an index the element changes is written; the rest stay verbatim."""
    walls = [FirstUnit(i, WALL, 40.5 + i, 30.5, float(i)) for i in range(5)]
    writes = [image.rotation_write for image in plan_in_place(walls, element, _RECT, False).images]
    assert writes == [None if index is None else (WRITE_WALL, index) for index in expected]
    radian = [FirstUnit(0, WALL, 40.5, 30.5, _RADIAN_3)]
    assert plan_in_place(radian, element, _RECT, True).images[0].rotation_write is None


def test_only_angle_consts_get_a_facing_write_and_cliffs_trees_and_inert_keep_theirs():
    firsts = _firsts()
    plan = plan_in_place(firsts, ROTATE_CW, _RECT, False)
    for first, image in zip(firsts, plan.images, strict=True):
        if first.unit_const in (ARCHER, VILLAGER):
            assert image.rotation_write == (
                WRITE_ANGLE, facing_image(first.rotation, ROTATE_CW, unit_rotation.angle_count_for(first.unit_const))
            )
        elif first.unit_const != WALL:
            assert image.rotation_write is None, first.key


def test_an_occupant_lands_on_its_holders_image_wherever_it_was_stored():
    holder = FirstUnit("castle", CASTLE, *render_units.span_anchor(40, 30, 4, 4), 0.0)
    away = FirstUnit("away", ARCHER, 70.5, 10.5, 0.0, holder=0)
    on = FirstUnit("on", ARCHER, holder.x, holder.y, 0.0, holder=0)
    for element in ELEMENTS[1:]:
        castle, *occupants = plan_in_place([holder, away, on], element, (40, 30, 44, 34), False).images
        assert all((o.x, o.y) == (castle.x, castle.y) for o in occupants), element


def test_kept_orientation_counts_as_the_pasted_turn_does():
    firsts = [
        FirstUnit(0, CLIFF, *render_units.span_anchor(40, 30, 3, 3), 0.0),
        FirstUnit(1, BRIDGE, *render_units.span_anchor(45, 30, 2, 3), 0.0),
        FirstUnit(2, GATE_NE, *render_units.span_anchor(40, 34, 4, 1), 0.0),
        FirstUnit(3, MOLE, *render_units.span_anchor(45, 36, 3, 1), 0.0),
    ]
    assert plan_in_place(firsts, "r", _RECT, False).kept_orientation == 2
    assert plan_in_place(firsts, "mx", _RECT, False).kept_orientation == 1


def test_image_on_map_wants_the_whole_footprint():
    assert image_on_map(ARCHER, 0.0, 119.9, 120, 120)
    assert not image_on_map(ARCHER, -0.5, 3.5, 120, 120)
    assert not image_on_map(ARCHER, 3.5, 120.0, 120, 120)
    assert image_on_map(CASTLE, *render_units.span_anchor(116, 0, 4, 4), 120, 120)
    assert not image_on_map(CASTLE, *render_units.span_anchor(117, 0, 4, 4), 120, 120)
    assert not image_on_map(GATE_NE, *render_units.span_anchor(-1, 5, 4, 1), 120, 120)
    assert image_on_map(GATE_NE, *render_units.span_anchor(10, 0, 4, 1), 120, 120)
    assert not image_on_map(GATE_NE, 12.0, -0.5, 120, 120)


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize(("run", "span"), [((607, 606, 605), (2, 3)), ((906, 905, 904), (1, 1))], ids=["classic", "wooden"])
def test_a_turned_bridge_keeps_its_top_and_bottom_at_the_screen_top_and_bottom_ends(element, run, span):
    """TASK-329, TASK-706: an `a` bridge (bottom at low x, middle, top at high
    x; classic 607-605 or 1x1 wooden 906-904) under each D4 element takes its
    orientation siblings, a top piece at the end nearer the screen top and a
    bottom piece at the other, measured through tile_screen_origin() rather
    than the top-ward table."""
    from descape import object_orientation

    sx, sy = span
    pieces = [_unit(c, *render_units.span_anchor(sx * k, 0, sx, sy)) for k, c in enumerate(run)]
    turned = transform_block(_block(3 * sx, sy, pieces), element, False)
    assert turned.kept_orientation == 0
    block = turned.block
    proj = iso_geometry.canvas_size_and_origin(block.width, block.height, 32, 0, 0)

    def screen_y(unit) -> float:
        tiles = render_units.occupied_tiles_for(unit.unit_const, unit.dx, unit.dy, block.width, block.height)
        assert len(tiles) == sx * sy, (element, unit)
        return sum(iso_geometry.tile_screen_origin(tx, ty, 0, proj)[1] for tx, ty in tiles) / len(tiles)

    # Units keep their order, so where the run reverses the first piece's image is the new top.
    by_end = {object_orientation.group_state(u.unit_const)[1]: u for u in block.units}
    assert sorted(by_end, key=str) == [0, 1, None], element
    assert screen_y(by_end[0]) < screen_y(by_end[None]) < screen_y(by_end[1]), element
    assert len({tile for u in block.units for tile in render_units.occupied_tiles_for(
        u.unit_const, u.dx, u.dy, block.width, block.height)}) == 3 * sx * sy, "the pieces still tile the turned block"


@pytest.mark.parametrize("element", [e for e in ELEMENTS if e != "id"])
def test_a_carried_corner_piece_takes_its_gate_images_corner_on_an_image_end_tile(element):
    """TASK-532: for every gate with a corner const, a piece with `gate` set
    lands on an end tile of its gate's image as corner_piece() of the image's
    const; one without keeps reorient_gate_const()'s verbatim const, as the
    paste turn and Mirror Map hand it."""
    from descape import mirror_tools, placement_rules

    gates = sorted(c for group in gate_orientation.groups().values() for c in group if gate_orientation.corner_piece(c))
    assert len(gates) >= 32
    for gate in gates:
        corner = gate_orientation.corner_piece(gate)
        x, y = render_units.span_anchor(2, 2, *tile_span(gate, render_units.NON_BUILDING_SPAN))
        ends = placement_rules.end_tiles(gate, x, y)
        firsts = [FirstUnit(0, gate, x, y, 0.0)]
        firsts += [FirstUnit(1 + i, corner, tx + 0.5, ty + 0.25, 0.0, gate=0) for i, (tx, ty) in enumerate(ends)]
        firsts.append(FirstUnit(3, corner, ends[0][0] + 0.5, ends[0][1] + 0.5, 0.0))
        plan = plan_in_place(firsts, element, (0, 0, 8, 8), False)
        image = plan.images[0]
        image_ends = set(placement_rules.end_tiles(image.unit_const, image.x, image.y))
        carried = plan.images[1:3]
        assert {p.unit_const for p in carried} == {gate_orientation.corner_piece(image.unit_const)}, (gate, element)
        assert {(int(p.x), int(p.y)) for p in carried} == image_ends, (gate, element)
        assert all(p.rotation_write is None for p in carried)
        assert plan.images[3].unit_const == mirror_tools.reorient_gate_const(corner, element) == corner
