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

from descape import iso_geometry, render, unit_rotation
from descape.mirror_tools import AXIS_SWAPPING, TRANSFORMS, compose, facing_image
from descape.region_clipboard import RegionBlock, RegionUnit
from descape.region_transform import (
    FLIP_HORIZONTAL,
    FLIP_VERTICAL,
    ROTATE_CCW,
    ROTATE_CW,
    anchor_for,
    lands_on_map,
    pivot2_of,
    transform_block,
)

ELEMENTS = list(TRANSFORMS)
ARCHER = 4  # 1x1, ANGLE (16 facings)
HOUSE = 70  # 2x2, INERT
CASTLE = 82  # 4x4, INERT
GATE_NE = 64  # (4, 1)
GATE_DIAGONAL = 659  # 4x4, sparse diagonal footprint
OAK = 349  # 1x1, VARIANT
WALL = 117  # 1x1, five wall shapes
CLIFF = 264  # 3x3 cliff
BRIDGE = 605  # (2, 3), no orientation sibling
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
    "2x2 house": _unit(HOUSE, *render.span_anchor(1, 3, 2, 2)),
    "4x4 castle": _unit(CASTLE, *render.span_anchor(4, 1, 4, 4)),
    "NE gate": _unit(GATE_NE, *render.span_anchor(1, 0, 4, 1)),
    "diagonal gate": _unit(GATE_DIAGONAL, *render.span_anchor(5, 2, 4, 4)),
}


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("name", list(_FOOTPRINT_UNITS))
def test_a_units_footprint_image_is_the_transform_of_its_footprint(element, name):
    w, h = 9, 6
    unit = _FOOTPRINT_UNITS[name]
    source_tiles = render.occupied_tiles_for(unit.unit_const, unit.dx, unit.dy, w, h)
    assert source_tiles, "the source footprint must lie inside the block"
    (image,) = transform_block(_block(w, h, [unit]), element, False).block.units
    n = max(w, h)
    low_x, low_y = _low_corner(element, w, h)
    expected = {
        (ix - low_x, iy - low_y) for ix, iy in (TRANSFORMS[element](x, y, n) for x, y in source_tiles)
    }
    new_w, new_h = (h, w) if element in AXIS_SWAPPING else (w, h)
    got = render.occupied_tiles_for(image.unit_const, image.dx, image.dy, new_w, new_h)
    assert set(got) == expected


@pytest.mark.parametrize("element", ELEMENTS)
def test_unit_fields_follow_the_mirror_rules(element):
    units = [
        _unit(ARCHER, 1.5, 1.5, rotation=1.25),
        _unit(OAK, 2.5, 1.5, rotation=37.0),
        _unit(HOUSE, *render.span_anchor(3, 0, 2, 2), rotation=7.0),
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
            _unit(CASTLE, *render.span_anchor(2, 1, 4, 4)),
            _unit(ARCHER, 4.0, 3.0, garrison_slot=1),  # stored on its holder's point
            _unit(OAK, 0.5, 0.5),
        ]
        turned = transform_block(_block(w, h, units), element, False).block.units
        assert [u.unit_const for u in turned] == [ARCHER, CASTLE, ARCHER, OAK], element
        assert [u.garrison_slot for u in turned] == [-1, -1, 1, -1], element
        assert (turned[2].dx, turned[2].dy) == (turned[1].dx, turned[1].dy), element


def test_kept_orientation_counts_cliffs_always_and_unsquare_spans_under_a_swap():
    units = [
        _unit(CLIFF, *render.span_anchor(0, 0, 3, 3)),
        _unit(BRIDGE, *render.span_anchor(5, 0, 2, 3)),
        _unit(GATE_NE, *render.span_anchor(0, 4, 4, 1)),
        _unit(ARCHER, 8.5, 5.5),
    ]
    block = _block(9, 6, units)
    assert transform_block(block, "r", False).kept_orientation == 2
    assert transform_block(block, "mx", False).kept_orientation == 1


def test_rotate_cw_turns_a_facing_the_way_rotate_step_does():
    """rotate_step(+1) is clockwise on screen (AoE2 rotation grows clockwise),
    so ROTATE_CW must add a quarter turn to a facing, not take one away."""
    assert facing_image(0.0, ROTATE_CW) == pytest.approx(math.pi / 2)
    assert unit_rotation.rotate_step(0.0, 4, +1) == pytest.approx(math.pi / 2)
    assert facing_image(0.0, ROTATE_CCW) == pytest.approx(3 * math.pi / 2)
    assert compose(ROTATE_CW, ROTATE_CCW) == "id"


def test_the_flips_mirror_the_screen_left_right_and_top_bottom():
    """Through tile_screen_origin() itself: FLIP_HORIZONTAL negates sx about
    the square's centre and keeps sy; FLIP_VERTICAL the other way round."""
    n = 9
    proj = iso_geometry.canvas_size_and_origin(n, n, 32, 0, 0)

    def screen(x, y):
        return iso_geometry.tile_screen_origin(x, y, 0, proj)

    # Twice the centre tile's origin, so the reflection stays in integers.
    cx, cy = (2 * c for c in screen((n - 1) // 2, (n - 1) // 2))
    for x, y in [(0, 0), (2, 7), (8, 3), (5, 5), (1, 0)]:
        sx, sy = screen(x, y)
        hx, hy = screen(*TRANSFORMS[FLIP_HORIZONTAL](x, y, n))
        vx, vy = screen(*TRANSFORMS[FLIP_VERTICAL](x, y, n))
        assert (hx, hy) == (cx - sx, sy), (x, y)
        assert (vx, vy) == (sx, cy - sy), (x, y)


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
