"""Pure geometry coverage for descape/brush.py -- no Qt, no AoE2ScenarioParser.
Golden offset tables are transcribed as literal data (not recomputed from the
formula under test) so a formula change that quietly alters the footprint a
user sees fails loudly here, matching tests/test_fill_tools.py's own
pin-the-shape style.
"""

from __future__ import annotations

import pytest

from descape.brush import (
    BRUSH_SHAPE_CIRCLE,
    BRUSH_SHAPE_CIRCLE_OUTLINE,
    BRUSH_SHAPE_LINE_X,
    BRUSH_SHAPE_LINE_Y,
    BRUSH_SHAPE_PLUS,
    BRUSH_SHAPE_ROUNDED,
    BRUSH_SHAPE_SQUARE,
    BRUSH_SHAPE_SQUARE_OUTLINE,
    BRUSH_SHAPES,
    BRUSH_SIZE_MAX,
    BRUSH_SIZE_MIN,
    BRUSH_THICKNESS_MAX,
    BRUSH_THICKNESS_MIN,
    OUTLINE_SHAPES,
    brush_offsets,
    brush_tiles,
    clamp_brush_size,
    clamp_brush_thickness,
    line_tiles,
    shape_label,
)


def _square(lo: int, hi: int) -> set[tuple[int, int]]:
    return {(dx, dy) for dy in range(lo, hi + 1) for dx in range(lo, hi + 1)}


# Golden footprints, sizes 1-9, transcribed from the pinned table in the
# implementation plan. lo/hi bound the square each size's offsets live in.
_BOUNDS = {n: (-((n - 1) // 2), n // 2) for n in range(1, 10)}

_SQUARE_COUNTS = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
_CIRCLE_COUNTS = {1: 1, 2: 4, 3: 9, 4: 12, 5: 21, 6: 32, 7: 37, 8: 52, 9: 69}

_CIRCLE_SIZE4 = {
    (dx, dy)
    for dx, dy in _square(*_BOUNDS[4])
    if (dx, dy) not in {(-1, -1), (2, -1), (-1, 2), (2, 2)}
}
_CIRCLE_SIZE5 = {
    (dx, dy)
    for dx, dy in _square(*_BOUNDS[5])
    if (dx, dy) not in {(-2, -2), (2, -2), (-2, 2), (2, 2)}
}


def test_square_offsets_match_pinned_table_sizes_1_to_9() -> None:
    for size, count in _SQUARE_COUNTS.items():
        lo, hi = _BOUNDS[size]
        offsets = set(brush_offsets(size, BRUSH_SHAPE_SQUARE))
        assert offsets == _square(lo, hi)
        assert len(offsets) == count


def test_circle_offsets_match_pinned_table_sizes_1_to_9() -> None:
    for size, count in _CIRCLE_COUNTS.items():
        assert len(set(brush_offsets(size, BRUSH_SHAPE_CIRCLE))) == count


def test_circle_size4_matches_pinned_shape() -> None:
    assert set(brush_offsets(4, BRUSH_SHAPE_CIRCLE)) == _CIRCLE_SIZE4


def test_circle_size5_matches_pinned_shape() -> None:
    assert set(brush_offsets(5, BRUSH_SHAPE_CIRCLE)) == _CIRCLE_SIZE5


def test_circle_equals_square_for_sizes_1_to_3() -> None:
    # A tile-space circle at these sizes samples every tile center inside
    # its bounding square -- there is no rule that excludes a corner
    # without also excluding everything, so circle and square coincide.
    # This is a real, deliberate consequence of the formula, not an
    # oversight -- pinned explicitly so a "fix" is a conscious decision.
    for size in (1, 2, 3):
        assert set(brush_offsets(size, BRUSH_SHAPE_SQUARE)) == set(brush_offsets(size, BRUSH_SHAPE_CIRCLE))


def test_size_1_is_a_single_tile_both_shapes() -> None:
    assert brush_offsets(1, BRUSH_SHAPE_SQUARE) == ((0, 0),)
    assert brush_offsets(1, BRUSH_SHAPE_CIRCLE) == ((0, 0),)


def test_even_size_anchor_is_up_left_of_center() -> None:
    # Size 2: cursor tile sits at the up-left corner of the 2x2 block.
    assert set(brush_offsets(2, BRUSH_SHAPE_SQUARE)) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    # Size 4: offset range is -1..2 on both axes, not -2..1 or symmetric.
    offsets = brush_offsets(4, BRUSH_SHAPE_SQUARE)
    dxs = {dx for dx, _ in offsets}
    dys = {dy for _, dy in offsets}
    assert dxs == {-1, 0, 1, 2}
    assert dys == {-1, 0, 1, 2}


def test_offsets_are_memoized() -> None:
    assert brush_offsets(5, BRUSH_SHAPE_CIRCLE) is brush_offsets(5, BRUSH_SHAPE_CIRCLE)
    assert brush_offsets(5, BRUSH_SHAPE_SQUARE) is brush_offsets(5, BRUSH_SHAPE_SQUARE)


def test_offsets_are_deterministic_and_unique() -> None:
    for size in range(BRUSH_SIZE_MIN, BRUSH_SIZE_MAX + 1):
        for shape in BRUSH_SHAPES:
            for thickness in range(BRUSH_THICKNESS_MIN, BRUSH_THICKNESS_MAX + 1):
                offsets = brush_offsets(size, shape, thickness)
                assert len(offsets) == len(set(offsets))
                assert offsets == brush_offsets(size, shape, thickness)
                assert list(offsets) == sorted(offsets, key=lambda o: (o[1], o[0])), "row-major order"


def test_size_is_clamped_not_raised() -> None:
    assert clamp_brush_size(0) == BRUSH_SIZE_MIN
    assert clamp_brush_size(-5) == BRUSH_SIZE_MIN
    assert clamp_brush_size(100) == BRUSH_SIZE_MAX
    assert brush_offsets(0, BRUSH_SHAPE_SQUARE) == brush_offsets(BRUSH_SIZE_MIN, BRUSH_SHAPE_SQUARE)
    assert brush_offsets(999, BRUSH_SHAPE_SQUARE) == brush_offsets(BRUSH_SIZE_MAX, BRUSH_SHAPE_SQUARE)


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        brush_offsets(3, "triangle")


def test_brush_tiles_clips_at_every_corner_of_a_square_map() -> None:
    w = h = 20
    for cx, cy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        tiles = brush_tiles(cx, cy, 5, BRUSH_SHAPE_SQUARE, w, h)
        assert all(0 <= x < w and 0 <= y < h for x, y in tiles)
        assert len(tiles) < 25  # a 5x5 centered on a corner must clip


def test_brush_tiles_center_on_map_returns_full_footprint() -> None:
    w = h = 40
    tiles = brush_tiles(20, 20, 5, BRUSH_SHAPE_SQUARE, w, h)
    assert len(tiles) == 25


def test_brush_tiles_never_wraps_a_row_on_a_non_square_map() -> None:
    # A flat-index clamp (y * width + x, then bounds-check the flat value)
    # would let a footprint straddling the right edge wrap onto the start of
    # the next row -- same hazard fill_tools.py's own neighbor clamp guards
    # against. width != height here specifically, since this app cannot
    # open such a map today and this pure function's own contract is the
    # only place that case can be exercised at all.
    width, height = 40, 8
    tiles = brush_tiles(39, 4, 5, BRUSH_SHAPE_SQUARE, width, height)
    for x, y in tiles:
        assert 0 <= x < width
        assert 0 <= y < height
        assert x <= 39
        assert abs(y - 4) <= 2


def test_brush_tiles_off_map_center_returns_empty() -> None:
    assert brush_tiles(-5, -5, 3, BRUSH_SHAPE_SQUARE, 10, 10) == []


def test_brush_tiles_size_1_is_the_bare_cursor_tile() -> None:
    assert brush_tiles(4, 4, 1, BRUSH_SHAPE_SQUARE, 10, 10) == [(4, 4)]
    assert brush_tiles(4, 4, 1, BRUSH_SHAPE_CIRCLE, 10, 10) == [(4, 4)]


# line_tiles golden paths, derived by hand from the tile-centre line's
# edge crossings (x steps first at an exact corner tie).
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0, 0), (3, 0), [(1, 0), (2, 0), (3, 0)]),
        ((0, 0), (0, -2), [(0, -1), (0, -2)]),
        ((0, 0), (2, 2), [(1, 0), (1, 1), (2, 1), (2, 2)]),
        ((0, 0), (1, 3), [(0, 1), (1, 1), (1, 2), (1, 3)]),
        ((2, 1), (0, 0), [(1, 1), (1, 0), (0, 0)]),
        ((5, 5), (5, 5), []),
    ],
    ids=["horizontal", "vertical-negative", "diagonal-tie", "steep", "negative-both", "zero-length"],
)
def test_line_tiles_golden_paths(start, end, expected) -> None:
    assert line_tiles(*start, *end) == expected


def test_line_tiles_is_4_connected_and_ends_at_the_target() -> None:
    for x0, y0, x1, y1 in [(0, 0, 7, 3), (4, 9, -2, 1), (-3, 2, 5, -6), (10, 10, 3, 11), (0, 0, -5, -5)]:
        path = line_tiles(x0, y0, x1, y1)
        assert len(path) == abs(x1 - x0) + abs(y1 - y0)
        assert path[-1] == (x1, y1)
        prev = (x0, y0)
        for tile in path:
            assert abs(tile[0] - prev[0]) + abs(tile[1] - prev[1]) == 1
            assert min(x0, x1) <= tile[0] <= max(x0, x1)
            assert min(y0, y1) <= tile[1] <= max(y0, y1)
            prev = tile


# -- TASK-257's shapes -----------------------------------------------------------

# Tile counts for sizes 1-9, from an independent probe (plans' golden table, 2026-10-13).
_SHAPE_COUNTS = {
    (BRUSH_SHAPE_ROUNDED, 1): [1, 4, 5, 12, 21, 32, 45, 60, 77],
    (BRUSH_SHAPE_PLUS, 1): [1, 4, 5, 12, 9, 20, 13, 28, 17],
    (BRUSH_SHAPE_LINE_X, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (BRUSH_SHAPE_LINE_Y, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (BRUSH_SHAPE_SQUARE_OUTLINE, 1): [1, 4, 8, 12, 16, 20, 24, 28, 32],
    (BRUSH_SHAPE_SQUARE_OUTLINE, 2): [1, 4, 9, 16, 24, 32, 40, 48, 56],
    (BRUSH_SHAPE_SQUARE_OUTLINE, 3): [1, 4, 9, 16, 25, 36, 48, 60, 72],
    (BRUSH_SHAPE_SQUARE_OUTLINE, 4): [1, 4, 9, 16, 25, 36, 49, 64, 80],
    (BRUSH_SHAPE_CIRCLE_OUTLINE, 1): [1, 4, 8, 8, 12, 20, 16, 20, 32],
    (BRUSH_SHAPE_CIRCLE_OUTLINE, 2): [1, 4, 9, 12, 20, 28, 28, 40, 48],
    (BRUSH_SHAPE_CIRCLE_OUTLINE, 3): [1, 4, 9, 12, 21, 32, 36, 48, 60],
    (BRUSH_SHAPE_CIRCLE_OUTLINE, 4): [1, 4, 9, 12, 21, 32, 37, 52, 68],
}

_CIRCLE_OUTLINE_9_T1 = """
..#####..
.##...##.
##.....##
#.......#
#.......#
#.......#
##.....##
.##...##.
..#####..
"""

_PLUS_6 = """
..##..
..##..
######
######
..##..
..##..
"""


def _picture(size: int, text: str) -> set[tuple[int, int]]:
    lo, _hi = _BOUNDS[size]
    rows = text.strip().splitlines()
    return {(lo + col, lo + row) for row, line in enumerate(rows) for col, ch in enumerate(line) if ch == "#"}


@pytest.mark.parametrize(("shape", "thickness"), list(_SHAPE_COUNTS))
def test_new_shape_counts_match_the_golden_table(shape: str, thickness: int) -> None:
    counts = [len(brush_offsets(size, shape, thickness)) for size in range(1, 10)]
    assert counts == _SHAPE_COUNTS[(shape, thickness)]


def test_circle_outline_9_and_plus_6_match_their_pictures() -> None:
    assert set(brush_offsets(9, BRUSH_SHAPE_CIRCLE_OUTLINE, 1)) == _picture(9, _CIRCLE_OUTLINE_9_T1)
    assert set(brush_offsets(6, BRUSH_SHAPE_PLUS)) == _picture(6, _PLUS_6)


def test_rounded_square_is_the_square_minus_its_corners_from_size_3() -> None:
    for size in range(1, 10):
        lo, hi = _BOUNDS[size]
        corners = {(lo, lo), (hi, lo), (lo, hi), (hi, hi)} if size >= 3 else set()
        assert set(brush_offsets(size, BRUSH_SHAPE_ROUNDED)) == _square(lo, hi) - corners


def _neighbours(tile, diagonal: bool):
    x, y = tile
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return [(x + dx, y + dy) for dx, dy in steps]


def _reach(start, allowed, diagonal: bool) -> set:
    seen, stack = {start}, [start]
    while stack:
        for tile in _neighbours(stack.pop(), diagonal):
            if tile in allowed and tile not in seen:
                seen.add(tile)
                stack.append(tile)
    return seen


@pytest.mark.parametrize("shape", sorted(OUTLINE_SHAPES))
def test_every_ring_is_8_connected_and_encloses_its_hole(shape: str) -> None:
    """Loops over the live constants, so a raised BRUSH_SIZE_MAX fails here instead of shipping a gapped ring."""
    filled_shape = BRUSH_SHAPE_SQUARE if shape == BRUSH_SHAPE_SQUARE_OUTLINE else BRUSH_SHAPE_CIRCLE
    for size in range(BRUSH_SIZE_MIN, BRUSH_SIZE_MAX + 1):
        for thickness in range(BRUSH_THICKNESS_MIN, BRUSH_THICKNESS_MAX + 1):
            ring = set(brush_offsets(size, shape, thickness))
            filled = set(brush_offsets(size, filled_shape))
            assert ring <= filled
            assert _reach(next(iter(ring)), ring, diagonal=True) == ring, (size, thickness)
            hole = filled - ring
            if size - 2 * thickness < 1:
                assert not hole, "nothing to subtract: the filled shape"
                continue
            assert hole, (size, thickness)
            lo, hi = _BOUNDS[size]
            box = _square(lo - 1, hi + 1)
            flood = _reach(next(iter(hole)), box - ring, diagonal=False)
            assert flood == hole, f"size {size} thickness {thickness}: the hole leaks out of the ring"


def test_thickness_is_ignored_for_a_shape_without_one() -> None:
    for shape in set(BRUSH_SHAPES) - OUTLINE_SHAPES:
        assert brush_offsets(7, shape, 3) is brush_offsets(7, shape)
        assert brush_tiles(20, 20, 7, shape, 40, 40, thickness=3) == brush_tiles(20, 20, 7, shape, 40, 40)


def test_outline_thickness_is_clamped() -> None:
    assert clamp_brush_thickness(0) == BRUSH_THICKNESS_MIN
    assert clamp_brush_thickness(99) == BRUSH_THICKNESS_MAX
    for shape in OUTLINE_SHAPES:
        assert brush_offsets(9, shape, 0) is brush_offsets(9, shape, BRUSH_THICKNESS_MIN)
        assert brush_offsets(9, shape, 99) is brush_offsets(9, shape, BRUSH_THICKNESS_MAX)


def test_an_outline_with_no_room_for_a_hole_is_the_filled_shape() -> None:
    assert brush_offsets(3, BRUSH_SHAPE_SQUARE_OUTLINE, 2) == brush_offsets(3, BRUSH_SHAPE_SQUARE)
    assert brush_offsets(8, BRUSH_SHAPE_CIRCLE_OUTLINE, 4) == brush_offsets(8, BRUSH_SHAPE_CIRCLE)


def test_plus_and_lines_keep_the_even_size_anchor() -> None:
    # Size 2 covers the same 2x2 every shape does; size 4 runs -1..2 like the square.
    assert set(brush_offsets(2, BRUSH_SHAPE_PLUS)) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    plus4 = set(brush_offsets(4, BRUSH_SHAPE_PLUS))
    assert {dx for dx, _ in plus4} == {dy for _, dy in plus4} == {-1, 0, 1, 2}
    assert plus4 == {(dx, dy) for dx, dy in _square(-1, 2) if dx in (0, 1) or dy in (0, 1)}
    assert brush_offsets(4, BRUSH_SHAPE_LINE_X) == ((-1, 0), (0, 0), (1, 0), (2, 0))
    assert brush_offsets(4, BRUSH_SHAPE_LINE_Y) == ((0, -1), (0, 0), (0, 1), (0, 2))


def test_brush_tiles_passes_thickness_and_clips() -> None:
    tiles = brush_tiles(0, 0, 9, BRUSH_SHAPE_SQUARE_OUTLINE, 20, 20, thickness=2)
    assert all(0 <= x < 20 and 0 <= y < 20 for x, y in tiles)
    # The 5x5 quadrant at the corner holding the cursor: everything but the 3x3 hole's (0..2) part.
    assert set(tiles) == _square(0, 4) - _square(0, 2)
    assert len(brush_tiles(10, 10, 9, BRUSH_SHAPE_SQUARE_OUTLINE, 20, 20, thickness=2)) == 56


def test_shape_labels_name_the_lines_by_the_view() -> None:
    assert shape_label(BRUSH_SHAPE_LINE_X, isometric=True) == "Line /"
    assert shape_label(BRUSH_SHAPE_LINE_Y, isometric=True) == "Line \\"
    assert shape_label(BRUSH_SHAPE_LINE_X, isometric=False) == "Line -"
    assert shape_label(BRUSH_SHAPE_LINE_Y, isometric=False) == "Line |"
    for isometric in (True, False):
        assert shape_label(BRUSH_SHAPE_SQUARE, isometric) == "Square"
        assert shape_label(BRUSH_SHAPE_ROUNDED, isometric) == "Rounded square"
        assert all(label.isascii() for label in (shape_label(s, isometric) for s in BRUSH_SHAPES))
    with pytest.raises(ValueError):
        shape_label("triangle", True)


def test_square_stays_first_in_the_shape_order() -> None:
    assert BRUSH_SHAPES == (
        BRUSH_SHAPE_SQUARE, BRUSH_SHAPE_CIRCLE, BRUSH_SHAPE_ROUNDED, BRUSH_SHAPE_SQUARE_OUTLINE,
        BRUSH_SHAPE_CIRCLE_OUTLINE, BRUSH_SHAPE_PLUS, BRUSH_SHAPE_LINE_X, BRUSH_SHAPE_LINE_Y,
    )
