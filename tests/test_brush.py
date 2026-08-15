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
    BRUSH_SHAPE_SQUARE,
    BRUSH_SIZE_MAX,
    BRUSH_SIZE_MIN,
    brush_offsets,
    brush_tiles,
    clamp_brush_size,
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
        for shape in (BRUSH_SHAPE_SQUARE, BRUSH_SHAPE_CIRCLE):
            offsets = brush_offsets(size, shape)
            assert len(offsets) == len(set(offsets))
            assert offsets == brush_offsets(size, shape)


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
