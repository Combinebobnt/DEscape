"""Pure geometry coverage for descape/shape_tools.py -- no Qt, no
AoE2ScenarioParser. Golden tile sets are transcribed as literal data rather
than recomputed from the formula under test, matching tests/test_brush.py's
own pin-the-shape style.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from descape.shape_tools import (
    SNAP_DIRECTIONS,
    line_tiles,
    rect_bounds,
    rect_perimeter_tiles,
    rect_tiles,
    snap_line_delta,
    snap_square_delta,
    snap_wall_delta,
    wall_path_tiles,
)

_W = _H = 20


def test_a_zero_length_line_is_the_anchor_tile_alone():
    assert line_tiles(4, 4, 4, 4, _W, _H) == [(4, 4)]


def test_a_horizontal_line_runs_in_drag_order():
    assert line_tiles(2, 5, 6, 5, _W, _H) == [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]


def test_a_reversed_drag_starts_at_its_own_anchor():
    assert line_tiles(6, 5, 2, 5, _W, _H) == [(6, 5), (5, 5), (4, 5), (3, 5), (2, 5)]


def test_a_diagonal_line_is_one_tile_per_step():
    assert line_tiles(0, 0, 4, 4, _W, _H) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_a_two_to_one_line_pairs_every_x_step():
    # Transcribed, not recomputed: this run's shape (two tiles per y) is the
    # thing a Bresenham change would silently alter.
    assert line_tiles(0, 0, 6, 3, _W, _H) == [
        (0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3),
    ]


def test_a_line_running_off_the_map_keeps_only_its_on_map_tiles():
    tiles = line_tiles(-3, 2, 3, 2, _W, _H)
    assert tiles == [(0, 2), (1, 2), (2, 2), (3, 2)]


def test_clipping_does_not_bend_the_line():
    # The on-map part of an off-map drag must lie on the SAME line an
    # unclipped rasterization would produce -- clipping the endpoints first
    # would change the slope and move these tiles.
    clipped = line_tiles(-10, -5, 10, 5, _W, _H)
    unclipped = line_tiles(-10, -5, 10, 5, 1000, 1000)
    assert clipped == [(x, y) for x, y in unclipped if 0 <= x < _W and 0 <= y < _H]


def test_a_line_wholly_off_the_map_paints_nothing():
    assert line_tiles(-5, -5, -2, -2, _W, _H) == []


def test_rect_bounds_normalizes_either_drag_direction():
    assert rect_bounds(7, 9, 2, 3) == (2, 3, 7, 9)
    assert rect_bounds(2, 3, 7, 9) == (2, 3, 7, 9)


def test_a_filled_rectangle_covers_every_tile_in_its_bounds():
    tiles = rect_tiles(2, 3, 4, 5, _W, _H, filled=True)
    assert tiles == [
        (2, 3), (3, 3), (4, 3),
        (2, 4), (3, 4), (4, 4),
        (2, 5), (3, 5), (4, 5),
    ]


def test_an_outlined_rectangle_is_its_border_only():
    assert rect_tiles(2, 3, 4, 5, _W, _H, filled=False) == [
        (2, 3), (3, 3), (4, 3),
        (2, 4), (4, 4),
        (2, 5), (3, 5), (4, 5),
    ]


def test_a_single_row_rectangle_is_its_own_perimeter_with_no_duplicates():
    tiles = rect_perimeter_tiles(2, 7, 5, 7, _W, _H)
    assert tiles == [(2, 7), (3, 7), (4, 7), (5, 7)]
    assert len(tiles) == len(set(tiles))


def test_a_single_column_rectangle_is_its_own_perimeter_with_no_duplicates():
    tiles = rect_perimeter_tiles(6, 2, 6, 5, _W, _H)
    assert tiles == [(6, 2), (6, 3), (6, 4), (6, 5)]
    assert len(tiles) == len(set(tiles))


def test_a_one_tile_rectangle_is_one_tile_either_way():
    assert rect_tiles(3, 3, 3, 3, _W, _H, filled=True) == [(3, 3)]
    assert rect_tiles(3, 3, 3, 3, _W, _H, filled=False) == [(3, 3)]


def test_a_perimeter_never_repeats_a_corner():
    tiles = rect_perimeter_tiles(1, 1, 8, 6, _W, _H)
    assert len(tiles) == len(set(tiles))
    # 2 * (w + h) - 4 for an 8x6 border.
    assert len(tiles) == 2 * (8 + 6) - 4


def test_a_rectangle_straddling_the_edge_is_clipped_on_both_axes():
    assert rect_tiles(-2, -1, 1, 1, _W, _H, filled=True) == [
        (0, 0), (1, 0), (0, 1), (1, 1),
    ]
    # The clipped-away left and top edges do not reappear on the map border:
    # only the real right edge (x == 1) and bottom edge (y == 1) survive.
    assert rect_perimeter_tiles(-2, -1, 1, 1, _W, _H) == [(1, 0), (0, 1), (1, 1)]


def test_a_rectangle_wholly_off_the_map_paints_nothing():
    assert rect_tiles(-9, -9, -4, -4, _W, _H, filled=True) == []
    assert rect_perimeter_tiles(-9, -9, -4, -4, _W, _H) == []


def test_there_are_sixteen_distinct_snap_directions():
    assert len(SNAP_DIRECTIONS) == 16
    assert len(set(SNAP_DIRECTIONS)) == 16
    assert set(SNAP_DIRECTIONS) == {
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2),
    }


def test_a_zero_length_drag_snaps_to_nothing():
    assert snap_line_delta(0, 0) == (0, 0)


def _run_lengths(tiles: list[tuple[int, int]], slow: int) -> list[int]:
    """Consecutive tile counts along the slower-moving axis -- the `##  ##  ##`
    pattern a lattice line reads as, expressed as numbers."""
    runs: list[list[int]] = []
    for tile in tiles:
        if runs and tile[slow] == runs[-1][0]:
            runs[-1][1] += 1
        else:
            runs.append([tile[slow], 1])
    return [count for _, count in runs]


def _interior_runs(dx: int, dy: int) -> list[int]:
    """Run lengths with the first and last dropped. Both ends are inclusive
    endpoints, so they are half-runs by construction on any slope steeper
    than 1:1 -- regularity is a claim about the middle. A pure horizontal or
    vertical line is a single run with no middle to strip, and is trivially
    regular."""
    tiles = line_tiles(40, 40, 40 + dx, 40 + dy, 200, 200)
    slow = 1 if abs(dx) >= abs(dy) else 0
    runs = _run_lengths(tiles, slow)
    return runs[1:-1] if len(runs) > 2 else runs


@pytest.mark.parametrize("direction", SNAP_DIRECTIONS)
def test_an_exact_multiple_snaps_to_itself(direction):
    vx, vy = direction
    assert snap_line_delta(vx * 7, vy * 7) == (vx * 7, vy * 7)


@pytest.mark.parametrize("direction", SNAP_DIRECTIONS)
def test_every_snapped_endpoint_is_a_whole_multiple_of_some_direction(direction):
    vx, vy = direction
    # Nudged a little off-axis so the snap has something to correct, but by
    # less than half a primitive step -- a bigger nudge crosses the bisector
    # into a neighbouring direction, which is correct behaviour and not what
    # this case is about.
    sx, sy = snap_line_delta(vx * 9 + (1 if vy else 0), vy * 9 + (1 if vx else 0))
    assert any(
        k >= 1 and (sx, sy) == (ux * k, uy * k)
        for ux, uy in SNAP_DIRECTIONS
        for k in [round((sx * ux + sy * uy) / (ux * ux + uy * uy))]
    )


@pytest.mark.parametrize("direction", SNAP_DIRECTIONS)
def test_every_snapped_line_rasterizes_to_a_regular_run(direction):
    """The discriminating case for the snap spec: an endpoint on an exact
    primitive multiple makes Bresenham emit uniform interior runs. Projecting
    onto the ray and rounding, the obvious alternative implementation, gives
    a ragged `## ### ##` instead -- see the control case below."""
    vx, vy = direction
    assert len(set(_interior_runs(vx * 7, vy * 7))) == 1


def test_an_unsnapped_delta_can_rasterize_ragged():
    """The control the test above needs to mean anything: (6, 4) is not a
    whole multiple of any primitive direction, and its interior runs are not
    uniform. Snapping is what removes this."""
    assert len(set(_interior_runs(6, 4))) > 1
    assert snap_line_delta(6, 4) != (6, 4)


def test_a_snap_prefers_the_nearest_direction():
    assert snap_line_delta(10, 1) == (10, 0)
    assert snap_line_delta(10, 9) == (10, 10)
    assert snap_line_delta(10, 5) == (10, 5)
    assert snap_line_delta(1, 10) == (0, 10)
    assert snap_line_delta(-10, 5) == (-10, 5)
    assert snap_line_delta(-4, -8) == (-4, -8)


def test_a_drag_shorter_than_the_primitive_still_draws_one_repeat():
    # round() would give k == 0 here, collapsing the line onto its anchor and
    # making Shift look broken on a short drag.
    assert snap_line_delta(1, 0) == (1, 0)
    dx, dy = snap_line_delta(1, 1)
    assert (dx, dy) == (1, 1)


def test_a_square_takes_the_longer_axis_and_keeps_each_sign():
    assert snap_square_delta(10, 3) == (10, 10)
    assert snap_square_delta(3, 10) == (10, 10)
    assert snap_square_delta(-10, 3) == (-10, 10)
    assert snap_square_delta(10, -3) == (10, -10)
    assert snap_square_delta(-3, -10) == (-10, -10)
    assert snap_square_delta(0, 0) == (0, 0)


# --- Wall Run (2026-09-19 wall-runs plan) ------------------------------

# The 8 primitive directions a wall run can express, as (dx, dy) unit steps.
_WALL_DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]


@pytest.mark.parametrize("step", _WALL_DIRECTIONS)
def test_a_straight_wall_run_walks_one_direction(step):
    """A drag along a primitive direction is a pure run with no bend: the
    diagonal segment or the axis-aligned one is the whole path."""
    dx, dy = step
    tiles = wall_path_tiles(20, 20, 20 + dx * 6, 20 + dy * 6, 40, 40)
    assert tiles == [(20 + dx * i, 20 + dy * i) for i in range(7)]


def test_a_shallow_drag_bends_exactly_once_diagonal_first():
    """The load-bearing shape difference from line_tiles(): 45-degree
    alignment, so an arbitrary angle splits into one pure diagonal segment
    and one axis-aligned one rather than into a Bresenham staircase."""
    tiles = wall_path_tiles(0, 0, 10, 3, 40, 40)
    assert tiles == [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
    ]
    # One bend: the step direction changes exactly once along the path.
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(tiles)]
    assert sum(1 for a, b in pairwise(steps) if a != b) == 1


def test_a_steep_drag_bends_onto_the_y_axis():
    assert wall_path_tiles(0, 0, 2, 5, 40, 40) == [
        (0, 0), (1, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    ]


@pytest.mark.parametrize(
    "span", [(0, 0, 9, 4), (9, 4, 0, 0), (5, 5, 1, 12), (5, 5, 12, 1), (7, 7, 7, 7)]
)
def test_every_wall_path_step_is_one_tile_and_contiguous(span):
    tiles = wall_path_tiles(*span, 40, 40)
    assert tiles[0] == span[:2]
    assert tiles[-1] == span[2:]
    for (ax, ay), (bx, by) in pairwise(tiles):
        assert max(abs(bx - ax), abs(by - ay)) == 1


def test_a_single_tile_drag_is_one_tile():
    assert wall_path_tiles(6, 6, 6, 6, 40, 40) == [(6, 6)]


def test_a_wall_path_clips_after_rasterizing_like_a_line():
    """Same contract line_tiles() documents: the visible part of an off-map
    drag still lies on the path the preview showed, so clipping cannot move
    the bend."""
    tiles = wall_path_tiles(2, 2, -6, 2, 40, 40)
    assert tiles == [(2, 2), (1, 2), (0, 2)]
    assert wall_path_tiles(-4, -4, -2, -2, 40, 40) == []


def test_snap_wall_delta_only_ever_yields_a_primitive_direction():
    """8, not snap_line_delta()'s 16 -- a 1:2 or 2:1 result would name a
    direction wall_path_tiles() cannot draw."""
    seen = set()
    for dx in range(-9, 10):
        for dy in range(-9, 10):
            if (dx, dy) == (0, 0):
                continue
            sx, sy = snap_wall_delta(dx, dy)
            unit = max(abs(sx), abs(sy))
            seen.add((sx // unit, sy // unit))
    assert seen == set(_WALL_DIRECTIONS)


def test_snap_wall_delta_picks_the_nearer_of_axis_and_diagonal():
    assert snap_wall_delta(10, 1) == (10, 0)
    assert snap_wall_delta(1, 10) == (0, 10)
    assert snap_wall_delta(10, 9) == (10, 10)
    assert snap_wall_delta(-10, 6) == (-10, 10)
    assert snap_wall_delta(-6, -10) == (-10, -10)
    assert snap_wall_delta(0, 0) == (0, 0)


def test_a_snapped_wall_drag_keeps_its_length():
    """snap_square_delta()'s own rule: the snap grows toward the drag rather
    than pulling the endpoint back toward the anchor."""
    for dx, dy in [(10, 6), (6, -10), (-9, 9), (12, 1)]:
        sx, sy = snap_wall_delta(dx, dy)
        assert max(abs(sx), abs(sy)) == max(abs(dx), abs(dy))
