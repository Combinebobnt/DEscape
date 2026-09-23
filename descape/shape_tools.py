"""Tile-space geometry for the Draw Line and Draw Rectangle tools. Pure
index math, no Qt, no config -- the same split brush.py has, and for the
same reason: the interesting part of a shape tool is which tiles it names,
and that part is worth testing without a QApplication.

Every function here clips to the map, matching brush.brush_tiles()'s own
contract rather than leaving it to call sites: a shape drag legitimately
runs off the map edge (the endpoint holds its last valid tile, which can
still be a tile whose brush footprint straddles the border), so an
unclipped return would be a bug waiting at every caller.

Shift-constrain lives here too, in TILE space. Screen space is not an
option: in Sloped the projection warps per corner with elevation, so a
screen angle is not well-defined along a line's own length, and the same
drag in Flat and in Stepped would paint different tiles.
"""

from __future__ import annotations

# The 16 primitive directions Shift snaps a line to -- slopes 0, 1:2, 1:1,
# 2:1 and vertical, reflected across both axes. Not 15-degree increments:
# measured against Bresenham, a 15-degree snap displaces the endpoint by at
# most 0.13x the drag length, so under ~8 tiles it does nothing at all, and
# 15/30/60/75 rasterize to irregular staircases indistinguishable from an
# unsnapped line. Only small-integer slopes read as deliberate on a lattice.
SNAP_DIRECTIONS: tuple[tuple[int, int], ...] = tuple(
    (sx * ax, sy * ay)
    for ax, ay in ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1))
    for sx in (1, -1)
    for sy in (1, -1)
    if not (ax == 0 and sx == -1) and not (ay == 0 and sy == -1)
)


def line_tiles(
    x0: int, y0: int, x1: int, y1: int, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """Bresenham from (x0, y0) to (x1, y1) inclusive, clipped to the map, in
    draw order from the anchor outward. Clipped after rasterizing rather
    than before: clipping the endpoints first would change the line's slope,
    so the visible part of an off-map drag would no longer lie on the line
    the user is previewing."""
    tiles = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < map_width and 0 <= y < map_height:
            tiles.append((x, y))
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x += sx
        if err2 <= dx:
            err += dx
            y += sy
    return tiles


def wall_path_tiles(
    x0: int, y0: int, x1: int, y1: int, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """A wall run's path from (x0, y0) to (x1, y1) inclusive, clipped
    to the map, in draw order from the anchor outward -- same clip-after-
    rasterize contract line_tiles() documents, and for the same reason.

    Not Bresenham: in-game a wall drag is always 45-degree aligned, an
    arbitrary angle splitting into one pure diagonal segment and one
    axis-aligned segment, which reads as a bent wall rather than as a
    staircase. So this emits min(|dx|, |dy|) corner-touching diagonal steps
    first, then the axis-aligned remainder to the endpoint: two segments,
    exactly one bend.

    Diagonal-segment-first is a choice, not a measurement -- the game may
    put the axis-aligned half first. It is on the in-game verification list
    for that reason. The diagonal steps being 8-connected rather than
    4-connected is measured, though: a fifth of all corpus walls sit on a
    diagonal-only neighbour mask (see
    unit_sprites.wall_variant_from_neighbours8()).
    """
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    diagonal = min(abs(dx), abs(dy))
    straight = max(abs(dx), abs(dy)) - diagonal
    tiles = []
    x, y = x0, y0
    steps = [(sx, sy)] * diagonal
    steps += [(sx, 0) if abs(dx) > abs(dy) else (0, sy)] * straight
    for step_x, step_y in [(0, 0), *steps]:
        x += step_x
        y += step_y
        if 0 <= x < map_width and 0 <= y < map_height:
            tiles.append((x, y))
    return tiles


def snap_wall_delta(dx: int, dy: int) -> tuple[int, int]:
    """Shift on a wall-run drag: the drag snaps to one of the 8 primitive
    directions (4 axis-aligned, 4 diagonal), keeping its own length along
    that direction.

    8, not snap_line_delta()'s 16: a wall run has no shallow-angle form at
    all (wall_path_tiles() bends rather than staircases), so the 1:2 and 2:1
    members of SNAP_DIRECTIONS would offer directions the tool cannot
    express, and a Shift drag would land somewhere the preview then
    contradicts.
    """
    if dx == 0 and dy == 0:
        return 0, 0
    if abs(dx) >= 2 * abs(dy):
        return dx, 0
    if abs(dy) >= 2 * abs(dx):
        return 0, dy
    # A diagonal keeps the longer axis's length, so the snap never shortens
    # a drag back toward its anchor -- snap_square_delta()'s own rule.
    side = max(abs(dx), abs(dy))
    return (side if dx >= 0 else -side), (side if dy >= 0 else -side)


def rect_bounds(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """The drag's two corners normalized to (left, top, right, bottom), both
    ends inclusive. Unclipped on purpose -- every consumer below clips its
    own emitted tiles, and a clipped bounds would move the perimeter ring
    onto the map edge instead of letting it fall off."""
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def rect_tiles(
    x0: int, y0: int, x1: int, y1: int, map_width: int, map_height: int, *, filled: bool
) -> list[tuple[int, int]]:
    if not filled:
        return rect_perimeter_tiles(x0, y0, x1, y1, map_width, map_height)
    left, top, right, bottom = rect_bounds(x0, y0, x1, y1)
    return [
        (x, y)
        for y in range(max(top, 0), min(bottom, map_height - 1) + 1)
        for x in range(max(left, 0), min(right, map_width - 1) + 1)
    ]


def rect_perimeter_tiles(
    x0: int, y0: int, x1: int, y1: int, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """The rectangle's one-tile-thick border, clipped, in row-major order
    with no duplicates (a 1xN or Nx1 rectangle is its own perimeter, and a
    naive four-edge walk would emit its corners twice). Also the Flag B
    preview path for a filled rectangle: the ring is what the user reads as
    "the rectangle" anyway, and it drops a 480-square from ~230k tiles to
    ~1.9k."""
    left, top, right, bottom = rect_bounds(x0, y0, x1, y1)
    tiles = []
    for y in range(max(top, 0), min(bottom, map_height - 1) + 1):
        if y in (top, bottom):
            xs = range(max(left, 0), min(right, map_width - 1) + 1)
        else:
            # sorted(set(...)), not the literal pair: a one-column rectangle
            # has left == right, and the naive pair would emit it twice per
            # interior row.
            xs = sorted({x for x in (left, right) if 0 <= x < map_width})
        tiles.extend((x, y) for x in xs)
    return tiles


def snap_line_delta(dx: int, dy: int) -> tuple[int, int]:
    """The Shift-constrained delta for a raw drag delta: the primitive
    direction in SNAP_DIRECTIONS closest in angle to (dx, dy), repeated a
    whole number of times.

    The whole-multiple step is the load-bearing one, not a rounding detail.
    Landing the endpoint on an exact integer multiple of the primitive
    vector is what makes Bresenham emit a perfectly regular run (`##  ##  ##`
    for a 1:2, never `## ### ##`); projecting onto the ray and rounding the
    components separately does not.
    """
    if dx == 0 and dy == 0:
        return 0, 0
    best = None
    best_cos = None
    raw_len = (dx * dx + dy * dy) ** 0.5
    for vx, vy in SNAP_DIRECTIONS:
        cos = (dx * vx + dy * vy) / (raw_len * (vx * vx + vy * vy) ** 0.5)
        if best_cos is None or cos > best_cos:
            best_cos = cos
            best = (vx, vy)
    vx, vy = best
    k = round((dx * vx + dy * vy) / (vx * vx + vy * vy))
    # A drag shorter than half a primitive vector rounds k to 0, which would
    # collapse the line to its anchor and make Shift feel broken. One repeat
    # is the shortest thing the snap can honestly draw.
    if k < 1:
        k = 1
    return vx * k, vy * k


def snap_square_delta(dx: int, dy: int) -> tuple[int, int]:
    """Shift on Rectangle squares it: the longer axis wins, and each axis
    keeps its own sign, so the square grows in the direction actually
    dragged rather than snapping back toward the anchor."""
    side = max(abs(dx), abs(dy))
    return (side if dx >= 0 else -side), (side if dy >= 0 else -side)
