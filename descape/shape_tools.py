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

import math

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


# Quadrant-arc moves in doubled coordinates (one tile is 2): away from the
# top apex toward the side apex. Their order is the DP's tie-break order.
_ARC_E, _ARC_SE, _ARC_S = (2, 0), (2, -2), (0, -2)
_ARC_MOVES = (_ARC_E, _ARC_SE, _ARC_S)
# The DP only walks tiles this close to the curve, in doubled units (1.5 tiles).
_ARC_BAND = 3.0


def ellipse_perimeter_tiles(
    x0: int, y0: int, x1: int, y1: int, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """The outline of the ellipse inscribed in the inclusive tile box, for
    Wall Circle (GH #125): a closed, thin, 8-connected ring (every tile has
    exactly two ring tiles among its 8 neighbours), touching all four box
    sides, symmetric under both flips of the box, clipped, row-major, no
    duplicates. A box under 3 tiles on either axis is rect_perimeter_tiles(),
    so 1xN is a straight run and 2x2 four towers.

    8-connected because a 4-connected staircase gives each stair tile one
    orthogonal neighbour per axis, so a diagonal stretch would be a chain of
    towers; here it gets the {NE,SW}/{NW,SE} diagonal shapes instead
    (unit_sprites.wall_variant_from_neighbours8()).

    Not a plain midpoint rasterizer: following the curve alone, a quadrant
    arc can meet its mirror at an apex in a one-neighbour spur or a
    three-neighbour corner (the 4xN and Nx4 boxes do). So one quadrant arc
    is the cheapest path, by summed squared distance to the curve, among
    the monotone paths that rule both out (_ellipse_quadrant_arc()),
    mirrored into the other three.
    """
    left, top, right, bottom = rect_bounds(x0, y0, x1, y1)
    if right - left < 2 or bottom - top < 2:
        return rect_perimeter_tiles(x0, y0, x1, y1, map_width, map_height)
    half_w, half_h = right - left, bottom - top  # semi-axes, doubled
    arc = _ellipse_quadrant_arc(half_w, half_h, _ARC_BAND) or _ellipse_quadrant_arc(half_w, half_h, None)
    ring = {
        ((su * u + left + right) // 2, (sv * v + top + bottom) // 2)
        for u, v in arc
        for su in (1, -1)
        for sv in (1, -1)
    }
    return sorted(
        ((x, y) for x, y in ring if 0 <= x < map_width and 0 <= y < map_height),
        key=lambda tile: (tile[1], tile[0]),
    )


def _ellipse_quadrant_arc(half_w: int, half_h: int, band: float | None) -> list[tuple[int, int]] | None:
    """One quadrant of the ring in doubled coordinates about the box centre
    (u right, v up), from the top apex (u = half_w % 2, v = half_h) to the
    side apex (u = half_w, v = half_h % 2). Moves are E, SE and S; the first
    is never S and the last never E (either would leave a spur or a
    three-neighbour tile at the apex once mirrored), and an E never touches
    an S (that corner tile would see both of its ring neighbours' neighbours).
    A shortest path over (tile, last move), in order of u - v, which every
    move raises. None when `band` cuts every path off."""
    w2, h2 = half_w * half_w, half_h * half_h

    def cost(u: int, v: int) -> float:
        # Squared Sampson distance: |F| / |grad F|, the first-order distance to the curve.
        f = u * u / w2 + v * v / h2 - 1.0
        grad = 2.0 * math.sqrt(u * u / (w2 * w2) + v * v / (h2 * h2))
        return (f / grad) ** 2 if grad else math.inf

    start, end = (half_w % 2, half_h), (half_w, half_h % 2)
    best: dict[tuple, tuple[float, tuple | None]] = {(start, None): (cost(*start), None)}
    buckets: dict[int, set] = {start[0] - start[1]: {(start, None)}}
    for k in range(start[0] - start[1], end[0] - end[1] + 1, 2):
        # Sorted so ties resolve the same way every run.
        for key in sorted(buckets.pop(k, ()), key=lambda s: (s[0], -1 if s[1] is None else _ARC_MOVES.index(s[1]))):
            (u, v), last = key
            so_far = best[key][0]
            for move in _ARC_MOVES:
                if (last is None and move == _ARC_S) or {last, move} == {_ARC_E, _ARC_S}:
                    continue
                q = (u + move[0], v + move[1])
                if q[0] > half_w or q[1] < end[1]:
                    continue
                step = cost(*q)
                if band is not None and step > band * band:
                    continue
                new_key = (q, move)
                total = so_far + step
                if new_key not in best or total < best[new_key][0]:
                    best[new_key] = (total, key)
                    buckets.setdefault(q[0] - q[1], set()).add(new_key)
    finals = [(best[(end, move)][0], i, move) for i, move in enumerate((_ARC_SE, _ARC_S)) if (end, move) in best]
    if not finals:
        return None
    key: tuple | None = (end, min(finals)[2])
    path = []
    while key is not None:
        path.append(key[0])
        key = best[key][1]
    return path[::-1]


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
