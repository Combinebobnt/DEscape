"""Brush footprints for the drag-stroke edit tools (Terrain, Elevate, Set
Elevation). Pure tile-index math, no Qt, no config -- brush size/shape is
deliberately session-only (see viewer.py's brush widgets), so nothing here
touches descape.settings.

Size is a diameter/edge length, not a radius: size 1 is a single tile, size 3
is a 3x3 square, size 5 is a 5x5 square. Even sizes have no true center tile;
the anchor convention below puts the cursor tile at the upper-left of the
four center tiles rather than rounding outward, so a size-2 brush at (x, y)
covers exactly (x, y), (x+1, y), (x, y+1), (x+1, y+1).

The circle shape is approximated by testing tile centers against a radius in
TILE space, not screen space -- deliberately. DEscape's Stepped view is a
dimetric projection, so a circle that looked round on screen would be an
ellipse in tile space (and would paint an ellipse). This module does the
opposite: paint a true tile-space circle, and accept that Stepped mode will
render it as an on-screen ellipse. That is expected, not a bug to chase.

At sizes 1-3 a tile-space circle rasterizes identically to the square: with
so few tile centers to sample, nothing falls outside the radius. They start
to diverge at size 4. See tests/test_brush.py for the pinned footprint table
across sizes 1-9.

The other six shapes (TASK-257) are all defined on the same box and anchor:

- **Rounded square** is the square minus its four corner tiles, from size 3.
  On a grid this small nothing else fits between Square and Circle: it equals
  Circle at sizes 4 to 6 and Plus at size 3.
- **Square outline** and **Circle outline** are the shape minus the same
  shape `2 * thickness` smaller, concentric (same parity, so the same
  centre). Thickness is measured inward. When `size - 2 * thickness < 1`
  nothing is left to subtract and the result is the filled shape, so no
  size/thickness coupling is needed in the UI.
- **Plus** is the box's centre row and column: 1-wide arms at odd sizes, and
  2-wide arms over the four centre tiles at even sizes, which have no centre
  tile. Size 2 is the full 2x2.
- **Line X** and **Line Y** run along the tile axes through the cursor tile,
  1 wide and `size` long, with the even-size anchor rule along their length.
  They are a fixed tile set like the circle, so they read as the two
  diagonals in the isometric views; shape_label() names them for the view.

Raising elevation with a hollow or thin shape lets
`set_tiles_elevation()`'s gap-fill rule write extra tiles: a hole 1 or 2
tiles wide fills solid, a wider square hole loses its four corners, and a
Plus from size 5 gains its four inner corners. Lowering writes the ring
alone. tests/test_elevation_tools.py pins the counts, so a later propagation
change is a conscious one.
"""

from __future__ import annotations

from functools import cache

BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 9

BRUSH_SHAPE_SQUARE = "square"
BRUSH_SHAPE_CIRCLE = "circle"
BRUSH_SHAPE_ROUNDED = "rounded"
BRUSH_SHAPE_SQUARE_OUTLINE = "square_outline"
BRUSH_SHAPE_CIRCLE_OUTLINE = "circle_outline"
BRUSH_SHAPE_PLUS = "plus"
BRUSH_SHAPE_LINE_X = "line_x"
BRUSH_SHAPE_LINE_Y = "line_y"

# The Shape combo's order: index 0 stays Square, the default.
BRUSH_SHAPES = (
    BRUSH_SHAPE_SQUARE,
    BRUSH_SHAPE_CIRCLE,
    BRUSH_SHAPE_ROUNDED,
    BRUSH_SHAPE_SQUARE_OUTLINE,
    BRUSH_SHAPE_CIRCLE_OUTLINE,
    BRUSH_SHAPE_PLUS,
    BRUSH_SHAPE_LINE_X,
    BRUSH_SHAPE_LINE_Y,
)
OUTLINE_SHAPES = frozenset({BRUSH_SHAPE_SQUARE_OUTLINE, BRUSH_SHAPE_CIRCLE_OUTLINE})
_FILLED_OF_OUTLINE = {BRUSH_SHAPE_SQUARE_OUTLINE: BRUSH_SHAPE_SQUARE, BRUSH_SHAPE_CIRCLE_OUTLINE: BRUSH_SHAPE_CIRCLE}

BRUSH_THICKNESS_MIN = 1
# A thicker ring than this leaves no hole at any size up to BRUSH_SIZE_MAX.
BRUSH_THICKNESS_MAX = (BRUSH_SIZE_MAX - 1) // 2

# ASCII only: box-drawing glyphs are missing from some Windows UI fonts.
_LABELS = {
    BRUSH_SHAPE_SQUARE: "Square",
    BRUSH_SHAPE_CIRCLE: "Circle",
    BRUSH_SHAPE_ROUNDED: "Rounded square",
    BRUSH_SHAPE_SQUARE_OUTLINE: "Square outline",
    BRUSH_SHAPE_CIRCLE_OUTLINE: "Circle outline",
    BRUSH_SHAPE_PLUS: "Plus",
}
# Screen x grows with x + y and screen y with y - x in the isometric views
# (iso_geometry.tile_screen_origin), so the tile x axis runs up-right there.
_LINE_LABELS = {
    (BRUSH_SHAPE_LINE_X, True): "Line /",
    (BRUSH_SHAPE_LINE_Y, True): "Line \\",
    (BRUSH_SHAPE_LINE_X, False): "Line -",
    (BRUSH_SHAPE_LINE_Y, False): "Line |",
}


def clamp_brush_size(size: int) -> int:
    return max(BRUSH_SIZE_MIN, min(BRUSH_SIZE_MAX, size))


def clamp_brush_thickness(thickness: int) -> int:
    return max(BRUSH_THICKNESS_MIN, min(BRUSH_THICKNESS_MAX, thickness))


def shape_label(shape: str, isometric: bool) -> str:
    """The Shape combo's text for `shape`. The two lines are named by how they
    run on screen, which depends on the view; every other label does not."""
    if shape in (BRUSH_SHAPE_LINE_X, BRUSH_SHAPE_LINE_Y):
        return _LINE_LABELS[(shape, bool(isometric))]
    try:
        return _LABELS[shape]
    except KeyError:
        raise ValueError(f"Unknown brush shape: {shape!r}") from None


def brush_offsets(size: int, shape: str, thickness: int = 1) -> tuple[tuple[int, int], ...]:
    """(dx, dy) offsets from the cursor tile for a brush of the given size,
    shape and (outline shapes only) thickness, in stable row-major order (y
    ascending, then x ascending). size and thickness are clamped rather than
    raising -- callers pass straight through from a QSpinBox already
    range-limited to that span, so an out-of-range value here would only ever
    be a defensive edge, not a real one. Thickness is ignored for a shape
    that has none, so each real footprint has one cached tuple."""
    size = clamp_brush_size(size)
    thickness = clamp_brush_thickness(thickness) if shape in OUTLINE_SHAPES else 1
    return _offsets(size, shape, thickness)


def _box(size: int) -> tuple[int, int]:
    """The (lo, hi) offset span of a size-`size` box: the cursor tile is the
    upper-left of the four centre tiles at an even size."""
    return -((size - 1) // 2), size // 2


@cache
def _offsets(size: int, shape: str, thickness: int) -> tuple[tuple[int, int], ...]:
    lo, hi = _box(size)
    span = range(lo, hi + 1)
    if shape == BRUSH_SHAPE_SQUARE:
        return tuple((dx, dy) for dy in span for dx in span)
    if shape == BRUSH_SHAPE_CIRCLE:
        center = (lo + hi) / 2
        radius_sq = (size / 2) ** 2
        return tuple((dx, dy) for dy in span for dx in span if (dx - center) ** 2 + (dy - center) ** 2 <= radius_sq)
    if shape == BRUSH_SHAPE_ROUNDED:
        if size < 3:
            return _offsets(size, BRUSH_SHAPE_SQUARE, 1)
        corners = {(lo, lo), (hi, lo), (lo, hi), (hi, hi)}
        return tuple((dx, dy) for dy in span for dx in span if (dx, dy) not in corners)
    if shape in OUTLINE_SHAPES:
        filled = _offsets(size, _FILLED_OF_OUTLINE[shape], 1)
        inner = size - 2 * thickness
        if inner < 1:
            return filled
        # Same parity, so the same centre: the smaller box is this one moved in by `thickness`.
        hole = set(_offsets(inner, _FILLED_OF_OUTLINE[shape], 1))
        return tuple(offset for offset in filled if offset not in hole)
    if shape == BRUSH_SHAPE_PLUS:
        # The centre row and column; an even size has two of each.
        middle = {0} if size % 2 else {0, 1}
        return tuple((dx, dy) for dy in span for dx in span if dx in middle or dy in middle)
    if shape == BRUSH_SHAPE_LINE_X:
        return tuple((dx, 0) for dx in span)
    if shape == BRUSH_SHAPE_LINE_Y:
        return tuple((0, dy) for dy in span)
    raise ValueError(f"Unknown brush shape: {shape!r}")


def brush_tiles(
    cx: int, cy: int, size: int, shape: str, map_width: int, map_height: int, thickness: int = 1
) -> list[tuple[int, int]]:
    """brush_offsets(size, shape, thickness) translated to tile (cx, cy) and clipped to
    the map. width/height are checked separately (never a single flat-index
    clamp) so a footprint straddling an edge can never wrap into the next
    row -- same hazard fill_tools.py's neighbor clamp guards against. Works
    for map_width != map_height even though nothing in this app can
    currently open such a map (see viewer.py's non-square load-time note);
    this function's own contract doesn't depend on that."""
    return [
        (cx + dx, cy + dy)
        for dx, dy in brush_offsets(size, shape, thickness)
        if 0 <= cx + dx < map_width and 0 <= cy + dy < map_height
    ]


def line_tiles(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """The 4-connected (supercover) tile path from (x0, y0) to (x1, y1),
    excluding the start tile and including the end: exactly |dx| + |dy|
    tiles, each one edge-step from the last. Used to fill the cursor-tile
    gaps Qt's motion compression leaves in a slow drag stroke; 4-connected
    because a real cursor crosses tile edges, not corners.

    Integer-only: step x while the next x edge is no farther along the
    tile-centre line than the next y edge. At an exact corner tie x steps
    first, so the path is deterministic."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    x, y = x0, y0
    ix = iy = 0
    path = []
    for _ in range(dx + dy):
        if iy == dy or (ix < dx and (2 * ix + 1) * dy <= (2 * iy + 1) * dx):
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        path.append((x, y))
    return path
