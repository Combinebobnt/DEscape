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
"""

from __future__ import annotations

from functools import lru_cache

BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 9

BRUSH_SHAPE_SQUARE = "square"
BRUSH_SHAPE_CIRCLE = "circle"
BRUSH_SHAPES = (BRUSH_SHAPE_SQUARE, BRUSH_SHAPE_CIRCLE)


def clamp_brush_size(size: int) -> int:
    return max(BRUSH_SIZE_MIN, min(BRUSH_SIZE_MAX, size))


@lru_cache(maxsize=None)
def brush_offsets(size: int, shape: str) -> tuple[tuple[int, int], ...]:
    """(dx, dy) offsets from the cursor tile for a brush of the given size
    and shape, in stable row-major order (y ascending, then x ascending).
    size is clamped to [BRUSH_SIZE_MIN, BRUSH_SIZE_MAX] rather than raising --
    callers pass straight through from a QSpinBox already range-limited to
    that span, so an out-of-range value here would only ever be a defensive
    edge, not a real one."""
    size = clamp_brush_size(size)
    lo = -((size - 1) // 2)
    hi = size // 2
    if shape == BRUSH_SHAPE_SQUARE:
        return tuple((dx, dy) for dy in range(lo, hi + 1) for dx in range(lo, hi + 1))
    if shape != BRUSH_SHAPE_CIRCLE:
        raise ValueError(f"Unknown brush shape: {shape!r}")
    center = (lo + hi) / 2
    radius_sq = (size / 2) ** 2
    return tuple(
        (dx, dy)
        for dy in range(lo, hi + 1)
        for dx in range(lo, hi + 1)
        if (dx - center) ** 2 + (dy - center) ** 2 <= radius_sq
    )


def brush_tiles(
    cx: int, cy: int, size: int, shape: str, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """brush_offsets(size, shape) translated to tile (cx, cy) and clipped to
    the map. width/height are checked separately (never a single flat-index
    clamp) so a footprint straddling an edge can never wrap into the next
    row -- same hazard fill_tools.py's neighbor clamp guards against. Works
    for map_width != map_height even though nothing in this app can
    currently open such a map (see viewer.py's non-square load-time note);
    this function's own contract doesn't depend on that."""
    return [
        (cx + dx, cy + dy)
        for dx, dy in brush_offsets(size, shape)
        if 0 <= cx + dx < map_width and 0 <= cy + dy < map_height
    ]
