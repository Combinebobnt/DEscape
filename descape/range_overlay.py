"""Qt-free geometry for View > Range Rings -- GH #49's base range ring.

Split the way grid_overlay already is: this module decides WHERE the ring
goes, map_view's QGraphicsPathItem decides how it looks. Nothing here imports
PyQt5.

Nothing here draws an ellipse. The ring is a true circle in TILE space, and
isometric projection is what renders it as the 2:1 ellipse the issue's
screenshot shows, so every sample goes through iso_geometry's own forward
map rather than through a hand-written ellipse. That also keeps the ring on
the same projection the sprites and the footprint outlines use.

Distance is measured on the ground plane, the rule descape/ruler.py already
states for the Ruler ("AoE2 resolves unit range on the ground plane, so a
unit standing on a hill is not further away"). Only the height the ring is
DRAWN at follows the building, and that is one value for the whole ring,
passed in by the caller.
"""

from __future__ import annotations

import math

from descape import iso_geometry, terrain_palette
from descape.unit_stats_table import unit_stats

# Samples per ring, rounded UP to a multiple of eight. Eight, not four:
# sampling from theta 0 in 90-degree steps hits the tile-space axis extremes,
# but the SCREEN extremes of the projected ellipse sit at 45 + k*90 degrees,
# and the ring's bounding box is only the ellipse's when both sets are
# sampled.
SEGMENT_MULTIPLE = 8
MIN_SEGMENTS = 64
SEGMENTS_PER_TILE = 16


def ring_radius_tiles(range_tiles: float, span: tuple[int, int]) -> float:
    """The ring's radius in tiles, measured from the FOOTPRINT EDGE: a 4x4
    Castle with range 8 draws at 10 tiles.

    The one named place that rule lives, so the in-game pass can flip it in a
    single edit. The alternative is plain `range_tiles`, measured from the
    building's centre; only a side-by-side against the game's own circle
    settles which one AoE2 draws. This shape matches the game resolving reach
    from a unit's collision radius, and degenerates to `range_tiles` for a
    1x1."""
    return float(range_tiles) + max(span) / 2.0


def ring_radius_for_const(unit_const: int) -> float | None:
    """The ring radius for a placed `unit_const`, or None when it draws no
    ring: not a building, or a building with no range.

    BUILDING_TILE_SPANS is the codebase's own is-a-building membership test
    (render._unit_color reads it that way), and is read live here rather than
    snapshotted, since tests inject synthetic consts into it. `range` is
    unit_stats' `displayed_range`, the same number the Inspector's Range row
    shows, so the ring and that row can never disagree."""
    span = terrain_palette.BUILDING_TILE_SPANS.get(unit_const)
    if span is None:
        return None
    range_tiles = float(unit_stats(unit_const).get("range", 0.0))
    if range_tiles <= 0.0:
        return None
    return ring_radius_tiles(range_tiles, span)


def ring_segments(radius_tiles: float) -> int:
    """How many samples a ring of this radius takes, a multiple of
    SEGMENT_MULTIPLE. A 10-tile ring is 160, cheap for one QPainterPath."""
    wanted = max(MIN_SEGMENTS, math.ceil(SEGMENTS_PER_TILE * max(0.0, float(radius_tiles))))
    return math.ceil(wanted / SEGMENT_MULTIPLE) * SEGMENT_MULTIPLE


def ring_points(
    cx: float,
    cy: float,
    radius_tiles: float,
    style: str,
    tile_px: int | None = None,
    proj: iso_geometry.IsoProjection | None = None,
    rise_px: int = 0,
) -> list[tuple[float, float]]:
    """The ring around continuous map point (cx, cy) as a polygon in scene
    space, plain tuples so this module stays Qt-free (unit_pick.unit_polygons'
    own convention). Implicitly closed: the last point joins the first, and
    map_view's _add_closed_polygons() is what closes the subpath.

    Flat's scene space is (mx * tile_px, my * tile_px) (grid_axes()' own flat
    branch), so the ring is a TRUE circle there and Flat + Isometric's view
    transform shears it into the right ellipse for free. Stepped and Sloped
    project each sample through map_point_to_screen() at the caller's single
    `rise_px`."""
    segments = ring_segments(radius_tiles)
    samples = (
        (
            cx + radius_tiles * math.cos(2.0 * math.pi * i / segments),
            cy + radius_tiles * math.sin(2.0 * math.pi * i / segments),
        )
        for i in range(segments)
    )
    if style == "flat":
        if tile_px is None:
            raise ValueError("flat needs tile_px")
        return [(mx * tile_px, my * tile_px) for mx, my in samples]
    if style in ("stepped", "sloped"):
        if proj is None:
            raise ValueError(f"{style} needs proj")
        points = []
        for mx, my in samples:
            sx, sy = iso_geometry.map_point_to_screen(mx, my, int(rise_px), proj)
            points.append((float(sx), float(sy)))
        return points
    raise ValueError(f"unknown terrain style {style!r}")
