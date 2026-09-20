"""Qt-free geometry, LOD and colours for View > Grid, the tile-grid overlay.
The Qt half is viewer_canvas.GridItem, which turns these scene-space lines
into prebuilt QLineF batches and decides per paint which ranks are worth
drawing.

Split the way edge_ticks already is: this module decides WHERE a line
belongs, the item decides how it looks. Nothing here imports PyQt5.

grid_axes() is the ground-plane lattice, every line at elevation 0, which is
the same reference the Distance Ticks ruler and the Ruler tool use. That is
why its endpoints come from edge_ticks.iso_corner: each ruler tick lands
exactly on the end of a grid line at any tick interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from descape import edge_ticks, iso_geometry

Point = tuple[int, int]
Line = tuple[Point, Point]
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class GridAxis:
    """Every grid line of one family, in scene space. `axis` is "x" for the
    lines of constant x (which run along y) and "y" for the others.
    `minor_step` is the unnormalized scene-space offset from one line to the
    next; the item maps it as a delta and measures it in device space, which
    is what survives Flat's rotate-and-squash."""

    axis: str
    lines: tuple[Line, ...]
    majors: tuple[bool, ...]
    minor_step: tuple[float, float]


def grid_axes(
    map_w: int,
    map_h: int,
    proj: iso_geometry.IsoProjection | None = None,
    tile_px: int | None = None,
) -> tuple[GridAxis, GridAxis]:
    """The (x, y) line families of a map_w x map_h grid: map_w + 1 lines of
    constant x and map_h + 1 of constant y. Dispatches on `proj` the way
    edge_ticks.edge_runs does: Stepped and Sloped pass one, Flat passes
    `tile_px`."""
    xs = range(map_w + 1)
    ys = range(map_h + 1)
    if proj is not None:
        x_lines = tuple((edge_ticks.iso_corner(i, 0, proj), edge_ticks.iso_corner(i, map_h, proj)) for i in xs)
        y_lines = tuple((edge_ticks.iso_corner(0, j, proj), edge_ticks.iso_corner(map_w, j, proj)) for j in ys)
        x_step = (float(proj.half_w), float(-proj.half_h))
        y_step = (float(proj.half_w), float(proj.half_h))
    elif tile_px is not None:
        x_lines = tuple(((i * tile_px, 0), (i * tile_px, map_h * tile_px)) for i in xs)
        y_lines = tuple(((0, j * tile_px), (map_w * tile_px, j * tile_px)) for j in ys)
        x_step = (float(tile_px), 0.0)
        y_step = (0.0, float(tile_px))
    else:
        raise ValueError("grid_axes needs either proj (iso) or tile_px (flat)")
    return (
        GridAxis("x", x_lines, tuple(is_major(i) for i in xs), x_step),
        GridAxis("y", y_lines, tuple(is_major(j) for j in ys), y_step),
    )


def is_major(index: int) -> bool:
    """Every MAJORS_PER_MINOR-th grid line is a major, index 0 included, so
    the grid's majors line up with a one-tile ruler's."""
    return edge_ticks.is_major(index, 1)


# LOD, on the DEVICE-space gap between adjacent lines. Zoom is fit-relative,
# so a 480x480 at minimum zoom is under 2 device px per tile, and an ungated
# per-tile grid there is solid noise.
MIN_GRID_SPACING_PX = 6.0  # between MAJORS
MIN_MINOR_SPACING_PX = 5.0  # between MINORS


@dataclass(frozen=True)
class GridLod:
    """What a given on-screen density leaves worth drawing. No hysteresis,
    so the same view state always yields the same lines."""

    draw_grid: bool
    draw_minors: bool


def grid_lod(minor_spacing_px: float) -> GridLod:
    """Monotonic by construction: minors drop before the grid does."""
    draw_grid = minor_spacing_px * edge_ticks.MAJORS_PER_MINOR >= MIN_GRID_SPACING_PX
    return GridLod(draw_grid=draw_grid, draw_minors=draw_grid and minor_spacing_px >= MIN_MINOR_SPACING_PX)


# Appearance, Settings > Appearance. Blend is a signed strength, not a colour:
# 0 is invisible, negative darkens the terrain under the line towards black,
# positive whitens it. Source-over with an opaque black at alpha a is exactly
# dst * (1 - a), and with white dst * (1 - a) + a, so the two halves are a
# darken and a lighten blend without any QPainter composition mode.
BLEND_MIN = -100
BLEND_MAX = 100
BLEND_DEFAULT = -50
# Midpoint and half-span of the retired grid_lightness value space, read by
# blend_for_lightness() for a config written before this slider existed.
LEGACY_LIGHTNESS_MID = 120

# Thickness stays a stop space, like settings.ELEV_STEP_PCT_STOPS, so its
# slider's value space is the stop index.
THICKNESS_STOPS: tuple[int, ...] = (1, 2, 3, 4)  # device pixels
THICKNESS_DEFAULT = 1

# The two ranks separate by alpha, not lightness: a lightness split would eat
# one end of the slider's own range. These are the ceilings a full deflection
# reaches; everything between scales down from them.
MINOR_ALPHA = 90
MAJOR_ALPHA = 170


def _snap(value: int, stops: tuple[int, ...]) -> int:
    return min(stops, key=lambda stop: (abs(stop - value), stop))


def clamp_blend(value: int) -> int:
    return max(BLEND_MIN, min(BLEND_MAX, int(value)))


def blend_for_lightness(lightness: int) -> int:
    """A legacy grid_lightness (a 0..240 grey) as a blend, so a config
    written before this slider keeps roughly the look it had: the greys below
    the midpoint were darkening the view, the ones above lightening it."""
    return clamp_blend(round((lightness - LEGACY_LIGHTNESS_MID) / LEGACY_LIGHTNESS_MID * BLEND_MAX))


def snap_thickness(value: int) -> int:
    return _snap(value, THICKNESS_STOPS)


def thickness_index(value: int) -> int:
    return THICKNESS_STOPS.index(snap_thickness(value)) + 1


def thickness_for_index(index: int) -> int:
    return THICKNESS_STOPS[index - 1]


def grid_colors(blend: int) -> tuple[RGBA, RGBA]:
    """(minor, major) RGBA for a blend strength: black below zero, white
    above, both fully transparent at zero."""
    value = clamp_blend(blend)
    channel = 255 if value > 0 else 0
    strength = abs(value) / BLEND_MAX
    return (
        (channel, channel, channel, round(MINOR_ALPHA * strength)),
        (channel, channel, channel, round(MAJOR_ALPHA * strength)),
    )


# --- Follow Terrain Elevation ------------------------------------------------
#
# draped_lines() is deliberately a separate function rather than a flag on
# grid_axes(): the ground-plane lattice must stay at elevation 0 alongside the
# ruler and the tick strip, which are elevation-0 by design.

# Tiles of slack around the visible rect, so a scroll does not rebuild per pixel.
WINDOW_PAD_TILES = 8

Segments = np.ndarray  # (n, 4) int64: x0, y0, x1, y1


def corner_point(cx: int, cy: int, rise: int, proj: iso_geometry.IsoProjection) -> Point:
    """Scene point of grid corner (cx, cy) lifted `rise` canvas pixels off
    the ground plane. rise=0 is exactly edge_ticks.iso_corner()."""
    ox, oy = iso_geometry.tile_screen_origin(cx, cy, 0, proj)
    return ox, oy + proj.half_h - rise


def _corner_points(cx: np.ndarray, cy: np.ndarray, rise: np.ndarray, proj) -> tuple[np.ndarray, np.ndarray]:
    sx = proj.origin_x + (cx + cy) * proj.half_w
    sy = proj.origin_y + (cy - cx) * proj.half_h + proj.half_h - rise
    return sx, sy


def draped_lines(
    tiles: np.ndarray,
    style: str,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray | None = None,
    corner_rise: np.ndarray | None = None,
) -> tuple[Segments, Segments]:
    """(minor, major) grid segments over `tiles`, an (n, 2) int (x, y) array,
    lifted onto the terrain.

    Stepped is per-tile slabs: a tile's four corners all sit at its own
    elevation, so a boundary between two differing neighbours is drawn twice,
    once at each side's height, which is what the cliff between them looks
    like. Sloped is one shared lattice through corner_rise, whose blended
    corners make every shared edge identical from both sides.

    Each tile contributes its x-low and y-low edges, plus its x-high / y-high
    edge at the map border or, in Stepped, where the neighbour's height
    differs; so no edge is ever emitted twice at the same place. Collinear
    runs along a grid line are then merged, which makes a flat map's output
    exactly grid_axes()' lines.

    Stepped drops the copies a nearer tile's slab would bury. The overlay has
    no depth test, so without this a behind-and-lower tile's copy paints on
    top of the front tile it sits inside -- and elev_step is only half half_h
    by default, so it lands INSIDE that tile's face, reading as a grid line
    drawn mid-tile rather than as the cliff below it."""
    tiles = np.asarray(tiles, dtype=np.int64).reshape(-1, 2)
    empty = np.zeros((0, 4), dtype=np.int64)
    if style == "stepped":
        if elevations is None or corner_rise is not None:
            raise ValueError("stepped drapes from elevations only")
        map_h, map_w = elevations.shape
    elif style == "sloped":
        if corner_rise is None or elevations is not None:
            raise ValueError("sloped drapes from corner_rise only")
        map_h, map_w = corner_rise.shape[0] - 1, corner_rise.shape[1] - 1
    else:
        raise ValueError(f"no draped grid for style {style!r}")
    if len(tiles) == 0:
        return empty, empty
    x, y = tiles[:, 0], tiles[:, 1]

    # Rows of (orient, line, pos, r_start, r_end): orient 0 is a line of
    # constant x = line running along y = pos .. pos+1, orient 1 the transpose.
    rows = []
    if style == "stepped":
        rise = elevations[y, x].astype(np.int64) * proj.elev_step
        # depth_order paints ascending d = y - x, so the WEST and SOUTH
        # neighbours are in front of this tile and the EAST and NORTH ones
        # behind it. A boundary's two copies are emitted only where the one
        # in front cannot bury the other.
        west = x == 0
        inner = ~west
        keep = ~(inner & (elevations[y, np.maximum(x - 1, 0)] > elevations[y, x]))
        rows.append(np.stack([np.zeros_like(x[keep]), x[keep], y[keep], rise[keep], rise[keep]], axis=1))
        rows.append(np.stack([np.ones_like(x), y, x, rise, rise], axis=1))
        east = x + 1 == map_w
        inner = ~east
        east |= inner & (elevations[y, np.minimum(x + 1, map_w - 1)] != elevations[y, x])
        south = y + 1 == map_h
        inner = ~south
        south |= inner & (elevations[np.minimum(y + 1, map_h - 1), x] < elevations[y, x])
        rows.append(np.stack([np.zeros_like(x[east]), x[east] + 1, y[east], rise[east], rise[east]], axis=1))
        rows.append(np.stack([np.ones_like(x[south]), y[south] + 1, x[south], rise[south], rise[south]], axis=1))
    else:
        cr = corner_rise.astype(np.int64)
        rows.append(np.stack([np.zeros_like(x), x, y, cr[y, x], cr[y + 1, x]], axis=1))
        rows.append(np.stack([np.ones_like(x), y, x, cr[y, x], cr[y, x + 1]], axis=1))
        east = x + 1 == map_w
        south = y + 1 == map_h
        xe, ye = x[east], y[east]
        rows.append(np.stack([np.zeros_like(xe), xe + 1, ye, cr[ye, xe + 1], cr[ye + 1, xe + 1]], axis=1))
        xs, ys = x[south], y[south]
        rows.append(np.stack([np.ones_like(xs), ys + 1, xs, cr[ys + 1, xs], cr[ys + 1, xs + 1]], axis=1))
    edges = np.concatenate(rows)

    # Merge: collinear edges share (orient, line, slope, intercept) and are
    # consecutive in pos.
    orient, line, pos, r0, r1 = edges.T
    slope = r1 - r0
    intercept = r0 - pos * slope
    order = np.lexsort((pos, intercept, slope, line, orient))
    orient, line, pos, r0, r1, slope, intercept = (
        a[order] for a in (orient, line, pos, r0, r1, slope, intercept)
    )
    same = (
        (orient[1:] == orient[:-1])
        & (line[1:] == line[:-1])
        & (slope[1:] == slope[:-1])
        & (intercept[1:] == intercept[:-1])
        & (pos[1:] == pos[:-1] + 1)
    )
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    ends = np.concatenate((starts[1:], [len(pos)])) - 1

    o, ln = orient[starts], line[starts]
    p0, p1 = pos[starts], pos[ends] + 1
    rs, re = r0[starts], r1[ends]
    # orient 0: corner (line, pos); orient 1: corner (pos, line).
    cx0 = np.where(o == 0, ln, p0)
    cy0 = np.where(o == 0, p0, ln)
    cx1 = np.where(o == 0, ln, p1)
    cy1 = np.where(o == 0, p1, ln)
    sx0, sy0 = _corner_points(cx0, cy0, rs, proj)
    sx1, sy1 = _corner_points(cx1, cy1, re, proj)
    segments = np.stack([sx0, sy0, sx1, sy1], axis=1).astype(np.int64)
    major = ln % edge_ticks.MAJORS_PER_MINOR == 0
    return segments[~major], segments[major]
