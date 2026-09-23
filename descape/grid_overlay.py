"""Qt-free geometry, LOD and colours for View > Grid.

Two consumers. The chunk compositors bake the grid per tile from GridBake and
owned_edges() (Follow Terrain Elevation on, and always in Flat). The Qt half,
viewer_canvas.GridItem, draws grid_axes()' ground lattice for the follow-off
mode and the Settings > Appearance slider preview.

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
# Baked: canvas px at the composited mip, so ~0.5x-1x that on screen. As the
# follow-off overlay: device px.
THICKNESS_STOPS: tuple[int, ...] = (1, 2, 3, 4)
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


# --- The baked grid ------------------------------------------------------------
#
# With Follow Terrain Elevation on (and always in Flat) the grid is not a scene
# item: each chunk compositor draws it per tile, between that tile's own
# terrain and its own units, so a sprite is never crossed by a line and nearer
# raised terrain occludes a farther one by painter's order alone.


@dataclass(frozen=True)
class GridBake:
    """What a chunk compositor bakes into its pixels. Frozen and compared by
    value, like view_layers.LayerState, so a no-op set_grid() costs nothing.
    `thickness` is in CANVAS pixels at the mip being composited."""

    enabled: bool = False
    minor: RGBA = (0, 0, 0, 0)
    major: RGBA = (0, 0, 0, 0)
    thickness: int = THICKNESS_DEFAULT

    @property
    def paints(self) -> bool:
        return self.enabled and (self.minor[3] > 0 or self.major[3] > 0)


DEFAULT_GRID = GridBake()


def grid_bake(enabled: bool, blend: int, thickness: int) -> GridBake:
    minor, major = grid_colors(blend)
    return GridBake(enabled=bool(enabled), minor=minor, major=major, thickness=snap_thickness(thickness))


def owned_edges(
    tx: int, ty: int, map_w: int, map_h: int, elevations: np.ndarray | None = None
) -> tuple[tuple[str, bool], ...]:
    """The (edge, is_major) pairs tile (tx, ty) draws. Every tile owns its
    x-low and y-low edges; x-high and y-high only at the map border or, given
    Stepped `elevations`, where that neighbour's height differs (the cliff
    lip). Owning two sides is what composites a shared edge once, not twice."""
    edges = [("x_low", is_major(tx)), ("y_low", is_major(ty))]
    here = None if elevations is None else elevations[ty, tx]
    if tx + 1 == map_w or (here is not None and elevations[ty, tx + 1] != here):
        edges.append(("x_high", is_major(tx + 1)))
    if ty + 1 == map_h or (here is not None and elevations[ty + 1, tx] != here):
        edges.append(("y_high", is_major(ty + 1)))
    return tuple(edges)


def bake_lod(minor_spacing_canvas_px: float) -> GridLod:
    """grid_lod() on one mip's own minor spacing at residual magnification
    1.0, so it never drops a line that is genuinely visible. On the shipped
    ladders (MIP_MIN_TILE_PIXELS = 16) it never fires: a guard for a future
    ladder, not a live behaviour."""
    return grid_lod(minor_spacing_canvas_px)
