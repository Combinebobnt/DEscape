"""Qt-free placement geometry for the ruler-style distance ticks drawn in
the void just outside the map's own border. The Qt half is
viewer.EdgeTickItem, which maps these scene-space anchors through the
painter's world transform and then draws every mark in device coordinates.

Split the way iso_geometry.ground_outline_corners already is: this module
decides WHERE a mark belongs in scene space, the item decides how it looks
on screen. Nothing here imports PyQt5, so the whole placement layer is
testable without a QApplication, and none of it can reach descape.render or
the chunk caches.

Anchors are grid CORNERS, not tile centers, and always at elevation 0. That
is the same ground-plane reference ground_outline_corners uses, and for the
same reason: anchoring to the rendered silhouette instead would make the
ruler a ragged staircase that moved whenever a border tile was raised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from descape import iso_geometry

# Minor tick every N tiles, an exclusive pair surfaced as View > Distance
# Ticks. 5 is the default because majors then land every 20 tiles, hitting
# the far edge exactly on 5 of the 7 real AoE2:DE map sizes
# (scenario_new.STANDARD_MAP_SIZES), where every-4's 16-tile majors hits
# only 3 of 7 (144, 240, 480).
TICK_INTERVALS: tuple[int, ...] = (4, 5)
TICK_INTERVAL_DEFAULT = 5

# Every Nth minor tick is a major, and only majors carry a number.
MAJORS_PER_MINOR = 4

# Mark sizes, in DEVICE pixels, which is the whole point of the item's
# reset-transform trick. Scene-space lengths would be sheared and rotated by
# Flat's scale(1, 0.5) + rotate(-45), and sub-pixel at fit-to-view on a
# 480x480 (fit scale there is about 0.04).
MINOR_TICK_PX = 6.0
MAJOR_TICK_PX = 12.0
LABEL_GAP_PX = 3.0
LABEL_FONT_PX = 12
# The box a major's number is centered in and drawn into. Sized for the
# widest label any real map produces (3 digits, 480 being the largest of
# STANDARD_MAP_SIZES) at LABEL_FONT_PX, with room to spare.
LABEL_BOX_W_PX = 40.0
LABEL_BOX_H_PX = 16.0
# How far out along the outward ray that box's CENTER sits: clear of the
# major tick, plus the gap, plus half the box so its near edge is the thing
# the gap actually measures from.
LABEL_CENTER_PX = MAJOR_TICK_PX + LABEL_GAP_PX + LABEL_BOX_H_PX / 2.0

# The furthest any drawn pixel can land from its anchor: the label box's own
# far corner. Every other mark is strictly nearer, which is what makes this
# the single number scene_pad() has to cover.
DEVICE_REACH_PX = LABEL_CENTER_PX + math.hypot(LABEL_BOX_W_PX, LABEL_BOX_H_PX) / 2.0

# Multiplier on the scene-space pad, and NOT a round-number fudge. Under
# Flat's scale(1, 0.5) then rotate(-45) the transform's smallest singular
# value is sqrt(0.5) of the sqrt-determinant scale the caller measures and
# passes in, so a pad derived from that scale must be at least sqrt(2) times
# the device reach to still contain it along the squashed axis, a floor of
# 1.4142 that 2.0 clears with margin.
PAD_SAFETY = 2.0

# Level-of-detail ladder, all measured on the DEVICE-space gap between
# adjacent minor ticks: labels go first (they collide soonest), then the
# minors, then the edge stops being drawn at all. The thresholds are
# deliberately expressed against different quantities, since a label needs
# room next to the major it belongs to while a minor tick only needs to be
# distinguishable from its neighbour.
MIN_LABEL_SPACING_PX = 32.0  # between MAJORS
MIN_MINOR_SPACING_PX = 5.0  # between MINORS
MIN_MAJOR_SPACING_PX = 6.0  # between MAJORS


@dataclass(frozen=True)
class TickLod:
    """What a given on-screen density leaves worth drawing. Purely derived
    from the spacing, with no hysteresis, so the same view state always
    yields the same marks."""

    draw_edge: bool
    draw_minors: bool
    draw_labels: bool


def tick_lod(minor_spacing_px: float) -> TickLod:
    """The LOD verdict for a device-space gap of `minor_spacing_px` between
    adjacent minor ticks. Monotonic by construction: labels drop at a wider
    spacing than minors do, and the edge itself drops last."""
    major_spacing_px = minor_spacing_px * MAJORS_PER_MINOR
    draw_edge = major_spacing_px >= MIN_MAJOR_SPACING_PX
    return TickLod(
        draw_edge=draw_edge,
        draw_minors=draw_edge and minor_spacing_px >= MIN_MINOR_SPACING_PX,
        draw_labels=draw_edge and major_spacing_px >= MIN_LABEL_SPACING_PX,
    )


def scene_pad(scale: float) -> float:
    """Scene units the overlay's bounding rect must extend past the map on
    every side, for a view whose sqrt-determinant scale is `scale`.

    Constant device length means the scene-unit overhang grows without bound
    as the view zooms out, so this cannot be a fixed fraction of the map:
    at mip 8 on an 800px viewport the true overhang is 1.2x the canvas
    width, three times MapView.OVERSCROLL_FRACTION. Callers must therefore
    pass the smallest scale the view can reach, not its current one.

    A non-positive scale has no meaningful pad; 0.0 is returned rather than
    dividing by it, and the caller keeps whatever pad it already had."""
    if scale <= 0:
        return 0.0
    return PAD_SAFETY * DEVICE_REACH_PX / scale


@dataclass(frozen=True)
class EdgeRun:
    """One map edge's full set of ticks, in scene space.

    `edge` is "y0"/"y1" for the two edges along which x varies (an X ruler)
    and "x0"/"x1" for the two along which y varies. `outward` and
    `minor_step` are unnormalized scene-space directions; the item maps both
    as deltas and normalizes in device space, which is what survives Flat's
    rotate-and-squash."""

    edge: str
    anchors: tuple[tuple[int, int], ...]
    tiles: tuple[int, ...]
    majors: tuple[bool, ...]
    outward: tuple[float, float]
    minor_step: tuple[float, float]


def tick_indices(span: int, interval: int) -> tuple[int, ...]:
    """Grid-corner indices carrying a tick along an edge `span` tiles long.
    Inclusive of `span` itself only when the interval divides it, which is
    exactly when the ruler ends flush with the far corner."""
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval!r}")
    return tuple(range(0, span + 1, interval))


def is_major(index: int, interval: int) -> bool:
    """Whether the tick at grid index `index` is a major. Majors land every
    MAJORS_PER_MINOR minors, so index 0 is always one."""
    return index % (interval * MAJORS_PER_MINOR) == 0


def iso_corner(i: int, j: int, proj: iso_geometry.IsoProjection) -> tuple[int, int]:
    """Screen point of grid corner (i, j) at elevation 0, with i in
    [0, map_w] and j in [0, map_h].

    This is ground_outline_corners' own `corner()` generalised off the four
    extremes: the west tip of the diamond that tile (i, j) WOULD occupy.
    tile_screen_origin is pure linear arithmetic with no bounds check, so
    i == map_w and j == map_h are legal inputs, and ground_outline_corners
    already leans on exactly that. Integer throughout, no rounding."""
    ox, oy = iso_geometry.tile_screen_origin(i, j, 0, proj)
    return ox, oy + proj.half_h


def _iso_runs(map_w: int, map_h: int, interval: int, proj: iso_geometry.IsoProjection) -> tuple[EdgeRun, ...]:
    half_w, half_h = float(proj.half_w), float(proj.half_h)
    # Screen partials of the grid axes, read straight off tile_screen_origin:
    # sx = origin_x + (x+y)*half_w and sy = origin_y + (y-x)*half_h.
    d_x = (half_w, -half_h)
    d_y = (half_w, half_h)
    step_x = (d_x[0] * interval, d_x[1] * interval)
    step_y = (d_y[0] * interval, d_y[1] * interval)

    xs = tick_indices(map_w, interval)
    ys = tick_indices(map_h, interval)
    majors_x = tuple(is_major(i, interval) for i in xs)
    majors_y = tuple(is_major(j, interval) for j in ys)

    return (
        EdgeRun("y0", tuple(iso_corner(i, 0, proj) for i in xs), xs, majors_x,
                (-d_y[0], -d_y[1]), step_x),
        EdgeRun("y1", tuple(iso_corner(i, map_h, proj) for i in xs), xs, majors_x,
                d_y, step_x),
        EdgeRun("x0", tuple(iso_corner(0, j, proj) for j in ys), ys, majors_y,
                (-d_x[0], -d_x[1]), step_y),
        EdgeRun("x1", tuple(iso_corner(map_w, j, proj) for j in ys), ys, majors_y,
                d_x, step_y),
    )


def _flat_runs(map_w: int, map_h: int, interval: int, tile_px: int) -> tuple[EdgeRun, ...]:
    # FlatChunkCache.canvas_dims() is (map_w*tile_px, map_h*tile_px), so in
    # Flat the scene-space map is a plain axis-aligned square and its diamond
    # look is purely MapView's own view transform.
    xs = tick_indices(map_w, interval)
    ys = tick_indices(map_h, interval)
    majors_x = tuple(is_major(i, interval) for i in xs)
    majors_y = tuple(is_major(j, interval) for j in ys)
    step = float(tile_px * interval)

    return (
        EdgeRun("y0", tuple((i * tile_px, 0) for i in xs), xs, majors_x,
                (0.0, -1.0), (step, 0.0)),
        EdgeRun("y1", tuple((i * tile_px, map_h * tile_px) for i in xs), xs, majors_x,
                (0.0, 1.0), (step, 0.0)),
        EdgeRun("x0", tuple((0, j * tile_px) for j in ys), ys, majors_y,
                (-1.0, 0.0), (0.0, step)),
        EdgeRun("x1", tuple((map_w * tile_px, j * tile_px) for j in ys), ys, majors_y,
                (1.0, 0.0), (0.0, step)),
    )


def edge_runs(
    map_w: int,
    map_h: int,
    interval: int,
    proj: iso_geometry.IsoProjection | None = None,
    tile_px: int | None = None,
) -> tuple[EdgeRun, ...]:
    """All four edges' ticks, in (y0, y1, x0, x1) order.

    Dispatches on `proj` the way MapView.set_source already does: Stepped and
    Sloped share the iso diamond and pass one, Flat passes `tile_px` instead.
    Ticks point ALONG the grid axis they measure rather than perpendicular to
    the edge, so the ruler reads as the grid continuing past its own border.

    Built once per (style, dims, interval) and cached by the caller. The
    padded bounding rect spans the viewport, so Qt treats the overlay as
    exposed on essentially every repaint, including every invalidate_region
    during a drag-paint stroke; a 480x480 at interval 4 is 484 ticks."""
    if proj is not None:
        return _iso_runs(map_w, map_h, interval, proj)
    if tile_px is None:
        raise ValueError("edge_runs needs either proj (iso) or tile_px (flat)")
    return _flat_runs(map_w, map_h, interval, tile_px)
