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
terrain_overhang() is the other side of that choice: the raised silhouette
past that border, which the item clips out when the ruler is drawn behind
elevated terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
# The Settings > Appearance default; the live value is settings.
# get_distance_tick_font_px(). Every constant below that used to be sized
# off this one directly is now a function of an arbitrary font_px instead,
# so a config with no distance_tick_font_px key reproduces these exactly.
LABEL_FONT_PX = 12


def label_box_px(font_px: int) -> tuple[float, float]:
    """The box a major's number is centered in and drawn into, sized for
    the widest label any real map produces (3 digits, 480 being the largest
    of STANDARD_MAP_SIZES) at font_px, with room to spare.

    Multiply-before-divide, not a ratio constant: font_px * 40 / 12 computes
    480 / 12, and 12 divides 480 and 192 evenly, so this is exact at every
    integer font_px in [8, 24] (settings.DISTANCE_TICK_FONT_PX_MIN/_MAX) --
    (40 / 12) * font_px disagrees by a ULP at 10, 14 and 20."""
    return font_px * 40.0 / 12.0, font_px * 16.0 / 12.0


def label_center_px(font_px: int) -> float:
    """How far out along the outward ray a major's label box CENTER sits:
    clear of the major tick, plus the gap, plus half the box so its near
    edge is the thing the gap actually measures from."""
    return MAJOR_TICK_PX + LABEL_GAP_PX + label_box_px(font_px)[1] / 2.0


def label_reach_px(font_px: int) -> float:
    """How far out a major's numeric label box's own far corner can land."""
    return label_center_px(font_px) + math.hypot(*label_box_px(font_px)) / 2.0


# Clearance between a numeric label's reach and the axis letter's reach.
AXIS_LABEL_GAP_PX = 3.0


def axis_letter(edge: str) -> str:
    """The game-engine axis an edge's numbers measure. The y0/y1 edges are
    the ones along which x varies, per EdgeRun, and AoE2:DE calls that the X
    axis (see AoE2ScenarioParser's docs/images/map_coordinates_explained.png,
    drawn on a real DE editor screenshot)."""
    if edge in ("y0", "y1"):
        return "X"
    if edge in ("x0", "x1"):
        return "Y"
    raise ValueError(f"unknown edge {edge!r}")


def axis_label_box_px(font_px: int) -> tuple[float, float]:
    """The square box one axis letter is centered in, the height of a
    numeric label box. Multiply-before-divide for the same exactness."""
    side = font_px * 16.0 / 12.0
    return side, side


def axis_label_center_px(font_px: int) -> float:
    """How far out along the outward ray the axis letter's box center sits.

    Measured past both boxes' half-diagonals rather than their half-heights:
    the outward ray is diagonal in every iso view, so an axis-aligned number
    box reaches along it by up to its half-diagonal, and a letter placed off
    half-heights overlaps a neighbouring major's number."""
    return (
        label_reach_px(font_px)
        + AXIS_LABEL_GAP_PX
        + math.hypot(*axis_label_box_px(font_px)) / 2.0
    )


def axis_label_reach_px(font_px: int) -> float:
    return axis_label_center_px(font_px) + math.hypot(*axis_label_box_px(font_px)) / 2.0


def device_reach_px(font_px: int) -> float:
    """The furthest any drawn pixel can land from its anchor: the larger of
    the numeric label's and the axis letter's far corners. Every other mark
    is strictly nearer, which is what makes this the single number
    scene_pad() has to cover."""
    return max(label_reach_px(font_px), axis_label_reach_px(font_px))

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


def scene_pad(scale: float, font_px: int) -> float:
    """Scene units the overlay's bounding rect must extend past the map on
    every side, for a view whose sqrt-determinant scale is `scale` and a
    label font of `font_px`.

    Constant device length means the scene-unit overhang grows without bound
    as the view zooms out, so this cannot be a fixed fraction of the map:
    at mip 8 on an 800px viewport the true overhang is 1.2x the canvas
    width, three times MapView.OVERSCROLL_FRACTION. Callers must therefore
    pass the smallest scale the view can reach, not its current one.

    A non-positive scale has no meaningful pad; 0.0 is returned rather than
    dividing by it, and the caller keeps whatever pad it already had."""
    if scale <= 0:
        return 0.0
    return PAD_SAFETY * device_reach_px(font_px) / scale


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


# --- Settings > Appearance "Draw distance ticks behind elevated terrain" ----

# How far below the elevation-0 border an overhang hole reaches. A tick's
# root pixel has its centre just inside the map, so a hole that stopped at
# the border left that one pixel as a stray dot on the terrain.
ROOT_PX = 1


def stepped_corner_rise(elevations: np.ndarray, proj: iso_geometry.IsoProjection) -> np.ndarray:
    """Stepped's (h+1, w+1) per-corner rise in px: the highest flat top face
    touching each grid corner, elevation * elev_step over its up-to-4 tiles.
    Elevations are never negative, so a zero pad stands in for off-map."""
    # In uint8 (elevations stop at iso_geometry.MAX_ELEVATION): about 25x
    # faster than int64 on a 480x480, which keeps an edit's refresh near 1 ms.
    padded = np.pad(elevations.astype(np.uint8), 1)
    top = np.maximum(np.maximum(padded[:-1, :-1], padded[:-1, 1:]), np.maximum(padded[1:, :-1], padded[1:, 1:]))
    return top.astype(np.int64) * proj.elev_step


def _band_depth(corner_rise: np.ndarray, proj: iso_geometry.IsoProjection) -> int:
    """Rows in from an upper edge that terrain can still reach the border
    from: depth d sits 2 * d * half_h below it, plus one row of margin. 0
    when nothing rises at all."""
    max_rise = int(corner_rise.max()) if corner_rise.size else 0
    if max_rise <= 0:
        return 0
    return -(-max_rise // (2 * proj.half_h)) + 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each maximal True run."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist(), strict=True))


def _stepped_overhang(corner_rise: np.ndarray, proj: iso_geometry.IsoProjection) -> tuple[np.ndarray, ...]:
    h1, w1 = corner_rise.shape
    w, h = w1 - 1, h1 - 1
    depth = _band_depth(corner_rise, proj)
    if depth == 0:
        return ()
    band = np.zeros(corner_rise.shape, dtype=bool)
    band[: depth + 1, :] = True
    band[:, max(0, w - depth) :] = True
    j, i = np.nonzero(band)
    k = i + j
    lifted = proj.origin_y + (j - i) * proj.half_h + proj.half_h - corner_rise[j, i]
    env = np.full(w + h + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(env, k, lifted)
    lattice = np.arange(w + h + 1)
    # The upper boundary: corner (k, 0) up to the top tip, then (w, k - w).
    ground = proj.origin_y + proj.half_h - proj.half_h * np.minimum(lattice, 2 * w - lattice)
    xs = (proj.origin_x + lattice * proj.half_w).astype(np.float64)
    polys = []
    for start, end in _runs(env < ground):
        # Bracketed by the lattice points either side, so the edge crossings are covered.
        lo, hi = max(start - 1, 0), min(end + 1, w + h)
        top = np.column_stack((xs[lo : hi + 1], env[lo : hi + 1]))
        bottom = np.column_stack((xs[lo : hi + 1], ground[lo : hi + 1] + ROOT_PX))[::-1]
        polys.append(np.concatenate((top, bottom)).astype(np.float64))
    return tuple(polys)


def sloped_envelope(
    corner_rise: np.ndarray, proj: iso_geometry.IsoProjection
) -> tuple[np.ndarray, np.ndarray]:
    """(env, ground), one entry per scene-px column of the canvas: the
    topmost painted Sloped row over the band tiles, and the topmost row the
    same tiles paint at elevation 0. A column no band tile covers holds the
    int64 max in both.

    Each top is _sloped_column_runs' run_start, broadcast over the tiles,
    placed at the base_x/base_y render_sloped paints at (elevation-0
    tile_screen_origin, less d_min). test_edge_ticks.py pins it against
    iso_geometry.sloped_tile_outline, so the two cannot drift silently."""
    h1, w1 = corner_rise.shape
    w, h = w1 - 1, h1 - 1
    half_w, half_h = proj.half_w, proj.half_h
    big = np.iinfo(np.int64).max
    env = np.full(proj.canvas_w, big, dtype=np.int64)
    ground = np.full(proj.canvas_w, big, dtype=np.int64)
    cols, tops, _lens, _starts, _order = iso_geometry._diamond_column_runs(proj.tile_px)
    tops = tops[cols]

    def tiles(depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = np.zeros((h, w), dtype=bool)
        mask[:depth, :] = True
        mask[:, max(0, w - depth) :] = True
        y, x = np.nonzero(mask)
        base_x = (proj.origin_x + (x + y) * half_w)[:, None] + cols
        return y, x, base_x, proj.origin_y + (y - x) * half_h

    _y, _x, base_x, base_y = tiles(1)
    np.minimum.at(ground, base_x.ravel(), (base_y[:, None] + tops).ravel())

    y, x, base_x, base_y = tiles(max(_band_depth(corner_rise, proj), 1))
    nw, ne = corner_rise[y, x], corner_rise[y, x + 1]
    sw, se = corner_rise[y + 1, x], corner_rise[y + 1, x + 1]
    d_min = np.minimum(np.minimum(nw, ne), np.minimum(sw, se))
    nw, ne, se = (nw - d_min)[:, None], (ne - d_min)[:, None], (se - d_min)[:, None]
    left = cols < half_w
    k = np.where(left, cols, cols - half_w)
    a_up = np.where(left, nw, ne)
    b_up = np.where(left, ne, se)
    r_up = a_up + iso_geometry._round_div(2 * k * (b_up - a_up), 2 * (half_w - 1))
    np.minimum.at(env, base_x.ravel(), ((base_y - d_min)[:, None] + tops - r_up).ravel())
    return env, ground


def _staircase(start: int, values: np.ndarray) -> np.ndarray:
    """The pixel-exact top edge of columns start, start+1, ...: one
    horizontal segment per run of equal values."""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    seg_start = np.concatenate(([0], change))
    seg_end = np.concatenate((change, [len(values)]))
    level = values[seg_start]
    return np.column_stack(
        (np.column_stack((seg_start, seg_end)).ravel() + start, np.repeat(level, 2))
    ).astype(np.float64)


def _sloped_overhang(corner_rise: np.ndarray, proj: iso_geometry.IsoProjection) -> tuple[np.ndarray, ...]:
    if _band_depth(corner_rise, proj) == 0:
        return ()
    env, ground = sloped_envelope(corner_rise, proj)
    polys = []
    for start, end in _runs(env < ground):
        top = _staircase(start, env[start : end + 1])
        bottom = _staircase(start, ground[start : end + 1] + ROOT_PX)[::-1]
        polys.append(np.concatenate((top, bottom)))
    return tuple(polys)


def terrain_overhang(
    corner_rise: np.ndarray, proj: iso_geometry.IsoProjection, sloped: bool = False
) -> tuple[np.ndarray, ...]:
    """Scene-space polygons, each a float64 (N, 2) array, covering the
    projected terrain that lies outside the elevation-0 border the ruler
    is anchored on, down to ROOT_PX below that border. `corner_rise` is a
    (h+1, w+1) per-grid-corner rise in px: Sloped's cache.corner_rise (with
    `sloped`), or Stepped's stepped_corner_rise(). Rise only lifts terrain,
    so only the two upper edges (y0 and x1) can ever overhang, and a flat
    elevation-0 map gives ().

    Stepped works on the half_w corner lattice: per lattice x the highest
    lifted corner against the border corner's elevation-0 y. Each top-face
    edge spans one lattice interval, so interpolating those minima sits at
    or above the true envelope: the clip hides slightly more ruler, never
    less. Sloped cuts each screen column on its own, so it takes the
    per-px-column tops instead (sloped_envelope()). That is exact at the
    target level; a coarser Sloped level paints from its own corner_rise at
    its own projection, so there the clip can sit up to one level pixel off.

    A map whose minimum elevation is above 0 overhangs along both upper
    edges in full, by min_elev * elev_step scene px. That hides that much of
    every tick there, which is intended: the terrain really is painted over
    that strip."""
    if sloped:
        return _sloped_overhang(corner_rise, proj)
    return _stepped_overhang(corner_rise, proj)
