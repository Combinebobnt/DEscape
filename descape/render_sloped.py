"""The Sloped render: slope shade, tile quads, pick ids, marks and candidates, terrain, and the colour and pick-id rect composites."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from descape import (
    composite_backend,
    iso_geometry,
    settings,
    terrain_blend,
)
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render_bbox import BystanderGrid, _rect_candidates
from descape.render_paint import (
    INK_NONE,
    _clipped_paint,
    _clipped_paint_rgba,
    _clipped_paint_solid,
    _dims_and_elevations,
    _full_blend_keys,
    _paint_grid_iso,
    _record_ink,
    _rect_blend_keys,
    _restore_under,
    _terrain_grid_and_elevations,
    _top_block,
    tile_pixels_for_map,
)
from descape.render_sprites import _FARM_EDGE_BITS, SpriteLayer, _ink_floor, sprite_draws_by_anchor
from descape.render_units import NON_BUILDING_SPAN, _units_by_tile, unit_paint_offset
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    tile_span,
)
from descape.view_layers import DEFAULT_LAYERS, LayerState

# Phase 6 (Sloped): which corner_rise_px() rule is active. A named policy
# constant, not a structural choice -- switching it never touches
# sloped_quad_indices or anything downstream (see that function's own
# docstring).
#
# "max", MEASURED off the 2026-08-22 in-game captures, replacing the
# "average" that the 2026-08-09 preliminary read had guessed. The decisive
# evidence is a null result: a one-tile-lower pit inside a raised block
# renders in DE with a perfectly regular tile grid over it, no dimple at
# all. "min" would sink that corner a full level and "average" a quarter of
# one; only "max" predicts nothing. Eight further regions agree, each fit
# independently.
#
# One consequence worth knowing: with "max" over integer elevations every
# corner value is an integer number of levels, so a tile's shape is always
# one of a finite set -- which is what DE's own pre-authored slope tiles
# are. The general quad path reproduces that exactly; it is a superset, not
# an approximation of it.
SLOPE_CORNER_RULE = "max"

# Sloped has no skirts and no contact shadow to fall back on (adjacent
# tiles share corner heights by construction -- see corner_rise_px's own
# docstring -- so there is no vertical face left for either to draw): a
# directional (Lambert-style) shade is the user's explicit replacement
# depth cue (a deliberate design choice), NOT the
# direction-independent "slope magnitude only" CONTACT_SHADE's own comment
# argues for elsewhere in this module -- that argument doesn't carry over
# here because Sloped has no other depth cue left to fall back on, so this
# is a deliberate, confined departure, not an oversight.
#
# Anchored at horizontal, not textbook absolute Lambert
# (ambient + (1-ambient)*dot(n,L)): shade = 1 + STRENGTH*(dot(n,L) -
# dot(up,L)), i.e. the darkening is relative to what a FLAT face would
# already give under this same light, not to n.L in absolute terms.
# Written the textbook way instead, a flat map would come out uniformly
# darkened (or brightened) by dot(up,L) < 1, breaking
# render_terrain_sloped's flat-map byte-identity oracle against
# render_terrain_iso -- this form gives exactly shade=1.0 at n=up (a flat
# tile's normal) for ANY light direction, since dot(n,L)-dot(up,L) == 0
# there, while still being genuinely directional for anything tilted (a
# face tilted toward the light brightens, away from it darkens -- unlike a
# magnitude-only cue, which would shade a hill's two opposite faces
# identically).
#
# MEASURED off the 2026-08-22 in-game captures, replacing the look-and-feel
# guess that shipped until 2026-08-23. Both halves of that guess were wrong:
#
#   * its horizontal direction was INVERTED. It leaned toward -mapx/-mapy
#     (screen left); DE's light leans toward +mapx/+mapy (screen right and
#     slightly down). On a +mapx ridge the old constants left the flank DE
#     darkens almost unshaded (0.997) and darkened the one DE brightens
#     (0.824), where DE itself is at 0.727 / 1.186.
#   * its strength was too low AND too overhead to reach DE's range. At
#     lz = 0.869 the anchored form caps brightening at STRENGTH*(1-lz) =
#     +4.6%, so the lit side was essentially invisible. DE's light is far
#     more grazing (lz = 0.351), which is what buys a symmetric response.
#
# The old constants scored WORSE than no shading at all (weighted rms 0.177
# against the null's 0.130) precisely because they shaded the wrong side. The
# fitted ones score 0.050, and beat both the null and a direction-independent
# magnitude-only rival on all seven held-out regions -- which is what makes
# "DE is directional" a measurement rather than an eyeball.
SLOPE_LIGHT_DIR = tuple(
    c / math.sqrt(0.54**2 + 0.76**2 + 0.35**2) for c in (0.54, 0.76, 0.35)
)  # low sun leaning toward +mapx/+mapy, i.e. screen right; measured, not chosen
SLOPE_SHADE_STRENGTH = 0.425
# Guard rails, NOT part of the calibration. At the fitted strength the model
# spans [0.531, 1.276] over all 81 corner configs ([0.617, 1.257] over the
# one-level configs ordinary terrain actually produces), so these sit just
# outside that and never fire on legal geometry. They did not fire at the old
# strength either -- 0.6/1.25 were unreachable, which is why nothing noticed
# them for so long -- but they are much closer to live now, so a future
# strength change has to re-check the range rather than assume headroom.
# tests/test_sloped_render.py pins both the range and that fact.
SLOPE_SHADE_MIN = 0.50
SLOPE_SHADE_MAX = 1.35


@lru_cache(maxsize=1024)
def _slope_shade(tile_px: int, nw: int, ne: int, sw: int, se: int, elev_step: int) -> np.ndarray:
    """Per-pixel shading factor over one tile's DIAMOND footprint, in
    tile_uv_fractions(tile_px) order (equivalently diamond_indices' own) --
    multiply against sampled texture color, same "float32 factor, truncate
    back to uint8" contract render_iso.py's other shading (_shadow_factors)
    already uses.

    NOT positionally aligned with sloped_quad_indices' output, which is a
    variable-length resample and generally a different length entirely.
    Gather it through that call's fifth return value first:
    `shade[uv_idx]`. Slicing it to length instead (`shade[:dst_y.size]`)
    silently mis-shades rather than raising, because the two lengths can
    coincide -- tests/test_sloped_render.py's
    test_slope_shade_is_gathered_through_uv_idx pins exactly that case.

    nw/ne/sw/se are the tile's 4 corner rises AFTER normalization against
    their own minimum (same convention sloped_quad_indices itself takes) --
    only their DIFFERENCES matter for a gradient, so the normalization is
    harmless here, not just permitted.

    The surface gradient is a closed-form derivative of the bilinear patch
    tile_uv_fractions() parameterizes over the diamond -- the height field
    the corner rises define, which is a different derivation from
    sloped_quad_indices' own integer cuts on shared tile edges, not the
    same one at a different resolution. See
    tile_uv_fractions' docstring for the fx=1-fq, fy=fp identity this
    expands from: height(fx, fy) = NW(1-fx)(1-fy) + NE*fx*(1-fy) +
    SW*(1-fx)*fy + SE*fx*fy, so d(height)/d(fx) = (1-fy)*(NE-NW) +
    fy*(SE-SW) and d(height)/d(fy) = (1-fx)*(SW-NW) + fx*(SE-NE);
    substituting fx=1-fq, fy=fp gives the gx/gy lines below directly in
    terms of the (fp, fq) this module already has in hand.

    elev_step is the scale a gradient of exactly 1 elev_step-per-tile-width
    (the steepest ramp this project's own +-1-elevation-neighbor invariant
    ever produces, see iso_geometry.ELEV_STEP_DIVISOR's own comment) is
    normalized against, giving that steepest-real-ramp case a 45-degree
    tilt in the corresponding axis -- a natural reference already tied to
    an existing project constant, not a new unrelated tuning knob."""
    fp, fq = iso_geometry.tile_uv_fractions(tile_px)
    gx = (1 - fp) * (ne - nw) + fp * (se - sw)
    gy = fq * (sw - nw) + (1 - fq) * (se - ne)
    nx = -gx / elev_step
    ny = -gy / elev_step
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    lx, ly, lz = SLOPE_LIGHT_DIR
    dot = nx * lx + ny * ly + nz * lz
    shade = 1.0 + SLOPE_SHADE_STRENGTH * (dot - lz)
    return np.clip(shade, SLOPE_SHADE_MIN, SLOPE_SHADE_MAX).astype(np.float32)


def _render_tile_sloped(
    img: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    terrain_override: int | None = None,
    textures: bool = True,
    blend=None,
    fold: np.ndarray | None = None,
) -> None:
    """Sloped mode's counterpart to _render_tile_iso -- same texture-crop
    and offset/clip contract, but no skirt loop and no contact-shadow loop
    (see SLOPE_CORNER_RULE's own comment for why neither has anything to
    draw here), and a single sloped_quad_indices() paint instead of
    diamond_indices' uniform one, shaded via _slope_shade() instead of
    plain texture color.

    corner_rise is corner_rise_px()'s (h+1, w+1) whole-map output --
    precomputed once per render/patch (like _render_tile_iso's own
    `elevations` parameter), not recomputed per tile.

    Placed at elevation=0 (tile_screen_origin(tile.x, tile.y, 0, proj)),
    NOT tile.elevation: corner_rise's own values already encode the full
    absolute elevation-to-pixel scale directly (see corner_rise_px's
    docstring), so adding a second elevation*elev_step term here would
    double-count it. base_y is further shifted by -d_min, matching
    sloped_quad_indices' own normalization contract (see that function's
    docstring for why the caller, not that function, owns folding d_min
    back in).

    terrain_override (Track C6), when given, replaces tile.terrain_id for
    the texture lookup only -- mirrors _render_tile_iso's own parameter
    exactly (see that function's docstring): shading, corners, d_min and
    placement are all unaffected, and tile.terrain_id itself, and
    everything the caller derives from the tile object, stays untouched.

    textures=False (View > Layers): same flat-colour branch _render_tile_iso
    documents. _slope_shade()'s multiplier still applies on top, so a
    texture-less Sloped render keeps its slope shading. blend: the tile's
    blend key, as _render_tile_iso's. fold: iso_geometry.tile_folds()'s
    array, which picks how the texture and shade resample (never which
    pixels the tile covers); None is the fixed NE-SW fold."""
    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    d_min = min(d_nw, d_ne, d_sw, d_se)

    base_x, base_y, dst_y, dst_x, src_y, src_x, uv_idx = _sloped_tile_quad(
        tile.x, tile.y, tile_px, proj, d_nw, d_ne, d_sw, d_se, offset,
        nwse=iso_geometry.tile_nwse(fold, tile.x, tile.y),
    )
    # RAW d_nw..d_se, not the d_min-normalized ones: that is the tuple
    # _sloped_tile_quad passed sloped_quad_indices to get dst_y/dst_x.
    extent = iso_geometry.index_extent(iso_geometry.sloped_quad_indices, tile_px, d_nw, d_ne, d_sw, d_se)
    # Early reject: the one paint below IS the tile's whole reach, so wholly
    # outside here skips only no-ops. Must match _render_tile_sloped_ids'.
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        h, w = img.shape[0], img.shape[1]
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return

    top_block = _top_block(tile, tile_px, textures, blend, terrain_override)

    shade = _slope_shade(tile_px, d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min, proj.elev_step)
    native = composite_backend.native
    if native is not None:
        native.paint_sloped(img, base_y, base_x, dst_y, dst_x, src_y, src_x, uv_idx, top_block, shade)
        return
    top = top_block[src_y, src_x]
    shaded = np.clip(top.astype(np.float32) * shade[uv_idx][:, None], 0, 255).astype(np.uint8)
    _clipped_paint(img, base_y, base_x, dst_y, dst_x, shaded, extent=extent)


def _sloped_tile_quad(
    tx: int,
    ty: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    d_nw: int,
    d_ne: int,
    d_sw: int,
    d_se: int,
    offset: tuple[int, int] = (0, 0),
    nwse: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(base_x, base_y, dst_y, dst_x, src_y, src_x, uv_idx) -- the placement
    and warp-index computation _render_tile_sloped paints a tile's own
    pixels through, factored out so a conforming 1x1 unit marker
    (_draw_unit_sloped) can paint through the exact same call rather than a
    parallel copy. Inheriting this is the whole point: it is what makes the
    marker structurally unable to drift from the tile it stands on, the
    same way _render_tile_sloped_ids inherits agreement with the colour pass
    by copying its inputs rather than re-deriving them. `nwse` (the tile's
    fold) changes only src_y/src_x/uv_idx, so a caller that paints solid
    pixels may leave it False."""
    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, 0, proj)
    base_x -= off_x
    base_y -= off_y + min(d_nw, d_ne, d_sw, d_se)
    dst_y, dst_x, src_y, src_x, uv_idx = iso_geometry.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se, nwse)
    return base_x, base_y, dst_y, dst_x, src_y, src_x, uv_idx


PICK_ID_NONE = -1
"""The "no tile painted here" sentinel in a Sloped pick plane (Track C4).
Not 0: tile (0, 0) encodes to id 0 under y * map_w + x, so 0 is a real,
reachable tile id and a zero-filled plane would report the map's own west
corner for every unpainted background pixel."""


def _render_tile_sloped_ids(
    plane: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    map_w: int,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Paints one tile's own id into an int32 pick plane over exactly the
    pixels _render_tile_sloped() paints colour into -- Track C4's
    hit-testing backend.

    AGREEMENT WITH THE COLOUR PASS IS INHERITED, NOT ARGUED. Every quantity
    that decides WHICH pixels get touched is computed here the same way
    _render_tile_sloped() computes it, from the same inputs: the same four
    corner_rise lookups, the same d_min, the same tile_screen_origin() at
    elevation 0, the same -d_min subtraction folded into base_y, the same
    sloped_quad_indices() call (same lru_cache entry, so literally the same
    dst arrays), and the same _clipped_paint() masking. Only `values`
    differs. Getting any one of those wrong would be a SILENT disagreement
    between what the user sees and what a click resolves to, not a crash --
    which is why they are copied rather than re-derived, and why
    tests/test_sloped_pick.py mutation-checks the -d_min term specifically.

    src_y/src_x/uv_idx are deliberately unused: a tile id is constant across
    the tile, so there is no per-pixel quantity to gather. That makes this
    materially cheaper than the colour pass (no texture crop, no
    _slope_shade), which is the whole point of a separate ID-only walk.

    Terrain only -- units are NOT painted, and Track C5 deliberately kept it
    that way. Units paint after terrain into the same colour image, so a
    terrain-only plane reports the tile UNDER a unit pixel: exactly what
    tile picking and the tool highlight want, and exactly what unit picking
    must not use. unit_pick._pick_unit_sloped answers that second question
    analytically instead, and this plane's one job for it is the terrain
    tile the occlusion compare needs (SlopedChunkCache.pick_tile). Folding
    units in here would cost that closed form and reopen the pick plane's
    own cost measurement."""
    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    d_min = min(d_nw, d_ne, d_sw, d_se)

    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, 0, proj)
    base_x -= off_x
    base_y -= off_y + d_min

    dst_y, dst_x, _src_y, _src_x, _uv_idx = iso_geometry.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
    extent = iso_geometry.index_extent(iso_geometry.sloped_quad_indices, tile_px, d_nw, d_ne, d_sw, d_se)
    # The same early reject as _render_tile_sloped, on the same predicate, or
    # the colour image and this plane silently disagree.
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        h, w = plane.shape[0], plane.shape[1]
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return
    values = np.full(dst_y.shape[0], tile.y * map_w + tile.x, dtype=np.int32)
    _clipped_paint(plane, base_y, base_x, dst_y, dst_x, values, extent=extent)


def _draw_unit_sloped(
    img: np.ndarray,
    unit,
    color: tuple[int, int, int],
    tx: int,
    ty: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    rise_px: int,
    corners: tuple[int, int, int, int] | None = None,
    offset: tuple[int, int] = (0, 0),
    ink: np.ndarray | None = None,
    ink_rise: int = 0,
) -> None:
    """Sloped's counterpart to _draw_unit_iso -- same one-mark-per-call paint
    (see that function for why a multi-tile unit arrives here once per
    footprint tile), but takes a precomputed SCALAR PIXEL RISE (already
    derived from the UNIT'S OWN tile's 4 corners and its own sub-tile
    position by _paint_tile_and_units_sloped, see that function) instead of
    a whole-map elevations array to index. Sloped has no such array
    (corner_rise_px replaces it) -- allocating a throwaway full-size one
    just to satisfy _draw_unit_iso's existing signature would cost
    O(map_h * map_w) per UNIT, not per render, so this takes the scalar
    directly.

    **corners, not None, is a second mode, not an extra option on the same
    one.** When the caller's unit_tile_bounds() gives this unit a (1, 1)
    span AND its paint offset is zero, its own tile IS its whole footprint,
    so a conforming marker paints exactly that tile's own warped pixel set --
    via _sloped_tile_quad(), the same call _render_tile_sloped() paints the
    tile itself through, so the marker cannot drift from the tile it stands
    on. `rise_px` is ignored in that mode: the tile's own base_y already
    encodes the surface, and adding a rise on top would double-count it (see
    the plan's `d_min` trap). Every multi-tile footprint keeps the plain
    diamond at `rise_px`, unchanged -- a real building is a flat pad, and
    conforming per footprint tile would silently undo that -- and since off-grid
    placement's Stage 1 an off-centre 1x1 unit joins it there, since a
    conforming quad has no position term to express a sub-tile position with
    (see _paint_tile_and_units_sloped's docstring for what forced that).

    Pixels, not an elevation LEVEL, since Track C5: corner_rise is already
    in canvas pixels, so placing at tile_screen_origin(tx, ty, 0, proj) and
    subtracting rise_px is the whole conversion -- no elev_step round trip,
    and no widening of tile_screen_origin, whose docstring promises exact
    integer arithmetic. Matches _render_tile_sloped's own placement
    convention (elevation=0 plus a pixel shift) rather than introducing a
    second one.

    **No sprite path here, still.** A unit whose sprite paints via
    sprite_draws_by_anchor()/sprites.by_anchor (Track P3-g6) is skipped
    before it ever reaches this function -- see
    _paint_tile_and_units_sloped()'s skip_ids gate. This function draws the
    plain coloured mark for a unit with no resolved sprite, and for a farm
    only when sprites are off (Track C6): a farm with sprites on drapes as
    terrain instead, resolved in _paint_tile_and_units_sloped before this
    function is ever called for it (see that function's own farm case).

    ink/ink_rise: the two-pass ink plane and this mark's base rise in it, as
    _draw_unit_iso's. Its own argument because `corners` mode ignores rise_px."""
    if corners is not None:
        base_x, base_y, dst_y, dst_x, _src_y, _src_x, _uv_idx = _sloped_tile_quad(
            tx, ty, tile_px, proj, *corners, offset
        )
        # *corners is the raw (d_nw, d_ne, d_sw, d_se) _sloped_tile_quad
        # just keyed sloped_quad_indices on, byte for byte.
        extent = iso_geometry.index_extent(iso_geometry.sloped_quad_indices, tile_px, *corners)
        _clipped_paint_solid(img, base_y, base_x, dst_y, dst_x, color, extent=extent)
        if ink is not None:
            _record_ink(ink, base_y, base_x, dst_y, dst_x, ink_rise)
        return
    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    off_x, off_y = offset
    base_y, base_x = _sloped_mark_base(unit, tx, ty, rise_px, proj)
    extent = iso_geometry.index_extent(iso_geometry.diamond_indices, tile_px)
    _clipped_paint_solid(img, base_y - off_y, base_x - off_x, dst_y, dst_x, color, extent=extent)
    if ink is not None:
        _record_ink(ink, base_y - off_y, base_x - off_x, dst_y, dst_x, ink_rise)


def _sloped_mark_base(unit, tx: int, ty: int, rise_px: int, proj: iso_geometry.IsoProjection) -> tuple[int, int]:
    """(base_y, base_x) of a Sloped plain-diamond mark on footprint tile (tx, ty)."""
    # Positioned by its CENTRE, the same way _draw_unit_iso is (off-grid
    # placement, Stage 1). rise_px is an int, so a zero paint offset reduces
    # this to tile_screen_origin(tx, ty, 0) shifted up by rise_px exactly --
    # the expression this line used to be, byte for byte.
    dx, dy = unit_paint_offset(unit)
    cx, cy = iso_geometry.map_point_to_screen(tx + 0.5 + dx, ty + 0.5 + dy, rise_px, proj)
    return cy - proj.half_h, cx - proj.half_w


def _sloped_mark_conforms(unit) -> bool:
    """Whether a Sloped mark paints its own tile's warped quad (see
    _paint_tile_and_units_sloped) rather than a plain diamond."""
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    return span_x <= 1 and span_y <= 1 and unit_paint_offset(unit) == (0.0, 0.0)


def _sloped_mark_rise(unit, corner_rise: np.ndarray, fold: np.ndarray | None = None) -> int:
    ux, uy = int(unit.x), int(unit.y)
    return iso_geometry.unit_rise_px(
        corner_rise, ux, uy, unit.x - ux, unit.y - uy, iso_geometry.tile_nwse(fold, ux, uy)
    )


def _sloped_tile_floor(corner_rise: np.ndarray, tx: int, ty: int) -> int:
    """A Sloped tile's lowest drawn corner: its height as a two-pass occluder,
    and the base rise of a mark conforming to it."""
    return min(
        int(corner_rise[ty, tx]), int(corner_rise[ty, tx + 1]),
        int(corner_rise[ty + 1, tx]), int(corner_rise[ty + 1, tx + 1]),
    )


def _unit_mark_sloped(
    unit, tx: int, ty: int, proj: iso_geometry.IsoProjection, corner_rise: np.ndarray,
    fold: np.ndarray | None = None,
) -> tuple[tuple[int, int] | None, int]:
    """The native UnitPack's view of one Sloped mark: (None when it conforms
    to its tile's quad, else its plain diamond's (base_y, base_x); its base
    rise, as _paint_objects_sloped records it)."""
    if _sloped_mark_conforms(unit):
        return None, _sloped_tile_floor(corner_rise, tx, ty)
    rise_px = _sloped_mark_rise(unit, corner_rise, fold)
    return _sloped_mark_base(unit, tx, ty, rise_px, proj), rise_px


def _paint_tile_and_units_sloped(
    img: np.ndarray,
    tile,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
    blend=None,
    fold: np.ndarray | None = None,
) -> None:
    """Sloped's counterpart to _paint_tile_and_units_iso -- same
    "terrain then this tile's own units, interleaved in depth_order" step,
    and like it the sprites-off path only: with sprites on,
    _paint_candidates_sloped runs the two halves below as two passes (see
    _paint_tile_and_units_iso's docstring for the rule).

    A CENTRED 1x1 unit -- everything but a multi-tile building, a farm, or
    an off-centre unit -- instead conforms to its own tile's warped footprint
    exactly, via _draw_unit_sloped()'s `corners` mode (Track C6): its own
    tile IS its whole footprint, so there is no separate rise term and no
    risk of the marker spilling past the tile's painted pixels the way a
    plain diamond did on a ramp. Multi-tile buildings are unaffected and
    still paint via the PIXEL RISE below; a farm is multi-tile too, but
    drapes as real terrain instead of painting via _draw_unit_sloped() at all
    whenever sprites are on -- see this function's own farm case, below the
    _render_tile_sloped() call.

    **"Centred" is off-grid placement's Stage 1, and it is the narrower of two
    possible deviations from C6.** A conforming quad has no position term, so
    it cannot show a sub-tile position at all; an off-centre 1x1 unit
    therefore joins the plain-diamond path below, at its own point. Keeping
    C6 for every 1x1 unit instead would have left Sloped holding still while
    Stepped's marks moved, which breaks the flat-map byte-identity of
    render_terrain_sloped against render_terrain_iso -- a hard invariant, and
    the harder of the two to give up. C6's ramp-spill fix survives for the
    88.0% of units that sit at an exact (0.5, 0.5).

    Multi-tile units are drawn via _draw_unit_sloped() at the PIXEL RISE
    iso_geometry.unit_rise_px() reports for the unit's own sub-tile
    position inside its OWN tile -- i.e. on the surface this same function
    just painted, evaluated at the point the unit stands on (Track C5's
    Step 2). Until C5 that was an average of the own tile's 4 corners,
    quantized back to an integer elevation level: a surface the renderer
    never draws (the two differ on every non-planar tile, by up to a full
    level in 62 of 81 corner configurations) at a resolution coarser than
    the one it draws at. A unit is still a flat PAD -- one height for the
    whole footprint, since buildings are flat in-game -- C5 changed only
    how that height is computed.

    **The lookup is per-entry, over the unit's own tile, not over `tile`.**
    Since _units_by_tile() buckets a unit into every footprint tile, `tile`
    here is usually a FOOTPRINT tile rather than the unit's own one, and
    interpolating its corners would make the slab conform to the slope
    tile-by-tile -- silently undoing the flat pad described above (this is
    exactly what the 1x1 path above is allowed to do, since there a footprint
    tile and the own tile are the same tile by construction). The own
    tile's corners are always in bounds: unit_tile_bounds() only admitted
    this unit because its own tile is on-map, and corner_rise is
    (map_h + 1, map_w + 1).

    map_w/map_h are unused since _draw_unit_sloped() stopped re-deriving
    bounds; they stay only to keep this signature parallel with
    _paint_tile_and_units_iso()'s, which the two paths are deliberately
    written to mirror. Not load-bearing.

    sprites (Track P3-g6): a unit in sprites.skip_ids draws as a real sprite
    at its anchor tile instead of a mark here -- same skip/blit shape
    _paint_tile_and_units_iso() uses. A farm-terrain tile
    (sprites.farm_by_tile) is a THIRD case alongside the diamond and the
    sprite, resolved right here rather than in a separate pass (Track C6,
    mirroring _paint_tile_and_units_iso's own farm handling): it changes
    what _render_tile_sloped paints as this tile's own terrain, not
    something composited on top of it, so the override has to reach that
    call before it runs. The perimeter stroke that follows uses
    sloped_tile_edge_indices at this same tile's own base_x/base_y (the
    _sloped_tile_quad placement _render_tile_sloped itself paints through),
    not _render_tile_iso's elevation-based one -- copying that placement
    convention here would silently land the outline in the wrong place
    (this function's own `d_min` trap, a second time)."""
    _paint_terrain_sloped(
        img, tile, tile_px, proj, corner_rise, map_w, map_h, offset=offset, sprites=sprites,
        textures=textures, grid=grid, blend=blend, fold=fold,
    )
    _paint_objects_sloped(
        img, tile, units_by_tile, tile_px, proj, corner_rise, offset=offset, sprites=sprites, fold=fold
    )


def _paint_terrain_sloped(
    img: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
    blend=None,
    fold: np.ndarray | None = None,
) -> None:
    """A tile's terrain half: its terrain (a farm's override included), the
    farm outline, the grid. `sprites` is read for its farm tiles only; `fold`
    reaches the terrain's resample only (outline and grid are dst-only)."""
    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    farm = sprites.farm_by_tile.get((tile.x, tile.y)) if sprites is not None else None
    terrain_override = farm[0] if farm is not None else None
    _render_tile_sloped(
        img, tile, tile_px, proj, corner_rise, offset=offset,
        terrain_override=terrain_override, textures=textures, blend=blend, fold=fold,
    )
    if farm is not None:
        _, outline_color, edge_mask = farm
        base_x, base_y, _dst_y, _dst_x, _sy, _sx, _uv = _sloped_tile_quad(
            tile.x, tile.y, tile_px, proj, d_nw, d_ne, d_sw, d_se, offset
        )
        for bit, side in _FARM_EDGE_BITS:
            if not (edge_mask & bit):
                continue
            edge_dst_y, edge_dst_x = iso_geometry.sloped_tile_edge_indices(tile_px, side, d_nw, d_ne, d_sw, d_se)
            extent = iso_geometry.index_extent(
                iso_geometry.sloped_tile_edge_indices, tile_px, side, d_nw, d_ne, d_sw, d_se
            )
            _clipped_paint_solid(img, base_y, base_x, edge_dst_y, edge_dst_x, outline_color, extent=extent)
    if grid.paints:
        # Same slot and reasoning as Stepped's. _sloped_tile_quad's base, not
        # tile_screen_origin's, and corners normalized the way its frame is.
        gx, gy, *_ = _sloped_tile_quad(tile.x, tile.y, tile_px, proj, d_nw, d_ne, d_sw, d_se, offset)
        d_min = min(d_nw, d_ne, d_sw, d_se)
        corners = (d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min)
        _paint_grid_iso(img, tile.x, tile.y, tile_px, map_w, map_h, gy, gx, grid, corners=corners)


def _paint_objects_sloped(
    img: np.ndarray,
    tile,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    ink: np.ndarray | None = None,
    fold: np.ndarray | None = None,
) -> None:
    """A tile's objects half: its marks, then the sprites anchored on it. ink,
    when given, records every pixel they touch at the unit's base rise: the
    mark's own rise, or its tile's lowest corner when it conforms."""
    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    skip = sprites.skip_ids if sprites is not None else frozenset()
    for unit, color in units_by_tile.get((tile.x, tile.y), ()):
        if id(unit) in skip:
            continue  # its sprite paints instead, at its anchor tiles (_paint_objects_iso)
        if _sloped_mark_conforms(unit):
            # This unit's own tile IS its whole footprint (see
            # unit_tile_bounds' invariant), and units_by_tile only ever
            # buckets it there -- so `tile` here already IS that tile, and
            # its corners are the ones _render_tile_sloped just painted.
            #
            # Only at zero offset (off-grid placement, Stage 1). A conforming
            # quad has no position term to shift, so an off-centre unit falls
            # through to the plain-diamond path below, which does. That split
            # is what keeps render_terrain_sloped byte-identical to
            # render_terrain_iso on a flat map -- a hard project invariant,
            # and one a fractional unit is deliberately used to pin.
            corners = (d_nw, d_ne, d_sw, d_se)
            _draw_unit_sloped(
                img, unit, color, tile.x, tile.y, tile_px, proj, 0, corners=corners, offset=offset,
                ink=ink, ink_rise=min(corners),
            )
            continue
        rise_px = _sloped_mark_rise(unit, corner_rise, fold)
        _draw_unit_sloped(
            img, unit, color, tile.x, tile.y, tile_px, proj, rise_px, offset=offset, ink=ink, ink_rise=rise_px
        )

    # Sprites after this tile's marks, in by_anchor's order (_paint_objects_iso's comment).
    if sprites is not None:
        off_x, off_y = offset
        for draw, ax, ay, rise_px in sprites.by_anchor.get((tile.x, tile.y), ()):
            _clipped_paint_rgba(
                img, ay - draw.hotspot_y - off_y, ax - draw.hotspot_x - off_x, draw.rgba, ink, rise_px
            )


def _occlude_sloped(
    img: np.ndarray, terrain: np.ndarray, ink: np.ndarray, tile, tile_px: int,
    proj: iso_geometry.IsoProjection, corner_rise: np.ndarray, offset: tuple[int, int], floor: int,
) -> None:
    """Sloped's _occlude_iso: the tile's quad is its coverage, its lowest drawn
    corner its height, so it covers only ink its whole surface is above."""
    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    height = min(d_nw, d_ne, d_sw, d_se)
    if height <= floor:
        return
    base_x, base_y, dst_y, dst_x, _sy, _sx, _uv = _sloped_tile_quad(
        tile.x, tile.y, tile_px, proj, d_nw, d_ne, d_sw, d_se, offset
    )
    extent = iso_geometry.index_extent(iso_geometry.sloped_quad_indices, tile_px, d_nw, d_ne, d_sw, d_se)
    _restore_under(img, terrain, ink, base_y, base_x, dst_y, dst_x, height, extent=extent)


def _paint_candidates_sloped(
    img: np.ndarray,
    tiles: list,
    keys: list | None,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    corner_rise: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
    fold: np.ndarray | None = None,
) -> None:
    """Sloped's _paint_candidates_iso, same two passes and same pre-scan."""
    if sprites is None:
        for i, tile in enumerate(tiles):
            _paint_tile_and_units_sloped(
                img, tile, units_by_tile, tile_px, proj, corner_rise, map_w, map_h, offset=offset,
                textures=textures, grid=grid, blend=None if keys is None else keys[i], fold=fold,
            )
        return
    for i, tile in enumerate(tiles):
        _paint_terrain_sloped(
            img, tile, tile_px, proj, corner_rise, map_w, map_h, offset=offset, sprites=sprites,
            textures=textures, grid=grid, blend=None if keys is None else keys[i], fold=fold,
        )

    def mark_rise(unit, tile) -> int:
        if _sloped_mark_conforms(unit):
            return _sloped_tile_floor(corner_rise, tile.x, tile.y)
        return _sloped_mark_rise(unit, corner_rise, fold)

    floor = _ink_floor(tiles, units_by_tile, sprites, mark_rise)
    terrain = ink = None
    if any(_sloped_tile_floor(corner_rise, tile.x, tile.y) > floor for tile in tiles):
        terrain = img.copy()
        ink = np.full(img.shape[:2], INK_NONE, dtype=np.int32)
    for tile in tiles:
        if ink is not None:
            _occlude_sloped(img, terrain, ink, tile, tile_px, proj, corner_rise, offset, floor)
        _paint_objects_sloped(
            img, tile, units_by_tile, tile_px, proj, corner_rise, offset=offset, sprites=sprites, ink=ink,
            fold=fold,
        )


def _unit_rise_headroom_px(
    corner_rise: np.ndarray, elevations: np.ndarray, proj: iso_geometry.IsoProjection
) -> int:
    """How many canvas pixels ABOVE `elevation * elev_step` a Sloped unit's
    own placement can possibly land -- the exact `extra_top_px` bound
    _unit_screen_bbox_iso() needs (Track C5's Step 2).

    unit_rise_px() interpolates between the unit's OWN tile's four corner
    rises, so its value is bounded by that tile's own corner maximum; this
    takes the worst such excess over the whole map. Never negative under
    SLOPE_CORNER_RULE = "max" (a corner is the max over the tiles touching
    it, so it includes the tile's own elevation), and exactly 0 on a flat
    map, which is what keeps flat-map byte-identity against Stepped."""
    c = corner_rise
    tile_corner_max = np.maximum(
        np.maximum(c[:-1, :-1], c[:-1, 1:]), np.maximum(c[1:, :-1], c[1:, 1:])
    )
    return max(0, int((tile_corner_max - elevations * proj.elev_step).max()))


def sloped_elevations_and_proj(
    scenario: LoadedScenario,
) -> tuple[np.ndarray, np.ndarray, iso_geometry.IsoProjection]:
    """(elevations, corner_rise, proj) only -- no tile_grid, no
    compositing -- Sloped's counterpart to elevations_and_proj(), the cheap
    half a chunk-cache-backed viewer init needs (Track C3) without first
    paying for a full composite it's about to render lazily instead.

    corner_headroom_steps=1 (see IsoProjection.corner_headroom_px's own
    comment) -- Stepped/Flat's elevations_and_proj()/canvas_size_and_origin
    calls leave this at its 0 default."""
    w, h, tile_px, elevations = _dims_and_elevations(scenario)
    proj = iso_geometry.canvas_size_and_origin(
        w,
        h,
        tile_px,
        iso_geometry.MIN_ELEVATION,
        iso_geometry.MAX_ELEVATION,
        elev_step_pct=settings.get_elev_step_pct(),
        corner_headroom_steps=1,
    )
    corner_rise = iso_geometry.corner_rise_px(elevations, proj, rule=SLOPE_CORNER_RULE)
    return elevations, corner_rise, proj


def render_terrain_sloped_with_proj(
    scenario: LoadedScenario,
    with_units: bool = True,
    with_sprites: bool = False,
    grid: GridBake = DEFAULT_GRID,
    blend: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, iso_geometry.IsoProjection]:
    """render_terrain_sloped()'s real body, additionally returning the
    (h, w) elevations array, the (h+1, w+1) corner_rise array, and the
    IsoProjection the render was actually computed from -- Sloped's
    counterpart to render_terrain_iso_with_proj(), same reasons (Track C3's
    viewer wiring needs the extra values for hit-testing/patching against
    the exact snapshot that produced the pixels on screen).

    Canvas allocation intentionally matches render_terrain_iso_with_proj()'s
    OWN formula exactly (same skirt_headroom padding, even though Sloped
    paints no skirts) rather than a tighter Sloped-specific bound: keeping
    the two canvases the SAME shape for the same map is what makes a
    flat-map render_terrain_sloped output comparable to render_terrain_iso's
    at all -- see tests/test_sloped_render.py's byte-identity oracle.

    with_sprites (Track P3-g6): same "defaults False so this stays byte-
    identical to what it has always rendered" reasoning as
    render_terrain_iso_with_proj()'s own parameter, and what gives the
    stitched-chunk check a full-render ground truth to compare against.
    Farms drape (Track C6, with_farms's own default) whenever with_sprites
    is True, matching SlopedChunkCache's own call and Stepped parity -- see
    sprite_draws_by_anchor()'s docstring. blend: as
    render_terrain_iso_with_proj()'s."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    tile_grid, elevations = _terrain_grid_and_elevations(scenario)
    min_elev, max_elev = iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    proj = iso_geometry.canvas_size_and_origin(
        w,
        h,
        tile_px,
        min_elev,
        max_elev,
        elev_step_pct=settings.get_elev_step_pct(),
        corner_headroom_steps=1,
    )
    corner_rise = iso_geometry.corner_rise_px(elevations, proj, rule=SLOPE_CORNER_RULE)
    fold = iso_geometry.tile_folds(elevations)

    skirt_headroom = (max_elev - min_elev) * proj.elev_step
    img = np.zeros((proj.canvas_h + skirt_headroom, proj.canvas_w, 3), dtype=np.uint8)

    units_by_tile = _units_by_tile(scenario) if with_units else {}
    sprites = (
        sprite_draws_by_anchor(
            scenario, proj, elevations, corner_rise=corner_rise, hero_glow=DEFAULT_LAYERS.hero_glow, fold=fold
        )
        if (with_units and with_sprites)
        else None
    )
    keys = _full_blend_keys(scenario, blend)
    order = iso_geometry.depth_order(w, h).tolist()
    _paint_candidates_sloped(
        img, [tile_grid[y][x] for x, y in order], None if keys is None else [keys[y * w + x] for x, y in order],
        units_by_tile, tile_px, proj, corner_rise, w, h, sprites=sprites, grid=grid, fold=fold,
    )
    return img, elevations, corner_rise, proj


def render_terrain_sloped(
    scenario: LoadedScenario, with_units: bool = True, grid: GridBake = DEFAULT_GRID, blend: bool = True
) -> np.ndarray:
    """Sloped mode's top-level renderer -- the Phase 6 counterpart to
    render_terrain_iso(). Thin wrapper around
    render_terrain_sloped_with_proj() for callers that only need the
    pixels (render_scenario(), tools/dump_scenario.py,
    tools/gen_elevation_reference.py's own reference renders)."""
    img, _elevations, _corner_rise, _proj = render_terrain_sloped_with_proj(
        scenario, with_units=with_units, grid=grid, blend=blend
    )
    return img


def composite_rect_sloped(
    scenario: LoadedScenario,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    corner_rise: np.ndarray,
    proj: iso_geometry.IsoProjection,
    tile_px: int,
    units_by_tile: dict,
    building_bboxes: dict,
    with_units: bool = True,
    sprites: SpriteLayer | None = None,
    bystander_grid: BystanderGrid | None = None,
    layers: LayerState = DEFAULT_LAYERS,
    grid: GridBake = DEFAULT_GRID,
    unit_pack=None,
    terrain_ids: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
    key_map: terrain_blend.KeyMap | None = None,
    fold: np.ndarray | None = None,
) -> np.ndarray:
    """Sloped's counterpart to composite_rect_iso() -- same rect-keyed-core
    contract (see that function's own docstring for the full argument: a
    chunk's pixels can never depend on which OTHER rects happen to have
    been requested, since both the patch path and the chunk cache composite
    through this SAME function), same candidate enumeration via
    iso_geometry.tiles_in_screen_rect() and the same building-bystander
    merge, just painted via _paint_tile_and_units_sloped() instead of
    _paint_tile_and_units_iso().

    proj here must carry corner_headroom_steps=1 (see
    sloped_elevations_and_proj()) -- tiles_in_screen_rect()'s own candidate
    sweep uses proj.corner_headroom_px via tile_screen_bounds_swept(), so a
    proj built without it could under-enumerate candidates near a chunk's
    own edge.

    sprites (Track P3-g6): same null-when-units-off rule composite_rect_iso()
    applies, so a sprite layer built for a with_units=True render is never
    consulted once units are toggled off.

    bystander_grid: same optional index composite_rect_iso() takes, same
    None-keeps-the-full-walk default.

    layers: same contract composite_rect_iso() documents -- only the
    terrain-textures half is applied here; the farm half is already baked
    into whatever `sprites` was built with.

    unit_pack, terrain_ids: same N2 contract as composite_rect_iso()'s;
    layer_ids, key_map and blending likewise. fold: iso_geometry.tile_folds()
    of the elevations corner_rise came from (None only on a planar map,
    where every fold is NE-SW; iso_geometry.require_fold() raises otherwise)."""
    iso_geometry.require_fold(corner_rise, fold, "composite_rect_sloped()")
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    candidates = _rect_candidates(scenario, x0, y0, x1, y1, proj, with_units, building_bboxes, bystander_grid)

    scratch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sprite_layer = sprites if with_units else None
    blend = _rect_blend_keys(scenario, candidates, layers, terrain_ids, layer_ids, key_map)
    native = composite_backend.native
    if native is not None and unit_pack is not None and candidates.shape[0]:
        from descape import native_composite

        if native_composite.composite_sloped(
            native, scratch, scenario, x0, y0, candidates, corner_rise, proj, tile_px,
            unit_pack if with_units else None, layers.terrain_textures, grid, terrain_ids=terrain_ids,
            blend=blend, fold=fold,
        ):
            return scratch
    keys = None if blend is None else blend.as_list()
    # Same direct terrain index as composite_rect_iso, same clamping argument.
    _paint_candidates_sloped(
        scratch, [mm.terrain[cy * w + cx] for cx, cy in candidates.tolist()], keys, units_by_tile, tile_px, proj,
        corner_rise, w, h, offset=(x0, y0), sprites=sprite_layer, textures=layers.terrain_textures, grid=grid,
        fold=fold,
    )
    return scratch


def composite_ids_rect_sloped(
    scenario: LoadedScenario,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    corner_rise: np.ndarray,
    proj: iso_geometry.IsoProjection,
    tile_px: int,
) -> np.ndarray:
    """composite_rect_sloped()'s ID-plane twin: the half-open screen rect
    [x0, x1) x [y0, y1) as a fresh (y1-y0, x1-x0) int32 array holding, per
    pixel, the id (y * map_w + x) of the tile whose surface covers it, or
    PICK_ID_NONE where no tile does. Track C4's hit-testing primitive.

    Same rect-keyed-core contract as composite_rect_sloped() (see that
    function): the same candidates from the same
    iso_geometry.tiles_in_screen_rect() call, walked in the same depth
    order, overwriting each other in the same sequence -- so the topmost id
    at a pixel is the tile whose colour won that pixel, and occlusion by a
    taller neighbour resolves identically. proj must carry
    corner_headroom_steps=1 for the same reason it must there.

    THE ONE DELIBERATE DIVERGENCE, and why it is safe: no unit pass and no
    building "bystander" merge. A bystander is by definition a tile that
    tiles_in_screen_rect() did NOT return, i.e. one whose own terrain does
    not overlap this rect -- it is merged in there solely so its UNIT mark
    can paint, and its terrain paint is already a whole-tile no-op that
    _clipped_paint() discards. Dropping it therefore cannot change a single
    id. The merge also re-lexsorts into the same depth order it was already
    in, so the surviving candidates' relative order is untouched either
    way."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)

    plane = np.full((y1 - y0, x1 - x0), PICK_ID_NONE, dtype=np.int32)
    # Same direct terrain index as composite_rect_iso, same clamping argument.
    for cx, cy in candidates.tolist():
        tile = mm.terrain[cy * w + cx]
        _render_tile_sloped_ids(plane, tile, tile_px, proj, corner_rise, w, offset=(x0, y0))
    return plane
