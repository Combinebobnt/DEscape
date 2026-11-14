"""The Stepped render: skirt, seam and contact shade, tiles and terrain, projection, candidates, rect composite and region refresh."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from descape import (
    composite_backend,
    iso_geometry,
    settings,
    terrain_blend,
)
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render_bbox import BystanderGrid, _building_bboxes_iso, _rect_candidates, dirty_screen_bbox_iso
from descape.render_paint import (
    INK_NONE,
    _clipped_paint,
    _clipped_paint_rgba,
    _clipped_paint_solid,
    _dims_and_elevations,
    _full_blend_keys,
    _paint_grid_iso,
    _rect_blend_keys,
    _restore_under,
    _terrain_grid_and_elevations,
    _top_block,
    tile_pixels_for_map,
)
from descape.render_sprites import _FARM_EDGE_BITS, SpriteLayer, _ink_floor, sprite_draws_by_anchor
from descape.render_units import _draw_unit_iso, _unit_mark_rise_iso, _units_by_tile
from descape.scenario_io import LoadedScenario
from descape.view_layers import DEFAULT_LAYERS, LayerState

# Two-tone darkening multiplier applied to a skirt face's own top-texture
# sample, distinct per side per the parent plan's decision #5. First-pass
# values (right side darker, mimicking a consistent upper-left light
# source) with no in-game reference to calibrate against yet -- revisit
# once Phase 3 makes real Stepped output visible to eyeball.
SKIRT_SHADE = {"left": 0.75, "right": 0.55}

# Contact-shadow darkening at the row touching a raised column's own top
# edge (see iso_geometry.shadow_quad_indices) -- a single symmetric scalar,
# NOT a per-side dict like SKIRT_SHADE: asymmetry here would reintroduce
# the directional-light claim the ambient-occlusion framing exists to
# avoid (the contact shadow reads as "something is in front of this,"
# not "light comes from this direction"). This shades a tile because of a
# NEIGHBOR's height, never because of its own absolute elevation -- not a
# reintroduction of the per-tile elevation brightness removed at the
# user's request (see render_tile()'s own docstring).
#
# Starts lighter than SKIRT_SHADE's own darkest value (0.55): a darkened
# player-color unit dot sitting on a shadowed back tile must still read as
# the right player (see _render_tile_iso's docstring and the "known
# consequence" this trades off: with sprites off, unit marks on shadowed
# tiles get darkened too, since the terrain/mark paint order is interleaved
# per tile, _paint_tile_and_units_iso()).
CONTACT_SHADE = 0.65

# How far up-screen the contact shadow ramps back to no darkening, as a
# divisor of half_h -- so the ramp is a constant fraction of a tile at every
# tile_px, and independent of the elevation delta that produced the band.
#
# Not a cosmetic knob: shading the band's WHOLE exposed sliver (this is
# where that design decision landed) makes shadow length
# inversely proportional to step height, since a taller step hides more of
# its neighbor. Rendered, a 1-level step then darkens 51.6% of the neighbor
# tile at elev_step_pct=50 and 82.0% at 10, and every up-screen tile reads
# as a filled triangle -- a lattice of dark triangles rather than relief.
# An ambient-occlusion band should instead be a roughly fixed screen-space
# width hugging the occluder's silhouette, which is what this gives.
#
# 4 (a quarter of half_h, i.e. 4px at tile_px=64) chosen by A/B render at
# elev_step_pct 10/50/100 against half_h//2, which still left visible
# triangle texture at 100 where the sliver is only ~14 rows tall. The band
# GEOMETRY is untouched by this -- capping is expressed purely as falloff,
# so the band remains exactly the exposed sliver and simply reaches factor
# 1.0 (an exact no-op multiply) beyond the ramp.
CONTACT_RAMP_DIVISOR = 4

# Darkening for the 1px seam line along a tile's own two up-screen diamond
# edges (see iso_geometry.seam_edge_indices) -- a single symmetric scalar
# for the same reason CONTACT_SHADE is one, not a per-side dict like
# SKIRT_SHADE: asymmetry would reintroduce the directional-light claim the
# ambient-occlusion framing exists to avoid.
#
# The seam COMPLEMENTS the contact-shadow band, it does not replace it: the
# band supplies soft occlusion wherever a lower neighbor is genuinely
# visible, the seam guarantees the silhouette contour exists everywhere --
# including near a tile's apex, where the band has tapered out, and at
# elev_step_pct=200, where the band is empty by design.
#
# Slightly darker than CONTACT_SHADE's 0.65 because it is one pixel and has
# to register at a glance. A/B-rendered against 0.45 and 0.70 at
# elev_step_pct 25/50/100/200 on a 6-level pyramid and on scattered single
# raised tiles. Legibility is NOT what separates them -- measured, all
# three clear the terrain texture's own grain by a wide margin (mean
# darkening vs mean |pixel - its down-screen neighbor| on the unseamed
# render, tile_px=64 grass: 6.3x at 0.45, 4.6x at 0.60, 3.4x at 0.70), so
# "0.70 is too faint" would have been an eyeball claim the numbers do not
# support. 0.60 is chosen on WEIGHT relative to the band it sits beside:
# at 0.45 the contour is heavier than the contact shadow whose silhouette
# it is tracing (0.45 < CONTACT_SHADE's 0.65), which inverts the intended
# reading -- the band is the feature, the seam only guarantees its contour
# exists. 0.60 is the one candidate just darker than CONTACT_SHADE rather
# than well past it.
SEAM_SHADE = 0.60


@lru_cache(maxsize=2)
def _skirt_lut(side: str) -> np.ndarray:
    """256-entry uint8 lookup table for one skirt side's SKIRT_SHADE
    darkening, indexed directly by a raw top-texture byte. Built by the
    same clip-and-truncate expression the per-pixel form used, which is
    what makes it byte-identical rather than merely close (both SKIRT_SHADE
    factors are below 1, so neither clip bound can fire on uint8 input).

    Cached, so a test that patches SKIRT_SHADE has to call
    _skirt_lut.cache_clear() afterwards, the same way _seam_factors below
    documents for SEAM_SHADE. The table itself is only ever read; indexing
    it returns a fresh array, so callers keep the fresh-array contract
    _clipped_paint's `values` argument had before."""
    return np.clip(np.arange(256, dtype=np.float32) * SKIRT_SHADE[side], 0, 255).astype(np.uint8)


@lru_cache(maxsize=256)
def _seam_factors(tile_px: int, side: str) -> np.ndarray:
    """float32 darkening factors aligned 1:1 with the seam indices for
    `side` -- iso_geometry.seam_edge_indices(tile_px, side) for "up_left"/
    "up_right", or seam_apex_indices(tile_px) for "apex", the two-column
    once-per-tile pass. A flat SEAM_SHADE for every pixel, since the seam
    is 1px and so has no falloff to express.

    "apex" is served HERE rather than by its own lru_cache'd function on
    purpose: tests neutralise SEAM_SHADE to 1.0 and call
    _seam_factors.cache_clear() to get an exact no-op control render. A
    second cache would keep serving the old 0.60 through that clear, so
    the control would silently stop being a control. _shadow_factors takes
    its own "apex" side for the same reason.

    Constant, but still an ARRAY, not a scalar: _clipped_darken does
    `factors[in_bounds]` and `factors[:, None]`, both of which need a real
    1-D array of the same length as dst_y. And float32, not float64 -- the
    same bit-identity pinning _shadow_factors' own dtype note is about
    (float64 here would make the full-canvas and scratch-canvas paint paths
    differ by an LSB).

    Length comes from seam_edge_indices itself rather than being
    recomputed from the diamond's used-column count, so the 1:1 alignment
    is structural instead of an invariant two functions have to maintain
    separately -- same shape as _shadow_factors calling shadow_quad_indices.

    Deliberately NOT scaled with half_h. A fixed 1px is what keeps the seam
    from compounding with the band: the band drawn onto tile N reaches N's
    own top-edge row, exactly where N's seam goes, and _shadow_factors is
    at its 1.0 no-op endpoint there in all but two configurations of
    tile_px {8,16,32,64,128} x elev_step_pct {25,50,100,200} --
    (16, 25) and (8, 50), where 2 pixels per band sit on a span == 1
    column near the apex and carry the full CONTACT_SHADE. Worst case is
    SEAM_SHADE * CONTACT_SHADE = 0.39, dark but nowhere near black --
    accepted, and pinned by tests/test_seam_line.py so it cannot silently
    grow. Any thicker seam reaches into depth < span - 1 rows and compounds
    generally. The relative weight of 1px does grow on small mips (25% of
    a tile_px=8 diamond's rows vs 3.1% at 64), which is acceptable rather
    than a flaw: the band degenerates the same way there, since its own
    ramp floors at 1px too."""
    if side == "apex":
        _dst_y, dst_x = iso_geometry.seam_apex_indices(tile_px)
    else:
        _dst_y, dst_x = iso_geometry.seam_edge_indices(tile_px, side)
    return np.full(dst_x.size, SEAM_SHADE, dtype=np.float32)


@lru_cache(maxsize=2048)  # N2 fetches 45 per level per composite; four levels must not thrash
def _shadow_factors(tile_px: int, rise_px: int, side: str) -> np.ndarray:
    """float32 darkening factors aligned 1:1 with
    iso_geometry.shadow_quad_indices(tile_px, rise_px, side)'s own output
    (same call, same cache key shape) -- CONTACT_SHADE at depth=0 (the row
    touching the caster's diamond), ramping linearly back to exactly 1.0
    (no darkening) over the next CONTACT_RAMP_DIVISOR-th of half_h rows,
    and staying at 1.0 for the rest of that column's exposed sliver.

    The ramp length is a fixed fraction of a tile, NOT the column's own
    span and NOT rise_px -- see CONTACT_RAMP_DIVISOR's own comment for the
    measurements behind that. Normalizing on span (what this did until
    2026-08-15) spread one step's worth of darkening over up to 82% of the
    neighbor tile, so the whole up-screen half of a hill read as a lattice
    of dark triangles.

    min(span, ramp), not a bare ramp: near the caster's apex the wedge has
    tapered to fewer rows than the ramp itself, and ramping over the full
    length there would leave the band's last row still visibly darkened,
    i.e. a hard truncation edge exactly where the taper is most visible.
    Compressing the ramp into the shorter sliver fades it out properly.

    INCLUSIVE endpoint (`eff - 1`), unlike the exclusive `depth / span`
    this used until 2026-08-15. Exclusive was there only to dodge a
    divide-by-zero at span == 1, and it means no column ever actually
    reaches 1.0 -- harmless over a 30-row span (last row lands at 0.99)
    but not over a 3-row one near the apex, which ends at 0.88 and so
    keeps exactly the truncation edge the paragraph above is about. The
    max(1, ...) makes the endpoint safe instead of avoiding it: it covers
    both eff == 1 (a single-pixel apex column, which stays at
    CONTACT_SHADE either way) and half_h < CONTACT_RAMP_DIVISOR
    (tile_px=8). One consequence worth stating: the ramp darkens ramp - 1
    rows, not ramp, since its last row is the 1.0 endpoint itself.

    Reaching EXACTLY 1.0 rather than merely close is deliberate and safe:
    _clipped_darken's multiply-and-truncate is an exact round trip at
    factor 1.0, so the tail of a long sliver is a genuine no-op rather
    than a slow fade the caller pays for and nobody can see. The band
    GEOMETRY is unchanged by any of this -- it is still exactly the
    exposed sliver (a deliberate choice), which is what
    keeps this a render_iso.py policy change with no geometry, test-oracle or
    swept-bbox consequences.

    float32, not float64: _clipped_darken multiplies this against a uint8
    image and truncates back to uint8 either way, and pinning the
    intermediate dtype is what keeps the full-canvas and scratch-canvas
    paint paths bit-identical."""
    if side == "apex":
        _dst_y, _dst_x, depth, span = iso_geometry.shadow_apex_indices(tile_px, rise_px)
    else:
        _dst_y, _dst_x, depth, span = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
    _half_w, half_h = iso_geometry.half_dims(tile_px)
    ramp = max(1, half_h // CONTACT_RAMP_DIVISOR)
    if side == "apex":
        # HALF the band's ramp, and this is a measured choice, not a knob.
        # The wedge's own span is the diagonal's full exposure (24 rows at
        # tile_px=64, elev_step_pct=50), so a shared ramp would render it
        # 4 rows thick where the band it bridges has already tapered to 2
        # at the junction. That step reads as a horizontal barb hanging off
        # every tile's apex, which is the "crisp line that visibly thickens
        # into a lump" failure a prior design pass worried about. Halving
        # matches the junction thickness exactly, so the
        # contour runs at even weight through the join.
        ramp = max(1, ramp // 2)
    eff = np.minimum(span, np.int64(ramp))
    denom = np.maximum(eff - 1, np.int64(1))
    # denom.astype(np.float32) is mandatory, not redundant -- do NOT
    # simplify it away: float32 / int64 promotes to float64, which breaks
    # the bit-identity pinning this function's whole dtype note is about.
    t = np.minimum(depth.astype(np.float32) / denom.astype(np.float32), np.float32(1.0))
    return (1 - (1 - CONTACT_SHADE) * (1 - t)).astype(np.float32)


def _clipped_darken(
    img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, factors: np.ndarray, *, extent=None
) -> None:
    """img[base_y+dst_y, base_x+dst_x] *= factors, in place, dropping any
    destination pixel outside img's own bounds -- the multiplicative,
    darkening-only counterpart to _clipped_paint (see that function's own
    docstring for why clipping matters at all: a scratch-canvas call site
    composites a rect where most candidate tiles only partially overlap).

    The clip exists for the scratch-canvas `offset` call site only. It
    used to be load-bearing at the full-canvas call site too, back when a
    contact-shadow band was a constant rise_px-tall strip that could
    overshoot its neighbor entirely -- that is no longer true: the band is
    now a subset of the back neighbor's own diamond by construction (see
    iso_geometry.shadow_quad_indices), and every real tile's diamond is in
    bounds on the full canvas by construction too, so the full-canvas call
    site no longer drops shadow pixels. Kept unconditional anyway, same as
    _clipped_paint: the scratch-canvas case still needs it for partial-rect
    overlap, and one shared implementation is what keeps the two paths from
    drifting.

    Must stay MULTIPLICATIVE and bounded below by a non-zero factor (see
    CONTACT_SHADE): `0 * f == 0` is what keeps a still-unpainted background
    pixel exactly zero (the zero-initialized scratch canvas's own byte-
    identity invariant -- see composite_rect_iso's docstring), and a
    factor that could reach exactly 0 would turn an already-painted pixel
    into a false "unpainted" gap for verify_iso_render.check_full_coverage,
    whose oracle uses "black <=> unpainted" as its own stand-in.

    extent is _clipped_paint's own scalar pre-check, same contract and same
    two branches; see that function's docstring. The fast path here is the
    identical gather/multiply/cast/scatter chain below with only the mask
    skipped, so it is byte-identical rather than merely equivalent."""
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        h, w = img.shape[0], img.shape[1]
        if base_y + y_lo >= 0 and base_y + y_hi < h and base_x + x_lo >= 0 and base_x + x_hi < w:
            ay = base_y + dst_y
            ax = base_x + dst_x
            img[ay, ax] = (img[ay, ax] * factors[:, None]).astype(np.uint8)
            return
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax, factors = ay[in_bounds], ax[in_bounds], factors[in_bounds]
    img[ay, ax] = (img[ay, ax] * factors[:, None]).astype(np.uint8)


def _render_tile_iso(
    img: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    terrain_override: int | None = None,
    textures: bool = True,
    blend=None,
) -> None:
    """Paints one tile's skirts (if it's higher than its "left" or "right"
    neighbor -- see iso_geometry.skirt_quad_indices' docstring for exactly
    which two of a tile's four grid-neighbors those are, and why only
    those two can ever be visible), then its own top diamond, then a
    contact shadow onto whatever's already painted behind it (if it's
    higher than its "up_left"/"up_right" back neighbor -- see
    iso_geometry.shadow_quad_indices' docstring) into img, in place -- the
    iso counterpart to render_tile(), same "one function so the two paths
    can't drift" discipline that function already documents.

    Top face: plain terrain color/texture, no elevation-based brightness --
    same removal as render_tile(), same reasoning (see that function's
    docstring): real per-tile Z displacement already shows elevation here,
    a brightness hint on top of that is redundant. Skirt (vertical cliff
    face) shading, the contact shadow and the seam line are unrelated and
    unaffected -- SKIRT_SHADE/CONTACT_SHADE/SEAM_SHADE below are fixed
    darkening, not elevation-dependent (a tile is shaded because of a
    height DIFFERENCE against a neighbor, never because of its own
    absolute elevation), and stay.

    Known consequence, accepted, with sprites off only. Unit marks on a
    shadowed back tile get darkened too: a unit's own tile is always safe
    (drawn after this function returns for that tile), but a mark sitting on
    a tile this one's shadow reaches was already painted in an earlier
    depth_order iteration (_paint_tile_and_units_iso's interleave). If a
    darkened unit dot stops reading as its player's color, the fix is a
    lighter CONTACT_SHADE, not trying to exempt units. With sprites on, all
    terrain is down before any object (_paint_candidates_iso), so a shadow
    darkens ground only, never a sprite or a mark standing behind the step.

    offset shifts the tile's canvas-absolute screen position by (-offset_x,
    -offset_y) before painting, and every write is clipped to img's own
    bounds (see _clipped_paint) -- render_terrain_iso_with_proj()'s full-map
    call site passes the default (0, 0) against a full-sized canvas (a
    no-op: every tile is already proven to land fully in bounds there), so
    this parameter is additive, not a behavior change for that established,
    verified path. refresh_region_iso() (Phase 4) is the real caller: img
    there is a small scratch canvas covering just the dirty region, offset
    is that region's own (x0, y0), and most candidate tiles only partially
    overlap it by construction -- clipping is what keeps this function the
    single shared implementation both paths use instead of a parallel,
    could-drift copy.

    terrain_override, when given, replaces tile.terrain_id for the texture
    lookup only -- skirts, seam and contact shadow all sample from the same
    top_block this produces, so a farm tile's stepped cliff faces carry
    crop texture rather than whatever grass/dirt the tile was actually
    painted with (P3 farm-terrain: a farm renders as its footprint's real
    terrain, not a coloured mark -- see SpriteLayer.farm_by_tile). This is
    the ONLY thing terrain_override changes; tile.terrain_id itself, and
    everything the caller derives from the tile object, is untouched.

    textures=False (View > Layers) skips the texture lookup and takes the
    flat-colour branch -- see _tile_block()'s own note. Skirts and the seam
    follow automatically, since they sample the top_block that branch
    produces rather than fetching their own crop. blend: the tile's blend
    key (_top_block()); skirts sample the blended block, so a bleed reaches
    the cliff faces too (accepted)."""
    off_x, off_y = offset
    # elevations[tile.y, tile.x], NOT tile.elevation, for every read below --
    # identical in every other mode (Risk #6's whole point is keeping them in
    # sync), but Flat + Isometric View (Flat+Isometric plan) deliberately
    # forces this array to 0 while leaving the scenario's real tile.elevation
    # untouched. Reading the tile attribute directly bypassed that: it both
    # positioned every terrain diamond at its real height AND mixed a real
    # own-elevation against a forced-zero neighbor in every delta below
    # (tile.elevation - elevations[neighbour]), producing exactly the
    # Stepped-looking grid of skirts/shadows/seams the flattening was
    # supposed to remove.
    own_elev = int(elevations[tile.y, tile.x])
    base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, own_elev, proj)
    base_x -= off_x
    base_y -= off_y

    # Every positive neighbour delta in px, read once: 0 for an off-map
    # neighbour or a non-positive delta, the exact "draws nothing" gates
    # each loop below applies. elev_step >= 1, so px > 0 iff delta > 0.
    tx, ty = tile.x, tile.y
    step = proj.elev_step
    drop_left = max(own_elev - int(elevations[ty, tx - 1]), 0) * step if tx > 0 else 0
    drop_right = max(own_elev - int(elevations[ty + 1, tx]), 0) * step if ty + 1 < map_h else 0
    rise_ul = max(own_elev - int(elevations[ty - 1, tx]), 0) * step if ty > 0 else 0
    rise_ur = max(own_elev - int(elevations[ty, tx + 1]), 0) * step if tx + 1 < map_w else 0
    rise_diag = max(own_elev - int(elevations[ty - 1, tx + 1]), 0) * step if tx + 1 < map_w and ty > 0 else 0

    # Early reject, before the texture crop: the union of every sub-paint's
    # own extent, so wholly outside here means each paint below was a no-op.
    # Terrain only, never lifted to the caller; see iso_tile_extent.
    ext = iso_geometry.iso_tile_extent(tile_px, drop_left, drop_right, rise_ul, rise_ur, rise_diag)
    if ext is not None:
        y_lo, y_hi, x_lo, x_hi = ext
        h, w = img.shape[0], img.shape[1]
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return

    top_block = _top_block(tile, tile_px, textures, blend, terrain_override)

    # Every paint is collected first, in paint order, then run on one backend
    # below, so the gates stay shared and the native call is one per tile.
    skirts = []
    for side, drop_px in (("left", drop_left), ("right", drop_right)):
        if drop_px <= 0:
            continue  # map edge, or not higher than that neighbor -- no visible drop
        dst_y, dst_x, src_y, src_x = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
        # Same producer, same key: the extent memo rides the very arrays
        # the line above just fetched. See iso_geometry.index_extent.
        extent = iso_geometry.index_extent(iso_geometry.skirt_quad_indices, tile_px, drop_px, side)
        skirts.append((dst_y, dst_x, src_y, src_x, _skirt_lut(side), extent))

    diamond = iso_geometry.diamond_indices(tile_px)
    darkens = []

    # Seam line: a 1px contour along this tile's OWN two up-screen diamond
    # edges wherever the neighbor behind that edge is lower -- this tile's
    # own pixels, so it has to run after the top-face paint, and
    # before the contact-shadow loop below only for readability (the two
    # write disjoint rows: the seam is at tops[c], the band strictly above
    # it, off this diamond entirely).
    #
    # Its own loop, NOT folded into the shadow loop: that one `continue`s
    # on an empty band, which is the whole elev_step_pct=200 case -- and
    # that case, where a caster fully hides its neighbor and the up-screen
    # half of a hill would otherwise be featureless, is precisely what the
    # seam exists to fix. See iso_geometry.seam_edge_indices for why the
    # band alone leaves each terrace edge dashed at every other pct too.
    seam_qualified = False
    for side, rise_px in (("up_left", rise_ul), ("up_right", rise_ur)):
        if rise_px <= 0:
            continue  # map edge, or no height discontinuity -- a seam would be a grid outline on flat ground
        seam_qualified = True
        seam_dst_y, seam_dst_x = iso_geometry.seam_edge_indices(tile_px, side)
        extent = iso_geometry.index_extent(iso_geometry.seam_edge_indices, tile_px, side)
        darkens.append((seam_dst_y, seam_dst_x, _seam_factors(tile_px, side), extent))

    # The two apex columns, once, if EITHER side qualified -- they belong
    # to neither side's range (see iso_geometry.seam_apex_indices): the old
    # strict partition left a one-column hole at every tile apex on a run
    # where only one side drew, which is what made the line a dash rather
    # than a contour at coarse tile_px. Drawing them here rather than
    # widening the qualifying side is what keeps SEAM_SHADE from squaring
    # on the tile's most visible column when both neighbors are lower.
    if seam_qualified:
        apex_dst_y, apex_dst_x = iso_geometry.seam_apex_indices(tile_px)
        extent = iso_geometry.index_extent(iso_geometry.seam_apex_indices, tile_px)
        darkens.append((apex_dst_y, apex_dst_x, _seam_factors(tile_px, "apex"), extent))

    # Contact shadow: darkens whatever's ALREADY painted behind this tile
    # (a smaller-d, earlier-painted tile in depth_order) when this tile is
    # higher than that back neighbor -- the up-screen counterpart to the
    # skirt loop above, using the tile's own two BACK-facing edges instead
    # of its two front-facing ones (see iso_geometry.shadow_quad_indices'
    # docstring for exactly which neighbors "up_left"/"up_right" mean and
    # why paint order makes this safe: the band never touches this tile's
    # own diamond/skirts or any same-d tile, only already-painted smaller-d
    # ones). No texture is sampled -- this only darkens img in place via
    # _clipped_darken, never paints new color.
    for side, rise_px in (("up_left", rise_ul), ("up_right", rise_ur)):
        if rise_px <= 0:
            continue  # map edge, or not higher than that back neighbor -- no shadow to cast
        s_dst_y, s_dst_x, _depth, _span = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
        if s_dst_y.size == 0:
            # This tile fully hides that neighbor -- nothing exposed to
            # shade. Skipping is not needed for correctness (the empty path
            # is inert), but at elev_step_pct=200 EVERY band on the map is
            # empty, and without this a 480x480 map pays ~460k pointless
            # cache lookups per full render.
            continue
        # Never None here: the .size guard just above already returned on
        # the only empty case, and index_extent returns None only for that.
        extent = iso_geometry.index_extent(iso_geometry.shadow_quad_indices, tile_px, rise_px, side)
        darkens.append((s_dst_y, s_dst_x, _shadow_factors(tile_px, rise_px, side), extent))

    # The band's two apex columns, once, onto the DIAGONAL back neighbor
    # (x+1, y-1). Neither side of the loop above can reach them: each
    # apex column pairs with one of the diamond's two unused columns, so
    # shadow_quad_indices' `used[partner]` mask zeroes it, correctly (see
    # iso_geometry.shadow_apex_indices). That left a 2px hole at every
    # junction along a terrace, which is why a run of adjacent casters
    # read as separate blocks rather than one band.
    #
    # The diagonal's elevation is a GATE here, never a size: the extent
    # comes from the caster alone, which is what keeps this clear of the
    # taper regress a prior design pass ruled out. Paint order is safe
    # because (x+1, y-1) is d-2 under depth_order's y-x key,
    # so it is always already painted, unlike a same-d tile.
    #
    # Gated on seam_qualified too, not just the diagonal: a caster can be
    # higher than its diagonal while level with BOTH direct back
    # neighbors, and darkening the apex there would be a lone floating
    # mark with no band on either side of it to bridge.
    if seam_qualified and rise_diag > 0:
        a_dst_y, a_dst_x, _depth, _span = iso_geometry.shadow_apex_indices(tile_px, rise_diag)
        if a_dst_y.size:
            # Non-None for the same reason the band's own extent is.
            extent = iso_geometry.index_extent(iso_geometry.shadow_apex_indices, tile_px, rise_diag)
            darkens.append((a_dst_y, a_dst_x, _shadow_factors(tile_px, rise_diag, "apex"), extent))

    native = composite_backend.native
    if native is not None:
        native.render_tile_iso(img, base_y, base_x, top_block, skirts, diamond, darkens)
        return
    for dst_y, dst_x, src_y, src_x, lut, extent in skirts:
        _clipped_paint(img, base_y, base_x, dst_y, dst_x, lut[top_block[src_y, src_x]], extent=extent)
    dst_y, dst_x, src_y, src_x = diamond
    extent = iso_geometry.index_extent(iso_geometry.diamond_indices, tile_px)
    _clipped_paint(img, base_y, base_x, dst_y, dst_x, top_block[src_y, src_x], extent=extent)
    for dst_y, dst_x, factors, extent in darkens:
        _clipped_darken(img, base_y, base_x, dst_y, dst_x, factors, extent=extent)


def render_terrain_iso(
    scenario: LoadedScenario, with_units: bool = True, grid: GridBake = DEFAULT_GRID, blend: bool = True
) -> np.ndarray:
    """The Stepped-mode counterpart to render_terrain(): real per-tile
    vertical displacement via iso_geometry's projection primitives, instead
    of render_terrain()'s flat screen position (neither mode applies any
    elevation-based brightness -- see render_tile()'s own docstring for why
    that was removed). Returns a (canvas_h, canvas_w, 3) uint8 array -- a
    different shape than render_terrain()'s (h*tile_px, w*tile_px, 3) for the same
    scenario, so callers must not assume the two are interchangeable
    (overlay_units()'s own shape assertion would catch a mismatch
    immediately if something tried to combine them).

    with_units draws every unit at its own tile's elevation, interleaved
    with terrain in the same depth order (Phase 5) -- see
    _paint_tile_and_units_iso()'s docstring for the occlusion rule, and for
    what changes once sprites are on.

    Thin wrapper around render_terrain_iso_with_proj() for callers that only
    need the pixels (tools/dump_scenario.py, save_png) -- see that function
    for the version Phase 3's viewer uses, which also returns the elevations
    array and IsoProjection screen_to_tile() needs (Risk #6: those two must
    never drift from what actually got painted)."""
    img, _elevations, _proj = render_terrain_iso_with_proj(scenario, with_units=with_units, grid=grid, blend=blend)
    return img


def elevations_and_proj(scenario: LoadedScenario) -> tuple[np.ndarray, iso_geometry.IsoProjection]:
    """(elevations, proj) only -- no tile_grid, no compositing -- the cheap
    half of render_terrain_iso_with_proj() that Phase B-C's IsoChunkCache-
    backed viewer init needs to size a cache and answer hit-tests, without
    first paying for a full ~1-2s composite it's about to render lazily,
    chunk by chunk, instead. Skips
    building the (h, w) tile_grid nested list _terrain_grid_and_elevations()
    builds alongside elevations -- the chunk-cache path never needs a
    prebuilt grid, since composite_rect_iso() fetches each candidate tile
    lazily via mm.get_tile() per chunk.

    Computes proj via the exact same iso_geometry.canvas_size_and_origin(w,
    h, tile_px, MIN_ELEVATION, MAX_ELEVATION) call render_terrain_iso_with_
    proj() itself makes (see that function's own comment on the fixed
    range) -- the one-liner is duplicated, not shared, but it has no
    decision logic to drift between the two call sites, so this doesn't
    reopen Risk #6 (two callers computing subtly different projections) the
    way two independently-derived formulas would."""
    w, h, tile_px, elevations = _dims_and_elevations(scenario)
    proj = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION,
        elev_step_pct=settings.get_elev_step_pct(),
    )
    return elevations, proj


def render_terrain_iso_with_proj(
    scenario: LoadedScenario,
    with_units: bool = True,
    with_sprites: bool = False,
    grid: GridBake = DEFAULT_GRID,
    blend: bool = True,
) -> tuple[np.ndarray, np.ndarray, iso_geometry.IsoProjection]:
    """render_terrain_iso()'s real body, additionally returning the (h, w)
    elevations array and IsoProjection the render was actually computed
    from. render_terrain_iso() itself only exists to keep its established
    callers' return type unchanged (decision #1: additive, no drift) --
    Phase 3's viewer needs the extra two values itself, for
    iso_geometry.screen_to_tile() hit-testing against the exact same
    projection and elevation snapshot that produced the pixels currently on
    screen (per Risk #6, IsoProjection is deliberately immutable and meant
    to be constructed once and threaded through, never rebuilt by a second
    caller who could compute a subtly different one). blend=False keeps the
    pre-blending render (blend_effective())."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    tile_grid, elevations = _terrain_grid_and_elevations(scenario)
    # Fixed legal range, not this file's own observed min/max -- see
    # iso_geometry.MIN_ELEVATION/MAX_ELEVATION's own comment for why: it's
    # what keeps the canvas shape (and therefore every screen position)
    # stable across an elevation edit, which Phase 4's incremental redraw
    # (refresh_region_iso) depends on.
    min_elev, max_elev = iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    proj = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, min_elev, max_elev, elev_step_pct=settings.get_elev_step_pct()
    )

    # canvas_size_and_origin() reserves no headroom for skirts -- a
    # compositing-policy question it explicitly leaves to its caller, see
    # its own docstring. A skirt can hang at most the map's full observed
    # elevation range below its own tile, so pad by that worst case.
    skirt_headroom = (max_elev - min_elev) * proj.elev_step
    img = np.zeros((proj.canvas_h + skirt_headroom, proj.canvas_w, 3), dtype=np.uint8)

    units_by_tile = _units_by_tile(scenario) if with_units else {}
    # with_sprites defaults False so this stays byte-identical to what it has
    # always rendered; P3-g4's measurement is what decides whether sprites
    # become the default. Opting in here is what gives the stitched-chunk
    # check a full-render ground truth to compare against.
    # hero_glow follows the caches' default layer state, so stitched-vs-full stays exact on heroes.
    sprites = (
        sprite_draws_by_anchor(scenario, proj, elevations, hero_glow=DEFAULT_LAYERS.hero_glow)
        if (with_units and with_sprites)
        else None
    )
    keys = _full_blend_keys(scenario, blend)
    order = iso_geometry.depth_order(w, h).tolist()
    _paint_candidates_iso(
        img, [tile_grid[y][x] for x, y in order], None if keys is None else [keys[y * w + x] for x, y in order],
        units_by_tile, tile_px, proj, elevations, w, h, sprites=sprites, grid=grid,
    )
    return img, elevations, proj


def composite_rect_iso(
    scenario: LoadedScenario,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    elevations: np.ndarray,
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
) -> np.ndarray:
    """Composites the half-open screen rect [x0, x1) x [y0, y1) in
    isolation and returns it as a fresh (y1-y0, x1-x0, 3) uint8 array --
    Track B's "rect-keyed core": both
    refresh_region_iso() (once per edit) and Phase B-B's IsoChunkCache
    (once per visible chunk) composite through this SAME function, just
    over different rects, so a chunk's pixels can never depend on which
    OTHER rects happen to have been requested or in what order --
    tools/verify_iso_chunks.py's load-bearing correctness bar.

    units_by_tile/building_bboxes (see _units_by_tile()/
    _building_bboxes_iso()) are REQUIRED, precomputed by the caller rather
    than computed here: _units_by_tile() alone walks every unit in the
    scenario (~11k on this project's bigger real files), which would
    dominate if repeated once per chunk. Pass {} for both when
    with_units=False.

    Uses iso_geometry.tiles_in_screen_rect() for the terrain candidate list
    -- an O(candidates) analytic enumeration, not the O(w*h) full-grid scan
    refresh_region_iso() used before Phase B-A/B-B (that scan is still
    fine at its old once-per-edit cadence, but wrong to repeat per chunk;
    measured on the largest real map, the analytic form beats the scan by
    ~10-80x at chunk-sized and quarter-canvas rects but is ~2.5x slower at
    a full-canvas rect, 17ms vs 6.5ms). "Bystander" buildings -- whose footprint
    can overlap this rect even when their own center tile's terrain diamond
    doesn't (every footprint tile draws at the CENTER tile's elevation, not
    its own -- see _unit_iso_footprint/_draw_unit_iso) -- are merged in via
    building_bboxes, deduplicated against the terrain candidates, then the
    combined (small) set is re-sorted into the same depth_order (ascending
    d = y-x, x ascending tiebreak) tiles_in_screen_rect() already used, so
    the final paint order exactly matches what filtering a full
    depth_order(w, h) array would have produced -- just without ever
    building that full array.

    bystander_grid is an optional build_bystander_grid() index over that same
    building_bboxes dict, narrowing that merge from a full walk to the cells
    this rect touches. None (the default) keeps the full walk, which is what
    refresh_region_iso(), the tools and the tests use.

    layers (View > Layers) reaches the leaf as a plain `textures` bool, so
    the dataclass stays at this boundary and _render_tile_iso keeps an
    honest signature. The farm layer is NOT applied here: it is decided
    where the SpriteLayer is built (sprite_draws_by_anchor's with_farms), so
    by the time `sprites` arrives the farm tiles are already in or out.

    Never mutates elevations -- only dirty_screen_bbox_iso() does that (see
    its own docstring for why that separation matters for chunk-order
    independence).

    unit_pack (Batch F N2): the level's native_composite.UnitPack, built from
    this same units_by_tile and sprites (or native_composite.NO_UNITS when
    with_units is off). Given one and the native backend, the whole loop
    below runs as one native call; without one, native runs per tile.

    terrain_ids: the calling cache's per-tile terrain-id mirror for that
    native call (see native_composite's docstring). None reads the tiles
    live; the per-tile loop below always does.

    Blending (layers.terrain_blend, blend_effective()) reads its keys from
    terrain_ids and layer_ids, the cache's whole-map mirrors (fresh ones
    when either is None), through key_map, the cache's memo of them
    (_rect_blend_keys()). The whole-rect kernel composes each blended top
    block itself, byte-identical to terrain_blend.blended_block()."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    candidates = _rect_candidates(scenario, x0, y0, x1, y1, proj, with_units, building_bboxes, bystander_grid)

    # Scratch canvas local to the rect, not the full map -- composited tiles
    # write into it via _render_tile_iso's offset/clip support (a candidate
    # tile's own footprint can extend past the rect on any side, since
    # "intersects" isn't "is contained"), so the caller gets back exactly
    # this rect's own pixels with no separate "clear first" step needed
    # (scratch already starts at zero).
    #
    # Every real pixel in the rect does get painted by some candidate, not
    # just left at that zero background: candidates is a strict superset of
    # "every tile a full render would paint into this rect" (the swept-
    # elevation-range footprint test tiles_in_screen_rect() uses can only
    # ever be looser than the real, current-elevation footprint test full
    # rendering uses, since the current elevation is always inside
    # [min_elev, max_elev]), and compositing that superset in depth_order,
    # through the SAME _paint_candidates_iso() loop the full render uses
    # (interleaved with sprites off, terrain then objects with them on),
    # reproduces exactly what a full render's own proven-gap-free tiling
    # (Phase 2's check_full_coverage) would have painted there, units
    # included: a rect's candidates hold every sprite whose ink touches it
    # and every tile whose terrain could cover that ink.
    scratch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sprite_layer = sprites if with_units else None
    blend = _rect_blend_keys(scenario, candidates, layers, terrain_ids, layer_ids, key_map)
    native = composite_backend.native
    if native is not None and unit_pack is not None and candidates.shape[0]:
        from descape import native_composite

        if native_composite.composite_iso(
            native, scratch, scenario, x0, y0, candidates, elevations, proj, tile_px,
            unit_pack if with_units else None, layers.terrain_textures, grid, terrain_ids=terrain_ids,
            blend=blend,
        ):
            return scratch
    # The per-tile loop wants the keys themselves; only it pays for the list.
    keys = None if blend is None else blend.as_list()
    # .tolist() unboxes in C, and every candidate is already clamped on-map
    # (tiles_in_screen_rect + unit_occupied_tiles), so get_tile's checks can't fire.
    _paint_candidates_iso(
        scratch, [mm.terrain[cy * w + cx] for cx, cy in candidates.tolist()], keys, units_by_tile, tile_px, proj,
        elevations, w, h, offset=(x0, y0), sprites=sprite_layer, textures=layers.terrain_textures, grid=grid,
    )
    return scratch


def refresh_region_iso(
    img: np.ndarray,
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    tile_px: int,
    with_units: bool = True,
) -> tuple[int, int, int, int] | None:
    """Stepped mode's incremental counterpart to refresh_tiles() -- Phase 4
    of the parent plan, extended in Phase 5 to redraw units in the same
    pass (see _paint_tile_and_units_iso()'s docstring for why that has to be
    one shared per-tile step, not terrain fully repainted followed by a
    separate unit overlay: a unit's occlusion relative to terrain depends
    on paint ORDER, and a bolt-on "units after" pass would draw every
    affected unit on top regardless of whether a later-depth terrain tile
    should actually cover it). Repaints only the screen region a set of
    just-edited tiles (or units -- see with_units below) could possibly
    affect, in place, instead of render_terrain_iso_with_proj()'s full
    ~1-2s recomposite.

    Phase B-B: now a thin wrapper -- dirty_screen_bbox_iso() for "what
    changed" (also where elevations gets mutated), composite_rect_iso() for
    "paint that rect" -- kept as its own function, with its own signature
    unchanged: the same two pieces back Phase B-B's chunk cache without an
    img array in the picture at all.

    **Its callers are this project's tools and tests only.** The live
    Stepped edit path is ViewerWindow._apply_dirty -> dirty_screen_bbox_iso
    -> IsoChunkCache.patch, which never comes through here; naming
    ViewerWindow as a caller (as this docstring used to) has been stale
    since the chunk cache landed. What this function is now is the
    byte-identity harness AROUND that real path -- verify_iso_units.py's
    incremental checks compare it against a fresh full render, which is why
    it is kept rather than deleted.

    with_units=False skips units entirely -- only used by callers that want
    a pure-terrain incremental patch; the real viewer always wants the
    default.

    Returns the repainted (x0, y0, x1, y1) canvas-pixel bbox for the caller
    to patch into its displayed pixmap, or None if there's nothing to draw
    -- see dirty_screen_bbox_iso()'s own docstring for exactly when that
    happens."""
    bbox = dirty_screen_bbox_iso(scenario, dirty_indices, elevations, proj, with_units)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox

    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    units_by_tile = _units_by_tile(scenario) if with_units else {}
    building_bboxes = _building_bboxes_iso(scenario, w, h, proj, elevations) if with_units else {}

    scratch = composite_rect_iso(
        scenario, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, building_bboxes, with_units
    )
    img[y0:y1, x0:x1] = scratch
    return bbox


def _paint_tile_and_units_iso(
    img: np.ndarray,
    tile,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
    blend=None,
) -> None:
    """Terrain, then that tile's own units: the single per-tile step of the
    sprites-off render, which the full loop and every rect composite call in
    the SAME depth_order position, so no two paths can draw marks in a
    different relative order than terrain occlusion requires (Phase 5).

    Sprites off, marks stay interleaved tile by tile: a terrain tile painted
    LATER in depth_order (closer to the camera) paints over a mark drawn
    EARLIER at a farther tile.

    Sprites on, that interleave cut every big sprite along a tile-edge
    staircase (GH #107): each later tile's terrain repainted over the ink
    overhanging it. So _paint_candidates_iso runs the two halves below as two
    passes over the same candidates: all terrain first, then objects, and a
    nearer tile covers earlier ink only where it stands strictly higher than
    that ink's base (_occlude_iso). Flat or lower ground never covers ink.

    A multi-tile unit contributes ONE diamond here, not its whole footprint:
    _units_by_tile() buckets it into every tile it covers, so each of its
    diamonds is painted at that tile's own depth position. That is what
    makes the interleaving argument above hold across a building's whole
    slab rather than only at its own tile.

    A farm-terrain tile (sprites.farm_by_tile) is a THIRD case alongside
    the diamond and the sprite, and it is resolved right here rather than
    in a separate pass: it changes what _render_tile_iso paints as this
    tile's own terrain, not something composited on top of it, so the
    override has to reach that call before it runs. The perimeter stroke
    that follows still respects the same per-tile depth_order position as
    everything else here -- an edge belongs to its own tile's diamond, so
    it paints at that tile's own moment, same as the seam line does."""
    _paint_terrain_iso(
        img, tile, tile_px, proj, elevations, map_w, map_h, offset=offset, sprites=sprites,
        textures=textures, grid=grid, blend=blend,
    )
    _paint_objects_iso(img, tile, units_by_tile, tile_px, proj, elevations, offset=offset, sprites=sprites)


def _paint_terrain_iso(
    img: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
    blend=None,
) -> None:
    """A tile's terrain half: its terrain (a farm's override included), the
    farm outline, the grid. `sprites` is read for its farm tiles only."""
    farm = sprites.farm_by_tile.get((tile.x, tile.y)) if sprites is not None else None
    terrain_override = farm[0] if farm is not None else None
    _render_tile_iso(
        img, tile, tile_px, proj, elevations, map_w, map_h,
        offset=offset, terrain_override=terrain_override, textures=textures, blend=blend,
    )
    if farm is not None:
        _, outline_color, edge_mask = farm
        off_x, off_y = offset
        # elevations[tile.y, tile.x], not tile.elevation -- must match
        # _render_tile_iso's own base position above (see its comment), or
        # this outline stops lining up with the diamond it's meant to trace.
        base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, int(elevations[tile.y, tile.x]), proj)
        base_x -= off_x
        base_y -= off_y
        for bit, side in _FARM_EDGE_BITS:
            if not (edge_mask & bit):
                continue
            dst_y, dst_x = iso_geometry.tile_edge_indices(tile_px, side)
            extent = iso_geometry.index_extent(iso_geometry.tile_edge_indices, tile_px, side)
            _clipped_paint_solid(img, base_y, base_x, dst_y, dst_x, outline_color, extent=extent)
    if grid.paints:
        # Part of the terrain half: the grid is ground, so it lands under
        # every sprite, and a restored terrain pixel (_occlude_iso) keeps it.
        gx, gy = iso_geometry.tile_screen_origin(tile.x, tile.y, int(elevations[tile.y, tile.x]), proj)
        _paint_grid_iso(
            img, tile.x, tile.y, tile_px, map_w, map_h, gy - offset[1], gx - offset[0], grid, elevations=elevations
        )


def _paint_objects_iso(
    img: np.ndarray,
    tile,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    ink: np.ndarray | None = None,
) -> None:
    """A tile's objects half: its marks, then the sprites anchored on it. ink,
    when given, is the two-pass ink plane (int32, INK_NONE where empty): every
    pixel a mark or a sprite touches is recorded there at the unit's base
    rise, last writer wins."""
    skip = sprites.skip_ids if sprites is not None else frozenset()
    for unit, color in units_by_tile.get((tile.x, tile.y), ()):
        if id(unit) in skip:
            continue  # its sprite paints instead, at its anchor tiles (below)
        ink_rise = _unit_mark_rise_iso(unit, proj, elevations) if ink is not None else 0
        _draw_unit_iso(
            img, unit, color, tile.x, tile.y, tile_px, proj, elevations, offset=offset, ink=ink, ink_rise=ink_rise
        )

    # Sprites after this tile's marks. Each piece of a sprite paints at its key
    # (unit_sprites.sprite_anchor_tile, or its slot), a multi-tile piece cut into
    # screen columns that each paint at their front footprint tile when that is
    # earlier (_sprite_columns, TASK-221): every pixel paints exactly once.
    if sprites is not None:
        off_x, off_y = offset
        for draw, ax, ay, rise_px in sprites.by_anchor.get((tile.x, tile.y), ()):
            _clipped_paint_rgba(
                img, ay - draw.hotspot_y - off_y, ax - draw.hotspot_x - off_x, draw.rgba, ink, rise_px
            )


def _occlude_iso(
    img: np.ndarray, terrain: np.ndarray, ink: np.ndarray, tile, tile_px: int,
    proj: iso_geometry.IsoProjection, elevations: np.ndarray, map_h: int, offset: tuple[int, int], floor: int,
) -> None:
    """Pass 2's occlusion step for one tile: inside its opaque coverage (its
    diamond and its two skirt quads, the producers _render_tile_iso paints
    them from), ink based below the tile's own height gives way to the
    terrain snapshot. Seams and contact shadows are darken-only, not coverage.
    floor is _ink_floor's: a tile no higher than it has nothing to cover."""
    tx, ty = tile.x, tile.y
    own_elev = int(elevations[ty, tx])
    step = proj.elev_step
    height = own_elev * step
    if height <= floor:
        return
    base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, own_elev, proj)
    base_x -= offset[0]
    base_y -= offset[1]
    drop_left = max(own_elev - int(elevations[ty, tx - 1]), 0) * step if tx > 0 else 0
    drop_right = max(own_elev - int(elevations[ty + 1, tx]), 0) * step if ty + 1 < map_h else 0
    for side, drop_px in (("left", drop_left), ("right", drop_right)):
        if drop_px <= 0:
            continue
        dst_y, dst_x, _sy, _sx = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
        extent = iso_geometry.index_extent(iso_geometry.skirt_quad_indices, tile_px, drop_px, side)
        _restore_under(img, terrain, ink, base_y, base_x, dst_y, dst_x, height, extent=extent)
    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)
    extent = iso_geometry.index_extent(iso_geometry.diamond_indices, tile_px)
    _restore_under(img, terrain, ink, base_y, base_x, dst_y, dst_x, height, extent=extent)


def _paint_candidates_iso(
    img: np.ndarray,
    tiles: list,
    keys: list | None,
    units_by_tile: dict,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
    sprites: SpriteLayer | None = None,
    textures: bool = True,
    grid: GridBake = DEFAULT_GRID,
) -> None:
    """Paints `tiles` (already in depth order; keys is their blend keys or
    None) into img: the loop of the full render and of composite_rect_iso's
    per-tile path, and the reference native_composite's kernel mirrors.

    Sprites off: one interleaved loop, _paint_tile_and_units_iso per tile.
    Sprites on, two passes over the same tiles. Pass 1 paints every terrain
    half. Pass 2 walks the tiles again and, per tile, first lets its terrain
    cover earlier ink based below it (_occlude_iso, against a snapshot of
    pass 1), then paints its objects.

    The pre-scan makes flat ground free: when no tile stands higher than the
    lowest ink base, nothing can be covered, so the snapshot, the ink plane
    and the occlusion step are all skipped. It never changes a pixel."""
    if sprites is None:
        for i, tile in enumerate(tiles):
            _paint_tile_and_units_iso(
                img, tile, units_by_tile, tile_px, proj, elevations, map_w, map_h, offset=offset,
                textures=textures, grid=grid, blend=None if keys is None else keys[i],
            )
        return
    for i, tile in enumerate(tiles):
        _paint_terrain_iso(
            img, tile, tile_px, proj, elevations, map_w, map_h, offset=offset, sprites=sprites,
            textures=textures, grid=grid, blend=None if keys is None else keys[i],
        )
    floor = _ink_floor(tiles, units_by_tile, sprites, lambda unit, _tile: _unit_mark_rise_iso(unit, proj, elevations))
    terrain = ink = None
    if any(int(elevations[tile.y, tile.x]) * proj.elev_step > floor for tile in tiles):
        terrain = img.copy()
        ink = np.full(img.shape[:2], INK_NONE, dtype=np.int32)
    for tile in tiles:
        if ink is not None:
            _occlude_iso(img, terrain, ink, tile, tile_px, proj, elevations, map_h, offset, floor)
        _paint_objects_iso(img, tile, units_by_tile, tile_px, proj, elevations, offset=offset, sprites=sprites, ink=ink)
