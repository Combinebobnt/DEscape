"""Renders a loaded scenario's Map + Units to an RGB numpy array.

Pure function, no Qt dependency -- backs both the headless PNG export and the
PyQt5 viewer's canvas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from descape import asset_source, iso_geometry, settings, unit_sprites
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    BUILDING_TILE_SPANS,
    FOUNDATION_TERRAIN,
    PLAYER_COLORS,
    RESOURCE_COLORS,
    TREE_COLOR,
    TREE_UNIT_IDS,
    color_for_terrain_id,
)
from descape.unit_filter import UnitFilter

NON_BUILDING_SPAN = (1, 1)


def _sprite_reach_px(proj: iso_geometry.IsoProjection) -> tuple[int, int, int, int]:
    """(left, up, right, down) canvas-pixel padding a dirty tile's own bbox
    needs so a sprite anchored on it cannot escape.

    The tile radius above is no substitute, and not because it is slightly too
    small: it models the terrain diamond, skirt and contact shadow, and no unit
    PIXEL extent at all. Sprites reach 404/650/424/200 native px from their
    hotspot (unit_sprites.MAX_SPRITE_REACH_*), which the elevation sweep
    happens to cover horizontally and badly under-covers upward.

    Integer ceil of reach * sprite_scale, plus 1px: sprite_for() rounds the
    scaled width and the scaled hotspot INDEPENDENTLY, so the scaled reach on
    the far side can come out one pixel past (w - hx) * scale."""
    scale = unit_sprites.sprite_scale(proj.half_w)
    return tuple(
        math.ceil(reach * scale) + 1
        for reach in (
            unit_sprites.MAX_SPRITE_REACH_LEFT,
            unit_sprites.MAX_SPRITE_REACH_UP,
            unit_sprites.MAX_SPRITE_REACH_RIGHT,
            unit_sprites.MAX_SPRITE_REACH_DOWN,
        )
    )

# Pixels per tile side. >1 so real per-tile texture crops (see
# asset_source.get_terrain_texture_array) show actual detail instead of
# collapsing to a single flat-colored pixel. Both values must evenly divide
# asset_source.LOADED_TEXTURE_SIZE -- see _crop_offset's assertion.
#
# Adaptive rather than one fixed constant: memory scales with tile_pixels
# SQUARED (a 480x480 "Ludicrous" map's rendered array is 2.83GB at 64px/tile,
# vs 708MB at 32px/tile -- measured, not estimated), so unconditionally using
# a large value that looks great on small/medium maps risks several GB of
# concurrent memory on the biggest real maps once the render->QPixmap
# pipeline's several copies are counted. LARGE_MAP_TILE_THRESHOLD=240 keeps
# ordinary-sized maps at full detail and only drops resolution once a map is
# big enough for that to actually matter. These are the base values before
# settings.get_graphics_quality() (Settings > Appearance) is factored in --
# see tile_pixels_for_map() -- which scales whichever of these applies by a
# power of two per quality stage away from GRAPHICS_QUALITY_DEFAULT.
SMALL_MAP_TILE_PIXELS = 64
LARGE_MAP_TILE_PIXELS = 32
LARGE_MAP_TILE_THRESHOLD = 240  # map_width or map_height beyond this, in tiles
# 240 is coincidentally also the largest of descape.scenario_new.STANDARD_MAP_SIZES
# short of the 480 "Ludicrous" outlier, and File > New Map's soft-confirm threshold for
# custom sizes -- three separate reasons landing on the same number. Every size that
# submenu can reach stays on one side or the other of a case this threshold already
# covered before that feature existed: standard sizes <=240 take this branch (as 120/
# 240 already did), custom sizes in 241..480 take the large branch bounded by 480,
# which is both the largest real map ever seen and the size this module's own memory
# measurement above was benchmarked against.


def tile_pixels_for_map(map_width: int, map_height: int) -> int:
    """The TILE_PIXELS value to render a map of this size at -- see the
    module-level comment above for why this is adaptive rather than fixed.
    A function of the map's tile dimensions plus the current
    settings.get_graphics_quality() value (scales the result by
    2**(quality - GRAPHICS_QUALITY_DEFAULT)), not a pure function of size
    alone -- but still safe for render_terrain() and overlay_units() (and
    viewer.py's MapView, for hover/click tile math) to each call
    independently and get a matching answer, since nothing changes graphics
    quality mid-render: it's only ever changed from the Settings dialog,
    between renders, never during one.

    Scaling is safe for _crop_offset's divides-LOADED_TEXTURE_SIZE-evenly
    requirement at every quality stage, since every value involved (16, 32,
    64, 128) is a power of two dividing a power-of-two texture size.

    The one exception: quality 4 ("Enhanced")'s doubling is clamped to a
    no-op on small maps (base=SMALL_MAP_TILE_PIXELS=64) rather than
    reaching 128 -- the module comment's own 2.83GB measurement at 64px/tile
    was already the largest benchmarked case, and memory scales with
    tile_pixels squared, so 128px/tile on a map that size is unmeasured and
    likely several times that. Enhanced still doubles normally on "large"
    maps (base=32 -> 64), which is exactly the benchmarked small-map value,
    so that direction is already proven safe."""
    is_large = map_width > LARGE_MAP_TILE_THRESHOLD or map_height > LARGE_MAP_TILE_THRESHOLD
    base = LARGE_MAP_TILE_PIXELS if is_large else SMALL_MAP_TILE_PIXELS
    delta = settings.get_graphics_quality() - settings.GRAPHICS_QUALITY_DEFAULT
    if delta > 0 and not is_large:
        delta = 0
    return base << delta if delta >= 0 else base >> (-delta)


def _crop_offset(x: int, y: int, texture_size: int, tile_px: int) -> tuple[int, int]:
    """Crop origin into a cached terrain texture array for tile (x, y), chosen
    so same-terrain neighboring tiles sample *contiguous* texture regions
    instead of independent, unrelated patches -- tile (x+1, y)'s crop picks up
    exactly where tile (x, y)'s left off. That makes a same-terrain area one
    continuous unrolled tiling of the source texture rather than visibly cut-up
    squares, which works because ground textures are themselves authored to
    tile seamlessly at their own edges.

    Relies on texture_size being an exact multiple of tile_px (enforced by the
    assertion below, checked against the real values every render rather than
    stated as a fixed pair here since tile_px is now adaptive per map size --
    see tile_pixels_for_map) so every crop window lands exactly on a tile_px
    boundary and never straddles the texture's own wrap point -- a plain
    slice is enough, no explicit wraparound handling needed. Would need
    revisiting (real modulo/wraparound indexing) if either tile_pixels_for_map
    return value or asset_source.LOADED_TEXTURE_SIZE ever changed to a
    non-dividing pair.

    The large-scale repeat this trades away (the same texture region recurring
    every texture_size/tile_px tiles) is deliberate: for noisy ground textures
    (grass, dirt, ...) with no single distinct landmark, that repeat is far
    less visible than a hard seam at every single tile boundary."""
    assert texture_size % tile_px == 0, (
        f"_crop_offset requires texture_size ({texture_size}) to be a multiple "
        f"of tile_px ({tile_px}) to stay seam-free without wraparound handling"
    )
    return (x * tile_px) % texture_size, (y * tile_px) % texture_size


def _tile_block(tile, tile_px: int) -> np.ndarray:
    """The tile_px x tile_px pixel content for one terrain tile -- a real
    texture crop when an AoE2DE install is configured, else terrain_palette's
    flat color. Split out of render_tile() (Phase B-E) so composite_rect_flat()
    can reuse the exact same per-tile content without going through
    render_tile()'s own img-mutating, tile.x/tile.y-derived placement -- the
    caller picks where this block goes, not this function."""
    texture = asset_source.get_terrain_texture_array(tile.terrain_id)
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        return texture[oy : oy + tile_px, ox : ox + tile_px]
    r, g, b = color_for_terrain_id(tile.terrain_id)
    return np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.uint8)


def render_tile(img: np.ndarray, tile, tile_px: int) -> None:
    """Paints one tile's tile_px x tile_px block into img, in place. The
    per-tile body shared by render_terrain() (full-map render) and
    refresh_tiles() (incremental, edit-driven redraw) -- kept as one function
    so the two paths can never drift apart on how a tile is textured.

    No elevation-based brightness shading -- removed at the user's explicit
    request (the "glowing" per-tile brightness hint was redundant once
    Stepped mode shows real per-tile Z-height displacement, and actively
    misleading in Flat mode, which has no other elevation cue at all).
    Plain terrain color/texture only."""
    px0, py0 = tile.x * tile_px, tile.y * tile_px
    img[py0 : py0 + tile_px, px0 : px0 + tile_px] = _tile_block(tile, tile_px)


def render_terrain(scenario: LoadedScenario) -> np.ndarray:
    """Terrain rendered at tile_pixels_for_map()-per-tile resolution
    (adaptive on map size -- see that function). Uses a real per-tile
    texture crop when an AoE2DE install is configured
    (asset_source.get_terrain_texture_array), falling back to
    terrain_palette's flat color per tile otherwise. Returns an
    (H*tile_px, W*tile_px, 3) uint8 array."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    img = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)

    for tile in mm.terrain:
        render_tile(img, tile, tile_px)
    return img


def refresh_tiles(
    img: np.ndarray, scenario: LoadedScenario, dirty_indices, tile_px: int
) -> None:
    """Re-paints only the tiles at dirty_indices into img, in place -- the
    incremental counterpart to render_terrain(), for use after an edit
    (descape.edit_history.EditHistory.apply/undo/redo all return exactly this
    kind of index list). A full render_terrain() call reallocates and repaints
    the whole map (up to ~2.8GB for the largest possible map at full
    resolution, see the module comment above) -- far too slow to call once per
    brush stroke, let alone per dragged tile.

    dirty_indices needs no dilation to neighboring tiles: render_tile()'s
    output depends only on the tile's own terrain_id (elevation isn't even
    read anymore -- see render_tile()'s own docstring), with no neighbor
    term (v1's known lack of real cliff geometry/cross-tile lighting, here
    working in this function's favor) -- the exact diffed set is always the
    exact set that needs repainting, nothing more."""
    terrain = scenario.map_manager.terrain
    for i in dirty_indices:
        render_tile(img, terrain[i], tile_px)


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
# consequence" this trades off -- units on shadowed tiles get darkened too,
# since there's no cheap way to exempt them without breaking the terrain/
# unit paint-order interleaving _paint_tile_and_units_iso() depends on).
CONTACT_SHADE = 0.65

# How far up-screen the contact shadow ramps back to no darkening, as a
# divisor of half_h -- so the ramp is a constant fraction of a tile at every
# tile_px, and independent of the elevation delta that produced the band.
#
# Not a cosmetic knob: shading the band's WHOLE exposed sliver (this is
# where PLAN_CONTACT_SHADOW.md's decision 1 landed) makes shadow length
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


@lru_cache(maxsize=256)
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
    exposed sliver (PLAN_CONTACT_SHADOW.md's decision 1), which is what
    keeps this a render.py policy change with no geometry, test-oracle or
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


def _clipped_darken(img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, factors: np.ndarray) -> None:
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
    whose oracle uses "black <=> unpainted" as its own stand-in."""
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax, factors = ay[in_bounds], ax[in_bounds], factors[in_bounds]
    img[ay, ax] = (img[ay, ax] * factors[:, None]).astype(np.uint8)


def _terrain_grid_and_elevations(scenario: LoadedScenario) -> tuple[list, np.ndarray]:
    """(tile_grid, elevations) for iso rendering -- tile_grid[y][x] is the
    real tile object (for its texture/terrain_id), elevations is the same
    data as a plain (h, w) int64 array (what iso_geometry.screen_to_tile
    and the neighbor-elevation lookups in _render_tile_iso() need). Built
    once per render since depth_order() visits tiles in paint order, not
    mm.terrain's own order, so random (x, y) access is needed here, unlike
    render_terrain()'s single flat loop."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_grid: list = [[None] * w for _ in range(h)]
    elevations = np.zeros((h, w), dtype=np.int64)
    for tile in mm.terrain:
        tile_grid[tile.y][tile.x] = tile
        elevations[tile.y, tile.x] = tile.elevation
    return tile_grid, elevations


def _clipped_paint(img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, values: np.ndarray) -> None:
    """img[base_y+dst_y, base_x+dst_x] = values, dropping any destination
    pixel that falls outside img's own bounds instead of letting it wrap
    (a numpy fancy-index with a negative or over-large coordinate doesn't
    raise -- it silently indexes from the other end / out of bounds, a real
    bug class, not a hypothetical one -- see refresh_region_iso's docstring
    for why Phase 4 needs this: img there is a small scratch canvas, not the
    full map canvas, so most tiles composited into it are only ever
    partially in bounds by construction, not as an edge case.

    render_terrain_iso_with_proj()'s full-canvas call site (offset (0, 0),
    img sized exactly to the canvas every tile is proven to fit inside --
    see tools/verify_iso_render.py's bounds/coverage checks) never actually
    drops a pixel HERE -- diamonds and skirts always land in bounds at the
    full-canvas call site; this exists for refresh_region_iso()'s scratch-
    canvas call site. Same for _clipped_darken() below (the contact-shadow
    compositor): its band is confined to the back neighbor's own diamond,
    so it too only ever drops pixels at the scratch-canvas call site -- see
    that function's own docstring."""
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax, values = ay[in_bounds], ax[in_bounds], values[in_bounds]
    img[ay, ax] = values


def _clipped_paint_rgba(img: np.ndarray, base_y: int, base_x: int, rgba: np.ndarray) -> None:
    """Alpha-composites an (h, w, 4) uint8 block onto img at (base_y, base_x),
    clipping to img's bounds -- _clipped_paint()'s counterpart for sprites
    (P3-g3).

    Deliberately a sibling rather than a widening of _clipped_paint(): that
    function's contract is an opaque scatter of per-pixel values at arbitrary
    (dst_y, dst_x) index arrays, and every existing caller depends on the
    overwrite being total. A sprite is the opposite shape -- a contiguous
    rectangle with a real alpha channel, most of it transparent -- so sharing
    one implementation would mean a branch inside the hottest scatter in the
    renderer for no gain.

    Clipping is a rectangle intersection rather than a per-pixel mask because
    the block is contiguous; the same "a scratch canvas is smaller than the
    thing being drawn into it" reasoning _clipped_paint()'s docstring gives
    applies here, only more often -- a sprite is far larger than a tile
    diamond, so partial containment is the normal case, not the edge one."""
    h, w = rgba.shape[:2]
    ih, iw = img.shape[:2]
    sy0, sx0 = max(0, -base_y), max(0, -base_x)
    sy1, sx1 = min(h, ih - base_y), min(w, iw - base_x)
    if sy0 >= sy1 or sx0 >= sx1:
        return
    block = rgba[sy0:sy1, sx0:sx1]
    dst = img[base_y + sy0 : base_y + sy1, base_x + sx0 : base_x + sx1]
    alpha = block[..., 3:4].astype(np.uint16)
    # Integer blend, so a fully opaque pixel reproduces the source byte exactly
    # (a float round-trip can land a 255-alpha pixel one off) -- which is what
    # keeps the stitched-chunk byte-identity check meaningful.
    dst[...] = ((block[..., :3].astype(np.uint16) * alpha + dst.astype(np.uint16) * (255 - alpha)) // 255).astype(np.uint8)


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

    Known consequence, accepted -- units on a shadowed back tile get
    darkened too: a unit's own tile is always safe (drawn after this
    function returns for that tile), but a unit sitting on a tile this
    one's shadow reaches was already painted in an earlier depth_order
    iteration, and there's no cheap way to exempt it -- a separate
    pre-unit shadow pass would break the terrain/unit paint-order
    interleaving _paint_tile_and_units_iso() exists to guarantee. If a
    darkened unit dot stops reading as its player's color, the fix is a
    lighter CONTACT_SHADE, not trying to exempt units.

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
    everything the caller derives from the tile object, is untouched."""
    terrain_id = tile.terrain_id if terrain_override is None else terrain_override
    texture = asset_source.get_terrain_texture_array(terrain_id)
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        top_block = texture[oy : oy + tile_px, ox : ox + tile_px]
    else:
        r, g, b = color_for_terrain_id(terrain_id)
        top_block = np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.uint8)

    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
    base_x -= off_x
    base_y -= off_y

    for side, nx, ny in (("left", tile.x - 1, tile.y), ("right", tile.x, tile.y + 1)):
        if not (0 <= nx < map_w and 0 <= ny < map_h):
            continue  # map edge -- no neighbor to drop toward, so no skirt
        delta = tile.elevation - int(elevations[ny, nx])
        if delta <= 0:
            continue  # this tile isn't higher than that neighbor -- no visible drop
        drop_px = delta * proj.elev_step
        dst_y, dst_x, src_y, src_x = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
        skirt = np.clip(top_block[src_y, src_x].astype(np.float32) * SKIRT_SHADE[side], 0, 255).astype(np.uint8)
        _clipped_paint(img, base_y, base_x, dst_y, dst_x, skirt)

    dst_y, dst_x, src_y, src_x = iso_geometry.diamond_indices(tile_px)
    _clipped_paint(img, base_y, base_x, dst_y, dst_x, top_block[src_y, src_x])

    # Seam line: a 1px contour along this tile's OWN two up-screen diamond
    # edges wherever the neighbor behind that edge is lower -- this tile's
    # own pixels, so it has to run after the top-face paint just above, and
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
    for side, nx, ny in (("up_left", tile.x, tile.y - 1), ("up_right", tile.x + 1, tile.y)):
        if not (0 <= nx < map_w and 0 <= ny < map_h):
            continue  # map edge -- nothing behind this edge to contour against
        if tile.elevation - int(elevations[ny, nx]) <= 0:
            continue  # no height discontinuity here -- a seam would be a grid outline on flat ground
        seam_qualified = True
        seam_dst_y, seam_dst_x = iso_geometry.seam_edge_indices(tile_px, side)
        _clipped_darken(img, base_y, base_x, seam_dst_y, seam_dst_x, _seam_factors(tile_px, side))

    # The two apex columns, once, if EITHER side qualified -- they belong
    # to neither side's range (see iso_geometry.seam_apex_indices): the old
    # strict partition left a one-column hole at every tile apex on a run
    # where only one side drew, which is what made the line a dash rather
    # than a contour at coarse tile_px. Drawing them here rather than
    # widening the qualifying side is what keeps SEAM_SHADE from squaring
    # on the tile's most visible column when both neighbors are lower.
    if seam_qualified:
        apex_dst_y, apex_dst_x = iso_geometry.seam_apex_indices(tile_px)
        _clipped_darken(img, base_y, base_x, apex_dst_y, apex_dst_x, _seam_factors(tile_px, "apex"))

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
    for side, nx, ny in (("up_left", tile.x, tile.y - 1), ("up_right", tile.x + 1, tile.y)):
        if not (0 <= nx < map_w and 0 <= ny < map_h):
            continue  # map edge -- no back neighbor to shadow onto
        delta = tile.elevation - int(elevations[ny, nx])
        if delta <= 0:
            continue  # this tile isn't higher than that back neighbor -- no shadow to cast
        rise_px = delta * proj.elev_step
        s_dst_y, s_dst_x, _depth, _span = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
        if s_dst_y.size == 0:
            # This tile fully hides that neighbor -- nothing exposed to
            # shade. Skipping is not needed for correctness (the empty path
            # is inert), but at elev_step_pct=200 EVERY band on the map is
            # empty, and without this a 480x480 map pays ~460k pointless
            # cache lookups per full render.
            continue
        _clipped_darken(img, base_y, base_x, s_dst_y, s_dst_x, _shadow_factors(tile_px, rise_px, side))

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
    nx, ny = tile.x + 1, tile.y - 1
    if seam_qualified and 0 <= nx < map_w and 0 <= ny < map_h:
        delta = tile.elevation - int(elevations[ny, nx])
        if delta > 0:
            rise_px = delta * proj.elev_step
            a_dst_y, a_dst_x, _depth, _span = iso_geometry.shadow_apex_indices(tile_px, rise_px)
            if a_dst_y.size:
                _clipped_darken(
                    img, base_y, base_x, a_dst_y, a_dst_x, _shadow_factors(tile_px, rise_px, "apex")
                )


def render_terrain_iso(scenario: LoadedScenario, with_units: bool = True) -> np.ndarray:
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
    _paint_tile_and_units_iso()'s docstring for why that interleaving, not
    a separate overlay pass, is what keeps occlusion correct.

    Thin wrapper around render_terrain_iso_with_proj() for callers that only
    need the pixels (tools/dump_scenario.py, save_png) -- see that function
    for the version Phase 3's viewer uses, which also returns the elevations
    array and IsoProjection screen_to_tile() needs (Risk #6: those two must
    never drift from what actually got painted)."""
    img, _elevations, _proj = render_terrain_iso_with_proj(scenario, with_units=with_units)
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
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    elevations = np.zeros((h, w), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    proj = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION,
        elev_step_pct=settings.get_elev_step_pct(),
    )
    return elevations, proj


def render_terrain_iso_with_proj(
    scenario: LoadedScenario, with_units: bool = True, with_sprites: bool = False
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
    caller who could compute a subtly different one)."""
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
    sprites = sprite_draws_by_anchor(scenario, proj, elevations) if (with_units and with_sprites) else None
    for x, y in iso_geometry.depth_order(w, h):
        _paint_tile_and_units_iso(
            img, tile_grid[y][x], units_by_tile, tile_px, proj, elevations, w, h, sprites=sprites
        )
    return img, elevations, proj


def _canvas_pixel_dims(proj: iso_geometry.IsoProjection) -> tuple[int, int]:
    """(width, height) in canvas pixels INCLUDING the skirt headroom below
    the canvas proper -- the same total shape render_terrain_iso_with_proj()
    allocates its img array to: (proj.canvas_h + skirt_headroom,
    proj.canvas_w, 3), skirt_headroom = (max_elev - min_elev) * elev_step
    (see that function's own comment for why). Lets dirty_screen_bbox_iso()
    clip a bbox to real canvas bounds without needing an actual img object
    to read .shape from -- the whole point of splitting it out of
    refresh_region_iso() in the first place (Phase B-B)."""
    skirt_headroom = (proj.max_elev - proj.min_elev) * proj.elev_step
    return proj.canvas_w, proj.canvas_h + skirt_headroom


def _building_bboxes_iso(
    units_by_tile: dict,
    w: int,
    h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """(px, py) -> that tile's own iso screen bbox, for every tile carrying
    at least one BUILDING (nonzero footprint radius) unit -- precomputed
    once per elevations snapshot so composite_rect_iso()'s "bystander"
    check (a building's footprint can overlap a rect even when its own
    center tile doesn't -- see that function's docstring) is an O(1) dict
    lookup per call instead of re-walking every unit and recomputing
    _unit_screen_bbox_iso() for it every time, which would dominate if
    repeated once per chunk (Phase B-B's whole reason for existing -- a
    chunk cache calls composite_rect_iso() far more often than
    refresh_region_iso()'s old once-per-edit cadence).

    One entry per (px, py): the UNION of every building at that tile's own
    bbox (real files do have tiles with >=2 building entries -- e.g. a
    decorative object placed on a building's own tile -- confirmed on 5 of
    this project's 16 example files, 20 tiles total). A union, not "the
    first with a usable bbox": refresh_region_iso()'s original per-tile
    loop tried entries until one's bbox *intersected the caller's rect*,
    which a first-bbox-wins scheme here can't reproduce (this function
    doesn't see any rect -- that test happens later, per-rect, in
    composite_rect_iso()). A union is still safe -- composite_rect_iso()'s
    own candidate set is already an accepted strict superset of "every tile
    a full render would paint" (see its docstring), so a union bbox
    flagging a tile as a bystander slightly more often than the tightest
    possible test would is a no-op extra paint, never a missed one.

    **Deliberately keyed on OWN tiles even though units_by_tile no longer
    is.** Since units_by_tile buckets a unit into every footprint tile, a
    naive pass over its items would recompute _unit_screen_bbox_iso() once
    per footprint tile -- 64 times for a Colosseum instead of once -- on the
    path whose ~15-20ms rebuild is already the expensive one here. The
    own-tile gate below restores exactly one bbox computation per unit, and
    keeps this dict's contents byte-identical to what own-tile bucketing
    produced. The gate is total because unit_tile_bounds() guarantees a
    unit's own tile is inside the bounds it was bucketed over.

    extra_top_px is passed straight through to _unit_screen_bbox_iso() --
    see that function for why only Sloped ever passes a nonzero value, and
    why it is a top-edge-only widening."""
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for (px, py), entries in units_by_tile.items():
        union: tuple[int, int, int, int] | None = None
        for unit, _color in entries:
            if (int(unit.x), int(unit.y)) != (px, py):
                # Reached through a footprint tile that isn't this unit's own;
                # it gets its bbox computed once, at its own tile's key.
                continue
            span_x, span_y = BUILDING_TILE_SPANS.get(unit.unit_const, NON_BUILDING_SPAN)
            if span_x <= 1 and span_y <= 1:
                # Single-tile objects only ever paint their own tile, so they
                # can't make a neighbour a bystander. Written as a span test
                # rather than a sentinel comparison so it stays true by
                # construction: the same 140 consts either way (clearance <= 0.5).
                continue
            bbox = _unit_screen_bbox_iso(unit, w, h, proj, elevations, extra_top_px)
            if bbox is None:
                continue
            if union is None:
                union = bbox
            else:
                union = (
                    min(union[0], bbox[0]),
                    min(union[1], bbox[1]),
                    max(union[2], bbox[2]),
                    max(union[3], bbox[3]),
                )
        if union is not None:
            out[(px, py)] = union
    return out


def _dirty_screen_bbox(
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    canvas_dims: tuple[int, int],
    with_units: bool = True,
    with_sprites: bool = False,
    sprite_band_radius: int = 0,
    unit_band_radius: int = 0,
    elevation_changed: set | None = None,
) -> tuple[int, int, int, int] | None:
    """Shared body of dirty_screen_bbox_iso()/dirty_screen_bbox_sloped() --
    see the former's docstring for the full contract, which is this
    function's contract too.

    canvas_dims is the ONE thing the two styles disagree about, which is why
    this is a parameter rather than a `_canvas_pixel_dims(proj)` call: the
    final clamp must match whatever canvas the caller's chunk cache actually
    holds, and Sloped's is the tighter of the two (see
    SlopedChunkCache.canvas_dims()) -- EXCEPT when with_sprites, where the
    Sloped caller passes the wider skirt-padded value instead (Track P3-g6's
    Step 0 fix: without it, a sprite reaching past Sloped's tight canvas gets
    its dirty bbox clamped away before the wider canvas ever gets composited
    into). Everything above the clamp is style-independent -- both styles
    project a tile the same way, and a sloped tile's extra corner headroom is
    already inside tile_screen_bounds_swept() via proj.corner_headroom_px.

    sprite_band_radius (Track P3-g6): how far the with_sprites band below
    dilates dirty_xy before seeding itself. 0 (the default) uses bare
    dirty_xy, exact for Stepped -- see that block's own comment for why only
    an edit to a sprite's OWN centre tile can move it there. Sloped's anchor
    instead reads its own tile's four CORNERS, each shared with up to four
    tiles, so an edit to any tile in a changed tile's 3x3 neighbourhood can
    move a Sloped sprite; its caller passes 1 instead (F2).

    unit_band_radius (draw-perf seed-dilation plan Step 3): the mirror-image
    parameter for the exact-footprint seed union below -- how far a changed
    tile's trigger reaches before testing which units' own tile it caught.
    0 (Stepped) is exact: a unit's footprint moves only when the edit lands
    on its own tile (_unit_iso_footprint draws it entirely at that tile's
    elevation). Sloped's dirty_screen_bbox_sloped() passes 1 regardless of
    with_sprites: unit_rise_px reads a unit's tile's four CORNERS, each
    shared with up to four tiles, so any tile in the changed tile's 3x3
    neighborhood can move a Sloped unit's footprint (fact 4) -- the same
    corner-sharing argument sprite_band_radius answers for sprites, just
    unconditional here rather than gated on with_sprites.

    Extracted rather than mirrored (Track C4 Step 3): with a single
    expression separating them, two near-identical 60-line bodies would be a
    drift hazard, not a safety margin -- the opposite of the
    testkit/qt_capture split, where the bodies genuinely disagreed.

    elevation_changed (draw-perf plan Step 3): if given, populated with the
    subset of dirty_xy whose elevation actually moved -- read here, at the
    one point that sees both the pre-edit value (still in `elevations`) and
    the post-edit one (mm.get_tile), before the loop below overwrites the
    array. A terrain-paint-only edit leaves this empty; callers use it to
    decide whether any elevation-dependent cache state needs a refresh at
    all, which the dirty set itself (terrain edits included) can't answer."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    dirty_xy = {(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices}
    if not dirty_xy:
        return None

    # Computed locally and unconditionally now (draw-perf seed-dilation plan
    # Step 3, fact 3): the seed union below needs this set itself, not just
    # whatever a caller wanted for its own cache-refresh decision. Aliased
    # to the caller's set when given, rather than copied, so this is still
    # the only bookkeeping either use needs.
    elevation_changed_local = set() if elevation_changed is None else elevation_changed
    for x, y in dirty_xy:
        if int(elevations[y, x]) != mm.get_tile(x, y).elevation:
            elevation_changed_local.add((x, y))

    for x, y in dirty_xy:
        elevations[y, x] = mm.get_tile(x, y).elevation

    if any(not (proj.min_elev <= int(elevations[y, x]) <= proj.max_elev) for x, y in dirty_xy):
        return None

    # Lateral expansion, floor of 1 always (draw-perf seed-dilation plan
    # Step 3): a tile's own skirt geometry samples its "left"/"right"
    # neighbor's elevation (see _render_tile_iso), so an edited tile can
    # change a *neighbor's* skirt even though the neighbor's own elevation
    # never changed -- +-1 is enough for that alone (see
    # iso_geometry.skirt_quad_indices' docstring for exactly which two of a
    # tile's four grid-neighbors can ever show a skirt facing it). This
    # alone is exact for any terrain-only edit: no unit moves, so no pixel
    # outside dilate(dirty, 1) changes value (fact 2) -- the footprint union
    # below is additive, not a replacement for this.
    seed = set(dirty_xy)
    for x, y in list(dirty_xy):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    seed.add((nx, ny))

    # Exact footprint union, only for triggered buildings (draw-perf
    # seed-dilation plan Step 3): a unit's footprint is drawn entirely at
    # its OWN tile's elevation (see _unit_iso_footprint), so it only moves
    # when an elevation edit lands on that tile -- unit_band_radius away at
    # most, not a blanket UNIT_FOOTPRINT_MAX_RADIUS dilation applied to
    # every dirty tile regardless of whether anything elevation-related
    # changed at all. Skipped entirely when elevation_changed_local is empty
    # (the terrain-paint-only case, which is most edits) -- no unit scan
    # runs, matching the with_sprites own-tile-scan block below.
    #
    # Dilates the (small) TRIGGER set, not a per-anchor ring scan over every
    # unit -- same inversion the with_sprites block below could take too
    # (its own comment flags the O(units) cost of the opposite direction on
    # an 11k-unit map).
    if with_units and elevation_changed_local:
        triggered = set(elevation_changed_local)
        for x, y in elevation_changed_local:
            for dx in range(-unit_band_radius, unit_band_radius + 1):
                for dy in range(-unit_band_radius, unit_band_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        triggered.add((nx, ny))
        for units in scenario.unit_manager.units:
            for u in units:
                if (int(u.x), int(u.y)) not in triggered:
                    continue
                bounds = unit_tile_bounds(u, w, h)
                if bounds is None:
                    continue
                fx0, fx1, fy0, fy1 = bounds
                for fx in range(fx0, fx1):
                    for fy in range(fy0, fy1):
                        seed.add((fx, fy))

    # Union screen bbox, each seed tile swept across the WHOLE legal
    # elevation range rather than just its current one -- this function has
    # no access to what a tile's elevation was *before* the edit already
    # applied to mm.terrain (mutated in place before this is called, same
    # contract refresh_tiles() has), so only a bound this generous
    # guarantees the tile's old, now-stale screen footprint (wherever it
    # actually was) falls inside the bbox and gets erased below. Cheap:
    # this is a handful of bbox corners, not a per-pixel cost. Since a
    # unit's footprint diamonds use the SAME per-tile sweep formula (just at
    # the center tile's elevation, always inside [min_elev, max_elev]), this
    # bbox is automatically big enough for any footprint centered on a seed
    # tile too -- no separate unit-specific bbox term needed here.
    x0 = y0 = x1 = y1 = None
    for x, y in seed:
        tile_x0, tile_y0, tile_x1, tile_y1 = iso_geometry.tile_screen_bounds_swept(x, y, proj)
        x0 = tile_x0 if x0 is None else min(x0, tile_x0)
        x1 = tile_x1 if x1 is None else max(x1, tile_x1)
        y0 = tile_y0 if y0 is None else min(y0, tile_y0)
        y1 = tile_y1 if y1 is None else max(y1, tile_y1)

    if with_sprites:
        # Three things a reader would otherwise have to re-derive:
        #
        # DIRTY, not seed. sprite_draws_by_anchor computes a sprite's anchor x
        # with no elevation term at all, and its anchor y from elevations[
        # unit.y, unit.x] -- the unit's OWN centre tile. So only an edit to
        # that tile can MOVE a sprite; a terrain tile repainted underneath one,
        # or occlusion revealed by a neighbour, is already covered by
        # composite_rect_iso's bystander set via merge_sprite_bboxes. Widening
        # per dirty tile rather than per dilated seed tile is exact here, not
        # an optimisation.
        #
        # The elevation sweep, for the same reason the seed union above sweeps:
        # this function cannot see what the tile's elevation was BEFORE the
        # edit, so the sprite's own stale position is only guaranteed inside
        # the bbox if every legal anchor height is covered.
        #
        # The half_w/half_h slack, because the anchor is only pinned to its
        # tile's diamond to within half a tile. _span_start has a half-tile
        # branch, so an even-span building sits on either parity (46 of 158
        # Mills in the example corpus do, and a gate is span (4, 1), mixed
        # within one unit) -- putting ax anywhere in [tile_x0, tile_x0 +
        # 2*half_w] and ay anywhere in [tile_origin_y, tile_origin_y +
        # 2*half_h]. BOTH bands are the tile's own diamond bounding box, and
        # they must stay that way: tile_screen_origin returns that box's
        # TOP-LEFT, so the y band runs from it, not symmetrically about it.
        #
        # The y band read [tile_origin_y -+ half_h] until 2026-08-24, matching
        # sprite_draws_by_anchor's own half-tile-high anchor. Both were wrong
        # together, so the suite stayed green while every sprite rendered
        # floating; fixing the anchor alone then under-covered the BOTTOM edge
        # by up to half_h, which is the stale-fragment bug this widening
        # exists to prevent. The two must move together.
        pad_l, pad_u, pad_r, pad_d = _sprite_reach_px(proj)
        elev_span = (proj.max_elev - proj.min_elev) * proj.elev_step

        # draw-perf plan Step 4: pad only tiles that can actually MOVE a
        # sprite -- a plain terrain tile carrying no unit never can (see the
        # comment above: only a sprite's own anchor tile does). Direct
        # own-tile scan over scenario.unit_manager.units, not
        # _units_by_tile()'s footprint bucketing -- that buckets a unit into
        # EVERY footprint tile, which would over-widen a multi-tile
        # building's other tiles here and obscure why. Ignores unit_filter
        # (this function has no access to it): a filtered-out unit's tile
        # padding a bit further than strictly needed is safe over-inclusion,
        # the same direction every other widening in this function already
        # accepts, never a missed one. Not the resolved SpriteLayer either --
        # at bbox time it still holds the PRE-edit anchor set, and this
        # function runs BEFORE the cache's own post-edit rebuild.
        anchor_tiles = {(int(u.x), int(u.y)) for units in scenario.unit_manager.units for u in units}
        # band_tiles is the set of ANCHOR tiles to pad around, not dirty
        # tiles: the padding below must be centered on where the sprite
        # actually sits (an anchor's own tile_screen_origin), never on
        # whichever dirty tile happened to trigger it -- those can be
        # different tiles under Sloped's ring (radius=1). An anchor is
        # triggered when some dirty tile falls within its own radius-ring
        # (Stepped, radius=0: only the anchor tile itself; Sloped, radius=1:
        # its 3x3 neighbourhood, since a Sloped anchor reads its tile's four
        # shared corners).
        if sprite_band_radius:
            band_tiles = set()
            for ax, ay in anchor_tiles:
                for dx in range(-sprite_band_radius, sprite_band_radius + 1):
                    for dy in range(-sprite_band_radius, sprite_band_radius + 1):
                        if (ax + dx, ay + dy) in dirty_xy:
                            band_tiles.add((ax, ay))
                            break
                    else:
                        continue
                    break
        else:
            band_tiles = dirty_xy & anchor_tiles
        # Unit MOVES (not applicable here): this function only ever sees
        # terrain/elevation edits (dirty_indices comes from the terrain
        # array), and neither can move a unit -- so there is no "old anchor
        # tile" case to union in. A future unit-move caller of this function
        # would need to add one; none exists today.
        for x, y in band_tiles:
            tx0, ty_hi = iso_geometry.tile_screen_origin(x, y, proj.max_elev, proj)
            x0 = min(x0, tx0 - pad_l)
            x1 = max(x1, tx0 + 2 * proj.half_w + pad_r)
            # y1's 2*half_h is REQUIRED (the band's far end). y0 keeps an extra
            # half_h beyond the band's near end deliberately, as slack rather
            # than a derived bound: the exact top is ty_hi, but this is an
            # under-repaint guard, and the only direction a too-tight bound
            # fails in is stale pixels. 16px of extra repaint per edit is not
            # worth being clever about.
            y0 = min(y0, ty_hi - proj.half_h - pad_u)
            y1 = max(y1, ty_hi + elev_span + 2 * proj.half_h + pad_d)

    canvas_w, canvas_h = canvas_dims
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def dirty_screen_bbox_iso(
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    with_units: bool = True,
    with_sprites: bool = False,
    elevation_changed: set | None = None,
) -> tuple[int, int, int, int] | None:
    """The (x0, y0, x1, y1) canvas-pixel bbox a just-applied edit could have
    invalidated -- refresh_region_iso()'s original "half 1" (dirty tiles ->
    screen bbox), split out in Phase B-B so the chunk cache can ask this
    question without an img array to write into. dirty_indices' tiles must
    already carry their new (post-edit) terrain_id/elevation/layer -- same
    contract refresh_tiles() and
    refresh_region_iso() itself have (edit_history.py mutates tiles before
    any of these are called).

    elevations is mutated in place to match, deliberately the SAME array
    object the caller's MapView._iso_elevations already holds (never a
    copy) -- that's what keeps hit-testing (screen_to_tile) from drifting
    out of sync with what's actually redrawn, per the parent plan's Risk
    #6, with no separate "now go update the snapshot" step for the caller
    to forget. This is the ONLY place in the Track B pipeline that ever
    writes to elevations -- composite_rect_iso() below is deliberately
    read-only, so a caller must always run this (or refresh_region_iso(),
    which now just wraps it) before compositing anything downstream of an
    edit.

    Returns None if there's nothing to invalidate (empty dirty_indices) or
    if an edited tile's new elevation has landed outside [proj.min_elev,
    proj.max_elev] -- which, now that render_terrain_iso_with_proj() sizes
    proj to the fixed legal range (see iso_geometry.MIN_ELEVATION/
    MAX_ELEVATION), should never actually happen given viewer.py's own
    tools clamp to that same range -- but this function doesn't assume its
    caller enforced that: proj was sized for a specific range, and a tile
    landing outside it would compute a negative or overlarge screen
    position that silently corrupts unrelated pixels via numpy fancy-index
    wraparound rather than erroring, which is a far worse failure mode than
    just telling the caller to fall back to a full re-render.

    elevation_changed (draw-perf plan Step 3): optional out-param, populated
    in place with the subset of edited tiles whose elevation actually moved
    -- see _dirty_screen_bbox()'s own docstring for why this is the only
    point that can answer that question. None (the default) skips the
    bookkeeping; only IsoChunkCache.patch()'s caller needs it."""
    # with_sprites is a PARAMETER as of P3-g's toggle, not a module global read
    # at call time: sprites are per-cache now, so this function cannot look the
    # answer up itself. The caller's obligation is therefore load-bearing --
    # this must be passed the same value IsoChunkCache._level is deciding on,
    # i.e. that cache's own sprites_enabled. If the two ever disagree about
    # whether sprites are on, the under-repaint bug this widening exists to fix
    # comes straight back. ViewerWindow._apply_dirty is the only production
    # caller that renders through an IsoChunkCache, and
    # tests/test_sprite_toggle_viewer.py pins that it passes the cache's flag.
    # Defaults to False so the legacy/tool callers that never enable sprites
    # keep their exact behavior. The `with_units and` conjunction matters too:
    # composite_rect_iso already nulls the sprite layer when units are off, so
    # there is nothing to widen for.
    return _dirty_screen_bbox(
        scenario, dirty_indices, elevations, proj, _canvas_pixel_dims(proj), with_units,
        with_sprites=with_units and with_sprites, elevation_changed=elevation_changed,
    )


def dirty_screen_bbox_sloped(
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    with_units: bool = True,
    with_sprites: bool = False,
    elevation_changed: set | None = None,
) -> tuple[int, int, int, int] | None:
    """Sloped's counterpart to dirty_screen_bbox_iso() -- Track C4's Step 3.
    Identical contract, including the in-place elevations mutation (Risk #6:
    the snapshot the pick plane is rasterized against and the pixels that
    get redrawn must never drift apart, with no separate "now update the
    snapshot" step for a caller to forget). Clamped to the SlopedChunkCache's
    own tighter canvas rather than Stepped's skirt-padded one -- see
    _dirty_screen_bbox()'s canvas_dims parameter -- UNLESS with_sprites, see
    below.

    corner_rise is deliberately NOT rebuilt here, even though every edit
    invalidates it: it is derived state owned by the cache, and
    SlopedChunkCache._refresh_source_caches() rebuilds it as patch()'s first
    action, reading the elevations this function has just mutated. Splitting
    it that way keeps one array with one owner instead of handing the caller
    a second "and now also refresh this" obligation -- exactly the trap the
    in-place elevations contract exists to avoid.

    The floor-1 dilation _dirty_screen_bbox() always applies is exactly the
    ONE-TILE RING Sloped needs around the union of every changed tile: an
    edit moves that tile's four corner values, each corner is shared with up
    to four tiles (see iso_geometry.corner_rise_px), so every tile in the
    changed tile's 3x3 neighbourhood is reshaped. Note the ring is around
    each CHANGED tile, not around a click point -- one elevation click
    propagates through MapManager._elevation_tile_recursion to many tiles,
    and dirty_indices is what carries them.

    unit_band_radius=1 is passed regardless of with_sprites (draw-perf
    seed-dilation plan Step 3, fact 4) -- unlike sprite_band_radius above,
    which is gated on with_sprites because sprites are the only thing it
    affects. unit_rise_px reads a unit's own tile's four corners, the same
    sharing the ring above exists for, so a Sloped unit's footprint can move
    from an edit up to one tile away from its own tile whenever units are on
    at all -- see _dirty_screen_bbox's own unit_band_radius docstring.

    with_sprites (Track P3-g6): must be the live cache's own sprites_enabled,
    same load-bearing-argument warning dirty_screen_bbox_iso's own docstring
    carries -- ViewerWindow._apply_dirty is pinned to pass it by
    tests/test_sprite_toggle_viewer.py. Two things change when it's set:

    - the sprite band's SEED (F2): a Sloped anchor reads its own tile's four
      corners rather than its centre tile alone (see
      _dirty_screen_bbox's own with_sprites comment for why DIRTY, not
      SEED, is exact for Stepped), so every tile in the one-tile ring around
      each dirty tile can move a sprite, not just the dirty tile itself.
    - the canvas clamp widens to _canvas_pixel_dims(proj) (the Step 0 fix):
      SlopedChunkCache.canvas_dims() reports that same wider bound once
      sprites are on, and this function's own final clamp must agree, or a
      sprite reaching into the newly-composited strip never gets a dirty
      bbox wide enough to reach it.

    elevation_changed: same optional out-param as dirty_screen_bbox_iso()'s
    own -- only SlopedChunkCache.patch()'s caller needs it."""
    canvas_dims = _canvas_pixel_dims(proj) if with_sprites else (proj.canvas_w, proj.canvas_h)
    return _dirty_screen_bbox(
        scenario, dirty_indices, elevations, proj, canvas_dims, with_units,
        with_sprites=with_sprites, sprite_band_radius=1 if with_sprites else 0,
        unit_band_radius=1, elevation_changed=elevation_changed,
    )


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

    Never mutates elevations -- only dirty_screen_bbox_iso() does that (see
    its own docstring for why that separation matters for chunk-order
    independence)."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)

    if with_units and building_bboxes:
        seen = {(int(cx), int(cy)) for cx, cy in candidates}
        bystanders = [
            (px, py)
            for (px, py), (ux0, uy0, ux1, uy1) in building_bboxes.items()
            if (px, py) not in seen and ux0 < x1 and ux1 > x0 and uy0 < y1 and uy1 > y0
        ]
        if bystanders:
            extra = np.array(bystanders, dtype=np.int64)
            combined = np.concatenate([candidates, extra], axis=0)
            xs, ys = combined[:, 0], combined[:, 1]
            order = np.lexsort((xs, ys - xs))  # primary key is the LAST arg: d=y-x, then x -- matches depth_order()
            candidates = combined[order]

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
    # [min_elev, max_elev]), and compositing that superset in depth_order --
    # terrain then that tile's own units, via the SAME
    # _paint_tile_and_units_iso() step the full render uses -- reproduces
    # exactly what a full render's own proven-gap-free tiling (Phase 2's
    # check_full_coverage) would have painted there, units included.
    scratch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sprite_layer = sprites if with_units else None
    for cx, cy in candidates:
        tile = mm.get_tile(int(cx), int(cy))
        _paint_tile_and_units_iso(
            scratch, tile, units_by_tile, tile_px, proj, elevations, w, h,
            offset=(x0, y0), sprites=sprite_layer,
        )
    return scratch


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
# depth cue (docs/PLAN_V2_6.md's Track C decisions), NOT the
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


def _slope_shade(tile_px: int, nw: int, ne: int, sw: int, se: int, elev_step: int) -> np.ndarray:
    """Per-pixel shading factor over one tile's DIAMOND footprint, in
    tile_uv_fractions(tile_px) order (equivalently diamond_indices' own) --
    multiply against sampled texture color, same "float32 factor, truncate
    back to uint8" contract render.py's other shading (_shadow_factors)
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
    back in)."""
    texture = asset_source.get_terrain_texture_array(tile.terrain_id)
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        top_block = texture[oy : oy + tile_px, ox : ox + tile_px]
    else:
        r, g, b = color_for_terrain_id(tile.terrain_id)
        top_block = np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.uint8)

    d_nw = int(corner_rise[tile.y, tile.x])
    d_ne = int(corner_rise[tile.y, tile.x + 1])
    d_sw = int(corner_rise[tile.y + 1, tile.x])
    d_se = int(corner_rise[tile.y + 1, tile.x + 1])
    d_min = min(d_nw, d_ne, d_sw, d_se)

    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, 0, proj)
    base_x -= off_x
    base_y -= off_y + d_min

    dst_y, dst_x, src_y, src_x, uv_idx = iso_geometry.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
    top = top_block[src_y, src_x]
    shade = _slope_shade(tile_px, d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min, proj.elev_step)
    shaded = np.clip(top.astype(np.float32) * shade[uv_idx][:, None], 0, 255).astype(np.uint8)
    _clipped_paint(img, base_y, base_x, dst_y, dst_x, shaded)


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
    values = np.full(dst_y.shape[0], tile.y * map_w + tile.x, dtype=np.int32)
    _clipped_paint(plane, base_y, base_x, dst_y, dst_x, values)


def _draw_unit_sloped(
    img: np.ndarray,
    unit,
    color: tuple[int, int, int],
    tx: int,
    ty: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    rise_px: int,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Sloped's counterpart to _draw_unit_iso -- same one-diamond-per-call
    paint (see that function for why a multi-tile unit arrives here once per
    footprint tile), but takes a precomputed SCALAR PIXEL RISE (already
    derived from the UNIT'S OWN tile's 4 corners and its own sub-tile
    position by _paint_tile_and_units_sloped, see that function) instead of
    a whole-map elevations array to index. Sloped has no such array
    (corner_rise_px replaces it) -- allocating a throwaway full-size one
    just to satisfy _draw_unit_iso's existing signature would cost
    O(map_h * map_w) per UNIT, not per render, so this takes the scalar
    directly.

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
    _paint_tile_and_units_sloped()'s skip_ids gate. This function only ever
    draws the plain coloured mark: for a unit with no resolved sprite, and
    always for a farm (Sloped's sprite_draws_by_anchor() call passes
    with_farms=False, so no farm ever enters skip_ids here)."""
    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    values = np.full((dst_y.shape[0], 3), color, dtype=np.uint8)
    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, 0, proj)
    _clipped_paint(img, base_y - off_y - rise_px, base_x - off_x, dst_y, dst_x, values)


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
) -> None:
    """Sloped's counterpart to _paint_tile_and_units_iso -- same
    "terrain then this tile's own units, interleaved in depth_order" step
    (see that function's own docstring for why interleaving is load-
    bearing, not a style choice).

    Units are drawn via _draw_unit_sloped() at the PIXEL RISE
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
    tile-by-tile -- silently undoing the flat pad described above. The own
    tile's corners are always in bounds: unit_tile_bounds() only admitted
    this unit because its own tile is on-map, and corner_rise is
    (map_h + 1, map_w + 1).

    map_w/map_h are unused since _draw_unit_sloped() stopped re-deriving
    bounds; they stay only to keep this signature parallel with
    _paint_tile_and_units_iso()'s, which the two paths are deliberately
    written to mirror. Not load-bearing.

    sprites (Track P3-g6): a unit in sprites.skip_ids draws as a real sprite
    at its anchor tile instead of a mark here -- same skip/blit shape
    _paint_tile_and_units_iso() uses, minus the farm-terrain-override case
    (sprite_draws_by_anchor() is always called with with_farms=False for
    Sloped, so sprites.farm_by_tile is always empty here; farms keep their
    plain mark unconditionally, see this repo's plan for why a paint-time
    skip alone would make them invisible instead of deferred)."""
    _render_tile_sloped(img, tile, tile_px, proj, corner_rise, offset=offset)
    skip = sprites.skip_ids if sprites is not None else frozenset()
    for unit, color in units_by_tile.get((tile.x, tile.y), ()):
        if id(unit) in skip:
            continue  # its sprite paints instead, once, at its anchor tile
        ux, uy = int(unit.x), int(unit.y)
        rise_px = iso_geometry.unit_rise_px(corner_rise, ux, uy, unit.x - ux, unit.y - uy)
        _draw_unit_sloped(img, unit, color, tile.x, tile.y, tile_px, proj, rise_px, offset=offset)

    # Sprites last within this tile's step, mirroring _paint_tile_and_units_iso's
    # own ordering rationale: a unit standing on this tile must not be cut by
    # its own tile's terrain.
    if sprites is not None:
        off_x, off_y = offset
        for draw, ax, ay in sprites.by_anchor.get((tile.x, tile.y), ()):
            _clipped_paint_rgba(
                img, ay - draw.hotspot_y - off_y, ax - draw.hotspot_x - off_x, draw.rgba
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
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    elevations = np.zeros((h, w), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
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
    scenario: LoadedScenario, with_units: bool = True, with_sprites: bool = False
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
    with_farms=False always, matching SlopedChunkCache's own call -- see
    sprite_draws_by_anchor()'s docstring for why."""
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

    skirt_headroom = (max_elev - min_elev) * proj.elev_step
    img = np.zeros((proj.canvas_h + skirt_headroom, proj.canvas_w, 3), dtype=np.uint8)

    units_by_tile = _units_by_tile(scenario) if with_units else {}
    sprites = (
        sprite_draws_by_anchor(scenario, proj, elevations, corner_rise=corner_rise, with_farms=False)
        if (with_units and with_sprites)
        else None
    )
    for x, y in iso_geometry.depth_order(w, h):
        _paint_tile_and_units_sloped(
            img, tile_grid[y][x], units_by_tile, tile_px, proj, corner_rise, w, h, sprites=sprites
        )
    return img, elevations, corner_rise, proj


def render_terrain_sloped(scenario: LoadedScenario, with_units: bool = True) -> np.ndarray:
    """Sloped mode's top-level renderer -- the Phase 6 counterpart to
    render_terrain_iso(). Thin wrapper around
    render_terrain_sloped_with_proj() for callers that only need the
    pixels (render_scenario(), tools/dump_scenario.py,
    tools/gen_elevation_reference.py's own reference renders)."""
    img, _elevations, _corner_rise, _proj = render_terrain_sloped_with_proj(scenario, with_units=with_units)
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
    consulted once units are toggled off."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)

    if with_units and building_bboxes:
        seen = {(int(cx), int(cy)) for cx, cy in candidates}
        bystanders = [
            (px, py)
            for (px, py), (ux0, uy0, ux1, uy1) in building_bboxes.items()
            if (px, py) not in seen and ux0 < x1 and ux1 > x0 and uy0 < y1 and uy1 > y0
        ]
        if bystanders:
            extra = np.array(bystanders, dtype=np.int64)
            combined = np.concatenate([candidates, extra], axis=0)
            xs, ys = combined[:, 0], combined[:, 1]
            order = np.lexsort((xs, ys - xs))
            candidates = combined[order]

    scratch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sprite_layer = sprites if with_units else None
    for cx, cy in candidates:
        tile = mm.get_tile(int(cx), int(cy))
        _paint_tile_and_units_sloped(
            scratch, tile, units_by_tile, tile_px, proj, corner_rise, w, h,
            offset=(x0, y0), sprites=sprite_layer,
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
    for cx, cy in candidates:
        tile = mm.get_tile(int(cx), int(cy))
        _render_tile_sloped_ids(plane, tile, tile_px, proj, corner_rise, w, offset=(x0, y0))
    return plane


def _blit_clipped(dst: np.ndarray, block: np.ndarray, x: int, y: int) -> None:
    """Blits block into dst at offset (x, y) -- both pixel-space, dst's own
    origin -- clipping to whichever of block's four edges fall outside
    dst's own bounds. composite_rect_flat()'s chunk-sized rects aren't
    always tile_px-aligned (Phase B-B's patch() intersects an edit's dirty
    bbox with each cached chunk's own bounds, not the tile grid), so a
    tile's tile_px x tile_px block can straddle the scratch canvas's edge
    on any side; a plain slice-assign would raise or silently misplace
    pixels there."""
    h, w = block.shape[:2]
    dh, dw = dst.shape[:2]
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(dw, x + w), min(dh, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    dst[dy0:dy1, dx0:dx1] = block[sy0:sy1, sx0:sx1]


def _flat_unit_draws(
    scenario: LoadedScenario, tile_px: int, unit_filter: UnitFilter = UnitFilter()
) -> tuple[np.ndarray, np.ndarray]:
    """Every unit's pixel-space draw, as (N,4) int32 half-open bboxes
    (x0,y0,x1,y1) plus (N,3) uint8 colors -- Phase B-E's precomputed input
    to composite_rect_flat(), built in EXACTLY overlay_units()'s own
    per-player/per-unit-list order. Off-map units (unit_tile_bounds()
    returns None) are dropped: an off-map unit paints nothing anywhere in
    overlay_units() either, so dropping it here can't change any pixel any
    caller of composite_rect_flat() will ever see -- it's not a
    dirty-tile-style subsetting the way refresh_units_over()'s fixed-point
    expansion has to worry about (see composite_rect_flat()'s own
    docstring for why that distinction is the whole correctness argument
    here).

    Like _units_by_tile()/_building_bboxes_iso() for the iso cache, this is
    REQUIRED to be precomputed by the caller (once per FlatChunkCache
    construction) rather than recomputed per chunk -- walking every unit
    (~11k on this project's bigger real files) would dominate a single
    chunk's cost otherwise."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    bboxes = []
    colors = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tile_x0, tile_x1, tile_y0, tile_y1 = bounds
            bboxes.append((tile_x0 * tile_px, tile_y0 * tile_px, tile_x1 * tile_px, tile_y1 * tile_px))
            colors.append(_unit_color(unit, player_color))
    if not bboxes:
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)
    return np.array(bboxes, dtype=np.int32), np.array(colors, dtype=np.uint8)


def composite_rect_flat(
    scenario: LoadedScenario,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tile_px: int,
    unit_draws: tuple[np.ndarray, np.ndarray] | None = None,
    with_units: bool = True,
) -> np.ndarray:
    """Composites the half-open canvas rect [x0, x1) x [y0, y1) for Flat
    mode in isolation and returns it as a fresh (y1-y0, x1-x0, 3) uint8
    array -- Phase B-E's rect-keyed core, the Flat counterpart to
    composite_rect_iso() (see that function's docstring for the shared
    "same function composites every caller's rect, so pixels can never
    depend on request order" property this one inherits).

    Correctness argument (why this needs none of refresh_units_over()'s
    fixed-point dirty-tile expansion): overlay_units() is a sequence of
    opaque axis-aligned rect overwrites onto a fully-painted terrain
    canvas, so restricting the OUTPUT RECT commutes with that draw
    sequence -- a unit whose bbox misses [x0,x1)x[y0,y1) contributes zero
    pixels inside it, so omitting it from the sum below cannot change the
    result at any pixel that IS in range, regardless of where that unit
    sits in the draw order. refresh_units_over()'s fixed-point expansion
    (and the "building erases a tree" bug its own docstring records) is
    the fix for a *units-subsetting* problem -- redrawing a chosen subset
    of units onto an ALREADY-PAINTED persistent canvas, where a skipped
    building can silently leave a stale tree visible underneath it. This
    function has no such problem: every call composites a fresh scratch
    from nothing, terrain first, so there's no stale prior paint for a
    dropped unit to ever leave behind. Units below are therefore filtered
    ONLY by geometric intersection with the rect (np.nonzero on the
    precomputed bbox array, which returns ascending indices -- i.e. still
    exactly overlay_units()' own relative order), never by tile
    membership, and with no dilation step.

    unit_draws, if given, must be _flat_unit_draws(scenario, tile_px)'s own
    return value, precomputed once by the caller (FlatChunkCache) and
    reused across many calls -- see that function's own docstring for why.

    Do NOT re-express render_terrain()/overlay_units() in terms of this
    function -- same reason Track B gives for composite_rect_iso(): the
    full render is the independent byte-identity
    ground truth tools/verify_iso_chunks.py-style checks compare against,
    and sharing one implementation would make that comparison a
    tautology, destroying the only real seam check either path has."""
    out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)

    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    tx0, ty0 = max(x0 // tile_px, 0), max(y0 // tile_px, 0)
    tx1 = min(-(-x1 // tile_px), tile_w)  # ceil division, clipped to map bounds
    ty1 = min(-(-y1 // tile_px), tile_h)
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            tile = mm.get_tile(tx, ty)
            _blit_clipped(out, _tile_block(tile, tile_px), tx * tile_px - x0, ty * tile_px - y0)

    if with_units and unit_draws is not None:
        bboxes, colors = unit_draws
        if len(bboxes):
            hits = np.nonzero(
                (bboxes[:, 0] < x1) & (bboxes[:, 2] > x0) & (bboxes[:, 1] < y1) & (bboxes[:, 3] > y0)
            )[0]
            for i in hits:
                bx0, by0, bx1, by1 = (int(v) for v in bboxes[i])
                ix0, iy0 = max(bx0, x0) - x0, max(by0, y0) - y0
                ix1, iy1 = min(bx1, x1) - x0, min(by1, y1) - y0
                out[iy0:iy1, ix0:ix1] = colors[i]
    return out


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
    building_bboxes = _building_bboxes_iso(units_by_tile, w, h, proj, elevations) if with_units else {}

    scratch = composite_rect_iso(
        scenario, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, building_bboxes, with_units
    )
    img[y0:y1, x0:x1] = scratch
    return bbox


def _span_start(coord: float, span: int) -> int:
    """Lowest tile of a `span`-wide footprint anchored on `coord`.

    Two branches, because only one of them has measured data behind it.

    A span-1 axis anchors at int(coord) -- the unit's own tile, by definition,
    and what every unit did before spans existed. This branch carries all
    non-buildings, whose coordinates are arbitrary floats: 291,148 axis values
    across the example corpus have 15,370 distinct fractional parts.

    A span > 1 axis works in half-tiles, which is exact because span > 1
    implies a building, and building coordinates are always exact multiples of
    0.5 (measured: 27,320 axis values, fractional part 0.0 or 0.5, nothing
    else). Rounding half up rather than flooring is what keeps the own-tile
    invariant on legal off-parity placements.

    Per axis, not per unit: a gate segment is (4, 1) and takes both branches.
    Applying the half-tile branch to a span-1 axis would move any unit whose
    fractional part exceeds 0.75 a whole tile (5,671 real corpus values do).
    """
    if span <= 1:
        return int(coord)
    return (round(coord * 2) - span + 1) // 2


def unit_tile_bounds(unit, tile_w: int, tile_h: int) -> tuple[int, int, int, int] | None:
    """(tile_x0, tile_x1, tile_y0, tile_y1) -- the tile-space bounding box
    _draw_unit() would paint for this unit, half-open like Python ranges.
    None if the unit is off-map. Split out from _draw_unit() so
    refresh_units_over() can cheaply test overlap with a dirty-tile set
    without touching img.

    **Size comes from clearance alone, never from position.** The engine and
    the in-game editor both allow a building to sit on the coordinate parity
    its size does not "expect" (46 of 158 Mills and 32 of 254 Castles in the
    example corpus do), so deriving the span from where a unit sits would
    render every off-grid House 3x3 -- exactly the bug this replaced. Position
    only picks the anchor; an off-grid building is drawn at its true size,
    quantized to the nearer tile.

    **Invariant, relied on downstream: (int(unit.x), int(unit.y)) is always
    inside the returned bounds.** _unit_iso_footprint reads elevations[py, px]
    for the whole slab, unit_pick's stepped gate keys on own_x/own_y, and
    _unit_screen_bbox_iso indexes tile_x1 - 1, which is only safe while the
    clamped range is non-empty. Don't change the anchor rule without
    re-checking all three."""
    px, py = int(unit.x), int(unit.y)
    if not (0 <= px < tile_w and 0 <= py < tile_h):
        return None
    span_x, span_y = BUILDING_TILE_SPANS.get(unit.unit_const, NON_BUILDING_SPAN)
    x0, y0 = _span_start(unit.x, span_x), _span_start(unit.y, span_y)
    tile_x0, tile_x1 = max(0, x0), min(tile_w, x0 + span_x)
    tile_y0, tile_y1 = max(0, y0), min(tile_h, y0 + span_y)
    return tile_x0, tile_x1, tile_y0, tile_y1


def _unit_color(unit, player_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """The color rule shared by both Flat's _draw_unit() and Stepped's
    _draw_unit_iso() (Phase 5 reuses this rather than re-deriving it):
    trees are always dark green, non-building resource/decoration objects
    get their real minimap color, everything else (including buildings) is
    colored by owning player -- all regardless of owner, matching AoE2's own
    minimap for the first two categories."""
    is_building = unit.unit_const in BUILDING_TILE_SPANS
    if unit.unit_const in TREE_UNIT_IDS:
        return TREE_COLOR
    if not is_building and unit.unit_const in RESOURCE_COLORS:
        return RESOURCE_COLORS[unit.unit_const]
    return player_color


def _draw_unit(img: np.ndarray, unit, player_color, tile_w: int, tile_h: int, tile_px: int) -> None:
    """Draws one unit's mark into img, in place -- a colored dot, sized to
    its real footprint and colored per _unit_color(), a 1-tile dot for
    everything else. Shared by overlay_units() (every unit, full map) and
    refresh_units_over() (only units overlapping a dirty-tile set, after an
    edit) so the two paths can never draw a unit differently."""
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    color = _unit_color(unit, player_color)

    # Tile-space bounds scaled to pixel space -- render_terrain() renders at
    # tile_px pixels per tile, not one pixel per tile. tile_px here must
    # match whatever render_terrain() used for this same scenario -- true as
    # long as both derive it via tile_pixels_for_map() rather than a
    # passed-in value, since it's a pure function of the map's own
    # (unchanging) dimensions.
    y0, y1 = tile_y0 * tile_px, tile_y1 * tile_px
    x0, x1 = tile_x0 * tile_px, tile_x1 * tile_px
    img[y0:y1, x0:x1] = color


def _unit_iso_footprint(
    unit, tile_w: int, tile_h: int, elevations: np.ndarray
) -> tuple[int, int, int, int, int] | None:
    """(tile_x0, tile_x1, tile_y0, tile_y1, elevation) for one unit's iso
    footprint -- the same tile-space bounds unit_tile_bounds() computes for
    Flat mode, plus the single elevation every footprint tile is drawn at:
    elevations[py, px], the
    unit's OWN floored tile's elevation -- not each footprint tile's own
    individual terrain elevation. That's Phase 5's whole plan for
    multi-tile buildings ("center tile's elevation for the whole
    footprint") -- a flat slab, deliberately not per-corner-interpolated
    (real sloped ramps are Phase 6, out of scope here). None if the unit is
    off-map.

    elevations[py, px] is safe because unit_tile_bounds() guarantees the
    unit's own tile lies inside the bounds it returns -- see its docstring.
    An even-span building has no centre tile at all, so "its own tile" is the
    right way to say this, not "its center"."""
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    px, py = int(unit.x), int(unit.y)
    return bounds + (int(elevations[py, px]),)


def _draw_unit_iso(
    img: np.ndarray,
    unit,
    color: tuple[int, int, int],
    tx: int,
    ty: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Stepped mode's counterpart to _draw_unit() (Phase 5): paints ONE of a
    unit's footprint diamonds -- the one covering tile (tx, ty) -- filled,
    fitting naturally into the same diamond shape terrain tiles paint in
    rather than the scaled-up rectangle Flat mode's _draw_unit() uses. A
    multi-tile building reaches this function once per footprint tile,
    because _units_by_tile() buckets it into every one of them, so each
    diamond lands at its own tile's moment in the depth walk instead of the
    whole slab landing at the unit's own tile's moment (which is what let
    later terrain tiles paint back over half of it).

    The diamond sits at the unit's OWN tile's elevation, not (tx, ty)'s, so
    a multi-tile building still renders as one flat slab at a single height
    -- see _unit_iso_footprint and unit_pick's "asymmetry 1". Reading that
    elevation directly rather than via _unit_iso_footprint() means this
    function has no off-map guard of its own; _units_by_tile() has already
    dropped anything unit_tile_bounds() rejects, which is why the index
    below is safe. offset/clipping mirror _render_tile_iso()'s own
    scratch-canvas contract, for refresh_region_iso()'s incremental redraw."""
    elevation = int(elevations[int(unit.y), int(unit.x)])
    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    values = np.full((dst_y.shape[0], 3), color, dtype=np.uint8)
    off_x, off_y = offset
    base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, elevation, proj)
    _clipped_paint(img, base_y - off_y, base_x - off_x, dst_y, dst_x, values)


def _unit_screen_bbox_iso(
    unit,
    tile_w: int,
    tile_h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """(sx0, sy0, sx1, sy1) canvas-pixel bbox a Stepped-mode unit's footprint
    would occupy -- the closed-form counterpart to actually drawing it
    (_draw_unit_iso() must still visit every footprint tile to paint its
    diamond, since each one's screen position differs, but the bbox extremes
    don't need that loop): screen_x = origin_x + (x+y)*half_w is monotonic
    increasing in both x and y independently, and screen_y = origin_y +
    (y-x)*half_h - elevation*elev_step is monotonic increasing in y and
    decreasing in x, with the SAME elevation for every footprint tile (see
    _unit_iso_footprint) -- so the bbox's four extremes are exactly at the
    footprint tile-range's own corners. Used by refresh_region_iso() to
    cheaply test a unit's overlap against a candidate region without an
    O(footprint size) scan, even across this project's ~11,000-unit real
    files. None if the unit is off-map.

    extra_top_px raises the TOP edge only (Track C5's Step 2), and is 0 for
    every Stepped caller so their bboxes stay byte-identical. Sloped needs
    it because the sentence above -- "a building's screen bbox only ever
    depends on proj + elevations, never on corner_rise", the grounds on
    which SlopedChunkCache reuses this function verbatim -- stops being
    true once a unit sits at unit_rise_px() rather than at
    elevation * elev_step. Under SLOPE_CORNER_RULE = "max" a corner is
    never BELOW its own tile's elevation and can be above it, so the
    interpolated rise only ever moves a unit UP the screen: sy1 is
    unaffected, and only sy0 needs the headroom. Under-covering here is a
    MISSED bystander paint (a stale pixel), not a no-op, so the caller
    passes a proven bound rather than a guess -- see
    SlopedChunkCache._refresh_source_caches()."""
    footprint = _unit_iso_footprint(unit, tile_w, tile_h, elevations)
    if footprint is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1, elevation = footprint
    half_w, half_h = proj.half_w, proj.half_h
    sx0, _ = iso_geometry.tile_screen_origin(tile_x0, tile_y0, elevation, proj)
    sx1, _ = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y1 - 1, elevation, proj)
    _, sy0 = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y0, elevation, proj)
    _, sy1 = iso_geometry.tile_screen_origin(tile_x0, tile_y1 - 1, elevation, proj)
    return sx0, sy0 - extra_top_px, sx1 + 2 * half_w, sy1 + 2 * half_h


def _units_by_tile(
    scenario: LoadedScenario, unit_filter: UnitFilter = UnitFilter()
) -> dict[tuple[int, int], list[tuple]]:
    """Every unit, grouped by EVERY tile of its footprint -- the key Stepped
    mode's compositor (_paint_tile_and_units_iso) uses to draw each of a
    unit's diamonds at exactly the point in depth_order's loop where that
    diamond belongs. A multi-tile building therefore appears in as many
    buckets as it has footprint tiles, and paints one diamond per bucket.

    Own-tile-only bucketing was the earlier shape, and it was a bug: the
    unit painted its whole footprint slab at its own tile's moment, and
    every footprint tile later in depth_order then painted its own terrain
    diamond back over it -- a span-4 building kept about half its footprint.

    Preserves overlay_units()'s own per-player/per-unit-list stacking order
    within each tile's bucket (built by iterating players/units in that same
    order), so units sharing a tile still paint in the same relative order
    Flat mode would.

    **Off-map units are dropped here, and that is load-bearing downstream.**
    This is now the only off-map guard in the Stepped draw path:
    _draw_unit_iso() indexes elevations[int(unit.y), int(unit.x)] directly
    rather than re-deriving bounds, so it relies on this function having
    already skipped anything unit_tile_bounds() rejects. Buckets over the
    CLAMPED bounds that function returns; its own-tile invariant guarantees
    (int(unit.x), int(unit.y)) is among them even for a building hanging off
    the top-left map edge, which is what keeps _building_bboxes_iso()'s
    own-tile gate total.

    unit_filter (phase 3) drops non-matching units outright rather than
    marking them -- so a hidden unit is absent from every downstream
    consumer at once, including phase 3's pick index, and cannot be
    selected through a tile it no longer paints on. A default UnitFilter()
    keeps every unit in its original relative order, which is what makes
    the filter's arrival a byte-identical no-op for every existing
    caller."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    buckets: dict[tuple[int, int], list] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tile_x0, tile_x1, tile_y0, tile_y1 = bounds
            entry = (unit, _unit_color(unit, player_color))
            for ty in range(tile_y0, tile_y1):
                for tx in range(tile_x0, tile_x1):
                    buckets.setdefault((tx, ty), []).append(entry)
    return buckets


@dataclass(frozen=True)
class SpriteLayer:
    """Everything the Stepped compositor needs to draw real sprites (P3-g3),
    precomputed once per elevations snapshot the same way units_by_tile and
    building_bboxes already are -- resolving and decoding a sprite per chunk
    would dominate a chunk's cost outright.

    by_anchor maps a tile to the sprites that paint AT that tile, in the
    units' own relative order. bboxes is keyed the same way and merges
    straight into building_bboxes, so composite_rect_iso's existing bystander
    machinery pulls a sprite's anchor tile into the candidate set with no
    change to that function's logic at all.

    skip_ids holds id(unit) for every unit that resolved to a sprite, so its
    coloured diamond is not drawn underneath -- a sprite has transparent
    pixels, so an un-skipped mark shows as a coloured fringe around the
    building rather than being hidden. Object identity is safe here because
    the scenario holds every unit alive for the whole life of this snapshot,
    and by_anchor/units_by_tile are built from those same objects.

    farm_by_tile is the farm-terrain counterpart (P3 farm-terrain plan):
    unlike a sprite, a farm has no
    single anchor tile worth compositing at -- it repaints its OWN terrain
    id at EVERY footprint tile, exactly where _render_tile_iso already
    paints that tile's terrain -- so this is keyed by every covered tile,
    not just one. Value is (terrain_id, outline_color, edge_mask):
    terrain_id feeds _render_tile_iso's terrain_override, outline_color and
    edge_mask (a bitmask of iso_geometry.tile_edge_indices sides that sit on
    the footprint's OUTER boundary) drive the perimeter stroke that keeps a
    placed farm visually distinct from hand-painted farm terrain. A unit
    that lands here also enters skip_ids, same reason as a sprite: its
    coloured slab must not draw underneath.
    """

    by_anchor: dict[tuple[int, int], list[tuple[object, int, int]]]
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]]
    skip_ids: frozenset[int]
    farm_by_tile: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]]


# iso_geometry.tile_edge_indices side names this farm-outline edge_mask packs,
# one bit per side -- see sprite_draws_by_anchor's farm branch for how a
# footprint tile's boundary sides are derived from its bounds.
EDGE_LEFT = 1
EDGE_RIGHT = 2
EDGE_UP_LEFT = 4
EDGE_UP_RIGHT = 8
_FARM_EDGE_BITS = (
    (EDGE_LEFT, "left"),
    (EDGE_RIGHT, "right"),
    (EDGE_UP_LEFT, "up_left"),
    (EDGE_UP_RIGHT, "up_right"),
)


def _terrain_overlay_for(unit_const: int) -> int | None:
    """The terrain id `unit_const` draws as instead of a coloured mark, or
    None. A real .sld always wins over a terrain override -- most consts
    carrying a foundation terrain (see FOUNDATION_TERRAIN's own docstring)
    are ordinary buildings whose foundation terrain is purely an in-game
    construction-outline hint, and must keep drawing as their sprite; Farm
    and its family are the only consts that end up with a foundation
    terrain AND no .sld today. Gated on TABLE membership, not decode
    success, so behavior stays deterministic across installs: a const
    whose .sld happens to be unreadable on THIS install must still fall
    back to the coloured mark, never silently pick up a terrain override
    instead."""
    if unit_const in unit_sprites.graphic_map():
        return None
    return FOUNDATION_TERRAIN.get(unit_const)


def sprite_draws_by_anchor(
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter = UnitFilter(),
    corner_rise: np.ndarray | None = None,
    with_farms: bool = True,
) -> SpriteLayer:
    """Resolves every visible unit to a real .sld sprite or a farm-terrain
    override, or leaves it to the coloured mark.

    A unit that resolves nowhere -- no graphic in the table and no
    foundation terrain, no install, a missing or unreadable file, or a
    frame that fails to decode -- is simply absent from every field here,
    which is exactly what makes the sprite/farm path strictly additive.

    **GAIA's `rotation` is never treated as an angle.** For GAIA objects it is
    a tree/doodad graphic-variant index (integers well outside [0, 2*pi)), so
    they resolve at angle 0 -- see AGENTS.md's hard rule.

    corner_rise (Track P3-g6): pass Sloped's (h+1, w+1) corner-rise field to
    anchor sprites at iso_geometry.unit_rise_px() -- the same surface
    _paint_tile_and_units_sloped()/unit_pick.unit_rise_px_for() already use --
    instead of Stepped's flat `elevation * elev_step`. None (the default)
    keeps Stepped's exact expression, including its rounding, so
    render_terrain_sloped's flat-map byte-identity oracle stays exact
    character for character.

    with_farms=False (Track P3-g6) suppresses the farm-terrain-override
    branch entirely -- neither farm_by_tile nor skip_ids gains an entry for
    a farm unit -- so Sloped keeps today's plain coloured mark for farms
    without a paint-time skip making them invisible. See
    _paint_tile_and_units_sloped's own farm handling for the other half of
    that deferral."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    half_w, half_h = proj.half_w, proj.half_h
    by_anchor: dict[tuple[int, int], list] = {}
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    skip_ids: set[int] = set()
    farm_by_tile: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}

    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            rotation = 0.0 if player_id == 0 else float(unit.rotation)
            team_index = scenario.team_indices[player_id]
            pieces = unit_sprites.sprite_pieces_for(unit.unit_const, rotation, team_index, half_w)
            if not pieces:
                terrain_id = _terrain_overlay_for(unit.unit_const) if with_farms else None
                if terrain_id is not None:
                    color = _unit_color(unit, player_color)
                    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
                    for ty in range(tile_y0, tile_y1):
                        for tx in range(tile_x0, tile_x1):
                            mask = 0
                            if tx == tile_x0:
                                mask |= EDGE_LEFT
                            if ty == tile_y1 - 1:
                                mask |= EDGE_RIGHT
                            if ty == tile_y0:
                                mask |= EDGE_UP_LEFT
                            if tx == tile_x1 - 1:
                                mask |= EDGE_UP_RIGHT
                            # Later unit wins on overlap -- overlapping farms
                            # can't happen in-game, but a scenario file can
                            # contain them, and a byte-identity test needs a
                            # deterministic answer.
                            farm_by_tile[(tx, ty)] = (terrain_id, color, mask)
                    skip_ids.add(id(unit))
                continue

            span_x, span_y = BUILDING_TILE_SPANS.get(unit.unit_const, NON_BUILDING_SPAN)
            # The UNCLAMPED footprint start, so a building hanging off a map
            # edge still anchors on its true centre rather than on the centre
            # of whatever survived clamping.
            fx = _span_start(unit.x, span_x) + span_x / 2
            fy = _span_start(unit.y, span_y) + span_y / 2
            ux, uy = int(unit.x), int(unit.y)
            # own_fx/own_fy: sub-tile fractions INSIDE the unit's own tile,
            # not to be confused with fx/fy above (the FOOTPRINT CENTRE in
            # continuous tile coords) -- unit_rise_px() needs the former, the
            # x/y placement below needs the latter. Same names, different
            # quantities is exactly how this file has produced green-suite
            # bugs before (see the half-tile floating-sprite bug this
            # function's own comment below records).
            own_fx, own_fy = unit.x - ux, unit.y - uy
            rise_px = (
                iso_geometry.unit_rise_px(corner_rise, ux, uy, own_fx, own_fy)
                if corner_rise is not None
                else int(elevations[uy, ux]) * proj.elev_step
            )
            ax = round(proj.origin_x + (fx + fy) * half_w)
            # The trailing `+ half_h` is not a fudge, and leaving it out is what
            # made every sprite float exactly half a tile above its ground
            # (reported from a live window 2026-08-24, and the reason this line
            # is commented at all).
            #
            # origin + ((fx+fy)*half_w, (fy-fx)*half_h) maps INTEGER tile coords
            # to tile_screen_origin's convention, which is the diamond's
            # BOUNDING-BOX TOP-LEFT -- not its centre. Feed that map a tile's
            # four continuous corners and you get a diamond centred half_h ABOVE
            # the one actually painted; the x term is centred for free (the two
            # +0.5s add), the y term is not (they cancel). So the ground point a
            # hotspot must land on is that map's output plus half_h.
            #
            # Only y needs it. Verified by measurement, not by eye: a visual
            # "the base lands on its footprint diamond" check passed while this
            # was wrong, because half a tile reads as plausible contact shadow.
            #
            # rise_px stays INSIDE this round() (Track P3-g6): pulling it out
            # can differ by 1px from _paint_tile_and_units_sloped's own
            # rounding and would break the flat-map byte-identity oracle this
            # expression is required to reduce to exactly.
            ay = round(proj.origin_y + (fy - fx) * half_h - rise_px) + half_h

            # Every piece paints at the unit's own anchor tile, offset by its
            # own (dx, dy) -- the degenerate placement case: real per-piece
            # depth slotting (a town centre villager standing between the
            # front and back pieces) is unplanned follow-up work. This still
            # closes the reported bug in full; it only loses cross-piece
            # unit sandwiching.
            anchor = unit_sprites.sprite_anchor_tile(*bounds)
            slot = by_anchor.setdefault(anchor, [])
            bbox = bboxes.get(anchor)
            for piece in pieces:
                px, py = ax + piece.dx, ay + piece.dy
                slot.append((piece.draw, px, py))
                h, w = piece.draw.rgba.shape[:2]
                x0, y0 = px - piece.draw.hotspot_x, py - piece.draw.hotspot_y
                piece_bbox = (x0, y0, x0 + w, y0 + h)
                bbox = piece_bbox if bbox is None else (
                    min(bbox[0], piece_bbox[0]),
                    min(bbox[1], piece_bbox[1]),
                    max(bbox[2], piece_bbox[2]),
                    max(bbox[3], piece_bbox[3]),
                )
            bboxes[anchor] = bbox
            skip_ids.add(id(unit))
    return SpriteLayer(
        by_anchor=by_anchor, bboxes=bboxes, skip_ids=frozenset(skip_ids), farm_by_tile=farm_by_tile
    )


def merge_sprite_bboxes(
    building_bboxes: dict[tuple[int, int], tuple[int, int, int, int]], sprites: SpriteLayer
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """building_bboxes unioned with a SpriteLayer's own, so composite_rect_iso
    pulls a sprite's anchor tile in as a bystander.

    **This is the half of P3-g3 that is easiest to skip and ships a VISIBLE
    chunk-seam clip rather than a latent bug.** _building_bboxes_iso is gated
    to span > 1 units, on the stated grounds that a single-tile object "can't
    make a neighbour a bystander". Sprites break that premise outright: a
    villager's sprite is a 200x200 native canvas against a 64x32 diamond, so
    even a 1x1 unit's pixels reach several tiles away and across chunk
    boundaries. Merging here lifts that gate for sprite-bearing units only,
    while a unit that fell back to a mark keeps today's behaviour exactly.

    Only ever grows a bbox, never shrinks one -- a coarser bystander flag is a
    no-op extra paint, per composite_rect_iso's own accepted tradeoff."""
    if not sprites.bboxes:
        return building_bboxes
    merged = dict(building_bboxes)
    for key, bbox in sprites.bboxes.items():
        prior = merged.get(key)
        merged[key] = bbox if prior is None else (
            min(prior[0], bbox[0]),
            min(prior[1], bbox[1]),
            max(prior[2], bbox[2]),
            max(prior[3], bbox[3]),
        )
    return merged


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
) -> None:
    """Terrain, then that tile's own units -- the single per-tile step both
    render_terrain_iso_with_proj()'s full loop and refresh_region_iso()'s
    incremental loop call, always in the SAME depth_order position, so the
    two paths can never draw units in a different relative order than
    terrain occlusion requires (Phase 5).

    "Paint all terrain, then all units on top" (a separate overlay pass
    after the whole region is final) was considered and rejected: a terrain
    tile painted LATER in depth_order (closer to the camera) must be able
    to visually occlude a unit painted EARLIER at a farther tile, which only
    holds if units are interleaved tile-by-tile like this -- a bolt-on
    "units after" pass would draw every affected unit on top regardless of
    whether a later-depth terrain tile should actually cover it, breaking
    exactly the occlusion property this project's own verify_iso_render.py
    already established for terrain.

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
    farm = sprites.farm_by_tile.get((tile.x, tile.y)) if sprites is not None else None
    terrain_override = farm[0] if farm is not None else None
    _render_tile_iso(
        img, tile, tile_px, proj, elevations, map_w, map_h,
        offset=offset, terrain_override=terrain_override,
    )
    if farm is not None:
        _, outline_color, edge_mask = farm
        off_x, off_y = offset
        base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
        base_x -= off_x
        base_y -= off_y
        color_arr = np.array(outline_color, dtype=np.uint8)
        for bit, side in _FARM_EDGE_BITS:
            if not (edge_mask & bit):
                continue
            dst_y, dst_x = iso_geometry.tile_edge_indices(tile_px, side)
            values = np.broadcast_to(color_arr, (dst_y.size, 3))
            _clipped_paint(img, base_y, base_x, dst_y, dst_x, values)
    skip = sprites.skip_ids if sprites is not None else frozenset()
    for unit, color in units_by_tile.get((tile.x, tile.y), ()):
        if id(unit) in skip:
            continue  # its sprite paints instead, once, at its anchor tile
        _draw_unit_iso(img, unit, color, tile.x, tile.y, tile_px, proj, elevations, offset=offset)

    # Sprites last within this tile's step, so a unit standing on it is not
    # cut by its own tile's terrain. A sprite paints ONCE, here at the tile
    # that comes last in depth_order among its footprint -- see
    # unit_sprites.sprite_anchor_tile for why that tile and not the unit's own.
    if sprites is not None:
        off_x, off_y = offset
        for draw, ax, ay in sprites.by_anchor.get((tile.x, tile.y), ()):
            _clipped_paint_rgba(
                img, ay - draw.hotspot_y - off_y, ax - draw.hotspot_x - off_x, draw.rgba
            )


def overlay_units(
    img: np.ndarray, scenario: LoadedScenario, unit_filter: UnitFilter = UnitFilter()
) -> np.ndarray:
    """Draws a colored dot per unit -- see _draw_unit() for the per-unit
    rules. Mutates and returns img.

    unit_filter is threaded here for the HEADLESS path only (render_scenario
    -> save_png, tools/dump_scenario.py): the viewer composites through
    composite_rect_flat()'s precomputed unit_draws, never through this
    function. refresh_units_over() deliberately does NOT gain the parameter
    -- it has no live callers at all today, and adding one to dead code
    would just be noise."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(tile_w, tile_h)
    assert img.shape[:2] == (tile_h * tile_px, tile_w * tile_px), (
        f"img shape {img.shape[:2]} doesn't match this scenario's expected "
        f"{(tile_h * tile_px, tile_w * tile_px)} at tile_px={tile_px} -- was "
        f"it rendered by render_terrain() for a different scenario?"
    )
    out = img.copy()

    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            _draw_unit(out, unit, player_color, tile_w, tile_h, tile_px)
    return out


def refresh_units_over(img: np.ndarray, scenario: LoadedScenario, dirty_tiles, tile_px: int) -> None:
    """Re-draws any unit mark whose footprint overlaps dirty_tiles (an
    iterable of (tile_x, tile_y) pairs) into img, in place. Units never move
    from a terrain/elevation edit, but overlay_units() draws *over* terrain
    -- refresh_tiles() only repaints raw terrain, so calling it alone after
    an edit silently erases whatever building footprint, tree, or resource
    dot was sitting on an edited tile. This is the fix: called right after
    refresh_tiles() with the same dirty set, it puts back the unit marks that
    would otherwise have been erased, without re-drawing the whole map's
    units.

    Redrawing is NOT as simple as "draw every unit whose footprint overlaps
    dirty_tiles": overlay_units() draws units in a fixed order (per player,
    then per unit in list order), so a later unit's footprint can paint over
    an earlier one's -- e.g. a tree drawn after the building it happens to
    sit inside of. If only the units directly overlapping dirty_tiles get
    redrawn, redrawing a building whose footprint happens to cover a *tree*
    that isn't itself touched would permanently erase that tree (confirmed:
    this was a real bug here, caught by tools/verify_write_path.py's callers
    comparing against a full render_scenario()). The fix is to first grow
    dirty_tiles to a fixed point -- whenever a unit's footprint overlaps the
    (growing) dirty set, its *entire* footprint joins the set too, and repeat
    until nothing new is added -- then do one redraw pass, in the same
    per-player/per-unit order overlay_units() uses, over every unit whose
    footprint overlaps the final, expanded set. That reproduces the exact
    same stacking order a full render would have produced for that region,
    without re-rendering tiles the edit never touched.

    A plain per-unit scan (matching stroke_dirty_indices' "linear scan is
    cheap enough at this project's map sizes" reasoning -- see
    edit_history.py) rather than any spatial index: this project's largest
    example file has under 11,000 units total. The fixed-point expansion adds
    more such scans, but stays cheap in practice since footprint overlaps are
    local -- it terminates as soon as a pass adds no new tiles."""
    dirty = set(dirty_tiles)
    if not dirty:
        return
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height

    # Left on the identity PLAYER_COLORS, not scenario.player_colors, unlike
    # every other call site: this function has no live caller anywhere in the
    # repo (production, tests, or tools) -- see overlay_units()'s docstring --
    # so there is no path through which a stored color override would ever
    # need to reach it.
    all_units = [
        (unit, PLAYER_COLORS[player_id % len(PLAYER_COLORS)], unit_tile_bounds(unit, tile_w, tile_h))
        for player_id, units in enumerate(scenario.unit_manager.units)
        for unit in units
    ]

    changed = True
    while changed:
        changed = False
        for _unit, _color, bounds in all_units:
            if bounds is None:
                continue
            tile_x0, tile_x1, tile_y0, tile_y1 = bounds
            if not any(tile_x0 <= x < tile_x1 and tile_y0 <= y < tile_y1 for x, y in dirty):
                continue
            new_tiles = {(x, y) for x in range(tile_x0, tile_x1) for y in range(tile_y0, tile_y1)} - dirty
            if new_tiles:
                dirty |= new_tiles
                changed = True

    for unit, color, bounds in all_units:
        if bounds is None:
            continue
        tile_x0, tile_x1, tile_y0, tile_y1 = bounds
        if any(tile_x0 <= x < tile_x1 and tile_y0 <= y < tile_y1 for x, y in dirty):
            _draw_unit(img, unit, color, tile_w, tile_h, tile_px)


def render_scenario(
    scenario: LoadedScenario, with_units: bool = True, isometric: bool = False, style: str | None = None
) -> np.ndarray:
    """isometric=True renders Stepped mode (render_terrain_iso) instead of
    Flat (render_terrain). with_units applies in every mode now (Phase 5 for
    Stepped, Phase 6 for Sloped): each draws units at their own tile's
    elevation, interleaved with terrain in depth order via that mode's own
    with_units parameter -- see _paint_tile_and_units_iso()'s/
    _paint_tile_and_units_sloped()'s docstrings for why that has to be
    interleaved rather than a separate overlay pass, the way Flat mode's
    overlay_units() draws over an already-finished terrain image.

    style, if given ("flat"/"stepped"/"sloped"), selects the render and
    overrides isometric -- Phase 6 (Sloped)'s own entry point, added
    without changing isometric's existing meaning or any existing call
    site: dump_scenario.py's --iso flag and save_png()'s own isometric
    passthrough both keep working exactly as before when style is left at
    its None default. isometric stays a valid way to select Stepped
    (style="stepped" is equivalent, not required)."""
    if style is not None:
        if style not in ("flat", "stepped", "sloped"):
            raise ValueError(f"style must be 'flat', 'stepped', or 'sloped', got {style!r}")
        if style == "sloped":
            return render_terrain_sloped(scenario, with_units=with_units)
        isometric = style == "stepped"
    if isometric:
        return render_terrain_iso(scenario, with_units=with_units)
    img = render_terrain(scenario)
    if with_units:
        img = overlay_units(img, scenario)
    return img


def save_png(
    scenario: LoadedScenario,
    out_path: str,
    with_units: bool = True,
    scale: int = 1,
    isometric: bool = False,
    style: str | None = None,
) -> None:
    from PIL import Image

    img = render_scenario(scenario, with_units=with_units, isometric=isometric, style=style)
    pil_img = Image.fromarray(img, mode="RGB")
    if scale != 1:
        pil_img = pil_img.resize(
            (pil_img.width * scale, pil_img.height * scale), Image.NEAREST
        )
    pil_img.save(out_path)
