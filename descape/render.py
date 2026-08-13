"""Renders a loaded scenario's Map + Units to an RGB numpy array.

Pure function, no Qt dependency -- backs both the headless PNG export and the
PyQt5 viewer's canvas.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

import numpy as np

from descape import asset_source, iso_geometry, settings
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    BUILDING_FOOTPRINTS,
    PLAYER_COLORS,
    RESOURCE_COLORS,
    TREE_COLOR,
    TREE_UNIT_IDS,
    color_for_terrain_id,
)

NON_BUILDING_RADIUS = (0, 0)

# Largest (rx, ry) in BUILDING_FOOTPRINTS -- the farthest a footprint tile
# can sit from a unit's own floored (px, py) position. Stepped mode's
# refresh_region_iso() (Phase 5) needs this to size its lateral seed
# dilation: an elevation edit on a building's own center tile moves EVERY
# one of that building's footprint diamonds (they're all drawn at the
# center's elevation -- see _unit_iso_footprint), not just the center tile
# itself, so the +-1 dilation that's enough for terrain skirts alone isn't
# enough once units are interleaved in.
UNIT_FOOTPRINT_MAX_RADIUS = max((max(rx, ry) for rx, ry in BUILDING_FOOTPRINTS.values()), default=0)

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


@lru_cache(maxsize=256)
def _shadow_factors(tile_px: int, rise_px: int, side: str) -> np.ndarray:
    """float32 darkening factors aligned 1:1 with
    iso_geometry.shadow_quad_indices(tile_px, rise_px, side)'s own output
    (same call, same cache key shape) -- factor = 1 at the contact row's
    own far edge, CONTACT_SHADE at depth=0 (the row touching the caster's
    diamond), linearly interpolated between. Normalized on rise_px, not an
    absolute pixel falloff: that's what makes a delta=1 step a tight
    contact band and a tall column a long soft gradient, rather than every
    delta getting the same fixed-width band regardless of how tall the
    drop actually is.

    float32, not float64: _clipped_darken multiplies this against a uint8
    image and truncates back to uint8 either way, and pinning the
    intermediate dtype is what keeps the full-canvas and scratch-canvas
    paint paths bit-identical."""
    _dst_y, _dst_x, depth = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
    return (1 - (1 - CONTACT_SHADE) * (1 - depth.astype(np.float32) / rise_px)).astype(np.float32)


def _clipped_darken(img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, factors: np.ndarray) -> None:
    """img[base_y+dst_y, base_x+dst_x] *= factors, in place, dropping any
    destination pixel outside img's own bounds -- the multiplicative,
    darkening-only counterpart to _clipped_paint (see that function's own
    docstring for why clipping matters at all: a scratch-canvas call site
    composites a rect where most candidate tiles only partially overlap).

    Unlike _clipped_paint, THIS function's full-canvas call site (offset
    (0, 0), against render_terrain_iso_with_proj()'s full map canvas) CAN
    legitimately drop pixels: shadow_quad_indices' dst_y goes negative by
    design (a shadow reaches above its own tile's row 0), and
    iso_geometry.tile_screen_bounds_swept's own widened headroom is a
    documented, accepted worst-case bound (see canvas_size_and_origin's
    "Canvas is not resized" note) -- the canvas itself is NOT grown to
    guarantee every theoretical shadow position
    stays in bounds, only wide/tall enough for every real example file
    measured so far. Rely on this clip for that rare top-edge overflow,
    same as the scratch-canvas case relies on it for partial-rect overlap.

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
    canvas call site. NOT true of every caller in this same pipeline any
    more: _clipped_darken() below (the contact-shadow compositor) CAN drop
    pixels even at the full-canvas call site, since a shadow's dst_y goes
    negative by design -- see that function's own docstring for why that's
    an accepted, documented tradeoff rather than a bug to fix here."""
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax, values = ay[in_bounds], ax[in_bounds], values[in_bounds]
    img[ay, ax] = values


def _render_tile_iso(
    img: np.ndarray,
    tile,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    map_w: int,
    map_h: int,
    offset: tuple[int, int] = (0, 0),
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
    face) shading and the contact shadow are unrelated and unaffected --
    SKIRT_SHADE/CONTACT_SHADE below are fixed darkening, not elevation-
    dependent (a tile darkens a NEIGHBOR because of a height difference,
    never itself because of its own absolute elevation), and stay.

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
    could-drift copy."""
    texture = asset_source.get_terrain_texture_array(tile.terrain_id)
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        top_block = texture[oy : oy + tile_px, ox : ox + tile_px]
    else:
        r, g, b = color_for_terrain_id(tile.terrain_id)
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
        s_dst_y, s_dst_x, _depth = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
        _clipped_darken(img, base_y, base_x, s_dst_y, s_dst_x, _shadow_factors(tile_px, rise_px, side))


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
    scenario: LoadedScenario, with_units: bool = True
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
    for x, y in iso_geometry.depth_order(w, h):
        _paint_tile_and_units_iso(img, tile_grid[y][x], units_by_tile, tile_px, proj, elevations, w, h)
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
    units_by_tile: dict, w: int, h: int, proj: iso_geometry.IsoProjection, elevations: np.ndarray
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
    possible test would is a no-op extra paint, never a missed one."""
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for (px, py), entries in units_by_tile.items():
        union: tuple[int, int, int, int] | None = None
        for unit, _color in entries:
            if BUILDING_FOOTPRINTS.get(unit.unit_const, NON_BUILDING_RADIUS) == NON_BUILDING_RADIUS:
                continue
            bbox = _unit_screen_bbox_iso(unit, w, h, proj, elevations)
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


def dirty_screen_bbox_iso(
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    with_units: bool = True,
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
    just telling the caller to fall back to a full re-render."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    dirty_xy = {(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices}
    if not dirty_xy:
        return None

    for x, y in dirty_xy:
        elevations[y, x] = mm.get_tile(x, y).elevation

    if any(not (proj.min_elev <= int(elevations[y, x]) <= proj.max_elev) for x, y in dirty_xy):
        return None

    # Lateral expansion: a tile's own skirt geometry samples its "left"/
    # "right" neighbor's elevation (see _render_tile_iso), so an edited tile
    # can change a *neighbor's* skirt even though the neighbor's own
    # elevation never changed (+-1 is enough for that alone -- see
    # iso_geometry.skirt_quad_indices' docstring for exactly which two of a
    # tile's four grid-neighbors can ever show a skirt facing it). But a
    # unit's footprint is drawn entirely at its OWN tile's elevation (see
    # _unit_iso_footprint), so if a dirty tile happens to be some building's
    # center, the edit moves footprint diamonds up to UNIT_FOOTPRINT_MAX_
    # RADIUS tiles away -- a full box in both axes, not just the 4-neighbor
    # cross skirts alone would need, since a footprint corner can be
    # diagonal from its own center. Dilating by whichever radius is larger
    # covers both needs with one sweep.
    radius = max(1, UNIT_FOOTPRINT_MAX_RADIUS) if with_units else 1
    seed = set(dirty_xy)
    for x, y in list(dirty_xy):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    seed.add((nx, ny))

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

    canvas_w, canvas_h = _canvas_pixel_dims(proj)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


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
    for cx, cy in candidates:
        tile = mm.get_tile(int(cx), int(cy))
        _paint_tile_and_units_iso(scratch, tile, units_by_tile, tile_px, proj, elevations, w, h, offset=(x0, y0))
    return scratch


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


def _flat_unit_draws(scenario: LoadedScenario, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Every unit's pixel-space draw, as (N,4) int32 half-open bboxes
    (x0,y0,x1,y1) plus (N,3) uint8 colors -- Phase B-E's precomputed input
    to composite_rect_flat(), built in EXACTLY overlay_units()'s own
    per-player/per-unit-list order. Off-map units (_unit_tile_bounds()
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
        player_color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
        for unit in units:
            bounds = _unit_tile_bounds(unit, tile_w, tile_h)
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
    unchanged, so every existing caller (viewer.py's ViewerWindow, this
    project's verify_iso_*.py scripts) is untouched: the same two pieces
    back Phase B-B's chunk cache without an img array in the picture at all.

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


# Default chunk size for IsoChunkCache -- measured, not guessed: benched
# CHUNK_PX in {256, 512, 1024} via tools/verify_iso_chunks.py on this
# project's largest real map (480x480, tile_px=32). 1024 wins cold
# full-canvas assembly by only ~8% over 512 (4.7s vs 5.1s) while being a
# much coarser invalidation granularity for a moving viewport; 256 loses on
# both cold assembly (~6s) and warm single-tile patch cost (~44ms vs ~28ms).
DEFAULT_CHUNK_PX = 512


class _ChunkCacheBase:
    """Grid/LRU bookkeeping shared by IsoChunkCache (Stepped, Phase B-B) and
    FlatChunkCache (Flat, Phase B-E) -- extracted because get_chunk/
    render_rect/patch/invalidate_region are pure chunk-grid arithmetic with
    zero mode-specific content, proven correct by tools/verify_iso_chunks.py's
    own byte-identity checks well before this split existed. This is NOT a
    weakening of composite_rect_iso()/composite_rect_flat()'s own "stay an
    independent implementation, never re-expressed in terms of the chunk
    path" rule -- that rule is about the COMPOSITOR (only ever reached here
    through the subclass's _composite_rect() hook, still two genuinely
    separate functions); this class owns LRU/grid bookkeeping only, the same
    thing any other chunk cache would.

    A subclass must, in its own __init__ (kept fully subclass-owned, not
    called from here, so each cache's own constructor signature/docstring
    stays exactly as-is): set self.chunk_px, call self._refresh_source_caches()
    once, then call self._init_max_chunks(max_chunks). It must also implement:
      - canvas_dims() -> (width, height) in canvas pixels
      - _composite_rect(x0, y0, x1, y1) -> (h, w, 3) uint8 array
      - _refresh_source_caches() -> None
      - a `style` class attribute ("stepped" / "flat") -- checked at the
        MapView.set_source() boundary (Phase B-E) to catch a cache wired to
        the wrong terrain style at construction time, rather than only once
        an edit exposes the mismatch later.

    mip is part of every cache key (Phase B-B); get_chunk()/render_rect()'s
    own literal "chunk 0" callers and patch()/invalidate_region()'s
    mip-independent chunk-index math are Phase B-D's to fix -- unchanged
    here, this extraction is behavior-preserving."""

    style: str = ""

    def _init_max_chunks(self, chunk_px: int, max_chunks: int | None) -> None:
        """Whole-canvas-at-chunk_px default, not some smaller fixed
        constant: the viewer always fitInView()s the full map on open, so
        the default/steady-state working set IS every chunk. A smaller cap
        would silently thrash (evict chunks the very next full repaint
        needs again) instead of ever reaching a warm, blit-only steady
        state -- the exact failure mode a chunk cache exists to avoid. See
        each subclass's own docstring for the measured memory cost of this
        default."""
        self.chunk_px = chunk_px
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        if max_chunks is None:
            canvas_w, canvas_h = self.canvas_dims()
            max_chunks = ((canvas_w + chunk_px - 1) // chunk_px) * ((canvas_h + chunk_px - 1) // chunk_px)
        self.max_chunks = max_chunks

    def canvas_dims(self) -> tuple[int, int]:
        raise NotImplementedError

    def _composite_rect(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        raise NotImplementedError

    def _refresh_source_caches(self) -> None:
        raise NotImplementedError

    def get_chunk(self, mip: int, cx: int, cy: int) -> np.ndarray:
        """Returns chunk (mip, cx, cy)'s composited pixels, from cache if
        present (moved to most-recently-used), else composited fresh via
        self._composite_rect() and inserted, evicting the least-recently-used
        entry past max_chunks. Clipped to canvas bounds at the high edge --
        a chunk straddling the canvas edge is smaller than chunk_px x
        chunk_px, same "ragged last chunk" shape any tile-based grid has."""
        key = (mip, cx, cy)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        canvas_w, canvas_h = self.canvas_dims()
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        x1, y1 = min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)
        chunk = self._composite_rect(x0, y0, x1, y1)
        self._cache[key] = chunk
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_chunks:
            self._cache.popitem(last=False)
        return chunk

    def render_rect(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Assembles pixels for [x0, x1) x [y0, y1) (clipped to canvas
        bounds) from chunks -- fetching/compositing each via get_chunk() as
        needed. The stitched result must be byte-identical to the
        corresponding crop of an independent full render regardless of
        chunk request order or what was already cached -- see
        tools/verify_iso_chunks.py (Stepped) / tests/test_flat_chunks.py
        (Flat)."""
        canvas_w, canvas_h = self.canvas_dims()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                chunk = self.get_chunk(0, cx, cy)
                chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                ox0, oy0 = max(x0, chunk_x0), max(y0, chunk_y0)
                ox1, oy1 = min(x1, chunk_x0 + chunk.shape[1]), min(y1, chunk_y0 + chunk.shape[0])
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                out[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = chunk[
                    oy0 - chunk_y0 : oy1 - chunk_y0, ox0 - chunk_x0 : ox1 - chunk_x0
                ]
        return out

    def patch(self, bbox: tuple[int, int, int, int]) -> None:
        """"Patch, don't drop": for
        every chunk CURRENTLY cached that bbox overlaps, recomposites just
        the intersected sub-rect via self._composite_rect() and writes it
        into the existing chunk array in place -- the cached chunk stays
        valid immediately, without paying a full chunk recomposite (or
        leaving a stale one on screen until the next get_chunk() eviction).
        Chunks NOT currently cached need no action: get_chunk() always
        composites fresh against the current state, so there's nothing
        stale to fix for those.

        Looks up the overlapping chunk-grid range directly (same index math
        as invalidate_region()) rather than scanning every cached entry --
        a real cost difference once max_chunks is in the hundreds and only
        a handful of chunks overlap a single edit's bbox.

        bbox must already reflect the edit; this method does not mutate any
        underlying source state itself (elevations, terrain) -- callers
        that need to (dirty_screen_bbox_iso() for Stepped) do so before
        calling this. Refreshes this cache's own per-edit derived state
        first (self._refresh_source_caches()) -- see each subclass's own
        docstring for what that means to it."""
        self._refresh_source_caches()
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        cx0, cy0 = px0 // self.chunk_px, py0 // self.chunk_px
        cx1, cy1 = (px1 - 1) // self.chunk_px, (py1 - 1) // self.chunk_px
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                # Mip hardcoded to 0: nothing writes a mip != 0 key today, so
                # this is behavior-preserving for now. Whoever adds mip levels
                # (Track B-D) must patch every cached mip whose footprint
                # overlaps bbox here, not just mip 0.
                chunk = self._cache.get((0, cx, cy))
                if chunk is None:
                    continue
                chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                chunk_x1, chunk_y1 = chunk_x0 + chunk.shape[1], chunk_y0 + chunk.shape[0]
                ix0, iy0 = max(px0, chunk_x0), max(py0, chunk_y0)
                ix1, iy1 = min(px1, chunk_x1), min(py1, chunk_y1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                patched = self._composite_rect(ix0, iy0, ix1, iy1)
                chunk[iy0 - chunk_y0 : iy1 - chunk_y0, ix0 - chunk_x0 : ix1 - chunk_x0] = patched

    def patch_rects(self, rects) -> None:
        """patch() for each rect in rects -- Phase B-E's Flat edits patch
        per-tile rects rather than one union bbox (a union over a scattered
        undo set can span the whole map, turning patch() into a full
        recomposite; see ViewerWindow._apply_dirty's Flat branch)."""
        for rect in rects:
            self.patch(rect)

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Evicts every cached chunk whose grid cell intersects bbox --
        forces a full recomposite from get_chunk() next time that chunk is
        requested, rather than trusting whatever's cached. Distinct from
        patch(): patch() keeps a cached chunk valid immediately at the cost
        of only the touched sub-rect; this drops it outright. Exists as its
        own primitive -- separate from patch() -- so a correctness bug
        elsewhere can't be masked by patch() quietly papering over it;
        tools/verify_iso_chunks.py's "invalidate_region round-trips" check
        calls this directly, forces a real recomposite via get_chunk(), and
        compares against a fresh full render."""
        px0, py0, px1, py1 = bbox
        cx0, cy0 = px0 // self.chunk_px, py0 // self.chunk_px
        cx1, cy1 = (px1 - 1) // self.chunk_px, (py1 - 1) // self.chunk_px
        for key in list(self._cache):
            _mip, cx, cy = key
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                del self._cache[key]


class IsoChunkCache(_ChunkCacheBase):
    """Qt-free LRU cache of composited Stepped-mode canvas chunks, keyed by
    (mip, chunk_x, chunk_y) -- Phase B-B of Track B.
    chunk_x/chunk_y are chunk-GRID indices: canvas pixel
    (chunk_x*chunk_px, chunk_y*chunk_px) is that chunk's own origin. mip is
    always 0 today (Phase B-D adds real mip levels); the key already
    carries it so B-D doesn't need a cache-key migration later.

    Each chunk is composited independently via composite_rect_iso() -- the
    SAME function render_terrain_iso_with_proj()'s full loop and
    refresh_region_iso()'s per-edit patch both reduce to -- so a chunk's
    pixels never depend on which OTHER chunks happen to be cached or in
    what order they were requested (this class's own load-bearing
    correctness bar, see tools/verify_iso_chunks.py). Grid/LRU mechanics
    (get_chunk/render_rect/patch/invalidate_region) live in _ChunkCacheBase,
    shared with Phase B-E's FlatChunkCache -- this class supplies only the
    Stepped-specific pieces: canvas_dims(), _composite_rect(), and
    _refresh_source_caches().

    Wired into viewer.py's Stepped mode as of Phase B-C, via MapCanvasItem;
    also exercised standalone by tools/verify_iso_chunks.py and its own
    bench.

    max_chunks defaults to covering the WHOLE canvas at chunk_px (see
    _ChunkCacheBase._init_max_chunks()'s own docstring for why).

    NOT free memory-wise: a fully-warmed cache holds roughly as many total
    pixel bytes as the old single canvas did (this project's biggest real
    map, 480x480 at chunk_px=512, needs 480 chunks), and render_rect() (see
    below) additionally allocates a fresh full-canvas-sized stitched output
    array on every full-viewport call -- so a full-viewport composite
    transiently needs the persistent cache AND that scratch buffer at once.
    Measured on that map: peak RSS for an open+warm+edit workflow went
    743MB -> 1127MB versus the pre-B-C single-buffer approach -- a real,
    accepted cost of avoiding thrashing at the default zoom, not a memory
    win. Pass an explicit
    smaller max_chunks (as tools/verify_iso_chunks.py's eviction tests do)
    to exercise real LRU eviction instead."""

    style = "stepped"

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
    ):
        self.scenario = scenario
        self.elevations = elevations
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self) -> None:
        """Recomputes units_by_tile/building_bboxes against the CURRENT
        self.elevations -- must run whenever elevations could have changed
        underneath this cache (construction, and every patch()), since a
        building's own screen bbox depends on its center tile's elevation
        (_unit_screen_bbox_iso). Cheap relative to a full chunk recomposite,
        but still real work at ~11k units on this project's bigger real
        files -- runs once per edit here, never once per chunk, which is
        the whole reason composite_rect_iso() takes these as precomputed
        arguments instead of computing them itself."""
        mm = self.scenario.map_manager
        w, h = mm.map_width, mm.map_height
        self.units_by_tile = _units_by_tile(self.scenario) if self.with_units else {}
        self.building_bboxes = (
            _building_bboxes_iso(self.units_by_tile, w, h, self.proj, self.elevations) if self.with_units else {}
        )

    def canvas_dims(self) -> tuple[int, int]:
        """(width, height) in canvas pixels, including skirt headroom --
        see _canvas_pixel_dims()."""
        return _canvas_pixel_dims(self.proj)

    def _composite_rect(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return composite_rect_iso(
            self.scenario,
            x0,
            y0,
            x1,
            y1,
            self.elevations,
            self.proj,
            self.tile_px,
            self.units_by_tile,
            self.building_bboxes,
            self.with_units,
        )


class FlatChunkCache(_ChunkCacheBase):
    """Flat mode's counterpart to IsoChunkCache -- Phase B-E of Track B. Same
    grid/LRU mechanics (_ChunkCacheBase), composited via composite_rect_flat()
    instead of composite_rect_iso()
    (see that function's own docstring for the correctness argument behind
    never needing refresh_units_over()'s fixed-point dirty-tile expansion
    here).

    max_chunks defaults to covering the WHOLE canvas at chunk_px, same as
    IsoChunkCache and for the same fitInView()-on-open reason -- unchanged
    here even though Flat's canvas is the LARGER of the two (675 MiB vs
    340 MiB at 480x480): capping this to something smaller would thrash
    at the app's own default zoom exactly
    the way IsoChunkCache's own docstring already argues against. Phase
    B-D's mip levels are the real fix for the fit-to-view memory cost, not
    a smaller cap here.

    _refresh_source_caches() is a documented no-op after construction:
    unlike Stepped's building bboxes (which move with their center tile's
    elevation, see IsoChunkCache._refresh_source_caches()), Flat's
    unit_draws (_flat_unit_draws()) derive only from unit.x/unit.y,
    unit_const, the owning player index, and map dimensions -- none of
    which any terrain or elevation edit touches. That makes patch() here
    genuinely cheaper than Stepped's per-edit ~15-20ms units_by_tile/
    building_bboxes rebuild, not just an equivalent no-op restated. If a
    future unit-editing feature (v3.5) ever makes unit_draws stale,
    invalidate_units() is the explicit way to force a rebuild -- don't
    "fix" this no-op into an unconditional rebuild instead, since that
    would silently reintroduce the per-edit cost this class exists to
    avoid paying for edits that were never about units at all."""

    style = "flat"

    def __init__(
        self,
        scenario: LoadedScenario,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
    ):
        self.scenario = scenario
        self.tile_px = tile_px
        self.with_units = with_units
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self) -> None:
        """See this class's own docstring: a documented no-op once
        unit_draws already exists (nothing a terrain/elevation edit touches
        can make it stale), except at construction, where it must actually
        build unit_draws the first time."""
        if not hasattr(self, "unit_draws"):
            self.unit_draws = _flat_unit_draws(self.scenario, self.tile_px) if self.with_units else None

    def invalidate_units(self) -> None:
        """Forces unit_draws to be rebuilt on the next patch()/construction-
        style refresh -- for a future unit-editing feature (v3.5) whose
        edits _refresh_source_caches()'s no-op would otherwise miss. Not
        called anywhere today; exists so that no-op doesn't become a trap
        once unit edits are real."""
        if hasattr(self, "unit_draws"):
            del self.unit_draws
        self._refresh_source_caches()

    def canvas_dims(self) -> tuple[int, int]:
        mm = self.scenario.map_manager
        return mm.map_width * self.tile_px, mm.map_height * self.tile_px

    def _composite_rect(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return composite_rect_flat(
            self.scenario, x0, y0, x1, y1, self.tile_px, unit_draws=self.unit_draws, with_units=self.with_units
        )


def _unit_tile_bounds(unit, tile_w: int, tile_h: int) -> tuple[int, int, int, int] | None:
    """(tile_x0, tile_x1, tile_y0, tile_y1) -- the tile-space bounding box
    _draw_unit() would paint for this unit, half-open like Python ranges.
    None if the unit is off-map. Split out from _draw_unit() so
    refresh_units_over() can cheaply test overlap with a dirty-tile set
    without touching img."""
    # Positions are tile-centered floats (e.g. 35.50, 8.50) -- floor, don't
    # round: round()'s round-half-to-even on an always-*.5 value collapses
    # two adjacent tiles onto one output pixel and skips the next one,
    # producing a checkerboard gap pattern that isn't there in the real
    # placement data.
    px, py = int(unit.x), int(unit.y)
    if not (0 <= px < tile_w and 0 <= py < tile_h):
        return None
    rx, ry = BUILDING_FOOTPRINTS.get(unit.unit_const, NON_BUILDING_RADIUS)
    tile_x0, tile_x1 = max(0, px - rx), min(tile_w, px + rx + 1)
    tile_y0, tile_y1 = max(0, py - ry), min(tile_h, py + ry + 1)
    return tile_x0, tile_x1, tile_y0, tile_y1


def _unit_color(unit, player_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """The color rule shared by both Flat's _draw_unit() and Stepped's
    _draw_unit_iso() (Phase 5 reuses this rather than re-deriving it):
    trees are always dark green, non-building resource/decoration objects
    get their real minimap color, everything else (including buildings) is
    colored by owning player -- all regardless of owner, matching AoE2's own
    minimap for the first two categories."""
    is_building = unit.unit_const in BUILDING_FOOTPRINTS
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
    bounds = _unit_tile_bounds(unit, tile_w, tile_h)
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
    footprint -- the same tile-space bounds _unit_tile_bounds() computes for
    Flat mode (see that function's docstring for why int(unit.x)/int(unit.y)
    is already the footprint's own center by construction), plus the single
    elevation every footprint tile is drawn at: elevations[py, px], the
    unit's OWN floored tile's elevation -- not each footprint tile's own
    individual terrain elevation. That's Phase 5's whole plan for
    multi-tile buildings ("center tile's elevation for the whole
    footprint") -- a flat slab, deliberately not per-corner-interpolated
    (real sloped ramps are Phase 6, out of scope here). None if the unit is
    off-map."""
    bounds = _unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    px, py = int(unit.x), int(unit.y)
    return bounds + (int(elevations[py, px]),)


def _draw_unit_iso(
    img: np.ndarray,
    unit,
    color: tuple[int, int, int],
    tile_w: int,
    tile_h: int,
    tile_px: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Stepped mode's counterpart to _draw_unit() (Phase 5): paints one unit's
    mark as a filled iso diamond per occupied footprint tile -- a building
    spanning multiple tiles gets one diamond per tile (fitting naturally
    into the same diamond shape terrain tiles paint in), not one scaled-up
    rectangle the way Flat mode's _draw_unit() draws it. Every footprint
    tile is placed at the unit's OWN tile's elevation (see
    _unit_iso_footprint), so a multi-tile building renders as one flat slab
    at a single height. offset/clipping mirror _render_tile_iso()'s own
    scratch-canvas contract, for refresh_region_iso()'s incremental redraw."""
    footprint = _unit_iso_footprint(unit, tile_w, tile_h, elevations)
    if footprint is None:
        return
    tile_x0, tile_x1, tile_y0, tile_y1, elevation = footprint

    dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
    values = np.full((dst_y.shape[0], 3), color, dtype=np.uint8)
    off_x, off_y = offset
    for ty in range(tile_y0, tile_y1):
        for tx in range(tile_x0, tile_x1):
            base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, elevation, proj)
            _clipped_paint(img, base_y - off_y, base_x - off_x, dst_y, dst_x, values)


def _unit_screen_bbox_iso(
    unit, tile_w: int, tile_h: int, proj: iso_geometry.IsoProjection, elevations: np.ndarray
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
    files. None if the unit is off-map."""
    footprint = _unit_iso_footprint(unit, tile_w, tile_h, elevations)
    if footprint is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1, elevation = footprint
    half_w, half_h = proj.half_w, proj.half_h
    sx0, _ = iso_geometry.tile_screen_origin(tile_x0, tile_y0, elevation, proj)
    sx1, _ = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y1 - 1, elevation, proj)
    _, sy0 = iso_geometry.tile_screen_origin(tile_x1 - 1, tile_y0, elevation, proj)
    _, sy1 = iso_geometry.tile_screen_origin(tile_x0, tile_y1 - 1, elevation, proj)
    return sx0, sy0, sx1 + 2 * half_w, sy1 + 2 * half_h


def _units_by_tile(scenario: LoadedScenario) -> dict[tuple[int, int], list[tuple]]:
    """Every unit, grouped by its own floored (x, y) tile -- the key
    Stepped mode's compositor (_paint_tile_and_units_iso) uses to draw a
    unit at exactly the point in depth_order's loop where it belongs (Phase
    5's plan: "for each (x, y) in that loop, after painting that tile, also
    draw any unit(s) whose floored position is (x, y)"). Preserves
    overlay_units()'s own per-player/per-unit-list stacking order within
    each tile's bucket (built by iterating players/units in that same
    order), so units sharing an exact tile still paint in the same relative
    order Flat mode would."""
    buckets: dict[tuple[int, int], list] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
        for unit in units:
            px, py = int(unit.x), int(unit.y)
            buckets.setdefault((px, py), []).append((unit, _unit_color(unit, player_color)))
    return buckets


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
    already established for terrain."""
    _render_tile_iso(img, tile, tile_px, proj, elevations, map_w, map_h, offset=offset)
    for unit, color in units_by_tile.get((tile.x, tile.y), ()):
        _draw_unit_iso(img, unit, color, map_w, map_h, tile_px, proj, elevations, offset=offset)


def overlay_units(img: np.ndarray, scenario: LoadedScenario) -> np.ndarray:
    """Draws a colored dot per unit -- see _draw_unit() for the per-unit
    rules. Mutates and returns img."""
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
        player_color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
        for unit in units:
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

    all_units = [
        (unit, PLAYER_COLORS[player_id % len(PLAYER_COLORS)], _unit_tile_bounds(unit, tile_w, tile_h))
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


def render_scenario(scenario: LoadedScenario, with_units: bool = True, isometric: bool = False) -> np.ndarray:
    """isometric=True renders Stepped mode (render_terrain_iso) instead of
    Flat (render_terrain). with_units applies in both modes now (Phase 5):
    Stepped draws each unit at its own tile's elevation, interleaved with
    terrain in depth order via render_terrain_iso()'s own with_units
    parameter -- see _paint_tile_and_units_iso()'s docstring for why that
    has to be interleaved rather than a separate overlay pass, the way Flat
    mode's overlay_units() draws over an already-finished terrain image."""
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
) -> None:
    from PIL import Image

    img = render_scenario(scenario, with_units=with_units, isometric=isometric)
    pil_img = Image.fromarray(img, mode="RGB")
    if scale != 1:
        pil_img = pil_img.resize(
            (pil_img.width * scale, pil_img.height * scale), Image.NEAREST
        )
    pil_img.save(out_path)
