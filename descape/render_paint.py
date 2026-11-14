"""The render's paint layer: tile-pixel tiers, top and tile blocks, blend keys, Flat patch rects, clipped paints, ink and grid."""

from __future__ import annotations

from functools import lru_cache
from itertools import repeat
from operator import attrgetter

import numpy as np

from descape import (
    asset_source,
    composite_backend,
    grid_overlay,
    iso_geometry,
    settings,
    terrain_blend,
)
from descape.grid_overlay import GridBake
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    color_for_terrain_id,
)
from descape.view_layers import DEFAULT_LAYERS, LayerState


def _drain(gen):
    """Runs a resumable walk (sprite_draws_by_anchor_sliced/_flat_icon_layer_
    sliced) to completion in one go and returns its StopIteration value --
    what makes each sliced generator's public wrapper a thin drain rather than
    a second copy of the walk.

    `except StopIteration` rather than a bare `for _ in gen: pass` because the
    payload rides the generator's RETURN value, which a for loop discards."""
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


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


# The finest zoom level each graphics quality builds (GH #241). Quality used to
# set only the base resolution above, so a fully zoomed-in Potatest view built
# the same 128 px level as Default, at the same memory and build cost. Past its
# finest level a view enlarges that level, as every quality already does past 128.
# Default reaches the 192 level, x1 art at an even 2x (TASK-601). Enhanced stays at 128
# until TASK-441 gives it 192 with x2 art at 1:1: it flips [4] here and in NATIVE_TIER_MAX.
FINEST_TILE_PIXELS = {
    1: 32,
    2: 64,
    3: iso_geometry.MIP_MAX_TILE_PIXELS,
    4: 128,
}
# The finest native tier (iso_geometry.NATIVE_TIERS) each quality's iso ladder may add, gated
# per quality so a later ceiling raise (TASK-202's 256) cannot admit 192 at Enhanced (TASK-441).
NATIVE_TIER_MAX = {
    1: 96,
    2: 96,
    3: 192,
    4: 96,
}


def finest_tile_pixels() -> int:
    """The finest mip level's tile_px at the current graphics quality."""
    return FINEST_TILE_PIXELS[settings.get_graphics_quality()]


def native_tier_max() -> int:
    """The finest native tier the current graphics quality's iso ladder may add."""
    return NATIVE_TIER_MAX[settings.get_graphics_quality()]


def _crop_offset(x: int, y: int, texture_size: int, tile_px: int) -> tuple[int, int]:
    """Crop origin into a cached terrain texture array for tile (x, y), chosen
    so same-terrain neighboring tiles sample *contiguous* texture regions
    instead of independent, unrelated patches -- tile (x+1, y)'s crop picks up
    exactly where tile (x, y)'s left off. That makes a same-terrain area one
    continuous unrolled tiling of the source texture rather than visibly cut-up
    squares, which works because ground textures are themselves authored to
    tile seamlessly at their own edges. Stepped and Sloped get the same
    continuity only because iso_geometry._inverse_sample reads the whole
    crop along these same axes (column = x, row = y).

    Where tile_px divides texture_size every crop window lands exactly on a
    tile_px boundary and never straddles the texture's own wrap point, so a
    plain slice of the texture is enough. The native tier's 96 on a 512
    texture (TASK-599) does not divide it: its crops straddle the wrap, so
    they slice asset_source.crop_source()'s wrap-padded copy instead, with
    the period still texture_size. Every caller crops from crop_source().

    The large-scale repeat this trades away (the same texture region recurring
    every texture_size/tile_px tiles) is deliberate: for noisy ground textures
    (grass, dirt, ...) with no single distinct landmark, that repeat is far
    less visible than a hard seam at every single tile boundary."""
    assert 0 < tile_px <= texture_size, (
        f"_crop_offset needs tile_px ({tile_px}) within one texture period ({texture_size})"
    )
    return (x * tile_px) % texture_size, (y * tile_px) % texture_size


def _top_block(
    tile, tile_px: int, textures: bool = True, blend=None, terrain_override: int | None = None
) -> np.ndarray:
    """The square tile_px x tile_px top block every terrain style builds
    first and only then places (Flat blits it, Stepped scatters it into the
    diamond and samples skirts from it, Sloped warps it): a real texture
    crop when an AoE2DE install is configured, else terrain_palette's flat
    colour. The one place a tile's own terrain pixels come from, so
    blending reaches all three styles, every level and every patch path
    through here, and Stepped == Sloped on a flat map holds by construction.

    blend: the tile's terrain_blend.influences() key, or None for today's
    plain crop. A farm (terrain_override) never blends: its key describes
    the raw terrain_id it covers, not the farm.

    textures=False (View > Layers) forces the flat-colour branch below. It
    adds no new rendering: it takes the exact path an unconfigured install
    already takes, which is what makes "textures off" provably equal to a
    no-install render rather than a second, drifting code path. The
    returned array may be a read-only view of a cached texture or block."""
    terrain_id = tile.terrain_id if terrain_override is None else terrain_override
    size = asset_source.terrain_texture_size(tile_px)
    texture = asset_source.terrain_texture_at(terrain_id, size) if textures else None
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        # key[0] is the mirror's tid: a stale mirror keeps the live tile's plain crop.
        if blend is not None and terrain_override is None and blend[0] == terrain_id:
            block = terrain_blend.blended_block(blend, ox, oy, tile_px)
            if block is not None:
                return block
        return asset_source.crop_source(texture, tile_px)[oy : oy + tile_px, ox : ox + tile_px]
    r, g, b = color_for_terrain_id(terrain_id)
    return np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.uint8)


def _tile_block(tile, tile_px: int, textures: bool = True, blend=None) -> np.ndarray:
    """Flat's per-tile content: _top_block() with no farm override (a draped
    farm tile calls _top_block() itself, GH #150). Split
    out of render_tile() (Phase B-E) so composite_rect_flat() can reuse the
    exact same per-tile content without going through render_tile()'s own
    img-mutating, tile.x/tile.y-derived placement."""
    return _top_block(tile, tile_px, textures, blend)


def blend_effective(layers: LayerState = DEFAULT_LAYERS, blend: bool = True) -> bool:
    """Whether terrain blending changes any pixel: the View > Layers row and
    the caller's blend flag on, textures on, and the install's blendomatic
    readable. Off, every render and patch path is byte-identical to the
    pre-blending renderer, and Flat's dirty rects need no dilation."""
    return blend and layers.terrain_blend and layers.terrain_textures and asset_source.get_blendomatic() is not None


# flat_tile_rects()' block side and bbox rule. tools/bench_draw_stroke.py, Dos Pilas, mip 0: one composite call
# 0.07 to 0.19 ms, each extra tile in it 2.6 to 6.6 us, so a bbox wins until it covers about 28 times the tiles.
FLAT_PATCH_BLOCK_TILES = 16
FLAT_PATCH_DENSITY = 8


def flat_tile_rects(tiles, tile_px: int) -> list:
    """Canvas-pixel rects covering the tiles `tiles` ((x, y) pairs, deduped
    here), for patch_rects(): per FLAT_PATCH_BLOCK_TILES block, the bbox of
    its tiles as one rect, or one rect a tile where that bbox would cover
    more than FLAT_PATCH_DENSITY times the block's tiles (a sparse scatter).
    Sorted and pairwise disjoint, each inside its own block, so one edit
    composites a tile once per resident level and derives its blend key once."""
    size = FLAT_PATCH_BLOCK_TILES
    blocks: dict[tuple[int, int], list] = {}
    for x, y in set(tiles):
        blocks.setdefault((x // size, y // size), []).append((x, y))
    rects = []
    for members in blocks.values():
        xs, ys = zip(*members, strict=True)
        x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        if (x1 - x0) * (y1 - y0) <= FLAT_PATCH_DENSITY * len(members):
            rects.append((x0 * tile_px, y0 * tile_px, x1 * tile_px, y1 * tile_px))
        else:
            rects.extend((x * tile_px, y * tile_px, (x + 1) * tile_px, (y + 1) * tile_px) for x, y in members)
    return sorted(rects)


def flat_dirty_rects(coords, map_w: int, map_h: int, tile_px: int, layers: LayerState) -> list:
    """Flat's patch_rects() input for the dirty tiles `coords` ((x, y) pairs):
    flat_tile_rects() of them, dilated first to the 8 neighbours when
    blending is effective, since a tile's blend reads them. The rects cover a
    superset of the dirty tiles and their ring: never assume one rect a tile."""
    if blend_effective(layers):
        coords = {
            (nx, ny)
            for x, y in coords
            for nx in range(max(x - 1, 0), min(x + 2, map_w))
            for ny in range(max(y - 1, 0), min(y + 2, map_h))
        }
    return flat_tile_rects(coords, tile_px)


def tile_field_mirror(scenario: LoadedScenario, field_name: str, idx: np.ndarray | None = None) -> np.ndarray:
    """Flat int64 `terrain_id` or `layer` per tile index (y * w + x), over idx
    or the whole map: the mirror a chunk cache keeps and blend keys read."""
    terrain = scenario.map_manager.terrain
    get = attrgetter(field_name)
    if idx is None:
        return np.fromiter(map(get, terrain), dtype=np.int64, count=len(terrain))
    # list.__getitem__ skips UuidList's pure-Python override, halving the read.
    tiles = map(list.__getitem__, repeat(terrain, idx.size), idx.tolist())
    return np.fromiter(map(get, tiles), dtype=np.int64, count=idx.size)


def _blend_keys(
    scenario: LoadedScenario, xs, ys, terrain_ids: np.ndarray | None = None, layer_ids: np.ndarray | None = None
) -> list:
    """terrain_blend.influences() per tile (xs[i], ys[i]), from the caller's
    whole-map flat mirrors, or from fresh ones read here when either is None
    (the tools, tests and refresh_region_iso, which keep no mirror)."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    if terrain_ids is None or layer_ids is None:
        terrain_ids = tile_field_mirror(scenario, "terrain_id")
        layer_ids = tile_field_mirror(scenario, "layer")
    return terrain_blend.influences(terrain_ids.reshape(h, w), layer_ids.reshape(h, w), xs, ys)


def _full_blend_keys(scenario: LoadedScenario, blend: bool) -> list | None:
    """_blend_keys() for every tile, indexed y * w + x, for the full-map
    renderers; None when blending is off (blend_effective())."""
    if not blend_effective(blend=blend):
        return None
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    ys, xs = np.divmod(np.arange(w * h), w)
    return _blend_keys(scenario, xs, ys)


def _rect_blend_keys(
    scenario, candidates: np.ndarray, layers: LayerState, terrain_ids, layer_ids, key_map: terrain_blend.KeyMap | None = None
) -> terrain_blend.RectKeys | None:
    """_blend_keys() for a composite_rect_*'s candidates as RectKeys, or None
    when blending is off or no candidate has a key (the plain path).

    key_map: the calling cache's KeyMap over the same two mirrors, which
    then derives only the tiles it has not seen since their last
    invalidate(), nor in their exact neighbourhood (its stash, bound to the
    mirrors here). Without one, or without both mirrors (tools, tests,
    the review pack, refresh_region_iso), a throwaway map derives every
    candidate, so there is one code path. _blend_keys is looked up at call
    time, so a caller that rebinds it still reaches the fill."""
    if not candidates.shape[0] or not blend_effective(layers):
        return None
    mm = scenario.map_manager
    w = mm.map_width
    if key_map is None or terrain_ids is None or layer_ids is None:
        key_map = terrain_blend.KeyMap(w, mm.map_height)
    else:
        assert terrain_ids.size == layer_ids.size == key_map.ids.size, "a key map sized for another map"
        key_map.bind((terrain_ids, layer_ids, terrain_blend.BLEND_TABLE, _blend_keys), mirrors=(terrain_ids, layer_ids))
    ids = key_map.lookup(
        candidates[:, 1] * w + candidates[:, 0], lambda xs, ys: _blend_keys(scenario, xs, ys, terrain_ids, layer_ids)
    )
    return terrain_blend.RectKeys(ids, key_map) if (ids >= 0).any() else None


def _dims_and_elevations(scenario: LoadedScenario) -> tuple[int, int, int, np.ndarray]:
    """(w, h, tile_px, elevations) -- the grid-free preamble both cheap-init
    entry points need, before each builds its own IsoProjection from it:
    elevations_and_proj() (Stepped/Flat) and sloped_elevations_and_proj().
    Only the preamble is shared; their canvas_size_and_origin() calls are
    not duplicates and stay separate (corner_headroom_steps differs).

    Never cache this, however tempting the lru_cache already imported above
    makes it. A cached return hands the SAME elevations array to two
    callers, and that array is mutated in place downstream -- deliberately
    so, see dirty_screen_bbox_iso()'s own contract -- which is exactly the
    Risk #6 drift the split return values exist to prevent.

    _terrain_grid_and_elevations() below keeps its own copy of these lines
    deliberately: it fills tile_grid in the same pass, so calling this
    instead would mean a second Python loop over up to ~230k tiles, and it
    never needs tile_px.

    Tests patch tile_pixels_for_map through testkit.render_family, which sets
    it in every render module that binds it."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    elevations = np.zeros((h, w), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    return w, h, tile_px, elevations


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


def _clipped_paint(
    img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, values: np.ndarray, *, extent=None
) -> None:
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
    that function's own docstring.

    extent, when given, is iso_geometry.index_extent(producer, *key) for
    the very producer call that made dst_y/dst_x: the INCLUSIVE
    (y_lo, y_hi, x_lo, x_hi) box they occupy in their own unoffset frame.
    It buys a scalar pre-check that skips the per-pixel mask entirely on
    the two decidable cases, wholly inside (scatter straight in) and wholly
    outside (return, matching the existing path's compaction to empty
    arrays, which is a no-op). Anything straddling an edge falls through to
    the mask below, so the accepted top-edge pixel drop SlopedChunkCache.
    canvas_dims documents still happens exactly where it did.

    KEYWORD-ONLY on purpose: no positional caller can bind it by accident,
    and extent=None is today's path byte-for-byte. Passing None from an
    empty index set is correct and expected, since that inert path already
    costs nothing."""
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        h, w = img.shape[0], img.shape[1]
        if base_y + y_lo >= 0 and base_y + y_hi < h and base_x + x_lo >= 0 and base_x + x_hi < w:
            img[base_y + dst_y, base_x + dst_x] = values
            return
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax, values = ay[in_bounds], ax[in_bounds], values[in_bounds]
    img[ay, ax] = values


def _clipped_paint_solid(img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, color, *, extent=None) -> None:
    """_clipped_paint with one (r, g, b) for every pixel: farm outlines and
    unit marks."""
    native = composite_backend.native
    if native is not None:
        r, g, b = color
        native.paint_solid(img, base_y, base_x, dst_y, dst_x, r, g, b)
        return
    _clipped_paint(img, base_y, base_x, dst_y, dst_x, np.full((dst_y.shape[0], 3), color, dtype=np.uint8), extent=extent)


# "No ink here" in a two-pass ink plane (_paint_candidates_iso): above every
# rise, so the occlusion test `ink < height` never fires on it.
INK_NONE = np.iinfo(np.int32).max


def _record_ink(ink: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, rise: int) -> None:
    """ink[base_y+dst_y, base_x+dst_x] = rise, clipped: a mark's pixels."""
    ay = base_y + dst_y
    ax = base_x + dst_x
    ok = (ay >= 0) & (ay < ink.shape[0]) & (ax >= 0) & (ax < ink.shape[1])
    ink[ay[ok], ax[ok]] = rise


def _record_ink_rgba(ink: np.ndarray, base_y: int, base_x: int, rgba: np.ndarray, rise: int) -> None:
    """ink = rise wherever _clipped_paint_rgba's block has alpha > 0. Not only
    opaque pixels: an occluded sprite's shadow and soft edge must go with it."""
    h, w = rgba.shape[:2]
    ih, iw = ink.shape
    sy0, sx0 = max(0, -base_y), max(0, -base_x)
    sy1, sx1 = min(h, ih - base_y), min(w, iw - base_x)
    if sy0 >= sy1 or sx0 >= sx1:
        return
    ink[base_y + sy0 : base_y + sy1, base_x + sx0 : base_x + sx1][rgba[sy0:sy1, sx0:sx1, 3] > 0] = rise


def _restore_under(
    img: np.ndarray, terrain: np.ndarray, ink: np.ndarray, base_y: int, base_x: int, dst_y, dst_x,
    height: int, *, extent=None,
) -> None:
    """Two-pass occlusion over one coverage op: wherever ink based below
    `height` sits on these pixels, the terrain snapshot comes back and the ink
    is cleared. Clipped like _clipped_paint, same extent contract."""
    h, w = ink.shape
    inside = False
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return
        inside = base_y + y_lo >= 0 and base_y + y_hi < h and base_x + x_lo >= 0 and base_x + x_hi < w
    ay = base_y + dst_y
    ax = base_x + dst_x
    if not inside:
        ok = (ay >= 0) & (ay < h) & (ax >= 0) & (ax < w)
        ay, ax = ay[ok], ax[ok]
    hit = ink[ay, ax] < height
    if hit.any():
        ay, ax = ay[hit], ax[hit]
        img[ay, ax] = terrain[ay, ax]
        ink[ay, ax] = INK_NONE


def _clipped_paint_rgba(
    img: np.ndarray, base_y: int, base_x: int, rgba: np.ndarray, ink: np.ndarray | None = None, ink_rise: int = 0
) -> None:
    """Alpha-composites an (h, w, 4) uint8 block onto img at (base_y, base_x),
    clipping to img's bounds -- _clipped_paint()'s counterpart for sprites
    (P3-g3). ink, when given, is the two-pass ink plane: every pixel the block
    touches (alpha > 0) is recorded there at ink_rise.

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
    if ink is not None:
        _record_ink_rgba(ink, base_y, base_x, rgba, ink_rise)
    native = composite_backend.native
    if native is not None:
        native.blit_rgba(img, base_y, base_x, rgba)
        return
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


def _clipped_lerp(
    img: np.ndarray, base_y: int, base_x: int, dst_y, dst_x, target_rgb, alpha, *, extent=None
) -> None:
    """img[base_y+dst_y, base_x+dst_x] lerped towards target_rgb by alpha/255,
    in place and clipped like _clipped_darken. target_rgb and alpha broadcast
    per pixel as (n, 3) and (n, 1), or as a constant (numpy only)."""
    native = composite_backend.native
    if native is not None and np.ndim(target_rgb) == 2 and np.ndim(alpha) == 2:
        native.lerp(img, base_y, base_x, dst_y, dst_x, target_rgb, alpha)
        return
    if extent is not None:
        y_lo, y_hi, x_lo, x_hi = extent
        h, w = img.shape[0], img.shape[1]
        if base_y + y_lo >= 0 and base_y + y_hi < h and base_x + x_lo >= 0 and base_x + x_hi < w:
            ay = base_y + dst_y
            ax = base_x + dst_x
            img[ay, ax] = ((target_rgb * alpha + img[ay, ax] * (255 - alpha)) // 255).astype(np.uint8)
            return
        if base_y + y_hi < 0 or base_y + y_lo >= h or base_x + x_hi < 0 or base_x + x_lo >= w:
            return
    ay = base_y + dst_y
    ax = base_x + dst_x
    in_bounds = (ay >= 0) & (ay < img.shape[0]) & (ax >= 0) & (ax < img.shape[1])
    if not np.all(in_bounds):
        ay, ax = ay[in_bounds], ax[in_bounds]
        if np.ndim(target_rgb) == 2:
            target_rgb = target_rgb[in_bounds]
        if np.ndim(alpha) == 2:
            alpha = alpha[in_bounds]
    # Integer math, same reason as _clipped_paint_rgba: a float round-trip can
    # land a pixel one off and quietly break the byte-identity oracles.
    img[ay, ax] = ((target_rgb * alpha + img[ay, ax] * (255 - alpha)) // 255).astype(np.uint8)


# Grid edge -> tile_edge_indices side: x-low faces (x-1, y), y-low (x, y-1),
# x-high (x+1, y), y-high (x, y+1).
_ISO_GRID_SIDES = {"x_low": "left", "y_low": "up_left", "x_high": "up_right", "y_high": "right"}


def _grid_stamp(pieces, grid: GridBake):
    """(dst_y, dst_x, target (n, 3), alpha (n, 1), extent) from per-edge
    (ys, xs, major) pixel sets, each pixel once and a major winning a tie, so
    no pixel is lerped twice where two edges or two thickness rows meet."""
    empty = np.zeros(0, dtype=np.int64)
    if not pieces:
        return empty, empty, None, None, None
    ys = np.concatenate([p[0] for p in pieces]).astype(np.int64)
    xs = np.concatenate([p[1] for p in pieces]).astype(np.int64)
    major = np.concatenate([np.full(p[0].size, p[2], dtype=bool) for p in pieces])
    if ys.size == 0:
        return empty, empty, None, None, None
    order = np.argsort(~major, kind="stable")
    ys, xs, major = ys[order], xs[order], major[order]
    key = (ys - ys.min()) * (int(xs.max()) + 1) + xs
    _, first = np.unique(key, return_index=True)
    ys, xs, major = ys[first], xs[first], major[first]
    rgba = np.where(major[:, None], np.array(grid.major, dtype=np.uint16), np.array(grid.minor, dtype=np.uint16))
    keep = rgba[:, 3] > 0
    ys, xs, rgba = ys[keep], xs[keep], rgba[keep]
    if ys.size == 0:
        return empty, empty, None, None, None
    extent = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    return ys, xs, np.ascontiguousarray(rgba[:, :3]), np.ascontiguousarray(rgba[:, 3:4]), extent


@lru_cache(maxsize=4096)
def _grid_stamp_iso(tile_px: int, edges: tuple, grid: GridBake, corners: tuple | None = None):
    """One tile's baked grid pixels in its own diamond's frame (Stepped) or
    sloped quad's frame (Sloped, `corners` normalized to d_min 0).

    A thickness of N stamps each 1px edge N times, offset vertically INWARD
    (about 0.89N perpendicular on a 2:1 edge) and clipped to this tile's own
    columns. Outward would spill into a neighbour whose later fill eats it, so
    a thick line sits wholly on the owning side of a shared edge instead of
    straddling it; Stepped and Sloped do it identically, keeping their
    flat-map equality."""
    if corners is None:
        def producer(side):
            return iso_geometry.tile_edge_indices(tile_px, side)
    else:
        def producer(side):
            return iso_geometry.sloped_tile_edge_indices(tile_px, side, *corners)
    tops = [producer("up_left"), producer("up_right")]
    bottoms = [producer("left"), producer("right")]
    ncol = int(max(p[1].max() for p in (*tops, *bottoms))) + 1
    top = np.full(ncol, np.iinfo(np.int64).max, dtype=np.int64)
    bottom = np.full(ncol, np.iinfo(np.int64).min, dtype=np.int64)
    for ey, ex in tops:
        top[ex] = ey
    for ey, ex in bottoms:
        bottom[ex] = ey
    rows = np.arange(grid.thickness, dtype=np.int64)
    pieces = []
    for edge, major in edges:
        side = _ISO_GRID_SIDES[edge]
        ey, ex = producer(side)
        step = -1 if side in ("left", "right") else 1
        ys = (ey[None, :] + step * rows[:, None]).ravel()
        xs = np.broadcast_to(ex, (rows.size, ex.size)).ravel()
        keep = (ys >= top[xs]) & (ys <= bottom[xs])
        pieces.append((ys[keep], xs[keep], major))
    return _grid_stamp(pieces, grid)


@lru_cache(maxsize=256)
def _grid_stamp_flat(tile_px: int, edges: tuple, grid: GridBake):
    """Flat's counterpart to _grid_stamp_iso: each edge is an N-px strip along
    the inside of the tile's own square block."""
    n = min(grid.thickness, tile_px)
    along, across = np.meshgrid(np.arange(tile_px, dtype=np.int64), np.arange(n, dtype=np.int64))
    along, across = along.ravel(), across.ravel()
    pieces = []
    for edge, major in edges:
        if edge == "x_low":
            pieces.append((along, across, major))
        elif edge == "x_high":
            pieces.append((along, tile_px - 1 - across, major))
        elif edge == "y_low":
            pieces.append((across, along, major))
        else:
            pieces.append((tile_px - 1 - across, along, major))
    return _grid_stamp(pieces, grid)


@lru_cache(maxsize=64)
def _grid_lod_edges_filter(minor_spacing_px: float, major_every: int) -> tuple[bool, bool]:
    lod = grid_overlay.bake_lod(minor_spacing_px, major_every)
    return lod.draw_grid, lod.draw_minors


def _grid_edges(
    tx: int, ty: int, map_w: int, map_h: int, minor_spacing_px: float, major_every: int, elevations=None,
) -> tuple:
    draw_grid, draw_minors = _grid_lod_edges_filter(minor_spacing_px, major_every)
    if not draw_grid:
        return ()
    edges = grid_overlay.owned_edges(tx, ty, map_w, map_h, elevations, major_every)
    return edges if draw_minors else tuple(e for e in edges if e[1])


def _paint_grid_iso(
    img, tx: int, ty: int, tile_px: int, map_w: int, map_h: int, base_y: int, base_x: int,
    grid: GridBake, elevations=None, corners=None,
) -> None:
    """One tile's baked grid, drawn at the base its own terrain painted at.
    Stepped passes `elevations` (cliff-lip edges), Sloped passes `corners`."""
    edges = _grid_edges(tx, ty, map_w, map_h, tile_px / 2, grid.major_every, elevations)
    if not edges:
        return
    dst_y, dst_x, target, alpha, extent = _grid_stamp_iso(tile_px, edges, grid, corners)
    if extent is not None:
        _clipped_lerp(img, base_y, base_x, dst_y, dst_x, target, alpha, extent=extent)


def _paint_grid_flat(img, tx: int, ty: int, tile_px: int, map_w: int, map_h: int, base_y: int, base_x: int,
                     grid: GridBake) -> None:
    """Shared by render_tile() and composite_rect_flat(): only this per-tile
    step is shared, like _tile_block, so the full render stays an independent
    oracle for the chunk path."""
    edges = _grid_edges(tx, ty, map_w, map_h, tile_px, grid.major_every)
    if not edges:
        return
    dst_y, dst_x, target, alpha, extent = _grid_stamp_flat(tile_px, edges, grid)
    if extent is not None:
        _clipped_lerp(img, base_y, base_x, dst_y, dst_x, target, alpha, extent=extent)


# Units or keys per step of a sliced level assembly (_LeveledChunkCache._assemble_level).
ASSEMBLY_SLICE = 512
