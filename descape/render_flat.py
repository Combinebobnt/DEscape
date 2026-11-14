"""The Flat render: tiles and terrain, tile refresh, Flat unit rows and icons, the rect composite, overlay_units and refresh_units_over."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from descape import (
    terrain_blend,
    unit_sprites,
)
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render_paint import (
    _clipped_paint_rgba,
    _drain,
    _full_blend_keys,
    _paint_grid_flat,
    _rect_blend_keys,
    _tile_block,
    _top_block,
    blend_effective,
    tile_pixels_for_map,
)
from descape.render_sprites import _paint_farm_outline_flat, _terrain_overlay_for
from descape.render_units import (
    _draw_unit,
    _unit_color,
    player_art_for,
    stored_rotation,
    unit_paint_offset,
    unit_tile_bounds,
    wall_variant_rotation_overrides,
)
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    PLAYER_COLORS,
)
from descape.unit_filter import UnitFilter
from descape.view_layers import DEFAULT_LAYERS, LayerState


def render_tile(
    img: np.ndarray,
    tile,
    tile_px: int,
    grid: GridBake = DEFAULT_GRID,
    map_dims: tuple[int, int] | None = None,
    blend=None,
    farm=None,
) -> None:
    """Paints one tile's tile_px x tile_px block into img, in place. The
    per-tile body shared by render_terrain() (full-map render) and
    refresh_tiles() (incremental, edit-driven redraw) -- kept as one function
    so the two paths can never drift apart on how a tile is textured.

    No elevation-based brightness shading -- removed at the user's explicit
    request (the "glowing" per-tile brightness hint was redundant once
    Stepped mode shows real per-tile Z-height displacement, and actively
    misleading in Flat mode, which has no other elevation cue at all).
    Plain terrain color/texture only. blend: the tile's blend key
    (_top_block()). farm: the tile's flat_farm_tiles() entry, painted as its
    farm terrain plus outline under the grid (GH #150), or None."""
    px0, py0 = tile.x * tile_px, tile.y * tile_px
    if farm is None:
        img[py0 : py0 + tile_px, px0 : px0 + tile_px] = _tile_block(tile, tile_px, blend=blend)
    else:
        img[py0 : py0 + tile_px, px0 : px0 + tile_px] = _top_block(tile, tile_px, blend=blend, terrain_override=farm[0])
        _paint_farm_outline_flat(img, py0, px0, tile_px, farm[1], farm[2])
    if grid.paints and map_dims is not None:
        _paint_grid_flat(img, tile.x, tile.y, tile_px, map_dims[0], map_dims[1], py0, px0, grid)


def render_terrain(
    scenario: LoadedScenario, grid: GridBake = DEFAULT_GRID, blend: bool = True, farm_tiles=None
) -> np.ndarray:
    """Terrain rendered at tile_pixels_for_map()-per-tile resolution
    (adaptive on map size -- see that function). Uses a real per-tile
    texture crop when an AoE2DE install is configured
    (asset_source.get_terrain_texture_array), falling back to
    terrain_palette's flat color per tile otherwise. Returns an
    (H*tile_px, W*tile_px, 3) uint8 array. blend=False keeps the
    pre-blending render (blend_effective()). farm_tiles: flat_farm_tiles()'
    by_tile, the oracle half of composite_rect_flat()'s drape (GH #150)."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(w, h)
    img = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)

    keys = _full_blend_keys(scenario, blend)
    farms = farm_tiles or {}
    for tile in mm.terrain:
        render_tile(
            img, tile, tile_px, grid, (w, h), None if keys is None else keys[tile.y * w + tile.x],
            farms.get((tile.x, tile.y)),
        )
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

    Matches render_terrain(blend=False) only: it paints no terrain blend, so
    dirty_indices needs no dilation to neighboring tiles. With blending a
    tile's pixels also depend on its 8 neighbours and its `layer`; the
    viewer's Flat path dilates through flat_dirty_rects() instead."""
    terrain = scenario.map_manager.terrain
    for i in dirty_indices:
        render_tile(img, terrain[i], tile_px)


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
    chunk's cost otherwise.

    A thin wrapper over _flat_unit_rows(), which also returns each row's unit
    identity and owner for FlatChunkCache's row splice."""
    bboxes, colors, _row_uid, _row_player = _flat_unit_rows(scenario, tile_px, unit_filter)
    return bboxes, colors


def _flat_unit_bbox(bounds: tuple[int, int, int, int], unit, tile_px: int) -> tuple[int, int, int, int]:
    """One unit's pixel-space rect from its unit_tile_bounds() result, shifted
    the same way _draw_unit() shifts it, so the chunked path and the
    full-canvas one draw every unit identically."""
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    dx, dy = unit_paint_offset(unit)
    off_x, off_y = round(dx * tile_px), round(dy * tile_px)
    return (
        tile_x0 * tile_px + off_x, tile_y0 * tile_px + off_y,
        tile_x1 * tile_px + off_x, tile_y1 * tile_px + off_y,
    )


def _flat_unit_rows(
    scenario: LoadedScenario, tile_px: int, unit_filter: UnitFilter = UnitFilter()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_flat_unit_draws()' walk, plus two side columns per row: row_uid,
    id(unit) as int64, and row_player, the owning player as int8. Rows are in
    player order, so row_player is non-decreasing, which is what lets
    FlatChunkCache.invalidate_units() find the end of a player's block with
    one searchsorted. One walk (_flat_unit_base_rows()), so the side columns
    can't disagree with the bboxes about which units have rows."""
    tiles, offsets, colors, row_uid, row_player = _flat_unit_base_rows(scenario, unit_filter)
    return flat_unit_bboxes(tiles, offsets, tile_px), colors, row_uid, row_player


def _flat_unit_base_rows(
    scenario: LoadedScenario, unit_filter: UnitFilter = UnitFilter()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The mip-independent half of _flat_unit_rows()' walk: (tiles (N,4)
    int32 in unit_tile_bounds()' (x0, x1, y0, y1) order, offsets (N,2)
    float64 from unit_paint_offset(), colors (N,3) uint8, row_uid int64,
    row_player int8). flat_unit_bboxes() turns the first two into any
    level's bboxes, so FlatChunkCache walks the units once per wholesale
    rebuild rather than once per level."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    tiles = []
    offsets = []
    colors = []
    row_uid = []
    row_player = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for unit in units:
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tiles.append(bounds)
            offsets.append(unit_paint_offset(unit))
            colors.append(_unit_color(unit, player_color))
            row_uid.append(id(unit))
            row_player.append(player_id)
    if not tiles:
        return (
            np.zeros((0, 4), dtype=np.int32), np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8),
        )
    return (
        np.array(tiles, dtype=np.int32), np.array(offsets, dtype=np.float64),
        np.array(colors, dtype=np.uint8), np.array(row_uid, dtype=np.int64), np.array(row_player, dtype=np.int8),
    )


def flat_unit_bboxes(tiles: np.ndarray, offsets: np.ndarray, tile_px: int) -> np.ndarray:
    """_flat_unit_bbox() for every row of _flat_unit_base_rows()' columns at
    once, as (N,4) int32 (x0, y0, x1, y1). np.rint rounds half to even, as
    round() does on a float, so the two agree on every row."""
    t = tiles.astype(np.int64) * tile_px
    off = np.rint(offsets * tile_px).astype(np.int64)
    ox, oy = off[:, 0], off[:, 1]
    return np.stack((t[:, 0] + ox, t[:, 2] + oy, t[:, 1] + ox, t[:, 3] + oy), axis=1).astype(np.int32)


def _flat_icon_layer(
    scenario: LoadedScenario, tile_px: int, unit_filter: UnitFilter = UnitFilter(), with_farms: bool = False
) -> tuple[dict[int, unit_sprites.SpriteDraw], int]:
    """Flat's real .sld icons (P3-g7), keyed by ROW INDEX into the bboxes
    array _flat_unit_draws() returns for the same arguments, plus that walk's
    own ROW COUNT. A unit that resolves nowhere is simply absent from the
    dict, and its coloured rect stays -- so this layer is strictly additive,
    exactly as the iso SpriteLayer is.

    **This walk and _flat_unit_draws()' are REQUIRED to stay in lockstep** --
    same per-player loop, same unit_filter gate, same unit_tile_bounds() is
    None drop -- or every row index past the first divergence keys an icon
    onto the wrong unit. The row count is returned for exactly that reason:
    FlatChunkCache._level_icons() asserts it EQUALS the draws' row count,
    which catches the likely divergence (a new `continue` that skips a unit
    without advancing `row`, shifting every later icon onto its neighbour
    while staying in range) and not merely an index past the end. The repo
    already treats this row index as a stable unit identity (unit_pick.py
    builds its own index in "EXACTLY _flat_unit_draws()'s order").

    Not a widening of _flat_unit_draws()' (bboxes, colors) tuple, deliberately:
    a third column would break four unpack sites in tests for no gain, and a
    side table keeps composite_rect_flat()'s icons=None path byte-identical to
    the pre-sprite one by construction rather than by measurement.

    Footprint size comes from the CLAMPED bounds, matching the rect
    _flat_unit_draws() actually paints, so a building hanging off a map edge
    fits what is on screen rather than what it would be inland.

    with_farms (GH #150): a draped unit with no icon gets FLAT_DRAPED, so
    its rect gives way to the farm drape composite_rect_flat() paints."""
    return _drain(_flat_icon_layer_sliced(scenario, tile_px, unit_filter, with_farms))


def _flat_icon_layer_sliced(
    scenario: LoadedScenario, tile_px: int, unit_filter: UnitFilter = UnitFilter(), with_farms: bool = False
) -> Generator[None, None, tuple[dict[int, unit_sprites.SpriteDraw], int]]:
    """_flat_icon_layer() as a resumable generator -- Flat's counterpart to
    sprite_draws_by_anchor_sliced(), same shape and same reasons (see that
    one's docstring). Returns _flat_icon_layer()'s whole (icons, row) tuple,
    row count included: that count is what FlatChunkCache._level_icons()
    asserts against the draws' own, so dropping it here would quietly disarm
    the row-desync guard on the warm path only."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    overrides = wall_variant_rotation_overrides(scenario)
    icons: dict[int, unit_sprites.SpriteDraw] = {}
    row = 0
    for player_id, units in enumerate(scenario.unit_manager.units):
        team_index = scenario.team_indices[player_id]
        for i, unit in enumerate(units):
            yield
            if not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            # `i` is the unit's position in the PLAYER'S full unit list (this
            # loop's own enumerate), not `row`, which only advances past units
            # unit_filter keeps.
            draw = _flat_unit_icon(
                unit, player_id, i, overrides, team_index, bounds, tile_px,
                player_art_for(scenario, player_id), with_farms,
            )
            if draw is not None:
                icons[row] = draw
            row += 1
    return icons, row


FLAT_DRAPED = unit_sprites.SpriteDraw(np.zeros((0, 0, 4), dtype=np.uint8), 0, 0)
"""_flat_unit_icon()'s answer for a farm drawn as Flat terrain (GH #150): an
icon-layer entry, compared by identity, that paints nothing in place of the
unit's rect, so the drape under it shows."""


def _flat_unit_icon(
    unit,
    player_id: int,
    index: int,
    overrides: dict[tuple[int, int], float],
    team_index: int,
    bounds: tuple[int, int, int, int],
    tile_px: int,
    art: tuple[int | None, int, int | None] | None = None,
    with_farms: bool = False,
) -> unit_sprites.SpriteDraw | None:
    """One unit's Flat icon, the per-unit body of _flat_icon_layer_sliced()
    and of FlatChunkCache's row splice, shared so the two cannot drift. index
    is the unit's position in its player's full list, the key `overrides`
    (wall_variant_rotation_overrides()) uses. Footprint size comes from the
    CLAMPED bounds, matching the rect _flat_unit_bbox() paints. `art` is the
    owner's player_art_for() triple (GH #48). with_farms: FLAT_DRAPED for a
    draped const with no icon, where None would keep its rect."""
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    rotation = stored_rotation(player_id, unit)
    # A wall/gate's shape isn't always recoverable from its own stored
    # rotation; see wall_variant_rotation_overrides()'s docstring.
    rotation = overrides.get((player_id, index), rotation)
    icon = unit_sprites.icon_for(
        unit.unit_const,
        rotation,
        team_index,
        (tile_x1 - tile_x0) * tile_px,
        (tile_y1 - tile_y0) * tile_px,
        getattr(unit, "reference_id", None),
        art,
    )
    if icon is None and with_farms and _terrain_overlay_for(unit.unit_const) is not None:
        return FLAT_DRAPED
    return icon


def composite_rect_flat(
    scenario: LoadedScenario,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tile_px: int,
    unit_draws: tuple[np.ndarray, np.ndarray] | None = None,
    with_units: bool = True,
    icons: dict[int, unit_sprites.SpriteDraw] | None = None,
    layers: LayerState = DEFAULT_LAYERS,
    grid: GridBake = DEFAULT_GRID,
    terrain_ids: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
    key_map: terrain_blend.KeyMap | None = None,
    farm_tiles: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] | None = None,
) -> np.ndarray:
    """Composites the half-open canvas rect [x0, x1) x [y0, y1) for Flat
    mode in isolation and returns it as a fresh (y1-y0, x1-x0, 3) uint8
    array -- Phase B-E's rect-keyed core, the Flat counterpart to
    composite_rect_iso() (see that function's docstring for the shared
    "same function composites every caller's rect, so pixels can never
    depend on request order" property this one inherits).

    Correctness argument (why this needs none of refresh_units_over()'s
    fixed-point dirty-tile expansion): overlay_units() is a sequence of
    axis-aligned rect draws onto a fully-painted terrain
    canvas, so restricting the OUTPUT RECT commutes with that draw
    sequence -- a unit whose bbox misses [x0,x1)x[y0,y1) contributes zero
    pixels inside it, so omitting it from the sum below cannot change the
    result at any pixel that IS in range, regardless of where that unit
    sits in the draw order.

    **Alpha does not weaken that argument, which is what P3-g7 rests on.**
    An icon's pixels are a function of the unit's own FOOTPRINT bbox alone,
    never of the requested rect: it is painted at its full footprint
    position through _clipped_paint_rgba() and the rect does the clipping,
    so it lands exactly where a full render would put it. What alpha does
    change is that overlapping units no longer fully occlude one another,
    making draw ORDER observable where opaque rects hid it -- and that order
    is already preserved, since np.nonzero returns ascending row indices,
    which is overlay_units()' own order. refresh_units_over()'s fixed-point expansion
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

    icons, if given, must be _flat_icon_layer()'s return value for those SAME
    arguments (P3-g7): a row index into unit_draws' bboxes -> that unit's
    footprint-fitted .sld icon, which REPLACES its coloured rect rather than
    overlaying it (a mark left under a mostly-transparent icon shows as a
    fringe -- the same reason the iso path has SpriteLayer.skip_ids). Rows
    with no entry keep the rect, so an unresolvable unit, a missing install
    and an undecodable frame all degrade to today's render. With icons=None
    the painted code path below is unchanged bytes, which is what makes
    "sprites off is byte-identical" a structural property rather than a
    measured one.

    layers: the terrain-textures and blend halves are read here; the farm
    layer arrives already applied, as farm_tiles and FLAT_DRAPED icons.
    Blending reads terrain_ids/layer_ids/key_map as composite_rect_iso() does.

    farm_tiles (GH #150): flat_farm_tiles()' by_tile. Each listed tile paints
    its farm terrain, then its 1px outline, then the grid, all in the tile
    loop, so the grid and every unit land over it. None is today's bytes.

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
    paint_grid = grid.paints
    keys = None
    if tx1 > tx0 and ty1 > ty0 and blend_effective(layers):
        ys, xs = np.mgrid[ty0:ty1, tx0:tx1]
        tiles = np.stack([xs.ravel(), ys.ravel()], axis=1)
        blend = _rect_blend_keys(scenario, tiles, layers, terrain_ids, layer_ids, key_map)
        keys = None if blend is None else blend.as_list()
    farms = farm_tiles or None
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            tile = mm.get_tile(tx, ty)
            key = None if keys is None else keys[(ty - ty0) * (tx1 - tx0) + tx - tx0]
            farm = None if farms is None else farms.get((tx, ty))
            if farm is None:
                block = _tile_block(tile, tile_px, layers.terrain_textures, key)
            else:
                block = _top_block(tile, tile_px, layers.terrain_textures, key, terrain_override=farm[0])
            _blit_clipped(out, block, tx * tile_px - x0, ty * tile_px - y0)
            # Units are a separate pass below, so anywhere in this loop is
            # "between terrain and units". Into `out`, never the block: it is
            # a view of the cached texture.
            if farm is not None:
                _paint_farm_outline_flat(out, ty * tile_px - y0, tx * tile_px - x0, tile_px, farm[1], farm[2])
            if paint_grid:
                _paint_grid_flat(out, tx, ty, tile_px, tile_w, tile_h, ty * tile_px - y0, tx * tile_px - x0, grid)

    if with_units and unit_draws is not None:
        bboxes, colors = unit_draws
        if len(bboxes):
            hits = np.nonzero(
                (bboxes[:, 0] < x1) & (bboxes[:, 2] > x0) & (bboxes[:, 1] < y1) & (bboxes[:, 3] > y0)
            )[0]
            for i in hits:
                bx0, by0, bx1, by1 = (int(v) for v in bboxes[i])
                draw = None if icons is None else icons.get(int(i))
                if draw is FLAT_DRAPED:
                    continue  # its farm_tiles drape is the whole draw
                if draw is None:
                    ix0, iy0 = max(bx0, x0) - x0, max(by0, y0) - y0
                    ix1, iy1 = min(bx1, x1) - x0, min(by1, y1) - y0
                    out[iy0:iy1, ix0:ix1] = colors[i]
                else:
                    # Centred on the WHOLE footprint (bx*/by*), never on its
                    # intersection with this rect -- centring on the clipped
                    # rect would recentre the icon per chunk and produce a
                    # seam-dependent image. _clipped_paint_rgba does the
                    # clipping; tests/test_flat_sprites.py's stitched-chunk
                    # check is what catches getting this wrong.
                    ih, iw = draw.rgba.shape[:2]
                    _clipped_paint_rgba(
                        out,
                        by0 + (by1 - by0 - ih) // 2 - y0,
                        bx0 + (bx1 - bx0 - iw) // 2 - x0,
                        draw.rgba,
                    )
    return out


def _painted_tile_bounds(unit, tile_w: int, tile_h: int) -> tuple[int, int, int, int] | None:
    """unit_tile_bounds() grown by one tile on whichever side the unit's mark
    is shifted toward (off-grid placement, Stage 1), clamped to the map.

    Grown, never moved: the invariant unit_tile_bounds' docstring names --
    (int(unit.x), int(unit.y)) is inside the returned bounds -- has three
    downstream consumers and must survive. This is a strictly wider box, so it
    does.
    """
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    dx, dy = unit_paint_offset(unit)
    if dx < 0:
        tile_x0 = max(0, tile_x0 - 1)
    elif dx > 0:
        tile_x1 = min(tile_w, tile_x1 + 1)
    if dy < 0:
        tile_y0 = max(0, tile_y0 - 1)
    elif dy > 0:
        tile_y1 = min(tile_h, tile_y1 + 1)
    return tile_x0, tile_x1, tile_y0, tile_y1


def overlay_units(
    img: np.ndarray,
    scenario: LoadedScenario,
    unit_filter: UnitFilter = UnitFilter(),
    with_sprites: bool = False,
    with_farms: bool = False,
) -> np.ndarray:
    """Draws a colored dot per unit -- see _draw_unit() for the per-unit
    rules. Mutates and returns img.

    with_farms (GH #150, with with_sprites): a draped const with no icon
    draws nothing here, its drape being render_terrain()'s farm_tiles.

    unit_filter is threaded here for the HEADLESS path only (render_scenario
    -> save_png, tools/dump_scenario.py): the viewer composites through
    composite_rect_flat()'s precomputed unit_draws, never through this
    function. refresh_units_over() deliberately does NOT gain the parameter
    -- it has no live callers at all today, and adding one to dead code
    would just be noise.

    with_sprites (P3-g7) resolves each unit's icon INLINE from this loop,
    deliberately NOT by calling _flat_icon_layer(). This function and
    composite_rect_flat() are two independent implementations on purpose --
    see that function's own docstring on why sharing one would make the
    byte-identity check a tautology -- so the icon logic lands twice, and this
    is the half that serves as the oracle."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    tile_px = tile_pixels_for_map(tile_w, tile_h)
    assert img.shape[:2] == (tile_h * tile_px, tile_w * tile_px), (
        f"img shape {img.shape[:2]} doesn't match this scenario's expected "
        f"{(tile_h * tile_px, tile_w * tile_px)} at tile_px={tile_px} -- was "
        f"it rendered by render_terrain() for a different scenario?"
    )
    out = img.copy()
    # Computed once here, not inside _flat_icon_for_unit -- that helper is
    # deliberately a second, independent implementation of the icon logic
    # (see this function's own docstring), but the connectivity table itself
    # is a pure function of scenario unit data and is meant to be shared
    # across all three rendering paths; see wall_variant_rotation_overrides().
    overrides = wall_variant_rotation_overrides(scenario) if with_sprites else {}

    for player_id, units in enumerate(scenario.unit_manager.units):
        player_color = scenario.player_colors[player_id]
        for i, unit in enumerate(units):
            if not unit_filter.matches(player_id, unit):
                continue
            icon = None
            if with_sprites:
                # team_indices is read INSIDE this branch on purpose: the
                # default sprite-free path must not start touching an
                # attribute the older duck-type test fixtures may not carry.
                icon = _flat_icon_for_unit(
                    scenario, unit, player_id, scenario.team_indices[player_id], tile_w, tile_h, tile_px,
                    overrides.get((player_id, i)),
                )
                if icon is None and with_farms and _terrain_overlay_for(unit.unit_const) is not None:
                    continue
            _draw_unit(out, unit, player_color, tile_w, tile_h, tile_px, icon)
    return out


def _flat_icon_for_unit(
    scenario: LoadedScenario, unit, player_id: int, team_index: int, tile_w: int, tile_h: int, tile_px: int,
    rotation_override: float | None = None,
) -> unit_sprites.SpriteDraw | None:
    """One unit's Flat icon, for overlay_units()' inline oracle path. Kept out
    of the loop body only for readability -- it deliberately does NOT share a
    walk with _flat_icon_layer(), whose whole value is being a second,
    independent implementation. rotation_override, when not None, is this
    unit's connectivity-derived wall/gate shape from
    wall_variant_rotation_overrides() -- computed once by the caller, shared
    with the other two rendering paths despite the walk itself staying
    independent; see overlay_units()'s docstring."""
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    rotation = stored_rotation(player_id, unit)
    if rotation_override is not None:
        rotation = rotation_override
    return unit_sprites.icon_for(
        unit.unit_const,
        rotation,
        team_index,
        (tile_x1 - tile_x0) * tile_px,
        (tile_y1 - tile_y0) * tile_px,
        getattr(unit, "reference_id", None),
        player_art_for(scenario, player_id),
    )


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
    more such scans, but stays cheap in practice since PAINTED-tile overlaps
    are local -- it terminates as soon as a pass adds no new tiles.

    "Painted", not "footprint", since off-grid placement's Stage 1: the expansion
    walks _painted_tile_bounds(), which is a unit's footprint grown by the
    tile its mark is shifted toward. So a straddling mark drags its neighbour
    into the dirty set and cascades through that tile's own units, which is
    the direction to err in -- a missed repaint leaves a stale mark."""
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
    # Widened by the paint offset (off-grid placement, Stage 1): a span-1 mark now
    # spills into the neighbouring tile it is shifted toward, so the plain
    # footprint would under-report which units this dirty set actually
    # repaints -- and a missed repaint here leaves a stale mark behind.
    all_units = [
        (unit, PLAYER_COLORS[player_id % len(PLAYER_COLORS)], _painted_tile_bounds(unit, tile_w, tile_h))
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
