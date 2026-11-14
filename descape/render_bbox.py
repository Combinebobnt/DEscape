"""The render's screen bboxes: building bboxes, dirty screen bboxes, BystanderGrid, rect candidates and prepare_rect_native."""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from descape import (
    composite_backend,
    iso_geometry,
    terrain_blend,
    unit_sprites,
)
from descape.grid_overlay import GridBake
from descape.render_paint import ASSEMBLY_SLICE, _drain, _rect_blend_keys
from descape.render_units import (
    NON_BUILDING_SPAN,
    _unit_screen_bbox_iso,
    anchor_tiles,
    unit_own_tile_index,
    unit_paint_offset,
    unit_tile_bounds,
)
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    tile_span,
)
from descape.unit_filter import UnitFilter
from descape.view_layers import LayerState


def _sprite_reach_px(proj: iso_geometry.IsoProjection) -> tuple[int, int, int, int]:
    """(left, up, right, down) canvas-pixel padding a dirty tile's own bbox
    needs so a sprite anchored on it cannot escape.

    The tile radius above is no substitute, and not because it is slightly too
    small: it models the terrain diamond, skirt and contact shadow, and no unit
    PIXEL extent at all. Sprites reach 404/700/424/316 native px from their
    hotspot (unit_sprites.sprite_reach(): MAX_SPRITE_REACH_* widened by any
    active mod art), which the elevation sweep
    happens to cover horizontally and badly under-covers upward.

    Integer ceil of reach * sprite_scale, plus 1px: sprite_for() rounds the
    scaled width and the scaled hotspot INDEPENDENTLY, so the scaled reach on
    the far side can come out one pixel past (w - hx) * scale."""
    scale = unit_sprites.sprite_scale(proj.half_w)
    left, right, up, down = unit_sprites.sprite_reach()
    return tuple(math.ceil(reach * scale) + 1 for reach in (left, up, right, down))


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
    scenario: LoadedScenario,
    w: int,
    h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
    unit_filter: UnitFilter = UnitFilter(),
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

    **Iterates units directly, at their own tile only** (Batch D's D7 --
    before this, the caller's units_by_tile bucketed a unit into every
    footprint tile, so a naive pass over its items would recompute
    _unit_screen_bbox_iso() once per footprint tile -- 64 times for a
    Colosseum instead of once -- on the path whose ~15-20ms rebuild is
    already the expensive one here; a since-removed own-tile gate restored
    the single computation but still paid the other 63 buckets' worth of
    dict/list walking to reach it). Visiting scenario.unit_manager.units
    directly costs exactly one visit per unit, full stop, and produces the
    identical dict: unit_filter here is the same UnitFilter every caller
    already threads through _units_by_tile() for this same elevations
    snapshot, so filtering happens once, in the one place it's needed.

    extra_top_px is passed straight through to _unit_screen_bbox_iso() --
    see that function for why only Sloped ever passes a nonzero value, and
    why it is a top-edge-only widening.

    A drain of _building_bboxes_iso_sliced(), the only copy of the loop."""
    return _drain(_building_bboxes_iso_sliced(scenario, w, h, proj, elevations, extra_top_px, unit_filter))


def _building_bboxes_iso_sliced(
    scenario: LoadedScenario,
    w: int,
    h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
    unit_filter: UnitFilter = UnitFilter(),
) -> Generator[None, None, dict[tuple[int, int], tuple[int, int, int, int]]]:
    """_building_bboxes_iso() as a resumable generator: at most ASSEMBLY_SLICE
    units per step, filtered-out ones included. The dict is the return value."""
    step = ASSEMBLY_SLICE
    seen = 0
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if seen and seen % step == 0:
                yield
            seen += 1
            if not unit_filter.matches(player_id, unit):
                continue
            bbox = _building_bbox_for(unit, w, h, proj, elevations, extra_top_px)
            if bbox is None:
                continue
            key = (int(unit.x), int(unit.y))
            prior = out.get(key)
            out[key] = bbox if prior is None else (
                min(prior[0], bbox[0]),
                min(prior[1], bbox[1]),
                max(prior[2], bbox[2]),
                max(prior[3], bbox[3]),
            )
    return out


def _building_bbox_for(
    unit,
    w: int,
    h: int,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    extra_top_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """One unit's own contribution to _building_bboxes_iso(), before the
    per-own-tile union: its _unit_screen_bbox_iso(), or None for a unit that
    contributes nothing (a centred single-tile object, or off-map). The
    filter is the caller's job. Extracted so the chunk caches' unit splices
    use the same skip predicate as the wholesale walk, not a copy of it."""
    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    if span_x <= 1 and span_y <= 1 and unit_paint_offset(unit) == (0.0, 0.0):
        # A CENTRED single-tile object only ever paints its own tile,
        # so it can't make a neighbour a bystander. Written as a span
        # test rather than a sentinel comparison so it stays true by
        # construction -- which is what let cliffs join the multi-tile
        # side of it just by gaining an OBJECT_TILE_SPANS entry.
        #
        # The offset condition is off-grid placement's Stage 1: an
        # off-centre mark straddles a tile boundary, so the premise
        # above stops holding for it and it needs a real bbox. Skipping
        # it would clip that mark at a chunk edge in every chunked
        # render while a full-canvas one looked fine.
        return None
    return _unit_screen_bbox_iso(unit, w, h, proj, elevations, extra_top_px)


def _observed_elevation_range(
    elevations: np.ndarray, pre_elevation: dict[tuple[int, int], int], xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) per tile (xs[i], ys[i]): the min and max over its 3x3
    neighbourhood (clipped at the map edge) of every elevation that
    neighbourhood held before or after the edit. `elevations` is post-edit;
    pre_elevation holds the old value of each tile the edit moved.

    3x3 is what a tile's painted extent reads. Stepped: its skirts drop to
    (x-1, y) and (x, y+1), its shadows and seams read (x, y-1), (x+1, y) and
    (x+1, y-1), and its units and sprites sit at its own height. Sloped: its
    four corners each blend the tiles touching that vertex, and a unit or
    sprite on it interpolates those same corners."""
    h, w = elevations.shape
    rx0, ry0 = max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1)
    rx1, ry1 = min(w, int(xs.max()) + 2), min(h, int(ys.max()) + 2)
    lo = elevations[ry0:ry1, rx0:rx1].copy()
    hi = lo.copy()
    for (px, py), pre in pre_elevation.items():
        if rx0 <= px < rx1 and ry0 <= py < ry1:
            lo[py - ry0, px - rx0] = min(lo[py - ry0, px - rx0], pre)
            hi[py - ry0, px - rx0] = max(hi[py - ry0, px - rx0], pre)
    # Edge padding only ever repeats a real neighbour, so min/max over it is
    # the map-edge clip. Every queried tile is 1 inside the region otherwise.
    rh, rw = lo.shape
    lo_p, hi_p = np.pad(lo, 1, mode="edge"), np.pad(hi, 1, mode="edge")
    windows = [(dy, dx) for dy in range(3) for dx in range(3)]
    lo3 = np.minimum.reduce([lo_p[dy : dy + rh, dx : dx + rw] for dy, dx in windows])
    hi3 = np.maximum.reduce([hi_p[dy : dy + rh, dx : dx + rw] for dy, dx in windows])
    return lo3[ys - ry0, xs - rx0], hi3[ys - ry0, xs - rx0]


# Dirty sets at least this large take _dirty_screen_bbox()'s array path
# (_dirty_seed_mask). Its fixed cost is a map-sized mask: break-even was ~81
# tiles on old-allies and June (2026-10-07), so a brush-9 step stays on the loop.
BBOX_ARRAY_PATH_MIN_TILES = 256


def _dirty_seed_mask(
    mm, dirty_indices, elevations: np.ndarray, proj: iso_geometry.IsoProjection, flatten_elevations: bool,
    elevation_changed: set, pre_elevation: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """_dirty_screen_bbox()'s per-tile part over index arrays: writes
    `elevations` and fills elevation_changed/pre_elevation exactly as its loop
    does, then returns (seed mask, dirty xs, dirty ys), the mask being the
    dirty tiles dilated by 1 and clipped to the map. None for an empty set,
    or for a tile outside proj's range (after the writes, as the loop).

    Tile i is (i % w, i // w), the library's own i_to_xy(); each tile's
    elevation is one property read rather than the loop's several."""
    w, h = mm.map_width, mm.map_height
    idx = np.unique(np.fromiter(dirty_indices, dtype=np.int64, count=len(dirty_indices)))
    if idx.size == 0:
        return None
    xs, ys = idx % w, idx // w
    if flatten_elevations:
        elevations[ys, xs] = 0
    else:
        terrain = mm.terrain
        new = np.fromiter((terrain[i].elevation for i in idx.tolist()), dtype=np.int64, count=idx.size)
        pre = elevations[ys, xs]
        moved = pre != new
        tiles = list(zip(xs[moved].tolist(), ys[moved].tolist(), strict=True))
        elevation_changed.update(tiles)
        pre_elevation.update(zip(tiles, pre[moved].tolist(), strict=True))
        elevations[ys, xs] = new
    written = elevations[ys, xs]
    if ((written < proj.min_elev) | (written > proj.max_elev)).any():
        return None
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[ys + 1, xs + 1] = True
    mask = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            mask |= padded[dy : dy + h, dx : dx + w]
    return mask, xs, ys


def _units_owning_tiles(scenario, tiles) -> list:
    """Every unit whose own `(int(u.x), int(u.y))` is in `tiles`, over all
    players and ignoring any filter, from unit_own_tile_index().

    Exactly the set a walk over scenario.unit_manager.units testing that own
    tile would keep: the index is keyed on the same expression, covers every
    player's list unfiltered, and is memoized on unit_gen, which every
    UnitEditModel mutator bumps and no terrain or elevation edit touches.
    tests/test_dirty_bbox_unit_index.py keeps that walk as its oracle."""
    own_index = unit_own_tile_index(scenario)
    return [u for t in tiles for _, _, u in own_index.get(t, ())]


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
    flatten_elevations: bool = False,
    extra_anchor_tiles: set[tuple[int, int]] | None = None,
    units_changed: bool = False,
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
    dilates its trigger set (see units_changed) before seeding itself. 0 (the
    default) uses it bare, exact for Stepped -- see that block's own comment
    for why only an edit to a sprite's OWN centre tile can move it there.
    Sloped's anchor instead reads its own tile's four CORNERS, each shared
    with up to four tiles, so an edit to any tile in a changed tile's 3x3
    neighbourhood can move a Sloped sprite; its caller passes 1 instead (F2).

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
    all, which the dirty set itself (terrain edits included) can't answer.

    flatten_elevations (Flat+Isometric plan, F2): Flat+Iso forces every
    tile's elevation to 0 so terrain reads as a real iso projection at
    elevation 0, not the scenario's real heights. Without this, an edited
    tile would silently re-raise itself to its real elevation on the next
    write below while every other tile stayed flat -- the array (shared,
    per Risk #6, with MapView and the pick plane) would corrupt out from
    under the mode rather than fail loudly. When set, the write loop below
    stores 0 instead of the tile's real elevation, which also means no
    tile's elevation ever "changes" here, so elevation_changed_local stays
    empty and the caller's cache-refresh gen bump is correctly skipped.

    extra_anchor_tiles (Batch D's D3): a unit edit's PRE-EDIT footprint
    tiles, unioned into anchor_tiles before the with_sprites band's
    `dilated & anchor_tiles` intersection below. anchor_tiles(scenario) is
    read live, i.e. POST-edit, so a unit's OLD anchor is never in it by the
    time this runs -- without this, a moved unit's old sprite fragment
    would never get padded into the bbox and would be left stale on
    screen. The caller (ViewerWindow._after_unit_mutation's D5 wiring) is
    responsible for also including these same tiles' indices in
    dirty_indices, since that is what gets them into `dilated` in the
    first place; this parameter only keeps them from being dropped by the
    anchor-tiles intersection once they're there. None (the default) is
    every non-unit-edit caller's exact prior behavior. Only matters when
    units_changed is set: without it the band never seeds from dirty_xy, so
    an old footprint tile whose elevation didn't move is never in `dilated`.

    units_changed (sprite-band terrain-only gate): whether this edit may have
    added, moved, removed or re-resolved a unit. When set, the with_sprites
    band seeds from every dirty tile, since a unit edit moves a sprite with
    no elevation change at all. When clear (the default), it seeds from
    elevation_changed_local alone: the only other thing that moves a sprite
    is its anchor's height, and a terrain-only repaint under one is already
    covered by the bystander index. Explicit rather than inferred from
    extra_anchor_tiles, which is an empty set for a pure add, and required
    under flatten_elevations, where elevation_changed_local is always empty.
    ViewerWindow._unit_edit_bbox() is the one caller that passes True."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height

    # Computed locally and unconditionally now (draw-perf seed-dilation plan
    # Step 3, fact 3): the seed union below needs this set itself, not just
    # whatever a caller wanted for its own cache-refresh decision. Aliased
    # to the caller's set when given, rather than copied, so this is still
    # the only bookkeeping either use needs.
    elevation_changed_local = set() if elevation_changed is None else elevation_changed
    # The old value itself, kept beside the set rather than in it: the set is
    # aliased to the caller's and its type is part of that contract.
    pre_elevation: dict[tuple[int, int], int] = {}

    # A large dirty set takes the array path (_dirty_seed_mask), same result.
    seed_mask = None
    if len(dirty_indices) >= BBOX_ARRAY_PATH_MIN_TILES:
        out = _dirty_seed_mask(mm, dirty_indices, elevations, proj, flatten_elevations, elevation_changed_local,
                               pre_elevation)
        if out is None:
            return None
        seed_mask, dirty_xs, dirty_ys = out
        dirty_xy = set(zip(dirty_xs.tolist(), dirty_ys.tolist(), strict=True)) if units_changed and with_sprites else None
    else:
        dirty_xy = {(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices}
        if not dirty_xy:
            return None

        if not flatten_elevations:
            for x, y in dirty_xy:
                pre = int(elevations[y, x])
                if pre != mm.get_tile(x, y).elevation:
                    elevation_changed_local.add((x, y))
                    pre_elevation[(x, y)] = pre

        for x, y in dirty_xy:
            elevations[y, x] = 0 if flatten_elevations else mm.get_tile(x, y).elevation

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
    # runs. The with_sprites band below gates the same way unless the
    # caller says units_changed.
    #
    # Dilates the (small) TRIGGER set, not a per-anchor ring scan over every
    # unit. The with_sprites block below dilates its own trigger set the same
    # way for its band (Batch A6). The triggered units come from the own-tile
    # index, not a walk over every unit (TASK-031.56, _units_owning_tiles).
    # Footprint tile -> the own tiles of units drawn on it. A footprint is
    # drawn at its unit's own-tile height, which the footprint tile's own
    # neighbourhood can be too far away to see.
    footprint_owners: dict[tuple[int, int], set[tuple[int, int]]] = {}
    if with_units and elevation_changed_local:
        triggered = set(elevation_changed_local)
        for x, y in elevation_changed_local:
            for dx in range(-unit_band_radius, unit_band_radius + 1):
                for dy in range(-unit_band_radius, unit_band_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        triggered.add((nx, ny))
        for u in _units_owning_tiles(scenario, triggered):
            bounds = unit_tile_bounds(u, w, h)
            if bounds is None:
                continue
            fx0, fx1, fy0, fy1 = bounds
            own = (int(u.x), int(u.y))
            if seed_mask is not None:
                seed_mask[fy0:fy1, fx0:fx1] = True
            for fx in range(fx0, fx1):
                for fy in range(fy0, fy1):
                    if seed_mask is None:
                        seed.add((fx, fy))
                    footprint_owners.setdefault((fx, fy), set()).add(own)

    # Union screen bbox, each seed tile swept over the elevations it was
    # actually painted at, before or after the edit (bbox-elev-sweep plan).
    # Not its own pre/post pair alone: a neighbour's edit moves this tile's
    # skirt bottom and Sloped corners, which _observed_elevation_range covers.
    # tests/test_bbox_elev_sweep.py's diff mask is what pins this.
    if seed_mask is None:
        seed_tiles = list(seed)
        sxs = np.fromiter((x for x, _ in seed_tiles), dtype=np.int64, count=len(seed_tiles))
        sys_ = np.fromiter((y for _, y in seed_tiles), dtype=np.int64, count=len(seed_tiles))
    else:
        mys, mxs = np.nonzero(seed_mask)
        sxs, sys_ = mxs.astype(np.int64), mys.astype(np.int64)
    lo, hi = _observed_elevation_range(elevations, pre_elevation, sxs, sys_)
    if footprint_owners:
        owners = list({o for owns in footprint_owners.values() for o in owns})
        olo, ohi = _observed_elevation_range(
            elevations, pre_elevation,
            np.array([x for x, _ in owners], dtype=np.int64), np.array([y for _, y in owners], dtype=np.int64),
        )
        owner_range = {o: (int(olo[i]), int(ohi[i])) for i, o in enumerate(owners)}
        if seed_mask is None:
            seed_index = {t: i for i, t in enumerate(seed_tiles)}
        else:
            index_grid = np.full(seed_mask.shape, -1, dtype=np.int64)
            index_grid[sys_, sxs] = np.arange(len(sxs))
            seed_index = {t: int(index_grid[t[1], t[0]]) for t in footprint_owners}
        for tile, owns in footprint_owners.items():
            i = seed_index[tile]
            for o in owns:
                lo[i] = min(lo[i], owner_range[o][0])
                hi[i] = max(hi[i], owner_range[o][1])
    tx0s, ty0s, tx1s, ty1s = iso_geometry.tile_screen_bounds_over(sxs, sys_, lo, hi, proj)
    x0, y0, x1, y1 = int(tx0s.min()), int(ty0s.min()), int(tx1s.max()), int(ty1s.max())

    if with_sprites:
        # Three things a reader would otherwise have to re-derive:
        #
        # MOVED tiles, not dirty (and never seed). sprite_draws_by_anchor
        # computes a sprite's anchor x with no elevation term at all, and its
        # anchor y from elevations[unit.y, unit.x] -- the unit's OWN centre
        # tile. So only an elevation change on that tile, or a unit edit, can
        # MOVE a sprite; a terrain tile repainted underneath one, or occlusion
        # revealed by a neighbour, is already covered by composite_rect_iso's
        # bystander set via merge_sprite_bboxes. Widening per moved tile
        # (every dirty tile when units_changed) is exact, not an optimisation.
        #
        # The elevation sweep, over the same observed range as the seed union
        # above: the sprite's stale position sits at the anchor's pre-edit
        # height and its new one at the post-edit height. The anchor's 3x3
        # neighbourhood bounds both, Sloped's corner interpolation included.
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
        # Batch D's D1b: memoized on scenario.unit_gen rather than recomputed
        # here every call -- named `live_anchor_tiles` locally so it can't
        # shadow the module-level anchor_tiles() function it now calls.
        live_anchor_tiles = anchor_tiles(scenario)
        if extra_anchor_tiles:
            # New set, never `|=`: live_anchor_tiles is the memo's own
            # cached object, and mutating it in place would corrupt every
            # future hit against the same scenario.unit_gen.
            live_anchor_tiles = live_anchor_tiles | extra_anchor_tiles
        # band_tiles is the set of ANCHOR tiles to pad around, not dirty
        # tiles: the padding below must be centered on where the sprite
        # actually sits (an anchor's own tile_screen_origin), never on
        # whichever trigger tile happened to catch it -- those can be
        # different tiles under Sloped's ring (radius=1). An anchor is
        # triggered when some trigger tile falls within its own radius-ring
        # (Stepped, radius=0: only the anchor tile itself; Sloped, radius=1:
        # its 3x3 neighbourhood, since a Sloped anchor reads its tile's four
        # shared corners).
        # Batch A6: dilate the small trigger set once and intersect rather than
        # ringing every anchor. radius 0 leaves it undilated (Stepped).
        r = sprite_band_radius
        trigger = dirty_xy if units_changed else elevation_changed_local
        dilated = {
            (tx + dx, ty + dy)
            for tx, ty in trigger
            for dx in range(-r, r + 1)
            for dy in range(-r, r + 1)
        }
        band_tiles = dilated & live_anchor_tiles
        # Unit MOVES: a terrain/elevation-only caller passes no
        # extra_anchor_tiles and no units_changed, so this pads only anchors
        # whose own height moved. D5's unit-edit wiring passes both, see
        # extra_anchor_tiles' and units_changed's own docstrings above.
        band = [(x, y) for x, y in band_tiles if 0 <= x < w and 0 <= y < h]
        if band:
            bxs = np.array([x for x, _ in band], dtype=np.int64)
            bys = np.array([y for _, y in band], dtype=np.int64)
            blo, bhi = _observed_elevation_range(elevations, pre_elevation, bxs, bys)
            tx0, ty_hi = iso_geometry.tile_screen_origin(bxs, bys, bhi, proj)
            ty_lo = ty_hi + (bhi - blo) * proj.elev_step
            x0 = min(x0, int(tx0.min()) - pad_l)
            x1 = max(x1, int(tx0.max()) + 2 * proj.half_w + pad_r)
            # y1's 2*half_h is REQUIRED (the band's far end). y0 keeps an extra
            # half_h beyond the band's near end deliberately, as slack rather
            # than a derived bound: the exact top is ty_hi, but this is an
            # under-repaint guard, and the only direction a too-tight bound
            # fails in is stale pixels. 16px of extra repaint per edit is not
            # worth being clever about.
            y0 = min(y0, int(ty_hi.min()) - proj.half_h - pad_u)
            y1 = max(y1, int(ty_lo.max()) + 2 * proj.half_h + pad_d)

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
    flatten_elevations: bool = False,
    extra_anchor_tiles: set[tuple[int, int]] | None = None,
    units_changed: bool = False,
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
    MAX_ELEVATION), should never actually happen given the viewer's own
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
    bookkeeping; only IsoChunkCache.patch()'s caller needs it.

    flatten_elevations (Flat+Isometric plan, F2): see _dirty_screen_bbox()'s
    own docstring. Only ViewerWindow's Flat+Iso render path passes True.

    extra_anchor_tiles (Batch D's D3) and units_changed: see
    _dirty_screen_bbox()'s own docstring -- passed straight through. Only a
    unit edit passes units_changed=True; a terrain or elevation edit leaves it
    False, which pads sprite reach only around anchors whose height moved."""
    # with_sprites is a PARAMETER as of P3-g's toggle, not a module global read
    # at call time: sprites are per-cache now, so this function cannot look the
    # answer up itself. The caller's obligation is therefore load-bearing --
    # this must be passed the same value _LeveledChunkCache._level is deciding on,
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
        flatten_elevations=flatten_elevations, extra_anchor_tiles=extra_anchor_tiles,
        units_changed=units_changed,
    )


def dirty_screen_bboxes_iso_grouped(
    scenario: LoadedScenario,
    groups,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    flatten_elevations: bool = False,
) -> list[tuple[int, int, int, int] | None] | None:
    """dirty_screen_bbox_iso(scenario, group, elevations, proj, with_sprites=
    False) for each tile-index group in `groups`, in one array pass
    (TASK-031.61 C1: a batch unit edit sizes one rect per block of tiles, and
    the per-group calls were 8-89 ms of a large stroke end or undo). Each
    answer is that call's own: the group's tiles dilated by 1 and clipped to
    the map, each swept over its observed elevations, clamped to the canvas;
    None for an empty group or one that lands off the canvas.

    Only for tiles whose elevation did not move, which is what lets the
    groups share a pass: nothing is written to `elevations` and no footprint
    is unioned. When a tile's elevation is not the one `elevations` holds (0,
    under flatten_elevations), or is outside proj's range, this returns None
    having written nothing, and
    the caller makes the per-group calls instead.
    tests/test_grouped_dirty_bbox.py keeps those calls as its oracle."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    out: list[tuple[int, int, int, int] | None] = [None] * len(groups)
    sizes = [len(group) for group in groups]
    total = sum(sizes)
    if not total:
        return out
    idx = np.fromiter((i for group in groups for i in group), dtype=np.int64, count=total)
    tiles = np.unique(idx)
    held = elevations[tiles // w, tiles % w]
    if flatten_elevations:
        moved = held != 0
    else:
        terrain = mm.terrain
        moved = held != np.fromiter((terrain[i].elevation for i in tiles.tolist()), dtype=np.int64, count=tiles.size)
    if moved.any() or ((held < proj.min_elev) | (held > proj.max_elev)).any():
        return None
    xs, ys = idx % w, idx // w
    # Each tile's 3x3, one row per offset. A neighbour off the map is replaced
    # by the tile itself, which is in the set anyway.
    offsets = np.arange(-1, 2, dtype=np.int64)
    nxs = xs + np.tile(offsets, 3)[:, None]
    nys = ys + np.repeat(offsets, 3)[:, None]
    outside = (nxs < 0) | (nxs >= w) | (nys < 0) | (nys >= h)
    nxs, nys = np.where(outside, xs, nxs).ravel(), np.where(outside, ys, nys).ravel()
    lo, hi = _observed_elevation_range(elevations, {}, nxs, nys)
    bounds = iso_geometry.tile_screen_bounds_over(nxs, nys, lo, hi, proj)
    tx0s, ty0s = (np.asarray(b).reshape(9, -1).min(axis=0) for b in bounds[:2])
    tx1s, ty1s = (np.asarray(b).reshape(9, -1).max(axis=0) for b in bounds[2:])
    # `idx` is in group order, so each non-empty group is one run of it.
    filled = [g for g, n in enumerate(sizes) if n]
    starts = np.cumsum([0] + [sizes[g] for g in filled[:-1]])
    canvas_w, canvas_h = _canvas_pixel_dims(proj)
    for g, gx0, gy0, gx1, gy1 in zip(
        filled,
        np.minimum.reduceat(tx0s, starts).tolist(),
        np.minimum.reduceat(ty0s, starts).tolist(),
        np.maximum.reduceat(tx1s, starts).tolist(),
        np.maximum.reduceat(ty1s, starts).tolist(),
        strict=True,
    ):
        x0, y0 = max(0, int(gx0)), max(0, int(gy0))
        x1, y1 = min(canvas_w, int(gx1)), min(canvas_h, int(gy1))
        if x1 > x0 and y1 > y0:
            out[g] = (x0, y0, x1, y1)
    return out


def dirty_screen_bbox_sloped(
    scenario: LoadedScenario,
    dirty_indices,
    elevations: np.ndarray,
    proj: iso_geometry.IsoProjection,
    with_units: bool = True,
    with_sprites: bool = False,
    elevation_changed: set | None = None,
    extra_anchor_tiles: set[tuple[int, int]] | None = None,
    units_changed: bool = False,
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
      _dirty_screen_bbox's own with_sprites comment for why MOVED tiles, not
      SEED, are exact for Stepped), so every tile in the one-tile ring around
      each moved tile can move a sprite, not just that tile itself.
    - the canvas clamp widens to _canvas_pixel_dims(proj) (the Step 0 fix):
      SlopedChunkCache.canvas_dims() reports that same wider bound once
      sprites are on, and this function's own final clamp must agree, or a
      sprite reaching into the newly-composited strip never gets a dirty
      bbox wide enough to reach it.

    elevation_changed: same optional out-param as dirty_screen_bbox_iso()'s
    own -- only SlopedChunkCache.patch()'s caller needs it.

    extra_anchor_tiles (Batch D's D3) and units_changed: see
    _dirty_screen_bbox()'s own docstring -- passed straight through, same
    contract as dirty_screen_bbox_iso()'s."""
    canvas_dims = _canvas_pixel_dims(proj) if with_sprites else (proj.canvas_w, proj.canvas_h)
    return _dirty_screen_bbox(
        scenario, dirty_indices, elevations, proj, canvas_dims, with_units,
        with_sprites=with_sprites, sprite_band_radius=1 if with_sprites else 0,
        unit_band_radius=1, elevation_changed=elevation_changed, extra_anchor_tiles=extra_anchor_tiles,
        units_changed=units_changed,
    )


@dataclass(frozen=True)
class BystanderGrid:
    """building_bboxes bucketed into a uniform cell_px grid over LEVEL pixels,
    origin (0, 0). This is the index composite_rect_iso()/
    composite_rect_sloped()'s bystander scan reads instead of walking every
    entry per rect.

    cells maps (gx, gy) -> that cell's ((px, py), bbox) entries. An entry is
    inserted into EVERY cell its bbox overlaps, so a bbox spanning several
    cells is stored several times and a read covering more than one cell can
    yield the same (px, py) twice. See _bystander_candidates() for why the
    dedupe there is load-bearing rather than tidy."""

    cell_px: int
    cells: dict[tuple[int, int], tuple[tuple[tuple[int, int], tuple[int, int, int, int]], ...]]


def build_bystander_grid(
    building_bboxes: dict[tuple[int, int], tuple[int, int, int, int]], cell_px: int
) -> BystanderGrid:
    """building_bboxes indexed by cell_px cell, for _bystander_candidates().

    Built once per building_bboxes build (a chunk cache's own per-level or
    per-refresh step), never per rect. That ratio is the whole point, since
    the rect reaching a compositor is either a whole chunk or an edit bbox
    clipped to one chunk, so a cache that uses its own chunk_px as cell_px
    turns the scan into a single dict hit.

    Cell (gx, gy) covers pixels [gx*c, (gx+1)*c) x [gy*c, (gy+1)*c), so a
    half-open bbox [ux0, ux1) spans cells ux0//c .. (ux1-1)//c inclusive.
    That is the same conversion _bystander_candidates() applies to its rect,
    which keeps the pair exactly as strict as the predicate it replaces. Python
    floor division is correct for the negative coordinates real bboxes carry
    (a building hanging off the top-left map edge, or Sloped's extra_top_px
    headroom lifting sy0 above the canvas).

    The assert is load-bearing, not decorative. A degenerate ux1 <= ux0 entry
    lands in no cell here, but the predicate it replaces still accepts one for
    any rect that strictly contains [ux1, ux0], so a degenerate entry would
    make the grid path differ from the walk. Both producers rule it out by
    construction (_unit_screen_bbox_iso's x extent is sx1 - sx0 + 2*half_w
    with sx1 >= sx0 and half_w >= 2, its y extent likewise; a sprite piece's
    is _draw_for_entry's max(1, ...) width/height; merge_sprite_bboxes only
    ever grows a bbox), so this pins that argument rather than guarding a
    case that happens.

    A drain of build_bystander_grid_sliced(), the only copy of the loop."""
    return _drain(build_bystander_grid_sliced(building_bboxes, cell_px))


def build_bystander_grid_sliced(
    building_bboxes: dict[tuple[int, int], tuple[int, int, int, int]], cell_px: int
) -> Generator[None, None, BystanderGrid]:
    """build_bystander_grid() as a resumable generator, at most ASSEMBLY_SLICE
    keys per step. The grid is the return value."""
    step = ASSEMBLY_SLICE
    cells: dict[tuple[int, int], list] = {}
    for seen, (key, bbox) in enumerate(building_bboxes.items()):
        if seen and seen % step == 0:
            yield
        entry = (key, bbox)
        for cell in _bbox_cells(key, bbox, cell_px):
            cells.setdefault(cell, []).append(entry)
    return BystanderGrid(cell_px=cell_px, cells={k: tuple(v) for k, v in cells.items()})


def _bbox_cells(key, bbox: tuple[int, int, int, int], cell_px: int):
    """Every cell a half-open bbox covers, after build_bystander_grid()'s
    degenerate-bbox assert. The one conversion the build and the patch share."""
    ux0, uy0, ux1, uy1 = bbox
    assert ux1 > ux0 and uy1 > uy0, f"degenerate building bbox at {key}: {bbox}"
    for gy in range(uy0 // cell_px, (uy1 - 1) // cell_px + 1):
        for gx in range(ux0 // cell_px, (ux1 - 1) // cell_px + 1):
            yield gx, gy


def patch_bystander_grid(
    grid: BystanderGrid,
    old_bboxes: dict[tuple[int, int], tuple[int, int, int, int] | None],
    building_bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> BystanderGrid:
    """grid, built over building_bboxes before a splice rewrote the keys in
    old_bboxes (key -> its bbox before, None if absent), brought up to date
    with building_bboxes as it is now. Equal to build_bystander_grid() of the
    current dict up to entry order within a cell, which cannot reach a pixel:
    _bystander_candidates() dedupes and re-sorts into a total depth order.

    Only the cells an old or new bbox of a changed key covers are rebuilt,
    into a copy of cells, so the input grid is never mutated. A key whose bbox
    did not change is dropped first, and a batch with none left returns grid
    itself. A cell left empty is deleted, as the full build never stores one."""
    changed = {k: old for k, old in old_bboxes.items() if building_bboxes.get(k) != old}
    if not changed:
        return grid
    c = grid.cell_px
    affected: set[tuple[int, int]] = set()
    added: dict[tuple[int, int], list] = {}
    for key, old in changed.items():
        if old is not None:
            affected.update(_bbox_cells(key, old, c))
        new = building_bboxes.get(key)
        if new is not None:
            entry = (key, new)
            for cell in _bbox_cells(key, new, c):
                affected.add(cell)
                added.setdefault(cell, []).append(entry)
    cells = dict(grid.cells)
    for cell in affected:
        entries = [e for e in cells.get(cell, ()) if e[0] not in changed]
        entries.extend(added.get(cell, ()))
        if entries:
            cells[cell] = tuple(entries)
        else:
            cells.pop(cell, None)
    return BystanderGrid(cell_px=c, cells=cells)


def _bystander_candidates(
    candidates: np.ndarray,
    building_bboxes: dict,
    grid: BystanderGrid | None,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    w: int,
) -> np.ndarray:
    """candidates merged with this rect's "bystander" buildings and re-sorted
    into depth order: composite_rect_iso()/composite_rect_sloped()'s shared
    bystander step (see composite_rect_iso()'s docstring for what a bystander
    is and why the merge exists).

    grid=None walks building_bboxes in full, which is the original shape and
    what refresh_region_iso(), the tools and the tests still take. A grid
    narrows the walk to the cells the rect touches; the predicate is then
    re-applied to every entry it yields, because two boxes sharing a cell need
    not overlap each other. The grid is a filter, never the test.

    Completeness: a bbox overlapping the rect shares at least one pixel with
    it, and the cell holding that pixel holds the bbox and is visited. So the
    grid path can only ever narrow the walk, never the result.

    The dedupe is the hazard the grid introduces. A bbox spanning several
    cells sits in each, so a multi-cell read can yield one (px, py) twice; a
    duplicated tile paints twice and _clipped_darken is MULTIPLICATIVE, so a
    doubled contact-shadow band is a visible pixel change, not a no-op.
    Adding each accepted key to `seen` handles that with the same mechanism
    that already excludes terrain candidates, so the 1x1 and multi-cell cases
    need no separate branch.

    seen packs (px, py) as py * w + px: injective because
    tiles_in_screen_rect() clamps candidates to [0, w) x [0, h) and
    building_bboxes' keys come from unit_occupied_tiles(), likewise clamped.
    .tolist() unboxes in C rather than N per-element int() calls."""
    seen = set((candidates[:, 1] * w + candidates[:, 0]).tolist())
    bystanders = []
    if grid is not None and x1 > x0 and y1 > y0:
        # An empty rect (x1 <= x0) has no cell range but still satisfies the
        # predicate for any bbox strictly containing it, so it takes the walk.
        c = grid.cell_px
        for gy in range(y0 // c, (y1 - 1) // c + 1):
            for gx in range(x0 // c, (x1 - 1) // c + 1):
                for (px, py), (ux0, uy0, ux1, uy1) in grid.cells.get((gx, gy), ()):
                    if ux0 < x1 and ux1 > x0 and uy0 < y1 and uy1 > y0:
                        key = py * w + px
                        if key not in seen:
                            seen.add(key)
                            bystanders.append((px, py))
    else:
        for (px, py), (ux0, uy0, ux1, uy1) in building_bboxes.items():
            if py * w + px not in seen and ux0 < x1 and ux1 > x0 and uy0 < y1 and uy1 > y0:
                bystanders.append((px, py))
    if not bystanders:
        return candidates
    extra = np.array(bystanders, dtype=np.int64)
    combined = np.concatenate([candidates, extra], axis=0)
    xs, ys = combined[:, 0], combined[:, 1]
    order = np.lexsort((xs, ys - xs))  # primary key is the LAST arg: d=y-x, then x, matching depth_order()
    # (d, x) determines a tile uniquely, so no two entries tie and the sort is
    # a total order, which is what makes grid iteration order irrelevant.
    return combined[order]


def _rect_candidates(scenario, x0, y0, x1, y1, proj, with_units, building_bboxes, bystander_grid) -> np.ndarray:
    """composite_rect_iso/_sloped's ordered candidates: the rect's terrain
    tiles plus its bystander buildings, in depth order."""
    w, h = scenario.map_manager.map_width, scenario.map_manager.map_height
    candidates = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)
    if with_units and building_bboxes:
        candidates = _bystander_candidates(candidates, building_bboxes, bystander_grid, x0, y0, x1, y1, w)
    return candidates


def prepare_rect_native(
    sloped: bool, scenario: LoadedScenario, x0: int, y0: int, x1: int, y1: int, heights: np.ndarray,
    proj: iso_geometry.IsoProjection, tile_px: int, building_bboxes: dict, with_units: bool,
    bystander_grid: BystanderGrid | None, layers: LayerState, grid: GridBake, unit_pack,
    terrain_ids: np.ndarray | None = None, layer_ids: np.ndarray | None = None,
    key_map: terrain_blend.KeyMap | None = None, snapshot: np.ndarray | None = None,
    fold: np.ndarray | None = None,
) -> tuple | None:
    """composite_rect_iso (heights = elevations) or composite_rect_sloped
    (heights = corner_rise) with every Python step done here and only the
    nogil kernel left, for margin_warm's worker threads (Batch F T2) and a
    pooled render_rect() (TASK-031.64). snapshot (Stepped only): a copy of
    heights the caller owns, passed to the kernel uncopied
    (native_composite.prepare_iso). fold (Sloped only): tile_folds() beside
    corner_rise, rebound with it, never written in place.

    Returns (kernel, scratch, args, holds): kernel(scratch, *args) paints what
    the composite would return, or returns False having painted nothing (take
    the per-tile path). None when this rect has no whole-rect native path. The
    args own snapshots of every input the GUI thread writes in place; holds
    keeps the unit pack alive, since the kernel reaches its sprite pixels
    through raw addresses the pack alone owns. terrain_ids: the calling
    cache's per-tile mirror (native_composite's docstring), None reads live.
    Blend keys are read here, on the calling thread, from both mirrors
    through key_map (_rect_blend_keys()); a worker never touches the map."""
    native = composite_backend.native
    if native is None or unit_pack is None:
        return None
    candidates = _rect_candidates(scenario, x0, y0, x1, y1, proj, with_units, building_bboxes, bystander_grid)
    if not candidates.shape[0]:
        return None
    blend = _rect_blend_keys(scenario, candidates, layers, terrain_ids, layer_ids, key_map)
    from descape import native_composite

    prepare = native_composite.prepare_sloped if sloped else native_composite.prepare_iso
    extra = {"fold": fold} if sloped else {"snapshot": snapshot}
    args = prepare(
        scenario, x0, y0, candidates, heights, proj, tile_px, unit_pack if with_units else None,
        layers.terrain_textures, grid, share=True, terrain_ids=terrain_ids, blend=blend, **extra,
    )
    if args is None:
        return None
    kernel = native.composite_sloped if sloped else native.composite_iso
    return kernel, np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8), args, (unit_pack,)
