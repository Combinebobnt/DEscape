"""The render's unit sprites: SpriteLayer, draped and farm tiles, draws by anchor, sprite resolution, SpriteMemo and sprite bbox merges."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

from descape import (
    game_dataset,
    games,
    iso_geometry,
    mod_graphics,
    unit_kind,
    unit_sprites,
)
from descape.render_paint import ASSEMBLY_SLICE, INK_NONE, _drain
from descape.render_units import (
    NON_BUILDING_SPAN,
    _span_start,
    _sprite_axis_centre,
    _unit_color,
    garrison_host_ids,
    player_art_for,
    stored_rotation,
    unit_occupied_tiles,
    unit_tile_bounds,
    wall_variant_rotation_overrides,
)
from descape.scenario_io import LoadedScenario
from descape.terrain_palette import (
    DOODAD_TILE_SPANS,
    FOUNDATION_TERRAIN,
    HERO_GLOW_CONSTS,
    STATIC_TILE_SPANS,
    TREE_UNIT_IDS,
    tile_span,
)
from descape.unit_filter import UnitFilter


def _ink_floor(tiles: list, units_by_tile: dict, sprites: SpriteLayer, mark_rise) -> int:
    """The lowest base rise among the tiles' marks and sprites, INK_NONE with
    none: no tile at or below it can occlude anything. mark_rise(unit, tile)
    is the rise the objects half would record for that mark."""
    floor = INK_NONE
    skip, by_anchor = sprites.skip_ids, sprites.by_anchor
    for tile in tiles:
        key = (tile.x, tile.y)
        for unit, _color in units_by_tile.get(key, ()):
            if id(unit) not in skip:
                floor = min(floor, mark_rise(unit, tile))
        for entry in by_anchor.get(key, ()):
            floor = min(floor, entry[3])
    return floor


def unit_sprite_draws_at(
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    corner_rise: np.ndarray | None,
    player_id: int,
    unit,
    rotation_override: float | None = None,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    fold: np.ndarray | None = None,
) -> list[tuple[object, int, int]]:
    """Every sprite piece `unit` would paint at its own stored position, as
    (draw, px, py) triples in canvas pixels, ordered the way the composite
    walk paints them -- row-major by anchor tile, matching `mm.terrain`.

    A thin call into _resolve_unit_sprite() rather than a re-derivation,
    deliberately: the anchor expression in there carries three separately
    documented traps (the trailing `+ half_h`, the fx/fy-versus-own_fx/own_fy
    split, and rise_px staying inside the round()), and a second copy of it
    would drift from the pixels the map actually draws.

    with_farms=False: a farm's overlay is terrain, not a sprite, so a farm
    resolves to nothing here and the caller falls back to the coloured mark.

    rotation_override is this unit's wall-connectivity-derived rotation, which
    the caller resolves ONCE -- wall_variant_rotation_overrides() is a walk
    over every unit in the file behind a memo, and calling it per mouse-move
    is exactly the cost a preview exists to avoid.

    tree_scale must be the WINDOW's current View > Layers value: a ghost left
    on the default 1.0 would drag a tree at full size and snap it small on
    drop. hero_glow likewise, or a hero would lose its ring mid-drag, and
    wall_alpha, or a wall would drag opaque. fold goes with corner_rise, as
    sprite_draws_by_anchor()'s."""
    overrides = {} if rotation_override is None else {(player_id, 0): rotation_override}
    # Unsliced: the ghost draws as scaled QImages, where adjacent column slices would seam.
    contribution = _resolve_unit_sprite(
        scenario, proj, elevations, UnitFilter(), corner_rise, overrides, False,
        player_id, 0, unit, tree_scale, hero_glow, wall_alpha, sliced=False, fold=fold,
    )
    if contribution is None:
        return []
    # by_anchor entries also carry the base rise; callers here place by pixel only.
    return [
        entry[:3]
        for key in sorted(contribution.by_anchor, key=lambda k: (k[1], k[0]))
        for entry in contribution.by_anchor[key]
    ]


@dataclass(frozen=True)
class SpriteLayer:
    """Everything the Stepped compositor needs to draw real sprites (P3-g3),
    precomputed once per elevations snapshot the same way units_by_tile and
    building_bboxes already are -- resolving and decoding a sprite per chunk
    would dominate a chunk's cost outright.

    by_anchor maps a tile to the sprites that paint AT that tile, in the
    units' own relative order with moved column slices last
    (_accumulate_contribution()), each as (draw, px, py, rise_px): the anchor
    pixel and the unit's base rise, which decides what terrain may cover its
    ink (_paint_candidates_iso). A multi-tile piece contributes one entry per
    column run (_sprite_columns(), TASK-221), each draw a slice of the piece
    with the same anchor pixel. bboxes is keyed the same way and merges
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

    by_anchor: dict[tuple[int, int], list[tuple[object, int, int, int]]]
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


# Consts whose foundation terrain drapes UNDER their sprite rather than
# instead of it: placed Pastures, whose annex-tree art (GH #66) is a hut,
# posts and fences standing on the pasture ground, not a ground cover itself.
DRAPED_SPRITE_CONSTS: frozenset[int] = frozenset({1893, 1897})
# Return of Rome (GH #67, 2026-10-17): Farm 50 and RiceFarm 1187 have a sprite there, only
# the vases a rotation picks; the field is their foundation terrain (7, 63), as in game.
_ROR_DRAPED_SPRITE_CONSTS: frozenset[int] = frozenset({50, 1187, 1893, 1897})
_DRAPED_BY_GAME = games.PerGame(
    "render: consts whose foundation terrain drapes under their sprite",
    {"aoe2": DRAPED_SPRITE_CONSTS, "ror": _ROR_DRAPED_SPRITE_CONSTS},
)


def draped_sprite_consts() -> frozenset[int]:
    """The active game's DRAPED_SPRITE_CONSTS, read at call time so a
    monkeypatch.setitem() of one game's set holds."""
    return _DRAPED_BY_GAME.active()


def _farm_footprint_tiles(
    bounds: tuple[int, int, int, int], terrain_id: int, color: tuple[int, int, int]
) -> dict[tuple[int, int], tuple[int, tuple[int, int, int], int]]:
    """One draped unit's farm_by_tile entries: every footprint tile maps to
    (terrain_id, outline colour, the EDGE_* bits on the footprint's outer
    boundary). Shared by the iso sprite walk and flat_farm_tiles()."""
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    farm_tiles: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}
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
            farm_tiles[(tx, ty)] = (terrain_id, color, mask)
    return farm_tiles


def flat_farm_tiles(
    scenario: LoadedScenario, unit_filter: UnitFilter = UnitFilter()
) -> tuple[dict[tuple[int, int], tuple[int, tuple[int, int, int], int]], set[int]]:
    """Top-down Flat's farm drape (GH #150): (by_tile, unit_ids), by_tile in
    SpriteLayer.farm_by_tile's shape and unit_ids the id(unit) of every unit
    that contributed. Walks players then units in list order, so a later
    farm wins an overlap as _accumulate_contribution() lets it. Mip-free:
    the caller paints each entry at its own tile_px."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    by_tile: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}
    unit_ids: set[int] = set()
    overlay: dict[int, int | None] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            const = unit.unit_const
            if const not in overlay:
                overlay[const] = _terrain_overlay_for(const)
            terrain_id = overlay[const]
            if terrain_id is None or not unit_filter.matches(player_id, unit):
                continue
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            color = _unit_color(unit, scenario.player_colors[player_id])
            by_tile.update(_farm_footprint_tiles(bounds, terrain_id, color))
            unit_ids.add(id(unit))
    return by_tile, unit_ids


def _paint_farm_outline_flat(
    img: np.ndarray, base_y: int, base_x: int, tile_px: int, color: tuple[int, int, int], edge_mask: int
) -> None:
    """A Flat farm tile's 1px owner-colour perimeter, along the inside of its
    block on each outer side: EDGE_LEFT is x-low, EDGE_UP_RIGHT x-high,
    EDGE_UP_LEFT y-low, EDGE_RIGHT y-high (iso_geometry.tile_edge_indices'
    neighbour vocabulary). Clipped to img."""
    h, w = img.shape[:2]
    strips = []
    if edge_mask & EDGE_LEFT:
        strips.append((base_y, base_y + tile_px, base_x, base_x + 1))
    if edge_mask & EDGE_UP_RIGHT:
        strips.append((base_y, base_y + tile_px, base_x + tile_px - 1, base_x + tile_px))
    if edge_mask & EDGE_UP_LEFT:
        strips.append((base_y, base_y + 1, base_x, base_x + tile_px))
    if edge_mask & EDGE_RIGHT:
        strips.append((base_y + tile_px - 1, base_y + tile_px, base_x, base_x + tile_px))
    for sy0, sy1, sx0, sx1 in strips:
        y0, y1, x0, x1 = max(sy0, 0), min(sy1, h), max(sx0, 0), min(sx1, w)
        if y1 > y0 and x1 > x0:
            img[y0:y1, x0:x1] = color


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
    instead. draped_sprite_consts() are the exception: their drape is
    unconditional, so it is just as deterministic."""
    if unit_const in unit_sprites.graphic_map() and unit_const not in draped_sprite_consts():
        return None
    return FOUNDATION_TERRAIN.get(unit_const)


def sprite_draws_by_anchor(
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter = UnitFilter(),
    corner_rise: np.ndarray | None = None,
    with_farms: bool = True,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    fold: np.ndarray | None = None,
) -> SpriteLayer:
    """Resolves every visible unit to a real .sld sprite or a farm-terrain
    override, or leaves it to the coloured mark.

    A unit that resolves nowhere -- no graphic in the table and no
    foundation terrain, no install, a missing or unreadable file, or a
    frame that fails to decode -- is simply absent from every field here,
    which is exactly what makes the sprite/farm path strictly additive.

    **GAIA's `rotation` goes through stored_rotation().** For GAIA variant
    consts it is a tree/doodad graphic-variant index (integers well outside
    [0, 2*pi)), for GAIA ANGLE consts a facing (GH #217) -- see AGENTS.md's
    hard rule.

    corner_rise (Track P3-g6): pass Sloped's (h+1, w+1) corner-rise field to
    anchor sprites at iso_geometry.unit_rise_px() -- the same surface
    _paint_tile_and_units_sloped()/unit_pick.unit_rise_px_for() already use --
    instead of Stepped's flat `elevation * elev_step`. None (the default)
    keeps Stepped's exact expression, including its rounding, so
    render_terrain_sloped's flat-map byte-identity oracle stays exact
    character for character. fold: iso_geometry.tile_folds() of the same
    elevations, the unit's own tile's fold for unit_rise_px(); None only on
    a planar corner_rise (sprite_draws_by_anchor_sliced() raises otherwise).

    with_farms=False (Track P3-g6) suppresses the farm-terrain-override
    branch entirely -- neither farm_by_tile nor skip_ids gains an entry for
    a farm unit -- so Sloped keeps today's plain coloured mark for farms
    without a paint-time skip making them invisible. See
    _paint_tile_and_units_sloped's own farm handling for the other half of
    that deferral.

    tree_scale < 1.0 (View > Layers > Small Trees) draws TREE_UNIT_IDS art
    at that fraction of its usual size. bboxes follow it for free: they are
    derived from each piece's own rgba shape and hotspot, so a shrunk sprite
    reports a shrunk box.

    hero_glow=True (View > Layers > Hero Glow) rings HERO_GLOW_CONSTS art in
    gold. Off by default here, so a caller that never names it stays
    byte-identical; the chunk caches pass their LayerState's own value, which
    defaults on. The bboxes widen with the ring the same way.

    wall_alpha < 1.0 (View > Layers > Translucent Walls) fades
    unit_kind.wall_consts() art (GH #35); see unit_sprites.sprite_pieces_for()."""
    return _drain(sprite_draws_by_anchor_sliced(
        scenario, proj, elevations, unit_filter, corner_rise, with_farms, tree_scale, hero_glow, wall_alpha,
        fold=fold,
    ))


@dataclass
class _SpriteContribution:
    """One unit's resolved contribution to a SpriteLayer -- Batch D's D2
    extraction, so D4's splice can re-resolve a single edited unit against
    an already-built layer instead of re-walking every unit. Exactly one of
    (by_anchor, farm_tiles) is ever non-empty: a unit resolves to a real sprite,
    a farm-terrain override, or (returned as None by the resolver, never as
    an instance of this class) nothing at all -- see _resolve_unit_sprite().
    The one exception is DRAPED_SPRITE_CONSTS (pastures), which carry both.

    by_anchor/bboxes are keyed per PIECE depth slot, and a multi-tile piece
    per screen-column run (_sprite_columns(), TASK-221), so a building
    contributes to more than one anchor tile (a town centre's `main` paints
    three tiles earlier than the rest of it). Every key is inside the unit's
    own footprint, which is what lets a single-unit splice treat them all as
    this unit's alone."""

    skip_id: int
    by_anchor: dict[tuple[int, int], list[tuple[object, int, int, int]]]
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]]
    farm_tiles: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]]


def _sprite_keeps_own_point(unit_const: int) -> bool:
    """A class-14 doodad or other static object with a footprint
    (DOODAD_TILE_SPANS, TASK-134; STATIC_TILE_SPANS, TASK-277) draws its sprite
    at its stored point and sorts it at its own tile, as before it had one. Its
    own tile is inside the footprint, so a splice still clears it. An invisible
    static (the flares) is the exception: its marker is the footprint itself,
    so it takes the building-like anchor and stays on its outline."""
    if unit_const in DOODAD_TILE_SPANS:
        return True
    return unit_const in STATIC_TILE_SPANS and unit_const not in unit_kind.invisible_consts()


# TASK-221: multi-tile sprite pieces paint per screen column (_sprite_columns()).
# False is the single-anchor render, the revert switch for tests, injects and the
# release diff. A memo invariant, so flipping it never serves the other state.
SPRITE_COLUMN_SLICING = True


def _sprite_columns(
    occupied: list[tuple[int, int]], key: tuple[int, int], x0: int, w: int, proj: iso_geometry.IsoProjection
) -> list[tuple[tuple[int, int], int, int, bool]]:
    """Where a multi-tile sprite piece's columns paint (TASK-221), as
    (anchor, c0, c1, moved) runs over its rgba's columns [0, w), the rgba's
    left edge at canvas x `x0`. `key` is the piece's own key (its clamped
    slot, or sprite_anchor_tile()), the tile the whole piece painted at
    before; `occupied` is unit_occupied_tiles().

    Screen strip s, [origin_x + s*half_w, origin_x + (s+1)*half_w), is
    covered by the tiles on lines x + y = s - 1 and s (tile_screen_origin()).
    Its front is the occupied tile there with the largest depth key
    (y - x, x), or, with none, the front of the nearest strip to its left.
    A strip paints at the earlier of its front and `key`, so a unit standing
    in front of a face sorts after that face's columns: in its own two
    columns a 1x1 tile outside the footprint never ties the front, and has
    the larger key exactly when it is nearer. A strip outside the occupied
    range (art overhanging the footprint, or a footprint clamped at the map
    edge) keeps `key`, as does a strip with no front at all. Adjacent strips
    with one anchor merge; moved is anchor != key."""
    fronts: dict[int, tuple[int, int]] = {}
    for tx, ty in occupied:
        for s in (tx + ty, tx + ty + 1):
            front = fronts.get(s)
            if front is None or (ty - tx, tx) > (front[1] - front[0], front[0]):
                fronts[s] = (tx, ty)
    lo, hi = min(fronts), max(fronts)
    key_depth = (key[1] - key[0], key[0])
    half_w, origin_x = proj.half_w, proj.origin_x
    runs: list[list] = []
    col = 0
    strip = (x0 - origin_x) // half_w
    front = None
    for s in range(lo, min(strip, hi + 1)):
        front = fronts.get(s, front)
    while col < w:
        end = min(w, origin_x + (strip + 1) * half_w - x0)
        anchor = key
        if lo <= strip <= hi:
            front = fronts.get(strip, front)
            if front is not None and (front[1] - front[0], front[0]) < key_depth:
                anchor = front
        if runs and runs[-1][0] == anchor:
            runs[-1][2] = end
        else:
            runs.append([anchor, col, end])
        col = end
        strip += 1
    if not runs:
        return [(key, 0, w, False)]
    return [(anchor, c0, c1, anchor != key) for anchor, c0, c1 in runs]


def _resolve_unit_sprite(
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter,
    corner_rise: np.ndarray | None,
    overrides: dict[tuple[int, int], float],
    with_farms: bool,
    player_id: int,
    i: int,
    unit,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    flag_hosts: frozenset[int] | None = None,
    sliced: bool = True,
    fold: np.ndarray | None = None,
) -> _SpriteContribution | None:
    """Resolves ONE unit to its sprite or farm-terrain-override contribution
    -- the per-unit body sprite_draws_by_anchor_sliced()'s loop inlined
    before Batch D's D2, extracted verbatim (byte-identical prereq) so a
    future single-unit splice can call the same logic. None means the loop's
    own `continue` (filtered out, off-map, or resolves to neither a sprite
    nor a farm override) -- the caller's contract is identical to that
    `continue`.

    A unit that draws a sprite and holds a garrison (garrison_host_ids(),
    GH #203) also paints its garrison flag pieces, slot-free, so they key on
    its anchor tile after its own pieces. `flag_hosts` is that set when the
    caller already holds it (a walk fetches it once); None looks it up.

    `i` must be the unit's own position in scenario.unit_manager.units
    [player_id] -- see wall_variant_rotation_overrides()'s own docstring for
    why that, and not a filtered/skipped counter, is the override dict's key.

    `tree_scale` (View > Layers > Small Trees) reaches only TREE_UNIT_IDS
    consts -- the same set Filters' Show Trees owns, reused rather than
    re-derived, and deliberately not terrain_units.TREE_CONSTS, which is that
    set intersected with the auto-place table and so a strict subset. This is
    where the "which consts" decision lives; unit_sprites just takes a float.
    `hero_glow` is gated here the same way, to HERO_GLOW_CONSTS (GH #39), and
    `wall_alpha` to unit_kind.wall_consts(), the set
    Filters' Show Walls owns, gates included (GH #35).

    A multi-tile unit that does not keep its own point paints each piece per
    screen column (TASK-221, _sprite_columns()) while `sliced` and
    SPRITE_COLUMN_SLICING hold: a column run anchored off the piece's own key
    is a unit_sprites.slice_draw() of it, keyed on its column's front tile,
    with a bbox from its own shape. `sliced=False` keeps whole pieces, for
    unit_sprite_draws_at()'s ghost."""
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    half_w = proj.half_w
    if not unit_filter.matches(player_id, unit):
        return None
    bounds = unit_tile_bounds(unit, tile_w, tile_h)
    if bounds is None:
        return None
    rotation = stored_rotation(player_id, unit)
    rotation = overrides.get((player_id, i), rotation)
    team_index = scenario.team_indices[player_id]
    # Short-circuits, so the wall set is only looked up with the wall row on.
    is_faded_wall = wall_alpha < 1.0 and unit.unit_const in unit_kind.wall_consts()
    art = player_art_for(scenario, player_id)
    reference_id = getattr(unit, "reference_id", None)
    pieces = unit_sprites.sprite_pieces_for(
        unit.unit_const, rotation, team_index, half_w,
        tree_scale if unit.unit_const in TREE_UNIT_IDS else 1.0,
        reference_id,
        hero_glow=hero_glow and unit.unit_const in HERO_GLOW_CONSTS,
        art=art,
        wall_alpha=wall_alpha if is_faded_wall else 1.0,
    )
    # A pasture (draped_sprite_consts()) carries both: its drape plus its annex-tree art.
    farm_tiles: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}
    terrain_id = _terrain_overlay_for(unit.unit_const) if with_farms else None
    if terrain_id is not None and (not pieces or unit.unit_const in draped_sprite_consts()):
        player_color = scenario.player_colors[player_id]
        farm_tiles = _farm_footprint_tiles(bounds, terrain_id, _unit_color(unit, player_color))
    if not pieces:
        if not farm_tiles:
            return None
        return _SpriteContribution(
            skip_id=id(unit), by_anchor={}, bboxes={}, farm_tiles=farm_tiles
        )
    if reference_id in (garrison_host_ids(scenario) if flag_hosts is None else flag_hosts):
        # A new list: sprite_pieces_for()'s may be a cached one.
        pieces = [*pieces, *unit_sprites.garrison_flag_pieces_for(unit.unit_const, team_index, half_w, art)]

    span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
    # The UNCLAMPED footprint start, so a building hanging off a map
    # edge still anchors on its true centre rather than on the centre
    # of whatever survived clamping.
    fx = _sprite_axis_centre(unit.x, span_x)
    fy = _sprite_axis_centre(unit.y, span_y)
    ux, uy = int(unit.x), int(unit.y)
    own_point = _sprite_keeps_own_point(unit.unit_const)
    if own_point:
        fx, fy = unit.x, unit.y
    # own_fx/own_fy: sub-tile fractions INSIDE the unit's own tile, not
    # to be confused with fx/fy above (where the footprint PAINTS, in
    # continuous tile coords) -- unit_rise_px() needs the former, the
    # x/y placement below needs the latter. Same names, different
    # quantities is exactly how this file has produced green-suite bugs
    # before (see the half-tile floating-sprite bug map_point_to_screen's
    # docstring records).
    own_fx, own_fy = unit.x - ux, unit.y - uy
    rise_px = (
        iso_geometry.unit_rise_px(corner_rise, ux, uy, own_fx, own_fy, iso_geometry.tile_nwse(fold, ux, uy))
        if corner_rise is not None
        else int(elevations[uy, ux]) * proj.elev_step
    )
    # The trailing `+ half_h` and rise_px-inside-the-round() are both
    # load-bearing; map_point_to_screen's own docstring records why.
    ax, ay = iso_geometry.map_point_to_screen(fx, fy, rise_px, proj)

    # A piece carrying its own depth slot paints at THAT footprint tile's
    # moment instead of the unit's single anchor, which is what lets a unit
    # stand between a town centre's back and front pieces. Slotting moves
    # ordering only, never a pixel: px/py below are unchanged either way.
    occupied = None if own_point else unit_occupied_tiles(unit, tile_w, tile_h)
    anchor = (ux, uy) if own_point else unit_sprites.sprite_anchor_tile(occupied)
    columns = sliced and SPRITE_COLUMN_SLICING and occupied is not None and len(occupied) > 1
    tile_x0, tile_x1, tile_y0, tile_y1 = bounds
    # The unclamped footprint start, matching how a slot offset was derived;
    # bounds is already map-clamped and is the wrong origin to add to.
    slot_x0, slot_y0 = _span_start(unit.x, span_x), _span_start(unit.y, span_y)
    by_anchor: dict[tuple[int, int], list[tuple[object, int, int, int]]] = {}
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for piece in pieces:
        px, py = ax + piece.dx, ay + piece.dy
        if piece.slot is None:
            key = anchor
        else:
            key = (
                min(max(slot_x0 + piece.slot[0], tile_x0), tile_x1 - 1),
                min(max(slot_y0 + piece.slot[1], tile_y0), tile_y1 - 1),
            )
        draw = piece.draw
        h, w = draw.rgba.shape[:2]
        x0, y0 = px - draw.hotspot_x, py - draw.hotspot_y
        runs = _sprite_columns(occupied, key, x0, w, proj) if columns else ((key, 0, w, False),)
        for run_key, c0, c1, moved in runs:
            if c0 == 0 and c1 == w and not moved:
                run_draw = draw
            else:
                run_draw = unit_sprites.slice_draw(draw, c0, c1, key if moved else None)
            by_anchor.setdefault(run_key, []).append((run_draw, px, py, rise_px))
            run_bbox = (x0 + c0, y0, x0 + c1, y0 + h)
            bbox = bboxes.get(run_key)
            bboxes[run_key] = run_bbox if bbox is None else (
                min(bbox[0], run_bbox[0]),
                min(bbox[1], run_bbox[1]),
                max(bbox[2], run_bbox[2]),
                max(bbox[3], run_bbox[3]),
            )
    return _SpriteContribution(
        skip_id=id(unit), by_anchor=by_anchor, bboxes=bboxes, farm_tiles=farm_tiles
    )


@dataclass(frozen=True)
class SpriteMemo:
    """Per-unit _SpriteContribution memo for one chunk-cache level, so a
    wholesale sprite-layer rebuild re-resolves only the units whose inputs
    changed and re-merges the rest (2026-09-25 sprite-contribution memo plan).

    `invariants` is everything _resolve_unit_sprite() reads that is the same
    for every unit of a walk (see _sprite_memo_invariants()). A walk given a
    memo whose invariants differ treats it as empty, so a Layers flip, a proj
    change or a Stepped/Sloped mix-up can never serve a stale contribution,
    whatever order a caller swaps its state in.

    `entries` maps id(unit) to (unit, key, contribution). Holding the unit
    keeps its id from being reused while the entry lives; the `is` check on
    lookup guards it anyway. contribution may be None (resolved to nothing).
    The unit filter is not part of either: matches() runs before every
    lookup, since a garrison edit changes it without touching any key field.

    Never mutated after construction: a walk builds a fresh one holding only
    the units it saw, so deleted units drop out, and a warm still walking an
    older memo keeps getting correct (fully keyed) answers from it."""

    invariants: tuple = ()
    entries: dict = field(default_factory=dict)


def _sprite_memo_invariants(
    proj, mm, corner_rise, with_farms, tree_scale, hero_glow, wall_alpha=1.0
) -> tuple:
    return (
        proj, mm.map_width, mm.map_height, corner_rise is not None, with_farms, tree_scale, hero_glow, wall_alpha,
        mod_graphics.generation(), unit_sprites.hi_res_generation(), game_dataset.generation(),
        SPRITE_COLUMN_SLICING,
    )


def _sprite_memo_key(
    scenario, heights: list, sloped: bool, overrides: dict, player_id: int, i: int, unit, flag_hosts: frozenset[int],
    folds: list | None = None,
) -> tuple:
    """Every per-unit input of _resolve_unit_sprite() that isn't a memo
    invariant. `heights` is elevations (Stepped) or corner_rise (Sloped) as
    nested lists, read with the resolver's own indices so an elevation edit
    under a unit changes its key; Sloped keys the four corners, which with
    x/y and the tile's fold (`folds`, tile_folds() as nested lists; a fold
    can flip with all four corners unchanged) fully determine
    unit_rise_px(). An index the resolver would fail on
    keys as None and the resolver decides. `flag_hosts` is the walk's
    garrison_host_ids(): whether the unit draws a garrison flag is set by
    other units' links, which no field of its own carries (GH #203)."""
    ux, uy = int(unit.x), int(unit.y)
    try:
        if sloped:
            row, below = heights[uy], heights[uy + 1]
            rise = (row[ux], row[ux + 1], below[ux], below[ux + 1], folds is not None and folds[uy][ux])
        else:
            rise = heights[uy][ux]
    except IndexError:
        rise = None
    reference_id = getattr(unit, "reference_id", None)
    return (
        unit.x, unit.y, unit.unit_const,
        overrides.get((player_id, i), stored_rotation(player_id, unit)),
        reference_id, player_id,
        scenario.team_indices[player_id], scenario.player_colors[player_id], rise,
        player_art_for(scenario, player_id), reference_id in flag_hosts,
    )


def _accumulate_contribution(
    contribution: _SpriteContribution,
    by_anchor: dict[tuple[int, int], list],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    skip_ids: set[int],
    farm_by_tile: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]],
) -> None:
    """Merges one unit's contribution into a layer's four collections, in
    place. The wholesale walk and the chunk caches' unit splice both call it,
    so a splice that feeds units in walk order builds the walk's exact layer.
    Copies out of the contribution, never aliasing its lists.

    A by_anchor list is its unmoved entries in walk order, then its moved
    column slices (unit_sprites.MovedSlice, TASK-221) stable-sorted by their
    piece's own key in depth order: each moved slice paints after every
    unmoved entry at its tile, and moved ones keep the order they painted in
    when each sat at its own key (pasture slots are not in depth order in
    the piece list)."""
    if contribution.farm_tiles:
        # Later unit wins on overlap -- overlapping farms can't
        # happen in-game, but a scenario file can contain them, and a
        # byte-identity test needs a deterministic answer.
        farm_by_tile.update(contribution.farm_tiles)
    moved_type = unit_sprites.MovedSlice
    for key, piece_draws in contribution.by_anchor.items():
        merged = by_anchor.setdefault(key, [])
        for entry in piece_draws:
            draw = entry[0]
            if type(draw) is moved_type:
                depth = (draw.key[1] - draw.key[0], draw.key[0])
                at = len(merged)
                while at and type(prev := merged[at - 1][0]) is moved_type and (
                    prev.key[1] - prev.key[0], prev.key[0]
                ) > depth:
                    at -= 1
            elif not merged or type(merged[-1][0]) is not moved_type:
                merged.append(entry)
                continue
            else:
                at = len(merged) - 1
                while at and type(merged[at - 1][0]) is moved_type:
                    at -= 1
            merged.insert(at, entry)
        own = contribution.bboxes[key]
        bbox = bboxes.get(key)
        bboxes[key] = own if bbox is None else (
            min(bbox[0], own[0]),
            min(bbox[1], own[1]),
            max(bbox[2], own[2]),
            max(bbox[3], own[3]),
        )
    skip_ids.add(contribution.skip_id)


def sprite_draws_by_anchor_sliced(
    scenario: LoadedScenario,
    proj: iso_geometry.IsoProjection,
    elevations: np.ndarray,
    unit_filter: UnitFilter = UnitFilter(),
    corner_rise: np.ndarray | None = None,
    with_farms: bool = True,
    tree_scale: float = 1.0,
    hero_glow: bool = False,
    wall_alpha: float = 1.0,
    memo: SpriteMemo | None = None,
    fold: np.ndarray | None = None,
) -> Generator[object, None, SpriteLayer | tuple[SpriteLayer, SpriteMemo]]:
    """sprite_draws_by_anchor() as a resumable generator -- one yield per
    unit, so level_warm.LevelWarmer can advance it a few milliseconds at a
    time and resume on the next event-loop tick. Each yield hands out the
    unit about to be resolved (no allocation; every drain ignores it), which
    Perf Trace's level-warm `max step` names. See that function for what
    the result actually is; this is a pure control-flow extraction of its
    body, and it is the ONLY copy of that walk (the public function above is
    a thin drain of this one), so the two can't drift.

    The SpriteLayer is the generator's RETURN value (StopIteration.value),
    never a yielded one: a partially-built layer must never be observable,
    since installing one would leave units silently missing from the level.

    The yield sits at the TOP of the per-unit body, before the filter and
    bounds `continue`s, so granularity is one yield per unit even for a walk
    that skips most of them -- a filtered-out unit costs a resume, not a
    whole unbounded run. Batch D's D2: the per-unit resolution itself is
    _resolve_unit_sprite() -- this loop only does the yield and the
    accumulation into by_anchor/bboxes/skip_ids/farm_by_tile.

    memo (a SpriteMemo, possibly empty): reuse its contributions for units
    whose key is unchanged and return (layer, fresh memo) instead of the
    layer alone. None keeps the exact memo-free walk. The merge below copies
    out of every contribution, never aliasing its lists, so a memoized one
    stays untouched by later layer splices."""
    iso_geometry.require_fold(corner_rise, fold, "sprite_draws_by_anchor()")
    overrides = wall_variant_rotation_overrides(scenario)
    flag_hosts = garrison_host_ids(scenario)
    by_anchor: dict[tuple[int, int], list] = {}
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    skip_ids: set[int] = set()
    farm_by_tile: dict[tuple[int, int], tuple[int, tuple[int, int, int], int]] = {}
    if memo is not None:
        invariants = _sprite_memo_invariants(
            proj, scenario.map_manager, corner_rise, with_farms, tree_scale, hero_glow, wall_alpha,
        )
        old = memo.entries if memo.invariants == invariants else {}
        entries: dict[int, tuple] = {}
        sloped = corner_rise is not None
        # A snapshot: a warm that outlives an edit under a unit is refused at install.
        heights = (corner_rise if sloped else elevations).tolist()
        folds = fold.tolist() if sloped and fold is not None else None

    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            yield unit
            if memo is None:
                contribution = _resolve_unit_sprite(
                    scenario, proj, elevations, unit_filter, corner_rise, overrides, with_farms,
                    player_id, i, unit, tree_scale, hero_glow, wall_alpha, flag_hosts=flag_hosts, fold=fold,
                )
            else:
                if not unit_filter.matches(player_id, unit):
                    # Carried over unchecked, so re-showing a filter costs no resolves.
                    entry = old.get(id(unit))
                    if entry is not None:
                        entries[id(unit)] = entry
                    continue
                key = _sprite_memo_key(scenario, heights, sloped, overrides, player_id, i, unit, flag_hosts, folds)
                entry = old.get(id(unit))
                if entry is not None and entry[0] is unit and entry[1] == key:
                    contribution = entry[2]
                else:
                    contribution = _resolve_unit_sprite(
                        scenario, proj, elevations, unit_filter, corner_rise, overrides, with_farms,
                        player_id, i, unit, tree_scale, hero_glow, wall_alpha, flag_hosts=flag_hosts, fold=fold,
                    )
                entries[id(unit)] = (unit, key, contribution)
            if contribution is not None:
                _accumulate_contribution(contribution, by_anchor, bboxes, skip_ids, farm_by_tile)
    layer = SpriteLayer(
        by_anchor=by_anchor, bboxes=bboxes, skip_ids=frozenset(skip_ids), farm_by_tile=farm_by_tile
    )
    if memo is None:
        return layer
    return layer, SpriteMemo(invariants, entries)


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
    no-op extra paint, per composite_rect_iso's own accepted tradeoff.

    A drain of merge_sprite_bboxes_sliced(), the only copy of the loop."""
    return _drain(merge_sprite_bboxes_sliced(building_bboxes, sprites))


def merge_sprite_bboxes_sliced(
    building_bboxes: dict[tuple[int, int], tuple[int, int, int, int]], sprites: SpriteLayer
) -> Generator[None, None, dict[tuple[int, int], tuple[int, int, int, int]]]:
    """merge_sprite_bboxes() as a resumable generator, at most ASSEMBLY_SLICE
    sprite keys per step. The merged dict is the return value."""
    if not sprites.bboxes:
        return building_bboxes
    step = ASSEMBLY_SLICE
    merged = dict(building_bboxes)
    for seen, (key, bbox) in enumerate(sprites.bboxes.items()):
        if seen and seen % step == 0:
            yield
        prior = merged.get(key)
        merged[key] = bbox if prior is None else (
            min(prior[0], bbox[0]),
            min(prior[1], bbox[1]),
            max(prior[2], bbox[2]),
            max(prior[3], bbox[3]),
        )
    return merged
