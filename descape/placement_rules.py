"""The two restrictions "Free placement" lifts (GH #176/#177, GH #174): the
terrain an object may stand on and the objects it may not overlap. Qt-free.
viewer.py asks refusal() for a Place Unit click and move_refusal() for a
drag-move: wall runs, nudges, Scatter, paste and the batch API stay
unrestricted. refused() and move_refused() give the same answers without
the message, for callers that ask per candidate (the Unit Brush, GH #170;
the Place Unit hover ghost and the drag previews, TASK-327; a group drop)
and build one message at most. Map Analysis (GH #173) is the second reader: it applies
terrain_refusal() and BlockerGrid to the units a file already holds. Both
read a mobile unit's terrain through laid_terrain(), so a unit on a bridge
is not on water (TASK-349); refusal() reads it memoized per unit edit.

Both rules read the game's own .dat, committed in object_catalog.json.

**Terrain.** An object's `restriction` names a .dat terrain-restriction row,
and every tile of its footprint must be one that row passes. The footprint
is render.occupied_tiles_for(), i.e. the tiles DEscape draws it on, so an
off-grid building is judged on its quantised tiles. An object with
`side_terrain` (docks, shore fish, box turtles) also needs a shore in the
ring of tiles around its footprint. "A shore" is the two literal ids (Beach
2, Ice 35) widened to the .dat's Beach row, because real files put shore fish
beside every beach type: 89.8% of 1018 corpus shore fish match the literal
ids and 95.0% match the row (measured 2026-10-12).

**Objects.** The game's clearance boxes rather than tile occupancy, so two
villagers can share a tile off the grid. boxes() is the whole rule. What is
solid is the .dat's non-zero obstruction_type plus the buildings, because
156 of the 491 BUILDING_TILE_SPANS consts carry obstruction 0. Of those,
_PASSABLE_BUILDING_CLASSES names the ones that really are passable; the 15
Town Centers and the Wonder shell (class 3) stay solid, since the game
obstructs them through annexes; and the rest (trophies, spawners, empty
annexes) are 1x1 with no clearance, so they have no shape either way. Fish
have obstruction 0 and are not buildings, so fish and farms both stack. A
class-14 doodad with a footprint (DOODAD_TILE_SPANS: mountains, ruins,
TASK-134) keeps its clearance box rather than its span: it is the game's own
rule, equals the span for 35 of the 46, and is what lets mountains overlap as
real maps have them (a mountain's box is 4x4 inside its 6x6 footprint). The
terrain rule reads the whole drawn footprint, as for every object.

Unknown data never refuses: a const or a terrain id the catalog lacks passes.
Units a Filters toggle hides still block, as in wall_run.blocked_tiles(), and
the message names the blocker and where it stands so a hidden one is findable.
"""

from __future__ import annotations

import math
import weakref

from descape import game_dataset, gate_orientation, object_catalog, render, terrain_catalog
from descape.terrain_palette import (
    BUILDING_TILE_OFFSETS,
    BUILDING_TILE_SPANS,
    DOODAD_TILE_SPANS,
    FOUNDATION_TERRAIN,
    name_for_terrain_id,
    tile_span,
)
from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_kind import mobile_unit_consts

# The .dat's Beach restriction row. No object uses it; it is what "a shore" widens to.
BEACH_RESTRICTION = 2

# Shapes that share an edge do not overlap. Coordinates are float32 in the file.
_SLACK = 1e-3

# .dat class -> why an obstruction-0 building of that class really is passable.
# Every other obstruction-0 building stays solid (module docstring).
_PASSABLE_BUILDING_CLASSES = {
    49: "farms, rice farms, pastures and fish traps: units walk and ships sail over them",
    11: "what a dead farm, pasture or fish trap leaves behind",
    14: "bridges, moles and a pasture's loose fence pieces: decks and litter units walk over",
    39: "the open state of every gate",
}

# (x0, y0, x1, y1) in continuous tile coordinates.
Box = tuple[float, float, float, float]


@game_dataset.cache(maxsize=1)
def _objects() -> dict[int, dict]:
    return object_catalog.dat_objects()


@game_dataset.cache(maxsize=1)
def _restrictions() -> dict[int, frozenset[int]]:
    return object_catalog.dat_terrain_restrictions()


@game_dataset.cache(maxsize=1)
def _terrains() -> dict[int, dict]:
    return object_catalog.dat_terrains()


def _terrain_name(terrain_id: int) -> str:
    """The game's own terrain name with an install, else the Terrain panel's label."""
    label = terrain_catalog.display_name(name_for_terrain_id(terrain_id))
    return object_catalog.resolve_name(terrain_id, label, _terrains().get(terrain_id), section="terrains")


# --- terrain -------------------------------------------------------------------


def laid_terrain(scenario) -> dict[tuple[int, int], frozenset[int]]:
    """tile -> the foundation terrain ids (FOUNDATION_TERRAIN) of the
    buildings standing on it, garrisoned ones excluded. A bridge lays 28
    (`Water 2D, Bridge`), which the land rows pass, over the shallow water or
    beach a file stores under it. No corpus mobile unit stands on a bridge
    tile, so what this changes rests on the .dat, not on a measurement."""
    mm = scenario.map_manager
    laid: dict[tuple[int, int], set[int]] = {}
    for units in scenario.unit_manager.units:
        for unit in units:
            terrain_id = FOUNDATION_TERRAIN.get(unit.unit_const)
            if terrain_id is None or unit.garrisoned_in_id != -1:
                continue
            tiles = render.occupied_tiles_for(unit.unit_const, unit.x, unit.y, mm.map_width, mm.map_height)
            for tile in tiles or ():
                laid.setdefault(tile, set()).add(terrain_id)
    return {tile: frozenset(ids) for tile, ids in laid.items()}


# scenario -> ((unit_gen, map width, map height), laid_terrain()).
_LAID_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _memoized_laid_terrain(scenario) -> dict[tuple[int, int], frozenset[int]]:
    """laid_terrain(), built once per unit edit (scenario.unit_gen, which every
    UnitEditModel mutator bumps) rather than per click or per group-move
    member. A scenario without unit_gen (a test fake) builds it fresh."""
    gen = getattr(scenario, "unit_gen", None)
    if gen is None:
        return laid_terrain(scenario)
    mm = scenario.map_manager
    key = (gen, mm.map_width, mm.map_height)
    cached = _LAID_MEMO.get(scenario)
    if cached is not None and cached[0] == key:
        return cached[1]
    laid = laid_terrain(scenario)
    _LAID_MEMO[scenario] = (key, laid)
    return laid


def _footprint_block(scenario, unit_const: int, tiles, laid=None) -> int | None:
    """The terrain id of the first footprint tile `unit_const`'s restriction
    row does not pass, or None. A tile whose stored terrain is refused still
    passes when `laid` (laid_terrain()) holds one the row passes."""
    passable = _restrictions().get(_objects().get(unit_const, {}).get("restriction"))
    if passable is None:
        return None
    mm = scenario.map_manager
    known = _terrains()
    for tx, ty in tiles:
        terrain_id = mm.terrain[ty * mm.map_width + tx].terrain_id
        if terrain_id in known and terrain_id not in passable:
            if laid and not passable.isdisjoint(laid.get((tx, ty), ())):
                continue
            return terrain_id
    return None


def _footprint_refusal(scenario, unit_const: int, tiles, laid=None) -> str | None:
    """_footprint_block() as a message, or None."""
    terrain_id = _footprint_block(scenario, unit_const, tiles, laid)
    if terrain_id is None:
        return None
    return f"{object_catalog.display_name(unit_const)} can't stand on {_terrain_name(terrain_id)}"


def _shore_block(scenario, unit_const: int, tiles) -> bool:
    """Whether a `side_terrain` const has no shore in the ring of tiles around
    its footprint. An unknown terrain id counts as shore."""
    side_terrain = _objects().get(unit_const, {}).get("side_terrain")
    if not side_terrain:
        return False
    shore = frozenset(side_terrain) | _restrictions().get(BEACH_RESTRICTION, frozenset())
    mm = scenario.map_manager
    width, height = mm.map_width, mm.map_height
    known = _terrains()
    footprint = set(tiles)
    ring = {
        (tx + dx, ty + dy)
        for tx, ty in footprint
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if 0 <= tx + dx < width and 0 <= ty + dy < height
    } - footprint
    for tx, ty in ring:
        terrain_id = mm.terrain[ty * width + tx].terrain_id
        if terrain_id in shore or terrain_id not in known:
            return False
    return True


def _shore_refusal(scenario, unit_const: int, tiles) -> str | None:
    """_shore_block() as a message, or None."""
    if not _shore_block(scenario, unit_const, tiles):
        return None
    return f"{object_catalog.display_name(unit_const)} needs a beach tile beside it"


def terrain_refusal(scenario, unit_const: int, x: float, y: float, laid=None) -> str | None:
    """Why `unit_const` cannot stand at (x, y) on this map's terrain, or None.
    An off-map point is not this rule's to refuse. `laid` is laid_terrain()'s
    map; refusal() passes it for a mobile unit only, as Map Analysis does."""
    mm = scenario.map_manager
    tiles = render.occupied_tiles_for(unit_const, x, y, mm.map_width, mm.map_height)
    if tiles is None:
        return None
    return _footprint_refusal(scenario, unit_const, tiles, laid) or _shore_refusal(scenario, unit_const, tiles)


def _terrain_blocked(scenario, unit_const: int, x: float, y: float, laid=None) -> bool:
    """terrain_refusal() is not None, without building the message."""
    mm = scenario.map_manager
    tiles = render.occupied_tiles_for(unit_const, x, y, mm.map_width, mm.map_height)
    if tiles is None:
        return False
    return _footprint_block(scenario, unit_const, tiles, laid) is not None or _shore_block(scenario, unit_const, tiles)


# --- objects -------------------------------------------------------------------


def _shape(unit_const: int):
    """`unit_const`'s solid shape, position-free: (span, tile offsets, None)
    for a diagonal gate, (span, None, half-extents) for a box, None for an
    object nothing collides with.

    Read live on every call, never cached per const: tests inject synthetic
    spans into BUILDING_TILE_SPANS, which is why tile_span() is a function."""
    entry = _objects().get(unit_const)
    if entry is None:
        return None
    if not entry.get("obstruction") and (
        unit_const not in BUILDING_TILE_SPANS or entry.get("class") in _PASSABLE_BUILDING_CLASSES
    ):
        return None
    span = tile_span(unit_const, render.NON_BUILDING_SPAN)
    offsets = BUILDING_TILE_OFFSETS.get(unit_const)
    if offsets is not None:
        return span, offsets, None
    if span != render.NON_BUILDING_SPAN and unit_const not in DOODAD_TILE_SPANS:
        # DEscape's span, not clearance: 2 * clearance equals it for 419 of 491 buildings, and gates differ.
        return span, None, (span[0] / 2, span[1] / 2)
    clearance = entry.get("clearance")
    if not clearance:
        return None
    return span, None, (clearance[0], clearance[1])


def _boxes_for(shape, x: float, y: float) -> tuple[Box, ...]:
    if shape is None:
        return ()
    span, offsets, half = shape
    if offsets is None:
        return ((x - half[0], y - half[1], x + half[0], y + half[1]),)
    # One unit square per real tile, so a diagonal gate's empty corners neither block nor are blocked.
    x0, y0 = render._span_start(x, span[0]), render._span_start(y, span[1])
    return tuple((x0 + ox, y0 + oy, x0 + ox + 1, y0 + oy + 1) for ox, oy in sorted(offsets))


def boxes(unit_const: int, x: float, y: float) -> tuple[Box, ...]:
    """The solid shape of a `unit_const` standing at (x, y), or () for an
    object that neither blocks nor is blocked.

    Solid means the catalog's `obstruction` key (the .dat's own non-zero
    obstruction_type) or membership of BUILDING_TILE_SPANS, minus
    _PASSABLE_BUILDING_CLASSES. Fish are neither, so they stack. A diagonal
    gate is one unit square per footprint tile; any other footprint wider
    than 1x1 is one box of its span centred on (x, y), except a doodad's;
    everything else, doodads included, is its `clearance` box, and nothing
    when that is zero (birds, butterflies).
    """
    return _boxes_for(_shape(unit_const), x, y)


def _overlap(mine: tuple[Box, ...], theirs: tuple[Box, ...]) -> bool:
    for ax0, ay0, ax1, ay1 in mine:
        for bx0, by0, bx1, by1 in theirs:
            if ax0 < bx1 - _SLACK and bx0 < ax1 - _SLACK and ay0 < by1 - _SLACK and by0 < ay1 - _SLACK:
                return True
    return False


def _reach(shape) -> float:
    """How far `shape` can extend from its own (x, y) on either axis."""
    span, offsets, half = shape
    # A tile-offset shape sits on quantised tiles, up to half a tile off its anchor.
    return max(span) / 2 + 0.5 if offsets is not None else max(half)


@game_dataset.cache(maxsize=1)
def _max_reach() -> float:
    """_reach() of the largest shape in the catalog: blocker()'s prefilter bound."""
    return max((_reach(shape) for shape in map(_shape, _objects()) if shape is not None), default=0.0)


def blocker(scenario, unit_const: int, x: float, y: float, exclude=()):
    """(player_id, unit) of the first unit a `unit_const` at (x, y) would
    overlap, in list order, or None. `exclude` is by identity. A garrisoned
    unit never blocks: it is inside its host, not on the map.

    One linear scan: 2 ms on the largest corpus file, so a click needs no
    index. A group move asks BlockerGrid instead."""
    shape = _shape(unit_const)
    if shape is None:
        return None
    mine = _boxes_for(shape, x, y)
    excluded = {id(unit) for unit in exclude}
    limit = _reach(shape) + _max_reach()
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if abs(unit.x - x) > limit or abs(unit.y - y) > limit:
                continue
            if unit.garrisoned_in_id != -1 or id(unit) in excluded:
                continue
            if _overlap(mine, boxes(unit.unit_const, unit.x, unit.y)):
                return player_id, unit
    return None


def _cells(shape_boxes: tuple[Box, ...]):
    for x0, y0, x1, y1 in shape_boxes:
        for cy in range(math.floor(y0), math.floor(y1) + 1):
            for cx in range(math.floor(x0), math.floor(x1) + 1):
                yield cx, cy


class BlockerGrid:
    """blocker() for many movers against one unit list: one scan into a tile
    hash, then a lookup per mover. A group move builds one with every member
    excluded. Answers exactly as blocker() does, the same unit included.

    Rows order by (player_id, index in that player's list), blocker()'s scan
    order, so a unit add() learns after an append to its player's list sorts
    exactly where a fresh build would put it."""

    def __init__(self, scenario, exclude=()):
        excluded = {id(unit) for unit in exclude}
        self._shapes: dict[int, object] = {}
        self._cells: dict[tuple[int, int], list] = {}
        self._next_index: dict[int, int] = {}
        for player_id, units in enumerate(scenario.unit_manager.units):
            self._next_index[player_id] = len(units)
            for index, unit in enumerate(units):
                if unit.garrisoned_in_id != -1 or id(unit) in excluded:
                    continue
                self._insert((player_id, index), player_id, unit)

    def _insert(self, order: tuple[int, int], player_id: int, unit) -> None:
        const = unit.unit_const
        if const not in self._shapes:
            self._shapes[const] = _shape(const)
        theirs = _boxes_for(self._shapes[const], unit.x, unit.y)
        if not theirs:
            return
        row = (order, player_id, unit, theirs)
        for cell in set(_cells(theirs)):
            self._cells.setdefault(cell, []).append(row)

    def add(self, player_id: int, unit) -> None:
        """Learns `unit`, just appended to `player_id`'s list (UnitEditModel.add())."""
        index = self._next_index.get(player_id, 0)
        self._next_index[player_id] = index + 1
        if unit.garrisoned_in_id == -1:
            self._insert((player_id, index), player_id, unit)

    def blocker(self, unit_const: int, x: float, y: float, exclude=()):
        mine = boxes(unit_const, x, y)
        excluded = {id(unit) for unit in exclude}
        first = None
        for cell in set(_cells(mine)):
            for row in self._cells.get(cell, ()):
                if first is not None and row[0] >= first[0]:
                    continue
                if id(row[2]) not in excluded and _overlap(mine, row[3]):
                    first = row
        return None if first is None else (first[1], first[2])


# --- both ----------------------------------------------------------------------


def refusal(
    scenario, unit_const: int, x: float, y: float, exclude=(), grid: BlockerGrid | None = None, *, laid=None
) -> str | None:
    """Why a `unit_const` cannot go at (x, y), terrain first and then the
    object in the way, as a whole clause for a status line, or None.
    `grid` replaces the scan; its own build-time exclusions still apply.
    A mobile unit's terrain counts a building's foundation (laid_terrain()),
    read only once the stored terrain refuses; `laid` is as refused()'s."""
    reason = terrain_refusal(scenario, unit_const, x, y)
    if reason is not None and unit_const in mobile_unit_consts():
        reason = terrain_refusal(
            scenario, unit_const, x, y, _memoized_laid_terrain(scenario) if laid is None else laid
        )
    if reason is not None:
        return reason
    if grid is not None:
        found = grid.blocker(unit_const, x, y, exclude)
    else:
        found = blocker(scenario, unit_const, x, y, exclude)
    if found is None:
        return None
    player_id, unit = found
    owner = "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}"
    return f"blocked by {object_catalog.display_name(unit.unit_const)} ({owner}) at ({unit.x:g}, {unit.y:g})"


def gate_companions(scenario, unit) -> list:
    """The corner pieces a gate `unit` keeps as separate solid units, or []
    (TASK-340). Real files store them on the gate's own end tiles: all 32
    corpus corners (GTBC2 95, both ends of a GTBA2 88, 1.5 tiles off its
    point) share their gate's owner, family and orientation (2026-10-29).
    A piece counts when it is gate_orientation.corner_piece()'s const, in
    the gate's owner list, with its point inside the gate's own boxes()."""
    corner = gate_orientation.corner_piece(unit.unit_const)
    if corner is None:
        return []
    own = next((units for units in scenario.unit_manager.units if any(u is unit for u in units)), None)
    if own is None:
        return []
    shape = boxes(unit.unit_const, unit.x, unit.y)
    return [
        piece
        for piece in own
        if piece.unit_const == corner
        and any(x0 <= piece.x < x1 and y0 <= piece.y < y1 for x0, y0, x1, y1 in shape)
    ]


def move_refusal(
    scenario, unit, x: float, y: float, exclude=(), grid: BlockerGrid | None = None, *, unit_const: int | None = None
) -> str | None:
    """refusal() for an existing `unit` going to (x, y). A garrisoned unit
    skips both rules: it stands wherever its host does, and with Show
    Garrisoned on it can be dragged alone or inside a selection. A gate's
    own corner pieces never block its move (gate_companions(), TASK-340).

    `unit_const` is the const it lands as, for a gate a drag turned (GH #280).
    `unit` stays the real one: gate_companions() finds it by identity."""
    if unit.garrisoned_in_id != -1:
        return None
    const = unit.unit_const if unit_const is None else unit_const
    return refusal(scenario, const, x, y, (*exclude, *gate_companions(scenario, unit)), grid)


def refused(
    scenario, unit_const: int, x: float, y: float, exclude=(), grid: BlockerGrid | None = None, *, laid=None
) -> bool:
    """refusal() is not None, without building a message: no display_name()
    or _terrain_name() call, which re-parses the language file per call.
    `laid` is a caller-held laid_terrain() map in place of the per-edit memo,
    for a caller that adds units between calls (the Unit Brush), since every
    add() bumps unit_gen and the memo would rebuild (~80 ms on 32k units)."""
    if _terrain_blocked(scenario, unit_const, x, y) and (
        unit_const not in mobile_unit_consts()
        or _terrain_blocked(
            scenario, unit_const, x, y, _memoized_laid_terrain(scenario) if laid is None else laid
        )
    ):
        return True
    if grid is not None:
        return grid.blocker(unit_const, x, y, exclude) is not None
    return blocker(scenario, unit_const, x, y, exclude) is not None


def move_refused(
    scenario, unit, x: float, y: float, exclude=(), grid: BlockerGrid | None = None, *, unit_const: int | None = None
) -> bool:
    """move_refusal() is not None, without building a message."""
    if unit.garrisoned_in_id != -1:
        return False
    const = unit.unit_const if unit_const is None else unit_const
    return refused(scenario, const, x, y, (*exclude, *gate_companions(scenario, unit)), grid)
