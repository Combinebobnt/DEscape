"""Planner for wall runs (2026-09-19 wall-runs plan, GH #31/#50), placed by
Place Unit with a wall const picked since GH #98 folded the Wall Run tool in.

Qt-free and scenario-shaped, the same split cliff_chain.py has: everything a
wall drag will write is decided here, and viewer.py's _commit_wall_run() only
opens the undo record and reports. That is what lets the before/after pair
the in-game verification needs go through the real write path without
instantiating a window.

**Nothing is resolved mid-drag.** A node's neighbour set is not complete
until the path is, so the whole plan is computed once from the committed
tile list -- ChainStroke's accumulate-then-commit rule, for the same reason.

Two kinds of path tile are skipped (GH #124): a connector tile is still a
neighbour, an occupied ("blocked") tile is a gap and never a neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from descape import gate_orientation, unit_kind, unit_sprites
from descape.render import occupied_tiles_for, span_anchor, stored_rotation, unit_occupied_tiles, unit_tile_bounds

# A wall piece is 1x1 for all 9 eligible consts (verified against
# BUILDING_TILE_SPANS), which is why a run is a tile path and never a
# lattice walk the way a cliff chain is.
_WALL_SPAN = (1, 1)

# What an unresolved node stores. Measured: 24 of 24 isolated corpus walls in
# integer-encoded files store 2. wall_variant_from_neighbours8() deliberately
# answers None there instead, so that "a lone wall is a tower" stays a
# caller's policy rather than part of that function's contract -- this is the
# caller, and this is the policy.
ISOLATED_VARIANT = 2


@dataclass(frozen=True)
class WallNode:
    """One new piece the drag will add. `variant` is written to `rotation`
    as a literal integer, never a radian re-encoding: the game re-derives
    the index either way, DEscape's own render reads a literal verbatim in
    an integer-only file and re-derives it in a radian one, and writing a
    literal cannot flip a file's file_is_radian() classification (that test
    is "any non-literal value anywhere")."""

    unit_const: int
    variant: int
    x: float
    y: float

    # add_many() reads exactly these two names off a spec, so a WallNode is
    # one -- no second parallel dataclass to keep in step with this one.
    @property
    def rotation(self) -> float:
        return float(self.variant)

    @property
    def initial_animation_frame(self) -> int:
        """Always 0, measured: all 8193 corpus wall placements store 0 here.
        Walls are the opposite case from cliffs, where the tool copies
        `rotation` into this field because 18,232 of 18,232 records agree."""
        return 0


@dataclass(frozen=True)
class ExistingWall:
    """A wall already on the map that a new run may need to re-derive. Only
    rotation_variant_eligible() consts appear here: that is exactly
    set_wall_variant()'s own scope, so a plan can never name something the
    model would refuse."""

    unit: object
    player_id: int
    tx: int
    ty: int


@dataclass(frozen=True)
class WallPlan:
    nodes: list[WallNode]
    # (player_id, unit, new_variant) for each pre-existing wall the run
    # changes the shape of.
    rewrites: list[tuple[int, object, int]]
    # Path tiles that already held a connector, so nothing was placed there.
    skipped: int
    # Path tiles left empty because blocked_tiles() named them (GH #124).
    blocked: int = 0


@dataclass
class SceneCensus:
    """What scenario_move_plan() reads off wall_scene()'s walk besides its two
    answers (GH #266), so a move pays one walk, not two."""

    # id(unit) -> owner, for every connector the walk left out by exclude_ids.
    owners: dict[int, int] = field(default_factory=dict)
    # Eligible walls whose render.stored_rotation() is not a literal index,
    # excluded and off-map ones included: unit_sprites.file_is_radian()'s inputs.
    radian: int = 0

    def note(self, player_id: int, unit, excluded: bool) -> None:
        if excluded:
            self.owners[id(unit)] = player_id
        if unit_sprites.rotation_variant_eligible(unit.unit_const) and not unit_sprites.is_literal_variant_index(
            stored_rotation(player_id, unit), 5
        ):
            self.radian += 1


def wall_scene(
    scenario,
    map_width: int,
    map_height: int,
    exclude_ids: frozenset[int] = frozenset(),
    census: SceneCensus | None = None,
) -> tuple[set[tuple[int, int]], list[ExistingWall]]:
    """The two reads plan_wall_run() needs off a scenario: every tile any
    connector occupies (walls AND gates, since a gate is a neighbour like any
    other), and the eligible existing walls with their anchor tiles.

    Straight off `unit_manager`, never `map_view._unit_index` -- that pick
    index is current or None outside Units mode (it may survive there, but
    may be None), and existing_cliffs() documents the same trap.

    `exclude_ids` holds `id(unit)` values left out of both answers (GH #194):
    the units a move carries, whose old tiles are no longer theirs.
    `census`, when given, is filled in by the same walk (SceneCensus).
    """
    connector_consts = unit_sprites.wall_connector_consts()
    tiles: set[tuple[int, int]] = set()
    walls: list[ExistingWall] = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if unit.unit_const not in connector_consts:
                continue
            excluded = id(unit) in exclude_ids
            if census is not None:
                census.note(player_id, unit, excluded)
            if excluded:
                continue
            occupied = unit_occupied_tiles(unit, map_width, map_height)
            if occupied is not None:
                tiles.update(occupied)
            if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            bounds = unit_tile_bounds(unit, map_width, map_height)
            if bounds is not None:
                walls.append(ExistingWall(unit=unit, player_id=player_id, tx=bounds[0], ty=bounds[2]))
    return tiles, walls


def blocked_tiles(
    scenario, map_width: int, map_height: int, exclude_ids: frozenset[int] = frozenset()
) -> set[tuple[int, int]]:
    """Every tile a wall run with "Walls skip occupied tiles" on must leave
    empty (GH #124): any object's footprint, any owner, except passable eye
    candy and connectors (wall_scene()'s business). Objects a Filters toggle
    hides still block, since they are in the file. `exclude_ids` is
    wall_scene()'s: the `id(unit)` values of units a move carries away.

    Obstacles (rocks, ruins, statues, mountains: unit_kind.obstacle_consts(),
    GH #149) block on their whole footprint: a mountain's is 6x6 through
    terrain_palette.DOODAD_TILE_SPANS (TASK-134, which closed TASK-209).

    The sparse unit_occupied_tiles() set, the same oracle wall_scene() uses,
    never a bounding rect. A function of its own rather than a third
    wall_scene() return value, which would touch all of that one's callers.
    """
    skip = unit_sprites.wall_connector_consts() | unit_kind.passable_eye_candy_consts()
    tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const in skip or id(unit) in exclude_ids:
                continue
            occupied = unit_occupied_tiles(unit, map_width, map_height)
            if occupied is not None:
                tiles.update(occupied)
    return tiles


def plan_wall_run(
    tiles,
    *,
    unit_const: int,
    existing_tiles: set[tuple[int, int]],
    existing_walls,
    blocked_tiles=frozenset(),
) -> WallPlan:
    """Everything the commit will write, for a drag that named `tiles`.

    Pure: no scenario, no model, no Qt. `existing_tiles`/`existing_walls`
    come from wall_scene(), `blocked_tiles` from blocked_tiles() (empty with
    the toggle off).

    A path tile that already holds a connector is skipped rather than
    stacked, and still counts as a neighbour for everything around it -- a
    run drawn along an existing wall should reshape it, not double it.
    A blocked tile is skipped too, but it is a gap: it never enters the
    neighbour set, so the pieces either side resolve as run ends. The
    connector rule is checked first, so a tile in both counts as `skipped`.
    """
    unique = list(dict.fromkeys(tiles))
    free = [tile for tile in unique if tile not in existing_tiles]
    placed = [tile for tile in free if tile not in blocked_tiles]
    skipped = len(unique) - len(free)
    blocked = len(free) - len(placed)
    all_tiles = existing_tiles | set(placed)

    nodes = []
    for tx, ty in placed:
        mask = unit_sprites.neighbour_mask(tx, ty, all_tiles, diagonals=True)
        variant = unit_sprites.wall_variant_from_neighbours8(mask)
        if variant is None:
            variant = ISOLATED_VARIANT
        x, y = span_anchor(tx, ty, *_WALL_SPAN)
        nodes.append(WallNode(unit_const=unit_const, variant=variant, x=x, y=y))

    # Only walls the new run actually touches: anything further away has the
    # same neighbour set it had before, so re-deriving it would be noise.
    placed_set = set(placed)
    neighbourhood = {
        (tx + dx, ty + dy)
        for tx, ty in placed_set
        for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS
    }
    rewrites = _neighbour_rewrites(existing_walls, neighbourhood, placed_set, all_tiles)
    return WallPlan(nodes=nodes, rewrites=rewrites, skipped=skipped, blocked=blocked)


def _neighbour_rewrites(
    existing_walls, neighbourhood, exclude, all_tiles, isolated: int | None = None
) -> list[tuple[int, object, int]]:
    """(player_id, unit, new_variant) for every existing wall in
    `neighbourhood` but not in `exclude` whose stored shape disagrees with
    the one `all_tiles` derives. Shared by plan_wall_run(),
    plan_gate_over_walls(), plan_gate_moves() and plan_wall_moves(), so they
    cannot drift on the compare.

    A wall left with no neighbour at all derives no shape: it is skipped, or
    compared against `isolated` when that is given. Adding connectors never
    leaves a wall neighbourless, so only the two move planners pass it
    (ISOLATED_VARIANT, GH #266)."""
    rewrites: list[tuple[int, object, int]] = []
    for wall in existing_walls:
        if (wall.tx, wall.ty) not in neighbourhood or (wall.tx, wall.ty) in exclude:
            continue
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, all_tiles, diagonals=True)
        derived = unit_sprites.wall_variant_from_neighbours8(mask)
        if derived is None:
            if isolated is None:
                continue
            derived = isolated
        # Against the DECODED index, never the raw float: in a radian file
        # index 3 is stored as 3.769911, and a raw compare would rewrite
        # every junction wall to a value it already had -- pulling its owner
        # into the touched-player set to be snapshotted for nothing, and
        # making the undo record misreport what the edit changed.
        if unit_sprites.variant_index(wall.unit.rotation, 5) == derived:
            continue
        rewrites.append((wall.player_id, wall.unit, derived))
    return rewrites


@dataclass(frozen=True)
class GatePlan:
    """What a Place Unit click with a gate const changes besides adding the
    gate (GH #159): the walls under its footprint go, and the walls beside it
    are re-derived against the gate's tiles."""

    # (player_id, unit) for every wall-family unit anchored on a footprint tile.
    removals: list[tuple[int, object]]
    # (player_id, unit, new_variant), as WallPlan.rewrites.
    rewrites: list[tuple[int, object, int]]

    def __bool__(self) -> bool:
        return bool(self.removals or self.rewrites)


def walls_under(gate_tiles, existing_walls) -> list[tuple[int, object]]:
    """(player_id, unit) for every wall-family unit anchored on one of
    `gate_tiles`: what a gate placed there replaces (GH #159). Pure, and
    model-free, so Place Unit's hover asks it too (TASK-327)."""
    footprint = set(gate_tiles)
    return [(wall.player_id, wall.unit) for wall in existing_walls if (wall.tx, wall.ty) in footprint]


def plan_gate_over_walls(gate_tiles, *, existing_tiles: set[tuple[int, int]], existing_walls) -> GatePlan:
    """Everything placing a gate on `gate_tiles` changes in the walls around
    it, in-game's replace rule. Pure, like plan_wall_run().

    `gate_tiles` is the sparse render.occupied_tiles_for() footprint, so a
    diagonal gate's bounding-box corners keep their walls. Only the 9
    wall-family consts are removed (existing_walls holds nothing else);
    a gate already under the footprint stays. Every wall is 1x1, so its
    anchor tile is its whole footprint. Gate tiles stay connectors after the
    swap, so the post-edit connector set is existing_tiles plus the footprint.
    """
    footprint = set(gate_tiles)
    removals = walls_under(footprint, existing_walls)
    neighbourhood = {
        (tx + dx, ty + dy)
        for tx, ty in footprint
        for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS
    }
    rewrites = _neighbour_rewrites(existing_walls, neighbourhood, footprint, existing_tiles | footprint)
    return GatePlan(removals=removals, rewrites=rewrites)


def gate_run_line(tiles) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """The line a gate's sparse footprint runs along, low end first, and its
    unit step (GH #194). A 1-wide or 1-tall footprint is its own line. A
    diagonal gate's is the bounding-box diagonal lying wholly inside it: 4 of
    its 6 tiles, the other two being the thickening beside the middle.
    None for a single tile or a shape with no such line."""
    footprint = set(tiles)
    if len(footprint) < 2:
        return None
    x0, x1 = min(x for x, _y in footprint), max(x for x, _y in footprint)
    y0, y1 = min(y for _x, y in footprint), max(y for _x, y in footprint)
    if y0 == y1:
        return sorted(footprint), (1, 0)
    if x0 == x1:
        return sorted(footprint), (0, 1)
    if x1 - x0 != y1 - y0:
        return None
    # y grows with x on one diagonal (a NW-SE run) and falls on the other (NE-SW).
    candidates = (
        ([(x0 + i, y0 + i) for i in range(x1 - x0 + 1)], (1, 1)),
        ([(x0 + i, y1 - i) for i in range(x1 - x0 + 1)], (1, -1)),
    )
    inside = [(line, step) for line, step in candidates if footprint.issuperset(line)]
    return inside[0] if len(inside) == 1 else None


@dataclass(frozen=True)
class GateMove:
    """One gate a move carries onto different tiles (GH #194): its sparse
    render.occupied_tiles_for() footprint before and after."""

    old_tiles: frozenset[tuple[int, int]]
    new_tiles: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class GateMovePlan:
    """What moving gates changes in the walls around them with "Moving a gate
    updates walls" on (GH #194): the walls under each new footprint go, the
    line each gate left is filled with the wall it was set in, and the walls
    beside both sites are re-derived."""

    # (player_id, unit) for every stationary wall anchored on a gate's new tile.
    removals: list[tuple[int, object]]
    # (player_id, node) for every new wall on a tile a gate left.
    fills: list[tuple[int, WallNode]]
    # (player_id, unit, new_variant), as WallPlan.rewrites.
    rewrites: list[tuple[int, object, int]]
    # Vacated line tiles left open because blocked_tiles named them.
    blocked: int = 0

    def __bool__(self) -> bool:
        # `blocked` alone changes nothing: such a plan is falsy and still carries the count.
        return bool(self.removals or self.fills or self.rewrites)


def plan_gate_moves(
    gates,
    *,
    static_tiles: set[tuple[int, int]],
    static_walls,
    moved_tiles,
    blocked_tiles=frozenset(),
    removable=None,
) -> GateMovePlan:
    """Everything moving `gates` (GateMove) changes in the walls that stay
    put. Pure, like plan_wall_run().

    `static_tiles`/`static_walls` are wall_scene() WITHOUT the units the move
    carries, `moved_tiles` every moved connector's new footprint, and
    `blocked_tiles` what a fill may not land on. `removable(unit)` answers
    whether the model will let a wall go; one it would refuse stays in the
    scene and is re-derived like any other neighbour.

    A fill copies the const and owner of the stationary wall that continues
    the gate's own line past its low end, else past its high end. Collinear
    only, on purpose: any wider rule would fill again behind a gate nudged
    sideways off its run, leaving a trail. Along its own axis the fill does
    follow the gate, since each fill is the next move's source.

    A wall the gate leaves with no neighbour at all becomes a tower
    (ISOLATED_VARIANT), as plan_wall_moves() rules for one a wall leaves
    (GH #266), so the two parts of a move agree on every wall.
    """
    landed = {tile for gate in gates for tile in gate.new_tiles}
    vacated = {tile for gate in gates for tile in gate.old_tiles}
    removed = [
        wall for wall in static_walls
        if (wall.tx, wall.ty) in landed and (removable is None or removable(wall.unit))
    ]
    removed_ids = {id(wall.unit) for wall in removed}
    kept: dict[tuple[int, int], ExistingWall] = {}
    for wall in static_walls:
        if id(wall.unit) not in removed_ids:
            kept.setdefault((wall.tx, wall.ty), wall)

    base = set(static_tiles) | set(moved_tiles)
    sources: dict[tuple[int, int], ExistingWall] = {}
    left_open: set[tuple[int, int]] = set()
    for gate in gates:
        run = gate_run_line(gate.old_tiles)
        if run is None:
            continue
        line, (dx, dy) = run
        (lx, ly), (hx, hy) = line[0], line[-1]
        source = kept.get((lx - dx, ly - dy)) or kept.get((hx + dx, hy + dy))
        if source is None:
            continue
        for tile in line:
            if tile in base or tile in sources:
                continue
            if tile in blocked_tiles:
                left_open.add(tile)
            else:
                sources[tile] = source

    post = base | set(sources)
    fills = []
    for (tx, ty), source in sources.items():
        mask = unit_sprites.neighbour_mask(tx, ty, post, diagonals=True)
        variant = unit_sprites.wall_variant_from_neighbours8(mask)
        if variant is None:
            variant = ISOLATED_VARIANT
        x, y = span_anchor(tx, ty, *_WALL_SPAN)
        fills.append((source.player_id, WallNode(unit_const=source.unit.unit_const, variant=variant, x=x, y=y)))

    neighbourhood = {
        (tx + dx, ty + dy)
        for tx, ty in vacated | landed | set(sources)
        for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS
    }
    rewrites = _neighbour_rewrites(
        static_walls, neighbourhood, {(wall.tx, wall.ty) for wall in removed}, post, isolated=ISOLATED_VARIANT
    )
    return GateMovePlan(
        removals=[(wall.player_id, wall.unit) for wall in removed],
        fills=fills,
        rewrites=rewrites,
        blocked=len(left_open),
    )


def scenario_gate_moves(scenario, map_width: int, map_height: int, moves, *, removable=None) -> GateMovePlan | None:
    """plan_gate_moves() for `moves`, [(unit, new_x, new_y), ...], read off
    the scenario as it stands BEFORE the move. Qt-free. The viewer plans
    through scenario_move_plan(), whose gate part builds these same inputs;
    the before/after pair tool's gate-move case calls this directly.

    `moves` names every unit the move carries, garrison occupants included:
    none of them is a wall to remove, a fill source or a blocker where it
    stood. A moved connector still counts as a neighbour on its new tiles,
    and any other moved object blocks a fill on its new ones.

    None when no gate among them changes its tile set (a 0.1 nudge usually
    keeps the footprint): re-deriving the neighbours there would turn a plain
    move into a wall edit.
    """
    gates = _gate_moves(moves, map_width, map_height)
    if not gates:
        return None
    moved = frozenset(id(unit) for unit, _x, _y in moves)
    static_tiles, static_walls = wall_scene(scenario, map_width, map_height, exclude_ids=moved)
    blocked = blocked_tiles(scenario, map_width, map_height, exclude_ids=moved)
    moved_tiles = _moved_connector_tiles(moves, map_width, map_height, blocked)
    return plan_gate_moves(
        gates,
        static_tiles=static_tiles,
        static_walls=static_walls,
        moved_tiles=moved_tiles,
        blocked_tiles=blocked,
        removable=removable,
    )


def _gate_moves(moves, map_width: int, map_height: int) -> list[GateMove]:
    """A GateMove for every gate among `moves` whose tile set changes."""
    gates = []
    for unit, x, y in moves:
        if not gate_orientation.is_gate(unit.unit_const):
            continue
        old = unit_occupied_tiles(unit, map_width, map_height)
        new = occupied_tiles_for(unit.unit_const, x, y, map_width, map_height)
        if old is not None and new is not None and set(old) != set(new):
            gates.append(GateMove(old_tiles=frozenset(old), new_tiles=frozenset(new)))
    return gates


def _moved_connector_tiles(moves, map_width: int, map_height: int, blocked=None) -> set[tuple[int, int]]:
    """Every moved connector's new footprint. Given a `blocked` set, every
    other mover but passable eye candy adds its new tiles to it, so no fill
    lands there (GH #194)."""
    connectors = unit_sprites.wall_connector_consts()
    passable = unit_kind.passable_eye_candy_consts()
    moved_tiles: set[tuple[int, int]] = set()
    for unit, x, y in moves:
        const = unit.unit_const
        if const in connectors:
            moved_tiles.update(occupied_tiles_for(const, x, y, map_width, map_height) or ())
        elif blocked is not None and const not in passable:
            blocked.update(occupied_tiles_for(const, x, y, map_width, map_height) or ())
    return moved_tiles


def _wall_tile(unit_const: int, x: float, y: float, map_width: int, map_height: int) -> tuple[int, int] | None:
    """The one tile a wall at (x, y) stands on (every wall is 1x1), or None off the map."""
    tiles = occupied_tiles_for(unit_const, x, y, map_width, map_height)
    return tiles[0] if tiles else None


@dataclass(frozen=True)
class WallMove:
    """One wall a move carries onto another tile (GH #266). A wall moved
    within its own tile is none: it counts as staying."""

    unit: object
    player_id: int
    old_tile: tuple[int, int]
    new_tile: tuple[int, int]


def plan_wall_moves(
    walls, *, pre_tiles, post_tiles, staying_walls, exclude=frozenset()
) -> tuple[list[tuple[int, object, int]], list[tuple[int, object, int]]]:
    """(rewrites, moved), both (player_id, unit, new_variant) lists: what
    moving `walls` (WallMove) changes in wall shapes (GH #266). Pure, like
    plan_wall_run().

    `pre_tiles`/`post_tiles` are every connector tile before and after the
    move, `staying_walls` the walls that keep their tile (ExistingWall), and
    `exclude` the tiles whose walls are not re-derived (a moved gate's
    removals).

    `moved`: a carried wall takes the shape its new tile derives over
    `post_tiles` (a tower with no neighbour) whenever that differs from its
    decoded index. Always, not only when its mask changed, so a wall left
    stale by an earlier move is fixed by moving it again.

    `rewrites`: a staying wall is re-derived, with the same tower rule, only
    when a tile in its 8-neighbourhood changed connector status: a moved
    wall's old tile that holds no connector afterwards, or its new tile that
    held none before. A tile another connector still covers changes nobody's
    mask, and re-deriving there would rewrite a radian file's noise for nothing.
    """
    moved = []
    for move in walls:
        mask = unit_sprites.neighbour_mask(*move.new_tile, post_tiles, diagonals=True)
        derived = unit_sprites.wall_variant_from_neighbours8(mask)
        if derived is None:
            derived = ISOLATED_VARIANT
        # The decoded index, as _neighbour_rewrites() compares.
        if unit_sprites.variant_index(move.unit.rotation, 5) != derived:
            moved.append((move.player_id, move.unit, derived))
    triggers = {move.old_tile for move in walls if move.old_tile not in post_tiles}
    triggers |= {move.new_tile for move in walls if move.new_tile not in pre_tiles}
    if not triggers:
        return [], moved
    neighbourhood = {
        (tx + dx, ty + dy)
        for tx, ty in triggers
        for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS
    }
    rewrites = _neighbour_rewrites(staying_walls, neighbourhood, exclude, post_tiles, isolated=ISOLATED_VARIANT)
    return rewrites, moved


@dataclass(frozen=True)
class MovePlan:
    """What a move changes in the walls (GH #266), with GH #194's gate part
    when "Moving a gate updates walls" applies: scenario_move_plan()'s answer."""

    # The gate part's, as GateMovePlan's; empty without one.
    removals: list[tuple[int, object]]
    fills: list[tuple[int, WallNode]]
    # (player_id, unit, new_variant): both parts' rewrites of walls that keep their tile, each unit once.
    rewrites: list[tuple[int, object, int]]
    # (player_id, unit, new_variant) for each wall the move carries onto another tile.
    moved: list[tuple[int, object, int]]
    blocked: int = 0
    # Whether the plan overwrites or removes the file's last non-literal wall
    # values, flipping unit_sprites.file_is_radian() to integer.
    radian_flip: bool = False

    def __bool__(self) -> bool:
        # `blocked` alone changes nothing, as GateMovePlan's.
        return bool(self.removals or self.fills or self.rewrites or self.moved)

    @property
    def wholesale(self) -> bool:
        """Whether the move needs a whole-list undo record and a whole-canvas
        refresh: removals and fills shift list indices, and a flipped radian
        verdict changes how walls outside the edit draw."""
        return bool(self.removals or self.fills or self.radian_flip)


def scenario_move_plan(
    scenario, map_width: int, map_height: int, moves, *, gate_walls: bool, removable=None
) -> MovePlan | None:
    """plan_wall_moves(), plus plan_gate_moves() when `gate_walls`, for
    `moves`, [(unit, new_x, new_y), ...] as scenario_gate_moves() takes them,
    read off the scenario as it stands BEFORE the move in one scene walk.
    Qt-free, so the viewer and the pair tool plan through the same code.

    None, before any walk, when no wall changes its tile and no gate does
    (or `gate_walls` is off). A wall moving onto or off the map is no
    WallMove; one moved within its tile stays. The wall part counts every
    moved connector (a co-moved gate too) on its new tiles and the gate
    part's fills, and leaves the walls the gate part removes alone. Both
    parts derive against the same tiles with the same tower rule, so they
    agree on any wall both name.
    """
    gates = _gate_moves(moves, map_width, map_height) if gate_walls else []
    carried = []
    for unit, x, y in moves:
        if not unit_sprites.rotation_variant_eligible(unit.unit_const):
            continue
        old = _wall_tile(unit.unit_const, unit.x, unit.y, map_width, map_height)
        new = _wall_tile(unit.unit_const, x, y, map_width, map_height)
        if old is not None and new is not None and old != new:
            carried.append((unit, old, new))
    if not gates and not carried:
        return None
    moved_ids = frozenset(id(unit) for unit, _x, _y in moves)
    census = SceneCensus()
    static_tiles, static_walls = wall_scene(scenario, map_width, map_height, exclude_ids=moved_ids, census=census)
    blocked = blocked_tiles(scenario, map_width, map_height, exclude_ids=moved_ids) if gates else None
    moved_tiles = _moved_connector_tiles(moves, map_width, map_height, blocked)
    gate_plan = GateMovePlan(removals=[], fills=[], rewrites=[])
    if gates:
        gate_plan = plan_gate_moves(
            gates,
            static_tiles=static_tiles,
            static_walls=static_walls,
            moved_tiles=moved_tiles,
            blocked_tiles=blocked,
            removable=removable,
        )

    connectors = unit_sprites.wall_connector_consts()
    carried_ids = {id(unit) for unit, _old, _new in carried}
    pre_tiles = set(static_tiles)
    staying = list(static_walls)
    for unit, x, y in moves:
        if unit.unit_const not in connectors:
            continue
        pre_tiles.update(unit_occupied_tiles(unit, map_width, map_height) or ())
        if id(unit) in carried_ids or not unit_sprites.rotation_variant_eligible(unit.unit_const):
            continue
        tile = _wall_tile(unit.unit_const, x, y, map_width, map_height)
        if tile is not None and tile == _wall_tile(unit.unit_const, unit.x, unit.y, map_width, map_height):
            staying.append(ExistingWall(unit=unit, player_id=census.owners[id(unit)], tx=tile[0], ty=tile[1]))
    post_tiles = static_tiles | moved_tiles | {(int(node.x), int(node.y)) for _p, node in gate_plan.fills}
    removed_ids = {id(unit) for _p, unit in gate_plan.removals}
    rewrites, moved = plan_wall_moves(
        [WallMove(unit=unit, player_id=census.owners[id(unit)], old_tile=old, new_tile=new) for unit, old, new in carried],
        pre_tiles=pre_tiles,
        post_tiles=post_tiles,
        staying_walls=staying,
        exclude={(wall.tx, wall.ty) for wall in static_walls if id(wall.unit) in removed_ids},
    )
    merged = list(gate_plan.rewrites)
    named = {id(unit) for _p, unit, _v in merged}
    merged += [entry for entry in rewrites if id(entry[1]) not in named]

    radian_flip = False
    # Writes are literal, so only a written or removed non-literal value can flip the verdict.
    if census.radian:
        written = {id(unit): (player_id, unit) for player_id, unit, _v in (*merged, *moved)}
        written.update((id(unit), (player_id, unit)) for player_id, unit in gate_plan.removals)
        overwritten = sum(
            not unit_sprites.is_literal_variant_index(stored_rotation(player_id, unit), 5)
            for player_id, unit in written.values()
        )
        radian_flip = overwritten == census.radian
    return MovePlan(
        removals=gate_plan.removals,
        fills=gate_plan.fills,
        rewrites=merged,
        moved=moved,
        blocked=gate_plan.blocked,
        radian_flip=radian_flip,
    )


def gate_move_players(plan: GateMovePlan) -> set[int]:
    """Every owner `plan` touches: of a removed wall, a fill and a rewritten
    wall. The undo record's player set is this plus the moved units' owners.
    Not touched_players(): that takes the one placing player, and a group
    move has none."""
    return (
        {player_id for player_id, _unit in plan.removals}
        | {player_id for player_id, _node in plan.fills}
        | {player_id for player_id, _unit, _variant in plan.rewrites}
    )


def move_plan_players(plan: MovePlan) -> set[int]:
    """gate_move_players() for a MovePlan, plus every carried wall's owner (GH #266)."""
    return gate_move_players(plan) | {player_id for player_id, _unit, _variant in plan.moved}


def touched_players(player: int, plan: WallPlan | GatePlan) -> list[int]:
    """The complete player set the undo record must snapshot, which has to be
    known BEFORE begin_unit_edit(): add()/add_many() are refused inside a
    fields_only edit, so the whole commit runs as one fields_only=False edit
    over both halves. Deferring everything to release is what makes this
    knowable at all. A GatePlan adds every removed wall's owner."""
    owners = {player} | {player_id for player_id, _unit, _variant in plan.rewrites}
    if isinstance(plan, GatePlan):
        owners |= {player_id for player_id, _unit in plan.removals}
    return sorted(owners)


def apply_wall_plan(model, player: int, plan: WallPlan) -> list:
    """Writes `plan` through UnitEditModel. The caller owns the undo record
    (begin_unit_edit/commit, via viewer's _unit_edit) and must have opened it
    with touched_players()'s answer."""
    units = model.add_many(player, plan.nodes) if plan.nodes else []
    for _player_id, unit, variant in plan.rewrites:
        model.set_wall_variant(unit, variant)
    return units


def apply_gate_plan(model, player: int, gate_const: int, x: float, y: float, plan: GatePlan):
    """Removes, adds the gate, then rewrites; returns the new gate. Removing
    first keeps the gate last in its list. The caller owns the undo record and
    must have opened it with touched_players(player, plan)."""
    model.remove_many([unit for _player_id, unit in plan.removals])
    gate = model.add(player, gate_const, x, y)
    for _player_id, unit, variant in plan.rewrites:
        model.set_wall_variant(unit, variant)
    return gate


def apply_gate_move_plan(model, plan: GateMovePlan) -> list:
    """Adds `plan`'s fills, then rewrites; returns the new walls. The caller
    owns the undo record, opened over gate_move_players(plan) plus the
    movers' owners, and has already run remove_many() on plan.removals (the
    only step that can refuse, so it goes before anything moves) and set
    every position."""
    by_owner: dict[int, list[WallNode]] = {}
    for player_id, node in plan.fills:
        by_owner.setdefault(player_id, []).append(node)
    units = []
    for player_id, nodes in by_owner.items():
        units.extend(model.add_many(player_id, nodes))
    for _player_id, unit, variant in plan.rewrites:
        model.set_wall_variant(unit, variant)
    return units


def apply_move_plan(model, plan: MovePlan) -> list:
    """apply_gate_move_plan() for a MovePlan (GH #266): the fills, the
    rewrites, then each carried wall's new index; returns the new walls. The
    caller owns the undo record, opened over move_plan_players(plan) plus the
    movers' owners, and has already run remove_many() on plan.removals and
    set every position."""
    units = apply_gate_move_plan(model, plan)
    for _player_id, unit, variant in plan.moved:
        model.set_wall_variant(unit, variant)
    return units
