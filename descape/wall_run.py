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

from dataclasses import dataclass

from descape import unit_kind, unit_sprites
from descape.render import span_anchor, unit_occupied_tiles, unit_tile_bounds

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


def wall_scene(scenario, map_width: int, map_height: int) -> tuple[set[tuple[int, int]], list[ExistingWall]]:
    """The two reads plan_wall_run() needs off a scenario: every tile any
    connector occupies (walls AND gates, since a gate is a neighbour like any
    other), and the eligible existing walls with their anchor tiles.

    Straight off `unit_manager`, never `map_view._unit_index` -- that pick
    index is current or None outside Units mode (it may survive there, but
    may be None), and existing_cliffs() documents the same trap.
    """
    connector_consts = unit_sprites.wall_connector_consts()
    tiles: set[tuple[int, int]] = set()
    walls: list[ExistingWall] = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if unit.unit_const not in connector_consts:
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


def blocked_tiles(scenario, map_width: int, map_height: int) -> set[tuple[int, int]]:
    """Every tile a wall run with "Walls skip occupied tiles" on must leave
    empty (GH #124): any object's footprint, any owner, except passable eye
    candy and connectors (wall_scene()'s business). Objects a Filters toggle
    hides still block, since they are in the file.

    Obstacles (rocks, ruins, statues, mountains: unit_kind.obstacle_consts(),
    GH #149) block, but only on their 1x1 anchor tile: like every
    non-building, unit_occupied_tiles() gives them NON_BUILDING_SPAN, so a
    run can pass through a 3x3 mountain's outer tiles. Accepted limitation;
    real spans would also change drawing and picking (unit_kind docstring).

    The sparse unit_occupied_tiles() set, the same oracle wall_scene() uses,
    never a bounding rect. A function of its own rather than a third
    wall_scene() return value, which would touch all of that one's callers.
    """
    skip = unit_sprites.wall_connector_consts() | unit_kind.passable_eye_candy_consts()
    tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const in skip:
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


def _neighbour_rewrites(existing_walls, neighbourhood, exclude, all_tiles) -> list[tuple[int, object, int]]:
    """(player_id, unit, new_variant) for every existing wall in
    `neighbourhood` but not in `exclude` whose stored shape disagrees with
    the one `all_tiles` derives. Shared by plan_wall_run() and
    plan_gate_over_walls(), so the two cannot drift on the compare."""
    rewrites: list[tuple[int, object, int]] = []
    for wall in existing_walls:
        if (wall.tx, wall.ty) not in neighbourhood or (wall.tx, wall.ty) in exclude:
            continue
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, all_tiles, diagonals=True)
        derived = unit_sprites.wall_variant_from_neighbours8(mask)
        if derived is None:
            continue
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
    removals = [(wall.player_id, wall.unit) for wall in existing_walls if (wall.tx, wall.ty) in footprint]
    neighbourhood = {
        (tx + dx, ty + dy)
        for tx, ty in footprint
        for dx, dy, _bit in unit_sprites.NEIGHBOUR_OFFSETS
    }
    rewrites = _neighbour_rewrites(existing_walls, neighbourhood, footprint, existing_tiles | footprint)
    return GatePlan(removals=removals, rewrites=rewrites)


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
