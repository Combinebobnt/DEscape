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
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import unit_sprites
from descape.render import span_anchor, unit_occupied_tiles, unit_tile_bounds

# A wall piece is 1x1 for all 8 eligible consts (verified against
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


def wall_scene(scenario, map_width: int, map_height: int) -> tuple[set[tuple[int, int]], list[ExistingWall]]:
    """The two reads plan_wall_run() needs off a scenario: every tile any
    connector occupies (walls AND gates, since a gate is a neighbour like any
    other), and the eligible existing walls with their anchor tiles.

    Straight off `unit_manager`, never `map_view._unit_index` -- that pick
    index is rebuilt only in Units mode, and existing_cliffs() documents the
    same trap.
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


def plan_wall_run(
    tiles,
    *,
    unit_const: int,
    existing_tiles: set[tuple[int, int]],
    existing_walls,
) -> WallPlan:
    """Everything the commit will write, for a drag that named `tiles`.

    Pure: no scenario, no model, no Qt. `existing_tiles`/`existing_walls`
    come from wall_scene().

    A path tile that already holds a connector is skipped rather than
    stacked, and still counts as a neighbour for everything around it -- a
    run drawn along an existing wall should reshape it, not double it.
    """
    placed = [tile for tile in dict.fromkeys(tiles) if tile not in existing_tiles]
    skipped = len(dict.fromkeys(tiles)) - len(placed)
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
    rewrites: list[tuple[int, object, int]] = []
    for wall in existing_walls:
        if (wall.tx, wall.ty) not in neighbourhood or (wall.tx, wall.ty) in placed_set:
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
    return WallPlan(nodes=nodes, rewrites=rewrites, skipped=skipped)


def touched_players(player: int, plan: WallPlan) -> list[int]:
    """The complete player set the undo record must snapshot, which has to be
    known BEFORE begin_unit_edit(): add()/add_many() are refused inside a
    fields_only edit, so the whole commit runs as one fields_only=False edit
    over both halves. Deferring everything to release is what makes this
    knowable at all."""
    return sorted({player} | {player_id for player_id, _unit, _variant in plan.rewrites})


def apply_wall_plan(model, player: int, plan: WallPlan) -> list:
    """Writes `plan` through UnitEditModel. The caller owns the undo record
    (begin_unit_edit/commit, via viewer's _unit_edit) and must have opened it
    with touched_players()'s answer."""
    units = model.add_many(player, plan.nodes) if plan.nodes else []
    for _player_id, unit, variant in plan.rewrites:
        model.set_wall_variant(unit, variant)
    return units
