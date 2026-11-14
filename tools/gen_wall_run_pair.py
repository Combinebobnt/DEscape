#!/usr/bin/env python3
"""Writes before/after .aoe2scenario pairs for confirming wall runs (Place
Unit with a wall picked, GH #98) in the real AoE2:DE editor -- the half of testing nothing in a terminal
session can drive. Both sides go through the real write path (UnitEditModel
+ scenario_write.write_scenario(), the same code the GUI uses), never a
hand-crafted byte patch.

Three edits, all deliberately asymmetric so "did anything change" becomes
"does this specific shape read back right":

  1. An L-shaped run whose far end T-junctions into an existing wall, in an
     integer-encoded file.
  2. The same shape in a radian-encoded file, so both encoding branches are
     checkable.
  3. A shallow drag, so wall_path_tiles()' two-segment split is visible and
     the diagonal-first guess is falsifiable.
  4. The L again, GAIA-owned (GH #98 dropped the GAIA refusal), in both an
     integer and a radian file: does the game keep a GAIA run's shapes?
  5. A Wall Rectangle ring (9x5, deliberately not square), in both an integer
     and a radian file. One of its short sides crosses the anchor wall's run,
     so the ring shares the anchor's tile (and any run tile under its far
     side) and shows a junction rewrite.
  6. Two Stone Gates placed over existing walls (GH #159), in both an
     integer and a radian file: one clicked onto the middle of a straight
     run along x, one across a run's bend where exactly one neighbouring
     wall reshapes. The walls under each gate are removed.
  7. A Stone Gate moved 5 tiles along its run with "Moving a gate updates
     walls" on (GH #194), in both an integer and a radian file. Here the
     "before" already holds the gate, placed as in 6. The "after" has the four
     walls under its new footprint removed and the four tiles it left filled
     with the run's own wall.
  8. A Wall Circle ring (GH #125: a 15x11 ellipse, deliberately not a
     circle), in both an integer and a radian file, placed as 5's ring is:
     its side apex sits on the anchor wall, so it shows a junction too. Its
     diagonal stretches are the shapes 3 and 4 in a closed ring.
  9. A wall moved MOVE_ROWS rows out of the middle of a straight x run (GH
     #266), in both an integer and a radian file, as three files: the
     untouched "before", a "stale" one with the move alone (set_position(),
     what DEscape wrote until GH #266) and the "after", planned and applied
     as the viewer's move does. In the after file the moved wall is a tower
     at its new tile and the two walls either side of the gap are run ends
     (also index 2); in the stale file all three keep their old index.
 10. A Stone Gate cycled from x to y (GH #339), in both an integer and a
     radian file. The "before" holds the x gate placed across a y run as in
     6 (one wall removed, three tiles sticking out); the "after" jumps it two
     orientations, as the Rotation field does, onto the run: the three walls
     under its new tiles are removed and the tiles it left stay empty.
 11. A mid-run wall replaced by a Stone Gate in Find and Replace (GH #339),
     in both an integer and a radian file. The gate keeps the wall's
     reference_id, covers it and the next three walls, which are removed,
     and the walls beside it keep the shape their neighbours give them.
 12. A wall deleted from the middle of a straight x run (GH #340), in both
     an integer and a radian file, as three files: the untouched "before",
     a "stale" one with the removal alone (what Delete wrote until GH #340)
     and the "after", with the two walls either side of the gap rewritten
     from a run piece (0) to a run end (2).
 13. Unit Line and Unit Shape (GH #365), in both an integer and a radian
     file, one undo record per shape as the tools commit them: a Shift
     Triangle of Stone Wall whose base T-joins the anchor wall's run, then on
     clear ground a Shift Right Triangle, a Rectangle and an Ellipse of Stone
     Wall (the old Wall Rectangle and Wall Circle checks) and an outline
     rectangle of Houses at Spacing 1. The wall shapes go through
     unit_shape.lattice_cells() and plan_wall_run(), the Houses through
     unit_shape.place_cells() with the game's placement rules on.
 14. A single wall meeting a diagonal run's interior (TASK-581), integer
     file only, as three files. On clear ground, three diagonal runs of 5
     placed through plan_wall_run(): site A a NE-SW run with one stub N of
     its middle, site B a NW-SE run with one stub W of its middle, site C a
     second NW-SE run with two stubs, N and E of its middle (a corner plus
     the pair). "before" holds the runs only, "stale" adds the stubs planned
     with the pre-TASK-581 table (both junction pieces at A and at B store
     2) and "after" the stubs as shipped (A's keep 4, B's keep 3, C's middle
     stores 2 either way). Every junction tile's stored index is printed
     per file.
 15. Find and Replace out of and back into a wall (TASK-584), in both an
     integer and a radian file, as four files at 12's site (a mid x-run
     wall whose two neighbours decode to 0): "before"; "stale", the wall
     replaced by an Outpost with replace_type() alone; "out", the same
     through scenario_replace_rewrites() and apply_replace_plan(), so its
     two neighbours become run ends; and "in", that Outpost replaced back
     into a Stone Wall the same way. In "out" and "in" every wall in the
     site's 8-neighbourhood, the new one included, stores what the file's
     own neighbours derive.

Writes into build/wall_run_pairs/, which is gitignored.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import gate_orientation, shape_tools, unit_shape, unit_sprites, wall_run
from descape.edit_history import EditHistory
from descape.place_rotation import PlaceRotation
from descape.render_units import occupied_tiles_for, snapped_anchor, span_anchor, unit_occupied_tiles, unit_tile_bounds
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel, reanchored, replaced_position

OUT = ROOT / "build" / "wall_run_pairs"
STONE_WALL = 117
PLAYER = 1
GAIA = 0

# (label, source file, path kind, owner)
CASES = [
    ("integer", "2_Joan_coop_1_v0_13.aoe2scenario", "L", PLAYER),
    ("radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "L", PLAYER),
    ("shallow", "2_Joan_coop_1_v0_13.aoe2scenario", "shallow", PLAYER),
    # A fourth, beyond the plan's three: the diagonal indices 3 and 4 are the
    # highest-risk thing here (nothing in this tree has ever written them),
    # and the plan's shallow case only exercises them in an integer file.
    ("shallow_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "shallow_nwse", PLAYER),
    # GH #98: a GAIA-owned L junctioning into the (player-owned) anchor wall.
    ("gaia_integer", "2_Joan_coop_1_v0_13.aoe2scenario", "L", GAIA),
    ("gaia_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "L", GAIA),
    # The 2026-09-21 wall enclosure plan: the Wall Rectangle tool's ring.
    ("ring_integer", "2_Joan_coop_1_v0_13.aoe2scenario", "ring", PLAYER),
    ("ring_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "ring", PLAYER),
    # GH #125: the Wall Circle tool's ring.
    ("circle_integer", "2_Joan_coop_1_v0_13.aoe2scenario", "circle", PLAYER),
    ("circle_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "circle", PLAYER),
]

STONE_GATE = 64  # spans 4 tiles along x; a click on tile t covers t-1..t+2
# (label, source file): GH #159's gates over walls.
GATE_CASES = [
    ("gate_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("gate_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
# (label, source file): GH #194's gate moved GATE_MOVE_TILES along its run.
GATE_MOVE_CASES = [
    ("gate_move_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("gate_move_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
GATE_MOVE_TILES = 5
STONE_GATE_Y = 88  # Stone Gate along y, two orientation steps from STONE_GATE
# (label, source file): GH #339's gate cycled from x to y across a y run.
GATE_CYCLE_CASES = [
    ("gate_cycle_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("gate_cycle_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
# (label, source file): GH #339's mid-run wall replaced by a Stone Gate.
GATE_REPLACE_CASES = [
    ("gate_replace_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("gate_replace_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
# (label, source file): GH #340's mid-run wall deleted.
DELETE_CASES = [
    ("delete_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("delete_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
# (label, source file): GH #266's wall moved MOVE_ROWS off its run.
MOVE_CASES = [
    ("move_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("move_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
MOVE_ROWS = 3
# (label, source file): GH #365's Unit Line / Unit Shape scene.
SHAPE_CASES = [
    ("task608_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("task608_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]
HOUSE = 70
# (name, kind, box width, box height, Shift): the shapes drawn on clear ground, each box's anchor at its low corner.
SHAPE_BOXES = [
    ("right triangle", shape_tools.RIGHT_TRIANGLE, 9, 9, True),
    ("rectangle", unit_shape.RECTANGLE, 9, 5, False),
    ("ellipse", unit_shape.ELLIPSE, 13, 9, False),
]
HOUSE_BOX = (8, 8)
# (label, source file): TASK-581's stub-on-a-diagonal-run junctions.
JUNCTION_CASES = [("junction", "2_Joan_coop_1_v0_13.aoe2scenario")]
OUTPOST = 598  # 1x1, player-ownable, not a connector
# (label, source file): TASK-584's Replace out of and back into a wall.
REPLACE_CASES = [
    ("replace_integer", "2_Joan_coop_1_v0_13.aoe2scenario"),
    ("replace_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario"),
]


def _busiest_wall(scenario, w: int, h: int):
    """An existing wall with an orthogonal neighbour, so the L's far end has
    a real run to T-junction into rather than a lone piece. Returns it and
    whether that run lies along x."""
    tiles, walls = wall_run.wall_scene(scenario, w, h)
    for wall in walls:
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, tiles)
        if mask in (unit_sprites.WEST | unit_sprites.EAST, unit_sprites.NORTH | unit_sprites.SOUTH):
            return wall, mask == unit_sprites.WEST | unit_sprites.EAST
    raise SystemExit("no straight-run wall found to junction into")


def _path(kind: str, wx: int, wy: int, w: int, h: int, along_x: bool):
    if kind == "ring":
        # 9 long along the anchor run's own axis, 5 across it, with a short
        # side through the anchor: that side crosses the run at one tile.
        # Extends away from the nearer map edge so the ring isn't clipped.
        if along_x:
            sx = -1 if wx > w // 2 else 1
            return shape_tools.rect_perimeter_tiles(wx, wy - 2, wx + sx * 8, wy + 2, w, h)
        sy = -1 if wy > h // 2 else 1
        return shape_tools.rect_perimeter_tiles(wx - 2, wy, wx + 2, wy + sy * 8, w, h)
    if kind == "circle":
        # GH #125's Wall Circle: a 15x11 ellipse (not a circle, so a
        # transposed write would show; an 11x7 one has no diagonal pieces),
        # placed as "ring" is: its side apex on the anchor, the long axis
        # along the anchor run, away from the nearer map edge.
        if along_x:
            sx = -1 if wx > w // 2 else 1
            return shape_tools.ellipse_perimeter_tiles(wx, wy - 5, wx + sx * 14, wy + 5, w, h)
        sy = -1 if wy > h // 2 else 1
        return shape_tools.ellipse_perimeter_tiles(wx - 5, wy, wx + 5, wy + sy * 14, w, h)
    if kind.startswith("shallow"):
        # Away from whichever edge the anchor wall happens to sit near -- a
        # clipped path would hide the very bend this case exists to show.
        # The two senses give the two diagonal indices: a drag whose x and y
        # move opposite ways is NE-SW (4), the same-way one is NW-SE (3).
        sy = -1 if wy > h // 2 else 1
        y0 = wy + sy * 9
        y1 = wy + sy * 5 if kind == "shallow" else wy + sy * 13
        return shape_tools.wall_path_tiles(wx - 14, y0, wx - 2, y1, w, h)
    # An L: in along y from above, then across x to the existing wall.
    leg_a = shape_tools.wall_path_tiles(wx - 8, wy - 8, wx - 8, wy, w, h)
    leg_b = shape_tools.wall_path_tiles(wx - 7, wy, wx - 1, wy, w, h)
    return leg_a + leg_b


def _gate_anchor(tx: int, ty: int, w: int, h: int) -> tuple[float, float]:
    """Place Unit's snapped anchor for a gate clicked on (tx, ty) with no cursor point (TASK-461)."""
    return snapped_anchor((tx, ty), None, tile_span(STONE_GATE, (1, 1)), w, h)


def _gate_plan_at(scenario, tx: int, ty: int, w: int, h: int):
    """viewer._gate_plan() for a Place Unit click on (tx, ty)."""
    tiles, walls = wall_run.wall_scene(scenario, w, h)
    footprint = occupied_tiles_for(STONE_GATE, *_gate_anchor(tx, ty, w, h), w, h)
    if footprint is None:
        return None, wall_run.GatePlan(removals=[], rewrites=[])
    return footprint, wall_run.plan_gate_over_walls(footprint, existing_tiles=tiles, existing_walls=walls)


def _gate_sites(scenario, w: int, h: int) -> list[tuple[int, int]]:
    """Two click tiles: the middle of a straight x run (walls from t-2 to t+3,
    so all four footprint tiles hold a wall and the run continues past both
    gate ends), fewest rewrites first; then a footprint over a bend whose plan
    removes walls and reshapes exactly one neighbour, most removals first and
    well clear of the first gate.

    The reshape must be the gate's doing: that wall's stored shape agreed with
    its neighbours before the gate. A corpus wall that already disagreed
    (about 1% do) would be rewritten by any edit beside it and prove nothing."""
    tiles, walls = wall_run.wall_scene(scenario, w, h)
    wall_tiles = {(wall.tx, wall.ty) for wall in walls}
    straight = [
        (len(_gate_plan_at(scenario, wall.tx, wall.ty, w, h)[1].rewrites), wall.tx, wall.ty)
        for wall in walls
        if all((wall.tx + dx, wall.ty) in wall_tiles for dx in range(-2, 4))
    ]
    if not straight:
        raise SystemExit("no straight x run of 6 walls to put a gate on")
    _n, sx, sy = min(straight)

    def derived(tx: int, ty: int):
        return unit_sprites.wall_variant_from_neighbours8(
            unit_sprites.neighbour_mask(tx, ty, tiles, diagonals=True)
        )

    bends = []
    for wall in walls:
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, tiles, diagonals=True)
        if bin(mask).count("1") != 2 or derived(wall.tx, wall.ty) != 2:
            continue
        # Footprint tx-1..tx+2 holds the bend for tx in bend-2..bend+1.
        for tx in range(wall.tx - 2, wall.tx + 2):
            if max(abs(tx - sx), abs(wall.ty - sy)) <= 8:
                continue
            _footprint, plan = _gate_plan_at(scenario, tx, wall.ty, w, h)
            if not plan.removals or len(plan.rewrites) != 1:
                continue
            _p, unit, _v = plan.rewrites[0]
            if derived(int(unit.x), int(unit.y)) == unit_sprites.variant_index(unit.rotation, 5):
                bends.append((-len(plan.removals), tx, wall.ty))
    if not bends:
        raise SystemExit("no bend where a gate reshapes exactly one neighbour")
    _n, bx, by = min(bends)
    return [(sx, sy), (bx, by)]


def _gate_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    sites = _gate_sites(loaded, w, h)

    before = OUT / f"{label}_before.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    write_scenario(loaded, before)

    model = UnitEditModel(loaded)
    history = EditHistory()
    lines = []
    footprints = []
    # One record per gate, as two Place Unit clicks would make.
    for tx, ty in sites:
        footprint, plan = _gate_plan_at(loaded, tx, ty, w, h)
        footprints.append(set(footprint))
        reshaped = ", ".join(
            f"({int(u.x)}, {int(u.y)}) {unit_sprites.variant_index(u.rotation, 5)} -> {v}"
            for _p, u, v in plan.rewrites
        ) or "none"
        model.begin_unit_edit(wall_run.touched_players(PLAYER, plan))
        wall_run.apply_gate_plan(model, PLAYER, STONE_GATE, *_gate_anchor(tx, ty, w, h), plan)
        model.commit_unit_edit("Place unit", history)
        lines.append(
            f"         gate clicked on tile ({tx}, {ty}), footprint x {min(footprint)[0]}..{max(footprint)[0]} "
            f"y {ty}: {len(plan.removals)} walls removed, reshaped: {reshaped}"
        )
    write_scenario(loaded, after, units=model)

    # The M check: read the written file back, no wall anchored on a new gate's tile.
    reread = load_map_and_units(after)
    gates = [
        set(unit_occupied_tiles(u, w, h) or ())
        for units in reread.unit_manager.units
        for u in units
        if u.unit_const == STONE_GATE and set(unit_occupied_tiles(u, w, h) or ()) in footprints
    ]
    assert len(gates) == len(sites), (len(gates), len(sites))
    _t, walls_back = wall_run.wall_scene(reread, w, h)
    under = [(wall.tx, wall.ty) for wall in walls_back if any((wall.tx, wall.ty) in g for g in gates)]
    assert not under, under
    print(f"{label:8s} {name} (owner {PLAYER})\n" + "\n".join(lines) + f"\n         {before.name} / {after.name}")


def _gate_move_site(scenario, w: int, h: int) -> tuple[int, int, int]:
    """(low, ty, owner): the low tile of a gate footprint inside a straight x
    run, and the player whose run it is. One wall of that one player on every
    tile from low-1 to low+9, so the gate (low..low+3) has a wall past its low
    end and, moved GATE_MOVE_TILES up, lands on four walls with one more past
    them. Nothing else stands on the four tiles it leaves, and the site with
    the fewest walls beside the run wins: those would reshape as it moves."""
    _tiles, walls = wall_run.wall_scene(scenario, w, h)
    owners: dict[tuple[int, int], list[int]] = {}
    for wall in walls:
        owners.setdefault((wall.tx, wall.ty), []).append(wall.player_id)
    blocked = wall_run.blocked_tiles(scenario, w, h)
    sites = []
    for (low, ty), (owner, *stacked) in sorted(owners.items()):
        # A GAIA gate would be an odd thing to look for in the game editor.
        if stacked or owner == GAIA:
            continue
        if any(owners.get((low + dx, ty)) != [owner] for dx in range(-1, GATE_MOVE_TILES + 5)):
            continue
        if any((low + dx, ty) in blocked for dx in range(4)):
            continue
        beside = sum((low + dx, ty + dy) in owners for dx in range(-2, GATE_MOVE_TILES + 6) for dy in (-1, 1))
        sites.append((beside, low, ty, owner))
    if not sites:
        raise SystemExit(f"no straight x run of {GATE_MOVE_TILES + 6} walls to move a gate along")
    _n, low, ty, owner = min(sites)
    return low, ty, owner


def _gate_move_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    low, ty, owner = _gate_move_site(loaded, w, h)
    x, y = span_anchor(low, ty, 4, 1)
    old = set(occupied_tiles_for(STONE_GATE, x, y, w, h))
    new = set(occupied_tiles_for(STONE_GATE, x + GATE_MOVE_TILES, y, w, h))

    before = OUT / f"{label}_before.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    model = UnitEditModel(loaded)
    history = EditHistory()
    # The "before": the gate placed over the run, as a Place Unit click does (GH #159).
    tiles, walls = wall_run.wall_scene(loaded, w, h)
    placed = wall_run.plan_gate_over_walls(old, existing_tiles=tiles, existing_walls=walls)
    model.begin_unit_edit(wall_run.touched_players(owner, placed))
    gate = wall_run.apply_gate_plan(model, owner, STONE_GATE, x, y, placed)
    model.commit_unit_edit("Place unit", history)
    write_scenario(loaded, before, units=model)

    # The "after": viewer.on_unit_move()'s plan-bearing branch, in its order.
    plan = wall_run.scenario_gate_moves(
        loaded, w, h, [(gate, x + GATE_MOVE_TILES, y)], removable=lambda unit: not model.referencing(unit)
    )
    assert plan, "the move planned nothing"
    reshaped = ", ".join(
        f"({int(u.x)}, {int(u.y)}) {unit_sprites.variant_index(u.rotation, 5)} -> {v}" for _p, u, v in plan.rewrites
    ) or "none"
    model.begin_unit_edit(sorted({owner} | wall_run.gate_move_players(plan)))
    model.remove_many([unit for _p, unit in plan.removals])
    model.set_position(gate, x + GATE_MOVE_TILES, y, gate.z)
    wall_run.apply_gate_move_plan(model, plan)
    model.commit_unit_edit("Move unit", history)
    write_scenario(loaded, after, units=model)

    # The M check, on the two files as written: 4 walls gone from under the
    # gate, 4 new ones on the tiles it left, and nothing else moved.
    def reread(path, footprint):
        scenario = load_map_and_units(path)
        gates = [
            u for units in scenario.unit_manager.units for u in units
            if u.unit_const == STONE_GATE and set(unit_occupied_tiles(u, w, h) or ()) == footprint
        ]
        assert len(gates) == 1, (path.name, len(gates))
        _t, walls_back = wall_run.wall_scene(scenario, w, h)
        return walls_back, sorted((wall.tx, wall.ty) for wall in walls_back)

    _walls, tiles_before = reread(before, old)
    walls_after, tiles_after = reread(after, new)
    assert (len(plan.removals), len(plan.fills), plan.blocked) == (4, 4, 0), (len(plan.removals), len(plan.fills), plan.blocked)
    assert set(tiles_before) - set(tiles_after) == new, "the walls under the moved gate were not all removed"
    assert set(tiles_after) - set(tiles_before) == old, "the tiles the gate left were not all filled"
    assert len(tiles_before) == len(tiles_after) and all(tiles_after.count(t) == 1 for t in old)
    filled = sorted((wall.tx, wall.player_id, wall.unit) for wall in walls_after if (wall.tx, wall.ty) in old)
    print(
        f"{label:8s} {name} (owner {owner})\n"
        f"         before: Stone Gate on tiles x {min(old)[0]}..{max(old)[0]} y {ty}, anchor ({x:g}, {y:g})\n"
        f"         after:  moved +{GATE_MOVE_TILES} to tiles x {min(new)[0]}..{max(new)[0]} y {ty}, "
        f"anchor ({x + GATE_MOVE_TILES:g}, {y:g})\n"
        f"         {len(plan.removals)} walls removed under it, {len(plan.fills)} filled on x {min(old)[0]}..{max(old)[0]} "
        f"(const {sorted({u.unit_const for _x, _p, u in filled})}, owner {sorted({p for _x, p, _u in filled})}, "
        f"indices read back {[unit_sprites.variant_index(u.rotation, 5) for _x, _p, u in filled]}), reshaped: {reshaped}\n"
        f"         {before.name} / {after.name}"
    )


def _owned_walls(scenario, w: int, h: int) -> dict[tuple[int, int], int]:
    """{tile: owner} for every tile holding exactly one wall, of a non-GAIA player."""
    _tiles, walls = wall_run.wall_scene(scenario, w, h)
    owners: dict[tuple[int, int], list[int]] = {}
    for wall in walls:
        owners.setdefault((wall.tx, wall.ty), []).append(wall.player_id)
    return {tile: found[0] for tile, found in owners.items() if len(found) == 1 and found[0] != GAIA}


def _gate_cycle_site(scenario, w: int, h: int) -> tuple[int, int, int]:
    """(tx, ty, owner): a straight y run of one player's walls on (tx, ty)..
    (tx, ty+3), where an x Stone Gate with its low corner on (tx, ty) sticks
    out over three tiles holding no connector and no blocker. Cycled two
    steps it turns along the run, onto (tx, ty+1)..(tx, ty+3). The site with
    the fewest other walls around both footprints wins."""
    owned = _owned_walls(scenario, w, h)
    connectors, _walls = wall_run.wall_scene(scenario, w, h)
    blocked = wall_run.blocked_tiles(scenario, w, h)
    sites = []
    for (tx, ty), owner in sorted(owned.items()):
        if tx + 4 > w or ty + 4 > h:
            continue
        if any(owned.get((tx, ty + dy)) != owner for dy in range(4)):
            continue
        if any((tx + dx, ty) in connectors or (tx + dx, ty) in blocked for dx in range(1, 4)):
            continue
        around = sum(
            (tx + dx, ty + dy) in connectors for dx in range(-1, 5) for dy in range(-1, 5) if dx != 0
        )
        sites.append((around, tx, ty, owner))
    if not sites:
        raise SystemExit("no straight y run of 4 walls with 3 free tiles beside its top")
    _n, tx, ty, owner = min(sites)
    return tx, ty, owner


def _gate_cycle_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    tx, ty, owner = _gate_cycle_site(loaded, w, h)
    x, y = span_anchor(tx, ty, 4, 1)
    old = set(occupied_tiles_for(STONE_GATE, x, y, w, h))
    turned = gate_orientation.cycle_const(STONE_GATE, 2)
    assert turned == STONE_GATE_Y, turned

    before = OUT / f"{label}_before.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    model = UnitEditModel(loaded)
    history = EditHistory()
    # The "before": the x gate placed across the run, as a Place Unit click does (GH #159).
    tiles, walls = wall_run.wall_scene(loaded, w, h)
    placed = wall_run.plan_gate_over_walls(old, existing_tiles=tiles, existing_walls=walls)
    assert len(placed.removals) == 1, len(placed.removals)
    model.begin_unit_edit(wall_run.touched_players(owner, placed))
    gate = wall_run.apply_gate_plan(model, owner, STONE_GATE, x, y, placed)
    model.commit_unit_edit("Place unit", history)
    write_scenario(loaded, before, units=model)

    # The "after": the Rotation field's jump to orientation 2, viewer._set_gate_field()'s
    # plan-bearing branch in its order (GH #339).
    nx, ny = reanchored(gate, turned)
    new = set(occupied_tiles_for(turned, nx, ny, w, h))
    plan = wall_run.scenario_gate_changes(
        loaded, w, h, [(gate, turned, nx, ny)], removable=lambda unit: not model.referencing(unit), fill=False
    )
    assert plan, "the cycle planned nothing"
    reshaped = ", ".join(
        f"({int(u.x)}, {int(u.y)}) {unit_sprites.variant_index(u.rotation, 5)} -> {v}" for _p, u, v in plan.rewrites
    ) or "none"
    model.begin_unit_edit(sorted({owner} | wall_run.gate_move_players(plan)))
    model.remove_many([unit for _p, unit in plan.removals])
    model.set_unit_const(gate, turned)
    wall_run.apply_gate_move_plan(model, plan)
    model.commit_unit_edit("Set unit Rotation", history)
    write_scenario(loaded, after, units=model)

    # The M check, on the two files as written: the gate turned onto the run, the three
    # walls under its new tiles gone, no wall on any gate tile, nothing filled.
    walls_before = _reread_gate(before, STONE_GATE, old, w, h)
    walls_after = _reread_gate(after, turned, new, w, h)
    assert new == {(tx, ty + dy) for dy in range(4)}, sorted(new)
    assert (len(plan.removals), plan.fills, plan.blocked) == (3, [], 0), (len(plan.removals), plan.fills, plan.blocked)
    assert set(walls_before) - set(walls_after) == new - {(tx, ty)}, "the walls under the turned gate were not all removed"
    assert not set(walls_after) & new and not set(walls_after) - set(walls_before)
    print(
        f"{label:8s} {name} (owner {owner})\n"
        f"         before: Stone Gate (x) on tiles x {tx}..{tx + 3} y {ty}, anchor ({x:g}, {y:g}), across a y run\n"
        f"         after:  turned to {turned} (y) on tiles x {tx} y {ty}..{ty + 3}, anchor ({nx:g}, {ny:g})\n"
        f"         {len(plan.removals)} walls removed under it, reshaped: {reshaped}\n"
        f"         {before.name} / {after.name}"
    )


def _reread_gate(path, const: int, footprint: set, w: int, h: int) -> list[tuple[int, int]]:
    """Reads `path` back: exactly one `const` gate on `footprint`, and every wall's tile."""
    scenario = load_map_and_units(path)
    gates = [
        u for units in scenario.unit_manager.units for u in units
        if u.unit_const == const and set(unit_occupied_tiles(u, w, h) or ()) == footprint
    ]
    assert len(gates) == 1, (path.name, len(gates))
    _t, walls_back = wall_run.wall_scene(scenario, w, h)
    return sorted((wall.tx, wall.ty) for wall in walls_back)


def _gate_replace_site(scenario, w: int, h: int) -> tuple[int, int, int]:
    """(tx, ty, owner): a straight x run of one player's walls on tx-1..tx+4,
    so a Stone Gate replacing the wall on tx covers tx..tx+3 with a run end
    left on either side. Fewest walls beside the run first."""
    owned = _owned_walls(scenario, w, h)
    sites = []
    for (tx, ty), owner in sorted(owned.items()):
        if any(owned.get((tx + dx, ty)) != owner for dx in range(-1, 5)):
            continue
        beside = sum((tx + dx, ty + dy) in owned for dx in range(-2, 6) for dy in (-1, 1))
        sites.append((beside, tx, ty, owner))
    if not sites:
        raise SystemExit("no straight x run of 6 walls to replace a wall in")
    _n, tx, ty, owner = min(sites)
    return tx, ty, owner


def _gate_replace_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    tx, ty, owner = _gate_replace_site(loaded, w, h)
    wall = next(
        u for u in loaded.unit_manager.units[owner]
        if unit_sprites.rotation_variant_eligible(u.unit_const) and (int(u.x), int(u.y)) == (tx, ty)
    )
    ref = wall.reference_id
    before = OUT / f"{label}_before.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    write_scenario(loaded, before)

    # The "after": viewer._replace_units()'s plan-bearing branch, in its order (GH #339).
    model = UnitEditModel(loaded)
    nx, ny = replaced_position(wall, STONE_GATE)
    new = set(occupied_tiles_for(STONE_GATE, nx, ny, w, h))
    plan = wall_run.scenario_gate_changes(
        loaded, w, h, [(wall, STONE_GATE, nx, ny)], removable=lambda unit: not model.referencing(unit), fill=False
    )
    assert plan, "the Replace planned nothing"
    reshaped = ", ".join(
        f"({int(u.x)}, {int(u.y)}) {unit_sprites.variant_index(u.rotation, 5)} -> {v}" for _p, u, v in plan.rewrites
    ) or "none"
    model.begin_unit_edit(sorted({owner} | wall_run.gate_move_players(plan)))
    model.remove_many([unit for _p, unit in plan.removals])
    model.replace_type(wall, STONE_GATE, map_w=w, map_h=h, planned={ref: STONE_GATE})
    wall_run.apply_gate_move_plan(model, plan)
    model.commit_unit_edit("Replace unit", EditHistory())
    write_scenario(loaded, after, units=model)

    # The M check, on the files as written: the replaced wall is the gate (same
    # reference_id), the three walls under it are gone, and the walls beside it
    # store the shape their neighbours give them.
    walls_after = _reread_gate(after, STONE_GATE, new, w, h)
    reread = load_map_and_units(after)
    gate = next(u for units in reread.unit_manager.units for u in units if u.reference_id == ref)
    assert gate.unit_const == STONE_GATE and new == {(tx + dx, ty) for dx in range(4)}, (gate.unit_const, sorted(new))
    assert len(plan.removals) == 3 and plan.fills == [], (len(plan.removals), plan.fills)
    assert not set(walls_after) & new, "a wall still stands under the new gate"
    connectors, walls_back = wall_run.wall_scene(reread, w, h)
    ends = {}
    for side in ((tx - 1, ty), (tx + 4, ty)):
        back = next(wb for wb in walls_back if (wb.tx, wb.ty) == side)
        stored = unit_sprites.variant_index(back.unit.rotation, 5)
        derived = unit_sprites.wall_variant_from_neighbours8(
            unit_sprites.neighbour_mask(*side, connectors, diagonals=True)
        )
        assert stored == derived, (side, stored, derived)
        ends[side] = stored
    print(
        f"{label:8s} {name} (owner {owner})\n"
        f"         wall ref {ref} on tile ({tx}, {ty}) in an x run replaced by Stone Gate {STONE_GATE}, "
        f"tiles x {tx}..{tx + 3} y {ty}, anchor ({nx:g}, {ny:g})\n"
        f"         {len(plan.removals)} walls removed under it, run ends beside it store {ends}, reshaped: {reshaped}\n"
        f"         {before.name} / {after.name}"
    )


def _delete_site(scenario, w: int, h: int):
    """(wall, west, east, rewrites): a non-GAIA wall in the interior of a
    straight x run (walls on t-2..t+2) whose two run neighbours decode to 0,
    a run piece, and which deleting turns into run ends (2), as
    scenario_removal_rewrites() plans it. The first such site, scanning by
    row then column."""
    _tiles, walls = wall_run.wall_scene(scenario, w, h)
    owned = _owned_walls(scenario, w, h)
    at = {(wall.tx, wall.ty): wall for wall in walls if (wall.tx, wall.ty) in owned}
    for (tx, ty), wall in sorted(at.items(), key=lambda item: (item[0][1], item[0][0])):
        if not all((tx + dx, ty) in at for dx in range(-2, 3)):
            continue
        west, east = at[(tx - 1, ty)], at[(tx + 1, ty)]
        if any(unit_sprites.variant_index(r.unit.rotation, 5) != 0 for r in (west, east)):
            continue
        rewrites = wall_run.scenario_removal_rewrites(scenario, w, h, [wall.unit])
        shapes = {id(unit): v for _p, unit, v in rewrites}
        if shapes.get(id(west.unit)) == 2 and shapes.get(id(east.unit)) == 2:
            return wall, west, east, rewrites
    raise SystemExit("no straight x run whose middle wall's delete turns both neighbours into run ends")


def _delete_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    wall, west, east, rewrites = _delete_site(loaded, w, h)
    refs = [r.unit.reference_id for r in (west, east)]
    gone = wall.unit.reference_id

    before = OUT / f"{label}_before.aoe2scenario"
    stale = OUT / f"{label}_stale.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    write_scenario(loaded, before)
    model = UnitEditModel(loaded)
    history = EditHistory()
    # The stale file is the removal alone, what Delete wrote until GH #340; the after file adds
    # the rewrites on top, the same end state as the viewer's single record.
    model.begin_unit_edit([wall.player_id])
    model.remove_many([wall.unit])
    model.commit_unit_edit("Delete unit", history)
    write_scenario(loaded, stale, units=model)
    model.begin_unit_edit(sorted({p for p, _u, _v in rewrites}))
    wall_run.apply_rewrites(model, rewrites)
    model.commit_unit_edit("Delete unit", history)
    write_scenario(loaded, after, units=model)

    # The M check, on the files as written: the two neighbours found by reference_id.
    def reread(path) -> list[int]:
        found = {u.reference_id: u for units in load_map_and_units(path).unit_manager.units for u in units}
        assert gone not in found, (path.name, gone)
        return [unit_sprites.variant_index(found[ref].rotation, 5) for ref in refs]

    stale_indices, after_indices = reread(stale), reread(after)
    assert stale_indices == [0, 0], (stale.name, stale_indices)
    assert after_indices == [2, 2], (after.name, after_indices)
    others = [(int(u.x), int(u.y)) for _p, u, _v in rewrites if u is not west.unit and u is not east.unit]
    print(
        f"{label:8s} {name} (owner {wall.player_id})\n"
        f"         deleted wall ref {gone} on tile ({wall.tx}, {wall.ty}) from the middle of an x run\n"
        f"         west ({west.tx}, {west.ty}) and east ({east.tx}, {east.ty}): stale file keeps {stale_indices}, "
        f"after file stores {after_indices}\n"
        f"         other walls reshaped: {others or 'none'}\n"
        f"         {before.name} / {stale.name} / {after.name}"
    )


def _move_site(scenario, w: int, h: int):
    """(wall, west, east, new_tile): a non-GAIA wall in the interior of a
    straight x run (walls on t-2..t+2), with its two run neighbours, where
    each of the three decodes to something other than 2 and stands alone on
    its tile, and no connector stands beside either neighbour off the run,
    so the move leaves both with one run neighbour. It goes MOVE_ROWS down
    (else up) onto a tile no object blocks with no connector in its
    8-neighbourhood, so it lands as a tower."""
    connectors = unit_sprites.wall_connector_consts()
    counts: dict[tuple[int, int], int] = {}
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const in connectors:
                for tile in unit_occupied_tiles(unit, w, h) or ():
                    counts[tile] = counts.get(tile, 0) + 1
    _tiles, walls = wall_run.wall_scene(scenario, w, h)
    at = {(wall.tx, wall.ty): wall for wall in walls}
    blocked = wall_run.blocked_tiles(scenario, w, h)

    def landing(tx: int, ty: int) -> bool:
        return (
            0 <= ty < h
            and (tx, ty) not in blocked
            and not any((tx + dx, ty + dy) in counts for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        )

    for (tx, ty), wall in sorted(at.items(), key=lambda item: (item[0][1], item[0][0])):
        if wall.player_id == GAIA:
            continue
        if not all((tx + dx, ty) in at for dx in range(-2, 3)):
            continue
        row = [at[(tx + dx, ty)] for dx in (-1, 0, 1)]
        if any(counts[(r.tx, r.ty)] != 1 or unit_sprites.variant_index(r.unit.rotation, 5) == 2 for r in row):
            continue
        if any((tx + dx, ty + dy) in counts for dx in (-1, 1) for dy in (-1, 1)):
            continue
        for dy in (MOVE_ROWS, -MOVE_ROWS):
            if landing(tx, ty + dy):
                return row[1], row[0], row[2], (tx, ty + dy)
    raise SystemExit(f"no straight x run with a wall to move {MOVE_ROWS} rows into the open")


def _move_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    wall, west, east, (nx, ny) = _move_site(loaded, w, h)
    trio = (wall, west, east)
    old_indices = [unit_sprites.variant_index(r.unit.rotation, 5) for r in trio]
    x, y = wall.unit.x, wall.unit.y + (ny - wall.ty)
    plan = wall_run.scenario_move_plan(loaded, w, h, [(wall.unit, x, y)], gate_walls=False)
    assert plan, "the move planned nothing"

    before = OUT / f"{label}_before.aoe2scenario"
    stale = OUT / f"{label}_stale.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    write_scenario(loaded, before)
    model = UnitEditModel(loaded)
    history = EditHistory()
    # The stale file is the move alone; the after file adds the plan's shapes on top, the
    # same end state as the viewer's single record (set_position(), then apply_move_plan()).
    model.begin_unit_edit([wall.player_id])
    model.set_position(wall.unit, x, y, wall.unit.z)
    model.commit_unit_edit("Move unit", history)
    write_scenario(loaded, stale, units=model)
    model.begin_unit_edit(sorted({wall.player_id} | wall_run.move_plan_players(plan)))
    wall_run.apply_move_plan(model, plan)
    model.commit_unit_edit("Move unit", history)
    write_scenario(loaded, after, units=model)

    # The M check, on the files as written: the three walls found by reference_id.
    refs = [r.unit.reference_id for r in trio]

    def reread(path) -> list[int]:
        found = {
            u.reference_id: u
            for units in load_map_and_units(path).unit_manager.units
            for u in units
            if u.reference_id in refs
        }
        assert (int(found[refs[0]].x), int(found[refs[0]].y)) == (nx, ny), (path.name, found[refs[0]].x, found[refs[0]].y)
        return [unit_sprites.variant_index(found[ref].rotation, 5) for ref in refs]

    stale_indices, after_indices = reread(stale), reread(after)
    assert stale_indices == old_indices, (stale.name, stale_indices, old_indices)
    assert after_indices == [2, 2, 2], (after.name, after_indices)
    others = [(int(u.x), int(u.y)) for _p, u, _v in plan.rewrites if u is not west.unit and u is not east.unit]
    print(
        f"{label:8s} {name} (owner {wall.player_id})\n"
        f"         moved wall: tile ({wall.tx}, {wall.ty}) -> ({nx}, {ny}), index {old_indices[0]} -> {after_indices[0]}\n"
        f"         west ({west.tx}, {west.ty}) {old_indices[1]} -> {after_indices[1]}, "
        f"east ({east.tx}, {east.ty}) {old_indices[2]} -> {after_indices[2]}; stale file keeps {stale_indices}\n"
        f"         other walls reshaped: {others or 'none'}\n"
        f"         {before.name} / {stale.name} / {after.name}"
    )


def _clear_boxes(occupied: set, taken: list, bw: int, bh: int, w: int, h: int):
    """Low corners of bw x bh boxes free of every object, with a 1-tile
    margin and then with none, and two tiles clear of the boxes already
    `taken`, nearest the map centre first so the shape is easy to find in
    the editor. The corpus maps are a third full, so the boxes are small."""
    cx, cy = w // 2, h // 2
    for margin in (1, 0):
        origins = sorted(
            ((x, y) for y in range(margin, h - bh - margin) for x in range(margin, w - bw - margin)),
            key=lambda o: (abs(o[0] - cx) + abs(o[1] - cy), o),
        )
        for x0, y0 in origins:
            # Two tiles between shapes whatever the margin, so no two join.
            if any(x0 - 2 < tx1 and tx0 < x0 + bw + 2 and y0 - 2 < ty1 and ty0 < y0 + bh + 2 for tx0, ty0, tx1, ty1 in taken):
                continue
            lo_x, lo_y, hi_x, hi_y = x0 - margin, y0 - margin, x0 + bw + margin, y0 + bh + margin
            if any((x, y) in occupied for y in range(lo_y, hi_y) for x in range(lo_x, hi_x)):
                continue
            yield x0, y0
    raise SystemExit(f"no clear {bw}x{bh} box")


def _run_end_site(scenario, occupied: set, w: int, h: int):
    """A straight run's end wall with ground past it for a 9x5 Shift Triangle
    whose base continues the run: (end wall's tile, the drag's anchor, its
    end, objects under the box). No other wall may touch the box; among the
    rest, the fewest objects under it, then nearest the map centre. The
    corpus has no run end with wholly clear ground past it."""
    tiles, walls = wall_run.wall_scene(scenario, w, h)
    sites = []
    for wall in walls:
        ex, ey = wall.tx, wall.ty
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (ex - dx, ey - dy) not in tiles or (ex - 2 * dx, ey - 2 * dy) not in tiles:
                continue
            ax, ay = ex + dx, ey + dy
            # The base along the run, the apex to the side nearer the map centre.
            side = (0, 1 if ay < h // 2 else -1) if dx else (1 if ax < w // 2 else -1, 0)
            bx, by = ax + dx * 8 + side[0] * 4, ay + dy * 8 + side[1] * 4
            lo_x, hi_x = min(ax, bx) - 1, max(ax, bx) + 1
            lo_y, hi_y = min(ay, by) - 1, max(ay, by) + 1
            if lo_x < 0 or lo_y < 0 or hi_x >= w or hi_y >= h:
                continue
            box = {(x, y) for y in range(lo_y, hi_y + 1) for x in range(lo_x, hi_x + 1)} - {(ex, ey)}
            if box & tiles:
                continue
            covered = len(box & occupied)
            sites.append((covered, abs(ax - w // 2) + abs(ay - h // 2), (ex, ey), (ax, ay), (bx, by)))
    if not sites:
        raise SystemExit("no run end with wall-free ground past it")
    covered, _d, wall_tile, anchor, end = min(sites)
    return wall_tile, anchor, end, covered


def _shape_wall(model, loaded, kind: str, anchor, end, shift: bool, w: int, h: int, history) -> str:
    """One Unit Line / Unit Shape wall drag, as the viewer's _commit_wall_run() writes it."""
    latch = unit_shape.latch_for(
        (STONE_WALL,), PLAYER, kind, filled=False, spacing=0, randomize=False, rules=True, pending=PlaceRotation()
    )
    cells = unit_shape.lattice_cells(latch, anchor, end, shift, w, h)
    existing_tiles, existing_walls = wall_run.wall_scene(loaded, w, h)
    plan = wall_run.plan_wall_run(
        cells, unit_const=STONE_WALL, existing_tiles=existing_tiles, existing_walls=existing_walls
    )
    model.begin_unit_edit(wall_run.touched_players(PLAYER, plan))
    wall_run.apply_wall_plan(model, PLAYER, plan)
    model.commit_unit_edit("Draw unit shape", history)
    xs = [x for x, _y in cells]
    ys = [y for _x, y in cells]
    corners = sorted((int(n.x), int(n.y)) for n in plan.nodes if n.variant == 2)
    return (
        f"drag ({anchor[0]}, {anchor[1]}) -> ({end[0]}, {end[1]}){' with Shift' if shift else ''}, "
        f"tiles x {min(xs)}..{max(xs)} y {min(ys)}..{max(ys)}: {len(plan.nodes)} placed, "
        f"{len(plan.rewrites)} rewritten, {plan.skipped} skipped, indices {sorted({n.variant for n in plan.nodes})}, "
        f"towers (2) at {corners}"
    )


def _shape_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    loaded = load_map_and_units(src)
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    tiles, _walls = wall_run.wall_scene(loaded, w, h)
    occupied = wall_run.blocked_tiles(loaded, w, h) | tiles
    (wx, wy), anchor, end, covered = _run_end_site(loaded, occupied, w, h)

    before = OUT / f"{label}_before.aoe2scenario"
    after = OUT / f"{label}_after.aoe2scenario"
    write_scenario(loaded, before)

    model = UnitEditModel(loaded)
    history = EditHistory()
    # The joined triangle: anchored just past a run's end wall, its base continuing the run.
    drawn = _shape_wall(model, loaded, shape_tools.TRIANGLE, anchor, end, True, w, h, history)
    lines = [f"joined triangle ({covered} object tiles under its box): {drawn}"]
    taken = [(min(anchor[0], end[0]), min(anchor[1], end[1]), max(anchor[0], end[0]) + 1, max(anchor[1], end[1]) + 1)]
    for shape_name, kind, bw, bh, shift in SHAPE_BOXES:
        x0, y0 = next(_clear_boxes(occupied, taken, bw, bh, w, h))
        taken.append((x0, y0, x0 + bw, y0 + bh))
        drawn = _shape_wall(model, loaded, kind, (x0, y0), (x0 + bw - 1, y0 + bh - 1), shift, w, h, history)
        lines.append(f"{shape_name}: {drawn}")

    # The Houses: the first clear box the game's placement rules accept whole (terrain can refuse a house).
    bw, bh = HOUSE_BOX
    latch = unit_shape.latch_for(
        (HOUSE,), PLAYER, unit_shape.RECTANGLE, filled=False, spacing=1, randomize=False, rules=True,
        pending=PlaceRotation(),
    )

    def add(spec):
        return model.add(spec.player, spec.unit_const, spec.x, spec.y, rotation=spec.rotation)

    for x0, y0 in _clear_boxes(occupied, taken, bw, bh, w, h):
        cells = unit_shape.lattice_cells(latch, (x0, y0), (x0 + bw - 1, y0 + bh - 1), False, w, h)
        model.begin_unit_edit([PLAYER])
        with model.batch_adds():
            stroke = unit_shape.place_cells(loaded, latch, cells, add, random.Random(608))
        if stroke.refused:
            # abort_unit_edit() does not roll the document back, so this box's houses come out first.
            for _owner, unit in stroke.placed:
                model.remove(unit)
            model.abort_unit_edit()
            continue
        model.commit_unit_edit("Draw unit shape", history)
        break
    houses = sorted((u.x, u.y) for _owner, u in stroke.placed)
    lines.append(
        f"houses: drag ({x0}, {y0}) -> ({x0 + bw - 1}, {y0 + bh - 1}) at Spacing 1: {len(houses)} placed at {houses}"
    )
    write_scenario(loaded, after, units=model)

    # Read back: every placed house is in the written file.
    reread = load_map_and_units(after)
    _t, walls_back = wall_run.wall_scene(reread, w, h)
    placed = set(houses)
    houses_back = sorted(
        (u.x, u.y) for units in reread.unit_manager.units for u in units if u.unit_const == HOUSE and (u.x, u.y) in placed
    )
    assert houses_back == houses, (len(houses_back), len(houses))
    # Every house the after file gained is a reported one: none left over from a refused box.
    gained = sum(u.unit_const == HOUSE for units in reread.unit_manager.units for u in units) - sum(
        u.unit_const == HOUSE for units in load_map_and_units(before).unit_manager.units for u in units
    )
    assert gained == len(houses), (gained, len(houses))
    print(f"{label} {name} (owner {PLAYER}), the joined run's end wall at tile ({wx}, {wy})")
    for line in lines:
        print(f"         {line}")
    print(f"         {len(walls_back)} walls read back; {before.name} / {after.name}")


def _pre_task581_table(mask: int) -> int | None:
    """wall_variant_from_neighbours8() as it was before TASK-581: any orthogonal bit defers to the 4-bit table."""
    if mask & unit_sprites.ORTHOGONAL_MASK:
        return unit_sprites.wall_variant_from_neighbours(mask & unit_sprites.ORTHOGONAL_MASK)
    return _SHIPPED_TABLE(mask)


_SHIPPED_TABLE = unit_sprites.wall_variant_from_neighbours8


def _junction_sites(loaded, w: int, h: int):
    """[(site, run tiles, stub tile groups, {junction tile: label})] on three clear 5x5 boxes."""
    tiles, _walls = wall_run.wall_scene(loaded, w, h)
    occupied = wall_run.blocked_tiles(loaded, w, h) | tiles
    taken: list = []
    sites = []
    for site in "ABC":
        x0, y0 = next(_clear_boxes(occupied, taken, 5, 5, w, h))
        taken.append((x0, y0, x0 + 5, y0 + 5))
        mid = (x0 + 2, y0 + 2)
        if site == "A":
            run = [(x0 + i, y0 + 4 - i) for i in range(5)]
            stubs = [[(x0 + 2, y0 + 1)]]
            junctions = {mid: "middle", (x0 + 3, y0 + 1): "middle's NE"}
        elif site == "B":
            run = [(x0 + i, y0 + i) for i in range(5)]
            stubs = [[(x0 + 1, y0 + 2)]]
            junctions = {mid: "middle", (x0 + 1, y0 + 1): "middle's NW"}
        else:
            run = [(x0 + i, y0 + i) for i in range(5)]
            stubs = [[(x0 + 2, y0 + 1), (x0 + 3, y0 + 2)]]
            junctions = {mid: "middle", (x0 + 1, y0 + 1): "middle's NW", (x0 + 3, y0 + 3): "middle's SE"}
        for group in stubs:
            junctions.update(dict.fromkeys(group, "stub"))
        sites.append((site, run, stubs, junctions))
    return sites


def _place_walls(model, loaded, tile_groups, w: int, h: int, history) -> None:
    """One plan_wall_run() record per group, as one Wall tool drag each."""
    for group in tile_groups:
        existing_tiles, existing_walls = wall_run.wall_scene(loaded, w, h)
        plan = wall_run.plan_wall_run(
            group, unit_const=STONE_WALL, existing_tiles=existing_tiles, existing_walls=existing_walls
        )
        model.begin_unit_edit(wall_run.touched_players(PLAYER, plan))
        wall_run.apply_wall_plan(model, PLAYER, plan)
        model.commit_unit_edit("Place wall run", history)


def _junction_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    files = {kind: OUT / f"{label}_{kind}.aoe2scenario" for kind in ("before", "stale", "after")}
    sites = None
    for kind in ("stale", "after"):
        loaded = load_map_and_units(src)
        mm = loaded.map_manager
        w, h = mm.map_width, mm.map_height
        found = _junction_sites(loaded, w, h)
        assert sites is None or [s[1] for s in found] == [s[1] for s in sites], "sites moved between loads"
        sites = found
        model = UnitEditModel(loaded)
        history = EditHistory()
        _place_walls(model, loaded, [run for _s, run, _st, _j in sites], w, h, history)
        if kind == "stale":
            write_scenario(loaded, files["before"], units=model)
            unit_sprites.wall_variant_from_neighbours8 = _pre_task581_table
        try:
            _place_walls(model, loaded, [g for _s, _r, stubs, _j in sites for g in stubs], w, h, history)
        finally:
            unit_sprites.wall_variant_from_neighbours8 = _SHIPPED_TABLE
        write_scenario(loaded, files[kind], units=model)

    # Read every junction tile back off the written files.
    stored = {}
    for kind, path in files.items():
        walls = wall_run.wall_scene(load_map_and_units(path), w, h)[1]
        stored[kind] = {(wall.tx, wall.ty): unit_sprites.variant_index(wall.unit.rotation, 5) for wall in walls}
    print(f"{label} {name} (owner {PLAYER})")
    for site, run, _stubs, junctions in sites:
        print(f"         site {site}, run {run[0]} .. {run[-1]}:")
        for tile, what in junctions.items():
            row = ", ".join(f"{kind} {stored[kind].get(tile, '-')}" for kind in files)
            print(f"           {what:12s} {tile}: {row}")
    a, b, c = sites
    for site, keep in ((a, 4), (b, 3)):
        pieces = [tile for tile, what in site[3].items() if what != "stub"]
        assert [stored["stale"][t] for t in pieces] == [2, 2], (site[0], "stale")
        assert [stored["after"][t] for t in pieces] == [keep, keep], (site[0], "after")
    assert stored["after"][(c[1][2])] == 2, "site C middle"
    print("         " + " / ".join(path.name for path in files.values()))


def _replace_one(model, loaded, unit, owner: int, const: int, w: int, h: int, plan_walls: bool):
    """viewer._replace_units() for one unit into a non-gate const, or replace_type() alone."""
    x, y = replaced_position(unit, const)
    plan = wall_run.scenario_replace_rewrites(loaded, w, h, [(owner, unit, const, x, y)]) if plan_walls else None
    model.begin_unit_edit(sorted({owner} | {p for p, _u, _v in (plan.rewrites if plan else [])}))
    model.replace_type(unit, const, map_w=w, map_h=h, planned={unit.reference_id: const})
    if plan:
        wall_run.apply_replace_plan(model, plan)
    model.commit_unit_edit("Replace unit", EditHistory())
    return plan


def _replace_case(label: str, name: str) -> None:
    src = ROOT / "examples" / name
    files = {kind: OUT / f"{label}_{kind}.aoe2scenario" for kind in ("before", "stale", "out", "in")}
    site = None
    for kind in ("stale", "planned"):
        loaded = load_map_and_units(src)
        mm = loaded.map_manager
        w, h = mm.map_width, mm.map_height
        wall, west, east, _rewrites = _delete_site(loaded, w, h)
        assert site is None or site == (wall.tx, wall.ty), "the site moved between loads"
        site, owner, ref = (wall.tx, wall.ty), wall.player_id, wall.unit.reference_id
        refs = {"west": west.unit.reference_id, "east": east.unit.reference_id}
        model = UnitEditModel(loaded)
        if kind == "stale":
            write_scenario(loaded, files["before"])
            _replace_one(model, loaded, wall.unit, owner, OUTPOST, w, h, plan_walls=False)
            write_scenario(loaded, files["stale"], units=model)
            continue
        out_plan = _replace_one(model, loaded, wall.unit, owner, OUTPOST, w, h, plan_walls=True)
        write_scenario(loaded, files["out"], units=model)
        in_plan = _replace_one(model, loaded, wall.unit, owner, STONE_WALL, w, h, plan_walls=True)
        write_scenario(loaded, files["in"], units=model)

    # The M check, on the files as written: every wall around the site stores its derived shape.
    lines = []
    found: dict[str, dict[str, int]] = {}
    for kind, path in files.items():
        reread = load_map_and_units(path)
        by_ref = {u.reference_id: u for units in reread.unit_manager.units for u in units}
        found[kind] = {side: unit_sprites.variant_index(by_ref[r].rotation, 5) for side, r in refs.items()}
        consts = by_ref[ref].unit_const
        if kind in ("out", "in"):
            connectors, walls_back = wall_run.wall_scene(reread, w, h)
            near = [wb for wb in walls_back if max(abs(wb.tx - site[0]), abs(wb.ty - site[1])) <= 1]
            assert (kind == "in") == any(wb.unit.reference_id == ref for wb in near), (kind, "the replaced unit")
            for wb in near:
                derived = unit_sprites.wall_variant_from_neighbours8(
                    unit_sprites.neighbour_mask(wb.tx, wb.ty, connectors, diagonals=True)
                )
                derived = wall_run.ISOLATED_VARIANT if derived is None else derived
                stored = unit_sprites.variant_index(wb.unit.rotation, 5)
                assert stored == derived, (path.name, (wb.tx, wb.ty), stored, derived)
        lines.append(f"           {kind:6s} unit ref {ref} is const {consts}, neighbours store {found[kind]}")
    assert all(found["out"][side] != found["stale"][side] for side in refs), (found["out"], found["stale"])
    print(
        f"{label} {name} (owner {owner}), wall ref {ref} on tile {site} in an x run, replaced by Outpost {OUTPOST} "
        f"then back into Stone Wall {STONE_WALL}\n"
        f"         out reshaped {len(out_plan.rewrites)}, in reshaped {len(in_plan.rewrites)}, "
        f"in shaped {[v for _u, v in in_plan.shaped]}"
    )
    for line in lines:
        print(line)
    print("         " + " / ".join(path.name for path in files.values()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for label, name, kind, owner in CASES:
        src = ROOT / "examples" / name
        loaded = load_map_and_units(src)
        mm = loaded.map_manager
        w, h = mm.map_width, mm.map_height
        wall, along_x = _busiest_wall(loaded, w, h)
        bounds = unit_tile_bounds(wall.unit, w, h)
        wx, wy = bounds[0], bounds[2]

        before = OUT / f"{label}_before.aoe2scenario"
        after = OUT / f"{label}_after.aoe2scenario"
        write_scenario(loaded, before)

        model = UnitEditModel(loaded)
        existing_tiles, existing_walls = wall_run.wall_scene(loaded, w, h)
        path = _path(kind, wx, wy, w, h, along_x)
        plan = wall_run.plan_wall_run(
            path,
            unit_const=STONE_WALL,
            existing_tiles=existing_tiles,
            existing_walls=existing_walls,
        )
        model.begin_unit_edit(wall_run.touched_players(owner, plan))
        wall_run.apply_wall_plan(model, owner, plan)
        model.commit_unit_edit("Place wall run", EditHistory())
        write_scenario(loaded, after, units=model)

        # Read the written file back rather than trusting the in-memory
        # objects: the round trip through scenario_write is the thing the
        # game will actually see.
        reread = load_map_and_units(after)
        _t, rewritten = wall_run.wall_scene(reread, w, h)
        placed_back = sorted(
            u.rotation for u in (x.unit for x in rewritten)
            if (int(u.x), int(u.y)) in {(int(n.x), int(n.y)) for n in plan.nodes}
        )
        assert len(placed_back) == len(plan.nodes), (len(placed_back), len(plan.nodes))
        shapes = sorted({n.variant for n in plan.nodes})
        print(
            f"{label:8s} {name} (owner {owner})\n"
            f"         anchor wall at tile ({wx}, {wy}), {len(plan.nodes)} pieces placed, "
            f"{len(plan.rewrites)} rewritten, {plan.skipped} skipped\n"
            f"         indices written: {shapes}, read back: {sorted(set(placed_back))}\n"
            f"         {before.name} / {after.name}"
        )
    for label, name in GATE_CASES:
        _gate_case(label, name)
    for label, name in GATE_MOVE_CASES:
        _gate_move_case(label, name)
    for label, name in MOVE_CASES:
        _move_case(label, name)
    for label, name in GATE_CYCLE_CASES:
        _gate_cycle_case(label, name)
    for label, name in GATE_REPLACE_CASES:
        _gate_replace_case(label, name)
    for label, name in DELETE_CASES:
        _delete_case(label, name)
    for label, name in SHAPE_CASES:
        _shape_case(label, name)
    for label, name in JUNCTION_CASES:
        _junction_case(label, name)
    for label, name in REPLACE_CASES:
        _replace_case(label, name)


if __name__ == "__main__":
    main()
