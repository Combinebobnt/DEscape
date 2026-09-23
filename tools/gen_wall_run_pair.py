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

Writes into build/wall_run_pairs/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import shape_tools, unit_sprites, wall_run
from descape.edit_history import EditHistory
from descape.render import unit_tile_bounds
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel

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


if __name__ == "__main__":
    main()
