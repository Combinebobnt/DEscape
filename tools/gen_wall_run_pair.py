#!/usr/bin/env python3
"""Writes before/after .aoe2scenario pairs for confirming the Wall Run tool
in the real AoE2:DE editor -- the half of testing nothing in a terminal
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

CASES = [
    ("integer", "2_Joan_coop_1_v0_13.aoe2scenario", "L"),
    ("radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "L"),
    ("shallow", "2_Joan_coop_1_v0_13.aoe2scenario", "shallow"),
    # A fourth, beyond the plan's three: the diagonal indices 3 and 4 are the
    # highest-risk thing here (nothing in this tree has ever written them),
    # and the plan's shallow case only exercises them in an integer file.
    ("shallow_radian", "C2_ElCid_coop_1_v0_16.aoe2scenario", "shallow_nwse"),
]


def _busiest_wall(scenario, w: int, h: int):
    """An existing wall with an orthogonal neighbour, so the L's far end has
    a real run to T-junction into rather than a lone piece."""
    tiles, walls = wall_run.wall_scene(scenario, w, h)
    for wall in walls:
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, tiles)
        if mask in (unit_sprites.WEST | unit_sprites.EAST, unit_sprites.NORTH | unit_sprites.SOUTH):
            return wall, tiles, walls
    raise SystemExit("no straight-run wall found to junction into")


def _path(kind: str, wx: int, wy: int, w: int, h: int):
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
    for label, name, kind in CASES:
        src = ROOT / "examples" / name
        loaded = load_map_and_units(src)
        mm = loaded.map_manager
        w, h = mm.map_width, mm.map_height
        wall, _tiles, _walls = _busiest_wall(loaded, w, h)
        bounds = unit_tile_bounds(wall.unit, w, h)
        wx, wy = bounds[0], bounds[2]

        before = OUT / f"{label}_before.aoe2scenario"
        after = OUT / f"{label}_after.aoe2scenario"
        write_scenario(loaded, before)

        model = UnitEditModel(loaded)
        existing_tiles, existing_walls = wall_run.wall_scene(loaded, w, h)
        path = _path(kind, wx, wy, w, h)
        plan = wall_run.plan_wall_run(
            path,
            unit_const=STONE_WALL,
            existing_tiles=existing_tiles,
            existing_walls=existing_walls,
        )
        model.begin_unit_edit(wall_run.touched_players(PLAYER, plan))
        wall_run.apply_wall_plan(model, PLAYER, plan)
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
            f"{label:8s} {name}\n"
            f"         anchor wall at tile ({wx}, {wy}), {len(plan.nodes)} pieces placed, "
            f"{len(plan.rewrites)} rewritten, {plan.skipped} skipped\n"
            f"         indices written: {shapes}, read back: {sorted(set(placed_back))}\n"
            f"         {before.name} / {after.name}"
        )


if __name__ == "__main__":
    main()
