#!/usr/bin/env python3
"""Writes build/task532/{before,turned,turned_diagonal,deleted,turned_left_behind}.aoe2scenario,
asking the real AoE2:DE editor whether a turned or deleted gate's corner
pieces read right (TASK-532).

Through the real write path, from examples/atilla_1_scn_resaved.aoe2scenario
(base-game civs), loaded through batch_api.load(): TASK-531's pair source. Its
player 4 GTBA2 88 at (88.5, 125.0) covers tiles y 123..126 in a Stone Wall
line along y, with two stacked GTBC2 95 corner pieces at each of y 123.5 and
126.5. A turn plans as Rotate does (wall_run.scenario_gate_changes(fill=False),
the pieces at placement_rules.companion_landing()) and applies through
remove_many(), set_unit_const(), set_position() and
wall_run.apply_gate_move_plan(); the delete plans
wall_run.scenario_removal_rewrites() and applies remove_many() and
wall_run.apply_rewrites(). Then write_scenario().

- "before" is a zero-edit write of the source.
- "turned" is Rotate's quarter turn (B 88 to A 64) on its low corner (88,
  123): the gate on tiles x 88..91 at y 123, the low-end pair at (88.5,
  123.5) and the high-end pair at (91.5, 123.5), all as 81.
- "turned_diagonal" is one 45-degree step (B 88 to D 667): the low-end pair
  on tile (88, 126), the high-end pair on tile (91, 123), all as 671. No
  real file shows a diagonal gate's pieces.
- "deleted" removes the gate and all four pieces; the walls beside reshape.
- "turned_left_behind" is the quarter turn without the carry, as DEscape
  wrote it before TASK-532: the pieces stay at y 123.5 and 126.5 as 95.

The tool refuses to run if the source's layout differs, and reads every
written file back.

    tools/gen_task532_pair.py [--source <file>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, gate_orientation, placement_rules, wall_run
from descape.edit_history import EditHistory
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel, reanchored

SOURCE = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
OUT = ROOT / "build" / "task532"
PLAYER = 4
GATE, CORNER, STONE_WALL = 88, 95, 117
X, GATE_Y = 88.5, 125.0
PIECE_YS = [123.5, 123.5, 126.5, 126.5]
WALL_YS = [121.5, 122.5, 127.5, 128.5]
AREA = (85, 120, 95, 130)  # the tiles every check reads


def _near(loaded, consts) -> list:
    """Player 4's units of `consts` inside AREA, sorted by (y, x)."""
    x0, y0, x1, y1 = AREA
    units = loaded.unit_manager.units[PLAYER]
    return sorted((u for u in units if u.unit_const in consts and x0 <= u.x < x1 and y0 <= u.y < y1), key=lambda u: (u.y, u.x))


def _site(loaded):
    """(gate, pieces): the source's gate and its four corner pieces, or SystemExit."""
    gates = [u for u in _near(loaded, {GATE}) if (u.x, u.y) == (X, GATE_Y)]
    pieces = [u for u in _near(loaded, {CORNER}) if u.x == X]
    walls = [u.y for u in _near(loaded, {STONE_WALL}) if u.x == X]
    if len(gates) != 1 or [u.y for u in pieces] != PIECE_YS or walls != WALL_YS:
        raise SystemExit(
            f"layout differs: gates {len(gates)}, pieces at {[u.y for u in pieces]}, walls at {walls}; "
            f"expected 1 GTBA2 at ({X:g}, {GATE_Y:g}), pieces at {PIECE_YS}, walls at {WALL_YS}"
        )
    found = placement_rules.companions_by_gate(loaded, [(PLAYER, gates[0])])
    if sorted(id(p) for _pid, p in found.get(id(gates[0]), [])) != sorted(id(p) for p in pieces):
        raise SystemExit("companions_by_gate() does not find the four pieces")
    return gates[0], pieces


def _turn(source: Path, steps: int, carry: bool):
    """(loaded, model) with the gate cycled `steps` orientations, its pieces too when `carry`."""
    loaded = batch_api.load(source)
    gate, pieces = _site(loaded)
    model = UnitEditModel(loaded)
    new_const = gate_orientation.cycle_const(gate.unit_const, steps)
    x, y = reanchored(gate, new_const)
    # Every row before any write, as the viewer's _gate_turn_rows(): a landing reads the gate's old footprint.
    rows = [(gate, new_const, x, y)]
    if carry:
        rows += [(p, *placement_rules.companion_landing(gate, new_const, x, y, p)) for p in pieces]
    mm = loaded.map_manager
    plan = wall_run.scenario_gate_changes(
        loaded, mm.map_width, mm.map_height, rows, removable=lambda unit: not model.referencing(unit), fill=False
    )
    # The viewer's wholesale branch, in its order (on_unit_rotate()).
    model.begin_unit_edit(sorted({PLAYER} | (wall_run.gate_move_players(plan) if plan else set())))
    if plan:
        model.remove_many([wall for _pid, wall in plan.removals])
    for unit, const, row_x, row_y in rows:
        model.set_unit_const(unit, const)
        if (unit.x, unit.y) != (row_x, row_y):
            model.set_position(unit, row_x, row_y, unit.z)
    if plan:
        wall_run.apply_gate_move_plan(model, plan)
    model.commit_unit_edit("Rotate unit", EditHistory())
    return loaded, model


def _delete(source: Path):
    """(loaded, model, reshaped) with the gate and its four pieces removed, as Delete does."""
    loaded = batch_api.load(source)
    gate, pieces = _site(loaded)
    model = UnitEditModel(loaded)
    mm = loaded.map_manager
    units = [gate, *pieces]
    walls = wall_run.scenario_removal_rewrites(loaded, mm.map_width, mm.map_height, units)
    model.begin_unit_edit(sorted({PLAYER} | {pid for pid, _u, _v in walls}))
    model.remove_many(units)
    wall_run.apply_rewrites(model, walls)
    model.commit_unit_edit("Delete unit", EditHistory())
    return loaded, model, len(walls)


def _check(path: Path, gate, pieces) -> list[str]:
    """Problems with `path`'s gate [(const, x, y)] and pieces [(const, x, y), ...], read back."""
    reloaded = batch_api.load(path)
    problems = []
    gate_consts = set(gate_orientation.orientation_siblings(GATE))
    corner_consts = set(gate_orientation.orientation_siblings(CORNER))
    got_gate = [(u.unit_const, u.x, u.y) for u in _near(reloaded, gate_consts)]
    if got_gate != gate:
        problems.append(f"{path.name}: gate {got_gate}, expected {gate}")
    got_pieces = [(u.unit_const, u.x, u.y) for u in _near(reloaded, corner_consts)]
    if got_pieces != sorted(pieces, key=lambda p: (p[2], p[1])):
        problems.append(f"{path.name}: {len(got_pieces)} pieces {got_pieces}, expected {pieces}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="atilla_1_scn_resaved or a copy of it")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    args = parser.parse_args()
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    source = batch_api.load(args.source)
    _site(source)
    before = args.out / "before.aoe2scenario"
    write_scenario(source, before, backup=False)
    print(f"wrote {before} (no edits)")
    problems = _check(before, [(GATE, X, GATE_Y)], [(CORNER, X, y) for y in PIECE_YS])

    left = [(CORNER, X, y) for y in PIECE_YS]
    for name, steps, carry, gate, pieces in (
        ("turned", 2, True, (64, 90.0, 123.5), [(81, 88.5, 123.5)] * 2 + [(81, 91.5, 123.5)] * 2),
        ("turned_diagonal", 1, True, (667, 90.0, 125.0), [(671, 91.5, 123.5)] * 2 + [(671, 88.5, 126.5)] * 2),
        ("turned_left_behind", 2, False, (64, 90.0, 123.5), left),
    ):
        loaded, model = _turn(args.source, steps, carry)
        path = args.out / f"{name}.aoe2scenario"
        write_scenario(loaded, path, units=model, backup=False)
        print(f"wrote {path}: gate as {gate}, pieces as {pieces}")
        problems += _check(path, [gate], pieces)

    loaded, model, reshaped = _delete(args.source)
    path = args.out / "deleted.aoe2scenario"
    write_scenario(loaded, path, units=model, backup=False)
    print(f"wrote {path}: gate and its 4 pieces removed, {reshaped} walls beside reshaped")
    problems += _check(path, [], [])

    print(
        f"In the AoE2:DE editor: player 4's Stone Gate near tile ({int(X)}, {int(GATE_Y)}), a wall line along y. "
        "turned: the gate runs along x on tiles 88..91 at y 123 with end posts at both ends and none left at (88, 126); "
        "turned_diagonal: note where the end posts show at the diagonal's two tips; "
        "deleted: no lone post remains; turned_left_behind: the old result, posts left on the column."
    )
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
