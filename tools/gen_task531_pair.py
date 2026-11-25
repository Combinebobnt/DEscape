#!/usr/bin/env python3
"""Writes build/task531/{before,after,left_behind}.aoe2scenario, asking the
real AoE2:DE editor whether a gate moved with its corner pieces reads as one
whole gate (TASK-531).

Through the real write path, from examples/atilla_1_scn_resaved.aoe2scenario
(base-game civs), loaded through batch_api.load(). Its player 4 GTBA2 88 at
(88.5, 125.0) covers tiles y 123..126 in a Stone Wall line along y, with two
stacked GTBC2 95 corner pieces at each of y 123.5 and 126.5. The move is one
tile +y with "Moving a gate updates walls" on, as Shift+Down does: planned
through placement_rules.companions_by_gate() and
wall_run.scenario_move_plan(gate_walls=True), applied through remove_many(),
set_position() and wall_run.apply_move_plan(), then write_scenario().

- "before" is a zero-edit write of the source.
- "after" moves the gate to y 126.0 (tiles 124..127) and its four pieces +1
  (124.5 twice, 127.5 twice); the wall on tile 127 is removed and a new
  Stone Wall fills tile 123.
- "left_behind" makes the same move without the carry, as DEscape wrote it
  before TASK-531: the pieces stay at 123.5 and 126.5 and tile 123 gets no
  fill, since the pieces left there still stand on it.

The tool refuses to run if the source's layout differs, and reads every
written file back.

    tools/gen_task531_pair.py [--source <file>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, placement_rules, wall_run
from descape.edit_history import EditHistory
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
OUT = ROOT / "build" / "task531"
PLAYER = 4
GATE, CORNER, STONE_WALL = 88, 95, 117
X = 88.5
GATE_Y, MOVED_Y = 125.0, 126.0
PIECE_YS = [123.5, 123.5, 126.5, 126.5]
WALL_YS = [121.5, 122.5, 127.5, 128.5]


def _column(loaded, const: int) -> list:
    """Player 4's units of `const` on x 88.5, sorted by y."""
    return sorted((u for u in loaded.unit_manager.units[PLAYER] if u.unit_const == const and u.x == X), key=lambda u: u.y)


def _site(loaded):
    """(gate, pieces): the source's gate and its four corner pieces, or SystemExit."""
    gates = [u for u in _column(loaded, GATE) if u.y == GATE_Y]
    pieces = _column(loaded, CORNER)
    walls = [u.y for u in _column(loaded, STONE_WALL) if 120 <= u.y < 130]
    if len(gates) != 1 or [u.y for u in pieces] != PIECE_YS or walls != WALL_YS:
        raise SystemExit(
            f"layout differs: gates {len(gates)}, pieces at {[u.y for u in pieces]}, walls at {walls}; "
            f"expected 1 GTBA2 at ({X:g}, {GATE_Y:g}), pieces at {PIECE_YS}, walls at {WALL_YS}"
        )
    found = placement_rules.companions_by_gate(loaded, [(PLAYER, gates[0])])
    if sorted(id(p) for _pid, p in found.get(id(gates[0]), [])) != sorted(id(p) for p in pieces):
        raise SystemExit("companions_by_gate() does not find the four pieces")
    return gates[0], pieces


def _move(source: Path, carry: bool):
    """(loaded, model, plan) with the gate moved one tile +y, its pieces too when `carry`."""
    loaded = batch_api.load(source)
    gate, _pieces = _site(loaded)
    model = UnitEditModel(loaded)
    moves = [(gate, gate.x, gate.y + 1)]
    if carry:
        found = placement_rules.companions_by_gate(loaded, [(PLAYER, gate)])
        moves += [(piece, piece.x, piece.y + 1) for _pid, piece in found[id(gate)]]
    mm = loaded.map_manager
    plan = wall_run.scenario_move_plan(
        loaded, mm.map_width, mm.map_height, moves, gate_walls=True, removable=lambda unit: not model.referencing(unit)
    )
    assert plan, "the move planned nothing"
    # The viewer's wholesale branch, in its order (_move_units()).
    model.begin_unit_edit(sorted({PLAYER} | wall_run.move_plan_players(plan)))
    model.remove_many([wall for _pid, wall in plan.removals])
    for unit, x, y in moves:
        model.set_position(unit, x, y, unit.z)
    wall_run.apply_move_plan(model, plan)
    model.commit_unit_edit("Nudge unit", EditHistory())
    return loaded, model, plan


def _check(path: Path, piece_ys: list[float], tile_123: bool) -> list[str]:
    reloaded = batch_api.load(path)
    problems = []
    gates = [u.y for u in _column(reloaded, GATE)]
    if MOVED_Y not in gates or GATE_Y in gates:
        problems.append(f"{path.name}: gate at y {gates}, expected {MOVED_Y:g}")
    pieces = [u.y for u in _column(reloaded, CORNER)]
    if pieces != piece_ys:
        problems.append(f"{path.name}: pieces at y {pieces}, expected {piece_ys}")
    walls = [u.y for u in _column(reloaded, STONE_WALL) if 120 <= u.y < 130]
    expected = [121.5, 122.5, *([123.5] if tile_123 else []), 128.5]
    if walls != expected:
        problems.append(f"{path.name}: walls at y {walls}, expected {expected} (tile 127 cleared)")
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

    problems = []
    for name, carry, piece_ys in (
        ("after", True, [124.5, 124.5, 127.5, 127.5]),
        ("left_behind", False, PIECE_YS),
    ):
        loaded, model, plan = _move(args.source, carry)
        path = args.out / f"{name}.aoe2scenario"
        write_scenario(loaded, path, units=model, backup=False)
        print(
            f"wrote {path}: gate to ({X:g}, {MOVED_Y:g}), pieces at y {piece_ys}, "
            f"{len(plan.removals)} wall removed, {len(plan.fills)} filled, {plan.blocked} occupied tiles left open"
        )
        problems += _check(path, piece_ys, tile_123=carry)
    print(
        f"In the AoE2:DE editor: player 4's Stone Gate near tile ({int(X)}, {int(GATE_Y)}), a wall line along y. "
        "after: the gate whole on tiles y 124..127 with its end posts at both ends and a Stone Wall at tile 123; "
        "left_behind: posts still at tiles 123 and 126 and tile 123 open."
    )
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
