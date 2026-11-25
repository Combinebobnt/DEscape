#!/usr/bin/env python3
"""Writes build/unit_state_pair/alive.aoe2scenario and states.aoe2scenario,
a before/after pair for checking what the AoE2:DE editor and a test game do
with each Unit State, which DEscape draws faded for every state but Alive.

Through the real write path, from descape/templates/blank_120x120.aoe2scenario
(the AoE2 template, loaded through batch_api.load() so AoE2's game data is
active): UnitEditModel.add() places every unit at Alive, set_status() sets
the states file's rows, then write_scenario(). Both files are read back and
every unit's state checked.

Nine rows, one per Unit State 0-8 in order, south to north on screen (row 0,
Foundation, at the bottom). Each row holds, left to right, a Villager (83), a
House (70), an Oak tree (411) and a Stone Wall (117) for Player 1, then the
same four for Player 2, then for GAIA. alive.aoe2scenario is the same layout
with every unit at Alive.

    tools/gen_unit_state_pair.py [--out <dir>]
    tools/gen_unit_state_pair.py --read-back <file>

--read-back prints every unit of a saved file as (const, owner, x, y, status),
so a state picked in the game's editor (its "Default" entry) can be read off
the byte it saves.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, render_units, unit_fields
from descape.edit_history import EditHistory
from descape.scenario_io import BLANK_TEMPLATE_PATH, is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel

OUT = ROOT / "build" / "unit_state_pair"
ALIVE_NAME = "alive.aoe2scenario"
STATES_NAME = "states.aoe2scenario"

VILLAGER, HOUSE, OAK, STONE_WALL = 83, 70, 411, 117
KINDS = (VILLAGER, HOUSE, OAK, STONE_WALL)
OWNERS = (1, 2, 0)
STATES = tuple(value for value, _label in unit_fields.UNIT_STATE_CHOICES)
ALIVE = 2
# Row r, object k: low corner (X0 + ROW_STEP*r + OBJECT_STEP*k, Y0 - ROW_STEP*r + OBJECT_STEP*k).
# (+1, -1) is straight up on screen and (+1, +1) straight right, so rows stack and objects line up.
X0, Y0 = 15, 60
ROW_STEP, OBJECT_STEP = 6, 3


@dataclass(frozen=True)
class Spot:
    """One placed unit: its row's state, owner, const and low-corner tile."""

    state: int
    owner: int
    const: int
    tile: tuple[int, int]

    def point(self) -> tuple[float, float]:
        span_x, span_y = tile_span(self.const, render_units.NON_BUILDING_SPAN)
        return self.tile[0] + span_x / 2, self.tile[1] + span_y / 2


def layout() -> list[Spot]:
    """Every unit, row by row in state order, then owner, then kind."""
    spots = []
    for row, state in enumerate(STATES):
        objects = [(owner, const) for owner in OWNERS for const in KINDS]
        for k, (owner, const) in enumerate(objects):
            x = X0 + ROW_STEP * row + OBJECT_STEP * k
            y = Y0 - ROW_STEP * row + OBJECT_STEP * k
            spots.append(Spot(state, owner, const, (x, y)))
    return spots


def build(states: bool = True, source: Path = BLANK_TEMPLATE_PATH):
    """(loaded, unit model, [(spot, unit)]) with every edit applied in memory. With
    `states` False every unit stays Alive (alive.aoe2scenario)."""
    loaded = batch_api.load(source)
    mm = loaded.map_manager
    spots = layout()
    for spot in spots:
        tiles = render_units.occupied_tiles_for(spot.const, *spot.point(), mm.map_width, mm.map_height)
        if tiles is None:
            raise SystemExit(f"{spot} leaves the {mm.map_width}x{mm.map_height} map")
    units = UnitEditModel(loaded)
    units.begin_unit_edit(sorted(set(OWNERS)))
    placed = [(spot, units.add(spot.owner, spot.const, *spot.point())) for spot in spots]
    units.commit_unit_edit("Place the Unit State rows", EditHistory())
    if states:
        units.begin_unit_edit(sorted(set(OWNERS)), fields_only=True)
        for spot, unit in placed:
            if spot.state != ALIVE:
                units.set_status(unit, spot.state)
        units.commit_unit_edit("Set each row's Unit State", EditHistory())
    return loaded, units, placed


def _rows(path: Path) -> list[tuple[int, int, float, float, int]]:
    loaded = batch_api.load(path)
    return sorted(
        (int(u.unit_const), owner, float(u.x), float(u.y), int(u.status))
        for owner, player_units in enumerate(loaded.unit_manager.units)
        for u in player_units
    )


def read_back(path: Path) -> int:
    """Prints every unit of `path` as (const, owner, x, y, status)."""
    rows = _rows(path)
    labels = dict(unit_fields.UNIT_STATE_CHOICES)
    print(f"{path}: {len(rows)} units")
    for const, owner, x, y, status in rows:
        print(f"  ({const}, {owner}, {x:g}, {y:g}, {status})  {labels.get(status, 'unlabelled')}")
    return 0


def _check(path: Path, placed, states: bool) -> list[str]:
    expected = sorted(
        (spot.const, spot.owner, float(unit.x), float(unit.y), spot.state if states else ALIVE) for spot, unit in placed
    )
    got = _rows(path)
    return [] if got == expected else [f"{path.name}: {len(set(got) ^ set(expected))} units differ from the layout"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--read-back", type=Path, metavar="FILE", help="Print a saved file's units and stop")
    args = parser.parse_args()
    if args.read_back is not None:
        return read_back(args.read_back)
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, states in ((ALIVE_NAME, False), (STATES_NAME, True)):
        loaded, units, placed = build(states)
        out = args.out / name
        write_scenario(loaded, out, units=units, backup=False)
        print(f"wrote {out} ({len(placed)} units)")
        problems += _check(out, placed, states)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(
        f"\nRows south to north on screen: states {', '.join(f'{v} {label}' for v, label in unit_fields.UNIT_STATE_CHOICES)}."
        "\nEach row, left to right: Villager, House, Oak, Stone Wall for Player 1, then Player 2, then GAIA."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
