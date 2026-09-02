#!/usr/bin/env python3
"""Example phase 3.5a batch script: reassign every unit of a given
unit_const belonging to one player over to another player.

    .venv/bin/python3 batch_scripts/reassign_units_by_const.py \\
        examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario \\
        --unit-const 4 --from-player 1 --to-player 2

Demonstrates the one thing that changed about batch_api in phase 3.5a: a
unit edit only persists when it goes through a
descape.unit_model.UnitEditModel and that model is passed to
batch_api.save(units=...) -- mutating a Unit object's `player` directly
(the banned setter) and saving with no model would silently drop the edit,
same as before this phase existed. See batch_api.py's own module docstring
for why that containment is deliberate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import batch_api
from descape.unit_model import UnitEditModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--unit-const", type=int, required=True, help="unit_const to match")
    parser.add_argument("--from-player", type=int, required=True, help="Source player (0 = GAIA, 1-8)")
    parser.add_argument("--to-player", type=int, required=True, help="Destination player (0 = GAIA, 1-8)")
    args = parser.parse_args()

    scenario = batch_api.load(args.in_path)
    units = UnitEditModel(scenario)

    to_move = [u for u in scenario.unit_manager.get_player_units(args.from_player) if u.unit_const == args.unit_const]
    for unit in to_move:
        units.reassign(unit, args.to_player)

    batch_api.save(scenario, args.out_path, units=units)
    print(f"Reassigned {len(to_move)} unit(s) with unit_const={args.unit_const} from player {args.from_player} to {args.to_player}")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
