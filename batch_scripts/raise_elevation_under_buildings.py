#!/usr/bin/env python3
"""Example v2.5 batch script: raise the elevation under every one of a
player's buildings by a fixed amount.

    .venv/bin/python3 batch_scripts/raise_elevation_under_buildings.py \\
        examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario --player 3 --amount 1

This is the exact v2.5 example ("raise
elevation under every Player 3 building by 1") -- something the in-game
editor has no scripting surface to do at all short of clicking each building
by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import batch_api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--player", type=int, required=True, help="Player number (1-8) whose buildings to raise under")
    parser.add_argument("--amount", type=int, default=1, help="Elevation delta, can be negative (default: 1)")
    args = parser.parse_args()

    scenario = batch_api.load(args.in_path)
    buildings = batch_api.buildings_of(scenario, args.player)

    # raise_elevation_under_many, not a per-building raise_elevation_under
    # loop: buildings of the same player are often adjacent, and looping the
    # single-unit call compounds when one building's neighbor-propagation
    # reaches a not-yet-processed (or already-processed) neighbor's tile --
    # confirmed directly against real scenario data, see raise_elevation_under's
    # own docstring. raise_elevation_under_many computes every tile's target
    # from one pre-edit snapshot instead, so the result doesn't depend on
    # processing order.
    raised = batch_api.raise_elevation_under_many(scenario, buildings, args.amount)

    batch_api.save(scenario, args.out_path)
    print(f"{raised} unique tile(s) under player {args.player}'s {len(buildings)} building(s) had their elevation adjusted")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
