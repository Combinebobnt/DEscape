#!/usr/bin/env python3
"""Example batch script: scatter GAIA fish at random, reproducible positions
across water.

    .venv/bin/python3 batch_scripts/scatter_fish_in_water.py \\
        examples/2_Joan_coop_1_v0_13.aoe2scenario out.aoe2scenario --count 40 --seed 7

By default every water tile on the map is eligible. --pond-at X Y restricts
it to the one connected body of water containing that tile. Something the
in-game editor has no equivalent for: its only "random" placement is
rotation, and it has no scripting surface at all.

The pond is grown by a local BFS over an is_water() tile set rather than
descape.fill_tools.contiguous_region(), which keys on exact terrain_id and
would split a bay of WATER + WATER_DEEP + SHALLOWS into three regions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AoE2ScenarioParser.datasets.other import OtherInfo

from descape import batch_api
from descape.scatter import scatter_units
from descape.unit_model import UnitEditModel


def water_tiles(scenario) -> set[tuple[int, int]]:
    # Flat indexing, never mm.get_tile(): it raises on a non-square map.
    mm = scenario.map_manager
    width = mm.map_width
    return {(i % width, i // width) for i, tile in enumerate(mm.terrain) if batch_api.is_water(tile.terrain_id)}


def connected_body(water: set[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    if start not in water:
        return set()
    body = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in water and nxt not in body:
                body.add(nxt)
                stack.append(nxt)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--count", type=int, default=40, help="How many fish to place (default: 40)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed; omit for a different result every run")
    parser.add_argument(
        "--unit-const",
        type=int,
        default=OtherInfo.FISH_SALMON.ID,
        help=f"Fish unit_const (default: {OtherInfo.FISH_SALMON.ID}, Salmon)",
    )
    parser.add_argument("--pond-at", type=int, nargs=2, metavar=("X", "Y"), help="Only the water body containing X Y")
    args = parser.parse_args()

    scenario = batch_api.load(args.in_path)
    tiles = water_tiles(scenario)
    if args.pond_at is not None:
        tiles = connected_body(tiles, tuple(args.pond_at))
        if not tiles:
            parser.error(f"tile {tuple(args.pond_at)} is not water")

    units = UnitEditModel(scenario)
    placed = scatter_units(scenario, units, tiles, args.unit_const, count=args.count, seed=args.seed)

    batch_api.save(scenario, args.out_path, units=units)
    print(f"Placed {len(placed)} fish across {len(tiles)} eligible water tile(s)")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
