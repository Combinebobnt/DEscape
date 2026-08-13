#!/usr/bin/env python3
"""Example v2.5 batch script: recolor every DIRT_2 tile that borders water to
BEACH.

    .venv/bin/python3 batch_scripts/recolor_dirt_near_water.py \\
        examples/2_Joan_coop_2_v0_15.aoe2scenario out.aoe2scenario

This is the exact v2.5 example ("recolor all
DIRT_2 tiles bordering water to BEACH") -- a bulk terrain edit that's tedious
to do by hand with the in-game editor's brush tools, and something it has no
scripting surface to automate at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import batch_api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    scenario = batch_api.load(args.in_path)
    mm = scenario.map_manager

    to_recolor = [
        tile
        for tile in mm.terrain
        if tile.terrain_id == TerrainId.DIRT_2
        and any(batch_api.is_water(n.terrain_id) for n in batch_api.neighbors(scenario, tile.x, tile.y, diagonal=True))
    ]
    for tile in to_recolor:
        batch_api.set_terrain(tile, TerrainId.BEACH)

    batch_api.save(scenario, args.out_path)
    print(f"Recolored {len(to_recolor)} DIRT_2 tile(s) bordering water to BEACH")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
