#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming the automatic
shoreline in the real AoE2:DE editor -- the half of testing nothing in this
session can drive.

Deliberately asymmetric: an L-shaped water blob pushed against the map's own
west edge, so one run checks the clipping and the outside-corner behaviour at
once, where a centred square blob would check neither. Both files go through
the real write path (the terrain writes + beach_edges + scenario_write, the
same code the GUI uses), never a byte patch.

Writes into build/beach_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape.batch_api import set_terrain
from descape.beach_edges import apply_beach_ring
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.terrain_palette import name_for_terrain_id

FIXTURE = ROOT / "tests" / "fixtures" / "real_blank_240x240.aoe2scenario"
OUT_DIR = ROOT / "build" / "beach_pair"

WATER_DEEP = 22
BEACH_WIDTH = 1

# An L against the west edge: a vertical arm at x 0..3 and a horizontal arm
# running east from its foot. x starts at 0 on purpose -- the ring there has
# nowhere to go west, which is the clipping case.
ARM_V = [(x, y) for y in range(20, 40) for x in range(4)]
ARM_H = [(x, y) for y in range(36, 40) for x in range(4, 24)]
CORE = ARM_V + ARM_H


def main() -> int:
    if not FIXTURE.is_file():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_path = OUT_DIR / "before_blank_240x240.aoe2scenario"
    scenario = load_map_and_units(FIXTURE)
    write_scenario(scenario, before_path, backup=False)
    print(f"wrote {before_path}")

    scenario = load_map_and_units(FIXTURE)
    mm = scenario.map_manager
    for x, y in CORE:
        set_terrain(mm.terrain[y * mm.map_width + x], WATER_DEEP)
    changed = apply_beach_ring(mm, CORE, WATER_DEEP, None, BEACH_WIDTH)

    after_path = OUT_DIR / "after_l_shaped_water_with_beach.aoe2scenario"
    write_scenario(scenario, after_path, backup=False)
    print(f"wrote {after_path}")

    beaches: dict[int, int] = {}
    for index in changed:
        tid = mm.terrain[index].terrain_id
        beaches[tid] = beaches.get(tid, 0) + 1
    print()
    print(f"core: {len(CORE)} tiles set to {name_for_terrain_id(WATER_DEEP)}")
    print(f"ring: {len(changed)} tiles, width {BEACH_WIDTH}")
    for tid, count in sorted(beaches.items()):
        print(f"  {count} x {name_for_terrain_id(tid)} ({tid})")
    print()
    print("What to check in the AoE2:DE editor, opening the 'after' file:")
    print("  1. An L of deep water in the map's north-west, its long arm")
    print("     running down the west edge and its foot running east.")
    print("  2. A one-tile beach all the way around the L's outside, with a")
    print("     clean inside corner where the two arms meet.")
    print("  3. NO beach west of x=0 wrapping onto the previous row's east")
    print("     edge -- that is the clipping check.")
    print("  4. The 'before' file is the same map with none of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
