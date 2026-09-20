#!/usr/bin/env python3
"""Re-derives the three measurements
unit_sprites.wall_variant_from_neighbours8() pins in its docstring, over
examples/. Install-free (scenario_io.load_map_and_units only).

Committed for exactly the reason tools/scan_wall_rotation.py was: the prior
attempt at that measurement was never committed, so its figures stopped
being re-derivable. Run this whenever the 8-bit function, the connector set,
or the diagonal bit sense changes.

The masks come from the SHIPPED unit_sprites.neighbour_mask() rather than a
local copy, so the NE/SE/SW/NW sense measured here is by construction the
one the write path uses. Connector tiles come from
render.unit_occupied_tiles(), not (int(x), int(y)) -- a gate is multi-tile.

Reports, over the 20 loadable files (8193 wall placements as of
2026-09-19):

1. initial_animation_frame across every wall placement -- 0 on all 8193,
   the opposite of the cliff tool's rotation-copying case.
2. The orthogonal-neighbour histogram -- {0: 1756, 1: 1199, 2: 5174,
   3: 49, 4: 15}, so 21.4% of walls have no orthogonal neighbour at all and
   the 4-bit function answers None for every one of them.
3. Stored index by diagonal neighbour shape, for those walls, split by
   unit_sprites.file_is_radian(). The integer rows pin a variant index at
   effectively 100% ({NE, SW} -> 4 at 608/609, {NW, SE} -> 3 at 359/359,
   everything else -> 2 at 179/179); the radian rows are flat noise across
   all five indices, which is the established evidence that a radian file's
   stored values carry no shape information.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import unit_sprites
from descape.render import stored_rotation, unit_occupied_tiles, unit_tile_bounds
from descape.scenario_io import load_map_and_units

NE_SW = unit_sprites.NE | unit_sprites.SW
NW_SE = unit_sprites.NW | unit_sprites.SE


def main() -> None:
    files = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.aoe2scenario"))
    frames = Counter()
    ortho_hist = Counter()
    diag_rows: dict[str, Counter] = {}
    total_walls = 0
    loaded = 0

    for path in files:
        try:
            scenario = load_map_and_units(path)
        except Exception as exc:
            print(f"  skip {path.name}: {type(exc).__name__}")
            continue
        loaded += 1
        mm = scenario.map_manager
        tile_w, tile_h = mm.map_width, mm.map_height

        connector_consts = unit_sprites.wall_connector_consts()
        connector_tiles: set[tuple[int, int]] = set()
        for units in scenario.unit_manager.units:
            for unit in units:
                if unit.unit_const in connector_consts:
                    occupied = unit_occupied_tiles(unit, tile_w, tile_h)
                    if occupied is not None:
                        connector_tiles.update(occupied)

        walls = []
        for player_id, units in enumerate(scenario.unit_manager.units):
            for unit in units:
                if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                    continue
                walls.append((player_id, unit, stored_rotation(player_id, unit)))
        if not walls:
            continue
        is_radian = unit_sprites.file_is_radian([r for *_, r in walls], 5)
        kind = "radian" if is_radian else "integer"

        for _player_id, unit, rotation in walls:
            total_walls += 1
            frames[int(unit.initial_animation_frame)] += 1
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tx, ty = bounds[0], bounds[2]
            mask = unit_sprites.neighbour_mask(tx, ty, connector_tiles, diagonals=True)
            ortho = mask & unit_sprites.ORTHOGONAL_MASK
            ortho_hist[bin(ortho).count("1")] += 1
            if ortho:
                continue
            diag = mask & ~unit_sprites.ORTHOGONAL_MASK
            if diag == NE_SW:
                row = "two on NE-SW"
            elif diag == NW_SE:
                row = "two on NW-SE"
            elif diag in (unit_sprites.NE, unit_sprites.SW):
                row = "one on NE-SW"
            elif diag in (unit_sprites.NW, unit_sprites.SE):
                row = "one on NW-SE"
            elif diag == 0:
                row = "none at all"
            else:
                row = "one on each axis"
            diag_rows.setdefault(f"{kind}/{row}", Counter())[
                unit_sprites.variant_index(rotation, 5)
            ] += 1

    print(f"loaded {loaded}/{len(files)} files, {total_walls} wall placements")
    print(f"1. initial_animation_frame histogram: {dict(sorted(frames.items()))}")
    print(f"2. orthogonal-neighbour histogram: {dict(sorted(ortho_hist.items()))}")
    print("3. stored index by diagonal shape (no orthogonal neighbour):")
    for key in sorted(diag_rows):
        counts = diag_rows[key]
        total = sum(counts.values())
        best, best_n = counts.most_common(1)[0]
        print(f"   {key:34s} n={total:5d}  index {best} in {best_n}  all={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
