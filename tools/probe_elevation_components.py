#!/usr/bin/env python3
"""How big the shared-tile components an elevation splice re-derives get on
real files (2026-09-28 plan, Step 0). Run from a DEscape checkout root:

    .venv/bin/python3 tools/probe_elevation_components.py [FILE ...]

For every unit sharing a footprint or own tile with another (no filter,
on-map only), finds its render_cache._shared_tile_component() with no cap.
Prints a histogram of component sizes (one count per component) and the max,
against render_cache._ELEV_SPLICE_MAX_UNITS. Informational, read-only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

DEFAULT_FILES = (
    "examples/0_June_Event_Scenario.aoe2scenario",
    "examples/F7_2_Dos Pilas (648).aoe2scenario",
)


def probe(path: str) -> None:
    from descape import render, render_cache
    from descape.scenario_io import load_map_and_units
    from descape.unit_filter import UnitFilter

    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    unit_filter = UnitFilter()
    units_by_tile = render._units_by_tile(scenario, unit_filter)
    own_index = render.unit_own_tile_index(scenario)
    done: set[int] = set()
    sizes: list[int] = []
    biggest: list[tuple] = []
    total = 0
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            occupied = render.unit_occupied_tiles(unit, w, h)
            if occupied is None:
                continue
            total += 1
            own = (int(unit.x), int(unit.y))
            shared = len(own_index.get(own, ())) > 1 or any(len(units_by_tile.get(t, ())) > 1 for t in occupied)
            if not shared or id(unit) in done:
                continue
            seed = (player_id, i, unit, own, tuple(occupied))
            members = render_cache._shared_tile_component(
                scenario, units_by_tile, unit_filter, own_index, [seed], 10**9
            )
            if members is None:
                print(f"  unit {player_id}/{i} const {unit.unit_const} at {own}: not locatable")
                continue
            done.update(id(m[2]) for m in members)
            sizes.append(len(members))
            if len(members) >= max((len(b) for b in biggest), default=0):
                biggest = [members]
    hist = Counter(sizes)
    print(f"{Path(path).name}: {total} on-map units, {len(done)} on shared tiles in {len(sizes)} components")
    print("  size:count " + " ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    if biggest:
        m = biggest[0]
        consts = Counter(u.unit_const for _, _, u, _, _ in m)
        print(f"  max {len(m)} (cap {render_cache._ELEV_SPLICE_MAX_UNITS}), around {m[0][3]}, consts {dict(consts.most_common(5))}")


def main() -> None:
    for path in sys.argv[1:] or DEFAULT_FILES:
        probe(path)


if __name__ == "__main__":
    main()
