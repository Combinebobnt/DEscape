#!/usr/bin/env python3
"""CLI: prints a text summary of an .aoe2scenario file's Map and Units."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape.scenario_io import load_map_and_units
from descape.terrain_palette import name_for_terrain_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Path to an .aoe2scenario file")
    parser.add_argument("--png", type=Path, help="Also render a PNG to this path")
    parser.add_argument("--scale", type=int, default=1, help="PNG upscale factor")
    parser.add_argument(
        "--iso",
        action="store_true",
        help="Render --png in Stepped isometric mode (real per-tile elevation "
        "displacement) instead of Flat. No effect without --png. Units render "
        "here too, each at its own tile's elevation.",
    )
    args = parser.parse_args()

    s = load_map_and_units(args.scenario)
    mm, um = s.map_manager, s.unit_manager

    print(f"File:              {s.path.name}")
    print(f"Scenario version:  {s.scenario_version}")
    print(f"Map size:          {mm.map_width} x {mm.map_height}")
    print(f"Trigger tail:      {len(s.trigger_tail):,} bytes (not parsed)")
    print()

    terrain_hist = Counter(t.terrain_id for t in mm.terrain)
    print("Terrain histogram (top 10):")
    for tid, count in terrain_hist.most_common(10):
        pct = 100 * count / len(mm.terrain)
        print(f"  {name_for_terrain_id(tid):28s} {count:6d} tiles ({pct:4.1f}%)")
    print()

    elevations = [t.elevation for t in mm.terrain]
    print(f"Elevation range:   {min(elevations)}..{max(elevations)}")
    print()

    print("Units per player:")
    total = 0
    for player_id, units in enumerate(um.units):
        label = "GAIA" if player_id == 0 else f"Player {player_id}"
        print(f"  {label:10s} {len(units):6d} units")
        total += len(units)
    print(f"  {'Total':10s} {total:6d} units")

    if args.png:
        from descape.render import save_png

        save_png(s, str(args.png), scale=args.scale, isometric=args.iso)
        print(f"\nWrote {args.png}{' (Stepped isometric)' if args.iso else ''}")


if __name__ == "__main__":
    main()
