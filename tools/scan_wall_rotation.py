#!/usr/bin/env python3
"""Measures the wall/gate connectivity override across examples/, install-free
(no genieutils, no AoE2:DE install -- scenario_io.load_map_and_units only).

A prior attempt at this same measurement was never committed, so its
"~1410 of 2014" figures are not re-derivable -- this script exists so the
numbers below always are. Run it any time render.wall_variant_rotation_overrides()
changes.

Per file, reports:
  - the wall-const rotation int/radian split (int counts 0.0 too -- it's
    index 0 under both conventions and so isn't evidence for either),
  - the neighbour-mask -> today's-resolved-index correlation (sanity check
    against the 98.9/99.1/96-100% figures measured on the integer corpus),
  - the changed-tile count if the connectivity override were applied,
  - how many 0.0-rotation walls sit on a non-zero mask whose derived index
    isn't 0 -- the question that decides whether a literal-looking 0.0 can
    be trusted verbatim inside a file that also has real radian-encoded
    walls (see render.wall_variant_rotation_overrides()'s docstring).

The connector tile set (for computing neighbour masks) is built over ALL
units, every player, ignoring any filter -- a wall's real shape does not
depend on which players a viewer happens to be showing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import unit_sprites
from descape.render import (
    stored_rotation,
    unit_occupied_tiles,
    unit_tile_bounds,
    wall_variant_rotation_overrides,
)
from descape.scenario_io import load_map_and_units

_MASK_NAME = {
    0: "0000 (isolated)",
    unit_sprites.WEST | unit_sprites.EAST: "1100 (+-x)",
    unit_sprites.NORTH | unit_sprites.SOUTH: "0011 (+-y)",
}


def _connector_tiles(scenario) -> set[tuple[int, int]]:
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const not in unit_sprites.WALL_CONNECTOR_CONSTS:
                continue
            occupied = unit_occupied_tiles(unit, tile_w, tile_h)
            if occupied is not None:
                tiles.update(occupied)
    return tiles


def _neighbour_mask(tx: int, ty: int, connector_tiles: set[tuple[int, int]]) -> int:
    mask = 0
    if (tx - 1, ty) in connector_tiles:
        mask |= unit_sprites.WEST
    if (tx + 1, ty) in connector_tiles:
        mask |= unit_sprites.EAST
    if (tx, ty - 1) in connector_tiles:
        mask |= unit_sprites.NORTH
    if (tx, ty + 1) in connector_tiles:
        mask |= unit_sprites.SOUTH
    return mask


def scan_file(path: Path) -> dict:
    """Measures the SHIPPED render.wall_variant_rotation_overrides() against
    this file, so a regression in that function shows up here rather than in
    a second, hand-duplicated copy of its logic. The mask/today-index
    diagnostics below (for --correlation and the 0.0-on-nonzero-mask count)
    are informational only and use their own local mask computation, since
    they need the mask even for units the real function does NOT override."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    graphic_map = unit_sprites.graphic_map()
    connector_tiles = _connector_tiles(scenario)
    overrides = wall_variant_rotation_overrides(scenario)

    candidates = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            angle_count = int(graphic_map[unit.unit_const]["angle_count"])
            # The shipped seam, not a local copy of it -- same reason the
            # docstring gives for calling wall_variant_rotation_overrides()
            # rather than reimplementing it.
            rotation = stored_rotation(player_id, unit)
            candidates.append((player_id, i, unit, rotation, angle_count))

    int_count = sum(1 for *_, rotation, angle_count in candidates if unit_sprites.is_literal_variant_index(rotation, angle_count))
    radian_count = len(candidates) - int_count

    changed = 0
    mask_to_today_index: dict[int, Counter] = defaultdict(Counter)
    zero_rotation_on_nonzero_mask = 0

    for player_id, i, unit, rotation, angle_count in candidates:
        bounds = unit_tile_bounds(unit, tile_w, tile_h)
        if bounds is None:
            continue
        tx, ty = bounds[0], bounds[2]
        mask = _neighbour_mask(tx, ty, connector_tiles)
        today_index = unit_sprites.variant_index(rotation, angle_count)
        mask_to_today_index[mask][today_index] += 1

        if rotation == 0.0 and mask != 0:
            derived = unit_sprites.wall_variant_from_neighbours(mask)
            if derived is not None and derived != 0:
                zero_rotation_on_nonzero_mask += 1

        override = overrides.get((player_id, i))
        if override is not None and int(override) != today_index:
            changed += 1

    total_walls = len(candidates)
    return {
        "path": path.name,
        "total_walls": total_walls,
        "int_count": int_count,
        "radian_count": radian_count,
        "changed": changed,
        "mask_to_today_index": mask_to_today_index,
        "zero_rotation_on_nonzero_mask": zero_rotation_on_nonzero_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples")
    parser.add_argument("--correlation", action="store_true", help="Print the mask->today-index table across all files")
    args = parser.parse_args()

    files = sorted(
        p for p in args.scenario_dir.iterdir()
        if p.suffix in (".aoe2scenario", ".scx2") or p.name == "play_Test"
    )
    if not files:
        print(f"No scenario files found under {args.scenario_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'file':45s} {'walls':>6s} {'int':>6s} {'radian':>6s} {'changed':>8s}")
    total_changed_radian = 0
    total_changed_int = 0
    total_zero_on_nonzero = 0
    combined_mask_index: dict[int, Counter] = defaultdict(Counter)
    for path in files:
        try:
            result = scan_file(path)
        except Exception as exc:  # noqa: BLE001 -- a scan tool, not production code
            print(f"{path.name:45s} FAILED: {exc}")
            continue
        print(
            f"{result['path']:45s} {result['total_walls']:6d} {result['int_count']:6d} "
            f"{result['radian_count']:6d} {result['changed']:8d}"
        )
        if result["radian_count"] > 0:
            total_changed_radian += result["changed"]
        else:
            total_changed_int += result["changed"]
        total_zero_on_nonzero += result["zero_rotation_on_nonzero_mask"]
        for mask, counter in result["mask_to_today_index"].items():
            combined_mask_index[mask].update(counter)

    print()
    print(f"Changed tiles across radian-encoded files: {total_changed_radian}")
    print(f"Changed tiles across integer-only files (must be 0):  {total_changed_int}")
    print(f"0.0-rotation walls on a non-zero mask whose derived index != 0: {total_zero_on_nonzero}")

    if args.correlation:
        print()
        print("mask -> today's variant_index() distribution (all files combined):")
        for mask, counter in sorted(combined_mask_index.items()):
            label = _MASK_NAME.get(mask, format(mask, "04b"))
            total = sum(counter.values())
            dist = ", ".join(f"{idx}:{n}" for idx, n in sorted(counter.items()))
            print(f"  {label:14s} n={total:5d}  {dist}")


if __name__ == "__main__":
    main()
