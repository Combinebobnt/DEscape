#!/usr/bin/env python3
"""
Regenerates descape/tree_unit_ids.json: the set of unit_const ids that are tree
(or tree-like GAIA flora) objects.

Pure factual id/name/class correspondence data extracted from the game's own
unit table (via genieutils-py, which parses empires2_x2_p1.dat) -- not game
asset content itself, same reasoning as terrain_texture_map.json.

A unit counts as a tree if either is true:
  - its Genie engine unit class is 15 ("Trees" -- covers every biome's forest/
    bamboo/bush variants, felled-tree stumps, etc.)
  - its name contains "TREE" (case-insensitive) -- catches a couple of
    decorative outliers (e.g. "Sacred Tree") that the game data files under
    class 14 ("Special") instead of 15.

In-game, AoE2:DE's minimap renders every one of these as a flat dark green
regardless of the unit's own `minimap_color` field, which is 0 (unset) for
all of them -- confirmed by sampling: resource objects like GOLDM/PGOLD carry
real nonzero minimap_color palette indices, trees don't. That means the
minimap's tree coloring is a hardcoded engine special-case, not data-driven,
so the dark green used in descape/terrain_palette.py's TREE_COLOR is a
hand-matched approximation of that hardcoded color, not extracted from the
.dat file -- same situation terrain_palette.py's guessed (non-real-texture)
colors are already in.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its unit table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    args = parser.parse_args()

    from genieutils.datfile import DatFile

    dat_path = args.aoe2de_root / "resources/_common/dat/empires2_x2_p1.dat"
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = DatFile.parse(str(dat_path))
    gaia_units = data.civs[0].units

    TREE_CLASS = 15
    entries: list[tuple[int, str]] = []
    for unit_const, unit in enumerate(gaia_units):
        if unit is None:
            continue
        name = unit.name or ""
        if unit.class_ == TREE_CLASS or "TREE" in name.upper():
            entries.append((unit_const, name))
    entries.sort()

    out_path = Path(__file__).resolve().parent.parent / "descape" / "tree_unit_ids.json"
    out_path.write_text(
        json.dumps(
            {
                "_comment": "unit_const ids for tree/tree-like GAIA flora -- see tools/gen_tree_unit_ids.py",
                "ids": {str(uid): name for uid, name in entries},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(entries)} tree unit ids to {out_path}")


if __name__ == "__main__":
    main()
