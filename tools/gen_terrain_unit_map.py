#!/usr/bin/env python3
"""
Regenerates descape/terrain_unit_map.json: terrain_id -> the GAIA units the
in-game editor's Eye Candy option scatters when that terrain is painted
(species, per-tile placement odds, and whether it centers on the tile).

Pure factual id/density correspondence data extracted from the game's own
terrain table (via genieutils-py, which parses empires2_x2_p1.dat) -- not
game asset content itself, same reasoning as terrain_texture_map.json.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its terrain table.

Normally run through tools/regen_game_data.py, which also restamps
tools/game_data_stamp.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _game_root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "descape",
        help="Where to write terrain_unit_map.json (default: descape/)",
    )
    _game_root.add_dataset_argument(parser)
    args = parser.parse_args()
    _game_root.set_lenient(args.lenient)

    from genieutils.datfile import DatFile

    dat_path = _game_root.dat_path(args.aoe2de_root, args.dataset, args.dat)
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = DatFile.parse(str(dat_path))
    terrains = data.terrain_block.terrains

    mapping: dict[str, list[dict]] = {}
    for terrain_id, t in enumerate(terrains):
        count = t.number_of_terrain_units_used
        if not count:
            continue
        units = [
            {
                "id": t.terrain_unit_id[i],
                "density": t.terrain_unit_density[i],
                "centering": t.terrain_unit_centering[i],
            }
            for i in range(count)
        ]
        mapping[str(terrain_id)] = units

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "terrain_unit_map.json"
    out = {
        "_comment": (
            "terrain_id -> [{id, density, centering}, ...] GAIA units the "
            "in-game editor's Eye Candy option scatters over that terrain, in "
            "roll order. Pure factual id/density correspondence extracted from "
            "the game's own data table via genieutils-py, not game asset "
            "content itself. Regenerate with tools/gen_terrain_unit_map.py."
        ),
        "terrains": mapping,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(mapping)} terrain entries to {out_path}")


if __name__ == "__main__":
    main()
