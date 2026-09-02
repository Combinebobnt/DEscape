#!/usr/bin/env python3
"""
Regenerates descape/object_catalog.json: per-object and per-tech integer
fields from the game's own .dat tables, backing descape/object_catalog.py's
name-resolution chain (install string -> library enum name -> .dat short
code -> "UNKNOWN_<id>").

Pure factual id/field correspondence data extracted via genieutils-py, same
reasoning as terrain_texture_map.json/tree_unit_ids.json -- commits integers
and the .dat's own short internal codes only, never a display string. Real
in-game display names are resolved at runtime from the user's own install
(asset_source.resource_string(), keyed by the string_id committed here), not
bundled as text in this repo.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its object/tech tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _object_entries(units: list) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for unit_const, unit in enumerate(units):
        if unit is None or not unit.name:
            continue
        entries[str(unit_const)] = {
            "string_id": unit.language_dll_name,
            "class": unit.class_,
            "type": unit.type,
            "hidden": bool(unit.hide_in_editor),
            "icon": unit.icon_id,
            "code": unit.name,
        }
    return entries


def _tech_entries(techs: list) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for tech_id, tech in enumerate(techs):
        if tech is None or not tech.name:
            continue
        entries[str(tech_id)] = {
            "string_id": tech.language_dll_name,
            "civ": tech.civ,
            "icon": tech.icon_id,
            "code": tech.name,
        }
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    # civs[0] (Gaia) carries the master unit list -- every other civ's own
    # list is a copy with per-civ stat overrides, same source
    # gen_tree_unit_ids.py already reads.
    objects = _object_entries(data.civs[0].units)
    techs = _tech_entries(data.techs)

    out_path = Path(__file__).resolve().parent.parent / "descape" / "object_catalog.json"
    out = {
        "_comment": (
            "Per-object and per-tech integer fields from empires2_x2_p1.dat -- "
            "no display text, that is resolved at runtime from the user's own "
            "install (see descape/object_catalog.py, descape/asset_source.py). "
            "Regenerate with tools/gen_object_catalog.py."
        ),
        "objects": objects,
        "techs": techs,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(objects)} objects and {len(techs)} techs to {out_path}")


if __name__ == "__main__":
    main()
