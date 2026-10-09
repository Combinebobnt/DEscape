#!/usr/bin/env python3
"""
Regenerates descape/object_catalog.json: per-object and per-tech integer
fields from the game's own .dat tables, backing descape/object_catalog.py's
name-resolution chain (install string -> library enum name -> .dat short
code -> "UNKNOWN_<id>").

Pure factual id/field correspondence data extracted via genieutils-py, same
reasoning as terrain_texture_map.json/tree_unit_ids.json -- commits integers
and the .dat's own short internal codes only, never a display string.
An object's `no_graphic` key is present (always True) only when its .dat
standing_graphic[0] is -1 or the .dat's own `BLANK` graphic (no art file);
unit_kind.invisible_consts() is derived from it. The 2026-09-22 game patch
moved 20 helper consts (Empty TC annex, sheep/mole annexes, ...) from -1 to
`BLANK`, so the game treats the two alike; the user decided (2026-09-26) that
this also covers the consts that already pointed at `BLANK` (flares, debris).
Each object also carries `hero_mode`, the raw `unit.creatable.hero_mode` (0
with no creatable block), which object_catalog.derived_category() reads to
put a dat-only object under Heroes. The `terrains` section holds every
enabled .dat terrain as `{string_id, code, hidden}` (`code` is the terrain's
internal name, `hidden` its hide_in_editor), so terrain ids the library's
TerrainId lacks can still be listed and named.
An object's `resources` key (GH #145) is present only for the consts whose
.dat resource storages hold a positive food, wood, gold or stone amount, as
`{"food": n, ...}` with only the positive keys; player_stats sums it into
the map's resource totals.
An object's `obstruction` key (GH #149) is the raw .dat
`unit.obstruction_type`, present only when non-zero: 0 is the passable value
(grass, plants, rubble), anything else blocks movement.
unit_kind.obstacle_consts() is derived from it.
Real in-game display names are resolved at runtime from the user's own install
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


def _blank_graphic_ids(graphics: list) -> frozenset[int]:
    return frozenset(i for i, g in enumerate(graphics) if g is not None and g.name.strip().upper() == "BLANK")


def _object_entries(units: list, blank_ids: frozenset[int]) -> dict[str, dict]:
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
            "hero_mode": unit.creatable.hero_mode if unit.creatable is not None else 0,
        }
        # Omitted when false, so the committed diff is only the ~200 consts that need it.
        if unit.standing_graphic[0] == -1 or unit.standing_graphic[0] in blank_ids:
            entries[str(unit_const)]["no_graphic"] = True
        resources = _resources(unit.resource_storages)
        if resources:
            entries[str(unit_const)]["resources"] = resources
        # Omitted when 0 (passable), so the committed diff is only the consts that block.
        if unit.obstruction_type:
            entries[str(unit_const)]["obstruction"] = unit.obstruction_type
    return entries


# Genie resource type -> catalog key. 17 is the Genie enum's Fish Storage,
# gathered as food in-game. Every other type (4 population, ...) is not a
# gatherable amount and is left out.
_RESOURCE_KEYS = {0: "food", 17: "food", 1: "wood", 2: "stone", 3: "gold"}


def _resources(storages) -> dict[str, int]:
    """{food/wood/gold/stone: amount}, keys only for a positive amount; the
    whole key is omitted from the entry when this is empty."""
    totals: dict[str, float] = {}
    for storage in storages:
        key = _RESOURCE_KEYS.get(storage.type)
        if key is not None and storage.amount > 0:
            totals[key] = totals.get(key, 0) + storage.amount
    return {key: round(amount) for key, amount in totals.items()}


def _terrain_entries(terrains: list) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for terrain_id, terrain in enumerate(terrains):
        if terrain is None or not terrain.enabled:
            continue
        entries[str(terrain_id)] = {
            "string_id": terrain.string_id,
            "code": terrain.name,
            "hidden": bool(terrain.hide_in_editor),
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
    blank_ids = _blank_graphic_ids(data.graphics)
    if not blank_ids:
        raise SystemExit("No BLANK graphic in the .dat; the no_graphic rule needs re-checking")
    objects = _object_entries(data.civs[0].units, blank_ids)
    techs = _tech_entries(data.techs)
    terrains = _terrain_entries(data.terrain_block.terrains)

    out_path = Path(__file__).resolve().parent.parent / "descape" / "object_catalog.json"
    out = {
        "_comment": (
            "Per-object, per-terrain and per-tech integer fields and internal "
            "codes from empires2_x2_p1.dat -- "
            "no display text, that is resolved at runtime from the user's own "
            "install (see descape/object_catalog.py, descape/asset_source.py). "
            "Regenerate with tools/gen_object_catalog.py."
        ),
        "objects": objects,
        "techs": techs,
        "terrains": terrains,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(objects)} objects, {len(terrains)} terrains and {len(techs)} techs to {out_path}")


if __name__ == "__main__":
    main()
