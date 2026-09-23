#!/usr/bin/env python3
"""
Regenerates descape/garrison_table.json: which units can hold a garrison, how
many, and what kind, extracted from the game's own unit table via
genieutils-py -- not game asset content itself, same reasoning as
unit_stats.json / terrain_texture_map.json / unit_render_data.json.

**Use `data.civs[0]` (Gaia), same as every other generator here** -- see
tools/gen_unit_stats.py for the measurement behind that choice. Neither field
emitted here is among the 13 consts that differ between Gaia and a playable
civ (all 13 differ in `line_of_sight` only).

Three emissions, because the eligibility rule needs both sides of the pair:

- `hosts`: every const with `garrison_capacity > 0`, as
  `{"cap": n}` plus `"type": mask` when the const has a `building` section.
  `garrison_type` lives on `Unit.building`, so a unit host (a Transport Ship,
  a ram) structurally has none -- the key is omitted rather than zero-filled,
  since 0 is a real mask meaning "holds nothing" (a Market's own value).
- `classes`: `unit_const -> Unit.class_` for every const in the table. The
  host-side mask is a bitfield over unit *classes*, so the occupant side
  cannot be answered without it. descape/object_catalog.json carries the same
  field, but object_catalog.py is not a leaf module (it imports
  AoE2ScenarioParser and asset_source), and descape/garrison.py is -- one
  duplicated generated column is the cheaper of the two.
- `names`: omitted deliberately. display_name() already resolves those.

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
    units = data.civs[0].units

    hosts: dict[str, dict[str, int]] = {}
    classes: dict[str, int] = {}
    for unit_const, unit in enumerate(units):
        if unit is None:
            continue
        classes[str(unit_const)] = unit.class_
        if unit.garrison_capacity <= 0:
            continue
        entry: dict[str, int] = {"cap": int(unit.garrison_capacity)}
        if unit.building is not None:
            entry["type"] = int(unit.building.garrison_type)
        hosts[str(unit_const)] = entry

    out_path = Path(__file__).resolve().parent.parent / "descape" / "garrison_table.json"
    out_path.write_text(
        json.dumps(
            {
                "_comment": "garrison capacity/type and unit class -- see tools/gen_garrison_table.py",
                "hosts": dict(sorted(hosts.items(), key=lambda kv: int(kv[0]))),
                "classes": dict(sorted(classes.items(), key=lambda kv: int(kv[0]))),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(hosts)} garrison hosts and {len(classes)} unit classes to {out_path}")


if __name__ == "__main__":
    main()
