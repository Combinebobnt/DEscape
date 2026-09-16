#!/usr/bin/env python3
"""
Regenerates descape/unit_stats.json: per-unit_const base in-game stats (hit
points, attack, melee armour, pierce armour, range), extracted from the
game's own unit table via genieutils-py -- not game asset content itself,
same reasoning as terrain_texture_map.json / tree_unit_ids.json /
unit_render_data.json.

**Use `data.civs[0]` (Gaia), same as every other generator here.** Gaia is a
strict superset: 2,642 non-None units vs 2,137 for British, and there are zero
consts a playable civ has that Gaia lacks. A playable civ would drop every
GAIA-only object (trees, huntables, mines) -- exactly what this repo renders
most. Gaia's values are also safe for the five fields emitted here: only 13
consts differ between Gaia and British at all (e.g. herdables/sheep such as
305/594/833, at line_of_sight 3.0 vs 2.0), and all 13 differ in
`line_of_sight` only -- not one of the fields this table emits. HP, attack,
armour and range are byte-identical between the two.

**Field homes differ between two dataclasses -- the wrong-field bug this
script is most likely to ship:**
- `Unit.hit_points`
- `Unit.type_50.displayed_attack` / `.displayed_melee_armour` /
  `.displayed_range`
- `Unit.creatable.displayed_pierce_armour` -- NOT on `type_50`

Emission rules, keys omitted (not zero-filled) when the source value is
structurally absent, so absence is encoded structurally rather than by a
sentinel a reader could mistake for real data:
- `hp` from `Unit.hit_points`, omitted when `<= 0`. This covers two distinct
  cases with one rule: genie's own "no health" sentinel (-1, on 44 consts)
  and 573 pure decoratives (cliffs, rocks) that carry 0 with no combat block
  at all -- 23% of the table. A further 17 consts have `hp == 0` *with* a
  combat block, and those still get their attack/armour/range rows since the
  `hp <= 0` rule only ever drops the `hp` key itself. A const that ends up
  with an empty dict simply has no stats to show.
- `attack` / `melee_armour` / `range` only when `unit.type_50 is not None`.
- `pierce_armour` only when `unit.creatable is not None`.
- Everything else verbatim -- no clamping, no zero-filling, no
  sentinel-guessing beyond the `hp <= 0` rule above (a negative armour value,
  e.g. House's melee armour, is real data, not a bug). `displayed_*` is
  literally the field the game's own panel uses; second-guessing it is how a
  confident-but-wrong display gets shipped.
- JSON keys use genieutils' own British spelling (`melee_armour`,
  `pierce_armour`) so a misread field is visible at a glance.

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

    stats: dict[str, dict[str, float]] = {}
    for unit_const, unit in enumerate(units):
        if unit is None:
            continue
        entry: dict[str, float] = {}
        if unit.hit_points > 0:
            entry["hp"] = unit.hit_points
        if unit.type_50 is not None:
            entry["attack"] = unit.type_50.displayed_attack
            entry["melee_armour"] = unit.type_50.displayed_melee_armour
            entry["range"] = unit.type_50.displayed_range
        if unit.creatable is not None:
            entry["pierce_armour"] = unit.creatable.displayed_pierce_armour
        if entry:
            stats[str(unit_const)] = entry

    out_path = Path(__file__).resolve().parent.parent / "descape" / "unit_stats.json"
    out_path.write_text(
        json.dumps(
            {
                "_comment": "unit_const -> base in-game stats -- see tools/gen_unit_stats.py",
                "stats": dict(sorted(stats.items(), key=lambda kv: int(kv[0]))),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(stats)} unit stat entries to {out_path}")


if __name__ == "__main__":
    main()
