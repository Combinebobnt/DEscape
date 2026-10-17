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
Three more per-object keys back Place Unit's and Move's placement rules
(GH #176), each omitted when empty: `restriction` is the raw
`unit.terrain_restriction` when above 0, `clearance` is `unit.clearance_size`
as `[x, y]` half-extents in tiles (rounded to 4 places) unless both are 0,
and `side_terrain` lists the non-negative ids of `placement_side_terrain`
(the shore a dock or shore fish needs beside it). The top-level
`terrain_restrictions` section maps each restriction id a named object uses,
plus the Beach row 2 that the shore rule reads, to the enabled terrain ids
its .dat row passes (a value above 0).
GH #169's command-button keys, each omitted when empty: an object's `train`
lists `[building, button]` for each .dat `creatable.train_locations` entry
with a building id >= 0 and a button >= 1 (1..15 the 3x5 grid, 21..35 a
second page; 0 is no button), and a tech's `research` lists its
`research_locations` the same way. An object trained at the builder task
unit 118 (a building in a villager's build menu) also carries its raw
`interface_kind`, which picks the menu page: 2 economic, 10 military.
GH #230's `civs` key, only on objects with `train` and techs with
`research`: the sorted civ indices that can have it, omitted when every civ
can and `[]` when none can (heroes, retired techs). It is derived, not
measured: a fixed point per civ where a tech is available when its `civ` is
-1 or that civ, it is not disabled, and its available required techs, plus
one for an available building that initiates it, reach its required count.
Disabled means a type 102 command in the civ's tech tree effect, or in a
tech that fires at start (no requirements, no research location: "Disable
Regionals") unless the tree re-enables it (type 8, attribute 12, value 1).
A unit is available when it is enabled in the civ's own list or an available
tech enables it (type 2) and no available tech disables it, or an available
tech upgrades an available unit into it (type 3). Team bonuses, Antiquity
and full tech tree modes, and techs a building fires only in game (not
through `tech_id`) are outside it.
A tech's `gives` key lists the units with `train` it enables (type 2) or
upgrades into (type 3), omitted when none: a trigger that enables or
researches the tech makes them available (civ_availability.file_overrides()).
A tech with `research` also carries `age`, the age it becomes researchable
in: 2 Dark, 3 Feudal, 4 Castle, 5 Imperial, from the age techs (Feudal 101,
Castle 102, Imperial 103) among its required techs, else the highest age of
those it requires. Measured in game (2026-10-16, Spanish): a player starting
in a later age already has every generic tech of an earlier one (Loom at a
Feudal start, Bloodlines at Imperial, Chemistry only at Post-Imperial).
Real in-game display names are resolved at runtime from the user's own install
(asset_source.resource_string(), keyed by the string_id committed here), not
bundled as text in this repo.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its object/tech tables.

Normally run through tools/regen_game_data.py, which also restamps
tools/game_data_stamp.json.
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
        # GH #176's placement-rule keys, each omitted when empty (0 passes every terrain).
        if unit.terrain_restriction > 0:
            entries[str(unit_const)]["restriction"] = unit.terrain_restriction
        if any(unit.clearance_size):
            entries[str(unit_const)]["clearance"] = [round(size, 4) for size in unit.clearance_size]
        side_terrain = [terrain_id for terrain_id in unit.placement_side_terrain if terrain_id >= 0]
        if side_terrain:
            entries[str(unit_const)]["side_terrain"] = side_terrain
        if unit.creatable is not None:
            train = _button_slots((loc.unit_id, loc.button_id) for loc in unit.creatable.train_locations)
            if train:
                entries[str(unit_const)]["train"] = train
                if any(building == _BUILDER_UNIT for building, _button in train):
                    entries[str(unit_const)]["interface_kind"] = unit.interface_kind
    return entries


# The villager task unit the .dat trains every building at (VMBLD).
_BUILDER_UNIT = 118


def _button_slots(locations) -> list[list[int]]:
    """[building, button] for each (building, button) pair with a building and a real button."""
    return [[building, button] for building, button in locations if building >= 0 and button >= 1]


# The .dat's Beach restriction row. No object uses it, but it is the shore a
# `side_terrain` object needs beside it (descape/placement_rules.py).
_BEACH_RESTRICTION = 2


def _terrain_restriction_entries(restrictions: list, objects: dict[str, dict], terrains: dict[str, dict]) -> dict:
    """{restriction id: passable terrain ids} for every id an entry in
    `objects` carries plus the Beach row, limited to the enabled terrains
    `terrains` lists. A row value above 0 passes (it doubles as a damage
    multiplier)."""
    used = sorted({entry["restriction"] for entry in objects.values() if "restriction" in entry} | {_BEACH_RESTRICTION})
    terrain_ids = sorted(int(terrain_id) for terrain_id in terrains)
    entries: dict[str, list[int]] = {}
    for restriction_id in used:
        if restriction_id >= len(restrictions):
            raise SystemExit(f"Terrain restriction {restriction_id} has no .dat row; the placement rule needs re-checking")
        row = restrictions[restriction_id].passable_buildable_dmg_multiplier
        entries[str(restriction_id)] = [t for t in terrain_ids if t < len(row) and row[t] > 0]
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


# Effect command types and the tech attribute the civ availability model reads.
_ENABLE_UNIT = 2
_UPGRADE_UNIT = 3
_MODIFY_TECH = 8
_DISABLE_TECH = 102
_TECH_ENABLED_ATTRIBUTE = 12


def _civ_availability(data) -> dict[int, tuple[set[int], set[int]]]:
    """{civ index: (available tech ids, available unit ids)} for every civ
    but Gaia, by the fixed point the module docstring describes."""
    return {civ_index: _one_civ(data, civ_index) for civ_index in range(1, len(data.civs))}


def _commands(data, effect_id: int) -> list:
    return data.effects[effect_id].effect_commands if 0 <= effect_id < len(data.effects) else []


def _one_civ(data, civ_index: int) -> tuple[set[int], set[int]]:
    civ = data.civs[civ_index]
    tree = _commands(data, civ.tech_tree_id)
    disabled = {int(c.d) for c in tree if c.type == _DISABLE_TECH}
    reenabled = {c.a for c in tree if c.type == _MODIFY_TECH and c.b == _TECH_ENABLED_ATTRIBUTE and c.d == 1}
    for tech in data.techs:
        if (
            tech is not None
            and tech.name
            and tech.civ in (-1, civ_index)
            and tech.required_tech_count == 0
            and all(loc.location_id < 0 for loc in tech.research_locations)
        ):
            disabled |= {int(c.d) for c in _commands(data, tech.effect_id) if c.type == _DISABLE_TECH} - reenabled
    initiators: dict[int, list[int]] = {}
    for unit_id, unit in enumerate(civ.units):
        if unit is not None and unit.building is not None and unit.building.tech_id >= 0:
            initiators.setdefault(unit.building.tech_id, []).append(unit_id)
    enabled = {unit_id for unit_id, unit in enumerate(civ.units) if unit is not None and unit.enabled}
    techs: set[int] = set()
    turned_on: set[int] = set()
    turned_off: set[int] = set()
    upgrades: list[tuple[int, int]] = []
    units = set(enabled)
    changed = True
    while changed:
        changed = False
        for tech_id, tech in enumerate(data.techs):
            if tech_id in techs or tech is None or not tech.name or tech_id in disabled or tech.civ not in (-1, civ_index):
                continue
            met = sum(req in techs for req in tech.required_techs if req >= 0)
            met += any(unit_id in units for unit_id in initiators.get(tech_id, ()))
            if met < tech.required_tech_count:
                continue
            techs.add(tech_id)
            changed = True
            for c in _commands(data, tech.effect_id):
                if c.type == _ENABLE_UNIT:
                    (turned_on if c.b == 1 else turned_off).add(c.a)
                elif c.type == _UPGRADE_UNIT:
                    upgrades.append((c.a, c.b))
        reached = (enabled - turned_off) | turned_on
        reached |= {new for old, new in upgrades if old in units or old in reached}
        if reached != units:
            units = reached
            changed = True
    return techs, units


def _add_civs(entries: dict[str, dict], key: str, availability: dict, pick) -> None:
    """Sets `civs` on every entry carrying `key`, omitted when all civs have it."""
    for id_, entry in entries.items():
        if key not in entry:
            continue
        civs = [civ_index for civ_index, sets in availability.items() if int(id_) in pick(sets)]
        if len(civs) < len(availability):
            entry["civs"] = civs


# The age techs a requirement list names, and the age each one starts (Dark Age 104 is the first).
_AGE_TECHS = {104: 2, 101: 3, 102: 4, 103: 5}
_DARK_AGE = 2


def _add_ages(techs: dict[str, dict], data) -> None:
    """Sets `age` on every tech with `research`."""
    memo: dict[int, int] = {}

    def age(tech_id: int) -> int:
        if tech_id in memo:
            return memo[tech_id]
        memo[tech_id] = _DARK_AGE  # a requirement cycle reads as Dark Age
        required = [r for r in data.techs[tech_id].required_techs if 0 <= r < len(data.techs)]
        direct = [_AGE_TECHS[r] for r in required if r in _AGE_TECHS]
        memo[tech_id] = max(direct) if direct else max((age(r) for r in required), default=_DARK_AGE)
        return memo[tech_id]

    for id_, entry in techs.items():
        if "research" in entry:
            entry["age"] = age(int(id_))


def _add_gives(techs: dict[str, dict], objects: dict[str, dict], data) -> None:
    """Sets `gives` on every tech whose effect enables or upgrades into a unit with `train`."""
    trained = {int(id_) for id_, entry in objects.items() if "train" in entry}
    for id_, entry in techs.items():
        units = set()
        for c in _commands(data, data.techs[int(id_)].effect_id):
            if c.type == _ENABLE_UNIT and c.b == 1:
                units.add(c.a)
            elif c.type == _UPGRADE_UNIT:
                units.add(c.b)
        gives = sorted(units & trained)
        if gives:
            entry["gives"] = gives


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
        research = _button_slots((loc.location_id, loc.button_id) for loc in tech.research_locations)
        if research:
            entries[str(tech_id)]["research"] = research
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "descape",
        help="Where to write object_catalog.json (default: descape/)",
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
    availability = _civ_availability(data)
    _add_civs(objects, "train", availability, lambda sets: sets[1])
    _add_civs(techs, "research", availability, lambda sets: sets[0])
    _add_gives(techs, objects, data)
    _add_ages(techs, data)
    terrains = _terrain_entries(data.terrain_block.terrains)
    terrain_restrictions = _terrain_restriction_entries(data.terrain_restrictions, objects, terrains)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "object_catalog.json"
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
        "terrain_restrictions": terrain_restrictions,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(objects)} objects, {len(terrains)} terrains, {len(terrain_restrictions)} terrain "
        f"restrictions and {len(techs)} techs to {out_path}"
    )


if __name__ == "__main__":
    main()
