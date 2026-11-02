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
An object's `combat_level` key (GH #49) is the raw .dat `unit.combat_level`,
present only when non-zero: it equals a trigger effect's `object_type` filter
value on every sampled unit (archer 4, villager 3, boar 1), and
trigger_range.object_type_of() reads it.
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
tech that fires at start (no requirements, no research location, no cost:
"Disable Regionals") unless the tree re-enables it (type 8, attribute 12,
value 1). The model is a standard random map game: a tech whose
`full_tech_mode` is another mode's ("Disable Spies (Regicide only)" 11,
"[FTT] ..." -1, and scenario-only "[SCEN] ..." 13, left out on purpose)
never fires. The mode meanings come from the techs' names; a negative -N is
taken to mean every mode but N. A shadow tech, one with no research location
and a cost ("Paphos Shadow Tech", team bonuses), fires only where the civ's
tree or its own team bonus zeroes its cost (type 101: "Macedonian Team
Bonus") or an available building initiates it.
A unit is available when it is enabled in the civ's own list or an available
tech enables it (type 2, any non-zero B: 27 of the game's enables, all
named ones like "Hearth Troop (make avail)", store -1) and no available tech
disables it (B 0), or an
available tech upgrades an available unit into it (type 3), or nothing
enables it anywhere and it trains only at buildings exactly one civ has
(the Krepost's Konnik). A team bonus's effects on teammates, other game
modes, and techs a building fires only in game (not through `tech_id`) are
outside it.
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

import _game_root


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
        # Omitted when 0: the trigger "object type" filter's value (GH #49, trigger_range.object_type_of()).
        if unit.combat_level:
            entries[str(unit_const)]["combat_level"] = unit.combat_level
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


def _drop_unrowed_restrictions(restrictions: list, objects: dict[str, dict]) -> list[str]:
    """Removes the `restriction` key of each editor-hidden object whose id has no
    .dat row, returning those consts. The RoR .dat has two (GH #67: 1905
    PLACEHOLDER2, 1973 NESSIE); a visible one still fails the run below."""
    dropped = []
    for unit_const, entry in objects.items():
        if entry.get("hidden") and entry.get("restriction", 0) >= len(restrictions):
            del entry["restriction"]
            dropped.append(unit_const)
    return dropped


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
            _game_root.refuse(
                f"Terrain restriction {restriction_id} has no .dat row; the placement rule needs re-checking",
                "its objects are placed without a terrain check",
            )
            continue
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


_MODIFY_TECH_COST = 101
# full_tech_mode values a standard random map game leaves out: -1 full tech tree only ("[FTT] ..."),
# and each positive mode but 1, which only that mode fires (4 Empire Wars, 8 King of the Hill,
# 11 Regicide, 13 scenario, 26 Antiquity). 0 and 1 fire always; a negative mode -N fires outside N.
_FULL_TECH_TREE_ONLY = -1
_STANDARD_MODES = (0, 1)


def _civ_availability(data) -> dict[int, tuple[set[int], set[int]]]:
    """{civ index: (available tech ids, available unit ids)} for every civ
    but Gaia, by the fixed point the module docstring describes. A second
    pass adds the units only a unique building trains (_unique_building_units())."""
    civ_indices = range(1, len(data.civs))
    first = {civ_index: _one_civ(data, civ_index, {}) for civ_index in civ_indices}
    implicit = _unique_building_units(data, first)
    return {civ_index: _one_civ(data, civ_index, implicit) for civ_index in civ_indices}


def _unique_building_units(data, availability: dict) -> dict[int, set[int]]:
    """{unit id: its train locations} for each trainable unit nothing ever
    enables (no civ list, no type 2 enable, no type 3 upgrade) whose train
    locations are all buildings exactly one civ has: Krepost's Konnik
    (Bulgarians), Donjon's Serjeant and Spearman (Sicilians). The game trains
    those there though no command enables them; heroes and scenario units,
    whose locations several civs have, stay unavailable."""
    targets = {uid for civ in data.civs for uid, unit in enumerate(civ.units) if unit is not None and unit.enabled}
    for effect in data.effects:
        for c in effect.effect_commands:
            if c.type == _ENABLE_UNIT and c.b != 0:
                targets.add(c.a)
            elif c.type == _UPGRADE_UNIT:
                targets.add(c.b)
    holders: dict[int, int] = {}
    for _techs, units in availability.values():
        for unit_id in units:
            holders[unit_id] = holders.get(unit_id, 0) + 1
    implicit: dict[int, set[int]] = {}
    for unit_id, unit in enumerate(data.civs[0].units):
        if unit is None or unit.creatable is None or unit_id in targets:
            continue
        locations = {loc.unit_id for loc in unit.creatable.train_locations if loc.unit_id >= 0}
        if locations and all(holders.get(loc) == 1 for loc in locations):
            implicit[unit_id] = locations
    return implicit


def _commands(data, effect_id: int) -> list:
    return data.effects[effect_id].effect_commands if 0 <= effect_id < len(data.effects) else []


def _in_standard_game(tech) -> bool:
    mode = tech.full_tech_mode
    return mode in _STANDARD_MODES or (mode < 0 and mode != _FULL_TECH_TREE_ONLY)


def _tech_costs(data, tree: list) -> dict[int, dict[int, float]]:
    """{tech id: {resource: amount > 0}} after the civ tree's type 101 cost
    commands (C 0 sets, 1 adds, 2 multiplies)."""
    costs = {
        tech_id: {cost.type: cost.amount for cost in tech.resource_costs if cost.type >= 0 and cost.amount > 0}
        for tech_id, tech in enumerate(data.techs)
        if tech is not None
    }
    for c in tree:
        if c.type != _MODIFY_TECH_COST or c.a not in costs:
            continue
        old = costs[c.a].get(c.b, 0)
        costs[c.a][c.b] = c.d if c.c == 0 else old + c.d if c.c == 1 else old * c.d if c.c == 2 else old
    return {tech_id: {res: amount for res, amount in cost.items() if amount > 0} for tech_id, cost in costs.items()}


def _one_civ(data, civ_index: int, implicit: dict[int, set[int]]) -> tuple[set[int], set[int]]:
    civ = data.civs[civ_index]
    tree = _commands(data, civ.tech_tree_id)
    # The civ's own team bonus can zero a shadow tech's cost too (Macedonian Team Bonus 1269).
    costs = _tech_costs(data, tree + _commands(data, civ.team_bonus_id))
    disabled = {int(c.d) for c in tree if c.type == _DISABLE_TECH}
    reenabled = {c.a for c in tree if c.type == _MODIFY_TECH and c.b == _TECH_ENABLED_ATTRIBUTE and c.d == 1}
    for tech_id, tech in enumerate(data.techs):
        if (
            tech is not None
            and tech.name
            and tech.civ in (-1, civ_index)
            and _in_standard_game(tech)
            and tech.required_tech_count == 0
            and not costs[tech_id]
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
            if (
                tech_id in techs
                or tech is None
                or not tech.name
                or tech_id in disabled
                or tech.civ not in (-1, civ_index)
                or not _in_standard_game(tech)
            ):
                continue
            initiated = any(unit_id in units for unit_id in initiators.get(tech_id, ()))
            met = sum(req in techs for req in tech.required_techs if req >= 0) + initiated
            if met < tech.required_tech_count:
                continue
            # A shadow tech (no research location, a cost: "Paphos Shadow Tech") fires only where the
            # civ's tree zeroes its cost or a building initiates it.
            if costs[tech_id] and not initiated and all(loc.location_id < 0 for loc in tech.research_locations):
                continue
            techs.add(tech_id)
            changed = True
            for c in _commands(data, tech.effect_id):
                if c.type == _ENABLE_UNIT:
                    # B is 0 to disable; every non-zero B (1, and the -1 of "Hearth Troop (make avail)") enables.
                    (turned_on if c.b != 0 else turned_off).add(c.a)
                elif c.type == _UPGRADE_UNIT:
                    upgrades.append((c.a, c.b))
        reached = (enabled - turned_off) | turned_on
        reached |= {new for old, new in upgrades if old in units or old in reached}
        reached |= {unit_id for unit_id, locations in implicit.items() if locations & reached}
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
            if c.type == _ENABLE_UNIT and c.b != 0:
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
    _game_root.add_dataset_argument(parser)
    args = parser.parse_args()
    _game_root.set_lenient(args.lenient)

    from genieutils.datfile import DatFile

    dat_path = _game_root.dat_path(args.aoe2de_root, args.dataset, args.dat)
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = DatFile.parse(str(dat_path))
    # civs[0] (Gaia) carries the master unit list -- every other civ's own
    # list is a copy with per-civ stat overrides, same source
    # gen_tree_unit_ids.py already reads.
    blank_ids = _blank_graphic_ids(data.graphics)
    if not blank_ids:
        _game_root.refuse(
            "No BLANK graphic in the .dat; the no_graphic rule needs re-checking",
            "only objects with no standing graphic are marked no_graphic",
        )
    objects = _object_entries(data.civs[0].units, blank_ids)
    techs = _tech_entries(data.techs)
    availability = _civ_availability(data)
    _add_civs(objects, "train", availability, lambda sets: sets[1])
    _add_civs(techs, "research", availability, lambda sets: sets[0])
    _add_gives(techs, objects, data)
    _add_ages(techs, data)
    terrains = _terrain_entries(data.terrain_block.terrains)
    for unit_const in _drop_unrowed_restrictions(data.terrain_restrictions, objects):
        print(f"Hidden object {unit_const}'s terrain restriction has no .dat row; left out")
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
