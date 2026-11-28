#!/usr/bin/env python3
"""
Regenerates descape/terrain_classes.json: a terrain_id -> water/land family
and climate table.

This is pure factual classification data extracted from the game's own terrain
table (via genieutils-py, which parses empires2_x2_p1.dat) -- not game asset
content itself. It records what the game already believes about each terrain
slot, which is what lets the auto-beach edge pick a shoreline terrain by data
rather than by guessing from enum names.

The source field is genieutils.terrainblock.Terrain.is_water, which is not a
boolean despite the name -- it is a bitmask:

    0x01 medium water   0x02 deep water   0x04 shallow water
    0x08 shallows       0x10 beach        0x20 land
    +0x40 ice climate   +0x80 snow climate

Measured across all 200 slots, every enabled terrain has exactly one family
bit set, the high bits are only ever 0/0x40/0x80, and the only enabled slots
reading 0x00 are named OBSOLETE in the .dat itself. This script FAILS rather
than writing a file if any of that stops holding: a violation is the signal
that a game update changed the field's meaning, and papering over it would
silently mis-shore every map.

The 1999, 1997 and HD Edition .dats (--dataset aoc, aoe1, hd) have no such
byte: their values are the hand tables _AOC_IS_WATER, _AOE1_IS_WATER and
_HD_IS_WATER, each checked against that .dat's own ship restriction.

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

FAMILY_BITS = {
    0x01: "medium_water",
    0x02: "deep_water",
    0x04: "shallow_water",
    0x08: "shallows",
    0x10: "beach",
    0x20: "land",
}
CLIMATE_BITS = {0x00: "temperate", 0x40: "ice", 0x80: "snow"}

FAMILY_MASK = 0x3F
CLIMATE_MASK = 0xC0


def decode(raw: int) -> tuple[str, str]:
    """(family, climate) for one raw is_water byte, or a raise. Raising is
    the point -- see the module docstring."""
    family_bits = raw & FAMILY_MASK
    climate_bits = raw & CLIMATE_MASK
    if climate_bits not in CLIMATE_BITS:
        raise ValueError(f"unknown climate bits {climate_bits:#04x} in {raw:#04x}")
    if family_bits == 0:
        return "obsolete", CLIMATE_BITS[climate_bits]
    if family_bits not in FAMILY_BITS:
        raise ValueError(f"expected exactly one family bit, got {family_bits:#04x} in {raw:#04x}")
    return FAMILY_BITS[family_bits], CLIMATE_BITS[climate_bits]


# The 1999 .dat has no is_water byte. Each value is DE's byte for the same id,
# which keeps the 1999 terrain at that id, except 33, 34, 38 and 39: the 1999
# Snow Dirt, Snow Grass, Snow Road and Snow Road 2, which DE made obsolete.
_AOC_IS_WATER = {
    **dict.fromkeys(range(41), 0x20),
    1: 0x04, 2: 0x10, 4: 0x08, 15: 0x01, 22: 0x02, 23: 0x01,
    21: 0xA0, 26: 0x48, 32: 0xA0, 33: 0xA0, 34: 0xA0, 35: 0x60, 36: 0xA0, 37: 0x50, 38: 0xA0, 39: 0xA0,
}
# The 1997 .dat has none either. Its 23 terrains are Return of Rome's at the same ids and
# names (checked on the run), so each value is that game's byte (descape/ror/terrain_classes.json).
_AOE1_IS_WATER = {
    **dict.fromkeys(range(23), 0x20),
    1: 0x04, 2: 0x10, 4: 0x08, 15: 0x01, 21: 0xA0, 22: 0x02,
}
_AOE1_TERRAIN_NAMES = (
    "Grass", "Water", "Beach", "Thin River", "Shallows", "Jungle Edge", "Desert", "Crop", "Rows", "Wheat",
    "Forest", "Dirt", "Grass 2", "Desert Palm", "Desert Impas", "Water Impass", "Grass Impass", "Fog",
    "Forest Edge", "Pine Forest", "Jungle", "Pine Edge", "Water Dark",
)
# HD Edition's .dat has none either. Its ids 0..67 are DE's at the same names, so each is DE's
# byte, except the eight DE made obsolete (33, 34, 38, 39 snow land as in 1999; 43, 44, 61, 62
# land). 68..99 are HD's own modding slots, classed by their names (Reserved and Grass land,
# Shallows, Shallow Water, Normal Water, Deep Water); the ship check below confirms every water id.
_HD_IS_WATER = {
    **dict.fromkeys(range(100), 0x20),
    1: 0x04, 2: 0x10, 4: 0x08, 15: 0x01, 22: 0x02, 23: 0x01,
    21: 0xA0, 26: 0x48, 32: 0xA0, 33: 0xA0, 34: 0xA0, 35: 0x60, 36: 0xA0, 37: 0x50, 38: 0xA0, 39: 0xA0,
    51: 0x10, 52: 0x10, 53: 0x10, 54: 0x08, 55: 0x08, 57: 0x02, 58: 0x04, 59: 0x08,
    **dict.fromkeys(range(63, 68), 0x08),
    **dict.fromkeys(range(90, 95), 0x08), 95: 0x04, 96: 0x04, 97: 0x01, 98: 0x01, 99: 0x02,
}
# The HD slot names _HD_IS_WATER's 68..99 were classed by (2026-11-27).
_HD_SLOT_NAMES = (
    "Reserved", "Reserved", *["Grass"] * 20, *["Shallows"] * 5, "Shallow Water", "Shallow Water",
    "Normal Water", "Normal Water", "Deep Water",
)
_HAND_IS_WATER = {_game_root.AOC: _AOC_IS_WATER, _game_root.AOE1: _AOE1_IS_WATER, _game_root.HD: _HD_IS_WATER}
# A hand-classed .dat's ship restriction (the one its boats carry) and the families it must pass
# exactly, less the terrains no ship enters. The 1999 one passes beaches, the 1997 one does not.
_SHIP_FAMILIES = frozenset({"medium_water", "deep_water", "shallow_water", "shallows", "beach"})
_SHIP_CHECKS = {
    _game_root.AOC: (3, _SHIP_FAMILIES, frozenset()),
    _game_root.AOE1: (3, _SHIP_FAMILIES - {"beach"}, frozenset({15})),  # 15 Water Impass
    _game_root.HD: (3, _SHIP_FAMILIES, frozenset()),
}


def _raw_is_water(dataset: str, terrain_id: int, terrain) -> int:
    if dataset in _HAND_IS_WATER:
        return _HAND_IS_WATER[dataset].get(terrain_id, 0)
    return int(terrain.is_water) & 0xFF


def _ship_check(dataset: str, data, classes: dict[str, dict]) -> None:
    """Refuses a classic table whose water families differ from the .dat's own ship restriction."""
    restriction, families, impassable = _SHIP_CHECKS[dataset]
    row = data.terrain_restrictions[restriction].passable_buildable_dmg_multiplier
    passed = {terrain_id for terrain_id, value in enumerate(row) if value > 0}
    watery = {int(key) for key, entry in classes.items() if entry["family"] in families} - impassable
    if passed != watery:
        raise SystemExit(
            f"the {dataset} ship restriction passes {sorted(passed)} but the class table's ship terrains "
            f"are {sorted(watery)}: re-measure its hand is_water table"
        )


def _aoe1_name_check(terrains) -> None:
    """Refuses a 1997 .dat whose enabled terrains are not _AOE1_IS_WATER's ids and names."""
    names = {terrain_id: t.name for terrain_id, t in enumerate(terrains) if t.enabled}
    if names != dict(enumerate(_AOE1_TERRAIN_NAMES)):
        raise SystemExit(f"the 1997 terrains are {names}, not those _AOE1_IS_WATER was measured on")


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
        help="Where to write terrain_classes.json (default: descape/)",
    )
    _game_root.add_dataset_argument(parser)
    args = parser.parse_args()
    _game_root.set_lenient(args.lenient)

    dat_path = _game_root.dat_path(args.aoe2de_root, args.dataset, args.dat)
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = _game_root.parse_dat(dat_path)
    terrains = data.terrain_block.terrains
    if args.dataset == _game_root.AOE1:
        _aoe1_name_check(terrains)
    if args.dataset == _game_root.HD and tuple(t.name for t in terrains[68:100]) != _HD_SLOT_NAMES:
        raise SystemExit(f"HD terrains 68..99 are {[t.name for t in terrains[68:100]]}, not those _HD_IS_WATER was classed by")

    classes: dict[str, dict] = {}
    # (terrain id, report line)
    violations: list[tuple[int, str]] = []
    for i, t in enumerate(terrains):
        if not t.enabled:
            continue
        raw = _raw_is_water(args.dataset, i, t)
        try:
            family, climate = decode(raw)
        except ValueError as exc:
            violations.append((i, f"  id={i} name={t.name!r}: {exc}"))
            continue
        if family == "obsolete" and "OBSOLETE" not in (t.name or "").upper():
            violations.append((i, f"  id={i} name={t.name!r}: family bits 0 but not named OBSOLETE"))
            continue
        classes[str(i)] = {"raw": raw, "family": family, "climate": climate}

    if violations and _game_root.lenient():
        for terrain_id, line in violations:
            _game_root.refuse(
                f"is_water bitmask invariant: {line.strip()}", "terrain left out of the class table",
                kind="terrain_class", subject=terrain_id,
            )
    elif violations:
        print("The is_water bitmask invariant no longer holds:", file=sys.stderr)
        for _terrain_id, line in violations:
            print(line, file=sys.stderr)
        raise SystemExit(
            "Refusing to write: a game update has changed what this field means. "
            "Re-measure before touching descape/terrain_classes.py."
        )
    if args.dataset in _SHIP_CHECKS:
        _ship_check(args.dataset, data, classes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "terrain_classes.json"
    out = {
        "_comment": (
            "terrain_id -> {raw is_water byte, family, climate}. Pure factual "
            "classification extracted from the game's own data table via "
            "genieutils-py, not game asset content itself. family is one of "
            "medium_water/deep_water/shallow_water/shallows/beach/land/obsolete; "
            "climate is temperate/ice/snow. Regenerate with "
            "tools/gen_terrain_classes.py."
        ),
        "classes": classes,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for entry in classes.values():
        counts[entry["family"]] = counts.get(entry["family"], 0) + 1
    print(f"Wrote {len(classes)} entries to {out_path}")
    for family, count in sorted(counts.items()):
        print(f"  {family:<15} {count}")


if __name__ == "__main__":
    main()
