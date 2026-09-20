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

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its terrain table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    terrains = data.terrain_block.terrains

    classes: dict[str, dict] = {}
    violations: list[str] = []
    for i, t in enumerate(terrains):
        if not t.enabled:
            continue
        raw = int(t.is_water) & 0xFF
        try:
            family, climate = decode(raw)
        except ValueError as exc:
            violations.append(f"  id={i} name={t.name!r}: {exc}")
            continue
        if family == "obsolete" and "OBSOLETE" not in (t.name or "").upper():
            violations.append(
                f"  id={i} name={t.name!r}: family bits 0 but not named OBSOLETE"
            )
            continue
        classes[str(i)] = {"raw": raw, "family": family, "climate": climate}

    if violations:
        print("The is_water bitmask invariant no longer holds:", file=sys.stderr)
        for line in violations:
            print(line, file=sys.stderr)
        raise SystemExit(
            "Refusing to write: a game update has changed what this field means. "
            "Re-measure before touching descape/terrain_classes.py."
        )

    out_path = Path(__file__).resolve().parent.parent / "descape" / "terrain_classes.json"
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
