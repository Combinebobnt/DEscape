#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for checking GH #39's hero glow
against the real AoE2:DE editor: which of six units does the game ring in
gold, and does DEscape ring the same ones. Both files go through the real
write path (UnitEditModel + scenario_write.write_scenario(), the code the GUI
uses), never a hand-crafted byte patch.

The after file adds, for player 1, in one row two tiles apart (left to right):
HCHARL 165, HKHAN 1275, HLUBU 2032 (Three Kingdoms DLC), HWOLF 700, KINGX 434
and ARCHR 4. The first four are in DEscape's hero set, the last two are not.
HCHARL stands in for the plan's HLEIF 106, which is a longboat and would sit
on land here.
HWOLF settles the bit-128 question in tools/gen_unit_render_data.py.

Writes into build/hero_glow_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import terrain_classes
from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.terrain_palette import HERO_GLOW_CONSTS
from descape.unit_model import UnitEditModel

OUT = ROOT / "build" / "hero_glow_pair"
# Base-game civs only (GOTCHAS: a DLC civ blocks a tester without the DLC).
SOURCE = "atilla_1_scn_resaved.aoe2scenario"
PLAYER = 1
ROW = (165, 1275, 2032, 700, 434, 4)
STEP = 2


def _clear_row(loaded, length: int) -> tuple[int, int]:
    """The first row start, scanning out from the map centre, on land and with
    no unit within a tile of the row, so nothing hides or crowds the six."""
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    taken = {
        (int(u.x) + dx, int(u.y) + dy)
        for units in loaded.unit_manager.units for u in units
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    }
    for tile in mm.terrain:
        if terrain_classes.is_water_family(tile.terrain_id) or terrain_classes.is_beach_family(tile.terrain_id):
            taken.add((tile.x, tile.y))
    cx, cy = w // 2 - length // 2, h // 2
    for radius in range(max(w, h)):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                row = [(x + i, y) for i in range(length)]
                if all(0 <= tx < w and 0 <= ty < h and (tx, ty) not in taken for tx, ty in row):
                    return x, y
    raise SystemExit("no clear row found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / "examples" / SOURCE
    loaded = load_map_and_units(src)
    before = OUT / "hero_glow_before.aoe2scenario"
    after = OUT / "hero_glow_after.aoe2scenario"
    write_scenario(loaded, before)

    x0, y0 = _clear_row(loaded, STEP * len(ROW))
    model = UnitEditModel(loaded)
    model.begin_unit_edit([PLAYER])
    placed = []
    for i, const in enumerate(ROW):
        x, y = x0 + i * STEP + 0.5, y0 + 0.5
        model.add(PLAYER, const, x, y)
        placed.append((const, x, y))
    model.commit_unit_edit("Place hero glow row", EditHistory())
    write_scenario(loaded, after, units=model)

    # Read the written file back: the round trip is what the game will see.
    reread = load_map_and_units(after)
    back = {(u.unit_const, u.x, u.y) for u in reread.unit_manager.units[PLAYER]}
    assert all(p in back for p in placed), "a placed unit did not read back"
    print(f"{SOURCE}, player {PLAYER}, row from tile ({x0}, {y0}) along +x, {STEP} tiles apart:")
    for const, x, y in placed:
        print(f"  {const:5d} at ({x}, {y}): {'ringed' if const in HERO_GLOW_CONSTS else 'no ring'} in DEscape")
    print(f"wrote {before}\nwrote {after}")


if __name__ == "__main__":
    main()
