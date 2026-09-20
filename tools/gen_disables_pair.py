#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming GH #57's disable
lists in the real AoE2:DE editor -- the half of testing nothing in this
session can drive.

Deliberately asymmetric: one building disabled for P2 and one technology
for P5, every other player and category untouched. A
symmetric edit would turn the check into "did anything change"; this one makes
it "does *this player's* list, and only this one, read back right", which is
what would catch a write that silently applied to every player or to the
wrong category's block.

Both files go through the real write path (OptionsEditModel +
scenario_write.write_scenario(), the same code the GUI uses), never a byte
patch.

Writes into build/disables_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import disables_fields, object_catalog
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario

FIXTURE = ROOT / "tests" / "fixtures" / "real_blank_240x240.aoe2scenario"
OUT_DIR = ROOT / "build" / "disables_pair"

# Town Center (109) for P2, Loom (22) for P5. Both are unmistakable in the
# in-game lists and both are things a player would notice losing.
BUILDING_ID = 109
BUILDING_PLAYER = 2
TECH_ID = 22
TECH_PLAYER = 5


def main() -> int:
    if not FIXTURE.is_file():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_path = OUT_DIR / "before_no_disables.aoe2scenario"
    scenario = load_map_and_units(FIXTURE)
    write_scenario(scenario, before_path, backup=False)
    print(f"wrote {before_path}")

    scenario = load_map_and_units(FIXTURE)
    if not disables_fields.verify_disables_block(scenario):
        print("fixture failed the disables gate -- nothing to write", file=sys.stderr)
        return 1
    model = OptionsEditModel(scenario)
    model.set_value(
        disables_fields.disables_field_id("buildings", BUILDING_PLAYER), (BUILDING_ID,)
    )
    model.set_value(disables_fields.disables_field_id("techs", TECH_PLAYER), (TECH_ID,))

    after_path = OUT_DIR / "after_p2_building_p5_tech.aoe2scenario"
    write_scenario(scenario, after_path, options=model, backup=False)
    print(f"wrote {after_path}")

    reloaded = load_map_and_units(after_path)
    building = object_catalog.object_name(BUILDING_ID)
    tech = object_catalog.tech_name(TECH_ID)
    print()
    print("Read back from the written file:")
    for category, player in (("buildings", BUILDING_PLAYER), ("techs", TECH_PLAYER)):
        print(f"  P{player} {category}: {disables_fields.current_ids(reloaded, category, player)}")
    print(f"  gate re-verifies: {disables_fields.verify_disables_block(reloaded)}")
    print()
    print("What to check in the AoE2:DE editor, opening the 'after' file:")
    print("  1. Players tab -> Disable Objects -> Buildings, with P2 selected:")
    print(f"     '{building}' ({BUILDING_ID}) is in the Disabled List.")
    print("  2. Same control with P1, P3..P8 selected: the Buildings Disabled")
    print("     List is EMPTY. This is the asymmetry check -- a write that")
    print("     fanned out to every player fails here, not in step 1.")
    print("  3. Disable Objects -> Techs, with P5 selected:")
    print(f"     '{tech}' ({TECH_ID}) is in the Disabled List.")
    print("  4. Disable Objects -> Units, every player: EMPTY. A block-offset")
    print("     bug would land the edit in the wrong category here.")
    print("  5. Save the file from the in-game editor, reopen it, and confirm")
    print("     both lists survived -- that the game accepts the region, not")
    print("     just that it displays it.")
    print("  6. The 'before' file has every list empty, for every player.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
