#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming GH #22's Point of
View write in the real AoE2:DE editor -- the half of testing nothing in this
session can drive.

Deliberately asymmetric, twice over: P1's view moves to a corner tile whose
x and y differ (so a swapped-axes write is visible, not merely "something
changed"), and P2's is reset to the unset (-1, -1) pair while every other
player keeps the view the file already stored. A same-value edit, or one
applied to every player, fails to show either.

Both files go through the real write path (OptionsEditModel +
scenario_write.write_scenario(), the same code the GUI uses), never a byte
patch. The GAIA slot is left verbatim: this project never exposes it, and
whether the game's own Set View also rewrites it is one of the questions
this pair exists to answer.

Writes into build/pov_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import player_fields
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario

# York: 220x220, scenario 1.55, six active players, and every civ on it ships
# with the base game, so a tester without any DLC can still open the pair.
# It already stores real views: P1 at (165, 105), P2..P6 all at (171, 78).
FIXTURE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"
OUT_DIR = ROOT / "build" / "pov_pair"

# Near the low-x/high-y corner, well clear of both stored views, and x != y.
P1_VIEW = (12, 205)
RESET_PLAYER = 2


def _views(loaded) -> dict[int, tuple[int, int]]:
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    x_spec = specs.get(player_fields.POV_X_FIELD)
    y_spec = specs.get(player_fields.POV_Y_FIELD)
    if x_spec is None or y_spec is None:
        return {}
    return {
        player_id: (
            player_fields.current_value(loaded, x_spec, player_id),
            player_fields.current_value(loaded, y_spec, player_id),
        )
        for player_id in range(9)  # slot 0 is GAIA
    }


def main() -> int:
    if not FIXTURE.is_file():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_path = OUT_DIR / "before_york.aoe2scenario"
    scenario = load_map_and_units(FIXTURE)
    stored = _views(scenario)
    if not stored:
        print("fixture stores no Point of View array -- nothing to write", file=sys.stderr)
        return 1
    write_scenario(scenario, before_path, backup=False)
    print(f"wrote {before_path}")

    scenario = load_map_and_units(FIXTURE)
    model = OptionsEditModel(scenario)
    unset = player_fields.POV_UNSET
    for field, value in (
        (player_fields.POV_X_FIELD, P1_VIEW[0]),
        (player_fields.POV_Y_FIELD, P1_VIEW[1]),
    ):
        model.set_value(player_fields.player_field_id(field, 1), value)
    for field in (player_fields.POV_X_FIELD, player_fields.POV_Y_FIELD):
        model.set_value(player_fields.player_field_id(field, RESET_PLAYER), unset)

    after_path = OUT_DIR / "after_p1_corner_p2_reset.aoe2scenario"
    write_scenario(scenario, after_path, options=model, backup=False)
    print(f"wrote {after_path}")

    reloaded = _views(load_map_and_units(after_path))
    print()
    print("Read back from the written file (GAIA is slot 0):")
    for player_id, view in reloaded.items():
        was = stored[player_id]
        mark = "" if view == was else f"   <- was {was}"
        print(f"  P{player_id}: {view}{mark}")
    print()
    print("What to check in the AoE2:DE editor, opening the 'after' file:")
    print(f"  1. Players tab, P1 -> Point of View -> Go to View: it lands on"
          f" {P1_VIEW}, near the low-x/high-y corner, not on the town.")
    print("  2. Test-play as P1: the camera starts at that corner. This is")
    print("     what confirms the game reads Map.initial_player_views rather")
    print("     than the legacy player_data_3.initial_camera_* pair.")
    print(f"  3. P{RESET_PLAYER}: what does the editor show for a view of (-1, -1)?"
          " Go to View,")
    print("     and say what happens -- nothing, the map centre, or an error.")
    print("  4. P3..P6: still the view they had before (171, 78), untouched.")
    print("     A write that fanned out to every player fails here.")
    print("  5. Set P4's view from the in-game editor, save, and reopen the")
    print("     file in DEscape: does P4's marker move to where you set it,")
    print("     and did the game ALSO rewrite the GAIA slot (P0 above)?")
    print("  6. The 'before' file has P1 at (165, 105) and P2..P6 at (171, 78).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
