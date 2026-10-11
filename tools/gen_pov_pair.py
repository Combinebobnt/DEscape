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

--fixture picks another scenario (default: York). Its file names then carry
the fixture's stem, so York's pair is never overwritten. On a scenario
version 1.41 fixture the "after" file also unticks Collide and correcting,
since those files' Map flags were misframed before
library_compat.fix_structure() and the editor is what settles their order.

Writes into build/pov_pair/, which is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import option_fields, player_fields
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario

# York: 220x220, scenario 1.55, six active players, and every civ on it ships
# with the base game, so a tester without any DLC can still open the pair.
# It already stores real views: P1 at (165, 105), P2..P6 all at (171, 78).
FIXTURE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"
OUT_DIR = ROOT / "build" / "pov_pair"

RESET_PLAYER = 2
# How far (in tiles, either axis) P1's new view must sit from every stored one.
CLEARANCE = 16
MAP_FLAGS = ("collide_and_correct", "villager_force_drop", "lock_coop_alliances")
UNTICK_ON_V141 = "collide_and_correct"


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


def _flags(loaded) -> dict[str, int]:
    specs = {s.field_id: s for s in option_fields.specs_for(loaded)}
    return {name: option_fields.current_value(loaded, specs[name]) for name in MAP_FLAGS if name in specs}


def p1_target(width: int, height: int, stored: dict[int, tuple[int, int]]) -> tuple[tuple[int, int], str]:
    """P1's new view: the first corner tile, near low-x/high-y first, that is
    in-map, has x != y and is CLEARANCE tiles clear of every stored view.
    On York (220x220) this is (12, 205)."""
    lo_x, hi_x = width // 18, width - 1 - width // 15
    lo_y, hi_y = height // 18, height - 1 - height // 15
    corners = (
        ((lo_x, hi_y), "the low-x/high-y corner"),
        ((hi_x, lo_y), "the high-x/low-y corner"),
        ((lo_x, lo_y + 3), "the low-x/low-y corner"),
        ((hi_x, hi_y - 3), "the high-x/high-y corner"),
    )
    for (x, y), where in corners:
        clear = all(max(abs(x - sx), abs(y - sy)) >= CLEARANCE for sx, sy in stored.values())
        if x != y and 0 <= x < width and 0 <= y < height and clear:
            return (x, y), where
    raise SystemExit("no corner tile is clear of every stored view")


def _names(fixture: Path) -> tuple[str, str]:
    if fixture.resolve() == FIXTURE.resolve():
        return "before_york.aoe2scenario", "after_p1_corner_p2_reset.aoe2scenario"
    stem = fixture.stem.replace(" ", "_")
    return f"before_{stem}.aoe2scenario", f"after_p1_corner_p2_reset_{stem}.aoe2scenario"


def _print_york_checks(p1_view: tuple[int, int]) -> None:
    print("What to check in the AoE2:DE editor, opening the 'after' file:")
    print(f"  1. Players tab, P1 -> Point of View -> Go to View: it lands on"
          f" {p1_view}, near the low-x/high-y corner, not on the town.")
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


def _print_checks(
    stored: dict[int, tuple[int, int]], p1_view: tuple[int, int], where: str,
    flags_before: dict[str, int], flags_after: dict[str, int],
) -> None:
    def ticked(value: int) -> str:
        return "ticked" if value else "unticked"

    print("What to check in the AoE2:DE editor, opening the 'before' file first:")
    print(f"  1. Players tab, P1 -> Point of View -> Go to View: it lands on {stored[1]}.")
    if flags_before:
        print("  2. Map tab:", ", ".join(f"{name} {ticked(v)}" for name, v in flags_before.items()) + ".")
        print("     A different pattern here means the three flags sit in another order.")
    print("Then the 'after' file:")
    print(f"  3. P1 -> Go to View lands on {p1_view}, near {where}.")
    print(f"  4. P{RESET_PLAYER}: Go to View on an unset (-1, -1) view, and say what happens.")
    others = ", ".join(f"P{p} {stored[p]}" for p in range(3, 9))
    print(f"  5. P3..P8 unchanged: {others}.")
    if flags_after:
        changed = [f"{name} now {ticked(v)}" for name, v in flags_after.items() if v != flags_before.get(name)]
        same = [f"{name} still {ticked(v)}" for name, v in flags_after.items() if v == flags_before.get(name)]
        print("  6. Map tab:", ", ".join(changed + same) + ".")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", type=Path, default=FIXTURE, help="scenario to pair (default: York)")
    args = parser.parse_args(argv)
    fixture: Path = args.fixture
    if not fixture.is_file():
        print(f"missing fixture: {fixture}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before_name, after_name = _names(fixture)

    before_path = OUT_DIR / before_name
    scenario = load_map_and_units(fixture)
    stored = _views(scenario)
    if not stored:
        print("fixture stores no Point of View array -- nothing to write", file=sys.stderr)
        return 1
    flags_before = _flags(scenario)
    mm = scenario.map_manager
    p1_view, where = p1_target(mm.map_width, mm.map_height, stored)
    v141 = scenario.scenario_version == "1.41"
    write_scenario(scenario, before_path, backup=False)
    print(f"wrote {before_path}")

    scenario = load_map_and_units(fixture)
    model = OptionsEditModel(scenario)
    unset = player_fields.POV_UNSET
    for field, value in (
        (player_fields.POV_X_FIELD, p1_view[0]),
        (player_fields.POV_Y_FIELD, p1_view[1]),
    ):
        model.set_value(player_fields.player_field_id(field, 1), value)
    for field in (player_fields.POV_X_FIELD, player_fields.POV_Y_FIELD):
        model.set_value(player_fields.player_field_id(field, RESET_PLAYER), unset)
    if v141:
        model.set_value(UNTICK_ON_V141, 0)

    after_path = OUT_DIR / after_name
    write_scenario(scenario, after_path, options=model, backup=False)
    print(f"wrote {after_path}")

    after = load_map_and_units(after_path)
    reloaded = _views(after)
    flags_after = _flags(after) if v141 else {}
    print()
    print("Read back from the written file (GAIA is slot 0):")
    for player_id, view in reloaded.items():
        was = stored[player_id]
        mark = "" if view == was else f"   <- was {was}"
        print(f"  P{player_id}: {view}{mark}")
    for name, value in flags_after.items():
        was = flags_before.get(name)
        mark = "" if value == was else f"   <- was {was}"
        print(f"  {name}: {value}{mark}")
    print()
    if fixture.resolve() == FIXTURE.resolve():
        _print_york_checks(p1_view)
    else:
        _print_checks(stored, p1_view, where, flags_before if v141 else {}, flags_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
