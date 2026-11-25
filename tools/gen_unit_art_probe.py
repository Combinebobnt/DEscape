#!/usr/bin/env python3
"""Writes the probe scenario for TASK-140's in-game check: does a unit's art
follow its owner's Civilization or its Architecture, per unit group, in the
AoE2:DE Scenario Editor and in Test (play)?

Written through the real write path: examples/blank_map.aoe2scenario
(120x120 grass, 8 player slots) through batch_api.load(), the player count
and each player's civilization and architecture set through
OptionsEditModel, a water strip painted with batch_api.set_terrain(), the
units placed with UnitEditModel.add() and the militia put in its ram with
set_garrisoned_in(), then write_scenario() to
build/task140_probe/task140_unit_art_probe.aoe2scenario, read back.

    Player  Civilization  Architecture  Separates
    P1      Britons       Chinese       west vs asia: sails, monk, king, cart, merchant, ram flag
    P2      Chinese       Britons       the same, reversed
    P3      Athenians     Britons       antiquity lines and priestess vs west
    P4      Britons       Athenians     the same, reversed
    P5      Britons       Britons       control

Each player gets one row along +x at its own y: a House (buildings follow
Architecture, the anchor), Villager, Archer, Knight, Monk, King, Trade
Cart, Merchant and a Battering Ram holding a Militia (its garrison flag),
then a Galley, a Transport Ship and a Fishing Ship on the water strip
(x 64, 70, 76). P5 is the control: P1
and P4 share its civilization and differ only in architecture, P3 shares its
architecture and differs only in civilization. A unit that looks unlike
P5's on P1 or P4 follows Architecture; one that looks unlike it on P3
follows Civilization. P2 repeats P1 reversed, as a cross-check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.datasets.object_support import Civilization

from descape import batch_api, garrison, object_catalog, player_fields, render_units
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "blank_map.aoe2scenario"
OUT = ROOT / "build" / "task140_probe" / "task140_unit_art_probe.aoe2scenario"

HOUSE, VILLAGER, ARCHER, KNIGHT, MONK, KING = 70, 83, 4, 38, 125, 434
TRADE_CART, MERCHANT, BATTERING_RAM, MILITIA = 128, 1572, 1258, 74
GALLEY, TRANSPORT_SHIP, FISHING_SHIP = 539, 545, 13
LAND_ROW = (HOUSE, VILLAGER, ARCHER, KNIGHT, MONK, KING, TRADE_CART, MERCHANT, BATTERING_RAM)
# The Transport Ship's antiquity hull differs, not just its sails.
WATER_ROW = (GALLEY, TRANSPORT_SHIP, FISHING_SHIP)
MEDIUM_WATER = 23

# player -> (civilization, architecture), Civilization enum names.
PLAYERS = {
    1: ("BRITONS", "CHINESE"),
    2: ("CHINESE", "BRITONS"),
    3: ("ATHENIANS", "BRITONS"),
    4: ("BRITONS", "ATHENIANS"),
    5: ("BRITONS", "BRITONS"),
}
ROW_Y0, ROW_GAP = 12, 20  # player p's row is at y = ROW_Y0 + (p - 1) * ROW_GAP
LAND_X0, LAND_GAP = 8, 4
WATER_X = (60, 80)  # the painted strip, every y
WATER_X0, WATER_GAP = 64, 6


def _token(name: str) -> str:
    return Civilization[name].value


def _name(const: int) -> str:
    return object_catalog.combined_object_name(const)


def layout() -> list[tuple[int, int, int, int]]:
    """[(player, const, tile x, tile y)], the land row then the water row."""
    placed = []
    for player in PLAYERS:
        y = ROW_Y0 + (player - 1) * ROW_GAP
        placed += [(player, const, LAND_X0 + i * LAND_GAP, y) for i, const in enumerate(LAND_ROW)]
        placed += [(player, const, WATER_X0 + i * WATER_GAP, y) for i, const in enumerate(WATER_ROW)]
    return placed


def write_probe(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    loaded = batch_api.load(source)  # activates the file's game dataset (AGENTS.md)
    mm = loaded.map_manager
    if (mm.map_width, mm.map_height) != (120, 120) or any(loaded.unit_manager.units):
        raise SystemExit(f"{source.name} is not the empty 120x120 map this layout assumes")
    options = OptionsEditModel(loaded)
    options.set_value(player_fields.PLAYER_COUNT_FIELD_ID, len(PLAYERS))
    for player, (civ, arch) in PLAYERS.items():
        options.set_value(player_fields.player_field_id("civilization", player), _token(civ))
        options.set_value(player_fields.player_field_id("architecture", player), _token(arch))
    for tile in mm.terrain:
        if WATER_X[0] <= tile.x <= WATER_X[1]:
            batch_api.set_terrain(tile, MEDIUM_WATER)
    units = UnitEditModel(loaded)
    units.begin_unit_edit(sorted(PLAYERS))
    for player, const, tx, ty in layout():
        x, y = render_units.span_anchor(tx, ty, *tile_span(const, render_units.NON_BUILDING_SPAN))
        unit = units.add(player, const, x, y)
        if const == BATTERING_RAM:
            refusal = garrison.refusal(BATTERING_RAM, _name(BATTERING_RAM), 0, [MILITIA])
            if refusal is not None:
                raise SystemExit(refusal)
            units.set_garrisoned_in(units.add(player, MILITIA, x, y), unit.reference_id)
    units.commit_unit_edit("Place the TASK-140 unit art probe", EditHistory())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_scenario(loaded, out, options=options, units=units, backup=False)
    print(f"wrote {out}\n")
    return readback(out)


def readback(path: Path) -> int:
    """Each player's stored civilization and architecture, units and garrison, against the layout."""
    back = load_map_and_units(path)
    specs = {s.field_id: s for s in player_fields.specs_for(back)}
    failures = []
    for player, (civ, arch) in PLAYERS.items():
        got = tuple(player_fields.current_value(back, specs[f], player) for f in ("civilization", "architecture"))
        if got != (_token(civ), _token(arch)):
            failures.append(f"P{player}: stored {got}, wanted {(_token(civ), _token(arch))}")
        placed = back.unit_manager.units[player]
        wanted = sorted([*LAND_ROW, *WATER_ROW, MILITIA])
        if sorted(u.unit_const for u in placed) != wanted:
            failures.append(f"P{player}: units {sorted(u.unit_const for u in placed)}, wanted {wanted}")
        rams = {u.reference_id for u in placed if u.unit_const == BATTERING_RAM}
        if [u.garrisoned_in_id in rams for u in placed if u.unit_const == MILITIA] != [True]:
            failures.append(f"P{player}: the militia is not inside the ram")
        print(f"  P{player}: civilization {civ.title()}, architecture {arch.title()}, row y {ROW_Y0 + (player - 1) * ROW_GAP}")
    print("  land row, x " + ", ".join(f"{LAND_X0 + i * LAND_GAP} {_name(c)}" for i, c in enumerate(LAND_ROW)))
    print("  water row, x " + ", ".join(f"{WATER_X0 + i * WATER_GAP} {_name(c)}" for i, c in enumerate(WATER_ROW)))
    if failures:
        print("read-back FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nread back {path.name}: every player's civ, architecture, units and garrison as written.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The empty map to build on")
    parser.add_argument("--out", type=Path, default=OUT, help="Output scenario path")
    args = parser.parse_args()
    return write_probe(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
