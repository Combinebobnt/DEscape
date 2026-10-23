#!/usr/bin/env python3
"""Writes build/garrison_flags/garrison_flags.aoe2scenario for checking GH
#203's garrison flags against the real AoE2:DE editor: does each building
DEscape draws a flag on show one in the game, at the same spot and in the
same style?

Through the real write path, from examples/blank_map.aoe2scenario (120x120,
grass, flat, no units): the Players fields go through OptionsEditModel, the
units through UnitEditModel.add() (an occupant the way Garrison Add puts it:
on its host's point, garrisoned_in_id set at construction), the water band
through batch_api.set_terrain(), then write_scenario(). The file is read back
and every host, occupant and link checked.

Three players with three architecture sets, all in the Castle Age so each
building draws its own civ's art: P1 Franks (west flags), P2 Japanese
(asia), P3 Persians (orie). Each gets a garrisoned and an empty twin of every
host below, the garrisoned one first. On land, one row per host type
(rows stepping +12 x, -12 y, which on screen is straight up) and one column
per player (stepping +13, +13), the twin one tile past the host's own span
along +x+y, which puts the pair side by side on screen. In the water band
(x 104-119), each player's Dock twins touch the shore at x 104 and its
Transport Ship twins sit at x 116, clear of the docks on screen.

Mosque is the control: it holds units in game but has no garrison graphic,
so neither twin draws a flag. The Barracks' and Mosque's occupants are not
units garrison.refusal() accepts (their tables list no unit type), so the
game may refuse or eject them; every other occupant is one it accepts.

    tools/gen_garrison_flags_scenario.py [--source <file>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.datasets.object_support import Civilization, StartingAge

from descape import object_catalog, render, unit_sprites
from descape.batch_api import set_terrain
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed, load_map_and_units, refresh_player_render_context
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "blank_map.aoe2scenario"
OUT = ROOT / "build" / "garrison_flags"
FILE_NAME = "garrison_flags.aoe2scenario"

PLAYERS = {1: "FRANKS", 2: "JAPANESE", 3: "PERSIANS"}
AGE = StartingAge.CASTLE_AGE.value
WATER = 1  # Water, the ordinary sea
WATER_X0 = 104
TRANSPORT_X = 116

# 1258, the editor's Battering Ram: 35 is the catalog-hidden legacy one.
TOWN_CENTER, CASTLE, BARRACKS, MOSQUE, WATCH_TOWER, RAM = 109, 82, 12, 655, 79, 1258
DOCK, TRANSPORT = 45, 545
VILLAGER, MILITIA, MONK, FISHING_SHIP = 83, 74, 125, 13
# host -> the occupant its garrisoned twin holds.
OCCUPANT = {
    TOWN_CENTER: VILLAGER, CASTLE: VILLAGER, BARRACKS: MILITIA, MOSQUE: MONK, WATCH_TOWER: VILLAGER,
    RAM: MILITIA, DOCK: FISHING_SHIP, TRANSPORT: VILLAGER,
}
# Bottom row to top on screen. The Mosque's minaret is the tallest art, so it goes last.
LAND_ROWS = (TOWN_CENTER, CASTLE, BARRACKS, WATCH_TOWER, RAM, MOSQUE)
ROW_STEP, COLUMN_STEP, TWIN_GAP = 12, 13, 1
LAND_ORIGIN = (1, 62)


@dataclass(frozen=True)
class Twin:
    """One placed host: its owner, const, low-corner tile and whether it holds a unit."""

    player: int
    const: int
    tile: tuple[int, int]
    garrisoned: bool

    def point(self) -> tuple[float, float]:
        """The anchor a placed building sits at: its low corner plus half its span per axis."""
        span_x, span_y = render.tile_span(self.const, render.NON_BUILDING_SPAN)
        return self.tile[0] + span_x / 2, self.tile[1] + span_y / 2


def layout() -> list[Twin]:
    """Every host twin, garrisoned first per pair, in placement order."""
    twins: list[Twin] = []
    for column, player in enumerate(PLAYERS):
        for row, const in enumerate(LAND_ROWS):
            x = LAND_ORIGIN[0] + ROW_STEP * row + COLUMN_STEP * column
            y = LAND_ORIGIN[1] - ROW_STEP * row + COLUMN_STEP * column
            step = render.tile_span(const, render.NON_BUILDING_SPAN)[0] + TWIN_GAP
            twins += [Twin(player, const, (x, y), True), Twin(player, const, (x + step, y + step), False)]
        # From y 40, so no land row's art reaches the docks on screen.
        y0 = 40 + 25 * column
        twins += [Twin(player, DOCK, (WATER_X0, y0), True), Twin(player, DOCK, (WATER_X0, y0 + 5), False)]
        twins += [
            Twin(player, TRANSPORT, (TRANSPORT_X, y0 + 1), True), Twin(player, TRANSPORT, (TRANSPORT_X, y0 + 6), False),
        ]
    return twins


def _check_layout(twins: list[Twin], width: int, height: int) -> None:
    """Every footprint on the map, on its own terrain side, and clear of every other by a tile."""
    claimed: dict[tuple[int, int], Twin] = {}
    for twin in twins:
        tiles = render.occupied_tiles_for(twin.const, *twin.point(), width, height)
        span = render.tile_span(twin.const, render.NON_BUILDING_SPAN)
        if tiles is None or len(tiles) != span[0] * span[1]:
            raise SystemExit(f"{twin} leaves the {width}x{height} map")
        water = twin.const in (DOCK, TRANSPORT)
        if any((x >= WATER_X0) != water for x, _y in tiles):
            raise SystemExit(f"{twin} is not wholly on {'water' if water else 'land'}")
        for x, y in tiles:
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    other = claimed.get((nx, ny))
                    if other is not None and other != twin:
                        raise SystemExit(f"{twin} touches {other}")
        claimed.update(dict.fromkeys(tiles, twin))


def build(source: Path = SOURCE):
    """(loaded, options model, unit model, [(twin, host unit, occupant unit or None)]), every
    edit applied in memory and the render context (team colours, building art) refreshed."""
    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    twins = layout()
    _check_layout(twins, mm.map_width, mm.map_height)
    for tile in mm.terrain:
        if tile.x >= WATER_X0:
            set_terrain(tile, WATER)

    options = OptionsEditModel(loaded)
    options.set_value("player_count", len(PLAYERS))
    pending: dict[str, dict[int, object]] = {"civilization": {}, "architecture": {}, "starting_age": {}}
    for player, civ in PLAYERS.items():
        value = Civilization[civ].value
        for field_id, raw in (("civilization", value), ("architecture", value), ("starting_age", AGE)):
            options.set_value(f"player:{field_id}:{player}", raw)
            pending[field_id][player] = raw
    refresh_player_render_context(loaded, pending)

    units = UnitEditModel(loaded)
    units.begin_unit_edit(sorted(PLAYERS))
    placed = []
    for twin in twins:
        host = units.add(twin.player, twin.const, *twin.point())
        occupant = None
        if twin.garrisoned:
            occupant = units.add(
                twin.player, OCCUPANT[twin.const], host.x, host.y, host.z, garrisoned_in_id=host.reference_id
            )
        placed.append((twin, host, occupant))
    units.commit_unit_edit("Place GH #203 garrison flag twins", EditHistory())
    return loaded, options, units, placed


def _flag_note(loaded, twin: Twin) -> str:
    entry = unit_sprites.resolve_entry(twin.const, *loaded.player_art[twin.player])
    flags = (entry or {}).get("garrison_flags", ())
    if not flags:
        return "no flag (no garrison graphic)"
    return ", ".join(f"{f['file_name'].removesuffix('_garrison_flag_x1')} at ({f['dx']}, {f['dy']})" for f in flags)


def readback(path: Path, placed, expected_art) -> int:
    """Every host and occupant found again in `path` with its link, and each player's building art."""
    reloaded = load_map_and_units(path)
    problems = []
    if tuple(reloaded.player_art[p] for p in PLAYERS) != expected_art:
        problems.append(f"building art {[reloaded.player_art[p] for p in PLAYERS]} != {list(expected_art)}")
    by_spot = {(p, u.unit_const, u.x, u.y, u.garrisoned_in_id): u for p, us in enumerate(reloaded.unit_manager.units) for u in us}
    print(f"Read back {path}:")
    print(f"  {'player':8s} {'host':16s} {'tile':>10s} {'holds':14s} flag DEscape draws while it holds a unit")
    for twin, host, occupant in placed:
        found = by_spot.get((twin.player, twin.const, host.x, host.y, -1))
        held = None
        if occupant is not None and found is not None:
            held = by_spot.get((twin.player, occupant.unit_const, host.x, host.y, found.reference_id))
        if found is None or (occupant is not None) != (held is not None):
            problems.append(f"{twin}: host {'missing' if found is None else 'found'}, occupant {held is not None}")
        holds = object_catalog.combined_object_name(occupant.unit_const) if occupant is not None else "(empty)"
        name = object_catalog.combined_object_name(twin.const)
        tile = str(twin.tile)
        print(f"  P{twin.player} {PLAYERS[twin.player][:5]:5s} {name[:16]:16s} {tile:>10s} {holds[:14]:14s} "
              f"{_flag_note(reloaded, twin)}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print(f"\n{len(placed)} hosts, {sum(o is not None for _t, _h, o in placed)} garrisoned, {len(problems)} problems.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The blank scenario to build on")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    args = parser.parse_args()
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    if not args.source.is_file():
        print(f"missing source: {args.source}", file=sys.stderr)
        return 1
    loaded, options, units, placed = build(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / FILE_NAME
    write_scenario(loaded, out, options=options, units=units, backup=False)
    print(f"wrote {out}\n")
    return readback(out, placed, tuple(loaded.player_art[p] for p in PLAYERS))


if __name__ == "__main__":
    raise SystemExit(main())
