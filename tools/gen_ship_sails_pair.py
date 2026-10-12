#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for checking GH #20's ship sails
and butterflies against the real AoE2:DE editor: do DEscape's sails sit where
the game's do, in the owner's colour, and are the butterflies the game's size
and height?

Both files go through the real write path: "before" is a zero-edit
write_scenario(), "after" adds the units with UnitEditModel.add() then
write_scenario(), the code Place Unit runs. DEscape never writes a player's
civ, so the source decides it: the default's civs are all base-game (a tester
needs no DLC).

**Only players whose civ draws the Gaia table's ships get any.** DEscape
reads every unit's graphic from the .dat's civ 0, and a ship's sail set is
per civ. The civs whose shell equals civ 0's on all 20 sailing-ship consts are
British, French, Celts, Burgundians and Saxons (.dat civ names, indexes 1, 2,
13, 36, 60; measured 2026-10-12). Any other civ's ships carry different sails
in the game, which is a known gap and not what this pair tests.

The ships go in the largest water rectangle with no unit on it, one line per
qualifying player running along +y, the lines 2 tiles apart along +x:

    Galleon, War Galley, Galley, Transport Ship, Fire Ship, Trade Cog,
    Heavy Demolition Ship, Stoertebeker, then a Fishing Ship.

Those eight sailing ships use all 15 sail graphics between them. The Fishing
Ship is the control (one graphic, unchanged) and ends every line, and the gap
after the first line is 2 tiles wider, so a mirrored or transposed write reads
back visibly wrong. Each line faces its own direction, a quarter turn and a
sixteenth on from the last, so the pair covers several of the 16 facings.

The three butterflies (GAIA) go on the nearest flat, empty, dry strip of 8
tiles to the ships, at tiles 1, 3 and 6 of it: Butterflies 1, 2, 3.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import civ_art, object_catalog, place_rotation, player_fields, terrain_classes
from descape.edit_history import EditHistory
from descape.render import unit_occupied_tiles
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "2_Joan_coop_4_v0_13.aoe2scenario"
OUT = ROOT / "build" / "ship_sails_pair"
GAIA = 0

# .dat civ index -> its name there. Shell-equal to civ 0 on every sailing ship.
GAIA_SAIL_CIVS = {1: "British", 2: "French", 13: "Celts", 36: "Burgundians", 60: "Saxons"}

GALLEON, WAR_GALLEY, GALLEY, TRANSPORT, FIRE_SHIP = 442, 21, 539, 545, 529
TRADE_COG, HEAVY_DEMO, STOERTEBEKER, FISHING_SHIP = 17, 528, 114, 13
# One line's ships in order. The control comes last.
LINE = (GALLEON, WAR_GALLEY, GALLEY, TRANSPORT, FIRE_SHIP, TRADE_COG, HEAVY_DEMO, STOERTEBEKER, FISHING_SHIP)
BUTTERFLIES = (1608, 1609, 1610)
# unit_sprites.TEAM_COLORS' order, named for the printed table.
TEAM_COLOUR_NAMES = ("white", "blue", "red", "green", "yellow", "cyan", "purple", "grey", "orange")

SHIP_GAP = 2      # tiles between ships along a line, and between lines
FIRST_LINE_GAP = 2  # extra tiles after the first line
BUTTERFLY_STRIP = 8
BUTTERFLY_OFFSETS = (1, 3, 6)
LAND_MARGIN = 1


def ship_players(loaded) -> list[tuple[int, int]]:
    """(player_id, .dat civ index) for every player whose ships DEscape draws as the game does."""
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    if "civilization" not in specs:
        return []
    found = []
    for player_id in range(1, civ_art.PLAYER_COUNT):
        civ = civ_art.civ_index(player_fields.current_value(loaded, specs["civilization"], player_id))
        if civ in GAIA_SAIL_CIVS:
            found.append((player_id, civ))
    return found


def _free_tiles(loaded):
    """(water, land, elevation): unit-free tile sets and every tile's elevation."""
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    taken: set[tuple[int, int]] = set()
    for units in loaded.unit_manager.units:
        for u in units:
            taken.update(unit_occupied_tiles(u, w, h) or ())
    water, land, elevation = set(), set(), {}
    for tile in mm.terrain:
        key = (tile.x, tile.y)
        elevation[key] = tile.elevation
        if key in taken:
            continue
        if terrain_classes.is_beach_family(tile.terrain_id):
            continue
        (water if terrain_classes.is_water_family(tile.terrain_id) else land).add(key)
    return water, land, elevation


def largest_rect(free: set[tuple[int, int]], w: int, h: int) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the largest all-free rectangle, by the histogram walk."""
    best = (0, (0, 0, 0, 0))
    heights = [0] * w
    for y in range(h):
        heights = [heights[x] + 1 if (x, y) in free else 0 for x in range(w)]
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            current = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= current:
                sx, sh = stack.pop()
                if sh * (x - sx) > best[0]:
                    best = (sh * (x - sx), (sx, y - sh + 1, x - sx, sh))
                start = sx
            stack.append((start, current))
    return best[1]


def ship_layout(rect: tuple[int, int, int, int], players: list[tuple[int, int]]):
    """[(label, player, const, x, y, rotation)]: one line of LINE per player
    along +y from the rectangle's low corner, lines stepping along +x."""
    x0, y0, width, height = rect
    need_w = (len(players) - 1) * SHIP_GAP + FIRST_LINE_GAP + 1
    need_h = (len(LINE) - 1) * SHIP_GAP + 1
    if width < need_w or height < need_h:
        raise SystemExit(
            f"the largest unit-free water rectangle is {width}x{height} at ({x0}, {y0}); "
            f"{len(players)} lines of {len(LINE)} ships need {need_w}x{need_h}"
        )
    placed = []
    for line, (player, _civ) in enumerate(players):
        tx = x0 + line * SHIP_GAP + (FIRST_LINE_GAP if line else 0)
        facing = place_rotation.PlaceRotation(facing=line * 5 * math.tau / 16)
        for slot, const in enumerate(LINE):
            ty = y0 + slot * SHIP_GAP
            rotation = place_rotation.rotation_for(facing, const, player)
            label = f"P{player} line {line + 1} slot {slot + 1} {_name(const)}"
            placed.append((label, player, const, tx + 0.5, ty + 0.5, rotation))
    return placed


def butterfly_layout(land: set[tuple[int, int]], elevation: dict, near: tuple[int, int]):
    """[(label, GAIA, const, x, y, 0.0)] on the flat free land strip (along +x,
    with LAND_MARGIN free tiles round it) nearest `near`."""
    best = None
    for x, y in land:
        strip = [(x + dx, y + dy) for dx in range(-LAND_MARGIN, BUTTERFLY_STRIP + LAND_MARGIN)
                 for dy in range(-LAND_MARGIN, LAND_MARGIN + 1)]
        if any(t not in land for t in strip) or len({elevation[t] for t in strip}) != 1:
            continue
        key = ((x - near[0]) ** 2 + (y - near[1]) ** 2, x, y)
        if best is None or key < best:
            best = key
    if best is None:
        raise SystemExit(f"no flat empty land strip of {BUTTERFLY_STRIP} tiles found")
    _distance, x0, y0 = best
    return [
        (f"Butterflies {number} at strip tile {offset}", GAIA, const, x0 + offset + 0.5, y0 + 0.5, 0.0)
        for number, (const, offset) in enumerate(zip(BUTTERFLIES, BUTTERFLY_OFFSETS, strict=True), 1)
    ]


def _name(unit_const: int) -> str:
    return object_catalog.combined_object_name(unit_const)


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    loaded = load_map_and_units(source)
    players = ship_players(loaded)
    if not players:
        raise SystemExit(
            f"no player of {source.name} has a civ whose ships DEscape draws as the game does "
            f"({', '.join(GAIA_SAIL_CIVS.values())})"
        )
    mm = loaded.map_manager
    water, land, elevation = _free_tiles(loaded)
    rect = largest_rect(water, mm.map_width, mm.map_height)
    ships = ship_layout(rect, players)
    butterflies = butterfly_layout(land, elevation, (rect[0], rect[1]))

    out.mkdir(parents=True, exist_ok=True)
    before, after = out / "before.aoe2scenario", out / "after.aoe2scenario"
    write_scenario(loaded, before, backup=False)
    model = UnitEditModel(loaded)
    model.begin_unit_edit(sorted({player for _l, player, *_rest in ships + butterflies}))
    for _label, player, const, x, y, rotation in ships + butterflies:
        model.add(player, const, x, y, rotation=rotation)
    model.commit_unit_edit("Place GH #20 ships and butterflies", EditHistory())
    write_scenario(loaded, after, backup=False, units=model)

    print(f"wrote {before}\nwrote {after}\n")
    print(f"Source {source.name}, {mm.map_width}x{mm.map_height}.")
    print(f"Ships: water rectangle x {rect[0]}..{rect[0] + rect[2] - 1}, y {rect[1]}..{rect[1] + rect[3] - 1}.")
    for line, (player, civ) in enumerate(players, 1):
        colour = TEAM_COLOUR_NAMES[loaded.team_indices[player]]
        print(f"  line {line}: player {player} ({colour}), civ {civ} {GAIA_SAIL_CIVS[civ]}")
    skipped = [p for p in range(1, civ_art.PLAYER_COUNT) if p not in {player for player, _c in players}]
    print(f"  players given no ships (their civ's sails differ from the Gaia table's): {skipped}\n")
    return readback(after, ships + butterflies)


def readback(path: Path, placed) -> int:
    """Every placed unit found again in `path` at its tile, with its stored rotation."""
    units = load_map_and_units(path).unit_manager.units
    missing = 0
    print(f"Read back {path}:")
    print(f"  {'what':52s} {'const':>5s} {'tile':>10s} {'rotation':>9s}")
    for label, player, const, x, y, rotation in placed:
        hits = [u for u in units[player] if u.unit_const == const and (u.x, u.y) == (x, y)]
        if len(hits) != 1 or abs(hits[0].rotation - rotation) > 1e-5:
            missing += 1
            print(f"  {label:52s} {const:5d} {f'({int(x)}, {int(y)})':>10s}   MISSING or changed ({len(hits)} found)")
            continue
        print(f"  {label:52s} {const:5d} {f'({int(x)}, {int(y)})':>10s} {hits[0].rotation:9.4f}")
    print(f"\n{len(placed)} placed, {missing} missing.")
    return 1 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The scenario to add the units to")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    args = parser.parse_args()
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
