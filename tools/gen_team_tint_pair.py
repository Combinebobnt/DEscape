#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for measuring how AoE2:DE shades
player colour on unit sprites (GH #82), and a layout.json beside it that a
screenshot reader can project the units from.

Both files go through the real write path: "before" is a zero-edit
write_scenario(), "after" sets the players through OptionsEditModel (the code
Players mode saves with) and adds the units with UnitEditModel.add() (the code
Place Unit runs), then one write_scenario().

The after file:

- Players 1-8 take ColorIds 0-7 in slot order (blue, red, green, yellow,
  aqua, purple, grey, orange), the player count is 8, and every player's
  civilization and architecture is Britons. Every civ outside the Chronicles
  set draws Knight and Militia with the .dat's civ 0 graphics, which is what
  DEscape draws, and one civ gives every row the same House.
- Nine rows, GAIA first then players 1-8, each a Knight, a Militia and a
  House, all facing FACING. Rows step ROW_STEP tiles towards the screen's
  bottom (x - ROW_STEP, y + ROW_STEP); the three units of a row step
  COL_STEP tiles towards the screen's right (x + COL_STEP, y + COL_STEP),
  so a row reads left to right on screen and no two sprites overlap.
- The rows fall into two shots, A (GAIA and players 1-4) and B (players
  5-8), with an extra row step between them so each half is framed on its
  own at full zoom.

The source must hold a flat, dry, unit-free area for the block; the default,
the blank template, is a single terrain at one elevation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import civ_art, object_catalog, place_rotation, player_fields, terrain_classes
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.render_units import unit_occupied_tiles
from descape.scenario_io import BLANK_TEMPLATE_PATH, is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = BLANK_TEMPLATE_PATH
OUT = ROOT / "build" / "team_tint_pair"
GAIA = 0
PLAYERS = range(1, 9)
CIV_TOKEN, CIV_INDEX = "BRITON-CIV", 1  # a str16 token from scenario 1.56 on, the .dat index before

KNIGHT, MILITIA, HOUSE = 38, 74, 70
COLUMNS = (KNIGHT, MILITIA, HOUSE)
# Stored angle 8 of 16 on Knight and Militia: the facing showing them the most player-colour cloth.
FACING = 5 * math.pi / 4
ROW_STEP, COL_STEP = 3, 2
# Shot A holds rows 0-4, shot B rows 5-8, one more row step apart.
SHOT_B_FIRST_ROW, SHOT_GAP = 5, 1
MARGIN = 2
# ColorId order, which is TEAM_COLORS' order after its leading GAIA white.
COLOUR_NAMES = ("blue", "red", "green", "yellow", "aqua", "purple", "grey", "orange")


def row_players() -> list[int]:
    return [GAIA, *PLAYERS]


def layout(ax: int, ay: int) -> list[dict]:
    """One dict per unit, rows top to bottom: row, shot, player, const and the
    stored (x, y). The block's left end is (ax, ay + its height)."""
    units = []
    for row, player in enumerate(row_players()):
        step = row + (SHOT_GAP if row >= SHOT_B_FIRST_ROW else 0)
        bx, by = ax + block_rows() * ROW_STEP - step * ROW_STEP, ay + step * ROW_STEP
        for col, const in enumerate(COLUMNS):
            tx, ty = bx + col * COL_STEP, by + col * COL_STEP
            # A 2x2 House centres on the far corner of its column's tile, a 1x1 unit on the tile centre.
            x, y = (tx + 1.0, ty + 1.0) if const == HOUSE else (tx + 0.5, ty + 0.5)
            units.append({
                "row": row, "shot": "A" if row < SHOT_B_FIRST_ROW else "B", "player": player,
                "column": col, "const": const, "name": object_catalog.combined_object_name(const),
                "x": x, "y": y,
            })
    return units


def block_rows() -> int:
    return len(row_players()) - 1 + SHOT_GAP


def block_size() -> int:
    """Tiles the block spans along each axis, the House's 2x2 included."""
    return block_rows() * ROW_STEP + (len(COLUMNS) - 1) * COL_STEP + 2


def find_area(loaded) -> tuple[int, int]:
    """Low corner of the first square block, scanning out from the map centre,
    whose tiles plus MARGIN are dry land of one elevation with no unit."""
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    size = block_size()
    taken: set[tuple[int, int]] = set()
    for units in loaded.unit_manager.units:
        for u in units:
            taken.update(unit_occupied_tiles(u, w, h) or ())
    elevation = {}
    for tile in mm.terrain:
        elevation[(tile.x, tile.y)] = tile.elevation
        if terrain_classes.is_water_family(tile.terrain_id) or terrain_classes.is_beach_family(tile.terrain_id):
            taken.add((tile.x, tile.y))
    cx, cy = (w - size) // 2, (h - size) // 2
    for radius in range(max(w, h)):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if max(abs(x - cx), abs(y - cy)) != radius:
                    continue
                x0, y0, x1, y1 = x - MARGIN, y - MARGIN, x + size + MARGIN, y + size + MARGIN
                if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                    continue
                block = [(tx, ty) for ty in range(y0, y1) for tx in range(x0, x1)]
                if not any(t in taken for t in block) and len({elevation[t] for t in block}) == 1:
                    return x, y
    raise SystemExit(f"no flat, dry, unit-free {size}x{size} area (plus {MARGIN} tiles) found")


def player_edits(loaded) -> dict[str, object]:
    """{player field key: value} for the after file's Players edits."""
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    edits: dict[str, object] = {}
    for player in PLAYERS:
        edits[player_fields.player_field_id("color", player)] = player - 1
        for field_id in ("civilization", "architecture"):
            if field_id in specs:
                stored = player_fields.current_value(loaded, specs[field_id], player)
                edits[player_fields.player_field_id(field_id, player)] = CIV_TOKEN if isinstance(stored, str) else CIV_INDEX
    return edits


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    loaded = load_map_and_units(source)
    ax, ay = find_area(loaded)
    units = layout(ax, ay)

    out.mkdir(parents=True, exist_ok=True)
    before, after = out / "before.aoe2scenario", out / "after.aoe2scenario"
    write_scenario(loaded, before, backup=False)
    options = OptionsEditModel(loaded)
    for key, value in player_edits(loaded).items():
        options.set_value(key, value)
    try:
        options.set_value(player_fields.PLAYER_COUNT_FIELD_ID, len(PLAYERS))
        count_note = f"player count {len(PLAYERS)}"
    except KeyError:
        count_note = "player count unchanged (not writable on this file)"
    model = UnitEditModel(loaded)
    model.begin_unit_edit(sorted({u["player"] for u in units}))
    state = place_rotation.PlaceRotation(facing=FACING)
    for u in units:
        u["rotation"] = place_rotation.rotation_for(state, u["const"], u["player"])
        model.add(u["player"], u["const"], u["x"], u["y"], rotation=u["rotation"])
    model.commit_unit_edit("Place GH #82 team colour rows", EditHistory())
    write_scenario(loaded, after, backup=False, options=options, units=model)

    print(f"wrote {before}\nwrote {after}")
    return readback(after, units, out / "layout.json", source, count_note)


def readback(path: Path, units: list[dict], layout_path: Path, source: Path, count_note: str) -> int:
    """Every unit found again in `path` at its spot with its rotation, every
    player's colour and civ as set; then the layout table and layout.json."""
    loaded = load_map_and_units(path)
    problems = []
    for u in units:
        hits = [v for v in loaded.unit_manager.units[u["player"]] if v.unit_const == u["const"] and (v.x, v.y) == (u["x"], u["y"])]
        if len(hits) != 1 or abs(hits[0].rotation - u["rotation"]) > 1e-5:
            problems.append(f"{u['name']} of player {u['player']} at ({u['x']}, {u['y']}): {len(hits)} found")
    art = civ_art.player_art(loaded)
    for player in PLAYERS:
        if loaded.team_indices[player] != player:
            problems.append(f"player {player} reads back team index {loaded.team_indices[player]}, not {player}")
        if art[player][0] != CIV_INDEX:
            problems.append(f"player {player} reads back art {art[player]}, not civ {CIV_INDEX}")
    for u in units:
        u["team_index"] = loaded.team_indices[u["player"]]
        u["art"] = list(art[u["player"]]) if u["player"] != GAIA else None

    layout_path.write_text(json.dumps({
        "source": str(source), "after": str(path), "facing": FACING, "units": units,
    }, indent=1) + "\n")
    print(f"wrote {layout_path}\n")
    print(f"Source {source.name}; {count_note}; players 1-8 Britons, ColorIds 0-7 in slot order.")
    print(f"Every unit faces {FACING:.4f} rad. Columns left to right on screen: "
          + ", ".join(object_catalog.combined_object_name(c) for c in COLUMNS) + ".\n")
    print(f"  {'shot':4} {'row':>3}  {'owner':7} {'colour':8} {'tint':>4}  {'Knight':>12} {'Militia':>12} {'House':>12}")
    for row, player in enumerate(row_players()):
        cells = {u["const"]: f"({u['x']:g}, {u['y']:g})" for u in units if u["row"] == row}
        first = next(u for u in units if u["row"] == row)
        owner = "GAIA" if player == GAIA else f"P{player}"
        colour = "white" if player == GAIA else COLOUR_NAMES[loaded.team_indices[player] - 1]
        print(f"  {first['shot']:4} {row:>3}  {owner:7} {colour:8} {first['team_index']:>4}  "
              + " ".join(f"{cells[c]:>12}" for c in COLUMNS))
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"\n{len(units)} placed, {len(problems)} problems.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The scenario to add the units to")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    args = parser.parse_args()
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
