#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
whether a unit only a data mod defines survives DEscape's write path (GH #58).

Exploding Villagers (Workshop 93961) adds const 2750 (HDSQD, type 70), which
the install's .dat lacks; DEscape names it "Saboteur" (string 5588). With that
mod picked in File > Data Mod, the unit picker lists it as "Saboteur (data
mod)" and Place Unit writes it like any other const. Nothing
in-session can say whether the game, with the mod selected, shows it.

Both files go through the real write path: the mod's tables are activated the
way the viewer does it (data_mods.discover/entry_key/tables_for, then
game_dataset.set_data_mod()), "before" is a zero-edit write_scenario(), and
"after" adds the probes with UnitEditModel.add(), then write_scenario(). A
cache miss is derived first through data_mods.derive_command(), the same
subprocess the viewer starts. The source is atilla_1, whose civs are all
base-game (GOTCHAS).

Probes, all player 1, on the first flat, empty, dry 6x3 area scanning out
from the map centre:

- 2750 HDSQD at the area's low corner + (1.5, 1.5).
- A Villager (83) 3 tiles further along x, as a control: if the after file
  shows the Villager but not the HDSQD, the game dropped the mod's const; if
  it shows neither, the file itself did not load.

`--readback FILE` prints every probe const's player-1 objects near the area.
It only reads, so FILE may be anywhere, a mods/subscribed/ path included.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, batch_api, data_mods, game_dataset, object_catalog, terrain_classes
from descape.edit_history import EditHistory
from descape.render import unit_occupied_tiles
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
OUT = ROOT / "build" / "gh58_pair"
MOD = "subscribed//93961_Exploding Villagers"
PLAYER = 1
HDSQD, VILLAGER = 2750, 83
AREA_W, AREA_H = 6, 3
PROBES = ((HDSQD, 1.5, 1.5), (VILLAGER, 4.5, 1.5))
MATCH_RADIUS = 3.0


def activate_mod() -> game_dataset.DataModTables:
    install = asset_source.get_install_path()
    if install is None:
        raise SystemExit("no AoE2:DE install configured")
    mods = {mod.status_path: mod for mod in data_mods.discover()}
    if MOD not in mods:
        raise SystemExit(f"data mod not found in the profile: {MOD}")
    mod = mods[MOD]
    key = data_mods.entry_key(mod, install)
    tables = data_mods.tables_for(mod, key)
    if tables is None:
        argv, cwd = data_mods.derive_command(mod, install, key)
        print(f"deriving {mod.title} (cache miss) ...")
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise SystemExit(f"derive failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
        tables = data_mods.tables_for(mod, key)
    if tables is None or HDSQD not in tables.mod_only_ids:
        raise SystemExit(f"{mod.title}'s tables do not list {HDSQD} as mod-only")
    game_dataset.set_data_mod(tables)
    return tables


def find_area(loaded) -> tuple[int, int]:
    """Low corner of the first AREA_W x AREA_H square, scanning out from the map
    centre, of dry land at one elevation with no unit on it."""
    mm = loaded.map_manager
    w, h = mm.map_width, mm.map_height
    taken: set[tuple[int, int]] = set()
    for units in loaded.unit_manager.units:
        for u in units:
            taken.update(unit_occupied_tiles(u, w, h) or ())
    elevation = {}
    for tile in mm.terrain:
        elevation[(tile.x, tile.y)] = tile.elevation
        if terrain_classes.is_water_family(tile.terrain_id) or terrain_classes.is_beach_family(tile.terrain_id):
            taken.add((tile.x, tile.y))
    cx, cy = (w - AREA_W) // 2, (h - AREA_H) // 2
    for radius in range(max(w, h)):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if max(abs(x - cx), abs(y - cy)) != radius or x < 0 or y < 0 or x + AREA_W > w or y + AREA_H > h:
                    continue
                block = [(tx, ty) for ty in range(y, y + AREA_H) for tx in range(x, x + AREA_W)]
                if not any(t in taken for t in block) and len({elevation[t] for t in block}) == 1:
                    return x, y
    raise SystemExit("no flat empty land area found")


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    tables = activate_mod()
    out.mkdir(parents=True, exist_ok=True)
    before, after = out / "before.aoe2scenario", out / "after.aoe2scenario"

    loaded = batch_api.load(source)
    write_scenario(loaded, before, backup=False)
    ax, ay = find_area(loaded)
    model = UnitEditModel(loaded)
    model.begin_unit_edit([PLAYER])
    for const, dx, dy in PROBES:
        model.add(PLAYER, const, ax + dx, ay + dy)
    model.commit_unit_edit("Place a data mod's unit", EditHistory())
    write_scenario(loaded, after, backup=False, units=model)

    print(f"wrote {before}\nwrote {after}\n")
    print(f"Source {source.name}, data mod {tables.title} (mod-only ids {sorted(tables.mod_only_ids)}).")
    print(f"Probe area: tiles x {ax}..{ax + AREA_W - 1}, y {ay}..{ay + AREA_H - 1}, player {PLAYER}.\n")
    return readback(before, source) or readback(after, source)


def readback(path: Path, source: Path = SOURCE) -> int:
    """Each probe const's player-1 objects in `path` near where the probe goes;
    the area is re-derived from the untouched `source`."""
    ax, ay = find_area(batch_api.load(source))
    units = batch_api.load(path).unit_manager.units[PLAYER]
    print(f"Read back {path.name}:")
    for const, dx, dy in PROBES:
        want = (ax + dx, ay + dy)
        near = [u for u in units if u.unit_const == const
                and abs(u.x - want[0]) <= MATCH_RADIUS and abs(u.y - want[1]) <= MATCH_RADIUS]
        found = ", ".join(f"({u.x:.2f}, {u.y:.2f}) ref {u.reference_id}" for u in near) or "none"
        print(f"  {const:5d} {object_catalog.display_name(const):24s} want ({want[0]:.2f}, {want[1]:.2f}): {found}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The scenario to add the probes to")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--readback", type=Path, help="Only print the probes found in this file")
    args = parser.parse_args()
    if args.readback is not None:
        activate_mod()
        return readback(args.readback, args.source)
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
