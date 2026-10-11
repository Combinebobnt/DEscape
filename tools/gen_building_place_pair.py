#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
what it does with a building whose coordinates are not multiples of 0.5
(TASK-193, GH #163). No corpus file holds one, so nothing in-session can say
whether the game keeps such a point, snaps it, or refuses it.

Both files go through the real write path: "before" is a zero-edit
write_scenario(), "after" adds the probe objects with UnitEditModel.add()
then write_scenario(), the code Place Unit runs. The source's civs are all
base-game (GOTCHAS), so a tester without DLC can open the pair. It is
ring75, not the usual atilla_1: atilla's largest flat square with no unit
on it is 8x8, too small for the probes.

The probes sit in the first flat, empty, dry square found scanning out from
the map centre, all for player 1. Each axis gets its own fraction, so a
write that swaps or mirrors x and y reads back visibly wrong:

- Row A: five Houses (2x2) whose x fraction steps 0.0/0.2/0.4/0.6/0.8.
- Row B: the same staircase on y, as a column.
- Row C: a Barracks (3x3) on the grid as a control, and one at (.37, .71).
- Row D: a Castle (4x4), the same pair.
- Row E: an off-parity House (x.5, as 46 corpus Mills sit) beside an
  on-parity one. It settles how the game draws those corpus values.
- Row F: three fractional villagers, a 1x1 control (and TASK-008's case).

Markers: flags are eye candy but block (`.dat` obstruction 3), so they
cannot sit inside a footprint. Each building gets two Villagers just outside
its footprint instead: one on its centre's x (on the low-y side), one on its
centre's y (on the low-x side).

`--readback FILE` prints every probe's stored (x, y) in FILE, matched by
const to the nearest object, so "the game kept 10.37" can be told from "the
game rewrote it" after an in-game Save As. It only reads, so FILE may be
anywhere, a mods/subscribed/ path included.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import object_catalog, terrain_classes
from descape.edit_history import EditHistory
from descape.render import NON_BUILDING_SPAN, unit_occupied_tiles
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "ring75_v0_scx_resaved.aoe2scenario"
OUT = ROOT / "build" / "building_place_pair"
PLAYER = 1
AREA_W, AREA_H = 26, 24
MARGIN = 2
HOUSE, BARRACKS, CASTLE, VILLAGER = 70, 12, 82, 83
MARKER_GAP = 0.6
MATCH_RADIUS = 3.0


@dataclass(frozen=True)
class Probe:
    label: str
    unit_const: int
    dx: float  # offset from the area's low corner
    dy: float


def _buildings() -> list[Probe]:
    probes = []
    for i, frac in enumerate((0.0, 0.2, 0.4, 0.6, 0.8)):
        probes.append(Probe(f"A{i + 1} House x+{frac:.1f}", HOUSE, 2 + 4 * i + frac, 3.0))
    for i, frac in enumerate((0.0, 0.2, 0.4, 0.6, 0.8)):
        probes.append(Probe(f"B{i + 1} House y+{frac:.1f}", HOUSE, 23.0, 2 + 4 * i + frac))
    probes.append(Probe("C1 Barracks on grid", BARRACKS, 2.5, 8.5))
    probes.append(Probe("C2 Barracks (.37, .71)", BARRACKS, 8.37, 8.71))
    probes.append(Probe("D1 Castle on grid", CASTLE, 3.0, 15.0))
    probes.append(Probe("D2 Castle (.37, .71)", CASTLE, 10.37, 15.71))
    probes.append(Probe("E1 House on parity", HOUSE, 2.0, 21.0))
    probes.append(Probe("E2 House off parity x.5", HOUSE, 6.5, 21.0))
    return probes


def _markers(buildings: list[Probe]) -> list[Probe]:
    out = []
    for b in buildings:
        sx, sy = tile_span(b.unit_const, NON_BUILDING_SPAN)
        tag = b.label.split()[0]
        out.append(Probe(f"{tag} marker on x", VILLAGER, b.dx, b.dy - sy / 2 - MARKER_GAP))
        out.append(Probe(f"{tag} marker on y", VILLAGER, b.dx - sx / 2 - MARKER_GAP, b.dy))
    return out


def layout() -> list[Probe]:
    buildings = _buildings()
    villagers = [
        Probe("F1 Villager", VILLAGER, 11.23, 21.68),
        Probe("F2 Villager", VILLAGER, 13.61, 21.14),
        Probe("F3 Villager", VILLAGER, 15.87, 22.42),
    ]
    return buildings + villagers + _markers(buildings)


def find_area(loaded) -> tuple[int, int]:
    """Low corner of the first AREA_W x AREA_H square, scanning out from the map
    centre, whose tiles plus MARGIN are dry land of one elevation with no unit."""
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
                if max(abs(x - cx), abs(y - cy)) != radius:
                    continue
                x0, y0, x1, y1 = x - MARGIN, y - MARGIN, x + AREA_W + MARGIN, y + AREA_H + MARGIN
                if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                    continue
                block = [(tx, ty) for ty in range(y0, y1) for tx in range(x0, x1)]
                if any(t in taken for t in block):
                    continue
                if len({elevation[t] for t in block}) == 1:
                    return x, y
    raise SystemExit("no flat empty land area found")


def _name(unit_const: int) -> str:
    return object_catalog.combined_object_name(unit_const)


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    before, after = out / "before.aoe2scenario", out / "after.aoe2scenario"

    loaded = load_map_and_units(source)
    write_scenario(loaded, before, backup=False)
    ax, ay = find_area(loaded)
    probes = layout()
    model = UnitEditModel(loaded)
    model.begin_unit_edit([PLAYER])
    for p in probes:
        model.add(PLAYER, p.unit_const, ax + p.dx, ay + p.dy)
    model.commit_unit_edit("Place building free-placement probes", EditHistory())
    write_scenario(loaded, after, backup=False, units=model)

    elev = next(t.elevation for t in loaded.map_manager.terrain if (t.x, t.y) == (ax, ay))
    print(f"wrote {before}\nwrote {after}\n")
    print(f"Source {source.name}. Probe area: tiles x {ax}..{ax + AREA_W - 1}, y {ay}..{ay + AREA_H - 1}")
    print(f"({AREA_W}x{AREA_H}, flat at elevation {elev}, no unit or water within {MARGIN} tiles), all player {PLAYER}.\n")
    return readback(after, source)


def readback(path: Path, source: Path = SOURCE) -> int:
    """Each probe's stored (x, y) in `path`, matched to the nearest same-const
    player-1 object; the layout is re-derived from the untouched `source`."""
    ax, ay = find_area(load_map_and_units(source))
    units = load_map_and_units(path).unit_manager.units[PLAYER]
    print(f"Read back {path}:")
    print(f"  {'probe':28s} {'const':>5s} {'intended':>18s} {'stored':>18s}   delta")
    used: set[int] = set()
    missing = 0
    for p in layout():
        want = (ax + p.dx, ay + p.dy)
        candidates = [
            u for u in units
            if u.unit_const == p.unit_const and id(u) not in used
            and abs(u.x - want[0]) <= MATCH_RADIUS and abs(u.y - want[1]) <= MATCH_RADIUS
        ]
        intended = f"({want[0]:.2f}, {want[1]:.2f})"
        if not candidates:
            missing += 1
            print(f"  {p.label:28s} {p.unit_const:5d} {intended:>18s} {'MISSING':>18s}")
            continue
        u = min(candidates, key=lambda u: (u.x - want[0]) ** 2 + (u.y - want[1]) ** 2)
        used.add(id(u))
        stored = f"({u.x:.4f}, {u.y:.4f})"
        # + 0.0 turns float32 noise's -0.0 into 0.0
        ddx, ddy = round(u.x - want[0], 4) + 0.0, round(u.y - want[1], 4) + 0.0
        print(f"  {p.label:28s} {p.unit_const:5d} {intended:>18s} {stored:>18s}   ({ddx:+.4f}, {ddy:+.4f})")
    print(f"\n{missing} missing. Stored values are float32: a kept 10.2 reads back as 10.2000, a")
    print("snapped one as a different number. Consts: " + ", ".join(
        f"{c} {_name(c)}" for c in (HOUSE, BARRACKS, CASTLE, VILLAGER)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="The scenario to add the probes to")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--readback", type=Path, help="Only print each probe's stored (x, y) in this file")
    args = parser.parse_args()
    if args.readback is not None:
        return readback(args.readback, args.source)
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
