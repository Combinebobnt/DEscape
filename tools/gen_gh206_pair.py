#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
whether even-span buildings now land on its grid (GH #206), and why a tester's
Forage Bush vanished beside a half-tile-off Mill.

Same source, flat-area finder and write path as gen_building_place_pair.py:
"before" is a zero-edit write_scenario(), "after" adds the probes with
UnitEditModel.add() then write_scenario(), the code Place Unit runs.

- Row A (the fix): a Mill (2x2) and a Market (4x4) at render.snapped_anchor(),
  the values snapped Place Unit now writes. Each has a Villager on the tile
  just outside its footprint's low corner and one just outside its high one.
- Row B (the tester's repro): a Mill at (t + 0.5, u + 0.5), the old snapped
  value, with a player-1 Forage Bush on tile (t - 1, u - 1) and a Villager at
  (t - 0.5, u + 0.5). Beside it the fixed control, a Mill at (t' + 1, u + 1)
  with the same bush and Villager relative to its tile.
- Row C (owner): a lone Forage Bush for player 1 and one for GAIA.

`--readback FILE` prints every probe's stored (x, y) in FILE, matched by
owner and const to the nearest object. It only reads, so FILE may be
anywhere, a mods/subscribed/ path included.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_building_place_pair as base

from descape import object_catalog
from descape.edit_history import EditHistory
from descape.render import snapped_anchor
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_model import UnitEditModel

SOURCE = base.SOURCE
OUT = ROOT / "build" / "gh206_pair"
GAIA, PLAYER = 0, 1
MILL, MARKET, FORAGE_BUSH, VILLAGER = 68, 84, 59, base.VILLAGER
MATCH_RADIUS = 1.0  # under the 3.0 between A1's two markers, so a dropped one reads MISSING


@dataclass(frozen=True)
class Probe:
    label: str
    player: int
    unit_const: int
    dx: float  # offset from the area's low corner
    dy: float


def _fixed(label: str, const: int, tile: tuple[int, int]) -> list[Probe]:
    """A building at Place Unit's snapped anchor for `tile`, plus its two corner markers."""
    span = BUILDING_TILE_SPANS[const]
    x, y = snapped_anchor(tile, None, span, base.AREA_W, base.AREA_H)
    lo_x, lo_y = int(x - span[0] / 2), int(y - span[1] / 2)
    hi_x, hi_y = int(x + span[0] / 2), int(y + span[1] / 2)
    tag = label.split(maxsplit=1)[0]
    return [
        Probe(label, PLAYER, const, x, y),
        Probe(f"{tag} marker low", PLAYER, VILLAGER, lo_x - 0.5, lo_y - 0.5),
        Probe(f"{tag} marker high", PLAYER, VILLAGER, hi_x + 0.5, hi_y + 0.5),
    ]


def _repro(tag: str, mill: tuple[float, float], tile: tuple[int, int], what: str) -> list[Probe]:
    t, u = tile
    return [
        Probe(f"{tag} Mill {what}", PLAYER, MILL, *mill),
        Probe(f"{tag} Forage Bush (P1)", PLAYER, FORAGE_BUSH, t - 0.5, u - 0.5),
        Probe(f"{tag} Villager", PLAYER, VILLAGER, t - 0.5, u + 0.5),
    ]


def layout() -> list[Probe]:
    return [
        *_fixed("A1 Mill snapped", MILL, (3, 3)),
        *_fixed("A2 Market snapped", MARKET, (10, 3)),
        *_repro("B1", (4.5, 12.5), (4, 12), "at t + 0.5 (old)"),
        *_repro("B2", (13.0, 13.0), (12, 12), "at t + 1 (fixed)"),
        Probe("C1 Forage Bush (P1)", PLAYER, FORAGE_BUSH, 4.5, 20.5),
        Probe("C2 Forage Bush (GAIA)", GAIA, FORAGE_BUSH, 12.5, 20.5),
    ]


EXPECTED = """Expected in the AoE2:DE editor:
  Row A: each Mill/Market sits on whole tiles. Its two Villagers stand on the
         diagonal tiles just past its low and high corners, clear of the
         footprint but adjacent to it, the same at both ends.
  Row B: B1's (left) bush vanishes or is covered by its Mill; B2's (right)
         bush survives beside its Mill.
  Row C: both lone bushes exist. If C1 (player 1) is gone, ownership alone
         removes it, not the footprint."""


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
    ax, ay = base.find_area(loaded)
    model = UnitEditModel(loaded)
    model.begin_unit_edit([GAIA, PLAYER])
    for p in layout():
        model.add(p.player, p.unit_const, ax + p.dx, ay + p.dy)
    model.commit_unit_edit("Place GH #206 probes", EditHistory())
    write_scenario(loaded, after, backup=False, units=model)

    print(f"wrote {before}\nwrote {after}\n")
    print(f"Source {source.name}. Probe area: tiles x {ax}..{ax + base.AREA_W - 1}, y {ay}..{ay + base.AREA_H - 1}.")
    print("Rows run top to bottom in y: A at y+1..6, B at y+11..13, C at y+20.\n")
    print(EXPECTED + "\n")
    return readback(after, source)


def readback(path: Path, source: Path = SOURCE) -> int:
    """Each probe's stored (x, y) in `path`, matched to the nearest object of
    the same owner and const; the layout is re-derived from the untouched `source`."""
    ax, ay = base.find_area(load_map_and_units(source))
    units = load_map_and_units(path).unit_manager.units
    print(f"Read back {path}:")
    print(f"  {'probe':28s} {'own':>3s} {'const':>5s} {'intended':>18s} {'stored':>18s}")
    used: set[int] = set()
    missing = 0
    for p in layout():
        want = (ax + p.dx, ay + p.dy)
        candidates = [
            u for u in units[p.player]
            if u.unit_const == p.unit_const and id(u) not in used
            and abs(u.x - want[0]) <= MATCH_RADIUS and abs(u.y - want[1]) <= MATCH_RADIUS
        ]
        intended = f"({want[0]:.2f}, {want[1]:.2f})"
        if not candidates:
            missing += 1
            print(f"  {p.label:28s} {p.player:3d} {p.unit_const:5d} {intended:>18s} {'MISSING':>18s}")
            continue
        u = min(candidates, key=lambda u: (u.x - want[0]) ** 2 + (u.y - want[1]) ** 2)
        used.add(id(u))
        print(f"  {p.label:28s} {p.player:3d} {p.unit_const:5d} {intended:>18s} {f'({u.x:.4f}, {u.y:.4f})':>18s}")
    print(f"\n{missing} missing. Consts: " + ", ".join(
        f"{c} {object_catalog.combined_object_name(c)}" for c in (MILL, MARKET, FORAGE_BUSH, VILLAGER)))
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
