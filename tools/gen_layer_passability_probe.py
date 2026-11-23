#!/usr/bin/env python3
"""Writes one .aoe2scenario asking the real AoE2:DE editor whether layered
terrain (the editor's Layering Mode) counts for placement and pathing, or
only the base terrain under it does. DEscape's terrain rule
(descape/placement_rules.py) reads the base; only the game can say whether
that is right.

Three zones near (100..127, 100..121), each a base and a layer that
disagree, plus a real lake and real grass as controls. Deliberately
asymmetric, so no zone mirrors another:

- Lake (control): Water, Deep (22), no layer, x 100..127, y 100..111.
- Zone B, land over water: inside the lake, base 22 layered with Grass 1
  (0), x 110..115, y 103..108. Looks like an island.
- Zone A, water over land: base Grass 1 layered with Water, Deep (22),
  x 100..105, y 116..121. Looks like a pond.
- Zone C, the corpus pair: base Water, Shallow (1) layered with Beach, Wet
  (107), x 114..119, y 116..121. The commonest disagreeing pair in real files.

P1 units stand on each zone and on the controls; P1's Point of View is set
to the middle of it all. Every edit goes through the code the app uses
(terrain_target.write_tile, UnitEditModel.add, OptionsEditModel, then
write_scenario), never a byte patch. Before writing, the tool asserts the
current rule's view of each unit, so the probe probes what it claims; after
writing, it reloads the file and checks every tile and unit.

  (no flags)          write build/layer_passability_probe/probe.aoe2scenario
  --to-game           also copy it into the game profile's scenario folder
                      as layer_passability_probe.aoe2scenario
  --game-path         print where --to-game would copy, and write nothing
  --readback [FILE]   read an in-game Save As (default: the newest
                      layer_passability_probe*.aoe2scenario in that folder
                      other than the --to-game copy). Read-only, any path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, content_roots, object_catalog, placement_rules, player_fields, terrain_catalog
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_target import LAYER, TERRAIN, write_tile
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "tests" / "fixtures" / "real_blank_240x240.aoe2scenario"
OUT = ROOT / "build" / "layer_passability_probe"
OUT_NAME = "probe.aoe2scenario"
GAME_NAME = "layer_passability_probe.aoe2scenario"
GAME_GLOB = "layer_passability_probe*.aoe2scenario"
PLAYER = 1
POV = (113, 111)
MATCH_RADIUS = 1.5

GRASS_1, WATER_SHALLOW, WATER_DEEP, BEACH_WET = 0, 1, 22, 107
VILLAGER, FISHING_SHIP, GALLEY, KNIGHT = 83, 13, 539, 38
LAND_ROW, WATER_ROW = 7, 13  # the villager's and the fishing ship's .dat restriction rows

# Certainly base game: the 13 Age of Kings civs, as 1.56+ tokens and pre-1.56 ids.
BASE_CIV_TOKENS = frozenset({
    "BRITON-CIV", "BYZANTINE-CIV", "CELTIC-CIV", "CHINESE-CIV", "FRANKISH-CIV", "GOTHIC-CIV", "JAPANESE-CIV",
    "MONGOL-CIV", "PERSIAN-CIV", "SARACEN-CIV", "TEUTONIC-CIV", "TURKISH-CIV", "VIKING-CIV",
})
BASE_CIV_IDS = frozenset(range(1, 14))
BRITONS_TOKEN, BRITONS_ID = "BRITON-CIV", 1


@dataclass(frozen=True)
class Zone:
    letter: str
    title: str
    xs: range
    ys: range
    base: int
    layer: int  # -1: none

    def tiles(self) -> list[tuple[int, int]]:
        return [(x, y) for y in self.ys for x in self.xs]

    def span(self) -> str:
        return f"x {self.xs[0]}..{self.xs[-1]}, y {self.ys[0]}..{self.ys[-1]}"


LAKE = Zone("L", "Lake (control, real water)", range(100, 128), range(100, 112), WATER_DEEP, -1)
ZONE_B = Zone("B", "Zone B, land over water", range(110, 116), range(103, 109), WATER_DEEP, GRASS_1)
ZONE_A = Zone("A", "Zone A, water over land", range(100, 106), range(116, 122), GRASS_1, WATER_DEEP)
ZONE_C = Zone("C", "Zone C, the corpus pair", range(114, 120), range(116, 122), WATER_SHALLOW, BEACH_WET)
ZONES = (LAKE, ZONE_B, ZONE_A, ZONE_C)  # write order: B's base is the lake's
LAYERED = (ZONE_A, ZONE_B, ZONE_C)
# Plain grass, flat and empty, the source must hold around the probe (2 tiles of margin).
AREA_XS, AREA_YS = range(98, 130), range(98, 124)


@dataclass(frozen=True)
class Probe:
    label: str
    const: int
    x: float
    y: float
    refused: bool  # what placement_rules.terrain_refusal() must say today
    zone: str = ""  # the zone letter a missing unit FAILs for


PROBES = (
    Probe("zone A Villager", VILLAGER, 102.5, 118.5, False, "A"),
    Probe("zone B Fishing Ship", FISHING_SHIP, 112.5, 105.5, False, "B"),
    Probe("zone C Villager", VILLAGER, 116.5, 118.5, True, "C"),
    Probe("mover Galley (lake)", GALLEY, 103.5, 104.5, False),
    Probe("mover Knight (grass)", KNIGHT, 109.5, 118.5, False),
    Probe("mover Villager (grass)", VILLAGER, 109.5, 120.5, False),
    Probe("control Villager (lake)", VILLAGER, 124.5, 109.5, True),
)


def _terrain(terrain_id: int) -> str:
    return f"{terrain_catalog.display_name(name_for_terrain_id(terrain_id))} ({terrain_id})"


def _pair(base: int, layer: int) -> str:
    return _terrain(base) + (f" layered with {_terrain(layer)}" if layer != -1 else ", no layer")


def expected_tiles() -> dict[tuple[int, int], tuple[int, int]]:
    """(x, y) -> (terrain_id, layer) for every tile the probe writes; the rest stay as the source had them."""
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for zone in ZONES:
        for xy in zone.tiles():
            out[xy] = (zone.base, zone.layer)
    return out


def _tile_at(mm, x: int, y: int):
    return mm.terrain[y * mm.map_width + x]


def _zone_of(x: float, y: float) -> str:
    tx, ty = int(x), int(y)
    for zone in LAYERED:
        if tx in zone.xs and ty in zone.ys:
            return zone.letter
    return ""


# -- the game profile -------------------------------------------------------------


def game_folder() -> Path | None:
    """The active profile's scenario folder, or None when no profile resolves."""
    profile = content_roots.profile_path()
    if profile is None:
        return None
    profile_id = content_roots.active_profile_id(profile)
    if profile_id is None:
        return None
    return profile / profile_id / "resources" / "_common" / "scenario"


def _refuse_subscribed(path: Path) -> None:
    if is_workshop_subscribed(path):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {path}")


# -- write ------------------------------------------------------------------------


def _source_problem(loaded) -> str | None:
    mm = loaded.map_manager
    if not loaded.terrain_has_layer:
        return "the source's terrain has no layer field"
    if mm.map_width <= AREA_XS[-1] or mm.map_height <= AREA_YS[-1]:
        return f"the source is {mm.map_width}x{mm.map_height}; the probe needs at least {AREA_XS[-1] + 1} tiles"
    area = [_tile_at(mm, x, y) for y in AREA_YS for x in AREA_XS]
    if {(t.terrain_id, t.layer) for t in area} != {(GRASS_1, -1)} or len({t.elevation for t in area}) != 1:
        return f"x {AREA_XS[0]}..{AREA_XS[-1]}, y {AREA_YS[0]}..{AREA_YS[-1]} is not flat unlayered Grass 1"
    for units in loaded.unit_manager.units:
        for u in units:
            if int(u.x) in AREA_XS and int(u.y) in AREA_YS:
                return f"a unit (const {u.unit_const}) stands at ({u.x}, {u.y}) inside the probe area"
    return None


def _set_player(loaded, options: OptionsEditModel) -> str:
    """P1 active with a base-game civ (Britons if not) and its view on POV; returns the civ note."""
    if PLAYER not in player_fields.defined_player_ids(loaded):
        raise SystemExit(f"P{PLAYER} is not active in the source")
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    civ = player_fields.current_value(loaded, specs["civilization"], PLAYER)
    note = f"P{PLAYER} civ {civ} kept"
    if civ not in (BASE_CIV_TOKENS if isinstance(civ, str) else BASE_CIV_IDS):
        britons = BRITONS_TOKEN if isinstance(civ, str) else BRITONS_ID
        options.set_value(player_fields.player_field_id("civilization", PLAYER), britons)
        note = f"P{PLAYER} civ {civ} set to {britons}"
    options.set_value(player_fields.player_field_id(player_fields.POV_X_FIELD, PLAYER), POV[0])
    options.set_value(player_fields.player_field_id(player_fields.POV_Y_FIELD, PLAYER), POV[1])
    return note


def _assert_rule_view(loaded) -> None:
    """The current rule's answer for every probe, and the two rows it rests on."""
    objects = object_catalog.dat_objects()
    rows = object_catalog.dat_terrain_restrictions()
    assert objects[VILLAGER]["restriction"] == LAND_ROW, objects[VILLAGER]
    assert objects[FISHING_SHIP]["restriction"] == WATER_ROW, objects[FISHING_SHIP]
    land, water = rows[LAND_ROW], rows[WATER_ROW]
    assert WATER_DEEP not in land and WATER_SHALLOW not in land, "row 7 must refuse 22 and 1"
    assert GRASS_1 in land and BEACH_WET in land, "row 7 must pass 0 and 107"
    assert WATER_DEEP in water and GRASS_1 not in water, "row 13 must pass 22 and refuse 0"
    for p in PROBES:
        why = placement_rules.terrain_refusal(loaded, p.const, p.x, p.y)
        assert (why is not None) == p.refused, f"{p.label}: terrain_refusal() is {why!r}, expected refused={p.refused}"


def _tile_state(loaded) -> dict[tuple[int, int], tuple[int, int]]:
    return {(t.x, t.y): (t.terrain_id, t.layer) for t in loaded.map_manager.terrain}


def _unit_count(loaded) -> int:
    return sum(len(units) for units in loaded.unit_manager.units)


def _written_problems(path: Path, civ_wanted, base_tiles, base_units: int) -> list[str]:
    """Reload `path`: every tile's (terrain_id, layer) against the source's own outside the
    zones, every probe's const and point, P1's civ and view."""
    loaded = batch_api.load(path)
    expected = {**base_tiles, **expected_tiles()}
    problems = [
        f"tile {xy} holds {state}, expected {expected.get(xy)}"
        for xy, state in _tile_state(loaded).items()
        if state != expected.get(xy)
    ]
    count = _unit_count(loaded)
    if count != base_units + len(PROBES):
        problems.append(f"{count} units, expected the source's {base_units} plus {len(PROBES)}")
    p1 = loaded.unit_manager.units[PLAYER]
    problems.extend(
        f"{p.label}: no const {p.const} at ({p.x}, {p.y})"
        for p in PROBES
        if not any(u.unit_const == p.const and abs(u.x - p.x) < 1e-3 and abs(u.y - p.y) < 1e-3 for u in p1)
    )
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    civ = player_fields.current_value(loaded, specs["civilization"], PLAYER)
    if civ != civ_wanted:
        problems.append(f"P{PLAYER} civ {civ}, expected {civ_wanted}")
    view = tuple(player_fields.current_value(loaded, specs[f], PLAYER)
                 for f in (player_fields.POV_X_FIELD, player_fields.POV_Y_FIELD))
    if view != POV:
        problems.append(f"P{PLAYER} view {view}, expected {POV}")
    return problems


def write_probe(source: Path, out: Path) -> Path:
    _refuse_subscribed(out)
    if not source.is_file():
        raise SystemExit(f"missing source: {source}")
    loaded = batch_api.load(source)
    problem = _source_problem(loaded)
    if problem:
        raise SystemExit(problem)
    base_tiles, base_units = _tile_state(loaded), _unit_count(loaded)

    history = EditHistory()
    options = OptionsEditModel(loaded)
    civ_note = _set_player(loaded, options)
    mm = loaded.map_manager
    for zone in ZONES:
        for x, y in zone.tiles():
            tile = _tile_at(mm, x, y)
            if tile.terrain_id != zone.base:
                write_tile(tile, TERRAIN, zone.base)
            if zone.layer != -1:
                write_tile(tile, LAYER, zone.layer)
    _assert_rule_view(loaded)
    units = UnitEditModel(loaded)
    units.begin_unit_edit([PLAYER])
    for p in PROBES:
        units.add(PLAYER, p.const, p.x, p.y)
    units.commit_unit_edit("Layer passability probes", history)

    out.mkdir(parents=True, exist_ok=True)
    path = out / OUT_NAME
    write_scenario(loaded, path, units=units, options=options, backup=False)
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    civ_wanted = options.pending_values().get(
        player_fields.player_field_id("civilization", PLAYER),
        player_fields.current_value(loaded, specs["civilization"], PLAYER),
    )
    problems = _written_problems(path, civ_wanted, base_tiles, base_units)
    print(f"wrote {path} (from {source.name}; {civ_note}; P{PLAYER} view {POV})")
    if problems:
        for line in problems[:20]:
            print(f"  FAIL {line}", file=sys.stderr)
        raise SystemExit(f"reload check failed: {len(problems)} problems")
    print(f"reloaded: every tile and all {len(PROBES)} units as written\n")
    return path


def copy_to_game(path: Path) -> Path:
    folder = game_folder()
    if folder is None:
        raise SystemExit("no game profile resolves (set it in Settings > Mods); nothing copied")
    target = folder / GAME_NAME
    _refuse_subscribed(target)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    print(f"copied to {target}\n")
    return target


def print_checklist() -> None:
    print("Zones (tile coordinates as DEscape's hover readout shows them):")
    for zone in ZONES:
        print(f"  {zone.title}: {zone.span()} = {_pair(zone.base, zone.layer)}")
    print("  Real Grass 1 everywhere else.")
    print(f"Units, all P{PLAYER}:")
    for p in PROBES:
        print(f"  {p.label}: {object_catalog.display_name(p.const)} ({p.const}) at ({p.x}, {p.y})")
    print()
    print("H1 (AoE2:DE editor): open layer_passability_probe. Units tab, P1, try to place:")
    print(f"  a Villager on zone A ({ZONE_A.span()}), zone B ({ZONE_B.span()}) and the real lake")
    print("    (e.g. tile (120, 102));")
    print("  a Fishing Ship on zone A, zone B and real grass (e.g. tile (110, 114));")
    print(f"  a House on zone A; a Dock on zone B's west edge (x {ZONE_B.xs[0]}, y {ZONE_B.ys[0]}..{ZONE_B.ys[-1]}).")
    print("  Note each accepted or refused.")
    print("H2 (AoE2:DE Test as P1): are the zone A, B and C units and the control villager still there?")
    print(f"  Order the Galley onto zone B's centre ({ZONE_B.xs[0] + 3}, {ZONE_B.ys[0] + 3}); the Knight and the")
    print(f"  mover Villager onto zone A's centre ({ZONE_A.xs[0] + 3}, {ZONE_A.ys[0] + 3}), then zone C's")
    print(f"  ({ZONE_C.xs[0] + 3}, {ZONE_C.ys[0] + 3}); order each zone unit two tiles out and back.")
    print("  Note who reaches what. Then, back in the editor, Save As layer_passability_probe_saved")
    print("  and run this tool's --readback: no FAIL.")


# -- read back --------------------------------------------------------------------


def _default_readback() -> Path:
    folder = game_folder()
    if folder is None:
        raise SystemExit("no game profile resolves; pass the saved file to --readback")
    found = [p for p in folder.glob(GAME_GLOB) if p.name != GAME_NAME]
    if not found:
        raise SystemExit(f"no {GAME_GLOB} other than {GAME_NAME} in {folder}")
    return max(found, key=lambda p: p.stat().st_mtime)


def readback(path: Path) -> int:
    loaded = batch_api.load(path)
    mm = loaded.map_manager
    print(f"Read back {path}:")
    fails = 0
    used: set[int] = set()
    p1 = loaded.unit_manager.units[PLAYER] if len(loaded.unit_manager.units) > PLAYER else []
    for p in PROBES:
        near = [
            u for u in p1
            if u.unit_const == p.const and id(u) not in used
            and abs(u.x - p.x) <= MATCH_RADIUS and abs(u.y - p.y) <= MATCH_RADIUS
        ]
        if not near:
            tag = "FAIL" if p.zone else "    "
            fails += bool(p.zone)
            print(f"  {tag} {p.label:26s} const {p.const:4d} written ({p.x}, {p.y}): MISSING")
            continue
        u = min(near, key=lambda u: (u.x - p.x) ** 2 + (u.y - p.y) ** 2)
        used.add(id(u))
        print(f"  ok   {p.label:26s} const {p.const:4d} written ({p.x}, {p.y}): stored ({u.x:.4f}, {u.y:.4f})")
    for player, units in enumerate(loaded.unit_manager.units):
        for u in units:
            zone = _zone_of(u.x, u.y)
            if id(u) not in used and zone:
                print(f"  extra on zone {zone}: P{player} {object_catalog.display_name(u.unit_const)}"
                      f" ({u.unit_const}) on tile ({int(u.x)}, {int(u.y)})")
    expected = expected_tiles()
    for zone in ZONES:
        # The lake's own tiles only: zone B sits inside it.
        own = [xy for xy in zone.tiles() if expected[xy] == (zone.base, zone.layer)]
        wrong = [
            (x, y, (t.terrain_id, t.layer))
            for x, y in own
            if (t := _tile_at(mm, x, y)) is not None and (t.terrain_id, t.layer) != expected[(x, y)]
        ]
        if wrong:
            fails += 1
            x, y, got = wrong[0]
            print(f"  FAIL {zone.title}: {len(wrong)} tiles changed, e.g. ({x}, {y}) holds {got}")
        else:
            print(f"  ok   {zone.title}: every tile holds {_pair(zone.base, zone.layer)}")
    print(f"\n{fails} FAIL")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Plain-grass scenario to build on")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--to-game", action="store_true", help="Also copy the probe into the game profile")
    parser.add_argument("--game-path", action="store_true", help="Print where --to-game would copy, then stop")
    parser.add_argument("--readback", nargs="?", const="", type=str, metavar="FILE",
                        help="Only read back FILE (default: the newest in-game Save As)")
    args = parser.parse_args(argv)
    if args.game_path:
        folder = game_folder()
        print(folder / GAME_NAME if folder is not None else "no game profile resolves")
        return 0 if folder is not None else 1
    if args.readback is not None:
        return readback(Path(args.readback) if args.readback else _default_readback())
    path = write_probe(args.source, args.out)
    if args.to_game:
        copy_to_game(path)
    print_checklist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
