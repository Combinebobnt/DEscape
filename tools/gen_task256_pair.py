#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
whether a selection flipped or turned in place (TASK-256) reads right: the
facings, a gate's orientation and the wall shapes the game draws.

Both files go through the real write path. "before" is the source with a test
layout added (UnitEditModel.add() and raw terrain plus set_tiles_elevation(),
then write_scenario()); "after" is that same document with two in-place edits,
planned by region_transform.plan_in_place() and written through the same
UnitEditModel setters and tile writers the editor's Flip and Rotate Group use
(set_unit_const(), set_position(), set_rotation(), set_wall_variant();
paste_terrain() and set_tiles_elevation()), then write_scenario(). The source's
civs are all base-game (GOTCHAS), so a tester without DLC can open the pair. It
is ring75, as for tools/gen_building_place_pair.py: atilla_1 has no flat empty
area this size.

The layout sits in the first flat, empty, dry area found scanning out from the
map centre, all player 1:

- Group A, flipped horizontally on screen (the isometric view's left/right,
  element `a`) about its own footprint box: a Knight facing one way, a Stone
  Wall run along one diagonal with a Gate in it, and a House off centre. The
  Gate takes its other diagonal sibling and the walls their other run index.
- Region B, turned a quarter turn clockwise on screen about its centre: an
  8 x 8 square with a Road L, a Dirt stripe and a raised corner, holding a
  Villager facing one way, two Oaks and a House.

Run from the repo root; output goes to build/task256/ (gitignored).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import region_clipboard, region_transform, render, terrain_classes, unit_rotation
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tiles_elevation
from descape.mirror_tools import walls_are_radian
from descape.render import unit_occupied_tiles
from descape.scatter_dialog import units_region
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "ring75_v0_scx_resaved.aoe2scenario"
OUT = ROOT / "build" / "task256"
PLAYER = 1
AREA_W, AREA_H = 36, 16
MARGIN = 3
KNIGHT, VILLAGER, OAK, HOUSE, WALL, GATE = 38, 83, 349, 70, 117, 64
ROAD, DIRT = 24, 6
REGION_X, REGION_Y, REGION = 24, 4, 8  # Region B, local to the area


def _facing(unit_const: int, step: int) -> float:
    """Frame `step` of the const's own facing grid, so the game draws it exactly."""
    count = unit_rotation.angle_count_for(unit_const)
    return step * 2 * math.pi / count


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


def _group_a(ax: int, ay: int) -> list[tuple[int, float, float, float]]:
    """(const, x, y, rotation): a knight, a wall run with a gate in it, a house."""
    units = [(KNIGHT, ax + 2.5, ay + 2.5, _facing(KNIGHT, 3))]
    wy = ay + 9.5
    units += [(WALL, ax + x + 0.5, wy, 0.0) for x in (2, 3, 4)]
    units.append((GATE, *render.span_anchor(ax + 5, ay + 9, 4, 1), 0.0))
    units += [(WALL, ax + x + 0.5, wy, 0.0) for x in (9, 10, 11)]
    units.append((HOUSE, *render.span_anchor(ax + 10, ay + 2, 2, 2), 0.0))
    return units


def _region_b(rx: int, ry: int) -> list[tuple[int, float, float, float]]:
    return [
        (VILLAGER, rx + 1.5, ry + 5.5, _facing(VILLAGER, 2)),
        (OAK, rx + 6.5, ry + 1.5, 5.0),
        (OAK, rx + 6.5, ry + 2.5, 11.0),
        (HOUSE, *render.span_anchor(rx + 4, ry + 5, 2, 2), 0.0),
    ]


def _paint_region(mm, rx: int, ry: int) -> None:
    """A Road L along the region's top and left edges, a Dirt stripe, and a
    corner one level above the area's own."""
    base = mm.terrain[ry * mm.map_width + rx].elevation
    for y in range(ry, ry + REGION):
        for x in range(rx, rx + REGION):
            tile = mm.terrain[y * mm.map_width + x]
            if y == ry or x == rx:
                tile.terrain_id = ROAD
            elif x == rx + 3:
                tile.terrain_id = DIRT
    corner = [(x, y, base + 1) for y in range(ry + 5, ry + REGION) for x in range(rx + 5, rx + REGION)]
    set_tiles_elevation(mm, corner)


def _first_units(members) -> list[region_transform.FirstUnit]:
    return [region_transform.FirstUnit(i, u.unit_const, u.x, u.y, u.rotation) for i, u in enumerate(members)]


def _write(model, members, plan) -> None:
    """The viewer's _write_lift_units() order: gate sibling, position, facing or wall index."""
    for image in plan.images:
        unit = members[image.key]
        if image.unit_const != unit.unit_const:
            model.set_unit_const(unit, image.unit_const)
        model.set_position(unit, image.x, image.y, unit.z)
        if image.rotation_write is not None:
            kind, value = image.rotation_write
            if kind == region_transform.WRITE_ANGLE:
                model.set_rotation(unit, value)
            else:
                model.set_wall_variant(unit, int(value))


def _check_on_map(plan, mm) -> None:
    for image in plan.images:
        if not region_transform.image_on_map(image.unit_const, image.x, image.y, mm.map_width, mm.map_height):
            raise SystemExit(f"an image lands off the map: {image}")


def _describe(label: str, members, before) -> None:
    print(label)
    for unit, (const, x, y, rotation) in zip(members, before, strict=True):
        print(
            f"  ref {unit.reference_id:6d}  const {const:4d} -> {unit.unit_const:4d}  "
            f"({x:7.2f}, {y:7.2f}) -> ({unit.x:7.2f}, {unit.y:7.2f})  rotation {rotation:.4f} -> {unit.rotation:.4f}"
        )


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    before_path, after_path = out / "before.aoe2scenario", out / "after.aoe2scenario"

    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    ax, ay = find_area(loaded)
    rx, ry = ax + REGION_X, ay + REGION_Y
    _paint_region(mm, rx, ry)
    model = UnitEditModel(loaded)
    history = EditHistory()
    model.begin_unit_edit([PLAYER])
    group = [model.add(PLAYER, c, x, y, rotation=r) for c, x, y, r in _group_a(ax, ay)]
    region_units = [model.add(PLAYER, c, x, y, rotation=r) for c, x, y, r in _region_b(rx, ry)]
    model.commit_unit_edit("TASK-256 pair layout", history)
    write_scenario(loaded, before_path, backup=False, units=model)

    everything = [u for units in loaded.unit_manager.units for u in units]
    file_radian = walls_are_radian(everything)
    group_before = [(u.unit_const, u.x, u.y, u.rotation) for u in group]
    region_before = [(u.unit_const, u.x, u.y, u.rotation) for u in region_units]

    # Group A: Flip Horizontal (screen left/right in the isometric views, as in the game).
    rect_a = units_region(group, mm.map_width, mm.map_height)
    flip = region_transform.plan_in_place(_first_units(group), region_transform.FLIP_HORIZONTAL, rect_a, file_radian)
    _check_on_map(flip, mm)

    # Region B: Rotate Group clockwise, tiles and units together.
    rect_b = (rx, ry, rx + REGION, ry + REGION)
    tiles = dataclasses.replace(region_clipboard.copy_region(mm, loaded.unit_manager, *rect_b), units=())
    turn = region_transform.plan_in_place(
        _first_units(region_units), region_transform.ROTATE_CW, rect_b, file_radian, tiles=tiles
    )
    _check_on_map(turn, mm)

    model.begin_unit_edit([PLAYER], fields_only=True)
    _write(model, group, flip)
    _write(model, region_units, turn)
    model.commit_unit_edit("TASK-256 flip and turn", history)
    tx0, ty0 = turn.rect[0], turn.rect[1]
    region_clipboard.paste_terrain(mm, turn.block, tx0, ty0, carry_layer=loaded.terrain_has_layer)
    set_tiles_elevation(mm, region_clipboard.elevation_targets(turn.block, tx0, ty0, mm.map_width, mm.map_height))
    write_scenario(loaded, after_path, backup=False, units=model)

    print(f"wrote {before_path}\nwrote {after_path}\n")
    print(f"Source {source.name}, all player {PLAYER}. Walls are {'radian' if file_radian else 'integer'}-encoded.")
    print(f"Group A, footprint box tiles x {rect_a[0]}..{rect_a[2] - 1}, y {rect_a[1]}..{rect_a[3] - 1}, flipped")
    print("horizontally on screen: the Knight's facing mirrors left/right, the Gate takes its other")
    print("diagonal orientation (a different object type), and the walls beside it their other run.")
    _describe("Group A (const, position, rotation):", group, group_before)
    print(f"\nRegion B, tiles x {rx}..{rx + REGION - 1}, y {ry}..{ry + REGION - 1}, turned a quarter turn")
    print("clockwise on screen: its Road L, Dirt stripe and raised corner turn with it, the")
    print("Villager's facing turns a quarter, the Oaks keep their variants.")
    _describe("Region B units:", region_units, region_before)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
