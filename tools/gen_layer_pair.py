#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming layered terrain
(the in-game editor's Layering Mode) in the real AoE2:DE editor: the half of
testing nothing in this session can drive.

Every edit goes through the code the terrain tools use (terrain_target's
write_tile, fill_tools' flood_fill, then scenario_write), never a byte patch.
The edited copy is reloaded and every tile checked before the checklist
prints, so a mismatch here is a write-path bug, not something to look for in
the game.

Deliberately asymmetric, so a mirrored, shifted or dropped write shows:
an L of layered terrain against the west edge with a small hole off-centre
in it, a rectangle layered on its west half only, a strip written with its
own base terrain (which must stay unlayered), a row of 3x3 patches of the
pairs real scenarios use most, and a row of pairs no measured scenario has.

Writes into build/layer_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import terrain_blend
from descape.fill_tools import flood_fill
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_target import CLEAR, LAYER, TERRAIN, write_tile

FIXTURE = ROOT / "tests" / "fixtures" / "real_blank_240x240.aoe2scenario"
OUT_DIR = ROOT / "build" / "layer_pair"

GRASS_1, WATER_SHALLOW, BEACH, DIRT_3, SHALLOWS, DIRT_1 = 0, 1, 2, 3, 4, 6
FOREST_OAK, DIRT_2, FOREST_JUNGLE, WATER_DEEP, WATER_MEDIUM = 10, 11, 17, 22, 23
ROAD_BROKEN, SNOW = 25, 32

# An L against the west edge: a vertical arm at x 0..5 and a horizontal arm
# running east from its foot. x starts at 0 on purpose, the clipping case.
L_ARM_V = [(x, y) for y in range(20, 50) for x in range(6)]
L_ARM_H = [(x, y) for y in range(44, 50) for x in range(6, 30)]
L_SHAPE = L_ARM_V + L_ARM_H
# A 3x1 hole in the vertical arm, off-centre on both axes.
L_HOLE = [(2, 27), (3, 27), (4, 27)]

# A DIRT_1 rectangle, layered with DIRT_3 on its west half only.
RECT_X, RECT_Y = range(40, 56), range(20, 32)
RECT_WEST_X = range(40, 48)

# Written with its own base terrain: must read back, and look, unlayered.
OWN_BASE_STRIP = [(x, 40) for x in range(40, 56)]

PATCH_X0, PATCH_STEP, PATCH_SIZE = 20, 8, 3
TOP_ROW_Y, NOVEL_ROW_Y = 70, 80
# The six most common (base, layer) pairs across the 21 example scenarios
# (measured 2026-10-12: 139,604 layered tiles, 800 distinct pairs).
TOP_PAIRS = [
    (WATER_DEEP, WATER_MEDIUM),
    (WATER_SHALLOW, WATER_MEDIUM),
    (ROAD_BROKEN, DIRT_1),
    (DIRT_1, DIRT_3),
    (DIRT_1, DIRT_2),
    (GRASS_1, DIRT_1),
]
# Pairs none of those scenarios has, each with what it samples.
NOVEL_PAIRS = [
    (GRASS_1, WATER_DEEP, "water layered over land"),
    (WATER_DEEP, GRASS_1, "land layered over water"),
    (DIRT_1, FOREST_OAK, "a forest terrain as the layer"),
    (SNOW, BEACH, "a beach as the layer"),
    (GRASS_1, SHALLOWS, "shallows as the layer"),
    (GRASS_1, FOREST_JUNGLE, "a layer with no mask of its own (the fallback mask)"),
]


def _patch_tiles(column: int, y0: int) -> list[tuple[int, int]]:
    x0 = PATCH_X0 + column * PATCH_STEP
    return [(x, y) for y in range(y0, y0 + PATCH_SIZE) for x in range(x0, x0 + PATCH_SIZE)]


def _span(tiles: list[tuple[int, int]]) -> str:
    xs, ys = [x for x, _y in tiles], [y for _x, y in tiles]
    return f"x {min(xs)}..{max(xs)}, y {min(ys)}..{max(ys)}"


def _pair_text(base: int, layer: int) -> str:
    if layer == -1:
        return f"{name_for_terrain_id(base)} ({base}), no layer"
    return (
        f"{name_for_terrain_id(layer)} ({layer}) layered over {name_for_terrain_id(base)} ({base}), "
        f"mask {terrain_blend.layer_mask_name(layer)}"
    )


def apply_edits(mm) -> tuple[dict[tuple[int, int], tuple[int, int]], list[str]]:
    """Makes every edit on `mm`. Returns the (terrain_id, layer) each edited
    tile must now hold, and the checklist lines naming every patch."""
    width = mm.map_width
    expected: dict[tuple[int, int], tuple[int, int]] = {}
    lines: list[str] = []

    def tile(x: int, y: int):
        return mm.terrain[y * width + x]

    def note(title: str, tiles: list[tuple[int, int]], base: int, layer: int) -> None:
        for xy in tiles:
            expected[xy] = (base, layer)
        lines.append(f"{title}: {_span(tiles)} ({len(tiles)} tiles) = {_pair_text(base, layer)}")

    # 1. The L, then its hole.
    for x, y in L_SHAPE:
        write_tile(tile(x, y), LAYER, DIRT_1)
    for x, y in L_HOLE:
        write_tile(tile(x, y), CLEAR, DIRT_1)
    note("L, long arm down the west edge (less the hole below)", L_ARM_V, GRASS_1, DIRT_1)
    note("L, foot running east", L_ARM_H, GRASS_1, DIRT_1)
    note("hole in the L's long arm", L_HOLE, GRASS_1, -1)

    # 2. The rectangle. Its east half takes a throwaway layer first, so the
    # Layer fill stops at the half line by the pair rule, and a Clear layer
    # fill then removes the throwaway one.
    rect = [(x, y) for y in RECT_Y for x in RECT_X]
    west = [(x, y) for y in RECT_Y for x in RECT_WEST_X]
    east = [xy for xy in rect if xy not in set(west)]
    for x, y in rect:
        write_tile(tile(x, y), TERRAIN, DIRT_1)
    for x, y in east:
        write_tile(tile(x, y), LAYER, SNOW)
    filled = flood_fill(mm, RECT_WEST_X[0], RECT_Y[0], LAYER, DIRT_3)
    cleared = flood_fill(mm, RECT_X[-1], RECT_Y[-1], CLEAR, DIRT_3)
    if len(filled) != len(west) or len(cleared) != len(east):
        raise RuntimeError(f"rectangle fills covered {len(filled)} and {len(cleared)} tiles, not its two halves")
    note("rectangle, west half", west, DIRT_1, DIRT_3)
    note("rectangle, east half", east, DIRT_1, -1)

    # 3. A layer equal to the tile's own base is written as none.
    for x, y in OWN_BASE_STRIP:
        write_tile(tile(x, y), LAYER, tile(x, y).terrain_id)
    note("strip written with its own base", OWN_BASE_STRIP, GRASS_1, -1)

    # 4 and 5. The two rows of 3x3 patches.
    for column, (base, layer) in enumerate(TOP_PAIRS):
        patch = _patch_tiles(column, TOP_ROW_Y)
        for x, y in patch:
            write_tile(tile(x, y), TERRAIN, base)
            write_tile(tile(x, y), LAYER, layer)
        note(f"common pair {column + 1}", patch, base, layer)
    for column, (base, layer, what) in enumerate(NOVEL_PAIRS):
        patch = _patch_tiles(column, NOVEL_ROW_Y)
        for x, y in patch:
            write_tile(tile(x, y), TERRAIN, base)
            write_tile(tile(x, y), LAYER, layer)
        note(f"unusual pair {column + 1}, {what}", patch, base, layer)
    return expected, lines


def main() -> int:
    if not FIXTURE.is_file():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_path = OUT_DIR / "before_blank_240x240.aoe2scenario"
    scenario = load_map_and_units(FIXTURE)
    write_scenario(scenario, before_path, backup=False)
    print(f"wrote {before_path}")

    scenario = load_map_and_units(FIXTURE)
    if not scenario.terrain_has_layer:
        print("the fixture's terrain has no layer field", file=sys.stderr)
        return 1
    mm = scenario.map_manager
    untouched = {(t.terrain_id, t.layer) for t in mm.terrain}
    if untouched != {(GRASS_1, -1)}:
        print(f"the fixture is not plain unlayered GRASS_1: {sorted(untouched)}", file=sys.stderr)
        return 1
    expected, lines = apply_edits(mm)

    after_path = OUT_DIR / "after_layered_terrain.aoe2scenario"
    write_scenario(scenario, after_path, backup=False)
    print(f"wrote {after_path}")

    # Read back through the real load path: every tile, edited or not.
    reloaded = load_map_and_units(after_path).map_manager
    width = reloaded.map_width
    wrong = [
        (i % width, i // width, (t.terrain_id, t.layer))
        for i, t in enumerate(reloaded.terrain)
        if (t.terrain_id, t.layer) != expected.get((i % width, i // width), (GRASS_1, -1))
    ]
    layered = sum(1 for t in reloaded.terrain if t.layer != -1)
    want_layered = sum(1 for _base, layer in expected.values() if layer != -1)
    print()
    print(f"reloaded: {layered} layered tiles (expected {want_layered}), {len(wrong)} tiles differ from the plan")
    if wrong or layered != want_layered:
        for x, y, got in wrong[:10]:
            print(f"  ({x}, {y}): got {got}, expected {expected.get((x, y), (GRASS_1, -1))}", file=sys.stderr)
        return 1

    print()
    print("Patches (tile coordinates as DEscape's hover readout shows them):")
    for number, line in enumerate(lines, 1):
        print(f"  {number:2d}. {line}")
    print()
    print("What to check in the AoE2:DE editor, opening the 'after' file:")
    print("  1. The file loads with no error.")
    print("  2. Each patch above shows where listed. The L runs down the west edge")
    print("     and turns east at its foot, with a 3-tile gap of plain grass in its")
    print("     long arm. Only the rectangle's west half is layered.")
    print("  3. The own-base strip (patch 6) looks like the plain grass around it.")
    print("  4. Paint one of the common pairs beside its patch with the game's own")
    print("     Layering Mode: the two should look the same.")
    print("  5. Erase > Layered Terrain over a patch removes its layer and leaves")
    print("     the base terrain.")
    print("  6. Save the file from the game under a new name and open it in DEscape:")
    print("     hovering the patches shows the same layer= names as the list above.")
    print("  7. The 'before' file is the same map with none of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
