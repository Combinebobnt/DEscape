"""Coverage for descape/region_clipboard.py -- the Select tool's clipboard
(phase 2.8). Pure, no Qt: normalize_region()'s tile-rectangle arithmetic and
copy_region()/paste_terrain()/elevation_targets()/unit_paste_targets()'s data
transforms, against a real MapManager/UnitManager loaded from the shipped
blank template (same fixture tests/test_elevation_tools.py uses).

Verification step 4 of the plan is here (test_elevation_paste_matches_source_
plateau_including_pit): the check that decides whether set_tiles_elevation
was the right call for elevation paste over a raw-assign-plus-seam-sweep --
see elevation_targets()'s own docstring.
"""

from __future__ import annotations

from descape.elevation_tools import set_tiles_elevation
from descape.region_clipboard import (
    copy_region,
    elevation_targets,
    normalize_region,
    paste_terrain,
    unit_paste_targets,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_model import UnitEditModel


def _load():
    return load_map_and_units(BLANK_TEMPLATE_PATH)


# -- normalize_region ---------------------------------------------------


def test_normalize_region_both_drag_directions_agree() -> None:
    forward = normalize_region((10, 10), (13, 15), 120, 120)
    backward = normalize_region((13, 15), (10, 10), 120, 120)
    assert forward == backward == (10, 10, 14, 16)


def test_normalize_region_press_release_no_movement_selects_one_tile() -> None:
    assert normalize_region((5, 5), (5, 5), 120, 120) == (5, 5, 6, 6)


def test_normalize_region_clamps_by_intersection_off_map() -> None:
    # Anchor on-map, drag endpoint runs off the top-left and bottom-right --
    # both clamp rather than refuse.
    assert normalize_region((-5, -5), (3, 3), 120, 120) == (0, 0, 4, 4)
    assert normalize_region((115, 115), (200, 200), 120, 120) == (115, 115, 120, 120)


def test_normalize_region_wholly_off_map_is_none() -> None:
    assert normalize_region((-10, -10), (-5, -5), 120, 120) is None


# -- copy_region / paste_terrain round trip ------------------------------


def test_round_trip_at_same_origin_is_a_no_op() -> None:
    loaded = _load()
    mm = loaded.map_manager
    tx0, ty0, tx1, ty1 = 10, 10, 15, 14
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            mm.terrain[y * mm.map_width + x].terrain_id = (x + y) % 7

    block = copy_region(mm, loaded.unit_manager, tx0, ty0, tx1, ty1)
    before = [t.terrain_id for t in mm.terrain]
    paste_terrain(mm, block, tx0, ty0)
    after = [t.terrain_id for t in mm.terrain]
    assert before == after


def test_paste_near_map_edge_writes_the_part_that_fits() -> None:
    loaded = _load()
    mm = loaded.map_manager
    tx0, ty0, tx1, ty1 = 0, 0, 4, 4
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            mm.terrain[y * mm.map_width + x].terrain_id = 3
    block = copy_region(mm, loaded.unit_manager, tx0, ty0, tx1, ty1)

    # Paste anchored so half the block runs off the bottom-right edge.
    dest_x, dest_y = mm.map_width - 2, mm.map_height - 2
    paste_terrain(mm, block, dest_x, dest_y)
    for y in range(dest_y, min(mm.map_height, dest_y + block.height)):
        for x in range(dest_x, min(mm.map_width, dest_x + block.width)):
            assert mm.terrain[y * mm.map_width + x].terrain_id == 3
    # Nothing beyond the map edge was touched (would have raised on write,
    # so reaching here at all is most of the assertion).


def test_paste_resets_layer_to_minus_one() -> None:
    loaded = _load()
    mm = loaded.map_manager
    tx0, ty0, tx1, ty1 = 20, 20, 22, 22
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            tile = mm.terrain[y * mm.map_width + x]
            tile.terrain_id = 5
            tile.layer = 9
    block = copy_region(mm, loaded.unit_manager, tx0, ty0, tx1, ty1)

    dest_x, dest_y = 60, 60
    for y in range(dest_y, dest_y + 2):
        for x in range(dest_x, dest_x + 2):
            mm.terrain[y * mm.map_width + x].layer = 3
    paste_terrain(mm, block, dest_x, dest_y)
    for y in range(dest_y, dest_y + 2):
        for x in range(dest_x, dest_x + 2):
            assert mm.terrain[y * mm.map_width + x].layer == -1


# -- units ----------------------------------------------------------------


def test_copy_region_collects_units_by_direct_walk() -> None:
    # The regression this plan's direct-walk decision exists to prevent:
    # collecting units must not depend on unit_pick's on-demand index, which
    # is None while in Terrain mode (where the Select tool lives).
    loaded = _load()
    edits = UnitEditModel(loaded)
    edits.add(player=1, unit_const=83, x=12.5, y=12.5, z=0.0, rotation=1.5)
    edits.add(player=0, unit_const=83, x=50.5, y=50.5, z=0.0, rotation=0.0)  # outside the region

    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 15, 15)
    assert len(block.units) == 1
    u = block.units[0]
    assert u.player == 1
    assert u.dx == 2.5 and u.dy == 2.5


def test_unit_paste_targets_translates_and_drops_off_map() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    edits.add(player=1, unit_const=83, x=10.5, y=10.5, z=0.0, rotation=0.0)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)
    assert len(block.units) == 1

    on_map = unit_paste_targets(block, 50, 50, loaded.map_manager.map_width, loaded.map_manager.map_height)
    assert len(on_map) == 1
    _, x, y = on_map[0]
    assert (x, y) == (50.5, 50.5)

    off_map = unit_paste_targets(
        block, loaded.map_manager.map_width - 0, 0, loaded.map_manager.map_width, loaded.map_manager.map_height
    )
    assert off_map == []


def test_rotation_is_never_transformed_by_copy_or_paste() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    # An arbitrary, non-angle-looking value -- copy/paste must pass it
    # through byte-for-byte, per AGENTS.md's hard rule.
    edits.add(player=0, unit_const=349, x=30.5, y=30.5, z=0.0, rotation=37.0)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 30, 30, 31, 31)
    assert block.units[0].rotation == 37.0
    [(unit, _x, _y)] = unit_paste_targets(block, 40, 40, 120, 120)
    assert unit.rotation == 37.0


# -- elevation: verification step 4 (the plan's design gate) ---------------


def test_elevation_paste_matches_source_plateau_including_pit() -> None:
    """Copies a 3-level plateau with a one-tile pit onto flat ground, then
    checks (a) the pasted footprint is byte-exact to the source, pit
    included, and (b) an 8-connected sweep over the whole map finds no
    abs(delta) > 1 pair. If either fails, elevation_targets()/
    set_tiles_elevation() was the wrong call and Stage 3's elevation arm
    must fall back to raw-assign plus a seam sweep instead (see the plan's
    Risks section)."""
    loaded = _load()
    mm = loaded.map_manager

    sx0, sy0 = 30, 30
    size = 5
    # Built the same way a real user would (three brush strokes, each
    # legalized by set_tiles_elevation's own propagation) rather than a
    # hand-authored elevation matrix -- a directly-assigned matrix can encode
    # a jump the in-game editor could never produce, which would make this
    # test's own fixture illegal before any copy/paste touches it.
    outer = [(sx0 + x, sy0 + y, 1) for y in range(size) for x in range(size)]
    set_tiles_elevation(mm, outer)
    inner = [(sx0 + x, sy0 + y, 2) for y in range(1, size - 1) for x in range(1, size - 1)]
    set_tiles_elevation(mm, inner)
    top = [(sx0 + 2, sy0 + 2, 3)]
    set_tiles_elevation(mm, top)
    # The one-tile pit: a single-tile dip back to the middle level, placed
    # with the same single-target call a real "lower elevation" click makes.
    set_tiles_elevation(mm, [(sx0 + 2, sy0 + 2, 2)])

    plateau = [[mm.get_tile(sx0 + x, sy0 + y).elevation for x in range(size)] for y in range(size)]

    block = copy_region(mm, loaded.unit_manager, sx0, sy0, sx0 + size, sy0 + size)

    dx0, dy0 = 60, 60
    et = elevation_targets(block, dx0, dy0, mm.map_width, mm.map_height)
    set_tiles_elevation(mm, et)

    for y in range(size):
        for x in range(size):
            got = mm.get_tile(dx0 + x, dy0 + y).elevation
            assert got == plateau[y][x], f"mismatch at relative ({x}, {y}): {got} != {plateau[y][x]}"

    for y in range(mm.map_height):
        for x in range(mm.map_width):
            e = mm.get_tile(x, y).elevation
            for ddx, ddy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                nx, ny = x + ddx, y + ddy
                if 0 <= nx < mm.map_width and 0 <= ny < mm.map_height:
                    ne = mm.get_tile(nx, ny).elevation
                    assert abs(e - ne) <= 1, f"illegal jump between ({x}, {y}) and ({nx}, {ny}): {e} vs {ne}"
