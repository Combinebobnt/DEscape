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

import pytest
from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import library_compat, region_clipboard
from descape.elevation_tools import set_tiles_elevation
from descape.region_clipboard import (
    RegionBlock,
    RegionUnit,
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


def _add_raw_unit(loaded, player: int, x: float, y: float, reference_id: int, garrisoned_in_id: int, unit_const: int = 83) -> Unit:
    """Appends a Unit with a caller-chosen reference_id, bypassing
    UnitEditModel.add()'s own id generator -- needed to construct the -1
    sentinel and duplicate-reference_id cases, neither of which add() can
    produce."""
    unit = Unit(
        player=player,
        x=x,
        y=y,
        z=0.0,
        reference_id=reference_id,
        unit_const=unit_const,
        status=2,
        rotation=0.0,
        initial_animation_frame=0,
        garrisoned_in_id=garrisoned_in_id,
        caption_string_id=-1,
        caption_string="",
        uuid=loaded._scenario.uuid,
    )
    loaded.unit_manager.units[player].append(unit)
    return unit


def _ru(dx: float, dy: float, garrison_slot: int, unit_const: int = 83) -> RegionUnit:
    return RegionUnit(
        player=0,
        unit_const=unit_const,
        dx=dx,
        dy=dy,
        z=0.0,
        rotation=0.0,
        status=2,
        initial_animation_frame=0,
        caption_string_id=-1,
        caption_string="",
        garrison_slot=garrison_slot,
    )


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


def _construct_unit_while_poisoned(uuid) -> Unit:
    """Reproduces UnitManager.construct()'s own path while Unit is poisoned:
    caption_string_id/caption_string are passed as None, which the real
    disabled setter silently swallows rather than raising (only a non-None
    value raises), leaving the instance with no such attribute at all --
    unlike UnitEditModel.add(), which always passes -1/"" and so always hits
    the raise. depoison() afterward removes the class-level property, so the
    later read falls through to a missing instance attribute: plain
    AttributeError, not UnsupportedAttributeError."""

    def _raise(self_, val=None):
        if val is not None:
            raise UnsupportedAttributeError("synthetic poisoning for a test")

    Unit.caption_string_id = property(_raise, _raise)
    Unit.caption_string = property(_raise, _raise)
    return Unit(
        player=1,
        x=12.5,
        y=12.5,
        z=0.0,
        reference_id=99999,
        unit_const=83,
        status=2,
        rotation=0.0,
        initial_animation_frame=0,
        garrisoned_in_id=-1,
        caption_string_id=None,
        caption_string=None,
        uuid=uuid,
    )


def test_copy_region_survives_a_unit_parsed_while_poisoned() -> None:
    """The second live gap the 2026-09-12 plan recorded, reachable today via
    parse_triggers()'s own depoison(): copy_region()'s defensive read must
    catch AttributeError as well as UnsupportedAttributeError."""
    loaded = _load()
    unit = _construct_unit_while_poisoned(loaded._scenario.uuid)
    loaded.unit_manager.units[1].append(unit)
    try:
        library_compat.depoison()
        with pytest.raises(AttributeError):
            _ = unit.caption_string

        block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 15, 15)
        assert len(block.units) == 1
        assert block.units[0].caption_string == ""
        assert block.units[0].caption_string_id == -1
    finally:
        library_compat.depoison()


def test_unit_paste_targets_translates_and_drops_off_map() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    edits.add(player=1, unit_const=83, x=10.5, y=10.5, z=0.0, rotation=0.0)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)
    assert len(block.units) == 1

    on_map = unit_paste_targets(block, 50, 50, loaded.map_manager.map_width, loaded.map_manager.map_height)
    assert len(on_map) == 1
    assert (on_map[0].x, on_map[0].y) == (50.5, 50.5)

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
    [target] = unit_paste_targets(block, 40, 40, 120, 120)
    assert target.unit.rotation == 37.0


# -- garrison remap: copy_region()'s slot resolution ------------------------


def test_copy_region_resolves_garrison_slot_for_intra_region_holder() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    holder = edits.add(player=1, unit_const=79, x=10.5, y=10.5, z=0.0, rotation=0.0)
    edits.add(player=1, unit_const=4, x=10.5, y=10.5, z=0.0, rotation=0.0, garrisoned_in_id=holder.reference_id)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    holder_ru = next(u for u in block.units if u.unit_const == 79)
    occupant_ru = next(u for u in block.units if u.unit_const == 4)
    assert occupant_ru.garrison_slot == block.units.index(holder_ru)


def test_copy_region_holder_outside_rect_gives_minus_one() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    holder = edits.add(player=1, unit_const=79, x=5.5, y=5.5, z=0.0, rotation=0.0)
    edits.add(player=1, unit_const=4, x=10.5, y=10.5, z=0.0, rotation=0.0, garrisoned_in_id=holder.reference_id)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    assert len(block.units) == 1
    assert block.units[0].garrison_slot == -1


def test_copy_region_ungarrisoned_unit_ignores_minus_one_sentinel_collision() -> None:
    loaded = _load()
    # Unit A's own reference_id is the -1 junk sentinel -- a naive dict
    # lookup on garrisoned_in_id == -1 would match it. Unit B is plainly
    # ungarrisoned and must resolve to -1 without ever consulting that entry.
    _add_raw_unit(loaded, player=0, x=10.5, y=10.5, reference_id=-1, garrisoned_in_id=-1)
    _add_raw_unit(loaded, player=0, x=10.5, y=10.5, reference_id=100, garrisoned_in_id=-1)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    assert len(block.units) == 2
    assert all(u.garrison_slot == -1 for u in block.units)


def test_copy_region_duplicate_reference_id_among_captured_gives_minus_one() -> None:
    loaded = _load()
    _add_raw_unit(loaded, player=0, x=10.5, y=10.5, reference_id=555, garrisoned_in_id=-1)
    _add_raw_unit(loaded, player=0, x=10.5, y=10.5, reference_id=555, garrisoned_in_id=-1)
    _add_raw_unit(loaded, player=0, x=10.5, y=10.5, reference_id=200, garrisoned_in_id=555)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    occupant_ru = block.units[2]  # the unit constructed with garrisoned_in_id=555
    assert occupant_ru.garrison_slot == -1


def test_copy_region_cross_player_garrison_resolves() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    holder = edits.add(player=1, unit_const=79, x=10.5, y=10.5, z=0.0, rotation=0.0)
    edits.add(player=0, unit_const=83, x=10.5, y=10.5, z=0.0, rotation=0.0, garrisoned_in_id=holder.reference_id)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    holder_ru = next(u for u in block.units if u.unit_const == 79)
    occupant_ru = next(u for u in block.units if u.unit_const == 83)
    assert occupant_ru.garrison_slot == block.units.index(holder_ru)


def test_copy_region_two_occupants_share_one_holder() -> None:
    loaded = _load()
    edits = UnitEditModel(loaded)
    holder = edits.add(player=1, unit_const=79, x=10.5, y=10.5, z=0.0, rotation=0.0)
    edits.add(player=1, unit_const=4, x=10.5, y=10.5, z=0.0, rotation=1.0, garrisoned_in_id=holder.reference_id)
    edits.add(player=1, unit_const=4, x=10.5, y=10.5, z=0.0, rotation=2.0, garrisoned_in_id=holder.reference_id)
    block = copy_region(loaded.map_manager, loaded.unit_manager, 10, 10, 11, 11)

    holder_slot = block.units.index(next(u for u in block.units if u.unit_const == 79))
    occupant_slots = {u.garrison_slot for u in block.units if u.unit_const == 4}
    assert occupant_slots == {holder_slot}


# -- garrison remap: unit_paste_targets()'s ordering -------------------------


def test_unit_paste_targets_orders_holder_before_occupant_three_deep_chain() -> None:
    a = _ru(dx=0, dy=0, garrison_slot=1)  # in B
    b = _ru(dx=0, dy=0, garrison_slot=2)  # in C
    c = _ru(dx=0, dy=0, garrison_slot=-1)
    block = RegionBlock(width=1, height=1, terrain_ids=(0,), elevations=(0,), layers=(-1,), units=(a, b, c))

    targets = unit_paste_targets(block, 0, 0, 10, 10)
    assert [t.slot for t in targets] == [2, 1, 0]
    by_slot = {t.slot: t.holder_slot for t in targets}
    assert by_slot == {2: -1, 1: 2, 0: 1}


def test_unit_paste_targets_breaks_mutual_cycle_to_minus_one() -> None:
    a = _ru(dx=0, dy=0, garrison_slot=1)
    b = _ru(dx=0, dy=0, garrison_slot=0)
    block = RegionBlock(width=1, height=1, terrain_ids=(0,), elevations=(0,), layers=(-1,), units=(a, b))

    targets = unit_paste_targets(block, 0, 0, 10, 10)
    assert {t.slot for t in targets} == {0, 1}
    assert all(t.holder_slot == -1 for t in targets)


def test_unit_paste_targets_holder_slot_minus_one_when_holder_clipped_but_occupant_isnt() -> None:
    # Impossible on a real file (measured: occupant always shares the
    # holder's exact tile) -- only reachable via a hand-built RegionBlock.
    holder = _ru(dx=0, dy=0, garrison_slot=-1)
    occupant = _ru(dx=10, dy=0, garrison_slot=0)
    block = RegionBlock(
        width=11, height=1, terrain_ids=(0,) * 11, elevations=(0,) * 11, layers=(-1,) * 11, units=(holder, occupant)
    )

    targets = unit_paste_targets(block, -5, 0, 20, 20)
    assert [t.slot for t in targets] == [1]
    assert targets[0].holder_slot == -1


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


# -- translated_region (the move overlay's rect) -----------------------------


def test_translated_region_shifts_in_both_directions() -> None:
    assert region_clipboard.translated_region((2, 3, 6, 7), 4, 5, 40, 40) == (6, 8, 10, 12)
    assert region_clipboard.translated_region((6, 8, 10, 12), -4, -5, 40, 40) == (2, 3, 6, 7)


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (-4, 0, (0, 3, 2, 7)),  # clipped at the west edge
        (0, -5, (2, 0, 6, 2)),  # north
        (36, 0, (38, 3, 40, 7)),  # east
        (0, 35, (2, 38, 6, 40)),  # south
    ],
)
def test_translated_region_clamps_by_intersection_at_each_edge(dx, dy, expected) -> None:
    """Clamped rather than refused, exactly as normalize_region() clamps: a
    move legitimately drags off the map edge."""
    assert region_clipboard.translated_region((2, 3, 6, 7), dx, dy, 40, 40) == expected


@pytest.mark.parametrize(("dx", "dy"), [(-10, 0), (0, -10), (40, 0), (0, 40)])
def test_translated_region_is_none_when_fully_off_map(dx, dy) -> None:
    assert region_clipboard.translated_region((2, 3, 6, 7), dx, dy, 40, 40) is None
