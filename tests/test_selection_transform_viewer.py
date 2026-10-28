"""Flip and turn a selection in place (TASK-256), through a real offscreen
ViewerWindow: Edit > Flip Selection and Rotate Selection's Rotate Group pair on
a Units-mode selection and on a committed Select region, plus the top-down
Flat flip fix for a pasted region.

Built on tests/test_region_rotate.py's helpers. Every ViewerWindow() here must
call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest
from test_region_rotate import _add, _close, _last_status, _paste, _setup, _tiles
from test_region_select import _select_window

from descape import gate_orientation, region_transform, render, unit_model, unit_pick
from descape.mirror_tools import compose, facing_image
from descape.scatter_dialog import units_region

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

ARCHER = 4  # ANGLE, 16 facings
VILLAGER = 83
HOUSE = 70  # 2x2
CASTLE = 82  # 4x4
WALL = 117  # five wall shapes, solid
GATE = 64  # stone gate, (4, 1); its siblings are (4, 4), (1, 4), (4, 4)


def _top_down(window) -> None:
    """Top-down Flat: Flat with Isometric View off (ViewerWindow._render_style "flat")."""
    window.iso_action.setChecked(False)
    assert window._render_style == "flat"


def _units_window():
    window = _select_window()
    window.mode_combo.setCurrentText("Units")
    window._on_tool_selected("select")
    return window


def _select(window, units, player: int = 1) -> None:
    window._selection = [unit_pick.unit_key(player, unit) for unit in units]
    window._refresh_selection_view()


def _group(window):
    """A facing archer, a stone gate in a wall run and a 2x2 house off centre, selected."""
    units = _add(
        window,
        [
            (1, ARCHER, 40.5, 40.5, 1.25, None),
            (1, WALL, 42.5, 44.5, 0.0, None),
            (1, GATE, *render.span_anchor(43, 44, 4, 1), 7.0, None),
            (1, WALL, 47.5, 44.5, 0.0, None),
            (1, HOUSE, *render.span_anchor(46, 39, 2, 2), 0.0, None),
        ],
    )
    _select(window, units)
    return units


def _state(window) -> list:
    """Every unit's every field, with its reference_id, in list order."""
    return [
        (player, unit.reference_id, unit_model.unit_state(unit))
        for player, units in enumerate(window.scenario.unit_manager.units)
        for unit in units
    ]


def _expected(units, element):
    """plan_in_place() for `units` as they stand now, about their footprint box."""
    firsts = [region_transform.FirstUnit(i, u.unit_const, u.x, u.y, u.rotation) for i, u in enumerate(units)]
    return region_transform.plan_in_place(firsts, element, units_region(units, 120, 120), False)


def _key(window, key) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window.map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


def test_flip_horizontal_of_a_paste_in_top_down_flat_mirrors_left_right_on_screen():
    window = _select_window()
    try:
        _top_down(window)
        _setup(window, units=False)
        _paste(window)
        source = window._paste_move.block
        window.flip_horizontal_action.trigger()
        assert _last_status(window) == "Flipped pasted region horizontally"
        assert window._paste_move.element == "mx"
        expected = region_transform.transform_block(source, "mx", False).block
        x0, y0 = window._paste_move.anchor
        mm = window.scenario.map_manager
        got = tuple(
            mm.terrain[(y0 + y) * mm.map_width + x0 + x].terrain_id
            for y in range(expected.height)
            for x in range(expected.width)
        )
        assert got == expected.terrain_ids
        # The block's first row is its own first row reversed: left/right on screen.
        assert got[: expected.width] == tuple(reversed(source.terrain_ids[: source.width]))
        window.flip_vertical_action.trigger()
        assert window._paste_move.element == "r2"
    finally:
        _close(window)


def test_flip_in_the_isometric_views_keeps_the_screen_diagonal():
    window = _select_window()
    try:
        assert window._render_style != "flat"
        _setup(window, units=False)
        _paste(window)
        plain = _tiles(window)
        window.flip_horizontal_action.trigger()
        assert window._paste_move.element == region_transform.FLIP_HORIZONTAL
        assert _tiles(window) != plain
    finally:
        _close(window)


# --- a Units-mode selection, flipped and turned as one group --------------------


def test_flip_moves_the_group_about_its_shared_centre_keeping_identity_and_undo_restores_every_field():
    window = _units_window()
    try:
        units = _group(window)
        archer, wall, gate, _wall2, _house = units
        refs = [u.reference_id for u in units]
        plan = _expected(units, region_transform.FLIP_HORIZONTAL)
        before = _state(window)
        depth = len(window.edit_history.records)
        window.flip_horizontal_action.trigger()
        assert _last_status(window) == "Flipped 5 units horizontally (Enter to apply, Esc to undo)"
        assert len(window.edit_history.records) == depth + 1
        assert [u.reference_id for u in units] == refs
        assert [(u.unit_const, u.x, u.y) for u in units] == [(i.unit_const, i.x, i.y) for i in plan.images]
        assert gate.unit_const == gate_orientation.orientation_siblings(GATE)[2], "the (4, 1) gate became its (1, 4) sibling"
        assert float(wall.rotation) == 1.0, "a run along x became a run along y"
        assert archer.rotation == facing_image(1.25, region_transform.FLIP_HORIZONTAL, 16)
        # A trigger naming the archer still finds it, at its new tile.
        ref = window._unit_reference_index().get(archer.reference_id)
        assert ref is not None and ref.own_tile == (int(archer.x), int(archer.y))
        window.undo()
        assert _state(window) == before
    finally:
        _close(window)


def test_a_run_of_flips_and_turns_is_one_record_whose_undo_restores_the_first_state():
    window = _units_window()
    try:
        _group(window)
        before = _state(window)
        depth = len(window.edit_history.records)
        window.flip_horizontal_action.trigger()
        window.rotate_group_cw_action.trigger()
        window.flip_vertical_action.trigger()
        assert len(window.edit_history.records) == depth + 1
        assert window._live_lift("units").element == compose(
            region_transform.FLIP_VERTICAL, compose(region_transform.ROTATE_CW, region_transform.FLIP_HORIZONTAL)
        )
        assert _last_status(window) == "Flipped 5 units vertically"
        window.undo()
        assert _state(window) == before
        assert window._lift is None
    finally:
        _close(window)


def test_four_group_turns_give_back_the_first_state_exactly_with_the_lift_alive():
    window = _units_window()
    try:
        units = _add(
            window,
            [
                (1, ARCHER, 41.29999923706055, 40.70000076293945, 1.25, None),  # off-grid facing and point
                (1, HOUSE, 44.5, 40.5, 0.0, None),  # off parity
                (1, GATE, *render.span_anchor(40, 43, 4, 1), 7.0, None),
                (1, WALL, 46.5, 42.5, 3.0, None),
            ],
        )
        _select(window, units)
        before = _state(window)
        base = window.edit_history.peek_undo()
        for _ in range(4):
            window.rotate_group_cw_action.trigger()
            assert window._live_lift("units") is not None
        assert _state(window) == before
        assert window.edit_history.peek_undo() is base
        assert window._live_lift("units").element == "id"
    finally:
        _close(window)


def test_enter_keeps_the_result_and_the_next_flip_starts_a_new_record():
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        _group(window)
        depth = len(window.edit_history.records)
        window.flip_horizontal_action.trigger()
        flipped = _state(window)
        _key(window, Qt.Key_Return)
        assert _last_status(window) == "Applied flip"
        assert window._lift is None and _state(window) == flipped
        window.flip_horizontal_action.trigger()
        assert len(window.edit_history.records) == depth + 2
        assert _last_status(window).endswith("(Enter to apply, Esc to undo)")
    finally:
        _close(window)


def test_escape_reverts_and_keeps_the_selection_then_a_second_escape_clears_it():
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        _group(window)
        selection = list(window._selection)
        before = _state(window)
        window.flip_horizontal_action.trigger()
        window.rotate_group_ccw_action.trigger()
        _key(window, Qt.Key_Escape)
        assert _last_status(window) == "Reverted to before the first flip or turn"
        assert _state(window) == before
        assert window._selection == selection and window._lift is None
        _key(window, Qt.Key_Escape)
        assert window._selection == []
    finally:
        _close(window)


@pytest.mark.parametrize("between", ["reselect", "deselect", "nudge"])
def test_a_selection_change_or_another_edit_between_two_flips_makes_two_records(between):
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        units = _group(window)
        keys = list(window._selection)
        depth = len(window.edit_history.records)
        window.flip_horizontal_action.trigger()
        if between == "reselect":
            window.on_marquee_select(keys, Qt.NoModifier)  # the same keys, by a new gesture
        elif between == "deselect":
            window.deselect_units()
            _select(window, units)
        else:
            assert window.on_unit_nudge(1, 0, Qt.NoModifier)
        window.flip_horizontal_action.trigger()
        assert len(window.edit_history.records) == depth + (3 if between == "nudge" else 2)
    finally:
        _close(window)


def test_a_refused_turn_mid_lift_changes_nothing_and_free_placement_lets_it_through():
    window = _units_window()
    try:
        units = _add(window, [(1, HOUSE, *render.span_anchor(40, 40, 2, 2), 0.0, None), (1, ARCHER, 45.5, 40.5, 0.0, None)])
        _house, archer = units
        firsts = [region_transform.FirstUnit(i, u.unit_const, u.x, u.y, u.rotation) for i, u in enumerate(units)]
        rect = units_region(units, 120, 120)
        flip = region_transform.FLIP_HORIZONTAL
        turned = compose(region_transform.ROTATE_CW, flip)
        target = region_transform.plan_in_place(firsts, turned, rect, False).images[1]
        tile = (int(target.x), int(target.y))
        flipped = region_transform.plan_in_place(firsts, flip, rect, False).images
        assert all((int(i.x), int(i.y)) != tile for i in flipped), "the blocker must not block the first flip"
        _add(window, [(2, WALL, tile[0] + 0.5, tile[1] + 0.5, 0.0, None)])
        _select(window, units)
        window.flip_horizontal_action.trigger()
        lift, after_flip = window._lift, _state(window)
        depth, cursor = len(window.edit_history.records), window.edit_history.cursor
        window.rotate_group_cw_action.trigger()
        assert _last_status(window).startswith("Rotate Group: 1 of 2 units can't go there (first: blocked by ")
        assert window._lift is lift and _state(window) == after_flip
        assert (len(window.edit_history.records), window.edit_history.cursor) == (depth, cursor)
        window.unrestricted_check.setChecked(True)
        window.rotate_group_cw_action.trigger()
        assert (int(archer.x), int(archer.y)) == tile
        assert window._live_lift("units").element == turned
        assert len(window.edit_history.records) == depth
    finally:
        window.unrestricted_check.setChecked(False)
        _close(window)


def test_a_turn_off_the_map_is_refused_even_with_free_placement_and_creates_no_lift():
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        _select(window, _add(window, [(1, ARCHER, 50.5, 0.5, 0.0, None), (1, ARCHER, 59.5, 1.5, 0.0, None)]))
        window.unrestricted_check.setChecked(True)
        before = _state(window)
        depth = len(window.edit_history.records)
        window.rotate_group_cw_action.trigger()
        assert _last_status(window) == "Rotate Group: part of the selection would land off the map, move it inward first"
        assert _state(window) == before and len(window.edit_history.records) == depth
        assert window._lift is None
        _key(window, Qt.Key_Escape)  # no lift to revert: today's Escape clears the selection
        assert window._selection == []
    finally:
        window.unrestricted_check.setChecked(False)
        _close(window)


def test_a_gate_is_placed_by_its_turned_siblings_footprint():
    """The (4, 1) gate's turn is a (1, 4) sibling whose column reaches tile
    (41, 38), well clear of the old row at y = 40, where a wall stands."""
    window = _units_window()
    try:
        (gate,) = _add(window, [(1, GATE, *render.span_anchor(40, 40, 4, 1), 7.0, None)])
        _add(window, [(2, WALL, 41.5, 38.5, 0.0, None)])
        _select(window, [gate])
        before = _state(window)
        window.rotate_group_cw_action.trigger()
        assert _last_status(window).startswith("Rotate Group: 1 of 1 units can't go there (first: blocked by ")
        assert _state(window) == before and window._lift is None
    finally:
        _close(window)


def test_an_occupant_rides_onto_its_hosts_new_point_and_one_stored_away_leaves_the_pivot_alone():
    window = _units_window()
    try:
        castle, archer = _add(
            window, [(1, CASTLE, *render.span_anchor(40, 40, 4, 4), 0.0, None), (1, ARCHER, 47.5, 41.5, 0.0, None)]
        )
        (villager,) = _add(window, [(1, VILLAGER, 70.5, 70.5, 0.0, None)])
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Garrison", [1]):
            model.set_garrisoned_in(villager, castle.reference_id)
        _select(window, [castle, archer])
        window.rotate_group_cw_action.trigger()
        assert window._lift.rect == (40, 40, 48, 44), "the pivot is the castle and archer's box alone"
        assert (villager.x, villager.y) == (castle.x, castle.y)
        assert villager.garrisoned_in_id == castle.reference_id
    finally:
        _close(window)


def test_flip_horizontal_of_a_unit_group_in_top_down_flat_mirrors_left_right():
    window = _units_window()
    try:
        _top_down(window)
        units = _group(window)
        plan = _expected(units, "mx")
        window.flip_horizontal_action.trigger()
        assert _last_status(window) == "Flipped 5 units horizontally (Enter to apply, Esc to undo)"
        assert window._live_lift("units").element == "mx"
        assert [(u.x, u.y) for u in units] == [(i.x, i.y) for i in plan.images]
    finally:
        _close(window)


def test_flip_and_rotate_group_are_disabled_with_nothing_selected_in_triggers_mode_and_under_the_data_lock():
    window = _units_window()
    try:
        group = [*window._flip_actions, *window._rotate_group_actions]
        assert not any(action.isEnabled() for action in group)
        _group(window)
        assert all(action.isEnabled() for action in group)
        # An instance attribute, not monkeypatch.undo(), which would drop the autouse config isolation.
        window.game_data_lock_reason = lambda: "Read-only: locked"
        window._update_rotate_enabled()
        assert not any(action.isEnabled() for action in group)
        del window.game_data_lock_reason
        window._update_rotate_enabled()
        assert all(action.isEnabled() for action in group)
        window.mode_combo.setCurrentText("Triggers")
        assert not any(action.isEnabled() for action in group)
    finally:
        _close(window)


def test_per_unit_rotate_still_turns_each_selected_unit_on_its_own_spot():
    window = _units_window()
    try:
        units = _group(window)
        others = [u for u in units if u.unit_const != GATE]  # a gate re-anchors on its new sibling's span
        points = [(u.x, u.y) for u in others]
        window.rotate_cw_action.trigger()
        assert [(u.x, u.y) for u in others] == points
        assert window._lift is None
    finally:
        _close(window)


def test_while_the_place_preview_shows_flip_and_rotate_group_change_nothing():
    window = _units_window()
    try:
        _group(window)
        before = _state(window)
        window._place_ghost_active = lambda: True  # the hover ghost, which offscreen never shows
        window.flip_horizontal_action.trigger()
        assert _last_status(window) == "Flip: nothing changed while placing (Rotate turns the placement preview)"
        window.rotate_group_cw_action.trigger()
        assert _last_status(window).startswith("Rotate Group: nothing changed while placing")
        assert _state(window) == before and window._lift is None
    finally:
        _close(window)


# --- a committed Select region, flipped and turned in place ---------------------

_A, _B, _C = 2, 15, 1  # Beach, Water 2D (shoreless), Water Shallow: present in every DE version
_GRASS = 0  # Grass 1: land a fish is refused on
PERCH = 53  # a 2x2 deep-water fish: refused on land


def _paint(window, rect, pattern) -> None:
    """Terrain by `pattern(x, y)` over the half-open rect."""
    mm = window.scenario.map_manager
    tx0, ty0, tx1, ty1 = rect
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            mm.terrain[y * mm.map_width + x].terrain_id = pattern(x, y)


def _raise(window, targets) -> None:
    from descape.elevation_tools import set_tiles_elevation

    set_tiles_elevation(window.scenario.map_manager, targets)


def _region_tiles(window, rect):
    mm = window.scenario.map_manager
    tx0, ty0, tx1, ty1 = rect
    return [
        (t.terrain_id, t.elevation, t.layer)
        for t in (mm.terrain[y * mm.map_width + x] for y in range(ty0, ty1) for x in range(tx0, tx1))
    ]


def _copy(window, rect):
    from descape import region_clipboard

    return region_clipboard.copy_region(window.scenario.map_manager, window.scenario.unit_manager, *rect)


def test_flip_of_a_square_region_turns_its_tiles_and_units_in_place_as_one_record():
    window = _select_window()
    try:
        rect = (40, 40, 46, 46)
        _paint(window, rect, lambda x, y: (_A, _B, _C)[(x * 7 + y * 3) % 3])
        _raise(window, [(41, 41, 1), (42, 41, 1), (41, 42, 1)])
        archer, house = _add(
            window, [(1, ARCHER, 40.5, 44.5, 1.25, None), (1, HOUSE, *render.span_anchor(43, 41, 2, 2), 0.0, None)]
        )
        refs = [archer.reference_id, house.reference_id]
        window.on_region_selected(rect)
        first = _copy(window, rect)
        before_tiles, before_units = _region_tiles(window, (36, 36, 50, 50)), _state(window)
        depth = len(window.edit_history.records)
        window.flip_vertical_action.trigger()
        assert _last_status(window) == "Flipped region vertically (Enter to apply, Esc to undo)"
        assert len(window.edit_history.records) == depth + 1
        assert window._region == rect
        assert [archer.reference_id, house.reference_id] == refs
        turned = region_transform.transform_block(first, region_transform.FLIP_VERTICAL, False).block
        assert _copy(window, rect) == turned, "a copy of the flipped region is the flipped copy"
        window.undo()
        assert _region_tiles(window, (36, 36, 50, 50)) == before_tiles
        assert _state(window) == before_units
    finally:
        _close(window)


def test_a_non_square_region_turns_about_its_centre_and_leaves_the_tiles_it_vacates():
    window = _select_window()
    try:
        rect = (40, 40, 45, 43)  # 5 x 3; its 3 x 5 turn is (41, 39, 44, 44)
        around = (36, 36, 50, 50)
        _paint(window, (38, 37, 47, 46), lambda x, y: _C)
        _paint(window, rect, lambda x, y: (_A, _B)[(x + 2 * y) % 2])
        _raise(window, [(42, 41, 1)])  # the centre tile, which a quarter turn keeps
        (archer,) = _add(window, [(1, ARCHER, 40.5, 40.5, 0.0, None)])
        window.on_region_selected(rect)
        first = _copy(window, rect)
        start = _region_tiles(window, around)
        mm = window.scenario.map_manager

        def vacated() -> dict:
            tiles = {(x, y): mm.terrain[y * mm.map_width + x] for x in (40, 44) for y in range(40, 43)}
            return {xy: (t.terrain_id, t.elevation) for xy, t in tiles.items()}

        left = vacated()
        window.rotate_cw_action.trigger()
        assert _last_status(window) == "Turned region ↻ (Enter to apply, Esc to undo)"
        assert window._region == (41, 39, 44, 44)
        turned = region_transform.transform_block(first, region_transform.ROTATE_CW, False).block
        assert _copy(window, window._region) == turned
        assert vacated() == left, "a tile the turned region no longer covers kept its terrain and elevation"
        assert (int(archer.x), int(archer.y)) == (43, 39)
        for _ in range(3):
            window.rotate_cw_action.trigger()
        assert window._live_lift("region").element == "id"
        assert _region_tiles(window, around) == start
        assert window._region == rect
    finally:
        _close(window)


def test_escape_after_a_non_square_turn_restores_the_tiles_and_the_first_rectangle():
    from PyQt5.QtCore import Qt

    window = _select_window()
    try:
        rect = (40, 40, 45, 43)
        _paint(window, rect, lambda x, y: (_A, _B, _C)[(x + y) % 3])
        window.on_region_selected(rect)
        start = _region_tiles(window, (36, 36, 50, 50))
        window.rotate_group_cw_action.trigger()
        assert window._region == (41, 39, 44, 44)
        _key(window, Qt.Key_Escape)
        assert _last_status(window) == "Reverted to before the first flip or turn"
        assert _region_tiles(window, (36, 36, 50, 50)) == start
        assert window._region == rect and window._lift is None
        _key(window, Qt.Key_Escape)  # the second Escape clears the region, as it always has
        assert window._region is None
    finally:
        _close(window)


def test_an_elevation_step_at_the_regions_edge_comes_back_through_flips_and_turns():
    window = _select_window()
    try:
        rect = (40, 40, 44, 44)
        _raise(window, [(40, y, 2) for y in range(40, 44)])  # a ridge on the left edge, skirted outward
        window.on_region_selected(rect)
        around = (34, 34, 50, 50)  # the region grown well past the skirt
        start = _region_tiles(window, around)
        mm = window.scenario.map_manager

        def outside() -> dict:
            return {
                (x, y): mm.terrain[y * mm.map_width + x].elevation
                for y in range(around[1], around[3])
                for x in range(around[0], around[2])
                if not (40 <= x < 44 and 40 <= y < 44)
            }

        skirt = outside()
        window.flip_horizontal_action.trigger()
        assert _region_tiles(window, around) != start
        assert outside() != skirt, "the ridge's new skirt reached outside the region"
        window.flip_horizontal_action.trigger()
        assert _region_tiles(window, around) == start
        for _ in range(4):
            window.rotate_group_ccw_action.trigger()
        assert _region_tiles(window, around) == start
        window.rotate_group_ccw_action.trigger()
        window.undo()
        assert _region_tiles(window, around) == start
    finally:
        _close(window)


def test_a_fish_flips_with_its_water_and_is_not_refused_by_the_land_it_leaves_for():
    window = _select_window()
    try:
        rect = (40, 40, 46, 46)
        _paint(window, rect, lambda x, y: _C if x < 42 else _GRASS)  # water in the two left columns
        (fish,) = _add(window, [(0, PERCH, *render.span_anchor(40, 42, 2, 2), 0.0, None)])
        window.on_region_selected(rect)
        window.flip_horizontal_action.trigger()
        assert _last_status(window) == "Flipped region horizontally (Enter to apply, Esc to undo)"
        mm = window.scenario.map_manager
        tiles = render.occupied_tiles_for(PERCH, fish.x, fish.y, mm.map_width, mm.map_height)
        assert tiles and all(mm.terrain[y * mm.map_width + x].terrain_id == _C for x, y in tiles)
        assert (int(fish.x), int(fish.y)) != (41, 43), "the fish moved with its water"
    finally:
        _close(window)


@pytest.mark.parametrize("gate", ["map_is_square", "terrain_write_supported"])
def test_a_region_turn_is_refused_and_disabled_where_its_tiles_cannot_be_written(gate):
    window = _select_window()
    try:
        rect = (40, 40, 44, 44)
        _paint(window, rect, lambda x, y: (_A, _B)[x % 2])
        window.on_region_selected(rect)
        turn = [*window._rotate_actions, *window._flip_actions, *window._rotate_group_actions]
        assert all(action.isEnabled() for action in turn)
        start = _region_tiles(window, rect)
        setattr(window.scenario, gate, False)
        window._update_rotate_enabled()
        assert not any(action.isEnabled() for action in turn)
        window.on_selection_transform(region_transform.FLIP_HORIZONTAL)
        reason = "needs a square map" if gate == "map_is_square" else "terrain cannot be edited"
        assert reason in _last_status(window) and _last_status(window).startswith("Flip: ")
        window.on_unit_rotate(1)
        assert _last_status(window).startswith("Rotate: ")
        assert _region_tiles(window, rect) == start and window._lift is None
    finally:
        setattr(window.scenario, gate, True)
        _close(window)


def test_target_order_a_unit_selection_beats_the_region_and_rotate_alone_turns_the_region():
    window = _select_window()
    try:
        rect = (40, 40, 46, 46)
        _paint(window, rect, lambda x, y: (_A, _B, _C)[(x + 2 * y) % 3])
        units = _add(window, [(1, ARCHER, 60.5, 60.5, 0.0, None), (1, ARCHER, 62.5, 60.5, 0.0, None)])
        window.on_region_selected(rect)
        tiles = _region_tiles(window, rect)
        window.mode_combo.setCurrentText("Units")
        _select(window, units)
        assert window._transform_target() == "units" and not window._region_rotate_target()
        window.flip_horizontal_action.trigger()
        assert window._lift.kind == "units"
        assert _region_tiles(window, rect) == tiles, "Flip acted on the region, not the selection"
        window.deselect_units()
        assert window._transform_target() == "region"
        window.rotate_cw_action.trigger()
        assert window._lift.kind == "region"
        assert _region_tiles(window, rect) != tiles
    finally:
        _close(window)
