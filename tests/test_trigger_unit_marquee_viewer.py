"""TASK-168: a box dragged while a Unit[] pick is armed (Set Objects, or a list
field's Pick) writes every unit it covers as one undo record, through a real
offscreen ViewerWindow on tests/fixtures/triggers_120x120.aoe2scenario.

The fixture's three placed units: archer A on tile (50, 55), archer B on
(52, 57), the house on (40..41, 62..63), all player 1."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
PLAYER_ONE = 1
IDS = "selected_object_ids"
ARCHER_A_TILE = (50, 55)
ARCHER_B_TILE = (52, 57)
HOUSE_TILES = ((40, 62), (41, 63))
EMPTY_TILES = ((80, 80), (90, 90))
STACK_TILE = (100, 100)
VILLAGER = 83


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


def _window(style: str = "Flat"):
    window = conftest.shown_window(1100, 760)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(_gen().REFERENCE_TRIGGER)
    return window


def _ids(window, effect: int) -> list[int]:
    gen = _gen()
    if window.trigger_edits is not None:
        trigger = window.trigger_edits.manager().triggers[gen.REFERENCE_TRIGGER]
    else:
        from descape.scenario_io import parse_triggers

        trigger = parse_triggers(window.scenario).triggers[gen.REFERENCE_TRIGGER]
    return list(getattr(trigger.effects[effect], IDS))


def _records(window) -> int:
    return len(window.edit_history.records)


def _last_line(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _arm_list(window, effect: int) -> tuple:
    window.trigger_panel.select_entry("effect", effect)
    target = (_gen().REFERENCE_TRIGGER, "effect", effect, IDS, True)
    window.arm_unit_picker(target)
    assert window._unit_picker == target
    return target


def _box(view, tiles):
    """Viewport corners of the box round `tiles`' on-screen footprints, one pixel out."""
    from PyQt5.QtCore import QPointF

    rect = None
    for tile in tiles:
        bounds = view._tile_polygon(*tile).boundingRect()
        rect = bounds if rect is None else rect.united(bounds)
    rect = rect.adjusted(-1, -1, 1, 1)
    view.centerOn(rect.center())
    return QPointF(view.mapFromScene(rect.topLeft())), QPointF(view.mapFromScene(rect.bottomRight()))


def _mouse(kind, pos, button, buttons, modifiers=None):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier if modifiers is None else modifiers)


def _press_and_move(view, start, end, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    view.mousePressEvent(_mouse(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton, modifiers))
    for t in (0.5, 1.0):
        pos = start + (end - start) * t
        view.mouseMoveEvent(_mouse(QEvent.MouseMove, pos, Qt.NoButton, Qt.LeftButton, modifiers))


def _drag(view, tiles, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    start, end = _box(view, tiles)
    _press_and_move(view, start, end, modifiers)
    assert view._marquee_item is not None, "the box should show mid-drag"
    view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, modifiers))
    assert view._marquee_item is None


def _click(view, tile) -> None:
    from PyQt5.QtCore import QEvent, Qt

    pos = conftest.polygon_viewport_pos(view, *tile)
    view.mousePressEvent(_mouse(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
    view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))


ALL_THREE = (*HOUSE_TILES, ARCHER_A_TILE, ARCHER_B_TILE)


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
def test_a_box_over_three_units_writes_them_as_one_record(style) -> None:
    """Flat resolves the box by footprint tile, Stepped by on-screen anchor (_units_in_scene_rect)."""
    gen = _gen()
    window = _window(style)
    try:
        effect = gen.PATROL_IDS_EFFECT
        before = _ids(window, effect)
        assert before == list(gen.PATROL_SELECTED_IDS)
        _arm_list(window, effect)
        records = _records(window)
        _drag(window.map_view, ALL_THREE)
        assert _ids(window, effect) == [gen.REF_ARCHER_B, gen.REF_ARCHER_A, gen.REF_HOUSE]
        assert _records(window) == records + 1, "one box, one record"
        assert window._unit_picker is not None and window.map_view.unit_picker_active()
        assert _last_line(window) == "Box: added 2, removed 0"
        assert (PLAYER_ONE, gen.REF_HOUSE) in window.map_view._unit_select_keys
        window.undo()
        assert _ids(window, effect) == before
    finally:
        conftest.close_window(window)


def test_a_ctrl_box_toggles_the_units_it_covers() -> None:
    from PyQt5.QtCore import Qt

    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        records = _records(window)
        _drag(window.map_view, (ARCHER_A_TILE, ARCHER_B_TILE), Qt.ControlModifier)
        assert _ids(window, effect) == [gen.REF_ARCHER_A]
        assert _records(window) == records + 1
        assert _last_line(window) == "Box: added 1, removed 1"
    finally:
        conftest.close_window(window)


def test_a_shift_box_adds_like_a_plain_one() -> None:
    from PyQt5.QtCore import Qt

    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        _drag(window.map_view, (ARCHER_A_TILE, ARCHER_B_TILE), Qt.ShiftModifier)
        assert _ids(window, effect) == [gen.REF_ARCHER_B, gen.REF_ARCHER_A]
    finally:
        conftest.close_window(window)


def test_a_box_over_units_already_listed_writes_nothing() -> None:
    gen = _gen()
    window = _window()
    try:
        effect = gen.TASK_OBJECT_EFFECT
        _arm_list(window, effect)
        records = _records(window)
        _drag(window.map_view, (ARCHER_A_TILE, ARCHER_B_TILE))
        assert _ids(window, effect) == list(gen.TASK_SELECTED_IDS)
        assert _records(window) == records
        assert _last_line(window) == "Box: no units added"
        assert window._unit_picker is not None
    finally:
        conftest.close_window(window)


def test_a_box_over_empty_ground_writes_nothing_and_stays_armed() -> None:
    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        records = _records(window)
        _drag(window.map_view, EMPTY_TILES)
        assert _ids(window, effect) == list(gen.PATROL_SELECTED_IDS)
        assert _records(window) == records
        assert _last_line(window) == "Box: no units added"
        assert window._unit_picker is not None and window.map_view.unit_picker_active()
    finally:
        conftest.close_window(window)


def test_a_list_click_toggles_on_release_and_a_press_alone_writes_nothing() -> None:
    from PyQt5.QtCore import QEvent, Qt

    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        view = window.map_view
        records = _records(window)
        pos = conftest.polygon_viewport_pos(view, *ARCHER_A_TILE)
        view.mousePressEvent(_mouse(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        assert _ids(window, effect) == list(gen.PATROL_SELECTED_IDS), "a list pick waits for the release"
        assert _records(window) == records
        view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
        assert _ids(window, effect) == [gen.REF_ARCHER_B, gen.REF_ARCHER_A]
        assert _records(window) == records + 1
        _click(view, ARCHER_A_TILE)
        assert _ids(window, effect) == [gen.REF_ARCHER_B]
        assert window._unit_picker is not None
    finally:
        conftest.close_window(window)


def test_repeated_list_clicks_on_a_stack_walk_down_it() -> None:
    gen = _gen()
    window = _window()
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Stack", [PLAYER_ONE]):
            units = [model.add(PLAYER_ONE, VILLAGER, STACK_TILE[0] + 0.5, STACK_TILE[1] + 0.5) for _ in range(3)]
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        view = window.map_view
        assert len(view.stack_group_at(STACK_TILE)) == 3
        _click(view, STACK_TILE)
        first = _ids(window, effect)[-1]
        _click(view, STACK_TILE)
        second = _ids(window, effect)[-1]
        stacked = {u.reference_id for u in units}
        assert first in stacked and second in stacked and first != second, "the second click walks down"
        assert _ids(window, effect) == [*gen.PATROL_SELECTED_IDS, first, second]
    finally:
        conftest.close_window(window)


def test_rearming_between_scalar_and_list_targets_swaps_the_gesture() -> None:
    """set_unit_picker() stores `multi` before its unchanged-state guard: True to True."""
    from PyQt5.QtCore import QEvent, Qt

    gen = _gen()
    window = _window()
    try:
        effect = gen.TASK_OBJECT_EFFECT
        window.trigger_panel.select_entry("effect", effect)
        scalar = (gen.REFERENCE_TRIGGER, "effect", effect, "location_object_reference", False)
        listed = (gen.REFERENCE_TRIGGER, "effect", effect, IDS, True)
        view = window.map_view
        window.arm_unit_picker(scalar)
        assert not view._unit_picker_multi
        window.arm_unit_picker(listed)
        assert view.unit_picker_active() and view._unit_picker_multi
        window.arm_unit_picker(scalar)
        assert not view._unit_picker_multi
        pos = conftest.polygon_viewport_pos(view, *ARCHER_A_TILE)
        view.mousePressEvent(_mouse(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        trigger = window.trigger_edits.manager().triggers[gen.REFERENCE_TRIGGER]
        assert trigger.effects[effect].location_object_reference == gen.REF_ARCHER_A, "a scalar commits on press"
    finally:
        conftest.close_window(window)


def test_filter_hidden_units_are_skipped_and_a_listed_one_survives() -> None:
    from PyQt5.QtCore import Qt

    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        window.map_view.on_unit_pick((PLAYER_ONE, gen.REF_HOUSE))
        assert _ids(window, effect) == [gen.REF_ARCHER_B, gen.REF_HOUSE]
        window.disarm_unit_picker()
        window.show_buildings_action.setChecked(False)
        _arm_list(window, effect)
        _drag(window.map_view, ALL_THREE)
        assert _ids(window, effect) == [gen.REF_ARCHER_B, gen.REF_HOUSE, gen.REF_ARCHER_A], "the hidden house stays"
        _drag(window.map_view, ALL_THREE, Qt.ControlModifier)
        assert _ids(window, effect) == [gen.REF_HOUSE], "a Ctrl box cannot drop a hidden unit"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("how", ["escape", "right_click", "leave"])
def test_a_cancelled_drag_writes_nothing_and_leaves_no_box(how) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QKeyEvent

    gen = _gen()
    window = _window()
    try:
        effect = gen.PATROL_IDS_EFFECT
        _arm_list(window, effect)
        view = window.map_view
        records = _records(window)
        start, end = _box(view, ALL_THREE)
        _press_and_move(view, start, end)
        assert view._marquee_item is not None
        if how == "escape":
            view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        elif how == "right_click":
            view.mousePressEvent(_mouse(QEvent.MouseButtonPress, QPointF(5, 5), Qt.RightButton, Qt.RightButton))
        else:
            view.leaveEvent(QEvent(QEvent.Leave))
        assert view._marquee_item is None
        view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        assert _ids(window, effect) == list(gen.PATROL_SELECTED_IDS)
        assert _records(window) == records
        assert view._marquee_item is None
        armed = how == "leave"
        assert (window._unit_picker is not None) == armed and view.unit_picker_active() == armed
    finally:
        conftest.close_window(window)


def test_a_scalar_pick_draws_no_box() -> None:
    from PyQt5.QtCore import QEvent, Qt

    gen = _gen()
    window = _window()
    try:
        window.trigger_panel.select_entry("condition", gen.DESTROY_CONDITION)
        window.arm_unit_picker((gen.REFERENCE_TRIGGER, "condition", gen.DESTROY_CONDITION, "unit_object", False))
        view = window.map_view
        records = _records(window)
        start, end = _box(view, EMPTY_TILES)
        _press_and_move(view, start, end)
        assert view._marquee_item is None
        view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        assert _records(window) == records
        assert window._unit_picker is not None, "empty ground picks nothing, so a scalar pick stays armed"
    finally:
        conftest.close_window(window)


def test_the_garrison_pick_keeps_its_click_on_press_and_draws_no_box() -> None:
    from PyQt5.QtCore import QEvent, Qt

    gen = _gen()
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        window._selection = [(PLAYER_ONE, gen.REF_HOUSE)]
        window._refresh_selection_view()
        window._on_garrison_pick(True)
        view = window.map_view
        assert window._garrison_picker is not None and view.unit_picker_active()
        assert not view._unit_picker_multi
        picked = []
        view.on_unit_pick = picked.append
        pos = conftest.polygon_viewport_pos(view, *ARCHER_A_TILE)
        view.mousePressEvent(_mouse(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        assert picked == [(PLAYER_ONE, gen.REF_ARCHER_A)], "the garrison pick commits on press"
        start, end = _box(view, EMPTY_TILES)
        _press_and_move(view, start, end)
        assert view._marquee_item is None
    finally:
        conftest.close_window(window)


def test_the_list_hint_and_tooltips_name_the_box() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        effect = gen.TASK_OBJECT_EFFECT
        target = _arm_list(window, effect)
        assert _last_line(window).startswith(
            "Click a unit to add or remove it, or drag a box to add units (Ctrl toggles); Esc to finish"
        )
        assert "or drag a box to add them" in panel._pick_buttons[target].toolTip()
        assert "or drag a box to add them" in panel._group_rows["objects"].set_button.toolTip()
    finally:
        conftest.close_window(window)
