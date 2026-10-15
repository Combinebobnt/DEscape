"""Units mode's Select and Move tools, and Pan as pure camera there
(TASK-252, TASK-012, GH #47).

Every unit gesture used to run under Pan, through a mode-level MapView branch
that fired for any tool no earlier branch claimed. That branch is gated by
tool now: Select picks (click, Ctrl/Shift click, stack cycling, marquee),
Move drags, and Pan left-drags the camera as in every other mode.

Gestures are real press-move-release events through MapView's own handlers,
on the units fixture in Flat unless a test says otherwise. Tools are chosen
with `<tool>_action.setChecked(True)`, never `_on_tool_selected()`, which
leaves pan_action checked (GOTCHAS).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent

from descape import settings, viewer_common

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_VILLAGER = 83
# Clear ground in the fixture (asserted by _add_units), and the column of it the tests use.
_ROW = 63


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install (see test_unit_edit_viewer.py's twin)."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window(style: str = "Flat", tool: str | None = None, show: bool = True):
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    # Before the style switch: iso_action defaults checked, which would send Flat through real iso.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    if show:
        window.resize(1200, 800)
        window.show()
        QApplication.processEvents()
    if tool is not None:
        getattr(window, f"{tool}_action").setChecked(True)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _add_units(window, tiles, const: int = _VILLAGER):
    """One villager per tile, centred, for player 1, in one edit. Returns keys."""
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    assert not occupied & set(tiles), "the fixture already holds a unit on a test tile"
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [1]):
        units = [model.add(1, const, x + 0.5, y + 0.5) for x, y in tiles]
    return [(1, u.reference_id) for u in units]


def _make_stack(window, tile, count: int = 3):
    """`count` villagers at one point; keys top-first."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Stack", [1]):
        units = [model.add(1, _VILLAGER, tile[0] + 0.5, tile[1] + 0.5) for _ in range(count)]
    return [(1, u.reference_id) for u in reversed(units)]


def _select_keys(window, keys) -> None:
    window._selection = list(keys)
    window._refresh_selection_view()


def _position(window, key):
    entry = window.map_view._unit_index.entry_for_key(key)
    return entry.unit.x, entry.unit.y


def _vp(window, tile) -> QPointF:
    """Viewport centre of a tile, in Flat or the isometric styles."""
    view = window.map_view
    if view._terrain_style == "flat":
        return conftest.viewport_pos(view, *tile)
    return conftest.polygon_viewport_pos(view, *tile)


def _zoom_on(window, tile, factor: float = 4.0) -> None:
    """Zoomed in on `tile`, so a tile spans many pixels and the view can scroll."""
    view = window.map_view
    view.scale(factor, factor)
    view.centerOn(view.mapToScene(_vp(window, tile).toPoint()))


def _event(kind, pos, button, buttons, modifiers=Qt.NoModifier):
    return QMouseEvent(kind, pos, button, buttons, modifiers)


def _press(view, pos, modifiers=Qt.NoModifier) -> None:
    view.mousePressEvent(_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton, modifiers))


def _move(view, pos, modifiers=Qt.NoModifier, buttons=Qt.LeftButton) -> None:
    view.mouseMoveEvent(_event(QEvent.MouseMove, pos, Qt.NoButton, buttons, modifiers))


def _release(view, pos, modifiers=Qt.NoModifier) -> None:
    view.mouseReleaseEvent(_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton, modifiers))


def _gesture(view, start, end=None, modifiers=Qt.NoModifier) -> None:
    _press(view, start, modifiers)
    if end is not None:
        _move(view, end, modifiers)
    _release(view, end if end is not None else start, modifiers)


def _scroll(view) -> tuple[int, int]:
    return view.horizontalScrollBar().value(), view.verticalScrollBar().value()


def _spy(monkeypatch, view, name: str) -> list:
    calls: list = []
    monkeypatch.setattr(view, name, lambda *a: calls.append(a))
    return calls


# --- registry and toolbar -----------------------------------------------------


def test_move_is_a_units_only_non_edit_tool_on_v() -> None:
    move = next(t for t in settings.TOOLS if t.tool_id == "move")
    assert move.label == "Move"
    assert move.modes == ("units",)
    assert move.default_key == "V"
    assert not move.is_edit_tool
    assert "move" not in viewer_common.EDIT_TOOLS
    assert ("tool_move", "Move Tool", "V") in settings.REBINDABLE_ACTIONS
    assert settings.get_keybind("tool_move") == "V"
    assert viewer_common.tool_applicable("move", "units")
    for mode in ("view", "terrain", "triggers"):
        assert not viewer_common.tool_applicable("move", mode), mode
    ids = [t.tool_id for t in settings.TOOLS]
    assert ids.index("move") == ids.index("select") + 1


def test_select_and_move_show_and_enable_in_units_mode_and_move_hides_in_terrain() -> None:
    window = _window(show=False)
    try:
        for action in (window.select_action, window.move_action):
            assert action.isVisible() and action.isEnabled(), action.text()
        window.mode_combo.setCurrentText("Terrain")
        assert not window.move_action.isVisible()
        assert window.select_action.isVisible()
    finally:
        _close(window)


def test_the_units_bar_reads_in_order() -> None:
    """Unshown, so More Tools has measured nothing and every tool sits on the bar."""
    window = _window(show=False)
    try:
        tool_actions = {getattr(window, f"{t.tool_id}_action"): t.label for t in settings.TOOLS}
        shown = [tool_actions[a] for a in window.main_toolbar.actions() if a in tool_actions and a.isVisible()]
        assert shown == [
            "Pan", "Select", "Move", "Eyedropper", "Ruler", "Place Unit", "Wall Rectangle", "Convert", "Eraser",
        ]
        # Overflow hides a suffix of TOOLS, so the bar must be built in that order.
        all_tools = [a for a in window.main_toolbar.actions() if a in tool_actions]
        assert all_tools == [getattr(window, f"{t.tool_id}_action") for t in settings.TOOLS]
    finally:
        _close(window)


def test_the_units_mode_cursors_follow_the_tool() -> None:
    from PyQt5.QtWidgets import QGraphicsView

    window = _window(show=False)
    try:
        view = window.map_view
        window.select_action.setChecked(True)
        assert view.cursor().shape() == Qt.PointingHandCursor
        window.move_action.setChecked(True)
        assert view.cursor().shape() == Qt.SizeAllCursor
        window.pan_action.setChecked(True)
        assert not view.testAttribute(Qt.WA_SetCursor)
        assert view.dragMode() == QGraphicsView.ScrollHandDrag
        window.mode_combo.setCurrentText("Terrain")
        window.select_action.setChecked(True)
        assert view.cursor().shape() == Qt.CrossCursor
    finally:
        _close(window)


def test_paste_filters_hide_under_select_in_units_and_the_placement_boxes_show_under_move() -> None:
    window = _window(show=False, tool="select")
    try:
        assert window._current_tool == "select"
        assert not window.paste_terrain_param_action.isVisible()
        assert not window.paste_units_param_action.isVisible()
        assert not window.unrestricted_param_action.isVisible()
        window.move_action.setChecked(True)
        assert window.unrestricted_param_action.isVisible() and window.unrestricted_check.isEnabled()
        assert window.off_grid_param_action.isVisible()
        assert window.gate_walls_param_action.isVisible()
    finally:
        _close(window)


# --- Pan in Units mode ---------------------------------------------------------


def test_a_pan_drag_in_units_mode_scrolls_even_from_a_selected_unit(monkeypatch) -> None:
    window = _window()
    try:
        view = window.map_view
        (key,) = _add_units(window, [(64, _ROW)])
        _select_keys(window, [key])
        _zoom_on(window, (64, _ROW))
        assert window._current_tool == "pan"
        moves = _spy(monkeypatch, view, "_on_unit_move")
        marquees = _spy(monkeypatch, view, "_on_marquee_select")
        start = _vp(window, (64, _ROW))
        before = _scroll(view)

        _press(view, start)
        assert view._unit_drag_key is None and view._marquee_start_pos is None
        end = start + QPointF(-40, -30)
        _move(view, end)
        _release(view, end)

        assert _scroll(view) == (before[0] + 40, before[1] + 30)
        assert window._selection == [key]
        assert _position(window, key) == (64.5, _ROW + 0.5)
        assert moves == [] and marquees == []
        # A later move with no button held is a hover, not a pan.
        after = _scroll(view)
        _move(view, end + QPointF(-25, -25), buttons=Qt.NoButton)
        assert _scroll(view) == after
    finally:
        _close(window)


def test_a_pan_click_on_a_unit_selects_nothing_and_the_selection_still_deletes() -> None:
    window = _window()
    try:
        view = window.map_view
        kept, other = _add_units(window, [(64, _ROW), (67, _ROW)])
        _select_keys(window, [kept])
        _press(view, _vp(window, (67, _ROW)))
        assert view._unit_drag_key is None and view._marquee_start_pos is None
        _release(view, _vp(window, (67, _ROW)))
        assert window._selection == [kept]

        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        assert view._unit_index.entry_for_key(kept) is None
        assert view._unit_index.entry_for_key(other) is not None
    finally:
        _close(window)


def test_pan_draws_no_unit_hover_outline() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        _add_units(window, [(64, _ROW)])
        _move(view, _vp(window, (64, _ROW)), buttons=Qt.NoButton)
        assert view._unit_hover_item is not None, "precondition: Select outlines the unit under the cursor"

        window.pan_action.setChecked(True)
        assert view._unit_hover_item is None, "switching to Pan with no move left the outline"
        _move(view, _vp(window, (64, _ROW)), buttons=Qt.NoButton)
        assert view._unit_hover_item is None
        assert view._pan_highlight_item is not None, "Pan shows its tile cue, as in every other mode"
    finally:
        _close(window)


@pytest.mark.parametrize("tool", ["pan", "select", "move", "eyedropper", "ruler", "wall_rect"])
@pytest.mark.parametrize("seed", ["drag_key", "marquee"])
def test_a_stale_gesture_from_a_swallowed_release_dies_at_the_next_press(monkeypatch, tool, seed) -> None:
    window = _window(tool=tool)
    try:
        view = window.map_view
        (key,) = _add_units(window, [(64, _ROW)])
        _zoom_on(window, (70, _ROW + 6))
        if seed == "drag_key":
            view._unit_drag_key = key
            view._unit_drag_press_pos = QPointF(0, 0)
        else:
            view._marquee_start_pos = QPointF(0, 0)
        moves = _spy(monkeypatch, view, "_on_unit_move")
        marquees = _spy(monkeypatch, view, "_on_marquee_select")
        start = _vp(window, (70, _ROW + 6))  # empty ground
        before = _scroll(view)

        _gesture(view, start, start + QPointF(-40, -30))

        assert moves == []
        # Select's own fresh marquee is the only one allowed to resolve.
        assert len(marquees) == (1 if tool == "select" else 0)
        assert _position(window, key) == (64.5, _ROW + 0.5)
        if tool == "pan":
            assert _scroll(view) != before
    finally:
        _close(window)


# --- Select in Units mode ------------------------------------------------------


def test_a_select_click_selects_and_ctrl_toggles_and_shift_adds() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (67, _ROW)])
        _gesture(view, _vp(window, (64, _ROW)))
        assert window._selection == [a]
        _gesture(view, _vp(window, (67, _ROW)), modifiers=Qt.ControlModifier)
        assert window._selection == [a, b]
        _gesture(view, _vp(window, (64, _ROW)), modifiers=Qt.ControlModifier)
        assert window._selection == [b]
        _gesture(view, _vp(window, (64, _ROW)), modifiers=Qt.ShiftModifier)
        _gesture(view, _vp(window, (64, _ROW)), modifiers=Qt.ShiftModifier)
        assert window._selection == [b, a]
        _gesture(view, _vp(window, (64, _ROW)))
        assert window._selection == [a]
    finally:
        _close(window)


def test_a_select_drag_from_a_unit_that_leaves_its_tile_runs_a_marquee_and_moves_nothing() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        depth = window.edit_history.cursor
        _press(view, _vp(window, (64, _ROW)))
        assert window._selection == [a]
        end = _vp(window, (67, _ROW + 1))
        _move(view, end)
        assert view._marquee_item is not None
        assert view._unit_ghost_item is None
        _release(view, end)
        assert window._selection == [a, b]
        assert _position(window, a) == (64.5, _ROW + 0.5)
        assert window.edit_history.cursor == depth
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
def test_a_select_press_that_jitters_inside_its_tile_stays_a_click(style) -> None:
    window = _window(style, tool="select")
    try:
        view = window.map_view
        assert view._terrain_style == style.lower()
        a, b = _add_units(window, [(64, _ROW), (67, _ROW)])
        _select_keys(window, [b])
        _zoom_on(window, (64, _ROW), 6.0)
        jitter = 6
        start = _vp(window, (64, _ROW))
        end = start + QPointF(jitter, 0)
        assert jitter > view.UNIT_DRAG_THRESHOLD_PX
        assert view._pick_tile(view.mapToScene(end.toPoint())) == (64, _ROW), "precondition: still inside the tile"

        _press(view, start)
        _move(view, end)
        assert view._marquee_item is None, "a rectangle was drawn inside the pressed tile"
        _release(view, end)
        assert window._selection == [a]
    finally:
        _close(window)


def test_a_select_press_on_a_unit_over_an_unresolved_tile_stays_a_click(monkeypatch) -> None:
    # A Stepped skirt face resolves no tile: jitter there must not draw a marquee.
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (67, _ROW)])
        _select_keys(window, [b])
        _zoom_on(window, (64, _ROW), 6.0)
        start = _vp(window, (64, _ROW))
        end = start + QPointF(6, 0)
        monkeypatch.setattr(view, "_pick_tile", lambda pos: None)

        _press(view, start)
        assert window._selection == [a], "precondition: the press hit the unit"
        _move(view, end)
        assert view._marquee_item is None, "a rectangle was drawn on jitter"
        _release(view, end)
        assert window._selection == [a]
    finally:
        _close(window)


@pytest.mark.parametrize("modifiers", [Qt.NoModifier, Qt.ControlModifier, Qt.ShiftModifier], ids=["plain", "ctrl", "shift"])
def test_a_marquee_from_a_unit_that_resolves_no_keys_keeps_the_press_time_selection(monkeypatch, modifiers) -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (67, _ROW)])
        _select_keys(window, [b])
        monkeypatch.setattr(view, "_units_in_scene_rect", lambda rect: [])
        _press(view, _vp(window, (64, _ROW)), modifiers)
        pressed = list(window._selection)
        assert pressed == ([a] if modifiers == Qt.NoModifier else [b, a])
        end = _vp(window, (64, _ROW + 3))
        _move(view, end, modifiers)
        assert view._marquee_item is not None
        _release(view, end, modifiers)
        assert window._selection == pressed
    finally:
        _close(window)


@pytest.mark.parametrize("modifiers", [Qt.ControlModifier, Qt.ShiftModifier], ids=["ctrl", "shift"])
@pytest.mark.parametrize("covers", ["a_and_b", "only_b"])
def test_a_modified_marquee_from_an_unselected_unit_selects_both(monkeypatch, modifiers, covers) -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        if covers == "only_b":
            # A rectangle from on top of A that misses A's anchor, as Stepped/Sloped can draw.
            monkeypatch.setattr(view, "_units_in_scene_rect", lambda rect: [b])
        _press(view, _vp(window, (64, _ROW)), modifiers)
        end = _vp(window, (67, _ROW + 1))
        _move(view, end, modifiers)
        _release(view, end, modifiers)
        assert sorted(window._selection) == sorted([a, b])
    finally:
        _close(window)


def test_a_ctrl_marquee_from_a_selected_unit_leaves_it_out_and_takes_the_rest() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        _select_keys(window, [a])
        _press(view, _vp(window, (64, _ROW)), Qt.ControlModifier)
        assert window._selection == []
        end = _vp(window, (67, _ROW + 1))
        _move(view, end, Qt.ControlModifier)
        _release(view, end, Qt.ControlModifier)
        assert window._selection == [b]
    finally:
        _close(window)


def test_select_in_units_mode_never_touches_the_region_state() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        region = (10, 10, 20, 20)
        view.set_region(region)
        window.on_region_selected(region)
        _add_units(window, [(64, _ROW)])
        _press(view, _vp(window, (64, _ROW)))
        _move(view, _vp(window, (70, _ROW + 4)))
        assert view._select_anchor is None and view._move_anchor is None
        _release(view, _vp(window, (70, _ROW + 4)))
        _gesture(view, _vp(window, (75, _ROW + 6)), _vp(window, (80, _ROW + 9)))
        assert view._region == region
        assert window._region == region
        assert view._select_anchor is None
    finally:
        _close(window)


def test_select_in_terrain_still_drags_a_region_after_a_units_round_trip() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        window.mode_combo.setCurrentText("Terrain")
        assert window._current_tool == "select"
        _gesture(view, _vp(window, (10, 10)), _vp(window, (14, 12)))
        assert view._region == (10, 10, 15, 13)
    finally:
        _close(window)


# --- Move ----------------------------------------------------------------------


def test_a_move_press_on_empty_ground_drags_nothing_and_keeps_the_selection() -> None:
    window = _window(tool="move")
    try:
        (a,) = _add_units(window, [(64, _ROW)])
        _select_keys(window, [a])
        assert window.on_move_press(window.map_view.mapToScene(_vp(window, (70, _ROW + 6)).toPoint()), Qt.NoModifier) is None
        assert window._selection == [a]
        _gesture(window.map_view, _vp(window, (70, _ROW + 6)))
        assert window._selection == [a]
    finally:
        _close(window)


def test_a_move_press_on_a_selected_unit_keeps_the_group() -> None:
    window = _window(tool="move")
    try:
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        _select_keys(window, [a, b])
        pos = window.map_view.mapToScene(_vp(window, (66, _ROW)).toPoint())
        assert window.on_move_press(pos, Qt.NoModifier) == b
        assert window._selection == [a, b]
        assert window._is_group_key(b)
    finally:
        _close(window)


def test_a_move_press_on_an_unselected_unit_replaces_the_group_with_it() -> None:
    window = _window(tool="move")
    try:
        a, b, c = _add_units(window, [(64, _ROW), (66, _ROW), (68, _ROW)])
        _select_keys(window, [a, b])
        pos = window.map_view.mapToScene(_vp(window, (68, _ROW)).toPoint())
        assert window.on_move_press(pos, Qt.NoModifier) == c
        assert window._selection == [c]
        assert "Selected Villager" in window.status_log.toPlainText().splitlines()[-1]
    finally:
        _close(window)


def test_move_drags_the_selected_unit_lower_in_a_stack_while_the_top_one_stays() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        tile = (64, _ROW)
        keys = _make_stack(window, tile)
        _gesture(view, _vp(window, tile))
        _gesture(view, _vp(window, tile))
        assert window._selection == [keys[1]]

        window.move_action.setChecked(True)
        _gesture(view, _vp(window, tile), _vp(window, (66, _ROW)))
        assert _position(window, keys[1]) == (66.5, _ROW + 0.5)
        assert _position(window, keys[0]) == (64.5, _ROW + 0.5)
        assert _position(window, keys[2]) == (64.5, _ROW + 0.5)
    finally:
        _close(window)


def test_ctrl_held_at_a_move_press_changes_no_membership() -> None:
    window = _window(tool="move")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        _select_keys(window, [a])
        _gesture(view, _vp(window, (64, _ROW)), modifiers=Qt.ControlModifier)
        assert window._selection == [a]
        _gesture(view, _vp(window, (66, _ROW)), modifiers=Qt.ControlModifier)
        assert window._selection == [b]
    finally:
        _close(window)


def test_one_undo_restores_a_move() -> None:
    window = _window(tool="move")
    try:
        (a,) = _add_units(window, [(64, _ROW)])
        depth = window.edit_history.cursor
        _gesture(window.map_view, _vp(window, (64, _ROW)), _vp(window, (66, _ROW + 2)))
        assert _position(window, a) == (66.5, _ROW + 2.5)
        assert window.edit_history.cursor == depth + 1
        window.undo()
        assert _position(window, a) == (64.5, _ROW + 0.5)
    finally:
        _close(window)


# --- a keyboard tool switch mid-gesture ------------------------------------------


def test_a_switch_to_pan_mid_drag_still_commits_the_move() -> None:
    window = _window(tool="move")
    try:
        view = window.map_view
        (a,) = _add_units(window, [(64, _ROW)])
        _press(view, _vp(window, (64, _ROW)))
        end = _vp(window, (66, _ROW))
        _move(view, end)
        window.pan_action.setChecked(True)
        _move(view, end)
        _release(view, end)
        assert _position(window, a) == (66.5, _ROW + 0.5)
    finally:
        _close(window)


def test_a_switch_to_pan_mid_marquee_still_commits_the_marquee() -> None:
    window = _window(tool="select")
    try:
        view = window.map_view
        a, b = _add_units(window, [(64, _ROW), (66, _ROW)])
        _press(view, _vp(window, (63, _ROW - 1)))
        end = _vp(window, (67, _ROW + 1))
        _move(view, end)
        window.pan_action.setChecked(True)
        _move(view, end)
        _release(view, end)
        assert window._selection == [a, b]
    finally:
        _close(window)


# --- the Garrison block's Pick from map returns to its tool ------------------------

_WATCH_TOWER = 79


def _armed_garrison_pick(window, tool: str):
    """A Watch Tower selected and Pick from map armed while `tool` was active."""
    (tower,) = _add_units(window, [(64, _ROW)], const=_WATCH_TOWER)
    getattr(window, f"{tool}_action").setChecked(True)
    _select_keys(window, [tower])
    window.units_panel.garrison_pick_button.click()
    assert window._garrison_picker == tower
    assert window._current_tool == "pan", "arming forces Pan"
    return tower


@pytest.mark.parametrize("cancel", ["escape", "right_click", "button"])
@pytest.mark.parametrize("tool", ["select", "move"])
def test_cancelling_the_garrison_pick_returns_to_the_tool_it_was_armed_from(tool, cancel) -> None:
    window = _window(show=False)
    try:
        view = window.map_view
        _armed_garrison_pick(window, tool)
        if cancel == "escape":
            view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        elif cancel == "right_click":
            view.mousePressEvent(_event(QEvent.MouseButtonPress, _vp(window, (64, _ROW)), Qt.RightButton, Qt.RightButton))
        else:
            window.units_panel.garrison_pick_button.click()
        assert window._garrison_picker is None
        assert not view.unit_picker_active()
        assert window._current_tool == tool
        assert getattr(window, f"{tool}_action").isChecked()
    finally:
        _close(window)


def test_choosing_another_tool_while_the_garrison_pick_is_armed_keeps_that_tool() -> None:
    window = _window(show=False)
    try:
        _armed_garrison_pick(window, "select")
        window.place_unit_action.setChecked(True)
        assert window._garrison_picker is None
        assert window._current_tool == "place_unit"
    finally:
        _close(window)


@pytest.mark.parametrize("ender", ["mode_switch", "load", "undo"])
def test_a_garrison_pick_ended_by_the_app_does_not_restore_the_tool(ender) -> None:
    window = _window(show=False)
    try:
        _armed_garrison_pick(window, "select")
        if ender == "mode_switch":
            window.mode_combo.setCurrentText("Terrain")
        elif ender == "load":
            window.load_scenario(FIXTURE_PATH)
        else:
            window.undo()
        assert window._garrison_picker is None
        assert window._garrison_return_tool is None
        assert window._current_tool == "pan"
    finally:
        _close(window)


# --- the Pan click hint --------------------------------------------------------------

_HINT = "Pan moves the view."


def _hints(window) -> list[str]:
    return [line for line in window.status_log.toPlainText().splitlines() if _HINT in line]


def test_the_first_pan_click_on_a_unit_logs_the_hint_once() -> None:
    window = _window()
    try:
        view = window.map_view
        _add_units(window, [(64, _ROW)])
        _gesture(view, _vp(window, (70, _ROW + 6)))  # empty ground: no hint, and the flag stays unspent
        assert _hints(window) == []
        _gesture(view, _vp(window, (64, _ROW)))
        assert _hints(window) == ["Pan moves the view. Select (S) picks units, Move (V) drags them."]
        _gesture(view, _vp(window, (64, _ROW)))
        assert len(_hints(window)) == 1
        assert window._selection == []
    finally:
        _close(window)


def test_a_pan_drag_that_starts_on_a_unit_logs_no_hint() -> None:
    window = _window()
    try:
        view = window.map_view
        _add_units(window, [(64, _ROW)])
        _zoom_on(window, (64, _ROW))
        start = _vp(window, (64, _ROW))
        _gesture(view, start, start + QPointF(-40, 0))
        assert _hints(window) == []
    finally:
        _close(window)


def test_the_pan_hint_names_an_unbound_tool_without_a_key() -> None:
    window = _window()
    try:
        settings.set_keybind("tool_move", "")
        window.apply_keybind("tool_move")
        _add_units(window, [(64, _ROW)])
        _gesture(window.map_view, _vp(window, (64, _ROW)))
        assert _hints(window) == ["Pan moves the view. Select (S) picks units, Move drags them."]
    finally:
        _close(window)


# --- mode switches ---------------------------------------------------------------


def test_terrain_with_select_into_units_lands_on_pan() -> None:
    window = _window(show=False)
    try:
        window.mode_combo.setCurrentText("Terrain")
        window.select_action.setChecked(True)
        window.mode_combo.setCurrentText("Units")
        assert window._current_tool == "pan"
    finally:
        _close(window)


def test_units_with_select_into_terrain_keeps_select_unless_the_terrain_is_read_only() -> None:
    window = _window(show=False, tool="select")
    try:
        window.mode_combo.setCurrentText("Terrain")
        assert window._current_tool == "select"
        window.mode_combo.setCurrentText("Units")
        window.scenario.terrain_write_supported = False
        window._update_tool_enabled()
        window.select_action.setChecked(True)
        assert window._current_tool == "select", "Select picks units on a read-only terrain too"
        window.mode_combo.setCurrentText("Terrain")
        assert window._current_tool == "pan"
    finally:
        _close(window)


def test_units_with_move_into_terrain_lands_on_pan() -> None:
    window = _window(show=False, tool="move")
    try:
        window.mode_combo.setCurrentText("Terrain")
        assert window._current_tool == "pan"
    finally:
        _close(window)
