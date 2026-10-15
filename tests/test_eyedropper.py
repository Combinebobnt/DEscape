"""Eyedropper tool wiring, driven through a real offscreen ViewerWindow --
same technique and default-tier rationale as tests/test_fill_tool.py.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_PICK_TERRAIN = 15  # GRASS_1, distinct from the blank template's terrain_id=0
_PICK_ELEVATION = 3


def _edit_window():
    """Loads the blank template and switches to Terrain mode with Eyedropper
    active. Caller must edit_history.mark_saved() + close()."""
    window = conftest.terrain_edit_window()
    window.eyedropper_action.setChecked(True)
    return window


def test_eyedropper_registered_as_keybind_and_toolbar_action() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    assert ("tool_eyedropper", "Eyedropper Tool", "I") in settings.REBINDABLE_ACTIONS

    window = ViewerWindow()
    try:
        assert window._keybind_actions["tool_eyedropper"] is window.eyedropper_action
        assert window.eyedropper_action.shortcut() == QKeySequence("I")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_eyedropper_disabled_with_no_map_and_outside_terrain() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.eyedropper_action.isEnabled()  # no map loaded yet

        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert not window.eyedropper_action.isVisible()  # still View mode

        window.mode_combo.setCurrentText("Terrain")
        assert window.eyedropper_action.isVisible()
        assert window.eyedropper_action.isEnabled()
        assert window.eyedropper_action.isEnabled() == window.draw_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_pick_loads_terrain_and_elevation() -> None:
    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        tile = mm.terrain[0]
        tile.terrain_id = _PICK_TERRAIN
        tile.elevation = _PICK_ELEVATION

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
        assert window.terrain_panel.view.current_value() == _PICK_TERRAIN
        assert window.elevation_level_spin.value() == _PICK_ELEVATION
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_pick_mutates_nothing() -> None:
    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        mm.terrain[0].terrain_id = _PICK_TERRAIN

        window.pick_tile_value(0, 0, 0)

        assert window.edit_history.records == []
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_pick_out_of_enum_terrain_leaves_the_picker_unchanged() -> None:
    """The corruption guard: a terrain id the picker doesn't list must leave
    the chosen terrain alone, never blank it to None, which Draw/Fill would
    then write straight into a tile's terrain_id."""
    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        mm.terrain[0].terrain_id = 9999
        before = window.terrain_panel.terrain_id()

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == before
        assert window.terrain_panel.terrain_id() is not None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_pick_out_of_range_elevation_leaves_spin_unchanged() -> None:
    from descape.viewer import ELEVATION_LEVEL_MAX

    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        mm.terrain[0].elevation = ELEVATION_LEVEL_MAX + 5
        before = window.elevation_level_spin.value()

        window.pick_tile_value(0, 0, 0)

        assert window.elevation_level_spin.value() == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_both_param_widgets_visible_while_eyedropper_active() -> None:
    """The one place the obvious implementation silently defeats the
    feature: param_widget == "" would otherwise hide Level right when the
    tool writes into it. The terrain half lands in the always-visible
    Terrain-mode sidebar page (GH #56)."""
    window = _edit_window()
    try:
        assert window.left_stack.currentWidget() is window.terrain_panel
        assert window.level_param_spin_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_paste_gate_on_region_not_active_tool_even_under_eyedropper() -> None:
    """Phase 2.8: Copy/Paste Region no longer depend on which tool is
    active (the old tool-scoped clipboard this test used to cover is
    retired). With Eyedropper active and no region there is nothing to
    copy, so both stay disabled; with a region already selected, Copy stays
    enabled right through Eyedropper, and copying it enables Paste too."""
    window = _edit_window()
    try:
        window.on_hover((0, 0))
        assert not window.copy_action.isEnabled()
        assert not window.paste_action.isEnabled()

        window.on_region_selected((0, 0, 2, 2))
        assert window.copy_action.isEnabled()
        window.copy_region()
        assert window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_forced_back_to_pan_on_close() -> None:
    window = _edit_window()
    try:
        assert window.eyedropper_action.isChecked()
        window.close_scenario()
        assert window.pan_action.isChecked()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_forced_back_to_pan_leaving_terrain_mode() -> None:
    window = _edit_window()
    try:
        assert window.eyedropper_action.isChecked()
        window.mode_combo.setCurrentText("View")
        assert window.pan_action.isChecked()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selecting_eyedropper_configures_map_view() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QGraphicsView

    from descape.viewer_common import CLICK_TOOLS, EDIT_TOOLS

    window = _edit_window()
    try:
        assert window._current_tool == "eyedropper"
        assert window.map_view._tool == "eyedropper"
        assert window.map_view.dragMode() == QGraphicsView.NoDrag
        assert window.map_view.cursor().shape() == Qt.CrossCursor
        assert "eyedropper" not in EDIT_TOOLS
        assert "eyedropper" in CLICK_TOOLS
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_pick_sets_both_widgets() -> None:
    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        tile = mm.terrain[0]
        tile.terrain_id = _PICK_TERRAIN
        tile.elevation = _PICK_ELEVATION

        window.on_click_edit(0, 0, 0)  # real MapView dispatch entry point

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
        assert window.terrain_panel.view.current_value() == _PICK_TERRAIN
        assert window.elevation_level_spin.value() == _PICK_ELEVATION
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the Target combo ---------------------------------------------------------

_LAYER_TERRAIN = 2  # BEACH, distinct from _PICK_TERRAIN and the template's 0


def _layered_tile(window):
    tile = window.scenario.map_manager.terrain[0]
    tile.terrain_id = _PICK_TERRAIN
    tile.layer = _LAYER_TERRAIN
    tile.elevation = _PICK_ELEVATION
    return tile


def test_layer_target_picks_the_layered_terrain_of_a_layered_tile() -> None:
    from descape.terrain_palette import name_for_terrain_id

    window = _edit_window()
    try:
        tile = _layered_tile(window)
        window.terrain_target_combo.setCurrentIndex(1)
        assert window._terrain_target() == "layer"

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == _LAYER_TERRAIN
        assert window.elevation_level_spin.value() == _PICK_ELEVATION
        assert (tile.terrain_id, tile.layer) == (_PICK_TERRAIN, _LAYER_TERRAIN)  # a pick mutates nothing
        last = window.status_log.toPlainText().splitlines()[-1]
        assert (
            f"Picked layer {name_for_terrain_id(_LAYER_TERRAIN)} (over {name_for_terrain_id(_PICK_TERRAIN)}), "
            f"elevation {_PICK_ELEVATION} from (0, 0)"
        ) in last
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_layer_target_picks_the_base_terrain_of_an_unlayered_tile() -> None:
    window = _edit_window()
    try:
        tile = window.scenario.map_manager.terrain[0]
        tile.terrain_id = _PICK_TERRAIN
        window.terrain_target_combo.setCurrentIndex(1)

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
        assert "Picked layer" not in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("index", [0, 2])
def test_terrain_and_clear_targets_pick_the_base_terrain_of_a_layered_tile(index: int) -> None:
    window = _edit_window()
    try:
        _layered_tile(window)
        window.terrain_target_combo.setCurrentIndex(index)

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
        assert "Picked layer" not in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_layer_target_on_a_file_with_no_layer_field_picks_the_base_terrain() -> None:
    window = _edit_window()
    try:
        _layered_tile(window)
        window.terrain_target_combo.setCurrentIndex(1)
        window.scenario.terrain_has_layer = False  # as a v1.21 document

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- Units mode: the click picks a unit ---------------------------------------

_UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_ARCHER_TILE = (20, 20)  # Player 2's archer, reference_id 300
_ARCHER_KEY = (2, 300)
_EMPTY_TILE = (60, 60)


def _units_window():
    """The units fixture in Flat Units mode with Eyedropper active, shown so a
    synthesized press maps through a real viewport."""
    window = conftest.shown_window()
    window.load_scenario(_UNITS_FIXTURE)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    window.eyedropper_action.setChecked(True)
    return window


def _record_click_callbacks(view) -> dict[str, list]:
    """Replaces every callback a press could reach with a recorder."""
    calls: dict[str, list] = {"eyedrop": [], "place": [], "select": [], "edit": [], "stroke": []}
    view.on_unit_eyedrop = lambda entry, tile, modifiers: calls["eyedrop"].append((entry, tile, modifiers))
    view._on_unit_place = lambda pos, modifiers: calls["place"].append(pos)
    view._on_click_select = lambda pos, modifiers: calls["select"].append(pos)
    view._on_click_edit = lambda x, y, modifiers: calls["edit"].append((x, y))
    view._on_stroke_start = lambda: calls["stroke"].append("start")
    return calls


def _press(view, viewport_pos, button=None, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QMouseEvent

    button = Qt.LeftButton if button is None else button
    modifiers = Qt.NoModifier if modifiers is None else modifiers
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, viewport_pos, button, button, modifiers))


def test_eyedropper_is_visible_and_selectable_in_units_mode() -> None:
    from descape import viewer_common

    assert viewer_common.tool_applicable("eyedropper", "units")
    assert viewer_common.tool_applicable("eyedropper", "terrain")
    assert not viewer_common.tool_applicable("eyedropper", "view")
    assert not viewer_common.tool_applicable("eyedropper", "triggers")

    window = conftest.terrain_edit_window()
    try:
        window.mode_combo.setCurrentText("Units")
        assert window.eyedropper_action.isVisible()
        assert window.eyedropper_action.isEnabled()

        window.eyedropper_action.setChecked(True)
        assert window._current_tool == "eyedropper"
        assert window.map_view._tool == "eyedropper"
        assert not window.pan_action.isChecked(), "nothing forces the tool back to Pan"

        window.mode_combo.setCurrentText("View")
        assert window.pan_action.isChecked()
        assert not window.eyedropper_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_units_mode_click_picks_a_unit_and_never_places_or_selects() -> None:
    """The routing trap: Eyedropper is click_only, and the Units branch sends
    every click_only tool's click to Place Unit."""
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        units_before = sum(len(units) for units in window.scenario.unit_manager.units)
        calls = _record_click_callbacks(view)

        _press(view, conftest.viewport_pos(view, *_ARCHER_TILE))

        assert calls["place"] == [], "the pick must not place a unit"
        assert calls["select"] == [] and calls["edit"] == [] and calls["stroke"] == []
        assert sum(len(units) for units in window.scenario.unit_manager.units) == units_before
        assert len(calls["eyedrop"]) == 1
        entry, tile, modifiers = calls["eyedrop"][0]
        assert (entry.player_id, entry.unit.reference_id) == _ARCHER_KEY
        assert tile == _ARCHER_TILE
        assert not modifiers & Qt.ShiftModifier

        _press(view, conftest.viewport_pos(view, *_ARCHER_TILE), modifiers=Qt.ShiftModifier)
        assert len(calls["eyedrop"]) == 2
        assert calls["eyedrop"][1][2] & Qt.ShiftModifier, "the handler reads Shift off the click"
        assert calls["place"] == [] and calls["select"] == []
    finally:
        conftest.close_window(window)


def test_a_units_mode_right_click_off_map_click_or_empty_ground_picks_nothing() -> None:
    from PyQt5.QtCore import QPointF, Qt

    window = _units_window()
    try:
        view = window.map_view
        calls = _record_click_callbacks(view)

        _press(view, conftest.viewport_pos(view, *_ARCHER_TILE), button=Qt.RightButton)
        assert all(not made for made in calls.values()), "a right click does nothing"

        assert view.pick_unit_at(view.mapToScene(conftest.viewport_pos(view, *_EMPTY_TILE).toPoint())) is None
        _press(view, conftest.viewport_pos(view, *_EMPTY_TILE))
        assert all(not made for made in calls.values()), "empty ground picks nothing"

        # Nothing can be hit off the canvas, so the pin is that the click is never hit-tested.
        hit_tests: list = []
        real_pick = view.pick_unit_cover_at
        view.pick_unit_cover_at = lambda pos, *args: hit_tests.append(pos) or real_pick(pos, *args)
        _press(view, conftest.viewport_pos(view, *_ARCHER_TILE))
        assert len(hit_tests) == 1 and len(calls["eyedrop"]) == 1, "the control: an on-map click is hit-tested"
        off_map = QPointF(view.mapFromScene(QPointF(-40.0, -40.0)))
        assert not view._map_rect.contains(view.mapToScene(off_map.toPoint()))
        _press(view, off_map)
        assert len(hit_tests) == 1 and len(calls["eyedrop"]) == 1, "a click off the map does nothing"
    finally:
        conftest.close_window(window)


def test_a_terrain_mode_click_still_picks_the_tile() -> None:
    window = conftest.shown_terrain_window()
    try:
        window.eyedropper_action.setChecked(True)
        view = window.map_view
        eyedrops: list = []
        view.on_unit_eyedrop = lambda *args: eyedrops.append(args)
        tile = window.scenario.map_manager.terrain[0]
        tile.terrain_id = _PICK_TERRAIN
        tile.elevation = _PICK_ELEVATION

        _press(view, conftest.polygon_viewport_pos(view, 0, 0))

        assert window.terrain_panel.terrain_id() == _PICK_TERRAIN
        assert window.elevation_level_spin.value() == _PICK_ELEVATION
        assert eyedrops == []
    finally:
        conftest.close_window(window)


def test_the_eyedropper_shows_its_params_in_terrain_mode_only() -> None:
    """Level and the Target combo belong to the tile pick. In Units mode the
    tool reads a unit into the sidebar and has no toolbar param at all."""
    window = _edit_window()
    try:
        assert window.level_param_spin_action.isVisible()
        assert window.terrain_target_param_action.isVisible()

        window.mode_combo.setCurrentText("Units")
        window.eyedropper_action.setChecked(True)
        assert window._current_tool == "eyedropper"
        assert not window.level_param_spin_action.isVisible()
        assert not window.elevation_level_spin.isEnabled()
        assert not window.terrain_target_param_action.isVisible()
        assert not window.terrain_target_combo.isEnabled()
        assert not window.tool_param_separator_action.isVisible()

        window.mode_combo.setCurrentText("Terrain")
        assert window._current_tool == "eyedropper", "applicable in Terrain too, so it stays"
        assert window.level_param_spin_action.isVisible()
        assert window.terrain_target_param_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()
