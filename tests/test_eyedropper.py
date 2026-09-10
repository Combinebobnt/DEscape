"""Eyedropper tool wiring, driven through a real offscreen ViewerWindow --
same technique and default-tier rationale as tests/test_fill_tool.py.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_PICK_TERRAIN = 15  # GRASS_1, distinct from the blank template's terrain_id=0
_PICK_ELEVATION = 3


def _edit_window():
    """Loads the blank template and switches to Terrain mode with Eyedropper
    active. Caller must edit_history.mark_saved() + close()."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
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

        assert window.terrain_combo.currentData() == _PICK_TERRAIN
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


def test_pick_out_of_enum_terrain_leaves_combo_unchanged() -> None:
    """The corruption guard: findData() returns -1 for a terrain id the
    picker doesn't list, and setCurrentIndex(-1) would blank the combo,
    making currentData() return None -- which Draw/Fill would then write
    straight into a tile's terrain_id."""
    window = _edit_window()
    try:
        mm = window.scenario.map_manager
        mm.terrain[0].terrain_id = 9999
        before = window.terrain_combo.currentData()

        window.pick_tile_value(0, 0, 0)

        assert window.terrain_combo.currentData() == before
        assert window.terrain_combo.currentData() is not None
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
    feature: param_widget == "" would otherwise hide both Terrain type and
    Level right when the tool writes into them."""
    window = _edit_window()
    try:
        assert window.terrain_param_combo_action.isVisible()
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

        assert window.terrain_combo.currentData() == _PICK_TERRAIN
        assert window.elevation_level_spin.value() == _PICK_ELEVATION
    finally:
        window.edit_history.mark_saved()
        window.close()
