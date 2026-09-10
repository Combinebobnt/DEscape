"""The Select tool's own registration/gating/gesture coverage -- phase 2.8.
Same offscreen technique tests/test_ruler_viewer.py documents (QT_QPA_
PLATFORM=offscreen, a real ViewerWindow, MapView's own mouse handlers driven
directly rather than through the session), since region_clipboard.py's own
pure tests (tests/test_region_clipboard.py) cannot reach any of the Qt
wiring: toolbar registration, keybind rows, mode-applicability hiding, the
paste-filter checkboxes' visibility, or the drag gesture itself.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape import settings, viewer_common
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _select_window(style: str = "Flat"):
    """A shown window with the blank template loaded and Select active.
    Caller must edit_history.mark_saved() + close()."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window.terrain_style_combo.setCurrentText(style)
    window._on_tool_selected("select")
    window.show()
    QApplication.processEvents()
    return window


def _viewport_pos(map_view, tile_x: int, tile_y: int):
    from PyQt5.QtCore import QPointF

    polygon = map_view._tile_polygon(tile_x, tile_y)
    assert polygon is not None, f"no footprint for ({tile_x}, {tile_y})"
    return QPointF(map_view.mapFromScene(polygon.boundingRect().center()))


def _mouse_event(kind, pos, button, buttons):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)


def _press(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mousePressEvent(
        _mouse_event(QEvent.MouseButtonPress, _viewport_pos(map_view, *tile), Qt.LeftButton, Qt.LeftButton)
    )


def _move(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseMoveEvent(
        _mouse_event(QEvent.MouseMove, _viewport_pos(map_view, *tile), Qt.NoButton, Qt.LeftButton)
    )


def _release(map_view, tile: tuple[int, int]) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseReleaseEvent(
        _mouse_event(QEvent.MouseButtonRelease, _viewport_pos(map_view, *tile), Qt.LeftButton, Qt.NoButton)
    )


def _drag(map_view, a: tuple[int, int], b: tuple[int, int]) -> None:
    _press(map_view, a)
    _move(map_view, b)
    _release(map_view, b)


# -- registration ------------------------------------------------------------


def test_select_is_a_registered_tool() -> None:
    tool_ids = [t.tool_id for t in settings.TOOLS]
    assert "select" in tool_ids
    select = next(t for t in settings.TOOLS if t.tool_id == "select")
    assert select.modes == ("terrain",)
    assert not select.is_edit_tool
    assert select.default_key == "S"


def test_keybind_rows_exist_with_the_expected_defaults() -> None:
    assert settings.get_keybind("tool_select") == "S"
    assert settings.get_keybind("edit_select_all") == "Ctrl+A"
    assert settings.get_keybind("edit_deselect") == "Ctrl+Shift+A"


def test_mode_gating_hides_select_outside_terrain_mode() -> None:
    window = _select_window()
    try:
        assert viewer_common.tool_applicable("select", "terrain")
        assert not viewer_common.tool_applicable("select", "units")
        assert not viewer_common.tool_applicable("select", "view")
        for mode_text, mode_id in [("Units", "units"), ("View", "view")]:
            window.mode_combo.setCurrentText(mode_text)
            assert not window.select_action.isVisible(), mode_id
        window.mode_combo.setCurrentText("Terrain")
        assert window.select_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paste_filter_checkboxes_visible_only_while_select_active() -> None:
    window = _select_window()
    try:
        assert window.paste_terrain_param_action.isVisible()
        assert window.paste_elevation_param_action.isVisible()
        assert window.paste_units_param_action.isVisible()
        assert window.tool_param_separator_action.isVisible()

        window._on_tool_selected("draw")
        assert not window.paste_terrain_param_action.isVisible()
        assert not window.paste_elevation_param_action.isVisible()
        assert not window.paste_units_param_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the drag gesture ---------------------------------------------------------


def test_drag_commits_a_region_matching_the_two_corners() -> None:
    window = _select_window()
    try:
        _drag(window.map_view, (10, 10), (13, 15))
        assert window._region == (10, 10, 14, 16)
        assert window.map_view._region == (10, 10, 14, 16)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_press_release_with_no_movement_selects_one_tile() -> None:
    window = _select_window()
    try:
        _drag(window.map_view, (5, 5), (5, 5))
        assert window._region == (5, 5, 6, 6)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_all_and_deselect() -> None:
    window = _select_window()
    try:
        mm = window.scenario.map_manager
        window.select_all()
        assert window._region == (0, 0, mm.map_width, mm.map_height)
        assert window.copy_action.isEnabled()

        window.deselect()
        assert window._region is None
        assert not window.copy_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_all_and_deselect_no_op_silently_with_no_scenario() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.select_all()
        window.deselect()
        assert window._region is None
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- Escape ordering: cancel a drag before clearing a committed region -------


def test_escape_cancels_an_in_progress_drag_without_touching_a_prior_region() -> None:
    window = _select_window()
    try:
        window.select_all()
        mm = window.scenario.map_manager
        whole_map = (0, 0, mm.map_width, mm.map_height)
        assert window._region == whole_map

        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent

        _press(window.map_view, (10, 10))
        _move(window.map_view, (12, 12))
        assert window.map_view._select_anchor is not None

        window.map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert window.map_view._select_anchor is None
        # The drag was cancelled, not committed -- the PRIOR committed
        # region (whole map, from select_all() above) must survive.
        assert window._region == whole_map
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_escape_clears_a_committed_region_when_no_drag_is_in_progress() -> None:
    window = _select_window()
    try:
        _drag(window.map_view, (10, 10), (12, 12))
        assert window._region is not None

        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent

        window.map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert window._region is None
        assert window.map_view._region is None
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- region persistence -------------------------------------------------------


def test_region_survives_a_style_switch() -> None:
    window = _select_window(style="Stepped")
    try:
        _drag(window.map_view, (10, 10), (12, 12))
        region = window._region
        assert region is not None

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window._region == region
        assert window.map_view._region == region
        # The overlay itself must have been rebuilt against the new
        # projection, not merely left as a stale reference to destroyed
        # scene items (scene().clear() runs on every style switch).
        assert window.map_view._region_fill_item is not None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_region_does_not_survive_a_new_document() -> None:
    window = _select_window()
    try:
        _drag(window.map_view, (10, 10), (12, 12))
        assert window._region is not None

        window.load_scenario(BLANK_TEMPLATE_PATH, untitled=True)
        assert window._region is None
        assert window.map_view._region is None
    finally:
        window.edit_history.mark_saved()
        window.close()
