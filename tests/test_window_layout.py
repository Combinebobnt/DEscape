"""Main-window layout (GH #64): the editor panel and the status log are docks
that float, re-dock and close, and the arrangement persists across launches.

The default shape is pinned too: the panel column runs the full height with
the log under the map beside it, which is easy to regress silently because no
other test reads it.

Every size assertion runs on a shown window after the layout settles:
resizeDocks() is a request that Qt resolves against real geometry.
"""

from __future__ import annotations

import pytest

from descape import settings
from descape.viewer_toolbar import DEFAULT_PANEL_WIDTH, WINDOW_STATE_VERSION
from testkit.qt_window import panel_width, set_panel_width

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# resizeDocks() can land a pixel or two off the request.
_SLACK = 3


def _window():
    """A shown, fixed-size offscreen ViewerWindow."""
    return conftest.shown_window(1500, 900)


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _relaunch(window):
    """Close `window` (which saves the layout) and open the next launch's window."""
    conftest.close_window(window)
    return _window()


def _view_menu(window):
    return next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")


def _assert_default_layout(window) -> None:
    from PyQt5.QtCore import Qt

    assert window.dockWidgetArea(window.panel_dock) == Qt.LeftDockWidgetArea
    assert window.dockWidgetArea(window.log_dock) == Qt.BottomDockWidgetArea
    for dock in (window.panel_dock, window.log_dock):
        assert not dock.isFloating() and dock.isVisible()
    assert abs(panel_width(window) - DEFAULT_PANEL_WIDTH) <= _SLACK
    assert window.main_toolbar.isVisible() and window.tool_options_toolbar.isVisible()


def test_the_panel_column_runs_full_height_with_the_log_under_the_map_beside_it() -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QStyle

    window = _window()
    try:
        _assert_default_layout(window)
        column = window.left_stack
        log = window.status_log
        column_bottom = column.mapTo(window, QPoint(0, column.height())).y()
        log_bottom = log.mapTo(window, QPoint(0, log.height())).y()
        assert column_bottom == log_bottom, "the panel column must reach the log's bottom edge"

        column_right = column.mapTo(window, QPoint(column.width(), 0)).x()
        log_left = log.mapTo(window, QPoint(0, 0)).x()
        map_left = window.map_view.mapTo(window, QPoint(0, 0)).x()
        # Flush against the separator, not merely somewhere to the right of it.
        separator = window.style().pixelMetric(QStyle.PM_DockWidgetSeparatorExtent, None, window)
        assert log_left - column_right == separator
        assert map_left == log_left
    finally:
        conftest.close_window(window)


def test_the_log_height_holds_across_a_window_resize() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.resizeDocks([window.log_dock], [220], Qt.Vertical)
        _settle()
        before = window.status_log.height()
        window.resize(1500, 1200)
        _settle()
        assert window.status_log.height() == before, "the map, not the log, absorbs a window resize"
    finally:
        conftest.close_window(window)


def test_the_log_at_either_extreme_still_leaves_a_usable_map_and_log() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.resizeDocks([window.log_dock], [10_000], Qt.Vertical)
        _settle()
        assert window.map_view.height() >= settings.MIN_MAP_PANE
        window.resizeDocks([window.log_dock], [0], Qt.Vertical)
        _settle()
        assert window.status_log.height() >= settings.MIN_LOG_PANE
    finally:
        conftest.close_window(window)


def test_a_float_and_redock_round_trip_keeps_the_pages_in_the_panel() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.panel_dock.setFloating(True)
        _settle()
        assert window.panel_dock.isFloating() and window.panel_dock.isWindow()
        window.mode_combo.setCurrentText("Triggers")
        assert window.left_stack.currentWidget() is window.trigger_panel
        window.panel_dock.setFloating(False)
        _settle()
        assert window.dockWidgetArea(window.panel_dock) == Qt.LeftDockWidgetArea
        assert window.panel_dock.widget() is window.left_stack
        window.mode_combo.setCurrentText("Units")
        assert window.left_stack.currentWidget() is window.units_panel
        assert window.units_panel.isVisible()
    finally:
        conftest.close_window(window)


def test_the_arrangement_is_restored_on_the_next_launch() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    window.addDockWidget(Qt.RightDockWidgetArea, window.panel_dock)
    set_panel_width(window, 500)
    window.log_dock.toggleViewAction().trigger()
    _settle()
    assert window.log_dock.isHidden()
    window = _relaunch(window)
    try:
        assert settings.get_window_state() is not None
        assert window.dockWidgetArea(window.panel_dock) == Qt.RightDockWidgetArea
        assert abs(panel_width(window) - 500) <= _SLACK
        assert window.log_dock.isHidden()
        assert not window.log_dock.toggleViewAction().isChecked()
    finally:
        conftest.close_window(window)


def test_a_floating_panel_comes_back_floating_and_hides_with_its_window() -> None:
    window = _window()
    window.panel_dock.setFloating(True)
    window.panel_dock.move(40, 50)
    _settle()
    conftest.close_window(window)
    assert not window.panel_dock.isVisible(), "a closed window's floating panel must not stay on screen"
    window = _window()
    try:
        assert window.panel_dock.isFloating() and window.panel_dock.isVisible()
    finally:
        conftest.close_window(window)


def test_a_float_saved_off_every_screen_comes_back_on_one() -> None:
    """A floating panel left on a monitor that is gone. Qt 5.15 clamps a
    restored float onto a screen itself; this pins that, so no guard of our own."""
    from PyQt5.QtGui import QGuiApplication

    window = _window()
    window.panel_dock.setFloating(True)
    window.panel_dock.move(20_000, 20_000)
    _settle()
    window = _relaunch(window)
    try:
        frame = window.panel_dock.frameGeometry()
        assert window.panel_dock.isFloating()
        assert any(screen.availableGeometry().contains(frame.topLeft()) for screen in QGuiApplication.screens())
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("blob", ["garbage", "newer version"])
def test_a_state_that_will_not_restore_falls_back_to_the_default_layout(blob: str) -> None:
    from PyQt5.QtCore import Qt

    if blob == "garbage":
        settings.set_window_state(b"\x00not a saved state")
    else:
        window = _window()
        window.addDockWidget(Qt.RightDockWidgetArea, window.panel_dock)
        window.log_dock.hide()
        _settle()
        state = bytes(window.saveState(WINDOW_STATE_VERSION + 1))
        conftest.close_window(window)
        settings.set_window_state(state)
    window = _window()
    try:
        _assert_default_layout(window)
    finally:
        conftest.close_window(window)


def test_the_first_launch_after_the_splitter_layout_keeps_its_widths(tmp_path) -> None:
    """The pre-GH #64 config held split_sizes and log_height; they seed the docks once."""
    (tmp_path / "config.yaml").write_text("split_sizes: [520, 900]\nlog_height: 150\n")
    window = _window()
    try:
        assert abs(panel_width(window) - 520) <= _SLACK
        assert abs(window.status_log.height() - 150) <= _SLACK
    finally:
        conftest.close_window(window)


def test_reset_layout_restores_the_default_from_a_scrambled_one() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.addDockWidget(Qt.RightDockWidgetArea, window.panel_dock)
        set_panel_width(window, 600)
        window.log_dock.setFloating(True)
        window.panel_dock_action.trigger()
        window.main_toolbar.hide()
        _settle()
        assert window.panel_dock.isHidden() and not window.panel_dock_action.isChecked()
        window.reset_layout_action.trigger()
        _settle()
        _assert_default_layout(window)
        assert window.panel_dock_action.isChecked()
    finally:
        conftest.close_window(window)


def test_a_mode_change_titles_the_panel_and_reshows_it_when_closed() -> None:
    window = _window()
    try:
        assert window.panel_dock.windowTitle() == "View"
        window.panel_dock.close()
        _settle()
        assert not window.panel_dock_action.isChecked()
        window.mode_combo.setCurrentText("Map Options")
        _settle()
        assert window.panel_dock.windowTitle() == "Map Options"
        assert window.panel_dock.isVisible() and window.panel_dock_action.isChecked()
        panels = next(a.menu() for a in _view_menu(window).actions() if a.text().replace("&", "") == "Panels")
        # The panel's entry keeps its own name; the dock's toggleViewAction() would read "Map Options".
        assert [a.text().replace("&", "") for a in panels.actions() if not a.isSeparator()] == [
            "Editor Panel",
            "Status Log",
            "Reset Layout",
        ]
    finally:
        conftest.close_window(window)


def test_widening_on_mode_entry_docked_floating_and_closed() -> None:
    from descape.trigger_panel import TriggerPanel

    window = _window()
    try:
        set_panel_width(window, settings.MIN_SPLIT_PANE)
        window.mode_combo.setCurrentText("Triggers")
        _settle()
        assert panel_width(window) >= TriggerPanel.MIN_USEFUL_WIDTH

        window.mode_combo.setCurrentText("View")
        window.panel_dock.setFloating(True)
        window.panel_dock.resize(settings.MIN_SPLIT_PANE, 500)
        _settle()
        window.mode_combo.setCurrentText("Triggers")
        _settle()
        assert window.panel_dock.isFloating()
        assert window.panel_dock.width() >= TriggerPanel.MIN_USEFUL_WIDTH

        window.mode_combo.setCurrentText("View")
        window.panel_dock.setFloating(False)
        set_panel_width(window, settings.MIN_SPLIT_PANE)
        window.panel_dock.hide()
        _settle()
        before = window.panel_dock.width()
        window._widen_left_column(TriggerPanel.MIN_USEFUL_WIDTH)
        _settle()
        assert window.panel_dock.isHidden() and window.panel_dock.width() == before
    finally:
        conftest.close_window(window)
