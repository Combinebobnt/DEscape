"""GH #387: the trigger panel's property form is the Trigger Details dock.

By default it sits under the editor panel in the left column, shown only in
Triggers mode; it can be dragged anywhere, floated or tabbed with another
dock, and Triggers mode brings it to the front.
"""

from __future__ import annotations

import os

import pytest
from trigger_panel_helpers import (
    _no_install_env,  # noqa: F401 -- autouse fixture
    _sectioned_window,
    _triggers_window,
)

from descape import settings
from descape.viewer_toolbar import _DEFAULT_PANEL_SHARE_PCT, WINDOW_STATE_VERSION
from testkit.qt_window import FONT_DPI_OVERRIDE_ENV

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _front_tab(window, dock) -> str:
    """The title of the tab in front of the tab group `dock` is in."""
    from PyQt5.QtWidgets import QTabBar

    for bar in window.findChildren(QTabBar):
        titles = [bar.tabText(i) for i in range(bar.count())]
        if bar.isVisible() and dock.windowTitle() in titles:
            return bar.tabText(bar.currentIndex())
    raise AssertionError(f"{dock.windowTitle()} is not in a tab group")


def _tab_with_log(window) -> None:
    """Trigger Details tabbed with the Status Log, the log in front."""
    window.tabifyDockWidget(window.trigger_detail_dock, window.log_dock)
    window.log_dock.raise_()
    _settle()
    assert _front_tab(window, window.log_dock) == "Status Log", "fixture: the log starts in front"


def _set_mode(window, label: str) -> None:
    window.mode_combo.setCurrentText(label)
    _settle()


def _assert_fits(widget, name: str) -> None:
    needed = widget.minimumSizeHint().height()
    assert widget.height() >= needed, f"{name} gets {widget.height()} px of its {needed}"


def test_a_fresh_config_docks_the_form_under_the_editor_panel() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        _settle()
        dock = window.trigger_detail_dock
        assert dock.widget() is window.trigger_panel.detail_stack
        assert window.dockWidgetArea(dock) == Qt.LeftDockWidgetArea
        assert not dock.isFloating() and dock.isVisible()
        assert dock.geometry().top() > window.panel_dock.geometry().bottom(), "under the panel"
        assert dock.geometry().left() == window.panel_dock.geometry().left()
        # The entry list and its buttons are one pane, with no bar left over from the form.
        assert window.trigger_panel.entry_tree.parentWidget() is window.trigger_panel.splitter.widget(1)
    finally:
        conftest.close_window(window)


def test_the_dock_shows_only_in_triggers_mode_and_comes_back_when_closed() -> None:
    window = _triggers_window()
    try:
        dock = window.trigger_detail_dock
        action = dock.toggleViewAction()
        assert dock.isVisible() and action.isEnabled() and action.isChecked()
        for mode in ("View", "Units", "Terrain"):
            _set_mode(window, mode)
            assert dock.isHidden(), mode
            assert not action.isEnabled(), f"View > Panels > Trigger Details is live in {mode} mode"
        _set_mode(window, "Triggers")
        assert dock.isVisible() and action.isEnabled()
        action.trigger()
        _settle()
        assert dock.isHidden() and not action.isChecked()
        _set_mode(window, "Units")
        _set_mode(window, "Triggers")
        assert dock.isVisible() and action.isChecked()
    finally:
        conftest.close_window(window)


def test_triggers_mode_brings_a_tabbed_dock_to_the_front() -> None:
    window = _triggers_window()
    try:
        _tab_with_log(window)
        _set_mode(window, "Units")
        assert window.log_dock.isVisible(), "leaving Triggers mode leaves the log showing"
        _set_mode(window, "Triggers")
        assert _front_tab(window, window.log_dock) == "Trigger Details"
    finally:
        conftest.close_window(window)


def test_opening_the_picker_brings_a_tabbed_dock_to_the_front() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        _tab_with_log(window)
        panel.entry_new_button.click()
        _settle()
        assert panel.detail_stack.currentIndex() == panel._DETAIL_PICKER
        assert _front_tab(window, window.log_dock) == "Trigger Details"
        # The picker has its own dock now, so the list it was opened from stays in view, greyed
        # while the picker is latched to it.
        assert panel.entry_tree.isVisible() and panel.entry_new_button.isVisible()
        assert not panel.entry_tree.isEnabled()
        panel._close_picker()
        assert panel.entry_tree.isEnabled()
    finally:
        conftest.close_window(window)


def test_closing_the_dock_cancels_an_open_picker_but_a_tab_behind_does_not() -> None:
    """A closed dock would strand the picker: the entry list greyed, New disabled, nothing to cancel it."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        _tab_with_log(window)
        panel.entry_new_button.click()
        _settle()
        window.log_dock.raise_()
        _settle()
        assert panel.detail_stack.currentIndex() == panel._DETAIL_PICKER, "a tab behind keeps the picker"
        panel.entry_new_button.click()
        _settle()
        window.trigger_detail_dock.toggleViewAction().trigger()
        _settle()
        assert window.trigger_detail_dock.isHidden()
        assert panel.detail_stack.currentIndex() == panel._DETAIL_FORM
        assert panel.entry_tree.isEnabled() and panel.entry_new_button.isEnabled()
    finally:
        conftest.close_window(window)


def test_a_tabbed_placement_holds_across_a_relaunch() -> None:
    window = _triggers_window()
    _tab_with_log(window)
    conftest.close_window(window)
    window = _triggers_window()
    try:
        assert window.log_dock in window.tabifiedDockWidgets(window.trigger_detail_dock)
        assert _front_tab(window, window.log_dock) == "Trigger Details"
    finally:
        conftest.close_window(window)


def test_a_layout_saved_before_the_dock_is_refused_for_the_default() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    window.addDockWidget(Qt.RightDockWidgetArea, window.panel_dock)
    _settle()
    state = bytes(window.saveState(WINDOW_STATE_VERSION - 1))
    conftest.close_window(window)
    settings.set_window_state(state)
    window = _triggers_window()
    try:
        assert window.dockWidgetArea(window.panel_dock) == Qt.LeftDockWidgetArea
        assert window.dockWidgetArea(window.trigger_detail_dock) == Qt.LeftDockWidgetArea
        assert not window.trigger_detail_dock.isFloating()
    finally:
        conftest.close_window(window)


def test_resizing_the_floating_dock_refits_the_form_height() -> None:
    """The panel's resizeEvent no longer sees the form's width, so a narrower
    dock must still re-measure the wrapped rows, or they draw over each other."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        dock = window.trigger_detail_dock
        dock.setFloating(True)
        dock.resize(900, 600)
        _settle()
        wide = panel.property_host.minimumHeight()
        dock.resize(240, 600)
        _settle()
        narrow = panel.property_host.minimumHeight()
        assert narrow > wide, "fixture: the narrow dock wraps rows"
        width = panel.property_area.viewport().width()
        assert narrow >= panel.property_form.heightForWidth(width)
    finally:
        conftest.close_window(window)


def test_the_default_split_gives_the_editor_panel_most_of_the_column() -> None:
    window = _triggers_window()
    try:
        # A fresh config's split is made at first show, in View mode with Trigger Details closed.
        _settle()
        panel, detail = window.panel_dock.height(), window.trigger_detail_dock.height()
        assert abs(panel * 100 // (panel + detail) - _DEFAULT_PANEL_SHARE_PCT) <= 5, f"{panel}:{detail}"
        window.reset_layout_action.trigger()
        _settle()
        panel, detail = window.panel_dock.height(), window.trigger_detail_dock.height()
        assert abs(panel * 100 // (panel + detail) - _DEFAULT_PANEL_SHARE_PCT) <= 1, f"{panel}:{detail}"
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
@pytest.mark.parametrize("height", [900, 720])
def test_no_left_column_widget_goes_below_its_minimum(height: int) -> None:
    """1500x900 is the test window, 720 the app's default height. A larger
    font can raise the window's own minimum past either (before GH #387 too:
    785 px at 120 DPI, 950 at 144), and then something has to give."""
    window = _sectioned_window()
    try:
        window.resize(1500, height)
        window.reset_layout_action.trigger()
        _settle()
        if window.minimumSizeHint().height() > height:
            pytest.skip(f"the window's minimum height is {window.minimumSizeHint().height()} px at this font")
        panel = window.trigger_panel
        assert window.trigger_detail_dock.geometry().top() > window.panel_dock.geometry().bottom(), "the docks overlap"
        for widget, name in (
            (window.trigger_detail_dock, "Trigger Details"),
            (panel.splitter.widget(0), "the trigger pane"),
            (panel.splitter.widget(1), "the entry pane"),
            (panel.tree, "tree"),
            (panel.entry_tree, "entry_tree"),
            (panel.detail_stack, "detail_stack"),
            (panel.property_area, "property_area"),
        ):
            _assert_fits(widget, name)
        if height == 900 and not os.environ.get(FONT_DPI_OVERRIDE_ENV):
            # In rows, not pixels, so a style's row height moves the floor with it.
            rows = panel.tree.viewport().height() // panel.tree.sizeHintForRow(0)
            assert rows >= 6, f"the trigger list shows {rows} whole rows (8 measured)"
    finally:
        conftest.close_window(window)


def test_the_trigger_list_trades_height_with_the_entry_list() -> None:
    """Without the form, the panel's split favours the trigger list 2:1."""
    window = _sectioned_window()
    try:
        _settle()
        top, bottom = window.trigger_panel.splitter.sizes()
        assert top > bottom * 3 // 2, f"{top}:{bottom}"
    finally:
        conftest.close_window(window)
