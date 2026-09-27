"""GH #38 residual: Edit > Undo/Redo while a multi-line box has focus.

While a text_edits._MultiLineEdit box is focused, the two menu actions act on
the box's own document undo stack and follow its availability; the commit to
EditHistory still happens once, on focus-out. Every check runs against a
trigger's Display Instructions message box and a Messages-mode box.

Needs a shown, activated offscreen window: Qt's focus tracking is a no-op on a
never-activated one (tests/test_unit_edit_viewer.py makes the same point).
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BOXES = ["trigger message", "messages hints"]


def _box_window(which: str):
    """A shown, active window with one editable multi-line box in view."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication
    from test_trigger_panel import TRIGGER_FIXTURE, _plant_prose_effect, _row_widget

    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.text_edits import _MultiLineEdit

    window = conftest.shown_window(1500, 900)
    if which == "trigger message":
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        _plant_prose_effect(window, "hold the ford")
        box = _row_widget(window.trigger_panel, "message")
    else:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.mode_combo.setCurrentText("Messages")
        box = window.messages_panel.widget_for("hints")
    assert isinstance(box, _MultiLineEdit) and box.isEnabled()
    # The trigger form's unparented old rows are only deleteLater()'d; left
    # pending, offscreen activation can land on one of them instead.
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
    return window, box


def _focus(widget) -> None:
    from PyQt5.QtWidgets import QApplication

    widget.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget


def _type(box, text: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QTextCursor
    from PyQt5.QtTest import QTest

    box.moveCursor(QTextCursor.End)
    QTest.keyClicks(box, text)
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt):
        QTest.keyRelease(box, key)


def _history(window) -> tuple[int, int]:
    return len(window.edit_history.records), window.edit_history.cursor


@pytest.mark.parametrize("which", BOXES)
def test_menu_undo_follows_the_focused_boxs_own_stack(which: str) -> None:
    window, box = _box_window(which)
    try:
        _focus(box)
        assert not window.undo_action.isEnabled(), "a fresh box has nothing to undo"
        _type(box, " xyz")
        assert window.undo_action.isEnabled()
        assert not window.redo_action.isEnabled()
        while box.document().isUndoAvailable():
            window.undo_action.trigger()
        assert not window.undo_action.isEnabled()
        assert window.redo_action.isEnabled()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_menu_undo_and_redo_edit_the_box_without_touching_edit_history(which: str) -> None:
    window, box = _box_window(which)
    try:
        _focus(box)
        before_text = box.toPlainText()
        before_history = _history(window)
        _type(box, " xyz")
        typed = box.toPlainText()
        assert typed != before_text

        window.undo_action.trigger()
        assert box.toPlainText() != typed
        window.redo_action.trigger()
        assert box.toPlainText() == typed
        assert _history(window) == before_history
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_opening_the_edit_menu_neither_commits_nor_greys_undo(which: str) -> None:
    """The trap: a popup focus-out must not commit the box, and focusWidget()
    stays on it, so Undo is still live in the open menu."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QMenu

    window, box = _box_window(which)
    try:
        _focus(box)
        finished = []
        box.editingFinished.connect(lambda: finished.append(True))
        _type(box, " xyz")
        before_history = _history(window)

        edit_menu = next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Edit")
        edit_menu.popup(window.mapToGlobal(QPoint(10, 10)))
        QApplication.processEvents()
        assert finished == []
        assert window.undo_action.isEnabled()
        window.undo_action.trigger()
        edit_menu.hide()
        QApplication.processEvents()

        assert finished == []
        assert _history(window) == before_history
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_focus_out_still_records_exactly_one_edit(which: str) -> None:
    window, box = _box_window(which)
    try:
        _focus(box)
        records_before, _cursor = _history(window)
        _type(box, " xyz")
        window.undo_action.trigger()
        window.redo_action.trigger()
        _focus(window.map_view)
        records_after, cursor_after = _history(window)
        assert records_after == records_before + 1
        assert cursor_after == records_after
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_with_focus_back_on_the_map_the_menu_follows_edit_history(which: str) -> None:
    window, box = _box_window(which)
    try:
        _focus(box)
        _type(box, " xyz")
        _focus(window.map_view)
        assert window._focused_text_box() is None
        assert window.undo_action.isEnabled(), "the focus-out commit is undoable"

        window.undo_action.trigger()
        assert window.edit_history.cursor == len(window.edit_history.records) - 1
        assert window.undo_action.isEnabled() == window.edit_history.can_undo
        assert window.redo_action.isEnabled()
        window.redo_action.trigger()
        assert window.edit_history.cursor == len(window.edit_history.records)
    finally:
        conftest.close_window(window)


def test_a_box_in_another_window_does_not_drive_this_windows_menu() -> None:
    """focusChanged is app-global; each window only follows its own boxes."""
    first, _first_box = _box_window("messages hints")
    second, second_box = _box_window("messages hints")
    try:
        _focus(second_box)
        _type(second_box, " xyz")
        assert second.undo_action.isEnabled()
        assert first._focused_text_box() is None
        assert not first.undo_action.isEnabled()
    finally:
        conftest.close_window(second)
        conftest.close_window(first)
