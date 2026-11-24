"""GH #38 residual: Edit > Undo/Redo while a multi-line box has focus.

While a text_edits._MultiLineEdit box is focused, the two menu actions act on
the box's own document undo stack and follow its availability; the commit to
EditHistory still happens once, on focus-out. Every check runs against a
trigger's Display Instructions message box and a Messages-mode box.

Needs a shown, activated offscreen window: Qt's focus tracking is a no-op on a
never-activated one (tests/test_unit_edit_viewer_place_move.py makes the same point).
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
    from trigger_panel_helpers import TRIGGER_FIXTURE, _plant_prose_effect, _row_widget

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


def _edit_menu(window):
    from PyQt5.QtWidgets import QMenu

    return next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Edit")


def _leave_the_menu_bar(window) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    _edit_menu(window).hide()
    QTest.keyClick(window.menuBar(), Qt.Key_Escape)
    QApplication.processEvents()


@pytest.mark.parametrize("which", BOXES)
def test_a_mouse_click_on_the_edit_menu_title_commits_nothing(which: str) -> None:
    """TASK-418: a spontaneous press through the window handle, so QMenuBar's own press path and
    Qt's click-to-focus step run, which popup() skips. The box stays focused and uncommitted."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window, box = _box_window(which)
    try:
        _focus(box)
        finished = []
        box.editingFinished.connect(lambda: finished.append(True))
        _type(box, " xyz")
        before_history = _history(window)

        bar = window.menuBar()
        edit_menu = _edit_menu(window)
        pos = bar.mapTo(window, bar.actionGeometry(edit_menu.menuAction()).center())
        QTest.mouseClick(window.windowHandle(), Qt.LeftButton, Qt.NoModifier, pos)
        QApplication.processEvents()
        assert edit_menu.isVisible(), "the click did not open the Edit menu"
        assert finished == []
        assert _history(window) == before_history
        assert window._focused_text_box() is box
    finally:
        _leave_the_menu_bar(window)
        conftest.close_window(window)


def _refetch_box(window, which: str):
    """The box again: an undo rebuilds the form."""
    from trigger_panel_helpers import _row_widget

    if which == "trigger message":
        return _row_widget(window.trigger_panel, "message")
    return window.messages_panel.widget_for("hints")


def _click_the_edit_title(window) -> None:
    """A spontaneous press through the window handle, as TASK-418's live runs clicked it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    bar = window.menuBar()
    pos = bar.mapTo(window, bar.actionGeometry(_edit_menu(window).menuAction()).center())
    QTest.mouseClick(window.windowHandle(), Qt.LeftButton, Qt.NoModifier, pos)
    QApplication.processEvents()


def _enter_the_menu_bar(window, entry: str) -> None:
    """The bar's keyboard mode, by Escape out of a mouse-opened Edit menu or a bare Alt tap (TASK-648)."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    if entry == "escape":
        _click_the_edit_title(window)
        assert _edit_menu(window).isVisible(), "the click did not open the Edit menu"
        QTest.keyClick(QApplication.activePopupWidget(), Qt.Key_Escape)
    else:
        target = QApplication.focusWidget()
        QTest.keyPress(target, Qt.Key_Alt)
        QTest.keyRelease(target, Qt.Key_Alt)
    QApplication.processEvents()
    assert QApplication.activePopupWidget() is None
    assert QApplication.focusWidget() is window.menuBar()


def _click_the_map(window) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    view = window.map_view
    QTest.mouseClick(window.windowHandle(), Qt.LeftButton, Qt.NoModifier, view.mapTo(window, view.rect().center()))
    QApplication.processEvents()
    assert QApplication.focusWidget() is view


@pytest.mark.parametrize("which", BOXES)
def test_escape_out_of_a_mouse_opened_edit_menu_commits_nothing(which: str) -> None:
    """TASK-648: Escape leaves the bar in keyboard mode with focus (MenuBarFocusReason), which holds
    the box uncommitted as the Undo target; a second Escape hands focus back to it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window, box = _box_window(which)
    try:
        _focus(box)
        finished = []
        box.editingFinished.connect(lambda: finished.append(True))
        _type(box, " xyz")
        before_history = _history(window)

        _enter_the_menu_bar(window, "escape")
        assert finished == []
        assert _history(window) == before_history
        assert window.focusWidget() is window.menuBar()
        assert window._focused_text_box() is box

        QTest.keyClick(window.menuBar(), Qt.Key_Escape)
        QApplication.processEvents()
        assert QApplication.focusWidget() is box
        assert finished == []
        assert _history(window) == before_history
        assert box.toPlainText().endswith(" xyz")
    finally:
        _leave_the_menu_bar(window)
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_alt_e_holds_the_box_and_undo_follows_its_stack(which: str) -> None:
    """TASK-648 retires TASK-125's coarser path: Alt+E commits nothing, and Undo with the bar holding
    focus (a window shortcut fires there) steps back the box's own typing, not EditHistory."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window, box = _box_window(which)
    try:
        _focus(box)
        before_text = box.toPlainText()
        before_history = _history(window)
        _type(box, " xyz")

        QTest.keyClick(box, Qt.Key_E, Qt.AltModifier)
        QApplication.processEvents()
        assert _edit_menu(window).isVisible(), "Alt+E did not open the Edit menu"
        assert QApplication.focusWidget() is window.menuBar()
        assert _history(window) == before_history, "Alt+E should commit nothing"
        assert window._focused_text_box() is box
        assert window.undo_action.isEnabled()

        window.undo_action.trigger()
        QApplication.processEvents()
        assert box.toPlainText() == before_text
        assert _history(window) == before_history
    finally:
        _leave_the_menu_bar(window)
        conftest.close_window(window)


@pytest.mark.parametrize("entry", ["escape", "alt"])
@pytest.mark.parametrize("which", BOXES)
def test_leaving_the_menu_bar_for_the_map_commits_the_box_once(which: str, entry: str) -> None:
    from PyQt5.QtWidgets import QApplication

    window, box = _box_window(which)
    try:
        _focus(box)
        before_text = box.toPlainText()
        records, cursor = _history(window)
        _type(box, " xyz")
        typed = box.toPlainText()

        _enter_the_menu_bar(window, entry)
        assert _history(window) == (records, cursor)
        _click_the_map(window)
        assert _history(window) == (records + 1, cursor + 1), "the map click should commit the held box once"
        assert window._focused_text_box() is None

        # The one record holds the typed text: Undo reverts to the old text and Redo brings it back.
        window.undo_action.trigger()
        QApplication.processEvents()
        assert _refetch_box(window, which).toPlainText() == before_text
        window.redo_action.trigger()
        QApplication.processEvents()
        assert _refetch_box(window, which).toPlainText() == typed
    finally:
        _leave_the_menu_bar(window)
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_a_mode_switch_from_the_menu_bar_commits_the_held_box(which: str) -> None:
    """A mode shortcut fires while the bar holds the box; the switch hides it, so it commits there."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window, box = _box_window(which)
    try:
        _focus(box)
        records, cursor = _history(window)
        _type(box, " xyz")
        _enter_the_menu_bar(window, "alt")
        window.mode_combo.setCurrentText(settings.MODE_LABELS["view"])
        QApplication.processEvents()
        assert _history(window) == (records + 1, cursor + 1), "the switch should commit the held box once"
        assert window._focused_text_box() is None, "a hidden box must not stay the Undo target"
        assert window._menu_bar_held_box() is None
    finally:
        _leave_the_menu_bar(window)
        conftest.close_window(window)


@pytest.mark.parametrize("which", BOXES)
def test_a_window_switch_keeps_the_held_box_pending(which: str) -> None:
    """TASK-648, G5: the bar loses focus to no widget and gets it back; a switch never commits a box."""
    from PyQt5.QtWidgets import QApplication, QWidget

    window, box = _box_window(which)
    other = QWidget()
    try:
        _focus(box)
        records, cursor = _history(window)
        _type(box, " xyz")

        _enter_the_menu_bar(window, "alt")
        other.show()
        other.activateWindow()
        QApplication.setActiveWindow(other)
        QApplication.processEvents()
        assert not window.isActiveWindow()
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()
        assert _history(window) == (records, cursor)
        assert window._focused_text_box() is box

        _click_the_map(window)
        assert _history(window) == (records + 1, cursor + 1)
    finally:
        other.close()
        other.deleteLater()
        _leave_the_menu_bar(window)
        conftest.close_window(window)


def test_paste_from_the_menu_bar_targets_the_held_box() -> None:
    """TASK-648: in Triggers mode Edit > Cut/Copy/Paste follow the held box, not the trigger list."""
    window, box = _box_window("trigger message")
    try:
        _focus(box)
        assert window.paste_action.text() == "&Paste"
        _enter_the_menu_bar(window, "alt")
        assert window.paste_action.text() == "&Paste"
        assert window._focused_editor() is box
    finally:
        _leave_the_menu_bar(window)
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


@pytest.mark.parametrize("which", BOXES)
def test_a_grip_drag_leaves_the_focused_boxs_undo_stack_and_edit_history_alone(which: str) -> None:
    """GH #139: resizing is not an edit. The box keeps focus, its own stack
    and text are untouched, and menu Undo still steps the typing back."""
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    window, box = _box_window(which)
    try:
        _focus(box)
        _type(box, " xyz")
        typed = box.toPlainText()
        steps = box.document().availableUndoSteps()
        before_history = _history(window)

        _drag(box.grip(), 3)
        QApplication.processEvents()
        assert box.visible_lines() == box.VISIBLE_LINES + 3
        assert QApplication.focusWidget() is box
        assert box.toPlainText() == typed
        assert box.document().availableUndoSteps() == steps
        assert _history(window) == before_history
        window.undo_action.trigger()
        assert box.toPlainText() != typed, "menu Undo still acts on the box's typing"
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
