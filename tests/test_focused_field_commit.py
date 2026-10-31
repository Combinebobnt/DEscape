"""TASK-505: Save and the unsaved-changes prompts commit a focused field first.

A field commits its typed value when focus leaves it. A shortcut, a window
close and a programmatic call leave no field, so these routes commit it
themselves through ViewerWindow._commit_focused_field().

Needs a shown, activated offscreen window, as tests/test_text_box_undo_menu.py
does: Qt's focus tracking is a no-op on a never-activated one.
"""

from __future__ import annotations

import shutil

import pytest
from test_autosave_viewer import _dirty, _slots
from test_text_box_undo_menu import _focus, _history, _type

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

MARKER = "probexyz"
SPIN_VALUE = 7
BOX_KINDS = ["hints box", "trigger message box"]
KINDS = [*BOX_KINDS, "trigger name", "script_name", "ai_map_type"]


def _window_on(tmp_path, kind: str):
    """A shown, active window on a copy of the fixture, with `kind`'s field in view."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication
    from test_trigger_panel import TRIGGER_FIXTURE, _plant_prose_effect, _row_widget

    from descape.scenario_io import BLANK_TEMPLATE_PATH

    path = tmp_path / "doc.aoe2scenario"
    shutil.copy(TRIGGER_FIXTURE if kind.startswith("trigger") else BLANK_TEMPLATE_PATH, path)
    window = conftest.shown_window(1500, 900)
    window.load_scenario(path)
    if kind == "hints box":
        window.mode_combo.setCurrentText("Messages")
        widget = window.messages_panel.widget_for("hints")
    elif kind.startswith("trigger"):
        window.mode_combo.setCurrentText("Triggers")
        if kind == "trigger message box":
            _plant_prose_effect(window, "hold the ford")
            widget = _row_widget(window.trigger_panel, "message")
        else:
            window.trigger_panel.select_trigger(0)
            widget = _row_widget(window.trigger_panel, "name")
    else:
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for(kind)
    # As in test_text_box_undo_menu: pending old rows can take the offscreen activation.
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
    return window, widget, path


def _type_into(kind: str, widget) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    _focus(widget)
    if kind in BOX_KINDS:
        _type(widget, f" {MARKER}")
        return
    if kind == "ai_map_type":
        widget.lineEdit().selectAll()
        QTest.keyClicks(widget, str(SPIN_VALUE))
    else:
        widget.end(False)
        QTest.keyClicks(widget, MARKER)
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt):
        QTest.keyRelease(widget, key)


def _press_save_key(window, widget) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    combo = window.save_action.shortcut()[0]
    modifiers = int(Qt.KeyboardModifierMask)
    QTest.keyClick(widget, combo & ~modifiers, Qt.KeyboardModifiers(combo & modifiers))
    QApplication.processEvents()


def _saved_value(path, kind: str):
    """`kind`'s value in the file on disk, read by a second window. The trigger
    box goes through the parsed manager, so nothing re-plants its effect."""
    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(path)
        if kind.startswith("trigger"):
            window.mode_combo.setCurrentText("Triggers")
            trigger = window.trigger_panel._manager().triggers[0]
            return trigger.effects[-1].message if kind == "trigger message box" else trigger.name
        if kind == "hints box":
            window.mode_combo.setCurrentText("Messages")
            return window.messages_panel.current_values()["hints"]
        window.mode_combo.setCurrentText("Map Options")
        return window.map_options_panel.current_values()[kind]
    finally:
        conftest.close_window(window)


def _recording_question(monkeypatch) -> list[str]:
    """Each prompt's text, answered Cancel. Never raises: an exception out of
    closeEvent() would abort the process rather than fail the test."""
    from PyQt5.QtWidgets import QMessageBox

    asked: list[str] = []

    def question(_parent, _title, text, *_args, **_kwargs):
        asked.append(text)
        return QMessageBox.Cancel

    monkeypatch.setattr("descape.viewer.QMessageBox.question", question)
    return asked


def _close_if_open(window) -> None:
    if window.isVisible():
        conftest.close_window(window)


@pytest.mark.parametrize("kind", KINDS)
def test_the_save_key_writes_a_focused_fields_typed_value(tmp_path, kind: str) -> None:
    from PyQt5.QtWidgets import QApplication

    window, widget, path = _window_on(tmp_path, kind)
    try:
        _type_into(kind, widget)
        records = len(window.edit_history.records)
        _press_save_key(window, widget)

        assert not window.windowTitle().startswith("*")
        assert len(window.edit_history.records) == records + 1
        assert not window.edit_history.is_dirty
        assert QApplication.focusWidget() is widget
        if kind in BOX_KINDS:
            assert window._focused_text_box() is widget
    finally:
        conftest.close_window(window)

    saved = _saved_value(path, kind)
    assert saved == SPIN_VALUE if kind == "ai_map_type" else MARKER in saved


def test_file_save_from_the_menu_writes_a_focused_boxs_typed_value(tmp_path) -> None:
    """The popup's focus-out is exempt in the box, so the menu route commits nothing by itself."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    window, box, path = _window_on(tmp_path, "hints box")
    try:
        _type_into("hints box", box)
        before = _history(window)
        file_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&File")
        file_menu.popup(QPoint(0, 0))
        QApplication.processEvents()
        assert _history(window) == before

        window.save_action.trigger()
        file_menu.hide()
        QApplication.processEvents()
        assert not window.edit_history.is_dirty
    finally:
        conftest.close_window(window)

    assert MARKER in _saved_value(path, "hints box")


def test_save_as_writes_a_focused_boxs_typed_value(tmp_path, monkeypatch) -> None:
    target = tmp_path / "saved_as.aoe2scenario"
    monkeypatch.setattr("descape.viewer.QFileDialog.getSaveFileName", lambda *_a, **_k: (str(target), ""))
    window, box, _path = _window_on(tmp_path, "hints box")
    try:
        _type_into("hints box", box)
        window.save_as()
        assert not window.edit_history.is_dirty
    finally:
        conftest.close_window(window)

    assert MARKER in _saved_value(target, "hints box")


PROMPT_ROUTES = {
    "window close": lambda window: window.close(),
    "Close Map": lambda window: window.close_scenario(),
    "New Map": lambda window: window.new_map(),
    "Open": lambda window: window.open_file(),
    "Resize Map": lambda window: window.resize_map(),
}


@pytest.mark.parametrize(
    ("route", "kind"),
    [(route, "hints box") for route in PROMPT_ROUTES]
    + [("window close", "script_name"), ("window close", "ai_map_type")],
)
def test_an_unsaved_changes_prompt_counts_a_focused_fields_typed_value(
    tmp_path, monkeypatch, route: str, kind: str
) -> None:
    from PyQt5.QtWidgets import QApplication, QDialog

    asked = _recording_question(monkeypatch)
    # A route that fails to ask must fail the test, not block in its next dialog.
    monkeypatch.setattr("descape.viewer.QFileDialog.getOpenFileName", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr("descape.viewer.ResizeDialog.exec_", lambda _self: QDialog.Rejected)
    window, widget, path = _window_on(tmp_path, kind)
    try:
        _type_into(kind, widget)
        PROMPT_ROUTES[route](window)
        QApplication.processEvents()

        assert len(asked) == 1, f"asked {asked}"
        assert window.isVisible()
        assert window.scenario is not None and window.scenario.path == path
        assert window.windowTitle().startswith("*")
        assert QApplication.focusWidget() is widget
    finally:
        _close_if_open(window)


def test_a_locked_documents_prompt_leaves_the_typed_value_for_the_save_after_it_unlocks(tmp_path, monkeypatch) -> None:
    from test_trigger_panel import _row_widget

    from descape import game_dataset, settings

    _recording_question(monkeypatch)
    kind = "trigger message box"
    window, _widget, path = _window_on(tmp_path, kind)
    try:
        window.set_game_data(game_dataset.ROR)
        assert window.game_data_lock_reason()
        widget = _row_widget(window.trigger_panel, "message")
        _type_into(kind, widget)
        assert window._confirm_discard_changes()
        window.set_game_data(settings.GAME_DATA_DEFAULT)
        widget = _row_widget(window.trigger_panel, "message")
        assert MARKER in widget.toPlainText(), "the switch back keeps the typed text"
        _focus(widget)

        _press_save_key(window, widget)
        assert MARKER in _saved_value(path, kind)
    finally:
        window.set_game_data(settings.GAME_DATA_DEFAULT)
        _close_if_open(window)


@pytest.mark.parametrize("kind", KINDS)
def test_an_untouched_focused_field_adds_no_record_and_no_prompt(tmp_path, monkeypatch, kind: str) -> None:
    asked = _recording_question(monkeypatch)
    window, widget, _path = _window_on(tmp_path, kind)
    try:
        _focus(widget)
        before = _history(window)
        window.close()

        assert asked == []
        assert _history(window) == before
        assert not window.isVisible()
    finally:
        _close_if_open(window)


def test_an_autosave_tick_leaves_a_focused_fields_typed_value_pending(tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    window, box, _path = _window_on(tmp_path, "hints box")
    try:
        _dirty(window)
        _type_into("hints box", box)
        typed = box.toPlainText()
        before = _history(window)
        window._autosave_tick()

        assert len(_slots()) == 1
        assert _history(window) == before
        assert box.toPlainText() == typed
        assert QApplication.focusWidget() is box
    finally:
        conftest.close_window(window)
