"""GH #30's History window: the dialog itself, and the ViewerWindow wiring
that feeds it. EditHistory's own jump_to/span_kinds/on_change rules are
covered without Qt in tests/test_edit_history.py.

Same offscreen technique tests/test_clipboard_dialog.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest

from descape.history_dialog import OPENED_FILE_ID, EditHistoryDialog

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _rows(count: int):
    """Row ids deliberately unequal to their own row numbers, so an
    index-for-id confusion cannot pass."""
    return [(1000 + i, f"edit {i}", "tile") for i in range(count)]


def _dialog(**callbacks):
    conftest.ensure_qapp()
    return EditHistoryDialog(None, **callbacks)


def _texts(dialog, column: int = 1):
    return [
        dialog.tree.topLevelItem(row).text(column) for row in range(dialog.tree.topLevelItemCount())
    ]


# -- the dialog on its own ---------------------------------------------------


def test_set_rows_prepends_the_opened_file_row() -> None:
    from PyQt5.QtCore import Qt

    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 0)
        assert dialog.tree.topLevelItemCount() == 4, "three records plus 'Opened file'"
        assert _texts(dialog) == ["Opened file", "edit 0", "edit 1", "edit 2"]
        ids = [dialog.tree.topLevelItem(r).data(0, Qt.UserRole) for r in range(4)]
        assert ids == [OPENED_FILE_ID, 1000, 1001, 1002]
        targets = [dialog.tree.topLevelItem(r).data(1, Qt.UserRole) for r in range(4)]
        assert targets == [0, 1, 2, 3], "row N stands for cursor N"
    finally:
        dialog.close()


def test_empty_state_label_shows_with_no_records() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows([], 0, 0)
        assert dialog.tree.topLevelItemCount() == 1, "'Opened file' alone"
        assert dialog.status.text() == dialog._EMPTY
    finally:
        dialog.close()


def test_the_current_row_is_bold_and_undone_rows_are_greyed() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 1, 0)
        bold = [dialog.tree.topLevelItem(r).font(1).bold() for r in range(4)]
        assert bold == [False, True, False, False]
        normal = dialog.tree.topLevelItem(0).foreground(1).color()
        greyed = [dialog.tree.topLevelItem(r).foreground(1).color() for r in (2, 3)]
        assert all(c != normal for c in greyed), "rows above the cursor read as undone"
        assert dialog.tree.topLevelItem(1).foreground(1).color() == normal
    finally:
        dialog.close()


def test_the_opened_file_row_is_bold_at_cursor_zero() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(2), 0, 0)
        assert dialog.tree.topLevelItem(0).font(1).bold()
        assert not dialog.tree.topLevelItem(1).font(1).bold()
    finally:
        dialog.close()


def test_the_saved_marker_sits_on_the_saved_row_only() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 2)
        markers = [bool(t) for t in _texts(dialog, 0)]
        assert markers == [False, False, True, False]
    finally:
        dialog.close()


def test_no_saved_marker_when_the_saved_state_fell_off_the_cap() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, None)
        assert not any(_texts(dialog, 0))
    finally:
        dialog.close()


def test_selection_survives_a_repopulate_by_id_not_by_row() -> None:
    """The discriminating case: the trim drops the oldest record, so every
    surviving row shifts down one while its id stays put."""
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1002)
        assert dialog.selected_row_id() == 1002
        dialog.set_rows(_rows(3)[1:], 2, None)
        assert dialog.selected_row_id() == 1002
        assert dialog.selected_target() == 2, "one row earlier after the trim"
    finally:
        dialog.close()


def test_double_click_and_the_button_both_report_the_selected_target() -> None:
    seen = []
    dialog = _dialog(on_jump=seen.append)
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1000)
        dialog.jump_button.click()
        assert seen == [1], "row 'edit 0' is cursor 1"
        dialog.select_row(OPENED_FILE_ID)
        dialog.tree.itemDoubleClicked.emit(dialog.tree.topLevelItem(0), 0)
        assert seen == [1, 0]
    finally:
        dialog.close()


def test_selecting_a_row_does_not_jump_on_its_own() -> None:
    """Only a double-click or the button may move the history: a plain
    keyboard walk down the list would otherwise undo the document."""
    seen = []
    dialog = _dialog(on_jump=seen.append)
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1000)
        dialog.set_rows(_rows(3), 1, 0)
        assert seen == []
    finally:
        dialog.close()


def test_the_jump_button_gates_on_a_selection() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(2), 2, 0)
        dialog.tree.setCurrentItem(None)
        assert not dialog.jump_button.isEnabled()
        dialog.select_row(1001)
        assert dialog.jump_button.isEnabled()
    finally:
        dialog.close()


# -- the window wiring -------------------------------------------------------


def _window():
    window = conftest.terrain_edit_window()
    window._on_tool_selected("draw")
    return window


def _paint(window, x: int, y: int, terrain_id: int) -> None:
    window.terrain_panel.set_terrain(terrain_id)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def test_the_dialog_is_non_modal_and_lists_every_record() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._show_history_dialog()
        dialog = window._history_dialog
        assert dialog is not None
        assert not dialog.isModal()
        assert dialog.tree.topLevelItemCount() == 3, "two records plus 'Opened file'"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_new_edit_with_the_dialog_open_refreshes_it_through_on_change() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        assert dialog.tree.topLevelItemCount() == 2
        _paint(window, 4, 4, _TERRAIN_B)
        assert dialog.tree.topLevelItemCount() == 3, "no push site had to call the refresh"
        assert dialog.tree.topLevelItem(2).font(1).bold(), "the newest row is current"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_jumping_to_zero_restores_the_tiles_and_the_clean_title() -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        before = mm.get_tile(3, 3).terrain_id
        window.edit_history.mark_saved()
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        assert window.windowTitle().startswith("*")

        window._show_history_dialog()
        window._on_history_jump(0)
        assert window.edit_history.cursor == 0
        assert mm.get_tile(3, 3).terrain_id == before
        assert not window.windowTitle().startswith("*"), "back onto the saved cursor"
        assert not window.undo_action.isEnabled()
        assert window.redo_action.isEnabled()

        window._on_history_jump(2)
        assert mm.get_tile(3, 3).terrain_id == _TERRAIN_A
        assert mm.get_tile(4, 4).terrain_id == _TERRAIN_B
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_reports_the_distance_and_the_entry_it_landed_on() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._on_history_jump(0)
        assert "Jumped back 2 steps to: Opened file" in window.status_log.toPlainText()
        window._on_history_jump(1)
        assert "Jumped forward 1 step to:" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_to_the_current_cursor_reports_rather_than_moving() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._on_history_jump(1)
        assert window.edit_history.cursor == 1
        assert "Already at that history entry" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stale_out_of_range_row_is_refused_not_raised() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._on_history_jump(9)
        assert window.edit_history.cursor == 1
        assert "Jump refused" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_dialog_tracks_a_jump_made_from_the_menu_undo() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._show_history_dialog()
        dialog = window._history_dialog
        window.undo()
        bold = [dialog.tree.topLevelItem(r).font(1).bold() for r in range(3)]
        assert bold == [False, True, False], "undo moved the bold row without a manual refresh"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_the_file_empties_the_dialog() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        window.edit_history.mark_saved()
        window.close_scenario()
        assert dialog.tree.topLevelItemCount() == 1, "reset() reached the dialog"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_history_action_is_registered_and_unbound_by_default() -> None:
    from descape import settings

    ids = [row[0] for row in settings.REBINDABLE_ACTIONS]
    assert "edit_history" in ids
    assert settings.get_keybind("edit_history") == ""


def test_a_hidden_dialog_is_not_rebuilt_but_reopens_current() -> None:
    """Close only hides a QDialog, so the on_change hook would otherwise keep
    rebuilding a few hundred invisible rows on every stroke."""
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        dialog.close()
        _paint(window, 4, 4, _TERRAIN_B)
        assert dialog.tree.topLevelItemCount() == 2, "no rebuild while hidden"
        window._show_history_dialog()
        assert dialog.tree.topLevelItemCount() == 3, "reopening catches it up"
        assert dialog.tree.topLevelItem(2).font(1).bold()
    finally:
        window.edit_history.mark_saved()
        window.close()
