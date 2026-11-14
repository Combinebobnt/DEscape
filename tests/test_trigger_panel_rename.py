"""Trigger panel: in-place rename by double click and the section dialogs."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _action,
    _answer_dialogs,
    _divided_window,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _record_count,
    _row_widget,
    _sectioned_window,
    _tagged_window,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #279: double-click a trigger to rename it in place --------------------


def _rename_window(window=None):
    """A window active enough for the inline editor to hold focus."""
    from PyQt5.QtWidgets import QApplication

    window = window if window is not None else _triggers_window()
    window.activateWindow()
    QApplication.processEvents()
    return window


def _double_click(panel, item, column: int | None = None, modifiers=None) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    column = panel._COL_NAME if column is None else column
    modifiers = Qt.NoModifier if modifiers is None else modifiers
    viewport = panel.tree.viewport()
    pos = panel.tree.visualRect(panel.tree.indexFromItem(item, column)).center()
    # The view needs the first click's press too: alone, a double-click event reads as a single press.
    QTest.mouseClick(viewport, Qt.LeftButton, modifiers, pos)
    QTest.mouseDClick(viewport, Qt.LeftButton, modifiers, pos)
    # A real double click ends with a release; QTest's leaves the button held process-wide.
    QTest.mouseRelease(viewport, Qt.LeftButton, modifiers, pos)
    QApplication.processEvents()


def _name_editor(panel):
    """The open inline editor, or None."""
    from PyQt5.QtWidgets import QLineEdit

    return next((edit for edit in panel.tree.viewport().findChildren(QLineEdit) if edit.isVisible()), None)


def _type_into(editor, text: str) -> None:
    from PyQt5.QtTest import QTest

    editor.selectAll()
    QTest.keyClicks(editor, text)


def _editor_key(editor, key) -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QTest.keyClick(editor, key)
    QApplication.processEvents()


def test_double_click_renames_a_trigger_in_place_in_one_undo_step() -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        before_name = _names(window)[0]
        item = panel._item_for_index[0]
        before = _record_count(window)

        _double_click(panel, item, panel._COL_ID)  # any column of the row
        editor = _name_editor(panel)
        assert editor is not None and editor.text() == before_name
        assert editor.selectedText() == before_name and editor.hasFocus()

        _type_into(editor, "Renamed in place")
        _editor_key(editor, Qt.Key_Return)
        assert _name_editor(panel) is None
        assert _names(window)[0] == "Renamed in place"
        assert panel._item_for_index[0].text(panel._COL_NAME).startswith("Renamed in place")
        assert _row_widget(panel, "name").text() == "Renamed in place"
        assert _record_count(window) == before + 1

        window.undo()
        assert _names(window)[0] == before_name
        assert panel._item_for_index[0].text(panel._COL_NAME).startswith(before_name)
        assert _row_widget(panel, "name").text() == before_name
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_disabled_row_seeds_its_raw_name_and_an_unchanged_enter_records_nothing() -> None:
    from PyQt5.QtCore import Qt

    from descape import trigger_fields

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.set_trigger_field(0, next(s for s in trigger_fields.TRIGGER_FIELDS if s.name == "enabled"), 0)
        panel._refresh_labels(0)
        name = _names(window)[0]
        item = panel._item_for_index[0]
        assert item.text(panel._COL_NAME) != name, "fixture: the row shows a decorated label"
        before = _record_count(window)

        _double_click(panel, item)
        editor = _name_editor(panel)
        assert editor.text() == name
        _editor_key(editor, Qt.Key_Return)
        assert _record_count(window) == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_away_keeps_the_typed_name_and_a_later_esc_renames_nothing() -> None:
    """The view commits and closes its editor itself on a current-row change,
    past the delegate's closeEditor signal: a Sections jump does that with the
    editor still focused."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        other_name = _names(window)[1]
        before = _record_count(window)

        _double_click(panel, panel._item_for_index[0])
        _type_into(_name_editor(panel), "Jumped away")
        panel.reveal_trigger(1)
        QApplication.processEvents()
        assert _name_editor(panel) is None
        assert _names(window)[0] == "Jumped away"
        assert panel.current_trigger_index() == 1
        assert _record_count(window) == before + 1

        _double_click(panel, panel._item_for_index[1])
        _editor_key(_name_editor(panel), Qt.Key_Escape)
        assert _names(window)[:2] == ["Jumped away", other_name]
        assert _record_count(window) == before + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("key_name", ["Key_Up", "Key_Down"])
def test_an_arrow_key_commits_and_stays_on_the_row_across_a_repartition(key_name: str) -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        before = _record_count(window)

        _double_click(panel, panel._item_for_index[2])
        _type_into(_name_editor(panel), "--- Arrow ---")
        _editor_key(_name_editor(panel), getattr(Qt, key_name))
        assert _name_editor(panel) is None
        assert _names(window)[2] == "--- Arrow ---"
        assert panel.selected_trigger_indices() == [2]
        assert _record_count(window) == before + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_esc_cancels_and_tab_commits_and_stays_on_the_row() -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        name = _names(window)[0]
        item = panel._item_for_index[0]
        before = _record_count(window)

        _double_click(panel, item)
        editor = _name_editor(panel)
        _type_into(editor, "Never kept")
        _editor_key(editor, Qt.Key_Escape)
        assert _name_editor(panel) is None
        assert _names(window)[0] == name and _record_count(window) == before

        for key, text in ((Qt.Key_Tab, "By Tab"), (Qt.Key_Backtab, "By Shift Tab")):
            _double_click(panel, panel._item_for_index[0])
            editor = _name_editor(panel)
            _type_into(editor, text)
            _editor_key(editor, key)
            assert _names(window)[0] == text
            assert panel.current_trigger_index() == 0, "the current row stays"
            assert _name_editor(panel) is None, "no second editor on the next row"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_double_click_opens_no_editor_where_it_is_not_a_rename() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _rename_window(_sectioned_window())
    try:
        panel = window.trigger_panel
        panel.select_triggers([0, 3])
        _double_click(panel, panel._item_for_index[5], modifiers=Qt.ControlModifier)
        assert len(panel.selected_trigger_indices()) > 1, "fixture: Ctrl adds to the selection"
        assert _name_editor(panel) is None, "a row in a multi-selection"

        panel.select_trigger(0)
        panel._editable = False
        _double_click(panel, panel._item_for_index[0])
        assert _name_editor(panel) is None, "a read-only panel"
        panel._editable = True

        panel.tree.setFocus()
        QTest.keyClick(panel.tree, Qt.Key_A)
        QApplication.processEvents()
        assert _name_editor(panel) is None, "a typed key keeps the tree's keyboard search"

        header = panel._item_for_index[2]
        assert header.childCount() and header.isExpanded()
        _double_click(panel, header)
        assert _name_editor(panel) is None
        assert not header.isExpanded(), "a header with members keeps its collapse"

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(panel._ROLE_COL, Qt.UserRole) is None
        _double_click(panel, synthetic)
        assert _name_editor(panel) is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_rename_across_the_divider_predicate_repartitions_by_enter_and_by_a_click_away() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QAbstractItemView, QApplication

    window = _rename_window(_sectioned_window())
    try:
        panel = window.trigger_panel
        sections = len(panel._sections)

        panel.select_trigger(5)
        _double_click(panel, panel._item_for_index[5])
        editor = _name_editor(panel)
        _type_into(editor, "--By Enter--")
        _editor_key(editor, Qt.Key_Return)
        assert len(panel._sections) == sections + 1
        assert panel.current_trigger_index() == 5
        assert panel.tree.state() == QAbstractItemView.NoState

        panel.select_trigger(3)
        _double_click(panel, panel._item_for_index[3])
        editor = _name_editor(panel)
        _type_into(editor, "--By Click--")
        # The row above: one below the renamed row can shift when the tree rebuilds.
        above = panel.tree.visualItemRect(panel._item_for_index[0]).center()
        QTest.mouseClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, above)
        QApplication.processEvents()
        assert _name_editor(panel) is None
        assert _names(window)[3] == "--By Click--"
        assert len(panel._sections) == sections + 2
        assert panel.current_trigger_index() == 0
        assert panel.selected_trigger_indices() == [0]
        assert panel.tree.state() == QAbstractItemView.NoState
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("opened_by", ["double_click", "menu"])
def test_a_locked_document_refuses_the_rename_at_commit(monkeypatch, opened_by: str) -> None:
    """Either way in (GH #370's menu included), the commit is what a locked document refuses."""
    from PyQt5.QtCore import Qt

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        name = _names(window)[0]
        before = _record_count(window)
        label = panel._item_for_index[0].text(panel._COL_NAME)

        if opened_by == "menu":
            _menu_rename(panel, 0)
        else:
            _double_click(panel, panel._item_for_index[0])
        editor = _name_editor(panel)
        assert editor is not None
        _type_into(editor, "Refused")
        monkeypatch.setattr(window, "_refuse_locked_edit", lambda: True)
        _editor_key(editor, Qt.Key_Return)
        assert _record_count(window) == before
        assert _names(window)[0] == name
        assert panel._item_for_index[0].text(panel._COL_NAME) == label
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_section_dialogs_report_through_the_funnels(monkeypatch) -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        asked = _answer_dialogs(monkeypatch, text="  Via dialog ")
        panel.request_new_section()
        assert _names(window)[4] == "--Via dialog--"
        panel.select_trigger(2)
        panel.request_section_rename()
        assert asked["getText"][-1][2] == "A", "prefilled with the current title"
        assert _names(window)[2] == "--Via dialog--"
        _answer_dialogs(monkeypatch, text=None)
        count = _record_count(window)
        panel.request_new_section()
        panel.request_section_rename()
        assert _record_count(window) == count, "a cancelled dialog records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_section_verbs_are_gated_like_move_up_and_down() -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        assert panel.sections_menu.toolTipsVisible(), "the disabled reasons live in the action tooltips"
        panel.select_trigger(0)
        assert panel.section_new_action.isEnabled()
        assert panel.section_move_action.isEnabled()
        assert not panel.section_rename_action.isEnabled(), "not a divider"
        panel.select_trigger(2)
        assert panel.section_rename_action.isEnabled()
        assert not panel.section_move_action.isEnabled(), "a header does not move into a section"
        assert "section header" in panel.section_move_action.toolTip()

        panel.select_trigger(0)
        panel.filter_edit.setText("a")
        for enabled in (panel.section_new_action.isEnabled(), panel.section_move_action.isEnabled()):
            assert not enabled
        assert "clear the filter" in panel.section_new_action.toolTip()
        panel.filter_edit.clear()

        panel.sort_combo.setCurrentIndex(1)
        panel.select_trigger(2)
        assert not panel.section_new_action.isEnabled()
        assert not panel.section_rename_action.isEnabled()
        assert "Display order" in panel.section_move_action.toolTip()
        panel.sort_combo.setCurrentIndex(0)
        assert panel.section_new_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_section_verbs_follow_the_tag_facet_and_a_single_section_file(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert panel.section_new_action.isEnabled()
        assert not panel.section_move_action.isEnabled(), "one section: nowhere else to go"
        assert "no other section" in panel.section_move_action.toolTip()
        panel.set_tag_filter("D1")
        assert not panel.section_new_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_is_off_on_a_bare_run_divider() -> None:
    window = _divided_window("------")
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        assert not panel.section_rename_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #370: Rename in the trigger list's right-click menu --------------------


def _menu_rename(panel, index: int) -> None:
    """Trigger the menu's Rename on trigger `index`'s row and let its queued open run."""
    from PyQt5.QtWidgets import QApplication

    _action(panel.trigger_context_menu(panel._item_for_index[index]), "Rename").trigger()
    QApplication.processEvents()


def test_gh370_menu_rename_opens_the_inline_editor_in_one_undo_step() -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        before_name = _names(window)[0]
        before = _record_count(window)

        _menu_rename(panel, 0)
        editor = _name_editor(panel)
        assert editor is not None and editor.text() == before_name and editor.hasFocus()
        _type_into(editor, "Renamed from the menu")
        _editor_key(editor, Qt.Key_Return)
        assert _name_editor(panel) is None
        assert _names(window)[0] == "Renamed from the menu"
        assert _record_count(window) == before + 1

        window.undo()
        assert _names(window)[0] == before_name
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_menu_rename_reaches_a_section_header_with_members() -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window(_sectioned_window())
    try:
        panel = window.trigger_panel
        sections = len(panel._sections)
        header = panel._item_for_index[2]
        assert header.childCount() and header.isExpanded(), "fixture: an expanded header with members"
        panel.select_trigger(2)

        before = _record_count(window)
        _menu_rename(panel, 2)
        editor = _name_editor(panel)
        assert editor is not None and editor.text() == "RISK!!!", "the title alone"
        _type_into(editor, "Renamed")
        _editor_key(editor, Qt.Key_Return)
        assert _names(window)[2] == "--Renamed--", "its own dashes kept"
        assert _record_count(window) == before + 1
        assert len(panel._sections) == sections
        assert panel._item_for_index[2].isExpanded()

        panel._item_for_index[2].setExpanded(False)
        panel.select_trigger(2)
        _menu_rename(panel, 2)
        editor = _name_editor(panel)
        assert editor is not None and editor.text() == "Renamed"
        _type_into(editor, "Collapsed")
        _editor_key(editor, Qt.Key_Return)
        assert _names(window)[2] == "--Collapsed--"
        assert not panel._item_for_index[2].isExpanded(), "a collapsed header stays collapsed"

        window.undo()
        assert _names(window)[2] == "--Renamed--"

        header = panel._item_for_index[6]
        assert _names(window)[6] == "------" and header.childCount(), "fixture: a bare run with a member"
        panel.select_trigger(6)
        _menu_rename(panel, 6)
        editor = _name_editor(panel)
        assert editor.text() == "------", "no title slot: the whole name"
        _type_into(editor, "plain six")
        _editor_key(editor, Qt.Key_Return)
        assert _names(window)[6] == "plain six"
        assert len(panel._sections) == sections - 1, "no longer a divider: its section merges up"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_a_header_title_reads_like_the_rename_section_dialog(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warned.append(args)))
    window = _rename_window(_divided_window())
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        before = _record_count(window)
        for text in ("x-", "", "   "):
            _menu_rename(panel, 2)
            _type_into(_name_editor(panel), text)
            _editor_key(_name_editor(panel), Qt.Key_Return)
            assert _names(window)[2] == "--A--"
        assert len(warned) == 1 and "would not read back" in warned[0][2], "a blank title changes nothing, silently"
        assert _record_count(window) == before

        _menu_rename(panel, 2)
        _type_into(_name_editor(panel), "  Spaced  ")
        _editor_key(_name_editor(panel), Qt.Key_Return)
        assert _names(window)[2] == "--Spaced--"
        assert _record_count(window) == before + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_double_click_on_a_header_without_members_edits_its_title() -> None:
    from PyQt5.QtCore import Qt

    window = _rename_window(_divided_window(at=3))
    try:
        panel = window.trigger_panel
        header = panel._item_for_index[3]
        assert header.parent() is None and not header.childCount(), "fixture: the last trigger, a header alone"
        panel.select_trigger(3)
        _double_click(panel, header)
        editor = _name_editor(panel)
        assert editor is not None and editor.text() == "A"
        _type_into(editor, "B")
        _editor_key(editor, Qt.Key_Return)
        assert _names(window)[3] == "--B--"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_menu_rename_is_off_where_it_is_not_a_rename() -> None:
    from PyQt5.QtCore import Qt

    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(panel._ROLE_COL, Qt.UserRole) is None
        rename = _action(panel.trigger_context_menu(synthetic), "Rename")
        assert not rename.isEnabled()
        assert "select only" not in rename.toolTip(), "no trigger there to select"

        panel.select_triggers([0, 3])
        rename = _action(panel.trigger_context_menu(panel._item_for_index[3]), "Rename")
        assert not rename.isEnabled()
        assert rename.toolTip().endswith("; select only this trigger)")
        assert rename.toolTip().count("(") == rename.toolTip().count(")") == 1

        panel.select_trigger(0)
        assert _action(panel.trigger_context_menu(panel._item_for_index[0]), "Rename").isEnabled()
        panel._editable = False
        assert not _action(panel.trigger_context_menu(panel._item_for_index[0]), "Rename").isEnabled()
        panel._editable = True
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_a_right_click_selects_the_row_so_rename_is_on() -> None:
    """The gate relies on Qt's right press selecting an unselected row alone."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        item = panel._item_for_index[1]
        pos = panel.tree.visualItemRect(item).center()
        QTest.mouseClick(panel.tree.viewport(), Qt.RightButton, Qt.NoModifier, pos)
        QApplication.processEvents()
        assert panel.selected_trigger_indices() == [1]
        assert _action(panel.trigger_context_menu(item), "Rename").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh370_the_editor_survives_the_menu_closing() -> None:
    """Clicked in the running menu, as a user does: the menu closes after the click."""
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _rename_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        item = panel._item_for_index[0]
        menu = panel.trigger_context_menu(item)
        rename = _action(menu, "Rename")
        QTimer.singleShot(0, lambda: QTest.mouseClick(menu, Qt.LeftButton, Qt.NoModifier, menu.actionGeometry(rename).center()))
        QTimer.singleShot(2000, menu.close)  # a missed click must not hang the run
        menu.exec_(panel.tree.viewport().mapToGlobal(panel.tree.visualItemRect(item).center()))
        QApplication.processEvents()
        editor = _name_editor(panel)
        assert editor is not None, "opened inside the menu, the editor closes with it"
        # Offscreen leaves the closed menu active; a desktop hands activation back itself.
        window.activateWindow()
        QApplication.processEvents()
        assert editor.hasFocus()
        _type_into(editor, "After the menu")
        _editor_key(editor, Qt.Key_Return)
        assert _names(window)[0] == "After the menu"
    finally:
        window.edit_history.mark_saved()
        window.close()
