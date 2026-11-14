"""Trigger panel: copy, cut and paste of triggers and entries, the context
menus (GH #137) and cross-document pastes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    V158_DONOR,
    _action,
    _active_triggers_window,
    _create_objects_window,
    _ctrl_select,
    _divided_window,
    _focus_on,
    _gh2_window,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _order,
    _record_count,
    _rename,
    _roomy_tree,
    _row_widget,
    _select_entries,
    _status,
    _triggers_window,
    _types_of,
    trigger_clipboard_refs,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def new_maps_are_1_58(monkeypatch):
    """File > New Map from the 1.58 donor, TRIGGER_FIXTURE's own version: the
    shipped donor is 1.59, and a paste across versions is refused (TASK-345)."""
    from descape import games, scenario_new

    real = scenario_new.blank_template_path_for
    default = real(games.DEFAULT_ID)
    monkeypatch.setattr(scenario_new, "blank_template_path_for", lambda game: V158_DONOR if real(game) == default else real(game))


def _row_buttons(layout) -> list:
    return [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]


def test_the_copy_button_fills_the_clipboard_without_adding_triggers() -> None:
    """GH #27/#28: with Paste beside it, "Copy" copies; Paste then enables."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        copy = next(b for b in _row_buttons(panel.trigger_buttons) if b.text() == "Copy")
        _ctrl_select(panel, 0, 2)
        assert copy.isEnabled()
        assert not panel.trigger_paste_button.isEnabled(), "nothing to paste yet"
        copy.click()
        assert len(window.trigger_panel._manager().triggers) == 4, "copying adds nothing"
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert [t.name for t in window._trigger_clipboard.triggers] == ["Fixture: setup", "Fixture: references"]
        assert panel.trigger_paste_button.isEnabled()
        panel.tree.clearSelection()
        assert not copy.isEnabled(), "nothing selected to copy"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_duplicate_buttons_keep_their_names_and_still_duplicate() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert panel.trigger_copy_button.text() == "Duplicate"
        assert panel.entry_copy_button.text() == "Duplicate"
        assert [b.text() for b in _row_buttons(panel.trigger_buttons)] == ["New", "Copy", "Paste", "Delete"]
        assert panel.trigger_copy_button in _row_buttons(panel.reorder_buttons)
        _ctrl_select(panel, 0)
        panel.trigger_copy_button.click()
        assert len(window.trigger_edits.manager().triggers) == 5
        assert window._trigger_clipboard is None, "Duplicate leaves the clipboard alone"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paste_with_no_current_trigger_appends(tmp_path: Path) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 3)
        window.copy_triggers()
        window.trigger_structural_edit("paste", [])
        assert list(window.trigger_edits.manager().trigger_display_order) == [0, 1, 2, 3, 4]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_clipboard_from_another_scenario_version_is_refused_out_loud(monkeypatch) -> None:
    """GH #3: Paste stays enabled, and Ctrl+V or the button logs why it
    refused, changing nothing. QMessageBox.warning is patched: unpatched it
    blocks forever offscreen."""
    import dataclasses

    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _triggers_window()
    try:
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel = window.trigger_panel
        _ctrl_select(panel, 0)
        window.copy_triggers()
        window._trigger_clipboard = dataclasses.replace(window._trigger_clipboard, scenario_version="1.56")
        window._sync_trigger_clipboard_state()
        expected = "Copied from a 1.56 scenario; this one is 1.58."
        assert panel.trigger_paste_button.isEnabled()
        assert window.paste_action.isEnabled()
        assert panel.trigger_paste_button.toolTip() == expected
        assert window.paste_action.toolTip() == expected
        window.status_log.clear()

        QTest.keyClick(window, Qt.Key_V, Qt.ControlModifier)
        QApplication.processEvents()
        assert f"Cannot paste these triggers: {expected}" in window.status_log.toPlainText()
        assert len(panel._manager().triggers) == 4

        window.status_log.clear()
        panel.trigger_paste_button.click()
        assert f"Cannot paste these triggers: {expected}" in window.status_log.toPlainText()
        assert len(panel._manager().triggers) == 4
        assert warnings == []
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert not window.edit_history.is_dirty, "no undo record was pushed"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_into_another_file_reports_and_undoes_in_one_step(new_maps_are_1_58) -> None:
    """GH #3 through the window: copy in the fixture, File > New Map, paste.
    Both outside links are cleared, variable 0 takes the source's name, and one
    undo removes the triggers and the name together."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.new_map()
        assert window.scenario.scenario_version == "1.58"
        window.mode_combo.setCurrentText("Triggers")
        assert panel.trigger_paste_button.isEnabled(), "the clipboard outlives the document"

        window.trigger_structural_edit("paste", [])
        manager = window.trigger_edits.manager()
        assert [t.name for t in manager.triggers] == ["Fixture: references (copy)", "Fixture: variable (copy)"]
        assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var")]
        # The 5 unit references are the fixture's own, on trigger 3.
        assert "Pasted 2 triggers from another scenario: cleared 2 trigger links and 5 unit references, named 1 variable." in (
            window.status_log.toPlainText()
        )

        window.undo()
        manager = window.trigger_edits.manager()
        assert (len(manager.triggers), len(manager.variables)) == (0, 0)
        assert not window.edit_history.is_dirty, "the paste was one undo step"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _variable_dialog_rows(panel) -> list[tuple[int, str]]:
    """Open Variables… through its button, as a user does, and read its list."""
    from PyQt5.QtWidgets import QApplication

    panel.variables_button.click()
    QApplication.processEvents()
    tree = panel._variables_dialog.tree
    return [(int(tree.topLevelItem(i).text(0)), tree.topLevelItem(i).text(1)) for i in range(tree.topLevelItemCount())]


def test_a_cross_document_pastes_variable_name_shows_in_the_variables_dialog_until_undone(new_maps_are_1_58) -> None:
    """GH #3's manual step, in the dialog. Passes when: the new map's Variables…
    list is empty before the paste; after pasting the fixture's triggers 2 and 3
    it lists exactly (0, "fixture_var"); one undo empties the open dialog; and
    reopening it through the button still lists nothing."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.new_map()
        window.mode_combo.setCurrentText("Triggers")
        assert _variable_dialog_rows(panel) == [], "fixture assumption: the new map names no variable"

        window.trigger_structural_edit("paste", [])
        assert _variable_dialog_rows(panel) == [(0, "fixture_var")]

        window.undo()
        tree = panel._variables_dialog.tree
        assert tree.topLevelItemCount() == 0, "the open dialog drops the name with the undo"
        assert _variable_dialog_rows(panel) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_from_a_1_58_file_into_a_new_map_is_refused() -> None:
    """TASK-345: File > New Map is a 1.59 document, so triggers copied from the
    1.58 fixture are refused there. Paste stays enabled, says why, and
    changes nothing (GH #3's refusal)."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.new_map()
        window.mode_combo.setCurrentText("Triggers")
        expected = "Copied from a 1.58 scenario; this one is 1.59."
        assert panel.trigger_paste_button.isEnabled()
        assert panel.trigger_paste_button.toolTip() == expected
        assert window.paste_action.toolTip() == expected
        window.status_log.clear()

        panel.trigger_paste_button.click()
        assert f"Cannot paste these triggers: {expected}" in window.status_log.toPlainText()
        assert len(panel._manager().triggers) == 0
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert not window.edit_history.is_dirty, "no undo record was pushed"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_after_reopening_the_same_file_counts_as_another_scenario() -> None:
    """GH #3's reopen check: a reload is a new document, so the links a copy
    carried are cleared even though the path is the same."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        window.status_log.clear()

        window.trigger_structural_edit("paste", [])
        assert [t.name for t in window.trigger_edits.manager().triggers][-2:] == [
            "Fixture: references (copy)",
            "Fixture: variable (copy)",
        ]
        # Variable 0 already carries the same name here, so none is renamed.
        assert "Pasted 2 triggers from another scenario: cleared 2 trigger links and 5 unit references." in (
            window.status_log.toPlainText()
        ), window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #137: right-click Cut/Copy/Paste on triggers, conditions and effects ----


@pytest.fixture
def _no_menu_exec(monkeypatch):
    """A stray context-menu event would exec a modal QMenu and hang the run."""
    from PyQt5.QtWidgets import QMenu

    shown: list = []
    monkeypatch.setattr(QMenu, "exec_", lambda self, *a, **k: shown.append(self))
    return shown


def _right_click(tree, item) -> None:
    """A real right-button press and release on `item`'s row, near its left
    edge: an entry row's rect runs far past the 340 px viewport."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    tree.scrollToItem(item)
    QApplication.processEvents()
    rect = tree.visualItemRect(item)
    point = QPoint(rect.left() + 8, rect.center().y())
    assert tree.viewport().rect().contains(point), (rect, tree.viewport().rect())
    QTest.mousePress(tree.viewport(), Qt.RightButton, Qt.NoModifier, point)
    QTest.mouseRelease(tree.viewport(), Qt.RightButton, Qt.NoModifier, point)
    QApplication.processEvents()


@pytest.mark.parametrize("clicked", [3, 1], ids=["unselected row", "member of the selection"])
def test_gh137_slice0_ab_a_right_click_on_the_trigger_tree(_no_menu_exec, clicked: int) -> None:
    """Slice 0 a/b: an unselected row becomes the whole selection; a member
    keeps the selection and only becomes current."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _roomy_tree(panel)
        _ctrl_select(panel, 0, 1, 2)
        _right_click(panel.tree, panel._item_for_index[clicked])
        expected = [3] if clicked == 3 else [0, 1, 2]
        assert panel.selected_trigger_indices() == expected
        assert panel.current_trigger_index() == clicked
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("clicked", [3, 1], ids=["unselected row", "member of the selection"])
def test_gh137_slice0_c_a_right_click_on_the_entry_tree(_no_menu_exec, clicked: int) -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        assert indices == [1, 2, 3]
        _select_entries(panel, ("effect", 0), ("effect", 1), ("effect", 2))
        _right_click(panel.entry_tree, panel._entry_item_for("effect", clicked))
        expected = [("effect", 3)] if clicked == 3 else [("effect", 0), ("effect", 1), ("effect", 2)]
        assert panel.selected_entry_refs() == expected
        assert panel.current_entry_ref() == ("effect", clicked)
    finally:
        window.edit_history.mark_saved()
        window.close()


def _context_menu(tree, item, shown: list):
    """Deliver a mouse QContextMenuEvent to `item`'s row on the viewport, the
    event a real right-click produces, and return the one QMenu it exec'd.
    QTest's mouse presses never synthesize this event, so _right_click alone
    leaves the customContextMenuRequested wiring unexercised."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QContextMenuEvent
    from PyQt5.QtWidgets import QApplication

    tree.scrollToItem(item)
    QApplication.processEvents()
    rect = tree.visualItemRect(item)
    point = QPoint(rect.left() + 8, rect.center().y())
    assert tree.viewport().rect().contains(point), (rect, tree.viewport().rect())
    shown.clear()
    event = QContextMenuEvent(QContextMenuEvent.Mouse, point, tree.viewport().mapToGlobal(point))
    QApplication.sendEvent(tree.viewport(), event)
    QApplication.processEvents()
    assert len(shown) == 1, f"the right-click opened {len(shown)} menus"
    return shown[0]


def test_gh137_a_right_click_on_the_trigger_tree_opens_its_menu_and_pastes_below_that_row(_no_menu_exec) -> None:
    """The trigger tree's customContextMenuRequested wiring. Passes when: a
    context-menu event on row 0 exec's one menu starting Cut, Copy, Paste; its
    Copy takes trigger 0; and with trigger 3 current, Paste from the menu a
    right-click on row 1 opened lands the copy (index 4) below row 1, display
    order [0, 1, 4, 2, 3], selected."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        menu = _context_menu(panel.tree, panel._item_for_index[0], _no_menu_exec)
        assert _texts(menu)[:3] == ["Cut", "Copy", "Paste"]
        _action(menu, "Copy").trigger()
        assert [t.name for t in window._trigger_clipboard.triggers] == ["Fixture: setup"]

        panel.select_trigger(3)
        menu = _context_menu(panel.tree, panel._item_for_index[1], _no_menu_exec)
        assert _texts(menu)[:3] == ["Cut", "Copy", "Paste"]
        _action(menu, "Paste").trigger()
        assert _order(window) == [0, 1, 4, 2, 3]
        assert panel.selected_trigger_indices() == [4]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_menu_cut_and_paste_of_an_activated_trigger_relinks_and_says_so(_no_menu_exec) -> None:
    """Cut and Paste of trigger 0, which trigger 2 ("Fixture: references")
    activates, both through right-click menus rather than
    window.cut_triggers()/paste_triggers(). Passes when: the Cut reports the
    one inbound link that will reconnect and leaves that link (now on index 1)
    at -1; the Paste from row 2's menu appends the trigger as "Fixture: setup"
    (no "(copy)"), re-points the link to its new index 3, and reports
    "reconnected 1 trigger link into them", in one undo record."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert [ce.trigger_id for ce in trigger_clipboard_refs(panel._manager().triggers[2])] == [0, 1], (
            "fixture assumption: trigger 2 activates trigger 0"
        )
        panel.select_trigger(0)
        _action(_context_menu(panel.tree, panel._item_for_index[0], _no_menu_exec), "Cut").trigger()
        assert "1 trigger link into the cut triggers will reconnect on Paste." in _status(window)
        assert [ce.trigger_id for ce in trigger_clipboard_refs(panel._manager().triggers[1])] == [-1, 0]

        panel.select_trigger(2)
        records = _record_count(window)
        window.status_log.clear()
        _action(_context_menu(panel.tree, panel._item_for_index[2], _no_menu_exec), "Paste").trigger()
        manager = panel._manager()
        assert [t.name for t in manager.triggers][-1] == "Fixture: setup", "no (copy) suffix"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [3, 0]
        assert "Pasted 1 trigger: reconnected 1 trigger link into them." in _status(window)
        assert _record_count(window) == records + 1, "the relink rides in the paste's record"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_right_click_on_the_entry_tree_opens_its_menu_and_pastes_below_that_row(_no_menu_exec) -> None:
    """The entry tree's customContextMenuRequested wiring. Passes when: a
    context-menu event on trigger 3's effect 1 exec's one menu starting Cut,
    Copy, Paste; its Copy takes effects 1 and 2; and on trigger 2, Paste from
    the menu a right-click on effect 0 opened lands them below it, effect
    types [8, 19, 11, 9], as one "Paste 2 effects" record."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        menu = _context_menu(panel.entry_tree, panel._entry_item_for("effect", 1), _no_menu_exec)
        assert _texts(menu)[:3] == ["Cut", "Copy", "Paste"]
        _action(menu, "Copy").trigger()
        assert window._entry_clipboard.label == "2 effects"

        panel.select_trigger(2)
        records = _record_count(window)
        menu = _context_menu(panel.entry_tree, panel._entry_item_for("effect", 0), _no_menu_exec)
        assert _texts(menu)[:3] == ["Cut", "Copy", "Paste"]
        _action(menu, "Paste").trigger()
        assert _types_of(window, 2) == [8, 19, 11, 9]
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Paste 2 effects"
    finally:
        window.edit_history.mark_saved()
        window.close()


_EDITABLE_WIDGETS = ["line edit", "text box", "spin box", "editable combo"]


def _editable_widget(window, which: str):
    """One focused-able editor of each class Edit > Cut/Copy/Paste must defer to.
    No production combo is editable today, so that one is a stand-in."""
    from PyQt5.QtWidgets import QComboBox

    panel = window.trigger_panel
    panel.select_trigger(0)
    if which == "line edit":
        return _row_widget(panel, "name")
    if which == "text box":
        return _row_widget(panel, "description")
    if which == "spin box":
        return _row_widget(panel, "description_order")
    combo = QComboBox(panel)
    combo.setEditable(True)
    combo.addItem("alpha")
    combo.show()
    return combo


def _editor_text(widget) -> str:
    from PyQt5.QtWidgets import QComboBox, QPlainTextEdit

    if isinstance(widget, QPlainTextEdit):
        return widget.toPlainText()
    if isinstance(widget, QComboBox):
        return widget.lineEdit().text()
    return widget.text()


def _select_all_text(widget) -> None:
    from PyQt5.QtWidgets import QComboBox

    (widget.lineEdit() if isinstance(widget, QComboBox) else widget).selectAll()


@pytest.mark.parametrize("route", ["keybind", "menu"])
@pytest.mark.parametrize("which", _EDITABLE_WIDGETS)
def test_gh137_slice0_g_cut_copy_paste_defer_to_a_focused_editor(which: str, route: str) -> None:
    """Slice 0 g, a hard requirement (wave 10): with an editable widget
    focused, Ctrl+X/C/V and Edit > Cut/Copy/Paste act on its text and never
    on the triggers."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    clipboard = QApplication.clipboard()
    try:
        widget = _editable_widget(window, which)
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()
        widget.setFocus()
        QApplication.processEvents()
        focused = QApplication.focusWidget()
        assert focused is widget or widget.isAncestorOf(focused), focused

        def press(verb: str) -> None:
            if route == "keybind":
                key = {"cut": Qt.Key_X, "copy": Qt.Key_C, "paste": Qt.Key_V}[verb]
                QTest.keyClick(QApplication.focusWidget(), key, Qt.ControlModifier)
            else:
                action = getattr(window, f"{verb}_action")
                assert action.isEnabled(), verb
                action.trigger()
            QApplication.processEvents()

        baseline = _editor_text(widget)
        assert baseline, "the editor starts with some text to copy"
        clipboard.setText("")
        _select_all_text(widget)
        press("copy")
        assert clipboard.text() == baseline
        assert window._trigger_clipboard is None, "Copy took the triggers"

        window.copy_triggers()  # a held block, so a stray trigger Paste would show
        count = len(window.trigger_panel._manager().triggers)
        clipboard.setText("42")
        _select_all_text(widget)
        press("paste")
        assert _editor_text(widget) == "42"
        assert len(window.trigger_panel._manager().triggers) == count, "Paste pasted triggers"

        _select_all_text(widget)
        press("cut")
        assert _editor_text(widget) == ""
        assert clipboard.text() == "42"
        assert len(window.trigger_panel._manager().triggers) == count, "Cut cut triggers"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        conftest.close_window(window)


def _texts(menu) -> list[str]:
    return [action.text() for action in menu.actions()]


def test_gh137_the_trigger_menu_offers_cut_copy_paste_first() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert _texts(menu) == [
            "Cut", "Copy", "Paste", "Rename", "Move Up", "Move Down", "Move to Top", "Move to Bottom",
            "Renumber IDs to Display Order", "", "Select Section", "Select Tag",
        ]
        assert menu.toolTipsVisible()
        assert _action(menu, "Cut").isEnabled() and _action(menu, "Copy").isEnabled()
        assert not _action(menu, "Paste").isEnabled(), "nothing to paste yet"

        panel.tree.clearSelection()
        menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert not _action(menu, "Cut").isEnabled() and not _action(menu, "Copy").isEnabled()

        panel.select_trigger(0)
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Copy").trigger()
        assert [t.name for t in window._trigger_clipboard.triggers] == ["Fixture: setup"]
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "Copy records nothing"
        paste = _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste")
        assert paste.isEnabled() and paste.toolTip() == (
            "Paste the copied triggers below this one, or below a collapsed section's end"
        )

        window._trigger_clipboard = dataclasses.replace(window._trigger_clipboard, scenario_version="1.56")
        window._sync_trigger_clipboard_state()
        paste = _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste")
        assert paste.toolTip() == "Copied from a 1.56 scenario; this one is 1.58."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task214_edit_menu_trigger_clipboard_tips_say_a_collapsed_section_counts_whole() -> None:
    """Edit > Cut/Copy/Paste Triggers explain GH #134's collapsed-section rule like the panel's own controls."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.copy_triggers()
        window._update_clipboard_actions()
        assert "(a collapsed section cuts whole)" in window.cut_action.toolTip()
        assert "(a collapsed section copies whole)" in window.copy_action.toolTip()
        assert window.paste_action.toolTip().endswith(", or below a collapsed section's end")
        assert panel.trigger_paste_button.toolTip() == panel._PASTE_TOOLTIP
        assert "below a collapsed section's end" in panel._PASTE_TOOLTIP
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_menu_paste_lands_below_the_clicked_row_not_the_current_one() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.copy_triggers()
        panel.select_trigger(3)
        _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste").trigger()
        assert _order(window) == [0, 1, 4, 2, 3]
        assert panel.selected_trigger_indices() == [4]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_paste_is_disabled_on_the_synthetic_leading_header() -> None:
    from PyQt5.QtCore import Qt

    window = _divided_window()
    try:
        panel = window.trigger_panel
        window.copy_triggers()
        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None
        assert not _action(panel.trigger_context_menu(synthetic), "Paste").isEnabled()
        assert _action(panel.trigger_context_menu(panel._item_for_index[2]), "Paste").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_is_one_record_and_its_paste_keeps_the_name_of_a_divider() -> None:
    """The divider case of decision 2: a cut header still heads a section once
    pasted. An undo of the cut leaves the inbound link on the original, so
    the paste after it leaves that link alone and says so."""
    from descape.trigger_organize import is_divider

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "--- Setup ---")
        panel.select_trigger(0)
        records = _record_count(window)
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Cut").trigger()
        manager = panel._manager()
        assert [t.name for t in manager.triggers] == ["Fixture: armour split", "Fixture: references", "Fixture: variable"]
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Cut trigger 0"
        assert window._trigger_clipboard.from_cut
        assert "1 trigger link into the cut triggers will reconnect on Paste." in _status(window)
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [-1, 0]
        assert panel.selected_trigger_indices() == [0], "lands where the cut row was"

        window.undo()
        manager = panel._manager()
        assert len(manager.triggers) == 4
        assert window._trigger_clipboard is not None, "the clipboard keeps the block"
        window.status_log.clear()
        panel.select_trigger(3)
        window.paste_triggers()
        manager = panel._manager()
        assert manager.triggers[4].name == "--- Setup ---" and is_divider(manager.triggers[4].name)
        assert panel._grouped and panel._item_for_index[4].parent() is None, "it heads its own section"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[2])] == [0, 1]
        assert "left 1 trigger link alone" in _status(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_then_paste_relinks_and_one_undo_takes_both_back() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        panel.select_trigger(2)
        records = _record_count(window)
        window.paste_triggers()
        manager = panel._manager()
        assert [t.name for t in manager.triggers][-1] == "Fixture: setup", "no (copy) suffix"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [3, 0]
        assert "Pasted 1 trigger: reconnected 1 trigger link into them." in _status(window)
        assert _record_count(window) == records + 1, "the relink rides in the paste's record"

        window.undo()
        manager = panel._manager()
        assert len(manager.triggers) == 3
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [-1, 0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_undo_then_paste_leaves_the_restored_link_alone() -> None:
    """The undo puts the inbound link back on the original, so the cut
    block's paste must not re-point it at the copy, and says it left it."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        window.undo()
        window.status_log.clear()
        window.paste_triggers()
        manager = panel._manager()
        assert manager.triggers[4].name == "Fixture: setup"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[2])] == [0, 1]
        status = _status(window)
        assert "left 1 trigger link alone (changed or removed since the cut)" in status, status
        assert "reconnected" not in status
    finally:
        window.edit_history.mark_saved()
        window.close()


_CLEARED_SINCE_COPY = "cleared 1 trigger link (target removed since the copy)"


_LINK_TARGET_CASES = ["untouched", "edited and undone", "deleted"]


def _change_link_target(window, panel, case: str) -> None:
    """Act on trigger 0, the copied block's link target, between copy and paste."""
    if case == "edited and undone":
        _rename(panel, 0, "Fixture: renamed")
        window.undo()
    elif case == "deleted":
        window.trigger_structural_edit("delete", [0])


@pytest.mark.parametrize("case", _LINK_TARGET_CASES, ids=[f"target {case}" for case in _LINK_TARGET_CASES])
def test_gh137_a_same_document_trigger_paste_says_it_cleared_a_link(case: str) -> None:
    """A link resolves by identity, so only a deleted target lands on -1, and
    that must not be silent. Since TASK-003.06 identity survives an undo (the
    restore writes content back into the same object), so a target edited and
    then undone still resolves."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        window.copy_triggers()
        _change_link_target(window, panel, case)
        window.status_log.clear()
        window.paste_triggers()
        manager = panel._manager()
        pasted = [t for t in manager.triggers if t.name == "Fixture: references (copy)"]
        assert len(pasted) == 1
        refs = [ce.trigger_id for ce in trigger_clipboard_refs(pasted[0])]
        assert refs == ([-1, 0] if case == "deleted" else [0, 1])
        status = _status(window)
        if case == "deleted":
            assert f"Pasted 1 trigger: {_CLEARED_SINCE_COPY}." in status, status
        else:
            assert "cleared" not in status, status
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("case", _LINK_TARGET_CASES, ids=[f"target {case}" for case in _LINK_TARGET_CASES])
def test_gh137_a_same_document_entry_paste_says_it_cleared_a_link(case: str) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        _change_link_target(window, panel, case)
        last = len(panel._manager().triggers) - 1
        panel.select_trigger(last)
        window.status_log.clear()
        window.paste_entries()
        assert panel._manager().triggers[last].name == "Fixture: variable"
        pasted = panel._manager().triggers[last].effects[-1]
        assert pasted.trigger_id == (-1 if case == "deleted" else 0)
        status = _status(window)
        if case == "deleted":
            assert f"Pasted 1 effect: {_CLEARED_SINCE_COPY}." in status, status
        else:
            assert "cleared" not in status, status
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_edit_cut_and_ctrl_x_cut_the_selected_triggers() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel = window.trigger_panel
        panel.select_trigger(1)
        panel.tree.setFocus()
        QApplication.processEvents()
        assert window.cut_action.text() == "Cu&t Triggers" and window.cut_action.isEnabled()
        QTest.keyClick(panel.tree, Qt.Key_X, Qt.ControlModifier)
        QApplication.processEvents()
        assert [t.name for t in panel._manager().triggers] == [
            "Fixture: setup", "Fixture: references", "Fixture: variable",
        ]
        assert window._trigger_clipboard.from_cut

        panel.select_trigger(2)
        window.cut_action.trigger()
        assert len(panel._manager().triggers) == 2
        assert panel.selected_trigger_indices() == [1], "the last row cut lands on the one above"

        window.mode_combo.setCurrentText("Terrain")
        assert window.cut_action.text() == "Cu&t Triggers"
        assert not window.cut_action.isEnabled(), "there is no Cut Region"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_failed_cut_leaves_the_clipboard_alone(monkeypatch) -> None:
    """_trigger_edit reports a raised body instead of re-raising, so the block
    is committed only once the triggers are really gone."""
    warnings = []
    window = _triggers_window()
    window._warn = lambda title, text: warnings.append(text)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        manager = window._ensure_trigger_edits().manager()

        def refuse(_indices):
            raise RuntimeError("refused")

        monkeypatch.setattr(manager, "remove_triggers", refuse)
        window.cut_triggers()
        assert len(warnings) == 1
        assert window._trigger_clipboard is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy order", "display order"])
def test_gh137_pasting_a_cut_block_on_a_legacy_order_file_says_it_runs_last(legacy: bool) -> None:
    from types import SimpleNamespace

    from descape.trigger_model import exec_order_value

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert exec_order_value(window.scenario) == 0
        if legacy:
            # Map Options' own route: a pending flag counts, as the status line reads it.
            window._set_exec_order(SimpleNamespace(label="execution order"), 1)
            assert window.trigger_edits.exec_order == 1
        panel.select_trigger(0)
        window.cut_triggers()
        window.paste_triggers()
        said = "the pasted triggers run last wherever they are listed" in _status(window)
        assert said is legacy
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy order", "display order"])
def test_task053_02_moving_triggers_on_a_legacy_order_file_warns_once(legacy: bool) -> None:
    """Move Up/Down and Move to Bottom permute display order only, which a
    legacy file does not run by: said on the first move, then not again."""
    from types import SimpleNamespace

    from descape.viewer_triggers import LEGACY_REORDER_MESSAGE

    window = _triggers_window()
    try:
        if legacy:
            window._set_exec_order(SimpleNamespace(label="execution order"), 1)
        for op, indices in (("move_down", [0]), ("move_up", [1]), ("move_bottom", [0])):
            window.trigger_structural_edit(op, indices)
            assert _status(window).count(LEGACY_REORDER_MESSAGE) == (1 if legacy else 0), op
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh2_a_cut_trigger_reconnects_its_links_when_pasted_after_a_renumber() -> None:
    from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

    from descape.trigger_model import ids_follow_display_order

    window = _gh2_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_down", [1])
        assert _names(window)[2] == "Fixture: references" and _names(window)[0] == "Fixture: setup"
        panel.select_trigger(0)
        window.cut_triggers()
        assert not ids_follow_display_order(_order(window)), "the cut leaves a renumber to do"
        window.trigger_structural_edit("renumber", [])
        window.paste_triggers()
        manager = panel._manager()
        holder = next(t for t in manager.triggers if t.name == "Fixture: references")
        linked = [manager.triggers[ce.trigger_id].name for ce in get_trigger_referencing_ce(holder) if ce.trigger_id >= 0]
        assert "Fixture: setup" in linked
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_focused_editor_retitles_edit_cut_copy_paste() -> None:
    """Focus moves retitle the three, a tree-to-editor move included, which
    _on_app_focus_changed's text-box early return used to swallow."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        name = _row_widget(panel, "name")
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel.tree.setFocus()
        QApplication.processEvents()
        assert [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)] == [
            "Cu&t Triggers", "&Copy Triggers", "&Paste Triggers",
        ]
        assert not window.paste_action.isEnabled(), "no triggers to paste"

        name.setFocus()
        QApplication.processEvents()
        assert [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)] == [
            "Cu&t", "&Copy", "&Paste",
        ]
        assert all(a.isEnabled() for a in (window.cut_action, window.copy_action, window.paste_action))
        assert window.cut_action.toolTip() == "Cut the selected text"

        name.setReadOnly(True)
        panel.tree.setFocus()
        QApplication.processEvents()
        name.setFocus()
        QApplication.processEvents()
        assert window.copy_action.isEnabled()
        assert not window.cut_action.isEnabled() and not window.paste_action.isEnabled(), "read-only text"

        panel.tree.setFocus()
        QApplication.processEvents()
        assert window.cut_action.text() == "Cu&t Triggers"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_the_entry_menu_offers_cut_copy_paste_with_its_own_clipboard() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        menu = panel.entry_context_menu(panel._entry_item_for("effect", 1))
        assert _texts(menu) == [
            "Cut", "Copy", "Paste", "Move Up", "Move Down", "Move to Top", "Move to Bottom", "", "Select Same Type",
        ]
        assert menu.toolTipsVisible()
        assert _action(menu, "Cut").isEnabled() and _action(menu, "Copy").isEnabled()
        assert not _action(menu, "Paste").isEnabled(), "nothing to paste yet"

        _action(menu, "Copy").trigger()
        assert window._entry_clipboard.label == "2 effects"
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "Copy records nothing"
        assert window._trigger_clipboard is None, "two slots: the trigger one is untouched"
        assert not panel.trigger_paste_button.isEnabled(), "the trigger tree's Paste ignores an entry block"
        assert not _action(panel.trigger_context_menu(panel._item_for_index[3]), "Paste").isEnabled()

        panel.select_trigger(2)
        menu = panel.entry_context_menu(panel.entry_tree.topLevelItem(0))
        assert not _action(menu, "Cut").isEnabled(), "no condition or effect selected"
        assert _action(menu, "Paste").isEnabled()
        panel.select_trigger(0)
        window.copy_triggers()
        assert window._entry_clipboard.label == "2 effects", "and copying triggers keeps the entry block"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_entry_paste_lands_below_the_clicked_effect_of_another_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        window.copy_entries_to_clipboard()
        panel.select_trigger(2)
        records = _record_count(window)
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste").trigger()
        assert _types_of(window, 2) == [8, 19, 11, 9]
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Paste 2 effects"
        assert panel.selected_entry_refs() == [("effect", 1), ("effect", 2)]

        window.undo()
        assert _types_of(window, 2) == [8, 9]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("row", [0, 1, 2], ids=["trigger row", "conditions heading", "effects heading"])
def test_gh137_entry_paste_on_a_non_entry_row_appends_to_each_list(row: int) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("condition", 1), ("effect", 0))
        window.copy_entries_to_clipboard()
        panel.select_trigger(2)
        _action(panel.entry_context_menu(panel.entry_tree.topLevelItem(row)), "Paste").trigger()
        assert _types_of(window, 2, "condition") == [10, 5]
        assert _types_of(window, 2) == [8, 9, 56]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_entries_is_one_record_and_pastes_into_another_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("condition", 1), ("effect", 0))
        records = _record_count(window)
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Cut").trigger()
        assert (len(_types_of(window, 3, "condition")), len(_types_of(window, 3))) == (3, 6)
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Cut 2 entries"
        assert window._entry_clipboard.label == "1 condition and 1 effect"

        window.undo()
        assert (len(_types_of(window, 3, "condition")), len(_types_of(window, 3))) == (4, 7)
        assert window._entry_clipboard is not None, "the clipboard keeps the block"
        panel.select_trigger(0)
        window.paste_entries()
        assert _types_of(window, 0, "condition") == [10, 5]
        assert _types_of(window, 0) == [20, 56]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_refused_entry_paste_says_why_and_records_nothing() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        window._entry_clipboard = dataclasses.replace(window._entry_clipboard, scenario_version="1.56")
        window._sync_entry_clipboard_state()
        expected = "Copied from a 1.56 scenario; this one is 1.58."
        paste = _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste")
        assert paste.isEnabled() and paste.toolTip() == expected
        records = _record_count(window)
        paste.trigger()
        assert f"Cannot paste these conditions/effects: {expected}" in _status(window)
        assert _record_count(window) == records
        assert len(_types_of(window, 3)) == 7
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_an_entry_paste_into_another_file_reports_and_undoes_in_one_step(tmp_path: Path, new_maps_are_1_58) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, *[("condition", i) for i in range(4)], *[("effect", i) for i in range(7)])
        window.copy_entries_to_clipboard()
        window.edit_history.mark_saved()
        window.new_map()
        assert window.scenario.scenario_version == "1.58"
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_structural_edit("new", [])
        panel.select_trigger(0)
        window.paste_entries()
        assert len(_types_of(window, 0)) == 7
        assert (
            "Pasted 4 conditions and 7 effects from another scenario: cleared 5 unit references, named 1 variable."
            in _status(window)
        ), _status(window)
        manager = panel._manager()
        assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var")]
        out = tmp_path / "pasted.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded_scenario = load_map_and_units(out)
        reloaded = parse_triggers(reloaded_scenario)
        assert len(reloaded.triggers[0].effects) == 7
        assert [(v.variable_id, v.name) for v in reloaded.variables] == [(0, "fixture_var")], "the name is saved"

        window.undo()
        manager = panel._manager()
        assert (len(manager.triggers[0].effects), len(manager.variables)) == (0, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_failed_entry_cut_leaves_the_clipboard_alone(monkeypatch) -> None:
    warnings = []
    window = _triggers_window()
    window._warn = lambda title, text: warnings.append(text)
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        trigger = window._ensure_trigger_edits().manager().triggers[3]

        def refuse(**_kwargs):
            raise RuntimeError("refused")

        monkeypatch.setattr(trigger, "remove_effect", refuse)
        window.cut_entries()
        assert len(warnings) == 1
        assert window._entry_clipboard is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_repopulate_pushes_the_entry_paste_refusal() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        window._entry_clipboard = dataclasses.replace(window._entry_clipboard, scenario_version="1.56")
        window.mode_combo.setCurrentText("Terrain")
        window.mode_combo.setCurrentText("Triggers")
        panel.select_trigger(3)
        paste = _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste")
        assert paste.toolTip() == "Copied from a 1.56 scenario; this one is 1.58."
    finally:
        window.edit_history.mark_saved()
        window.close()


def _edit_texts(window) -> list[str]:
    return [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)]


def test_gh137_edit_cut_copy_paste_follow_the_focused_tree() -> None:
    """A tree-to-tree focus move retitles the three, and an entry selection
    change re-gates them while the entry tree keeps focus."""
    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        assert _edit_texts(window) == ["Cu&t Conditions/Effects", "&Copy Conditions/Effects", "&Paste Conditions/Effects"]
        _select_entries(panel)
        panel.entry_tree.clearSelection()
        assert not window.cut_action.isEnabled() and not window.paste_action.isEnabled()
        _select_entries(panel, ("effect", 1))
        assert window.cut_action.isEnabled() and window.copy_action.isEnabled()

        _focus_on(panel.tree)
        assert _edit_texts(window) == ["Cu&t Triggers", "&Copy Triggers", "&Paste Triggers"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_two_row_entry_selection_pops_up_no_discarded_form_row() -> None:
    """select_entries() of two rows builds the form twice in one gesture. The
    first form's rows still have Qt's queued show pending when they are
    discarded; unhidden, it mapped one as a top-level window that took focus."""
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        # Only windows this gesture shows: an earlier test on the worker may have left one up.
        shown_before = [w for w in QApplication.topLevelWidgets() if w.isVisible()]
        panel.select_entries([("effect", 1), ("effect", 2)])
        QApplication.processEvents()
        strays = [
            w for w in QApplication.topLevelWidgets()
            if w.isVisible() and w not in shown_before and w is not window and not window.isAncestorOf(w)
        ]
        assert strays == []
        assert panel.entry_tree.hasFocus()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_ctrl_x_c_v_on_the_entry_tree_act_on_conditions_and_effects() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        QTest.keyClick(panel.entry_tree, Qt.Key_C, Qt.ControlModifier)
        QApplication.processEvents()
        assert window._entry_clipboard.label == "2 effects"
        assert window._trigger_clipboard is None

        panel.select_trigger(2)
        _focus_on(panel.entry_tree)
        _select_entries(panel, ("effect", 0))
        QTest.keyClick(panel.entry_tree, Qt.Key_V, Qt.ControlModifier)
        QApplication.processEvents()
        assert _types_of(window, 2) == [8, 19, 11, 9], "below the current effect"
        assert len(panel._manager().triggers) == 4, "no trigger was pasted"
        assert panel.entry_tree.hasFocus(), "selecting the two pasted rows keeps the focus"

        _select_entries(panel, ("effect", 3))
        QTest.keyClick(panel.entry_tree, Qt.Key_X, Qt.ControlModifier)
        QApplication.processEvents()
        assert _types_of(window, 2) == [8, 19, 11]
        assert window._entry_clipboard.label == "1 effect"
        assert len(panel._manager().triggers) == 4, "no trigger was cut"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("opener", ["new", "retype"])
def test_gh137_edit_cut_copy_paste_stay_off_while_the_type_picker_is_up(opener: str) -> None:
    """The right-click menus' gate: with New or Type… open on trigger T, a
    click on T's row and Ctrl+X must not cut T out from under the picker."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        window.copy_triggers()  # a held block, so a stray Paste would show
        if opener == "retype":
            _select_entries(panel, ("effect", 0))
        panel._request_entry_op(opener)
        QApplication.processEvents()
        assert panel.picker_showing()
        item = panel._item_for_index[3]
        panel.tree.scrollToItem(item)
        rect = panel.tree.visualItemRect(item)
        QTest.mouseClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(rect.left() + 8, rect.center().y()))
        QApplication.processEvents()
        assert panel.picker_showing() and panel.tree.hasFocus(), "the click leaves the picker up"
        assert _edit_texts(window) == ["Cu&t Triggers", "&Copy Triggers", "&Paste Triggers"]
        assert not any(a.isEnabled() for a in (window.cut_action, window.copy_action, window.paste_action))

        for name, key in (("X", Qt.Key_X), ("V", Qt.Key_V)):
            QTest.keyClick(panel.tree, key, Qt.ControlModifier)
            QApplication.processEvents()
            assert len(panel._manager().triggers) == 4, f"Ctrl+{name} acted with the picker up"
            assert panel.picker_showing()

        panel._close_picker()
        QApplication.processEvents()
        assert window.cut_action.isEnabled() and window.paste_action.isEnabled(), "closing the picker re-gates them"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()
