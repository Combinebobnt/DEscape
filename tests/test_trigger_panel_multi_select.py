"""Trigger panel: multi-select of triggers, Select All and the selection verbs."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _action,
    _activate,
    _ctrl_select,
    _key,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _release_modifiers,
    _rename,
    _triggers_window,
    trigger_clipboard_refs,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- multi-select and clipboard (GH #27/#28) ----------------------------------


def test_the_trigger_and_entry_trees_are_multi_select_and_the_picker_is_not() -> None:
    """The picker resolves to exactly one vocabulary row; both content trees
    take a set (#27, #60)."""
    from PyQt5.QtWidgets import QAbstractItemView

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for tree in (panel.tree, panel.entry_tree):
            assert tree.selectionMode() == QAbstractItemView.ExtendedSelection
        assert panel.picker_tree.selectionMode() == QAbstractItemView.SingleSelection
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selected_trigger_indices_follow_display_order_under_either_sort() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_up", [3])  # display order [0, 1, 3, 2]
        _ctrl_select(panel, 2, 0, 3)
        assert panel.selected_trigger_indices() == [0, 3, 2]
        panel.sort_combo.setCurrentIndex(1)  # File order
        _ctrl_select(panel, 2, 0, 3)
        assert panel.selected_trigger_indices() == [0, 3, 2]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_filtered_out_row_is_never_in_the_selection() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 1)
        panel.filter_edit.setText("setup")  # hides "Fixture: armour split"
        assert panel.selected_trigger_indices() == [0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_selection_shows_a_count_and_hides_the_entry_tree_contents() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        assert panel.detail_stack.currentIndex() == panel._DETAIL_MULTI
        assert panel.multi_label.text() == "2 triggers selected"
        assert panel.entry_tree.topLevelItemCount() == 0
        for button in (panel.entry_new_button, panel.entry_copy_button, panel.entry_delete_button):
            assert not button.isEnabled()

        panel.select_trigger(1)
        assert panel.detail_stack.currentIndex() == panel._DETAIL_FORM
        assert panel.entry_tree.topLevelItemCount() == 3, "Trigger, Conditions, Effects"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_deselecting_down_to_one_restores_the_form_without_a_current_change() -> None:
    """currentItemChanged does not fire when the current row stays put, so the
    selection handler must bring the entry tree back itself."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        panel._item_for_index[2].setSelected(False)
        assert panel.current_trigger_index() == 0
        assert panel.detail_stack.currentIndex() == panel._DETAIL_FORM
        assert panel.entry_tree.topLevelItemCount() == 3
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_triggers_keeps_the_whole_set() -> None:
    """Trap 7: setCurrentItem() clears the selection, so building the set and
    then setting the current item would leave one row selected."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_triggers([3, 1])
        assert panel.current_trigger_index() == 3
        assert panel.selected_trigger_indices() == [1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_same_document_rebuild_keeps_the_selected_set() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 3)
        panel.show_scenario(window.scenario)
        assert panel.current_trigger_index() == 1
        assert panel.selected_trigger_indices() == [1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_select_walk_saves_byte_identically(tmp_path: Path) -> None:
    """Slice 0 is read-only: selecting sets, changing them and clearing them
    must not build a model or dirty a byte."""
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for selection in ([0, 1], [0, 1, 2, 3], [3, 1], [2]):
            _ctrl_select(panel, *selection)
        panel.sort_combo.setCurrentIndex(1)
        _ctrl_select(panel, 0, 3)
        panel.sort_combo.setCurrentIndex(0)
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        out = tmp_path / "walked.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_follow_the_block_bounds() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        # The step gate; a split block's button stays on for its Ctrl+click gather (TASK-356).
        _ctrl_select(panel, 0, 2)
        assert panel._trigger_move_reasons() == (" (already first)", ""), "the leading member is at slot 0"
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        _ctrl_select(panel, 1, 3)
        assert panel._trigger_move_reasons() == ("", " (already last)"), "the trailing member is last"
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        _ctrl_select(panel, 1, 2)
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        # A gathered block at an end has nowhere to go that way.
        _ctrl_select(panel, 0, 1)
        assert not panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        _ctrl_select(panel, 2, 3)
        assert panel.trigger_move_up_button.isEnabled() and not panel.trigger_move_down_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_buttons_act_on_the_whole_selection() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 2)
        panel.trigger_move_down_button.click()
        assert panel._display_slots == [0, 3, 1, 2]
        assert panel.selected_trigger_indices() == [1, 2], "the moved block stays selected"
        assert [panel.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(4)] == [0, 3, 1, 2]

        panel.trigger_delete_button.click()
        names = [t.name for t in window.trigger_edits.manager().triggers]
        assert names == ["Fixture: setup", "Fixture: variable"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copying_two_triggers_duplicates_each_in_place_with_links_to_the_originals() -> None:
    """Copy is N independent duplicates: the copy of "references" still
    activates the original "setup". Copy + Paste is the linked-block verb."""
    from descape.trigger_clipboard import get_trigger_referencing_ce

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_up", [3])  # display [0, 1, 3, 2]
        window.trigger_structural_edit("copy", [0, 2])
        manager = window.trigger_edits.manager()
        names = [manager.triggers[i].name for i in manager.trigger_display_order]
        assert names == [
            "Fixture: setup", "Fixture: setup (copy)", "Fixture: armour split",
            "Fixture: variable", "Fixture: references", "Fixture: references (copy)",
        ]
        copies = [manager.triggers[i] for i in panel.selected_trigger_indices()]
        assert [t.name for t in copies] == ["Fixture: setup (copy)", "Fixture: references (copy)"]
        setup = next(i for i, t in enumerate(manager.triggers) if t.name == "Fixture: setup")
        assert get_trigger_referencing_ce(copies[1])[0].trigger_id == setup
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_then_paste_lands_the_block_below_the_current_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert not panel.trigger_paste_button.isEnabled(), "nothing to paste yet"
        _ctrl_select(panel, 0, 2)
        window.copy_triggers()
        assert panel.trigger_paste_button.isEnabled()
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "copying records nothing"

        panel.select_trigger(1)
        panel.trigger_paste_button.click()
        manager = window.trigger_edits.manager()
        assert list(manager.trigger_display_order) == [0, 1, 4, 5, 2, 3]
        assert panel.selected_trigger_indices() == [4, 5], "the pasted block is selected"
        # A linked block: the pasted "references" activates the pasted "setup".
        refs = [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[5])]
        assert refs == [4, 1]

        window.undo()
        assert not window.edit_history.is_dirty, "the paste was one undo step"
        assert len(window.trigger_edits.manager().triggers) == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ctrl_a_in_the_filter_box_selects_its_text_not_the_triggers() -> None:
    """GH #3: the line edit claims Ctrl+A before the window's Select All does."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        # Shortcut dispatch needs an active window, as in test_keybinds.py.
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        panel = window.trigger_panel
        panel.select_trigger(1)
        panel.filter_edit.setText("Fixture")  # matches every shipped trigger
        panel.filter_edit.setFocus()
        QApplication.processEvents()
        assert panel.filter_edit.hasFocus()
        assert panel.selected_trigger_indices() == [1]

        QTest.keyClick(panel.filter_edit, Qt.Key_A, Qt.ControlModifier)
        QApplication.processEvents()
        assert panel.filter_edit.selectedText() == "Fixture"
        assert panel.selected_trigger_indices() == [1]
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_select_all_in_triggers_mode_takes_every_visible_trigger() -> None:
    """Ctrl+A dispatches on mode like Ctrl+C: every trigger in Triggers mode,
    collapsed sections included, filtered rows excluded; the whole map elsewhere."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel._item_for_index[2].setExpanded(False)
        window.select_all_action.trigger()
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]
        assert window.deselect_action.isEnabled()
        window.deselect_action.trigger()
        assert panel.selected_trigger_indices() == []
        # The focused tree's own Ctrl+A lands here, not on the window's action.
        panel.tree.selectAll()
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]

        panel.filter_edit.setText("variable")
        window.select_all_action.trigger()
        # The divider stays on screen for its matching member, so it counts.
        assert panel.selected_trigger_indices() == [2, 3]

        window.mode_combo.setCurrentText("Terrain")
        window.select_all_action.trigger()
        mm = window.scenario.map_manager
        assert window._region == (0, 0, mm.map_width, mm.map_height)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ctrl_a_and_ctrl_shift_a_keys_on_the_focused_trigger_tree() -> None:
    """GH #3 by real key presses on the activated window, the tree focused.
    Passes when: with section 2 collapsed, Ctrl+A selects [0, 1, 2, 3] (the
    collapsed member 3 included); Ctrl+Shift+A clears it to []; and with
    "variable" typed in the filter and the tree focused again, Ctrl+A selects
    only the rows left visible, [2, 3]."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel._item_for_index[2].setExpanded(False)
        assert panel._item_for_index[3].parent() is panel._item_for_index[2], "fixture assumption"
        panel.select_trigger(1)
        _activate(window)
        panel.tree.setFocus()
        QApplication.processEvents()
        assert panel.tree.hasFocus()

        _key(panel.tree, Qt.Key_A, Qt.ControlModifier)
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]
        _key(panel.tree, Qt.Key_A, Qt.ControlModifier | Qt.ShiftModifier)
        assert panel.selected_trigger_indices() == []

        panel.filter_edit.setText("variable")
        panel.tree.setFocus()
        QApplication.processEvents()
        assert panel.tree.hasFocus()
        assert sorted(i for i, row in panel._item_for_index.items() if not row.isHidden()) == [2, 3], "fixture assumption"
        _key(panel.tree, Qt.Key_A, Qt.ControlModifier)
        assert panel.selected_trigger_indices() == [2, 3]
        # The reader skips hidden rows by itself; clearing the filter shows
        # whether Select All left the hidden ones out of the selection too.
        panel.filter_edit.clear()
        QApplication.processEvents()
        assert panel.selected_trigger_indices() == [2, 3], "Ctrl+A selected rows the filter hid"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_the_trees_own_ctrl_a_takes_collapsed_members_once_select_all_is_rebound() -> None:
    """With edit_select_all at its default, Ctrl+A on the focused tree is the
    window's shortcut (measured: the tree never sees the key). Rebound away,
    the key reaches the tree's own Select All. Passes when: Select All rebound
    to Ctrl+Alt+F9, Ctrl+A on the focused tree with section 2 collapsed still
    selects [0, 1, 2, 3], the collapsed member 3 included, not Qt's
    expanded-rows-only set."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        taken = [aid for aid in window._keybind_actions if settings.get_keybind(aid) == "Ctrl+Alt+F9"]
        assert not taken, f"Ctrl+Alt+F9 is not free: {taken}"
        settings.set_keybind("edit_select_all", "Ctrl+Alt+F9")
        window.apply_keybind("edit_select_all")
        assert window.select_all_action.shortcut() == QKeySequence("Ctrl+Alt+F9")
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel._item_for_index[2].setExpanded(False)
        panel.select_trigger(1)
        _activate(window)
        panel.tree.setFocus()
        QApplication.processEvents()
        assert panel.tree.hasFocus()

        _key(panel.tree, Qt.Key_A, Qt.ControlModifier)
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_ctrl_a_key_in_terrain_mode_selects_the_whole_map() -> None:
    """GH #3's other half by a real key press. Passes when: back in Terrain
    mode with the map view focused, Ctrl+A selects the whole map region and no
    trigger selection changes; Ctrl+Shift+A clears the region."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        window.mode_combo.setCurrentText("Terrain")
        _activate(window)
        window.map_view.setFocus()
        QApplication.processEvents()
        assert window.map_view.hasFocus()
        assert window._region is None, "fixture assumption"

        _key(window.map_view, Qt.Key_A, Qt.ControlModifier)
        mm = window.scenario.map_manager
        assert window._region == (0, 0, mm.map_width, mm.map_height)
        assert panel.selected_trigger_indices() == [1]
        _key(window.map_view, Qt.Key_A, Qt.ControlModifier | Qt.ShiftModifier)
        assert window._region is None
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_programmatic_selection_replaces_even_with_ctrl_held() -> None:
    """Ctrl+V arrives with Ctrl down, and the post-edit select_triggers() must
    still replace the selection rather than toggle it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)  # leaves Ctrl held
        panel.select_triggers([3, 1])
        assert panel.selected_trigger_indices() == [1, 3]
        panel.select_trigger(2)
        assert panel.selected_trigger_indices() == [2]
    finally:
        for key in (Qt.Key_Control,):
            QTest.keyRelease(window, key)
        window.edit_history.mark_saved()
        window.close()


def test_select_section_selects_the_divider_and_its_members() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        flat_menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert not _action(flat_menu, "Select Section").isEnabled(), "no sections while flat"

        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        assert panel._grouped
        _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Section").trigger()
        assert panel.selected_trigger_indices() == [2, 3]

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None
        _action(panel.trigger_context_menu(synthetic), "Select Section").trigger()
        assert panel.selected_trigger_indices() == [0, 1], "the synthetic header itself is never selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_tag_spans_sections_and_skips_hidden_rows() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "[a] one")
        _rename(panel, 3, "[a] two")
        _rename(panel, 2, "--- Second ---")
        assert not _action(panel.trigger_context_menu(panel._item_for_index[1]), "Select Tag").isEnabled()

        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [0, 3]

        panel.filter_edit.setText("two")
        _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_edit_copy_and_paste_dispatch_on_mode() -> None:
    """One QAction pair keeps Ctrl+C / Ctrl+V; in Triggers mode it retitles
    and acts on triggers, tracking the selection as it changes."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert window.copy_action.text() == "&Copy Triggers"
        assert window.paste_action.text() == "&Paste Triggers"
        assert not window.paste_action.isEnabled(), "nothing copied yet"

        panel.tree.clearSelection()
        assert not window.copy_action.isEnabled()
        _ctrl_select(panel, 1, 3)
        assert window.copy_action.isEnabled()

        window.copy_action.trigger()
        assert window._trigger_clipboard is not None and len(window._trigger_clipboard.triggers) == 2
        assert window.paste_action.isEnabled()

        panel.select_trigger(0)
        window.paste_action.trigger()
        assert list(window.trigger_edits.manager().trigger_display_order) == [0, 4, 5, 1, 2, 3]

        window.mode_combo.setCurrentText("Terrain")
        assert window.copy_action.text() == "&Copy Region"
        assert window.paste_action.text() == "&Paste Region"
    finally:
        window.edit_history.mark_saved()
        window.close()
