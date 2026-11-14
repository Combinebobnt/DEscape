"""Trigger panel: divider sections, collapsing, section verbs and moves
(GH #133, GH #134), and their saved bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _action,
    _ctrl_select,
    _display_names,
    _divided_window,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _open,
    _order,
    _record_count,
    _rename,
    _roomy_tree,
    _saved_body,
    _sectioned_window,
    _trigger_row_count,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- navigation affordances: the Sections menu ---------------------------------


def _section_actions(panel) -> list[tuple[str, int]]:
    return [(a.text(), a.data()) for a in panel._section_actions]


def _expanded(panel) -> list[bool]:
    return [panel.tree.topLevelItem(i).isExpanded() for i in range(panel.tree.topLevelItemCount())]


def _flat_sections_menu(panel) -> None:
    """Shown on a flat tree, with nothing to collapse, expand or jump to."""
    assert not panel.sections_button.isHidden()
    assert not panel.collapse_all_action.isEnabled()
    assert not panel.expand_all_action.isEnabled()
    assert panel._section_actions == []


def test_the_sections_button_lists_every_section_and_stays_when_flat() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _flat_sections_menu(panel)
        assert panel.sections_button.isEnabled()
        assert panel.section_new_action.isEnabled(), "a flat tree can still gain its first section"
        window.close()
        window = _sectioned_window()
        panel = window.trigger_panel
        assert not panel.sections_button.isHidden()
        assert panel.collapse_all_action.isEnabled() and panel.expand_all_action.isEnabled()
        texts = [a.text() for a in panel.sections_menu.actions() if not a.isSeparator()]
        assert texts[:5] == ["New Section…", "Rename Section…", "Move to Section", "Collapse All", "Expand All"]
        assert panel.sections_menu.actions()[2].menu() is panel.section_move_menu
        assert _section_actions(panel) == [
            ("(before the first section)", 0),
            ("--RISK!!!--", 2),
            ("--RISK!!!--", 4),
            ("(unnamed section)", 6),
        ]
        panel.sort_combo.setCurrentIndex(1)  # File order
        _flat_sections_menu(panel)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_clicking_the_synthetic_header_shows_no_trigger() -> None:
    """GH #1: the "(before the first section)" row is not a trigger, so a click
    on it leaves the entry tree empty rather than showing a stale trigger."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _roomy_tree(panel)
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel.select_trigger(1)
        assert panel.entry_tree.topLevelItemCount() > 0, "fixture assumption: a trigger is showing"

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None, "fixture assumption: row 0 is the synthetic header"
        pos = panel.tree.visualItemRect(synthetic).center()
        QTest.mouseClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
        QApplication.processEvents()
        assert panel.tree.currentItem() is synthetic, "the click did not land on the header"
        assert panel.current_trigger_index() is None
        assert panel.entry_tree.topLevelItemCount() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_the_only_divider_back_folds_the_tree_flat() -> None:
    """GH #1: undoing the divider by hand, not by Undo, drops the grouping."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 2, "--- Second ---")
        assert panel.collapse_all_action.isEnabled(), "fixture assumption: the rename grouped the tree"

        _rename(panel, 2, "Fixture: references")
        _flat_sections_menu(panel)
        assert all(panel.tree.topLevelItem(i).childCount() == 0 for i in range(panel.tree.topLevelItemCount()))
        assert _trigger_row_count(panel.tree) == panel.tree.topLevelItemCount() == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_collapsed_section_survives_a_rename_rebuild() -> None:
    """Collapse the second of two same-titled sections, then rename a
    different trigger across the divider predicate, which rebuilds the tree
    through _changed()'s escalation. Only that one section stays collapsed."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        assert panel._collapsed_keys == {("--RISK!!!--", 1)}

        _rename(panel, 7, "--- Last ---")
        assert [s.header_index for s in panel._sections] == [None, 2, 4, 6, 7], "the rename rebuilt the tree"
        assert not panel._item_for_index[4].isExpanded()
        assert panel._item_for_index[2].isExpanded(), "its same-titled twin is a different section"
        assert panel._item_for_index[6].isExpanded()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_all_survives_a_rebuild_and_expand_all_undoes_it() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)  # a header: selecting it expands no parent
        panel.collapse_all_action.trigger()
        assert _expanded(panel) == [False] * 4
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [False] * 4

        panel.expand_all_action.trigger()
        assert panel._collapsed_keys == set()
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [True] * 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_selects_the_divider_or_the_first_member_and_expands_it() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.collapse_all_action.trigger()
        panel._section_actions[2].trigger()
        assert panel.current_trigger_index() == 4
        assert panel.selected_trigger_indices() == [4]
        assert panel._item_for_index[4].isExpanded()
        assert ("--RISK!!!--", 1) not in panel._collapsed_keys, "a jumped-to section stays open on rebuild"

        panel._section_actions[0].trigger()
        assert panel.current_trigger_index() == 0, "the synthetic section's first member"
        assert panel.tree.topLevelItem(0).isExpanded()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_filter_reveals_a_match_in_a_collapsed_section_and_clearing_restores() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        panel.filter_edit.setText("needle")
        match = panel._item_for_index[5]
        assert not match.isHidden()
        assert match.parent().isExpanded()
        assert panel._collapsed_keys == {("--RISK!!!--", 1)}, "the filter pass records nothing"

        panel.filter_edit.clear()
        assert _expanded(panel) == [True, True, False, True]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_all_under_an_active_filter_keeps_the_matches_visible() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.filter_edit.setText("needle")
        panel.collapse_all_action.trigger()
        match = panel._item_for_index[5]
        assert not match.isHidden() and match.parent().isExpanded()

        panel.filter_edit.clear()
        assert _expanded(panel) == [False] * 4, "the collapse is what clearing the filter restores"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_to_a_filtered_out_section_is_disabled() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.filter_edit.setText("needle")
        panel.sections_menu.aboutToShow.emit()
        assert [a.isEnabled() for a in panel._section_actions] == [False, False, True, False]
        panel.filter_edit.clear()
        panel.sections_menu.aboutToShow.emit()
        assert all(a.isEnabled() for a in panel._section_actions)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_a_divider_to_another_divider_retitles_its_section() -> None:
    """No repartition, so no rebuild: the menu label and the collapse keys
    have to follow the new title on their own."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        _rename(panel, 2, "--- Renamed ---")
        assert [text for text, _ in _section_actions(panel)][1:3] == ["--- Renamed ---", "--RISK!!!--"]
        assert panel._collapsed_keys == {("--RISK!!!--", 0)}
        panel.select_trigger(0)
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [True, True, False, True]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_state_is_dropped_with_the_document() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.collapse_all_action.trigger()
        assert panel._collapsed_keys
        panel.clear_document()
        assert panel._collapsed_keys == set()
        _flat_sections_menu(panel)
        assert not panel.sections_button.isEnabled(), "no document"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- section management: New Section, Rename Section, Move to Section -----------


def test_new_section_groups_a_divider_free_file_in_the_default_format() -> None:
    from descape.trigger_organize import is_divider, sections

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert not panel._grouped, "fixture assumption: no dividers"
        before = _record_count(window)
        window.trigger_structural_edit("new_section", [1], "Later")
        names = _names(window)
        assert len(names) == 5 and names[4] == "--- Later ---"
        assert is_divider(names[4])
        assert _order(window) == [0, 1, 2, 3, 4], "the only section's end is the end"
        assert [s.header_index for s in sections(names, _order(window))] == [None, 4]
        assert panel._grouped
        assert panel.current_trigger_index() == 4
        assert _record_count(window) == before + 1
        assert window.trigger_edits.has_edits
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_section_lands_after_the_current_section_in_the_files_own_format() -> None:
    window = _divided_window()
    try:
        window.trigger_structural_edit("new_section", [0], "Middle")
        assert _names(window)[4] == "--Middle--", "the file's only titled divider is tight 2+2"
        assert _order(window) == [0, 1, 4, 2, 3]
        window.trigger_structural_edit("new_section", [], "Last")
        assert _order(window)[-1] == 5
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_section_leaves_collapsed_sections_collapsed() -> None:
    """GH #135: the rebuild restored the old current trigger with autoScroll
    on, expanding its collapsed section just before the new divider took over."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_triggers([5, 3])
        panel.collapse_all_action.trigger()
        keys = {("", 0), ("--RISK!!!--", 0), ("--RISK!!!--", 1), ("------", 0)}
        assert panel._collapsed_keys == keys
        assert _expanded(panel) == [False] * 4

        window.trigger_structural_edit("new_section", [5], "Z")
        assert [s.header_index for s in panel._sections] == [None, 2, 4, 8, 6]
        assert panel.current_trigger_index() == 8
        assert panel._collapsed_keys == keys
        assert _expanded(panel) == [False, False, False, True, False], "only the new section is open"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_undo_reverses_the_whole_new_section_gesture() -> None:
    window = _divided_window()
    try:
        order = _order(window)
        window.trigger_structural_edit("new_section", [0], "Middle")
        window.undo()
        assert len(_names(window)) == 4
        assert _order(window) == order
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_empty_or_unreadable_title_records_nothing(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warned.append(args)))
    window = _divided_window()
    try:
        before = _record_count(window)
        window.trigger_structural_edit("new_section", [0], "   ")
        assert not warned
        window.trigger_structural_edit("new_section", [0], "ends in a dash-")
        assert len(warned) == 1
        assert _record_count(window) == before and len(_names(window)) == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_to_section_lands_at_the_end_of_that_section() -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        window.trigger_structural_edit("move_to_section", [0], 2)
        assert _order(window) == [1, 2, 3, 0]
        assert panel.selected_trigger_indices() == [0]
        assert _record_count(window) == before + 1
        assert window.trigger_edits.has_edits, "a pure permutation must still dirty the model"
        window.trigger_structural_edit("move_to_section", [3], None)
        assert _order(window) == [1, 3, 2, 0]
        window.undo()
        assert _order(window) == [1, 2, 3, 0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_to_section_moves_a_selection_as_one_block() -> None:
    window = _divided_window(at=3)
    try:
        window.trigger_structural_edit("new_section", [3], "B")
        assert _order(window) == [0, 1, 2, 3, 4]
        window.trigger_structural_edit("move_to_section", [2, 0], 4)
        assert _order(window) == [1, 3, 4, 0, 2]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_move_that_changes_nothing_or_moves_a_header_records_nothing() -> None:
    window = _divided_window()
    try:
        before = _record_count(window)
        order = _order(window)
        window.trigger_structural_edit("move_to_section", [3], 2)
        window.trigger_structural_edit("move_to_section", [2], None)
        window.trigger_structural_edit("move_to_section", [0], 3)
        assert _record_count(window) == before
        assert _order(window) == order
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_empty_leading_section_is_a_move_target_at_slot_zero() -> None:
    window = _divided_window(at=0)
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        panel._populate_move_menu()
        actions = panel.section_move_menu.actions()
        assert [(a.text(), a.data(), a.isEnabled()) for a in actions] == [
            ("(before the first section)", None, True),
            ("--A--", 0, False),
        ]
        actions[0].trigger()
        assert _order(window) == [2, 0, 1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_move_menu_keys_on_the_header_so_duplicate_titles_stay_apart() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel._populate_move_menu()
        assert [(a.text(), a.data(), a.isEnabled()) for a in panel.section_move_menu.actions()] == [
            ("(before the first section)", None, False),
            ("--RISK!!!--", 2, True),
            ("--RISK!!!--", 4, True),
            ("(unnamed section)", 6, True),
        ]
        panel.section_move_menu.actions()[2].trigger()
        assert _order(window) == [1, 2, 3, 4, 5, 0, 6, 7]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_section_keeps_the_decoration_and_retitles_the_section() -> None:
    window = _divided_window("--- Old ----")
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        panel.select_trigger(2)
        panel.rename_section(2, "New")
        assert _names(window)[2] == "--- New ----"
        assert [s.title for s in panel._sections if s.header_index == 2] == ["--- New ----"]
        assert _record_count(window) == before + 1
        panel.rename_section(2, "New")
        assert _record_count(window) == before + 1, "an unchanged title records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_section_refuses_a_title_that_would_not_read_back(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warned.append(args)))
    window = _divided_window()
    try:
        before = _record_count(window)
        window.trigger_panel.rename_section(2, "x-")
        assert warned and _names(window)[2] == "--A--"
        assert _record_count(window) == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def _save_divided_fixture(tmp_path: Path) -> Path:
    """The fixture with trigger 2 saved as "--A--", as a real file."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    model.manager().triggers[2].name = "--A--"
    model.mark_dirty(2)
    base = tmp_path / "divided.aoe2scenario"
    write_scenario(loaded, base, triggers=model)
    return base


def test_a_saved_move_to_section_changes_only_the_display_order_bytes(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_structural_edit("move_to_section", [0], 2)
        edited = tmp_path / "moved.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
        regions = window.trigger_edits.regions
    finally:
        window.edit_history.mark_saved()
        window.close()
    before, after = _saved_body(base), _saved_body(edited)
    assert len(before) == len(after)
    diffs = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert diffs, "the move changed no byte"
    units_end = load_map_and_units(base).units_section_end
    start, end = units_end + regions.triggers_end, units_end + regions.display_order_end
    assert all(start <= i < end for i in diffs), "a byte outside the display-order array changed"
    reloaded = _open(edited)
    try:
        assert _order(reloaded) == [1, 2, 3, 0]
    finally:
        reloaded.close()


def test_a_saved_new_section_round_trips(tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_structural_edit("new_section", [0], "Middle")
        edited = tmp_path / "new_section.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded = _open(edited)
    try:
        names = _names(reloaded)
        assert len(names) == 5 and names[4] == "--Middle--"
        assert _order(reloaded) == [0, 1, 4, 2, 3]
        assert [s.header_index for s in reloaded.trigger_panel._sections] == [None, 4, 2]
    finally:
        reloaded.close()


def test_a_saved_retitle_leaves_every_other_trigger_blob_alone(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_panel.rename_section(2, "Longer title")
        edited = tmp_path / "retitled.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    before = TriggerEditModel(load_map_and_units(base))._blobs
    after_model = TriggerEditModel(load_map_and_units(edited))
    assert after_model.manager().triggers[2].name == "--Longer title--"
    after = after_model._blobs
    assert [b for i, b in enumerate(before) if i != 2] == [b for i, b in enumerate(after) if i != 2]
    assert before[2] != after[2]


def test_a_new_section_on_a_real_scenario_matches_its_format_and_survives_save(
    scenario_path, tmp_path
) -> None:
    from descape.scenario_write import write_scenario
    from descape.trigger_organize import divider_format, format_divider, is_divider, sections

    window = _open(scenario_path)
    try:
        if not window.scenario.trigger_read_supported or not window.scenario.trigger_write_supported:
            pytest.skip(f"{scenario_path.name}: triggers are not writable")
        panel = window.trigger_panel
        if panel._manager() is None or not panel._manager().triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")
        expected = format_divider("DEscape probe", divider_format(_names(window)))
        count = len(_names(window))
        window.trigger_structural_edit("new_section", [_order(window)[0]], "DEscape probe")
        assert _names(window)[count] == expected
        edited = tmp_path / scenario_path.name
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded = _open(edited)
    try:
        names, order = _names(reloaded), _order(reloaded)
        assert len(names) == count + 1 and names[count] == expected and is_divider(expected)
        assert count in [s.header_index for s in sections(names, order)]
    finally:
        reloaded.close()


# -- GH #134: a collapsed section header counts as its whole section ------------


def _ab_window(collapse_a: bool = True):
    """The trigger fixture as two sections, --- A --- 0: 1 | --- B --- 2: 3,
    renamed through the widget, with A collapsed by default."""
    window = _triggers_window()
    panel = window.trigger_panel
    _rename(panel, 0, "--- A ---")
    _rename(panel, 2, "--- B ---")
    assert [s.header_index for s in panel._sections] == [0, 2], "fixture assumption"
    panel._item_for_index[0].setExpanded(not collapse_a)
    return window


def _section_names(window) -> list[tuple[str, list[str]]]:
    panel = window.trigger_panel
    triggers = panel._manager().triggers
    return [(triggers[s.header_index].name, [triggers[m].name for m in s.member_indices]) for s in panel._sections]


def test_gh134_duplicating_a_collapsed_section_copies_it_whole_after_itself() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert not panel._item_for_index[0].isExpanded(), "selecting the header leaves it collapsed"
        records = _record_count(window)
        panel.trigger_copy_button.click()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split",
            "--- A (copy) ---", "Fixture: armour split (copy)",
            "--- B ---", "Fixture: variable",
        ]
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- A (copy) ---", ["Fixture: armour split (copy)"]),
            ("--- B ---", ["Fixture: variable"]),
        ]
        assert _record_count(window) == records + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_deleting_a_collapsed_section_removes_it_whole_in_one_record() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        records = _record_count(window)
        panel.trigger_delete_button.click()
        assert _display_names(window) == ["--- B ---", "Fixture: variable"]
        assert _record_count(window) == records + 1
        window.undo()
        assert _display_names(window) == ["--- A ---", "Fixture: armour split", "--- B ---", "Fixture: variable"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_cutting_a_collapsed_section_cuts_it_whole() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        assert _display_names(window) == ["--- B ---", "Fixture: variable"]
        assert [t.name for t in window._trigger_clipboard.triggers] == ["--- A ---", "Fixture: armour split"]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("collapse_b", [True, False], ids=["B collapsed", "B expanded"])
def test_gh134_a_copied_collapsed_section_pastes_as_its_own_section_after_b(collapse_b: bool) -> None:
    """Collapsed, the panel resolves the anchor to B's last member; expanded,
    the window's divider-block rule does, so B never loses its member."""
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_clipboard_copy_button.click()
        assert [t.name for t in window._trigger_clipboard.triggers] == ["--- A ---", "Fixture: armour split"]
        panel._item_for_index[2].setExpanded(not collapse_b)
        panel.select_trigger(2)
        panel.trigger_paste_button.click()
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- B ---", ["Fixture: variable"]),
            ("--- A (copy) ---", ["Fixture: armour split (copy)"]),
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("entry", ["edit", "button", "menu"])
def test_gh134_a_plain_paste_onto_a_collapsed_header_lands_at_its_section_end(entry: str) -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        window.copy_triggers()
        panel.select_trigger(0)
        if entry == "edit":
            window.paste_triggers()
        elif entry == "button":
            panel.trigger_paste_button.click()
        else:
            _action(panel.trigger_context_menu(panel._item_for_index[0]), "Paste").trigger()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split", "Fixture: variable (copy)", "--- B ---", "Fixture: variable",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_duplicating_an_expanded_header_alone_leaves_its_members() -> None:
    """The split guard: the copy is an empty section after A, not a header
    that takes A's member."""
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_copy_button.click()
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- A (copy) ---", []),
            ("--- B ---", ["Fixture: variable"]),
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_scattered_duplicates_each_land_after_their_source() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 3)
        panel.trigger_copy_button.click()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split", "Fixture: armour split (copy)",
            "--- B ---", "Fixture: variable", "Fixture: variable (copy)",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_move_up_and_down_stay_header_only_on_a_collapsed_section() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_move_down_button.click()
        assert _order(window) == [1, 0, 2, 3]
        window.undo()
        panel._item_for_index[2].setExpanded(False)
        panel.select_trigger(2)
        panel.trigger_move_up_button.click()
        assert _order(window) == [0, 2, 1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #133: Section Up / Section Down move a whole section -------------------


@pytest.mark.parametrize("collapse_a", [True, False], ids=["A collapsed", "A expanded"])
def test_gh133_section_down_swaps_two_sections_in_one_record(collapse_a: bool) -> None:
    """The two renames push records of their own, so the pin is that the top
    record is the section move and one undo restores the order with the
    divider names still in place."""
    from descape.edit_history import TriggerDiffRecord

    window = _ab_window(collapse_a=collapse_a)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert panel.section_down_button.isEnabled()
        assert not panel.section_up_button.isEnabled()
        records = _record_count(window)
        panel.section_down_button.click()
        assert _order(window) == [2, 3, 0, 1]
        assert _section_names(window) == [
            ("--- B ---", ["Fixture: variable"]),
            ("--- A ---", ["Fixture: armour split"]),
        ]
        assert _record_count(window) == records + 1
        record = window.edit_history.peek_undo()
        assert isinstance(record, TriggerDiffRecord) and record.label == 'Move section "--- A ---" down'
        assert record.touched == []
        assert panel.selected_trigger_indices() == [0], "the selection follows the moved section"
        assert panel.section_up_button.isEnabled() and not panel.section_down_button.isEnabled()

        window.undo()
        assert _order(window) == [0, 1, 2, 3]
        assert _names(window)[0] == "--- A ---" and _names(window)[2] == "--- B ---"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_a_member_moves_its_section_and_move_up_still_moves_a_divider_alone() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        panel.section_up_button.click()
        assert _order(window) == [2, 3, 0, 1]
        assert window.edit_history.peek_undo().label == 'Move section "--- B ---" up'
        window.undo()
        panel.select_trigger(2)
        panel.trigger_move_up_button.click()
        assert _order(window) == [0, 2, 1, 3], "▲ still re-partitions: B takes A's member"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_the_leading_run_neither_moves_nor_is_swapped_past() -> None:
    """(before) 0 | --- B --- 1: 2, 3. B's Section Up would hand trigger 0
    to B, so it is off and says why; nothing is below B either."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "--- B ---")
        panel.select_trigger(1)
        assert not panel.section_up_button.isEnabled()
        assert panel.section_up_button.toolTip().endswith(" (no section above this one)")
        assert not panel.section_down_button.isEnabled()
        assert panel.section_down_button.toolTip().endswith(" (no section below this one)")
        panel.select_trigger(0)
        for button in (panel.section_up_button, panel.section_down_button):
            assert not button.isEnabled()
            assert "before the first section" in button.toolTip()
        records = _record_count(window)
        window.trigger_structural_edit("section_up", [1])
        window.trigger_structural_edit("section_down", [0])
        assert _record_count(window) == records, "a refused move records nothing"
        assert _order(window) == [0, 1, 2, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_section_moves_are_gated_like_move_up_and_down() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        assert panel.section_down_button.isEnabled()

        panel.filter_edit.setText("a")
        panel.select_trigger(1)
        assert not panel.section_down_button.isEnabled()
        assert "clear the filter" in panel.section_down_button.toolTip()
        panel.filter_edit.clear()

        _rename(panel, 3, "[t] tagged")
        panel.set_tag_filter("t")
        panel.select_trigger(3)
        assert not panel.section_up_button.isEnabled()
        assert "clear the filter or tag" in panel.section_up_button.toolTip()
        panel.set_tag_filter(None)

        panel.sort_combo.setCurrentIndex(1)
        panel.select_trigger(1)
        assert not panel.section_down_button.isEnabled()
        assert "Display order" in panel.section_down_button.toolTip()
        panel.sort_combo.setCurrentIndex(0)
        panel.select_trigger(1)
        assert panel.section_down_button.isEnabled()

        _ctrl_select(panel, 1, 3)
        assert not panel.section_down_button.isEnabled() and not panel.section_up_button.isEnabled()
        assert "one section" in panel.section_down_button.toolTip()
        records = _record_count(window)
        window.trigger_structural_edit("section_down", [1, 3])
        assert _record_count(window) == records, "a selection spanning sections records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_a_flat_file_offers_no_section_move() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        assert not panel.section_up_button.isEnabled() and not panel.section_down_button.isEnabled()
        assert "no sections" in panel.section_up_button.toolTip()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- TASK-225: collapse state follows its section, not its (title, ordinal) -----


def _collapsed_by_member(window) -> dict[str, bool]:
    """Each non-empty section's collapsed state, named by its first member:
    same-titled headers cannot tell the sections apart."""
    panel = window.trigger_panel
    triggers = panel._manager().triggers
    return {
        triggers[s.member_indices[0]].name: not panel.tree.topLevelItem(p).isExpanded()
        for p, s in enumerate(panel._sections)
        if s.member_indices
    }


def test_task225_collapse_follows_a_same_titled_section_through_a_swap_undo_redo_and_jump() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "--- X ---")
        _rename(panel, 2, "--- X ---")
        panel._item_for_index[0].setExpanded(False)
        expected = {"Fixture: armour split": True, "Fixture: variable": False}
        assert _collapsed_by_member(window) == expected, "fixture assumption"
        panel.select_trigger(0)
        panel.section_down_button.click()
        assert _order(window) == [2, 3, 0, 1]
        assert _collapsed_by_member(window) == expected
        window.undo()
        assert _order(window) == [0, 1, 2, 3]
        assert _collapsed_by_member(window) == expected
        window.redo()
        assert _collapsed_by_member(window) == expected
        window._history_jump(window.edit_history.cursor - 1)
        assert _order(window) == [0, 1, 2, 3]
        assert _collapsed_by_member(window) == expected
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task225_duplicating_a_bare_ruler_leaves_the_later_rulers_their_own_state() -> None:
    """The copy is a third `------` between the two; the second keeps its
    collapse rather than taking the first's."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "------")
        _rename(panel, 2, "------")
        panel._item_for_index[2].setExpanded(False)
        expected = {"Fixture: armour split": False, "Fixture: variable": True}
        assert _collapsed_by_member(window) == expected, "fixture assumption"
        panel.select_trigger(0)
        panel.trigger_copy_button.click()
        assert [s.member_indices == () for s in panel._sections] == [False, True, False], "the copy splits nothing"
        assert _collapsed_by_member(window) == expected
    finally:
        window.edit_history.mark_saved()
        window.close()
