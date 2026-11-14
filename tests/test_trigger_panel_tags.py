"""Trigger panel: name tags, the tag facet and the Tags menu verbs."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _TAGGED_NAMES,
    _action,
    _answer_dialogs,
    _ctrl_select,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _rename,
    _tagged_window,
    _triggers_window,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- tag management: the Tags menu's Rename Tag… and Remove Tag Everywhere ----


def _visible_ids(panel) -> list[int]:
    return sorted(i for i, item in panel._item_for_index.items() if not item.isHidden())


def test_rename_tag_is_offered_only_on_a_chosen_tag_of_a_writable_file(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        assert panel.current_tag() is None
        assert not panel.tag_rename_action.isEnabled()
        assert not panel.tag_remove_action.isEnabled()
        assert panel.tag_rename_action.toolTip() == "Pick a tag in the tag filter first"

        panel.set_tag_filter("D1")
        assert panel.tag_rename_action.isEnabled()
        assert panel.tag_remove_action.isEnabled()
        # Not gated on the text filter or the sort mode.
        panel.filter_edit.setText("setup")
        panel.sort_combo.setCurrentIndex(1)
        assert panel.tag_rename_action.isEnabled()

        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        assert panel.current_tag() == "D1"
        assert not panel.tag_rename_action.isEnabled()
        assert not panel.tag_remove_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_disabled_tags_actions_say_why(tmp_path: Path) -> None:
    """TASK-330: the four tag verbs live in the Tags menu, and a disabled one
    carries its reason in the tooltip."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        actions = (panel.tag_add_action, panel.tag_remove_selected_action, panel.tag_rename_action,
                   panel.tag_remove_action)
        assert panel.tags_menu.toolTipsVisible()
        panel.tree.clearSelection()
        assert panel.current_tag() is None
        assert panel.tags_button.isEnabled(), "the menu opens with nothing to act on, to say why"
        assert [a.isEnabled() for a in actions] == [False] * 4
        assert [a.toolTip() for a in actions] == [
            "Select a trigger to tag",
            "The selected triggers carry no tag",
            "Pick a tag in the tag filter first",
            "Pick a tag in the tag filter first",
        ]

        panel.set_tag_filter("D1")
        _ctrl_select(panel, 0)
        assert [a.isEnabled() for a in actions] == [True] * 4
        assert [a.toolTip() for a in actions] == [
            "Add a tag to every selected trigger that lacks it",
            "Remove a tag from the selected triggers only",
            panel._TAG_RENAME_TIP,
            panel._TAG_REMOVE_ALL_TIP,
        ]
        assert "arrow" not in panel._TAG_RENAME_TIP

        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        assert not panel.tags_button.isEnabled(), "read-only: nothing in the menu could run"
        assert [a.isEnabled() for a in actions] == [False] * 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_picked_tag_disables_move_up_and_down_and_says_why(tmp_path: Path) -> None:
    """GH #1: the tag facet hides rows like the text filter, so a swap could
    land on a hidden neighbour; both buttons disable with the tag named as why."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("P1")
        assert _visible_ids(panel) == [2], "fixture assumption: one P1 trigger, away from both ends"
        panel.select_trigger(2)
        for button in (panel.trigger_move_up_button, panel.trigger_move_down_button):
            assert not button.isEnabled()
            assert "tag" in button.toolTip(), button.toolTip()

        panel.set_tag_filter(None)
        panel.select_trigger(2)
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        assert "tag" not in panel.trigger_move_up_button.toolTip()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_the_facets_tag_keeps_the_facet_on_it(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path)
    asked = _answer_dialogs(monkeypatch, text="  Intro ")
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("D1")
        assert _visible_ids(panel) == [0, 2, 3], "D1 chained second on [P1][D1] counts too"
        panel.tag_rename_action.trigger()
        assert asked["getText"] == [('Rename tag "D1"', "3 triggers carry this tag. New tag:", "D1")]
        assert asked["question"] == []
        assert _names(window) == ["[Intro] setup", "[d1]armour split", "[P1][Intro] references", "[Intro]: variable"]
        assert window.edit_history.records[-1].touched == [0, 2, 3]
        assert panel.current_tag() == "Intro"
        assert _visible_ids(panel) == [0, 2, 3]
        assert panel.tag_combo.findData("D1") < 0
        # Undo drops the facet to "All tags": the renamed tag is gone again.
        window.undo()
        assert _names(window) == _TAGGED_NAMES
        assert panel.current_tag() is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_cancelled_rename_calls_nothing(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path)
    _answer_dialogs(monkeypatch, text=None)
    try:
        window.trigger_panel.set_tag_filter("D1")
        window.trigger_panel.request_tag_rename()
        assert window.edit_history.records == []
    finally:
        window.close()


def test_declining_a_merge_calls_nothing(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    calls = []
    window = _tagged_window(tmp_path)
    monkeypatch.setattr(window.trigger_panel, "_on_tag_rename", lambda *args: calls.append(args))
    asked = _answer_dialogs(monkeypatch, text="D1", answer=QMessageBox.No)
    try:
        window.trigger_panel.set_tag_filter("d1")
        window.trigger_panel.request_tag_rename()
        assert asked["question"] == [
            'Merge tag "d1" (1 trigger) into existing tag "D1" (3 triggers)? One undo reverses it.'
        ]
        assert calls == []
        assert window.edit_history.records == []
    finally:
        window.close()


def test_accepting_a_merge_leaves_one_facet_holding_both(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window = _tagged_window(tmp_path)
    _answer_dialogs(monkeypatch, text="D1", answer=QMessageBox.Yes)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("d1")
        panel.request_tag_rename()
        assert panel.tag_combo.findData("d1") < 0
        assert panel.current_tag() == "D1"
        assert _visible_ids(panel) == [0, 1, 2, 3]
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_strips_without_asking_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    """GH #101: no confirmation, since one undo reverses it."""
    window = _tagged_window(tmp_path)
    asked = _answer_dialogs(monkeypatch)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("D1")
        panel.tag_remove_action.trigger()
        assert asked == {"getText": [], "question": []}
        assert _names(window) == ["setup", "[d1]armour split", "[P1] references", ": variable"]
        assert window.edit_history.records[-1].touched == [0, 2, 3]
        assert panel.current_tag() is None
        assert panel.tag_combo.findData("D1") < 0
        assert len(window.edit_history.records) == 1
        window.undo()
        assert _names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


class _Asked(list):
    """_answer_item's record, plus the item each prompt pre-selected."""

    defaults: list


def _answer_item(monkeypatch, text=None) -> _Asked:
    """Stub the panel's QInputDialog.getItem; record (title, label, items, editable)."""
    from PyQt5.QtWidgets import QInputDialog

    asked = _Asked()
    asked.defaults = []

    def get_item(parent, title, label, items, current=0, editable=True):
        asked.append((title, label, list(items), editable))
        asked.defaults.append(list(items)[current] if items else None)
        return (text, True) if text is not None else ("", False)

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(get_item))
    return asked


def test_add_tag_tags_the_selected_triggers_lacking_it_in_one_step(monkeypatch) -> None:
    """GH #101: existing tags are offered, a trigger already carrying the tag
    is left alone, and one undo reverses the lot."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.clearSelection()
        assert not panel.tag_add_action.isEnabled(), "nothing selected"
        _ctrl_select(panel, 0, 2)
        assert panel.tag_add_action.isEnabled()
        asked = _answer_item(monkeypatch, text="  Intro ")
        panel.tag_add_action.trigger()
        assert asked == [("Add tag", "Add a tag to 2 selected triggers:", [], True)]
        assert _names(window) == [
            "[Intro] Fixture: setup", "Fixture: armour split", "[Intro] Fixture: references", "Fixture: variable",
        ]
        assert panel.tag_combo.findData("Intro") >= 0
        assert panel.selected_trigger_indices() == [0, 2], "the selection survives"
        assert len(window.edit_history.records) == 1
        window.undo()
        assert _names(window) == ["Fixture: setup", "Fixture: armour split", "Fixture: references", "Fixture: variable"]
        window.redo()

        _ctrl_select(panel, 0, 1)
        asked = _answer_item(monkeypatch, text="Intro")
        panel.tag_add_action.trigger()
        assert asked[0][2] == ["Intro"], "existing tags are offered"
        assert _names(window)[:2] == ["[Intro] Fixture: setup", "[Intro] Fixture: armour split"]
        assert len(window.edit_history.records) == 2
        window.undo()
        assert _names(window)[:2] == ["[Intro] Fixture: setup", "Fixture: armour split"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_add_tag_chains_onto_another_tag_and_refuses_a_closer(tmp_path: Path, monkeypatch) -> None:
    """Add tag prepends, so adding to "[d1]..." chains in front
    ("[P1][D1] ..." is the corpus's own shape); a tag holding "]" would not
    read back, so it is refused before anything is recorded."""
    from PyQt5.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1)
        _answer_item(monkeypatch, text="a]b")
        panel.tag_add_action.trigger()
        assert len(warnings) == 1 and '"a]b"' in warnings[0]
        assert window.edit_history.records == []

        _answer_item(monkeypatch, text="New")
        panel.tag_add_action.trigger()
        assert _names(window)[1] == "[New] [d1]armour split"
        _answer_item(monkeypatch)  # cancelled
        panel.tag_add_action.trigger()
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_add_tag_skips_a_trigger_carrying_it_anywhere_in_the_chain(tmp_path: Path, monkeypatch) -> None:
    """"[P1][D1] references" already carries D1 as its second tag, so Add tag
    D1 leaves it alone; "[d1]..." (case differs) still gets one, in one step."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2)
        _answer_item(monkeypatch, text="D1")
        panel.tag_add_action.trigger()
        assert _names(window) == _TAGGED_NAMES
        assert window.edit_history.records == []

        _ctrl_select(panel, 1, 2)
        _answer_item(monkeypatch, text="D1")
        panel.tag_add_action.trigger()
        assert _names(window) == ["[D1] setup", "[D1] [d1]armour split", "[P1][D1] references", "[D1]: variable"]
        assert len(window.edit_history.records) == 1
        record = window.edit_history.records[-1]
        assert record.touched == [1] and record.label == 'Add tag "D1" (1 trigger)'
        window.undo()
        assert _names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_on_a_selection_strips_only_the_selected_carriers(tmp_path: Path, monkeypatch) -> None:
    """The facet's Remove tag stays the every-trigger form; this one asks
    which of the selection's tags to drop, and does not ask when there is one."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        assert panel.tag_remove_selected_action.isEnabled()
        asked = _answer_item(monkeypatch, text="D1")
        panel.tag_remove_selected_action.trigger()
        assert asked == [("Remove tag", "Remove a tag from 2 selected triggers:", ["D1", "P1"], False)]
        assert asked.defaults == ["D1"], "the first selected row's leading tag"
        assert _names(window) == ["setup", "[d1]armour split", "[P1] references", "[D1]: variable"]
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[-1].touched == [0, 2]
        window.undo()
        assert _names(window) == _TAGGED_NAMES

        _ctrl_select(panel, 2)
        asked = _answer_item(monkeypatch, text="D1")
        panel.tag_remove_selected_action.trigger()
        assert asked == [("Remove tag", "Remove a tag from 1 selected trigger:", ["D1", "P1"], False)]
        assert asked.defaults == ["P1"], "pre-selects the leading tag, not the sorted-first chained one"
        assert _names(window)[2] == "[P1] references", "the chained second tag, stripped in place"
        window.undo()
        assert _names(window) == _TAGGED_NAMES

        _ctrl_select(panel, 0)
        asked = _answer_item(monkeypatch, text="unused")
        panel.tag_remove_selected_action.trigger()
        assert asked == [], "one tag in the selection: nothing to choose"
        assert _names(window)[0] == "setup"
        window.undo()
        assert _names(window) == _TAGGED_NAMES
        _ctrl_select(panel, 0)
        window.trigger_structural_edit("new", [])
        _ctrl_select(panel, 4)
        assert not panel.tag_remove_selected_action.isEnabled(), "an untagged selection has nothing to remove"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_tag_made_by_renaming_a_trigger_joins_the_dropdown_at_once(tmp_path: Path) -> None:
    """GH #101: no mode switch needed for the facet to offer it."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "[Fresh] armour split")
        assert panel.tag_combo.findData("Fresh") >= 0
        assert panel.tag_combo.findData("d1") < 0, "its old tag had no other carrier"
        panel.set_tag_filter("Fresh")
        assert _visible_ids(panel) == [1]
        window.undo()
        assert panel.tag_combo.findData("Fresh") < 0
        assert panel.tag_combo.findData("d1") >= 0
    finally:
        window.edit_history.mark_saved()
        window.close()


_CHAINED_NAMES = ["[New] setup", "[Old] [New] x", "[Old] solo", "plain"]


def _combo_tags(panel) -> list[str]:
    return [panel.tag_combo.itemData(i) for i in range(1, panel.tag_combo.count())]


def test_a_tag_anywhere_in_the_chain_is_listed_counted_and_filtered(tmp_path: Path) -> None:
    """The TODO's own "[Old] [New] x": New is listed, counted and matched,
    not only the leading Old. New leads nowhere, so only the chain finds it."""
    window = _tagged_window(tmp_path, ["[Old] [New] x", "[Old] solo", "plain", "plain"])
    try:
        panel = window.trigger_panel
        assert _combo_tags(panel) == ["New", "Old"]
        assert panel._tag_count("New") == 1
        assert panel._tag_count("Old") == 2
        panel.set_tag_filter("New")
        assert _visible_ids(panel) == [0]
        panel.set_tag_filter("Old")
        assert _visible_ids(panel) == [0, 1]
        _rename(panel, 3, "--- Sec ---")
        assert panel._grouped, "fixture assumption"
        panel.set_tag_filter("New")
        assert _visible_ids(panel) == [0], "the grouped tree filters the same way"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_strips_a_chained_tag_in_place_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    asked = _answer_dialogs(monkeypatch)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("New")
        panel.tag_remove_action.trigger()
        assert asked == {"getText": [], "question": []}
        assert _names(window) == ["setup", "[Old] x", "[Old] solo", "plain"]
        record = window.edit_history.records[-1]
        assert len(window.edit_history.records) == 1
        assert record.touched == [0, 1] and record.label == 'Remove tag "New" (2 triggers)'
        assert panel.tag_combo.findData("New") < 0
        window.undo()
        assert _names(window) == _CHAINED_NAMES
        assert panel.tag_combo.findData("New") >= 0
        window.redo()
        assert _names(window) == ["setup", "[Old] x", "[Old] solo", "plain"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_tag_retags_a_chained_tag_in_place_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    asked = _answer_dialogs(monkeypatch, text="Newer")
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("New")
        panel.tag_rename_action.trigger()
        assert asked["getText"] == [('Rename tag "New"', "2 triggers carry this tag. New tag:", "New")]
        assert _names(window) == ["[Newer] setup", "[Old] [Newer] x", "[Old] solo", "plain"]
        record = window.edit_history.records[-1]
        assert len(window.edit_history.records) == 1 and record.touched == [0, 1]
        assert panel.current_tag() == "Newer"
        assert _visible_ids(panel) == [0, 1]
        window.undo()
        assert _names(window) == _CHAINED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_closer_in_a_chained_tags_new_name_is_refused(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path, ["[Old] [New] x", "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("New", "a]b")
        assert window.edit_history.records == []
        assert _names(window)[0] == "[Old] [New] x"
    finally:
        window.close()
    assert warnings == ['Tag "New" cannot be renamed to "a]b": it contains "]", which ends a [] tag in "[Old] [New] x".']


@pytest.mark.parametrize("name", ["[Old] (Old) x", "(Old) [Old] x"])
def test_a_mixed_chain_refusal_names_the_bracket_that_fails(tmp_path: Path, monkeypatch, name: str) -> None:
    """The refusal names the segment the new tag breaks, not the first one
    carrying the old tag: "a)b" is fine inside [] and only fails inside ()."""
    from PyQt5.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path, [name, "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("Old", "a)b")
        assert window.edit_history.records == []
        assert _names(window)[0] == name
    finally:
        window.close()
    assert warnings == [f'Tag "Old" cannot be renamed to "a)b": it contains ")", which ends a () tag in "{name}".']


def test_a_chain_that_fails_with_no_single_bad_segment_gets_the_generic_reason(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    from descape import trigger_organize

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(trigger_organize, "retag_in_chain", lambda name, old, new: "untagged")
    window = _tagged_window(tmp_path, ["[Old] x", "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("Old", "New")
        assert window.edit_history.records == []
    finally:
        window.close()
    assert warnings == ['Tag "Old" cannot be renamed to "New": it would not read back as that tag in "[Old] x".']


def test_remove_selected_offers_every_tag_in_the_chain(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1)
        asked = _answer_item(monkeypatch, text="New")
        panel.tag_remove_selected_action.trigger()
        assert asked == [("Remove tag", "Remove a tag from 1 selected trigger:", ["New", "Old"], False)]
        assert _names(window) == ["[New] setup", "[Old] x", "[Old] solo", "plain"]
        assert window.edit_history.records[-1].touched == [1]
        window.undo()
        assert _names(window) == _CHAINED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_tag_keys_on_the_leading_tag_and_matches_it_anywhere(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    try:
        panel = window.trigger_panel
        assert not _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Tag").isEnabled()
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [0, 1]
        _action(panel.trigger_context_menu(panel._item_for_index[1]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [1, 2]
    finally:
        window.close()


def test_a_tag_chained_on_by_a_rename_joins_the_dropdown_at_once(tmp_path: Path) -> None:
    """The leading tag is unchanged, so only a chain comparison notices."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "[d1] [Fresh] armour split")
        assert panel.tag_combo.findData("Fresh") >= 0
        panel.set_tag_filter("Fresh")
        assert _visible_ids(panel) == [1]
        window.undo()
        assert panel.tag_combo.findData("Fresh") < 0
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- tag verbs under a real Ctrl+Z, and the tag chain on a real file -----------


def _press_ctrl_z(window) -> None:
    """A real Ctrl+Z on the focused trigger tree of the activated window, so
    the Edit > Undo keybind fires, not a direct window.undo()."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    # tests/test_text_box_undo_menu.py's activation recipe: shortcuts only reach an active window.
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    window.activateWindow()
    QApplication.setActiveWindow(window)
    tree = window.trigger_panel.tree
    tree.setFocus()
    QApplication.processEvents()
    assert window.isActiveWindow() and QApplication.focusWidget() is tree
    QTest.keyClick(tree, Qt.Key_Z, Qt.ControlModifier)
    QTest.keyRelease(tree, Qt.Key_Control)
    QApplication.processEvents()


def test_the_tags_menu_verbs_each_take_back_with_one_ctrl_z(tmp_path: Path, monkeypatch) -> None:
    """TASK-003.04 (GH #101), through the Tags menu's own actions and a real
    Ctrl+Z key press. Pass conditions:
    - Add Tag to Selection… on three selected triggers, typing a new tag: all
      three gain it, in front of the tag each already carries, as one record;
    - one Ctrl+Z press restores all three names;
    - the same verb with a tag picked from the offered list (the document's
      own tags): each of the three gains it, and one Ctrl+Z restores them;
    - Remove Tag Everywhere on the picked facet tag opens no dialog, strips
      every carrier, and one Ctrl+Z restores them."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        menu = {a.text(): a for a in panel.tags_menu.actions() if not a.isSeparator()}

        _ctrl_select(panel, 1, 2, 3)
        asked = _answer_item(monkeypatch, text="New")
        menu["Add Tag to Selection…"].trigger()
        assert asked[0][1] == "Add a tag to 3 selected triggers:"
        assert _names(window) == [
            "[D1] setup", "[New] [d1]armour split", "[New] [P1][D1] references", "[New] [D1]: variable",
        ]
        assert len(window.edit_history.records) == 1
        _press_ctrl_z(window)
        assert _names(window) == _TAGGED_NAMES, "one Ctrl+Z takes back all three"
        assert window.edit_history.cursor == 0

        _ctrl_select(panel, 0, 1, 3)
        asked = _answer_item(monkeypatch, text="P1")
        menu["Add Tag to Selection…"].trigger()
        assert "P1" in asked[0][2], "the document's own tags are offered to pick"
        assert _names(window) == [
            "[P1] [D1] setup", "[P1] [d1]armour split", "[P1][D1] references", "[P1] [D1]: variable",
        ]
        _press_ctrl_z(window)
        assert _names(window) == _TAGGED_NAMES

        panel.set_tag_filter("D1")
        dialogs = _answer_dialogs(monkeypatch)
        menu["Remove Tag Everywhere"].trigger()
        assert dialogs == {"getText": [], "question": []}, "no question asked"
        assert _names(window) == ["setup", "[d1]armour split", "[P1] references", ": variable"]
        _press_ctrl_z(window)
        assert _names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


_EL_CID_4 = Path(__file__).resolve().parent.parent / "examples" / "C2_ElCid_coop_4_v0_13.aoe2scenario"


@pytest.mark.corpus
def test_the_tags_menu_reads_the_whole_tag_chain_on_el_cid_4(monkeypatch) -> None:
    """TASK-003.14 on the real file, in the body's order. Pass conditions:
    1. the tag dropdown lists both Obj and D1; picking D1 shows exactly the D1
       carriers, the 8 "[Obj] [D1] ..." rows among them;
    2. with D1 picked, Tags > Rename Tag… to X: those 8 read "[Obj] [X] ...",
       the leading-D1 rows read "[X] ...", nothing else changes, and one
       Ctrl+Z restores every name;
    3. D1 picked again, Tags > Remove Tag Everywhere asks nothing: the 8 read
       "[Obj] ...", no trigger carries D1, and one Ctrl+Z restores every name;
    4. one "[Obj] [D1] ..." trigger selected, Tags > Remove Tag from
       Selection… offers D1 and Obj, with Obj pre-selected."""
    from descape import trigger_organize

    if not _EL_CID_4.is_file():
        pytest.skip(f"{_EL_CID_4.name} is not in examples/")
    window = _window()
    try:
        window.load_scenario(_EL_CID_4)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        original = _names(window)
        chained = [i for i, n in enumerate(original) if n.startswith("[Obj] [D1] ")]
        carriers = [i for i, n in enumerate(original) if "D1" in trigger_organize.tag_chain(n)]
        assert len(chained) == 8, "corpus assumption: the body's 8 [Obj] [D1] rows"
        assert set(chained) < set(carriers)

        # 1.
        assert {"Obj", "D1"} <= set(_combo_tags(panel))
        panel.set_tag_filter("D1")
        assert panel.current_tag() == "D1"
        assert _visible_ids(panel) == carriers

        # 2.
        _answer_dialogs(monkeypatch, text="X")
        panel.tag_rename_action.trigger()
        renamed = _names(window)
        assert [renamed[i] for i in chained] == ["[Obj] [X] " + original[i][len("[Obj] [D1] "):] for i in chained]
        for i, name in enumerate(original):
            if i in carriers and i not in chained:
                assert renamed[i] == "[X]" + name[len("[D1]"):], (i, name, renamed[i])
            elif i not in carriers:
                assert renamed[i] == name, (i, name, renamed[i])
        assert len(window.edit_history.records) == 1
        _press_ctrl_z(window)
        assert _names(window) == original

        # 3.
        panel.set_tag_filter("D1")
        dialogs = _answer_dialogs(monkeypatch)
        panel.tag_remove_action.trigger()
        assert dialogs == {"getText": [], "question": []}, "no question asked"
        stripped = _names(window)
        assert [stripped[i] for i in chained] == ["[Obj] " + original[i][len("[Obj] [D1] "):] for i in chained]
        assert not any("D1" in trigger_organize.tag_chain(n) for n in stripped)
        assert window.edit_history.cursor == 1, "one record (the undone rename's redo was dropped)"
        _press_ctrl_z(window)
        assert _names(window) == original

        # 4.
        _ctrl_select(panel, chained[0])
        asked = _answer_item(monkeypatch)  # cancelled: only the offer is under test
        panel.tag_remove_selected_action.trigger()
        assert asked == [("Remove tag", "Remove a tag from 1 selected trigger:", ["D1", "Obj"], False)]
        assert asked.defaults == ["Obj"]
        assert _names(window) == original
    finally:
        window.edit_history.mark_saved()
        window.close()
