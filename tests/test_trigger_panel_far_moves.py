"""Trigger panel: five-place, edge and keyboard moves of triggers and entries
(GH #211).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _LONG_COUNT,
    _action,
    _activate,
    _active_triggers_window,
    _ctrl_select,
    _entry_menu_move_states,
    _entry_move_states,
    _entry_rows,
    _focus_on,
    _last_label,
    _long_window,
    _modifier_click,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _open,
    _order,
    _ordered_window,
    _positions,
    _record_count,
    _release_modifiers,
    _row_in_view,
    _save_long_fixture,
    _saved_body,
    _select_entries,
    _status,
    _tagged_window,
    _tree_order,
    _triggers_window,
    _types_of,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #211: five places, and the top and bottom of the section ----------------
#
# The shipped fixtures hold 4 and 8 triggers: too few to move five places with
# room to spare, or to overflow the tree.


def _with_block_at(indices: list[int], slot: int) -> list[int]:
    """The long fixture's identity order with `indices` taken out and put
    back, as one block, at `slot` of what is left."""
    rest = [index for index in range(_LONG_COUNT) if index not in indices]
    return rest[:slot] + list(indices) + rest[slot:]


def test_gh211_shift_click_moves_a_trigger_five_places_in_one_record(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        panel.select_trigger(20)
        QApplication.processEvents()
        assert panel.tree.verticalScrollBar().value() > 0, (
            "fixture assumption: the triggers overflow the tree, and this row sits below the fold"
        )

        _modifier_click(panel.trigger_move_up_button, Qt.ShiftModifier)

        assert _order(window) == _with_block_at([20], 15)
        assert _record_count(window) == 1
        assert _last_label(window) == "Move trigger 20 up 5 places"
        assert panel.selected_trigger_indices() == [20] and panel.current_trigger_index() == 20
        assert _row_in_view(panel.tree, panel._item_for_index[20])
        window.undo()
        assert _order(window) == list(range(_LONG_COUNT)), "one undo takes the five places back"
        assert not window.edit_history.is_dirty

        # Near the end the distance is the room that is left, and then the button is grey.
        panel.select_trigger(26)
        _modifier_click(panel.trigger_move_down_button, Qt.ShiftModifier)
        assert _order(window) == _with_block_at([26], 29)
        assert _last_label(window) == "Move trigger 26 down 3 places"
        assert not panel.trigger_move_down_button.isEnabled()
        records = _record_count(window)
        _modifier_click(panel.trigger_move_down_button, Qt.ShiftModifier)
        assert _record_count(window) == records

        # click() is one place even though this button's last mouse click held Shift,
        # and so is a mouse click with no modifier on its own event.
        panel.select_trigger(10)
        assert panel.trigger_move_up_button.click_modifiers == Qt.NoModifier
        panel.trigger_move_up_button.click()
        assert panel._display_position[10] == 9
        _modifier_click(panel.trigger_move_up_button, Qt.NoModifier)
        assert panel._display_position[10] == 8
        assert _last_label(window) == "Move trigger 10 up"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_split_selection_keeps_its_spacing_on_a_five_place_move(tmp_path: Path) -> None:
    """The distance is capped by the leading member's room, exactly the ▲▼
    gate, so the block never closes up against the end."""
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 4)
        _modifier_click(panel.trigger_move_up_button, Qt.ShiftModifier)
        assert _order(window)[:6] == [2, 0, 4, 1, 3, 5]
        assert _last_label(window) == "Move 2 triggers up 2 places"
        assert panel.selected_trigger_indices() == [2, 4]
        assert panel._trigger_move_reasons()[0] == " (already first)"
        # Still on for the Ctrl+click gather (TASK-356).
        assert panel.trigger_move_up_button.isEnabled()

        _modifier_click(panel.trigger_move_down_button, Qt.ShiftModifier)
        assert _order(window)[:9] == [0, 1, 3, 5, 6, 2, 7, 4, 8]
        assert _last_label(window) == "Move 2 triggers down 5 places"
        assert _record_count(window) == 2
        window.undo()
        assert _order(window)[:6] == [2, 0, 4, 1, 3, 5]
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_block_moved_five_down_ends_with_its_last_row_in_view(tmp_path: Path) -> None:
    """The current row is the block's first, so the tree's own scroll follows
    that one; going down it is the last row that leads."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 7)
        QApplication.processEvents()
        assert not _row_in_view(panel.tree, panel._item_for_index[12]), "fixture assumption: slot 12 is below the fold"

        _modifier_click(panel.trigger_move_down_button, Qt.ShiftModifier)

        assert (panel._display_position[1], panel._display_position[7]) == (6, 12)
        assert panel.selected_trigger_indices() == [1, 7]
        assert _row_in_view(panel.tree, panel._item_for_index[7])
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_ctrl_click_goes_to_the_edge_of_the_triggers_own_section(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path, dividers=(10, 20))
    try:
        panel = window.trigger_panel
        assert [s.header_index for s in panel._sections] == [None, 10, 20], "fixture assumption"
        panel.select_trigger(15)
        _modifier_click(panel.trigger_move_up_button, Qt.ControlModifier)
        assert _order(window) == _with_block_at([15], 11), "just below its own divider"
        assert _last_label(window) == "Move trigger 15 to the top of the section"
        assert panel.selected_trigger_indices() == [15]
        assert _row_in_view(panel.tree, panel._item_for_index[15])
        assert _record_count(window) == 1

        # Ctrl wins over Shift: the edge, not five places.
        _modifier_click(panel.trigger_move_down_button, Qt.ControlModifier | Qt.ShiftModifier)
        assert _order(window) == _with_block_at([15], 19), "the next row is the following divider"
        assert _last_label(window) == "Move trigger 15 to the bottom of the section"
        window.undo()
        assert _order(window) == _with_block_at([15], 11)
        window.undo()

        # A split selection gathers into one block.
        _ctrl_select(panel, 12, 17)
        _modifier_click(panel.trigger_move_up_button, Qt.ControlModifier)
        assert _order(window) == _with_block_at([12, 17], 11)
        assert _last_label(window) == "Move 2 triggers to the top of the section"
        assert panel.selected_trigger_indices() == [12, 17]
        window.undo()

        # The leading run's bottom stops before the first divider.
        panel.select_trigger(3)
        _modifier_click(panel.trigger_move_down_button, Qt.ControlModifier)
        assert _order(window) == _with_block_at([3], 9)
        assert panel._sections[0].member_indices == (0, 1, 2, 4, 5, 6, 7, 8, 9, 3)
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_refused_ctrl_click_records_nothing_and_says_why(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path, dividers=(10, 20))
    try:
        panel = window.trigger_panel
        up, down = panel.trigger_move_up_button, panel.trigger_move_down_button
        for selection, button, message in (
            ([10], up, "Cannot move to the top: a section header moves with Section Up/Down"),
            ([11], up, "Cannot move to the top: already there"),
            ([19], down, "Cannot move to the bottom: already there"),
            ([9, 11], down, "Cannot move to the bottom: the selection spans sections"),
        ):
            _ctrl_select(panel, *selection)
            assert button.isEnabled(), "one place that way is still possible"
            assert message not in _status(window)
            _modifier_click(button, Qt.ControlModifier)
            assert _status(window).splitlines().count(message) == 1
            assert _record_count(window) == 0, message
            assert _order(window) == list(range(_LONG_COUNT)), message
        assert not window.edit_history.is_dirty
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_task356_a_split_block_against_the_top_still_offers_ctrl_click(tmp_path: Path) -> None:
    """One place up is refused, gathering at the top is not, so ▲ stays on for
    the Ctrl+click its tooltip names; a plain click there still moves nothing."""
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        up, down = panel.trigger_move_up_button, panel.trigger_move_down_button
        _ctrl_select(panel, 0, 2)
        assert panel._trigger_move_reasons()[0] == " (already first)"
        assert panel._edge_move_reasons()[0] == ""
        assert up.isEnabled()

        _modifier_click(up, Qt.NoModifier)
        assert _record_count(window) == 0 and _order(window) == list(range(_LONG_COUNT))

        _modifier_click(up, Qt.ControlModifier)
        assert _order(window) == _with_block_at([0, 2], 0)
        assert _record_count(window) == 1
        # Gathered: both ways up are now refused, so ▲ goes grey.
        assert not up.isEnabled() and down.isEnabled()

        # The mirror case at the bottom.
        window.undo()
        last = _LONG_COUNT - 1
        _ctrl_select(panel, last - 2, last)
        assert down.isEnabled()
        _modifier_click(down, Qt.NoModifier)
        assert _record_count(window) == 1, "only the undone record"
        _modifier_click(down, Qt.ControlModifier)
        assert _order(window) == _with_block_at([last - 2, last], _LONG_COUNT - 2)
        assert not down.isEnabled()
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_task365_a_button_on_only_for_the_gather_says_so(tmp_path: Path) -> None:
    """Enabled for the Ctrl+click gather alone, ▲/▼ names it; once nothing is
    left to gather the button greys out with the plain refusal note."""
    from PyQt5.QtCore import Qt

    gather = ": Ctrl+click gathers the selection here)"
    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        up, down = panel.trigger_move_up_button, panel.trigger_move_down_button
        _ctrl_select(panel, 0, 2)
        assert up.isEnabled()
        assert up.toolTip() == panel._MOVE_UP_TIP + " (already first" + gather
        assert down.toolTip() == panel._MOVE_DOWN_TIP

        _modifier_click(up, Qt.ControlModifier)
        assert not up.isEnabled()
        assert up.toolTip() == panel._MOVE_UP_TIP + " (already first)"

        last = _LONG_COUNT - 1
        _ctrl_select(panel, last - 2, last)
        assert down.isEnabled()
        assert down.toolTip() == panel._MOVE_DOWN_TIP + " (already last" + gather
        _modifier_click(down, Qt.ControlModifier)
        assert not down.isEnabled()
        assert down.toolTip() == panel._MOVE_DOWN_TIP + " (already last)"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_move_with_no_room_records_nothing(tmp_path: Path) -> None:
    """No button allows it, but the funnel is public: a commit always pushes,
    so a block already against the end must not open an edit."""
    window = _long_window(tmp_path)
    try:
        window.trigger_structural_edit("move_up", [0])
        window.trigger_structural_edit("move_up", [0, 2], 5)
        window.trigger_structural_edit("move_down", [_LONG_COUNT - 1], 5)
        window.trigger_structural_edit("move_top", [0])
        window.trigger_structural_edit("move_bottom", [])
        assert _record_count(window) == 0
        assert _order(window) == list(range(_LONG_COUNT))
        assert not window.edit_history.is_dirty and not window.windowTitle().startswith("*")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_flat_five_place_move_patches_the_tree_without_a_repopulate(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path)
    try:
        panel = window.trigger_panel
        before_items = [panel.tree.topLevelItem(i) for i in range(_LONG_COUNT)]
        panel.select_trigger(0)
        assert panel._edge_move_reasons() == (" (already there)", "")

        _modifier_click(panel.trigger_move_down_button, Qt.ShiftModifier)

        after_items = [panel.tree.topLevelItem(i) for i in range(_LONG_COUNT)]
        assert [id(item) for item in after_items] == [
            id(item) for item in (*before_items[1:6], before_items[0], *before_items[6:])
        ], "a full repopulate would have replaced every QTreeWidgetItem"
        assert _tree_order(panel) == _order(window) == _with_block_at([0], 5)
        assert panel._display_slots == _order(window)
        assert _positions(panel) == [str(p) for p in range(_LONG_COUNT)], "every row in the span is relabelled"
        assert panel.selected_trigger_indices() == [0]

        # The cached partition still lists trigger 0 first; the edge gate reads the order.
        assert panel._edge_move_reasons() == ("", "")
        top = _action(panel.trigger_context_menu(panel._item_for_index[0]), "Move to Top")
        assert top.isEnabled()
        top.trigger()
        assert _order(window) == list(range(_LONG_COUNT))
        assert _last_label(window) == "Move trigger 0 to the top"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_grouped_five_place_move_across_a_divider_lands_expanded_and_in_view(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _long_window(tmp_path, dividers=(10, 20))
    try:
        panel = window.trigger_panel
        panel._item_for_index[10].setExpanded(False)
        assert ("--- S10 ---", 0) in panel._collapsed_keys, "fixture assumption: the section above is collapsed"
        panel.select_trigger(22)
        QApplication.processEvents()

        _modifier_click(panel.trigger_move_up_button, Qt.ShiftModifier)

        assert _order(window) == _with_block_at([22], 17)
        item = panel._item_for_index[22]
        assert item.parent() is panel._item_for_index[10], "it crossed the divider into the section above"
        assert item.parent().isExpanded()
        assert ("--- S10 ---", 0) not in panel._collapsed_keys
        assert panel.selected_trigger_indices() == [22] and panel.current_trigger_index() == 22
        assert _row_in_view(panel.tree, item)
        assert _last_label(window) == "Move trigger 22 up 5 places"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_a_saved_five_place_and_edge_move_change_only_the_display_order_bytes(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    base = _save_long_fixture(tmp_path, dividers=(10, 20))
    window = _open(base)
    try:
        names = _names(window)
        window.trigger_structural_edit("move_down", [12], 5)
        window.trigger_structural_edit("move_top", [25])
        moved = _order(window)
        assert moved[11:19] == [11, 13, 14, 15, 16, 17, 12, 18] and moved[20:23] == [20, 25, 21]
        edited = tmp_path / "moved.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
        regions = window.trigger_edits.regions
    finally:
        window.edit_history.mark_saved()
        window.close()
    before, after = _saved_body(base), _saved_body(edited)
    assert len(before) == len(after)
    diffs = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert diffs, "the moves changed no byte"
    units_end = load_map_and_units(base).units_section_end
    start, end = units_end + regions.triggers_end, units_end + regions.display_order_end
    assert all(start <= i < end for i in diffs), "a byte outside the display-order array changed"
    reloaded = _open(edited)
    try:
        assert _order(reloaded) == moved
        assert _names(reloaded) == names
    finally:
        reloaded.close()


def _move_menu_states(panel, index: int) -> dict[str, tuple[bool, str]]:
    menu = panel.trigger_context_menu(panel._item_for_index[index])
    return {a.text(): (a.isEnabled(), a.toolTip()) for a in menu.actions() if a.text().startswith("Move")}


def test_gh211_the_trigger_menu_moves_follow_their_gates(tmp_path: Path) -> None:
    window = _long_window(tmp_path, dividers=(10, 20), names={15: "[x] tagged"})
    try:
        panel = window.trigger_panel
        # A menu entry ignores modifiers, so its tip names no Shift+ or Ctrl+click (TASK-356).
        tips = (panel._MOVE_UP_MENU_TIP, panel._MOVE_DOWN_MENU_TIP, panel._MOVE_TOP_TIP, panel._MOVE_BOTTOM_TIP)
        assert not [tip for tip in tips if "click" in tip]
        texts = ("Move Up", "Move Down", "Move to Top of Section", "Move to Bottom of Section")

        def states(*selection) -> list[tuple[bool, str]]:
            _ctrl_select(panel, *selection)
            found = _move_menu_states(panel, selection[0])
            assert tuple(found) == texts
            # Each entry is exactly its gate: enabled with no reason, and the reason on its tooltip.
            reasons = (*panel._trigger_move_reasons(), *panel._edge_move_reasons())
            assert list(found.values()) == [(not r, tip + r) for tip, r in zip(tips, reasons, strict=True)]
            return [(enabled, tip[len(base):]) for (enabled, tip), base in zip(found.values(), tips, strict=True)]

        on = (True, "")
        assert states(15) == [on, on, on, on]
        assert states(11) == [on, on, (False, " (already there)"), on]
        assert states(19) == [on, on, on, (False, " (already there)")]
        header = (False, " (a section header moves with Section Up/Down)")
        assert states(10) == [on, on, header, header]
        spans = (False, " (the selection spans sections)")
        assert states(9, 11) == [on, on, spans, spans]
        assert states(0) == [(False, " (already first)"), on, (False, " (already there)"), on]
        assert states(29) == [on, (False, " (already last)"), on, (False, " (already there)")]
        # A split block against the top: one place up is off, gathering at the top is not.
        assert states(0, 2) == [(False, " (already first)"), on, on, on]

        # The menu's verbs are the buttons' own.
        _ctrl_select(panel, 15)
        menu = panel.trigger_context_menu(panel._item_for_index[15])
        _action(menu, "Move to Bottom of Section").trigger()
        assert _order(window) == _with_block_at([15], 19)
        _action(panel.trigger_context_menu(panel._item_for_index[15]), "Move Up").trigger()
        assert _order(window) == _with_block_at([15], 18)
        assert _record_count(window) == 2
        window.undo()
        window.undo()

        # File order view offers all four as ID moves (TASK-569); under a filter and a tag nothing moves.
        panel.sort_combo.setCurrentIndex(1)
        _ctrl_select(panel, 15)
        found = _move_menu_states(panel, 15)
        assert tuple(found) == ("Move Up", "Move Down", "Move to Top", "Move to Bottom"), "File order has no sections"
        assert [enabled for enabled, _ in found.values()] == [True] * 4
        assert found["Move Up"][1] == panel._ID_MOVE_UP_MENU_TIP
        panel.sort_combo.setCurrentIndex(0)

        off = (False, " (clear the filter or tag to reorder)")
        panel.filter_edit.setText("t")
        assert states(15) == [off] * 4
        panel.filter_edit.clear()
        panel.set_tag_filter("x")
        assert states(15) == [off] * 4
        panel.set_tag_filter(None)
        assert states(15) == [on, on, on, on]

        # With the type picker up no trigger verb is offered, whatever the gates say.
        panel._request_entry_op("new")
        assert panel.picker_showing()
        assert [enabled for enabled, _ in _move_menu_states(panel, 15).values()] == [False] * 4
        panel._close_picker()
        assert not window.edit_history.is_dirty and _order(window) == list(range(_LONG_COUNT))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh211_the_entry_menu_offers_top_and_bottom_under_the_move_gate(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel

        def edges(*refs) -> list[tuple[bool, str]]:
            _select_entries(panel, *refs)
            menu = panel.entry_context_menu(panel._entry_item_for(*refs[0]))
            top, bottom = _action(menu, "Move to Top"), _action(menu, "Move to Bottom")
            # The same gate as Move Up and Move Down, entry for entry.
            assert (top.isEnabled(), bottom.isEnabled()) == _entry_menu_move_states(panel, refs[0])
            assert top.toolTip().startswith(panel._ENTRY_MOVE_TOP_TIP)
            assert bottom.toolTip().startswith(panel._ENTRY_MOVE_BOTTOM_TIP)
            return [
                (top.isEnabled(), top.toolTip()[len(panel._ENTRY_MOVE_TOP_TIP):]),
                (bottom.isEnabled(), bottom.toolTip()[len(panel._ENTRY_MOVE_BOTTOM_TIP):]),
            ]

        # Conditions show as 2, 0, 3, 1 and effects as 4, 0, 6, 2, 5, 1, 3.
        assert edges(("condition", 2)) == [(False, " (already first)"), (True, "")]
        assert edges(("condition", 1)) == [(True, ""), (False, " (already last)")]
        assert edges(("condition", 0)) == [(True, ""), (True, "")]
        assert edges(("condition", 2), ("condition", 0)) == [(False, " (already first)"), (True, "")]
        assert edges(("condition", 2), ("condition", 3)) == [(True, ""), (True, "")], "a split block gathers"

        _select_entries(panel, ("condition", 3), ("effect", 2))
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 2)), "Move to Bottom").trigger()
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([2, 0, 1, 3], [4, 0, 6, 5, 1, 3, 2])
        assert _last_label(window) == "Move 2 entries to the bottom"
        assert _record_count(window) == 1
        assert panel.selected_entry_refs() == [("condition", 3), ("effect", 2)]
        assert edges(("condition", 3), ("effect", 2)) == [(True, ""), (False, " (already last)")]
        window.undo()
        assert not window.edit_history.is_dirty

        # The type picker takes the lower pane, and the entry verbs with it.
        _select_entries(panel, ("condition", 0))
        item = panel._entry_item_for("condition", 0)
        panel._request_entry_op("new")
        menu = panel.entry_context_menu(item)
        for text in ("Move to Top", "Move to Bottom"):
            assert not _action(menu, text).isEnabled()
            assert "close the type picker" in _action(menu, text).toolTip()
        panel._request_entry_op("move_top")
        assert not window.edit_history.is_dirty, "a gated panel must not reach the model"
        panel._close_picker()
        assert _types_of(window, ENTRY_ORDER_TRIGGER, "condition") == [10, 5, 6, 6], "the lists never move"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh211_modifier_clicks_move_each_kind_of_entry_inside_its_own_list(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        # Conditions show as 2, 0, 3, 1 and effects as 4, 0, 6, 2, 5, 1, 3.
        _select_entries(panel, ("effect", 0), ("condition", 2))

        _modifier_click(panel.entry_move_down_button, Qt.ShiftModifier)
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([0, 3, 1, 2], [4, 6, 2, 5, 1, 3, 0])
        assert _last_label(window) == "Move 2 entries down to 5 places"
        assert _record_count(window) == 1
        assert panel.selected_entry_refs() == [("condition", 2), ("effect", 0)]
        assert _entry_move_states(panel) == (True, False)

        _modifier_click(panel.entry_move_up_button, Qt.ControlModifier)
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([2, 0, 3, 1], [0, 4, 6, 2, 5, 1, 3])
        assert _last_label(window) == "Move 2 entries to the top"
        assert _record_count(window) == 2

        # Ctrl wins over Shift, and a click with no modifier is still one place.
        _modifier_click(panel.entry_move_down_button, Qt.ControlModifier | Qt.ShiftModifier)
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([0, 3, 1, 2], [4, 6, 2, 5, 1, 3, 0])
        panel.entry_move_up_button.click()
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([0, 3, 2, 1], [4, 6, 2, 5, 1, 0, 3])
        assert _last_label(window) == "Move 2 entries up"

        for _ in range(4):
            window.undo()
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == ([2, 0, 3, 1], [4, 0, 6, 2, 5, 1, 3])
        assert not window.edit_history.is_dirty, "one record per click"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_far_entry_moves_keep_the_scroll_and_the_moved_row_in_view() -> None:
    """Each move rebuilds the entry tree, which leaves it scrolled to the top."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape import brush

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        panel.select_entry("effect", 2)  # the fixture's Create Object
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(brush.BRUSH_SHAPE_SQUARE))
        window.stamp_create_objects(50, 50)
        QApplication.processEvents()
        bar = panel.entry_tree.verticalScrollBar()
        panel.select_entry("effect", 20)
        QApplication.processEvents()
        start = bar.value()
        assert start > 0, "fixture assumption: the entries overflow the tree, and this row sits below the fold"
        slot = _entry_rows(panel, "effect").index(20)
        assert slot + 5 < len(_entry_rows(panel, "effect")) - 1, "fixture assumption: room for five places and more"
        records = _record_count(window)

        def in_view() -> bool:
            return _row_in_view(panel.entry_tree, panel._entry_item_for("effect", 20))

        _modifier_click(panel.entry_move_down_button, Qt.ShiftModifier)
        assert _entry_rows(panel, "effect").index(20) == slot + 5
        assert _last_label(window) == "Move effect down to 5 places"
        assert bar.value() >= start, f"the list jumped up to {bar.value()}"
        assert in_view()

        _modifier_click(panel.entry_move_up_button, Qt.ControlModifier)
        assert _entry_rows(panel, "effect")[0] == 20
        assert _last_label(window) == "Move effect to the top"
        assert in_view()

        _modifier_click(panel.entry_move_down_button, Qt.ControlModifier)
        assert _entry_rows(panel, "effect")[-1] == 20
        assert _last_label(window) == "Move effect to the bottom"
        assert in_view()
        assert panel.selected_entry_refs() == [("effect", 20)]
        assert _record_count(window) == records + 3

        window.undo()
        assert _entry_rows(panel, "effect")[0] == 20
        window.undo()
        assert _entry_rows(panel, "effect").index(20) == slot + 5
        window.undo()
        assert _entry_rows(panel, "effect").index(20) == slot
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def _press(widget, key, modifiers) -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QTest.keyClick(widget, key, modifiers)
    QApplication.processEvents()


def test_gh211_alt_up_moves_the_trigger_or_the_entry_under_the_focus() -> None:
    from PyQt5.QtCore import Qt

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.tree)
        _press(panel.tree, Qt.Key_Up, Qt.AltModifier)
        assert _order(window) == [0, 1, 3, 2]
        assert _last_label(window) == "Move trigger 3 up" and _record_count(window) == 1
        assert panel.current_trigger_index() == 3, "the key moved the trigger, not the tree's cursor"
        assert panel.tree.hasFocus()

        # The entry tree has the focus: the same key moves the effect and leaves the triggers alone.
        _select_entries(panel, ("effect", 2))
        _focus_on(panel.entry_tree)
        _press(panel.entry_tree, Qt.Key_Up, Qt.AltModifier)
        assert _entry_rows(panel, "effect")[:3] == [0, 2, 1]
        assert _last_label(window) == "Move effect up" and _record_count(window) == 2
        assert _order(window) == [0, 1, 3, 2]
        assert panel.selected_entry_refs() == [("effect", 2)]
        _press(panel.entry_tree, Qt.Key_End, Qt.AltModifier)
        assert _entry_rows(panel, "effect")[-1] == 2
        assert _last_label(window) == "Move effect to the bottom"
        _press(panel.entry_tree, Qt.Key_Up, Qt.AltModifier | Qt.ShiftModifier)
        assert _entry_rows(panel, "effect").index(2) == 1
        assert _last_label(window) == "Move effect up to 5 places"
        assert _order(window) == [0, 1, 3, 2]
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_the_far_and_edge_keys_move_a_trigger_five_places_and_to_the_section_edge(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path, dividers=(10, 20))
    try:
        _activate(window)
        panel = window.trigger_panel
        panel.select_trigger(15)
        _focus_on(panel.tree)
        _press(panel.tree, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        assert _order(window) == _with_block_at([15], 20), "five places, across the divider"
        assert _last_label(window) == "Move trigger 15 down 5 places"
        _press(panel.tree, Qt.Key_Home, Qt.AltModifier)
        assert _order(window) == _with_block_at([15], 20), "it is the first trigger of its new section already"
        assert "Cannot move to the top: already there" in _status(window).splitlines()
        _press(panel.tree, Qt.Key_End, Qt.AltModifier)
        assert _order(window) == _with_block_at([15], 29)
        assert _last_label(window) == "Move trigger 15 to the bottom of the section"
        _press(panel.tree, Qt.Key_Home, Qt.AltModifier)
        assert _order(window) == _with_block_at([15], 20)
        _press(panel.tree, Qt.Key_Down, Qt.AltModifier)
        assert _order(window) == _with_block_at([15], 21)
        assert _record_count(window) == 4
        assert panel.selected_trigger_indices() == [15] and panel.tree.hasFocus()
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_gh211_alt_up_in_a_form_dropdown_moves_nothing() -> None:
    """The shortcuts belong to the two trees. Anywhere else the key is the
    focused widget's own, as Alt+Down is to a dropdown."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QComboBox

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 2))
        combos = [c for c in panel.property_host.findChildren(QComboBox) if c.isEnabled() and c.isVisible()]
        assert combos, "fixture assumption: this effect's form has a dropdown"
        _focus_on(combos[0])
        rows = _entry_rows(panel, "effect")
        records = _record_count(window)

        # One press: the dropdown takes the key as its own, which may rebuild the form.
        _press(combos[0], Qt.Key_Up, Qt.AltModifier)

        assert _order(window) == [0, 1, 2, 3]
        assert _entry_rows(panel, "effect") == rows
        assert not any(r.label.startswith("Move") for r in window.edit_history.records[records:])
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("view", ["filter", "tag", "type picker"])
def test_gh211_alt_up_is_gated_like_the_move_buttons(tmp_path: Path, view: str) -> None:
    """A shortcut has no disabled button in front of it, so the request path
    itself refuses: nothing is recorded and the tree keeps its order."""
    from PyQt5.QtCore import Qt

    window = _tagged_window(tmp_path)
    try:
        _activate(window)
        panel = window.trigger_panel
        panel.select_trigger(2)
        if view == "filter":
            panel.filter_edit.setText("e")
        elif view == "tag":
            panel.set_tag_filter("D1")
        else:
            panel._request_entry_op("new")
            assert panel.picker_showing()
        panel.select_trigger(2)
        _focus_on(panel.tree)
        assert not panel.trigger_move_up_button.isEnabled(), "fixture assumption: the button is off here"
        shown = _tree_order(panel)

        for key, modifiers in (
            (Qt.Key_Up, Qt.AltModifier),
            (Qt.Key_Down, Qt.AltModifier),
            (Qt.Key_Up, Qt.AltModifier | Qt.ShiftModifier),
            (Qt.Key_Home, Qt.AltModifier),
            (Qt.Key_End, Qt.AltModifier),
        ):
            _press(panel.tree, key, modifiers)

        assert _record_count(window) == 0
        assert _order(window) == [0, 1, 2, 3]
        assert _tree_order(panel) == shown
        assert not window.edit_history.is_dirty
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_task569_the_move_trigger_keys_move_ids_in_file_order(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _tagged_window(tmp_path)
    try:
        _activate(window)
        panel = window.trigger_panel
        panel.sort_combo.setCurrentIndex(1)
        names = _names(window)
        panel.select_trigger(2)
        _focus_on(panel.tree)
        _press(panel.tree, Qt.Key_Up, Qt.AltModifier)
        assert _names(window) == [names[i] for i in (0, 2, 1, 3)]
        _press(panel.tree, Qt.Key_End, Qt.AltModifier)
        assert _names(window) == [names[i] for i in (0, 1, 3, 2)]
        assert _order(window) == [_names(window).index(name) for name in names], "display slots kept"
        assert _record_count(window) == 2
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()
