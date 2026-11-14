"""Trigger panel: the add picker, structural buttons, display and execution
order, Move Up/Down, the position column, renumbering and ID moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _LONG_COUNT,
    TRIGGER_FIXTURE,
    _action,
    _ctrl_select,
    _display_names,
    _divided_window,
    _gh2_window,
    _group,
    _header,
    _last_label,
    _long_window,
    _modifier_click,
    _names,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _order,
    _pick,
    _positions,
    _record_count,
    _release_modifiers,
    _rename,
    _row_in_view,
    _row_widget,
    _sectioned_window,
    _select_first_condition,
    _status,
    _tree_order,
    _triggers_window,
    _window,
)

from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- 4b.6b: structural editing and the vocabulary picker ---------------------


def test_opening_and_cancelling_the_picker_records_nothing(tmp_path: Path) -> None:
    """6b's own "browsing is not editing". commit_trigger_edit() pushes
    unconditionally, so the window must be told about an accepted pick and
    nothing else -- not about the button click that opened the picker.
    """
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))

        panel._request_entry_op("new")
        assert panel.detail_stack.currentIndex() == 1, "the picker page did not come up"
        # Browsing the picker is not editing either: select a row, filter, and
        # only then cancel.
        _pick(panel, "effect", "send chat")
        panel.picker_filter.setText("chat")
        panel._close_picker()

        assert panel.detail_stack.currentIndex() == 0
        assert window.trigger_edits is None, "opening the picker must not build an edit model"
        assert not window.edit_history.is_dirty
        assert not window.windowTitle().startswith("*")

        out = tmp_path / "cancelled.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_offers_both_kinds_and_filters_across_them() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel._request_entry_op("new")

        conditions = panel.picker_tree.topLevelItem(0)
        effects = panel.picker_tree.topLevelItem(1)
        assert "Conditions" in conditions.text(0) and "Effects" in effects.text(0)
        # A floor rather than an exact count, which a library bump would break
        # for no real reason: docs/INGAME_EDITOR_REFERENCE.md's traversal found
        # 40 conditions and 100 effects, and newer versions only add.
        assert conditions.childCount() >= 40
        assert effects.childCount() >= 100

        # But type 0, the library's "none" placeholder, is not offered: the
        # in-game editor lists neither, and alphabetically it would sit at the
        # top of both groups.
        names = [
            group.child(c).text(0)
            for group in (conditions, effects)
            for c in range(group.childCount())
        ]
        assert "none" not in names
        assert len(names) == conditions.childCount() + effects.childCount()
        assert f"({conditions.childCount()})" in conditions.text(0), (
            "the heading count must match what the group actually offers"
        )

        panel.picker_filter.setText("timer")
        visible = [
            group.child(c).text(0)
            for group in (conditions, effects)
            for c in range(group.childCount())
            if not group.child(c).isHidden()
        ]
        assert visible, "a filter matching real types hid everything"
        assert all("timer" in name for name in visible)

        # A group left with nothing under it hides too, rather than sitting
        # there as an empty heading.
        panel.picker_filter.setText("zzz-no-such-type-zzz")
        assert conditions.isHidden() and effects.isHidden()
        assert panel.picker_add_button.isEnabled() is False
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_collapses_the_other_kind_when_one_group_is_highlighted() -> None:
    """GH #136: New with the Effects (or Conditions) heading, or one of its
    rows, highlighted opens that kind's group and collapses the other."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        effects = _group(panel, "effect")
        assert effects.childCount() > 0, "fixture assumption: trigger 0 has an effect"

        def picker_expanded() -> list[bool]:
            groups = [panel.picker_tree.topLevelItem(i) for i in range(panel.picker_tree.topLevelItemCount())]
            assert [g.text(0).split(" ")[0] for g in groups] == ["Conditions", "Effects"]
            return [g.isExpanded() for g in groups]

        def open_from(item) -> list[bool]:
            panel._set_current_entry_item(item)
            panel._request_entry_op("new")
            expanded = picker_expanded()
            panel._close_picker()
            return expanded

        assert open_from(_group(panel, "effect")) == [False, True]
        assert open_from(_group(panel, "condition")) == [True, False]
        assert open_from(_group(panel, "effect").child(0)) == [False, True]
        assert open_from(panel.entry_tree.topLevelItem(0)) == [True, True], "the Trigger row opens both"

        # A filter opens both so a collapsed group cannot hide a match; clearing restores.
        panel._set_current_entry_item(_group(panel, "effect"))
        panel._request_entry_op("new")
        panel.picker_filter.setText("timer")
        assert picker_expanded() == [True, True]
        panel.picker_filter.clear()
        assert picker_expanded() == [False, True]
        panel._close_picker()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_adds_to_the_trigger_it_was_opened_against() -> None:
    """The target is latched at open. The trigger tree stays live while the
    picker shows, so reading the selection back at accept time would put the
    condition on whichever trigger the user had clicked in the meantime.
    """
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        before_0 = _group(panel, "condition").childCount()

        panel._request_entry_op("new")
        # The user clicks another trigger while the picker is up.
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        _pick(panel, "condition", "timer")
        panel._accept_pick()

        from descape import library_compat

        manager = window.trigger_edits.manager()
        assert len(manager.triggers[0].conditions) == before_0 + 1, "the pick missed its trigger"
        vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
        added = manager.triggers[0].conditions[-1]
        assert vocabulary.conditions[added.condition_type].name == "timer"
        # Only the latched trigger's blob is dirty; the one selected in the
        # meantime still splices its original bytes.
        assert window.trigger_edits.is_dirty(0)
        assert not window.trigger_edits.is_dirty(1)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_new_condition_gets_this_documents_own_vocabulary_defaults() -> None:
    """The library builds a new entry from its module-level default_attributes,
    which _initialise_version_dependencies rewrites per load. This pins that
    what comes out matches the vocabulary JSON for the document's own version,
    which is what the property form is about to render it with.
    """
    from descape import library_compat

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)

        window.entry_structural_edit("new", 0, "condition", -1, 3)
        window.entry_structural_edit("new", 0, "effect", -1, 55)
        manager = window.trigger_edits.manager()

        for kind, entries, created in (
            ("condition", vocabulary.conditions, manager.triggers[0].conditions[-1]),
            ("effect", vocabulary.effects, manager.triggers[0].effects[-1]),
        ):
            definition = entries[3 if kind == "condition" else 55]
            # Over `attributes`, the fields this type actually displays, not
            # over every field in default_attributes: the library normalises
            # the armour/attack list slots to None on a type that does not use
            # them, and those are never shown for such a type anyway.
            for attribute in definition.attributes:
                assert getattr(created, attribute) == definition.default_attributes[attribute], (
                    f"new {kind} {definition.name}: {attribute} is not the vocabulary default"
                )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_triggers_do_not_reuse_a_name_the_document_already_has() -> None:
    """Duplicate names are legal in-game, but 4c's folders will key on names --
    trigger_id does not survive a reorder -- so a new trigger gets a free one."""
    window = _triggers_window()
    try:
        for _ in range(3):
            window.trigger_structural_edit("new", ())
        names = [t.name for t in window.trigger_edits.manager().triggers]
        assert names[-3:] == ["New trigger", "New trigger 2", "New trigger 3"]
        assert len(set(names)) == len(names)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_sizes_its_column_to_its_longest_type_name() -> None:
    """Found by screenshotting the panel, not by an assertion: the picker's one
    column kept Qt's ~100 px default and clipped every row to "accumu",
    "ai signa", "bring o". ResizeToContents is safe on a single column where it
    is not on the two-column detail tree -- there is no second column for the
    longest row to push off the pane.
    """
    from PyQt5.QtWidgets import QHeaderView

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel._request_entry_op("new")

        assert panel.picker_tree.header().sectionResizeMode(0) == QHeaderView.ResizeToContents
        longest = max(
            (
                group.child(c).text(0)
                for i in range(panel.picker_tree.topLevelItemCount())
                for group in [panel.picker_tree.topLevelItem(i)]
                for c in range(group.childCount())
            ),
            key=len,
        )
        metrics = panel.picker_tree.fontMetrics()
        assert panel.picker_tree.columnWidth(0) >= metrics.horizontalAdvance(longest), (
            f"the picker column is narrower than {longest!r}, so it is clipping rows"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_takes_the_whole_lower_pane_and_gives_it_back() -> None:
    """Also a screenshot finding: sharing the lower pane with the detail tree
    left 144 selectable types in five visible rows."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.entry_tree.isVisible()

        panel._request_entry_op("new")
        assert not panel.entry_tree.isVisible()
        assert not panel.entry_new_button.isVisible()

        panel._close_picker()
        assert panel.entry_tree.isVisible()
        assert panel.entry_new_button.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_structural_buttons_track_the_selection_and_the_write_support_flag() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.trigger_new_button.isEnabled()
        assert panel.trigger_copy_button.isEnabled()
        assert panel.entry_new_button.isEnabled()

        # The trigger row and the group headings are not entries, so there is
        # nothing for Copy/Delete to act on while one of them is selected.
        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(0))
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled()
        # And the funnel refuses it too, not just the button: the trigger row
        # reaches _current_entry() as kind "trigger", where Copy and Delete go
        # through _selected_entry_ref(), which drops it.
        panel._request_entry_op("retype")
        assert panel.detail_stack.currentIndex() == 0, "the trigger row opened a retype picker"
        panel.entry_tree.setCurrentItem(_group(panel, "effect"))
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_delete_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled()

        _select_first_condition(panel)
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_delete_button.isEnabled()
        assert panel.entry_retype_button.isEnabled()

        # And every one of them goes dead on a file that cannot be written.
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _select_first_condition(panel)
        for button in (
            panel.trigger_new_button,
            panel.trigger_copy_button,
            panel.trigger_delete_button,
            panel.entry_new_button,
            panel.entry_copy_button,
            panel.entry_delete_button,
            panel.entry_retype_button,
        ):
            assert not button.isEnabled(), f"{button.text()} is live on a read-only file"

        # And the gate holds below the buttons too, not just on them.
        panel._request_trigger_op("new")
        panel._request_entry_op("new")
        panel._request_entry_op("retype")
        assert window.trigger_edits is None, "a gated panel must not reach the model"
        assert panel.detail_stack.currentIndex() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- trigger reordering -------------------------------------------------------
#
# The shipped fixture holds an identity display order (4 triggers, 0..3), so
# every test above is blind to the bug this section exists to pin: the panel
# used to list triggers in raw list-index order (viewer.py's old
# `for index, trigger in enumerate(triggers)`), never consulting
# trigger_display_order at all. A non-identity order has to be forced in-test,
# the same technique tests/test_trigger_undo.py's
# test_undo_preserves_a_custom_display_order uses -- through the setter, as a
# *fresh* list (Trap 6: the library mutates a list handed to the setter in
# place, so reusing a variable this test still holds a reference to would
# corrupt it once the getter's list_changed side effect fires).


def _force_custom_display_order(window) -> list[int]:
    manager = window.trigger_panel._manager()
    custom = list(reversed(range(len(manager.triggers))))
    manager.trigger_display_order = list(custom)
    window.trigger_panel.show_scenario(window.scenario)
    return custom


def test_the_panel_sorts_by_display_order_by_default() -> None:
    """The regression 1a exists to prevent: without it, this shows [0,1,2,3]
    even though the file says otherwise."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        assert _tree_order(panel) == custom

        panel.sort_combo.setCurrentIndex(1)  # "File order (trigger ID)"
        assert _tree_order(panel) == list(range(len(custom)))

        panel.sort_combo.setCurrentIndex(0)  # back to "Display order"
        assert _tree_order(panel) == custom
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_current_trigger_index_returns_the_model_index_not_the_row() -> None:
    """The regression 1a exists to prevent, at the read side: under a
    non-identity display order the first *row* and the first *trigger* are
    different triggers."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.current_trigger_index() == custom[0]
        assert custom[0] != 0, "the fixture must start at identity for this to bite"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selection_survives_a_sort_mode_change() -> None:
    """The same trigger stays selected across a sort-mode toggle, even though
    its row moves -- select_trigger() looks it up by index via
    _item_for_index, not by remembering a row number."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        target_index = custom[2]  # not at row 0 in either sort order
        panel.select_trigger(target_index)
        assert panel.current_trigger_index() == target_index

        panel.sort_combo.setCurrentIndex(1)
        assert panel.current_trigger_index() == target_index
        panel.sort_combo.setCurrentIndex(0)
        assert panel.current_trigger_index() == target_index
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selection_survives_a_panel_rebuild() -> None:
    """show_scenario()'s own same-document restore path, under a non-identity
    order: _selection_state()/_restore_selection() operate on rows and stay
    correct as long as the sort mode does not change mid-restore, which it
    does not here."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        target_index = custom[1]
        panel.select_trigger(target_index)

        panel.show_scenario(window.scenario)
        assert panel.current_trigger_index() == target_index
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_execution_order_readout_states() -> None:
    """The three states 1c defines, and specifically that showing them never
    builds a TriggerEditModel -- reading it through
    self._pending_option_values()'s lazily-constructed self.trigger_edits
    would violate the same contract
    test_browsing_every_row_saves_byte_identically pins."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        # The shipped fixture: check it actually carries a value one way,
        # then drive the other two states directly against exec_order_value's
        # contract rather than hunting for corpus files with every state.
        from descape.trigger_model import exec_order_value

        value = exec_order_value(window.scenario)
        assert value is not None, "fixture assumption: this file stores the flag"
        expected = (
            "Executes in trigger-ID order (legacy)." if value else "Executes in display order."
        )
        assert expected in panel.status.text()
        assert window.trigger_edits is None, "the readout must not build an edit model"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_pending_exec_order_flip_updates_the_triggers_readout() -> None:
    """The mirror image of what _pending_option_values() already solves for
    Map Options: flipping the flag there and switching to Triggers mode must
    show the *pending* mode, not the file's stored byte -- otherwise the
    status line asserts the old execution mode as fact."""
    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        before = window.map_options_panel.current_values()["legacy_exec_order"]
        widget.setCurrentIndex(widget.findData(1 - before))
        assert window.trigger_edits is not None, "fixture assumption: the flip built a model"

        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        expected = (
            "Executes in trigger-ID order (legacy) - unsaved change."
            if (1 - before)
            else "Executes in display order - unsaved change."
        )
        assert expected in panel.status.text(), panel.status.text()

        # GH #2: a second flip lands back on the stored byte, so nothing is unsaved.
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")  # rebuilt on re-entry
        widget.setCurrentIndex(widget.findData(before))
        window.mode_combo.setCurrentText("Triggers")
        assert "unsaved change" not in panel.status.text(), panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_are_disabled_at_the_ends_of_display_order() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        count = panel.tree.topLevelItemCount()
        assert count >= 2, "fixture assumption: at least two triggers to move between"

        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert not panel.trigger_move_up_button.isEnabled()
        assert panel.trigger_move_down_button.isEnabled()

        panel.tree.setCurrentItem(panel.tree.topLevelItem(count - 1))
        assert panel.trigger_move_up_button.isEnabled()
        assert not panel.trigger_move_down_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_are_disabled_while_filtered_in_either_view() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        assert panel.trigger_move_up_button.isEnabled()

        for sort in (0, 1):  # Display order, File order (TASK-569)
            panel.sort_combo.setCurrentIndex(sort)
            panel.filter_edit.setText("Fixture")  # matches every shipped trigger
            assert not panel.trigger_move_up_button.isEnabled(), sort
            assert "filter" in panel.trigger_move_up_button.toolTip()
            panel.filter_edit.clear()
            assert panel.trigger_move_up_button.isEnabled(), sort
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_raising_move_shows_the_rolled_back_order_not_a_phantom_success() -> None:
    """The invariant this project holds for exactly this shape of failure: "a UI
    that catches such an error should say so rather than presenting the edit
    as a clean no-op" -- and presenting it as a clean *success* is worse.
    move_row() patches two rows blindly, without reading the model, purely
    from tree-row bounds; if structural_edit() raised for a *model*-level
    reason unrelated to those row bounds, move_row()'s own bounds check would
    not catch it, and patching the tree would show a swap that never actually
    happened in the model the next save reads from.

    A plain out-of-range move (moving trigger 0 up) does not reach this: both
    moved_display_order()'s bounds check and move_row()'s independently agree
    it is invalid, so move_row() itself already no-ops in that case -- this
    corrupts trigger_display_order directly (dropping index 1 from it, which
    the panel's tree has no way to know about) so structural_edit() raises
    (list.index() -> ValueError) for a reason move_row()'s own row-bounds
    check cannot see, and confirms *that* divergence is the one this guards.

    window._warn is stubbed: the factory's default raises on the expected warning.
    """
    window = _triggers_window()
    window._warn = lambda title, text: None
    try:
        panel = window.trigger_panel
        before = _tree_order(panel)
        assert before == [0, 1, 2, 3], "fixture assumption: identity order, 4 triggers"

        manager = window.trigger_panel._manager()
        corrupted = [0, 0, 2, 3]  # index 1 is gone; both row and target (1, 2) stay in-bounds
        manager.trigger_display_order = list(corrupted)

        window.trigger_structural_edit("move_down", [1])  # row 1 -> target 2, both valid rows

        model = window.trigger_edits
        assert list(model.manager().trigger_display_order) == corrupted, (
            "a raising move leaves the model's order exactly as it was -- "
            "structural_edit()'s except branch rolls back to the value it "
            "captured at entry, which is the already-corrupted array here"
        )
        # The buggy shape this test exists to catch would be [0, 2, 0, 3] or
        # similar: move_row(1, +1) blindly swapping rows 1 and 2 regardless of
        # what the model actually holds. The fix falls back to a full
        # repopulate on error, which re-reads the model and reports its real
        # (if corrupted) state honestly instead.
        assert _tree_order(panel) == corrupted, (
            "the tree must reflect the model's real order after a raising "
            "move, not a swap move_row() performed blindly"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_patches_one_row_without_a_full_repopulate() -> None:
    """The point of the "order" refresh tier: show_scenario() is not called,
    so the tree's own QTreeWidgetItem objects survive the move, not just
    their labels."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        before_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])

        assert _tree_order(panel)[0] == moved_index
        after_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        assert {id(i) for i in before_items} == {id(i) for i in after_items}, (
            "a full repopulate would have replaced every QTreeWidgetItem"
        )
        assert panel.current_trigger_index() == moved_index, "the moved trigger stays selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_then_move_down_returns_to_the_original_order() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        original = _tree_order(panel)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])
        assert _tree_order(panel) != original
        window.trigger_structural_edit("move_down", [moved_index])
        assert _tree_order(panel) == original
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_moving_a_trigger_undoes_independently_of_an_exec_order_edit() -> None:
    """tests/test_options_undo.py's
    test_an_exec_order_edit_and_a_trigger_edit_undo_independently establishes
    this at the model layer for the exec-order axis; this is the same claim
    for a display-order move, exercised through the window."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        original = _tree_order(panel)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()
        window.trigger_structural_edit("move_up", [moved_index])
        moved = _tree_order(panel)
        assert moved != original

        model = window._ensure_trigger_edits()
        assert model.exec_order_supported, "fixture assumption"
        with window._trigger_edit(model, "Set exec order") as m:
            m.set_exec_order(1 - m.exec_order)

        window.undo()  # undoes the exec-order flip only
        assert _tree_order(panel) == moved, "the move must survive undoing the later, unrelated edit"

        window.undo()  # undoes the move
        assert _tree_order(panel) == original
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- execution-position column ------------------------------------------------
#
# Column _COL_POS is always the trigger's 0-based position in
# trigger_display_order; only the header says whether that is also the
# execution position. See trigger_model.resolve_exec_mode().


def test_the_position_column_reads_monotonically_under_display_sort() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        assert custom != sorted(custom), "the forced order must not be identity"
        assert _positions(panel) == [str(p) for p in range(len(custom))]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_position_column_is_display_position_under_file_order_sort() -> None:
    """Non-monotonic by design under File order: it is still each trigger's
    display position, not its row. Not a bug to "fix"."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        panel.sort_combo.setCurrentIndex(1)  # "File order (trigger ID)"
        assert _tree_order(panel) == list(range(len(custom)))
        assert _positions(panel) == [str(custom.index(i)) for i in range(len(custom))]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("pending", "expected"), [(0, "Exec #"), (1, "Display #")], ids=["display-order file", "legacy file"]
)
def test_the_position_header_follows_the_execution_mode(pending: int, expected: str) -> None:
    """Driven through pending_exec_order, the same resolution a real stored
    byte goes through; the corpus test below covers stored values."""
    from descape.trigger_model import exec_order_value

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert exec_order_value(window.scenario) is not None, "fixture assumption: the flag is stored"
        panel.show_scenario(window.scenario, pending_exec_order=pending)
        assert _header(panel) == expected
        assert panel.tree.headerItem().toolTip(panel._COL_POS)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_pending_exec_order_flip_switches_the_position_header() -> None:
    """The column's counterpart to
    test_a_pending_exec_order_flip_updates_the_triggers_readout."""
    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        from descape.trigger_model import exec_order_value

        before = exec_order_value(window.scenario)
        assert _header(panel) == ("Display #" if before else "Exec #")

        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        widget.setCurrentIndex(widget.findData(1 - before))
        window.mode_combo.setCurrentText("Triggers")
        assert _header(panel) == ("Exec #" if before else "Display #")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_the_document_resets_the_position_header() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.show_scenario(window.scenario, pending_exec_order=0)
        assert _header(panel) == "Exec #"
        panel.clear_document()
        assert _header(panel) == "Display #"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_relabels_both_swapped_rows() -> None:
    """The fast "order" tier never repopulates, so without move_row()'s own
    setText both rows keep their old numbers. Item identity is asserted too,
    so this cannot pass by regressing to a full repopulate."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        before_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])

        after_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        assert set(map(id, before_items)) == set(map(id, after_items))
        assert _positions(panel) == [str(p) for p in range(len(after_items))]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_grouped_move_keeps_a_pending_exec_order_flip() -> None:
    """move_row()'s grouped fallback repopulates, and must forward the pending
    flip: dropping it snaps header and readout back to the file's stored mode.
    The divider rename and the sort switch are the panel's other two repopulates."""
    from descape.trigger_model import exec_order_value

    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        before = exec_order_value(window.scenario)
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        widget.setCurrentIndex(widget.findData(1 - before))
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        assert panel.tree.topLevelItemCount() >= 3, "fixture assumption: room to group and still move"

        panel.select_trigger(panel._display_slots[0])
        name = _row_widget(panel, "name")
        name.setText("--- grouped move test ---")
        name.editingFinished.emit()
        assert panel._grouped, "fixture assumption: a divider-named trigger groups the tree"

        flipped = "Exec #" if before else "Display #"
        assert _header(panel) == flipped
        moved = panel._display_slots[2]
        panel.select_trigger(moved)
        window.trigger_structural_edit("move_up", [moved])

        assert panel._display_position[moved] == 1, "the move itself landed"
        assert _header(panel) == flipped
        assert "unsaved change" in panel.status.text(), panel.status.text()

        panel.sort_combo.setCurrentIndex(1)  # File order
        assert "unsaved change" in panel.status.text(), panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_position_column_matches_display_order_on_real_files(scenario_path) -> None:
    """Every row's position cell is its index in trigger_display_order, the
    header matches resolve_exec_mode(), and a grouped tree's synthetic header
    row has no position. Covers the not-stored case (F7_3_York) and the
    legacy + non-identity files by walking the whole corpus."""
    from PyQt5.QtCore import Qt

    from descape.trigger_model import exec_order_value, resolve_exec_mode

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        panel = window.trigger_panel
        display = list(panel._manager().trigger_display_order)
        mode = resolve_exec_mode(exec_order_value(window.scenario), None)[0]
        assert _header(panel) == panel._POSITION_HEADERS[mode][0]

        seen = 0
        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            for item in [top] + [top.child(c) for c in range(top.childCount())]:
                index = item.data(0, Qt.UserRole)
                if index is None:
                    assert item.text(panel._COL_POS) == ""
                    continue
                assert item.text(panel._COL_POS) == str(display.index(index))
                seen += 1
        assert seen == len(display)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_no_reorder_or_move_triggers_call_is_introduced() -> None:
    """Reordering here means permuting trigger_display_order only -- ids are
    never renumbered. A tests/test_private_api_guard.py-style source scan, so
    the ID-renumbering path (reorder_triggers()/move_triggers(), both
    display-order-resetting traps -- see trigger_model.py's structural_edit()
    docstring) cannot creep into trigger_structural_edit()'s move_up/
    move_down branch unnoticed."""
    import inspect

    from descape.viewer import ViewerWindow

    # The actual call shape (".reorder_triggers(" / ".move_triggers("), not a
    # bare word match -- this docstring itself names both methods by way of
    # explaining why they must not appear as calls.
    # GH #133's Section Up/Down permute display order in their own helper, and so do GH #211's edge moves.
    source = (
        inspect.getsource(ViewerWindow.trigger_structural_edit)
        + inspect.getsource(ViewerWindow._move_section)
        + inspect.getsource(ViewerWindow._move_to_section_edge)
    )
    assert ".reorder_triggers(" not in source
    assert ".move_triggers(" not in source

    # GH #27 Trap 1: import_triggers() with any index but -1 routes through
    # move_triggers() -> reorder_triggers() inside the library, which a scan
    # for those two call shapes cannot see.
    from descape import trigger_clipboard

    clipboard_source = inspect.getsource(trigger_clipboard)
    for text in (source, clipboard_source):
        assert ".reorder_triggers(" not in text and ".move_triggers(" not in text
        for line in text.splitlines():
            if ".import_triggers(" in line:
                assert "index=-1)" in line, line
    assert ".import_triggers(" in clipboard_source


# -- GH #2: Renumber IDs to Display Order -----------------------------------------


def _renumber_action(panel, index: int = 0):
    return _action(panel.trigger_context_menu(panel._item_for_index[index]), "Renumber IDs to Display Order")


def test_gh2_renumber_is_offered_only_while_ids_differ_from_display_order() -> None:
    window = _gh2_window()
    try:
        panel = window.trigger_panel
        action = _renumber_action(panel)
        assert not action.isEnabled()
        assert action.toolTip() == panel._RENUMBER_TIP + " (IDs already follow display order)"
        panel.select_trigger(0)
        panel.trigger_move_down_button.click()
        assert _order(window) == [1, 0, 2, 3]
        action = _renumber_action(panel)
        assert action.isEnabled() and action.toolTip() == panel._RENUMBER_TIP
        panel.sort_combo.setCurrentIndex(1)
        assert _renumber_action(panel).isEnabled(), "File order view, where it is needed"
        panel.sort_combo.setCurrentIndex(0)
        panel.filter_edit.setText("Fixture")
        assert _renumber_action(panel).isEnabled(), "it acts on the whole list, so a filter does not gate it"
        panel.filter_edit.clear()
        panel._editable = False
        assert _renumber_action(panel).toolTip().endswith(" (this file's triggers cannot be edited)")
        panel._editable = True
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh2_renumber_is_one_record_and_file_order_then_matches_the_list() -> None:
    window = _gh2_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_down", [0])
        window.trigger_structural_edit("move_down", [2])
        listed = _display_names(window)
        panel.select_trigger(3)
        selected = _names(window)[3]
        records = _record_count(window)
        panel.sort_combo.setCurrentIndex(1)
        _renumber_action(panel, 3).trigger()
        assert _record_count(window) == records + 1
        assert window.edit_history.peek_undo().label == "Renumber trigger IDs to display order"
        assert _order(window) == [0, 1, 2, 3]
        names = _names(window)
        assert [names[i] for i in _tree_order(panel)] == listed, "File order view now lists the display order"
        assert [names[i] for i in panel.selected_trigger_indices()] == [selected]
        assert not _renumber_action(panel).isEnabled()
        window.undo()
        assert _display_names(window) == listed and _names(window)[3] == selected
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("mode", ["legacy", "display", "unknown"])
def test_gh2_renumber_says_what_it_changed_for_the_files_execution_order(mode: str, monkeypatch) -> None:
    from types import SimpleNamespace

    from descape.viewer_triggers import DISPLAY_RENUMBER_MESSAGE, LEGACY_RENUMBER_MESSAGE, UNKNOWN_RENUMBER_MESSAGE

    window = _gh2_window()
    try:
        if mode == "legacy":
            window._set_exec_order(SimpleNamespace(label="execution order"), 1)
        elif mode == "unknown":
            set_viewer_global(monkeypatch, "exec_order_value", lambda loaded: None)
        window.trigger_structural_edit("move_down", [0])
        window.trigger_structural_edit("renumber", [])
        said = {
            "legacy": LEGACY_RENUMBER_MESSAGE in _status(window),
            "display": DISPLAY_RENUMBER_MESSAGE in _status(window),
            "unknown": UNKNOWN_RENUMBER_MESSAGE in _status(window),
        }
        assert said == {key: key == mode for key in said}
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- TASK-569: Move Up/Down in File order view moves trigger IDs ------------------


def _ids_window():
    """_gh2_window() with display order [2, 0, 3, 1] (no ID follows its slot), sorted by File order."""
    window = _gh2_window()
    window.trigger_structural_edit("move_top", [2])
    window.trigger_structural_edit("move_down", [1])
    assert _order(window) == [2, 0, 3, 1], "fixture assumption"
    window.trigger_panel.sort_combo.setCurrentIndex(1)
    return window


def _link_targets(window) -> dict[str, list[str]]:
    from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

    names = _names(window)
    return {
        t.name: [names[ce.trigger_id] for ce in get_trigger_referencing_ce(t)]
        for t in window.trigger_panel._manager().triggers
        if get_trigger_referencing_ce(t)
    }


def test_task569_file_order_bounds_tooltips_and_section_buttons() -> None:
    window = _ids_window()
    try:
        panel = window.trigger_panel
        up, down = panel.trigger_move_up_button, panel.trigger_move_down_button
        panel.select_trigger(0)
        assert not up.isEnabled() and down.isEnabled(), "bounds are IDs, not display slots (trigger 0 sits at slot 1)"
        assert up.toolTip() == panel._ID_MOVE_UP_TIP + " (already first)"
        assert down.toolTip() == panel._ID_MOVE_DOWN_TIP
        panel.select_trigger(3)
        assert up.isEnabled() and not down.isEnabled()
        panel.select_trigger(2)
        assert up.isEnabled() and down.isEnabled()
        menu = panel.trigger_context_menu(panel._item_for_index[2])
        tips = {text: _action(menu, text).toolTip() for text in ("Move Up", "Move Down", "Move to Top", "Move to Bottom")}
        assert tips == {
            "Move Up": panel._ID_MOVE_UP_MENU_TIP,
            "Move Down": panel._ID_MOVE_DOWN_MENU_TIP,
            "Move to Top": panel._ID_MOVE_TOP_TIP,
            "Move to Bottom": panel._ID_MOVE_BOTTOM_TIP,
        }
        panel.select_trigger(0)
        assert not _action(panel.trigger_context_menu(panel._item_for_index[0]), "Move to Top").isEnabled()
        panel.sort_combo.setCurrentIndex(0)
        assert up.toolTip().startswith(panel._MOVE_UP_TIP), "the sort switch swaps the tips back"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task569_an_id_move_is_one_record_keeps_display_order_and_links() -> None:
    window = _ids_window()
    try:
        panel = window.trigger_panel
        names, listed, links = _names(window), _display_names(window), _link_targets(window)
        assert links == {names[2]: [names[0], names[1]]}, "fixture assumption"
        panel.select_trigger(1)
        records = _record_count(window)
        panel.trigger_move_down_button.click()
        assert _record_count(window) == records + 1 and _last_label(window) == "Move trigger 1 ID down"
        assert _names(window) == [names[0], names[2], names[1], names[3]]
        assert _display_names(window) == listed, "every trigger keeps its display slot"
        assert _link_targets(window) == links, "links follow their triggers"
        assert [_names(window)[i] for i in _tree_order(panel)] == _names(window), "File order lists by ID"
        assert [_names(window)[i] for i in panel.selected_trigger_indices()] == [names[1]]
        assert _names(window)[panel.current_trigger_index()] == names[1], "the selection follows the trigger"
        window.undo()
        assert _names(window) == names and _display_names(window) == listed and _link_targets(window) == links
        window.redo()
        assert _names(window) == [names[0], names[2], names[1], names[3]] and _link_targets(window) == links
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task569_modifier_clicks_and_the_menu_route_to_id_moves(monkeypatch) -> None:
    from PyQt5.QtCore import Qt

    window = _ids_window()
    try:
        panel = window.trigger_panel
        calls: list[tuple] = []
        monkeypatch.setattr(panel, "_on_trigger_structural", lambda op, indices, arg=None: calls.append((op, list(indices), arg)))
        panel.select_trigger(2)
        panel.trigger_move_up_button.click()
        _modifier_click(panel.trigger_move_up_button, Qt.ShiftModifier)
        _modifier_click(panel.trigger_move_down_button, Qt.ControlModifier)
        _action(panel.trigger_context_menu(panel._item_for_index[2]), "Move to Top").trigger()
        _action(panel.trigger_context_menu(panel._item_for_index[2]), "Move Down").trigger()
        assert calls == [
            ("move_ids", [2], -1),
            ("move_ids", [2], -5),
            ("move_ids", [2], "bottom"),
            ("move_ids", [2], "top"),
            ("move_ids", [2], 1),
        ]
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_task569_far_edge_and_split_id_moves_match_display_order_semantics(tmp_path: Path) -> None:
    from PyQt5.QtCore import Qt

    window = _long_window(tmp_path)
    window._warn = lambda title, text: pytest.fail(f"{title}: {text}")
    try:
        panel = window.trigger_panel
        panel.sort_combo.setCurrentIndex(1)
        names = _names(window)
        _ctrl_select(panel, 2, 4)
        _modifier_click(panel.trigger_move_up_button, Qt.ShiftModifier)
        assert _last_label(window) == "Move 2 trigger IDs up 2 places", "capped by the leading member's room"
        assert _names(window)[:6] == [names[i] for i in (2, 0, 4, 1, 3, 5)]
        assert sorted(_names(window)[i] for i in panel.selected_trigger_indices()) == sorted([names[2], names[4]])
        assert _order(window) == [_names(window).index(name) for name in names], "display slots kept"
        records = _record_count(window)
        panel.select_trigger(_LONG_COUNT - 1)
        window.trigger_structural_edit("move_ids", [_LONG_COUNT - 1], 1)
        assert _record_count(window) == records, "no room records nothing"
        _modifier_click(panel.trigger_move_up_button, Qt.ControlModifier)
        assert _last_label(window) == f"Move trigger {_LONG_COUNT - 1} ID to the start"
        assert _names(window)[0] == names[_LONG_COUNT - 1]
        assert _row_in_view(panel.tree, panel._item_for_index[0])
        records = _record_count(window)
        window.trigger_structural_edit("move_ids", [0], "top")
        assert _record_count(window) == records
        assert _status(window).rstrip().endswith("Cannot move trigger IDs: already first")
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("mode", ["legacy", "display", "unknown"])
def test_task569_an_id_move_says_once_what_it_means_for_the_files_execution_order(mode: str, monkeypatch) -> None:
    from types import SimpleNamespace

    from descape.viewer_triggers import DISPLAY_ID_MOVE_MESSAGE, LEGACY_REORDER_MESSAGE, UNKNOWN_ID_MOVE_MESSAGE

    window = _ids_window()
    try:
        if mode == "legacy":
            window._set_exec_order(SimpleNamespace(label="execution order"), 1)
        elif mode == "unknown":
            set_viewer_global(monkeypatch, "exec_order_value", lambda loaded: None)
        before, records = _status(window), _record_count(window)
        for arg in (1, -1, "bottom"):
            window.trigger_structural_edit("move_ids", [1], arg)
        assert _record_count(window) == records + 3, "every move happened, in every mode"
        said = _status(window)[len(before):]
        assert said.count(DISPLAY_ID_MOVE_MESSAGE) == (mode == "display")
        assert said.count(UNKNOWN_ID_MOVE_MESSAGE) == (mode == "unknown")
        assert LEGACY_REORDER_MESSAGE not in said, "an ID move is not a display-order move"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task569_section_buttons_stay_greyed_in_file_order() -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "--B--")  # 0 | --B-- 1 | --A-- 2, 3
        panel.select_trigger(3)
        assert panel.section_up_button.isEnabled(), "fixture assumption: Section Up works in Display order"
        panel.sort_combo.setCurrentIndex(1)
        panel.select_trigger(3)
        assert not panel.section_up_button.isEnabled() and not panel.section_down_button.isEnabled()
        assert panel.trigger_move_up_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task569_an_id_move_survives_a_write_and_reload(tmp_path: Path) -> None:
    from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _ids_window()
    try:
        names, listed = _names(window), _display_names(window)
        window.trigger_structural_edit("move_ids", [2], "top")
        out = tmp_path / "moved.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        manager = parse_triggers(load_map_and_units(out))
        reloaded = [t.name for t in manager.triggers]
        assert reloaded == [names[2], names[0], names[1], names[3]]
        assert [reloaded[i] for i in manager.trigger_display_order] == listed
        assert [reloaded[ce.trigger_id] for ce in get_trigger_referencing_ce(manager.triggers[0])] == names[:2]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #2: a pending exec-order flip across every repopulate ------------------


def _flip_exec_order(window) -> tuple[str, str]:
    """Flip Execution order in Map Options, then come back to Triggers. Returns
    the flipped mode's position header and its status readout."""
    window.mode_combo.setCurrentText("Map Options")
    # Map Options' value, not exec_order_value(): Triggers may not have parsed its section yet.
    before = window.map_options_panel.current_values()["legacy_exec_order"]
    widget = window.map_options_panel.widget_for("legacy_exec_order")
    assert widget.isEnabled(), widget.toolTip()
    widget.setCurrentIndex(widget.findData(1 - before))
    assert widget.currentData() == 1 - before
    window.mode_combo.setCurrentText("Triggers")
    header = "Exec #" if before else "Display #"
    readout = (
        "Executes in trigger-ID order (legacy) - unsaved change."
        if 1 - before
        else "Executes in display order - unsaved change."
    )
    return header, readout


def _saved_sectioned_file(tmp_path: Path) -> Path:
    """_sectioned_window()'s eight triggers in four sections, saved and closed,
    so a window opening it is grouped from the start."""
    from descape.scenario_write import write_scenario

    window = _sectioned_window()
    try:
        path = tmp_path / "sectioned.aoe2scenario"
        write_scenario(window.scenario, path, **window._edit_model_kwargs())
    finally:
        window.edit_history.mark_saved()
        window.close()
    return path


def test_a_pending_exec_order_flip_survives_each_grouped_repopulate(tmp_path: Path) -> None:
    """TASK-053.01 (GH #2) on _sectioned_window()'s layout, saved and reopened,
    because a flip made after its New trigger edits is refused (see the xfail
    below). Pass conditions: with Execution order flipped in Map Options
    before Triggers is ever entered, the position header reads the flipped
    mode and the status line carries "- unsaved change." after each of
    - opening Triggers on the file, grouped from the start;
    - Move Down of a trigger (grouped, move_row() repopulates);
    - renaming a "--RISK!!!--" divider to a plain name;
    - renaming a plain trigger to a "--- divider ---";
    - switching the sort to File order (header too);
    - switching back to Display order."""
    path = _saved_sectioned_file(tmp_path)
    window = _window()
    try:
        window.load_scenario(path)
        assert window.mode_combo.currentText() != "Triggers"
        header, readout = _flip_exec_order(window)
        panel = window.trigger_panel
        assert [s.header_index for s in panel._sections] == [None, 2, 4, 6], "fixture assumption"

        def still_flipped(step: str) -> None:
            assert _header(panel) == header, f"{step}: header {_header(panel)!r}"
            assert readout in panel.status.text(), f"{step}: {panel.status.text()!r}"

        still_flipped("opening Triggers on a file grouped from the start")
        window.trigger_structural_edit("move_down", [0])
        assert panel._display_position[0] == 1, "the move itself landed"
        still_flipped("Move Down")

        _rename(panel, 4, "plain four")
        assert [s.header_index for s in panel._sections] == [None, 2, 6], "the rename repartitioned"
        still_flipped("rename from a divider")

        _rename(panel, 5, "--- New ---")
        assert [s.header_index for s in panel._sections] == [None, 2, 5, 6], "the rename repartitioned"
        still_flipped("rename to a divider")

        panel.sort_combo.setCurrentIndex(1)  # File order
        still_flipped("File order")
        panel.sort_combo.setCurrentIndex(0)  # Display order
        assert panel._grouped
        still_flipped("back to Display order")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "edit",
    [
        "none",
        "rename",
        "new trigger",
        "delete trigger",
        "new trigger, saved",
    ],
)
def test_the_execution_order_row_stays_editable_after_a_trigger_edit(
    edit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found by TASK-053.01's sweep test (TASK-404). Pass conditions, on the
    trigger fixture in Triggers mode, after `edit`, then Map Options:
    - the Execution order row is enabled;
    - its tooltip is not the alignment-gate read-only reason;
    - the status-line notes carry no alignment-gate note.
    "none" and "rename" are the in-test controls: the same assertions pass
    there, so a structural-edit failure is the edit, not the probe. Without the
    fix both "new trigger" params fail; "delete trigger" passes either way and
    stays as a control."""
    from PyQt5.QtWidgets import QFileDialog

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if edit == "rename":
            _rename(panel, 0, "a much longer name than the fixture's own")
        elif edit.startswith("new trigger"):
            window.trigger_structural_edit("new", [])
        elif edit == "delete trigger":
            window.trigger_structural_edit("delete", [0])
        if edit.endswith(", saved"):
            target = tmp_path / "saved.aoe2scenario"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
            window.save_as()
            assert target.is_file(), "the save landed"
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        assert widget.isEnabled(), widget.toolTip()
        assert widget.toolTip() != window._EXEC_ORDER_READ_ONLY
        assert window._map_options_notes() == []
    finally:
        window.edit_history.mark_saved()
        window.close()
