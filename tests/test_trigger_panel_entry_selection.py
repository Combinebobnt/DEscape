"""Trigger panel: multi-entry selection, group edits of conditions and effects,
and Select Same Type.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _ARCHER,
    _ARMOUR_TRIGGER,
    _CREATE_OBJECT,
    _DISPLAY_INSTRUCTIONS,
    _SCOUT,
    TRIGGER_FIXTURE,
    _action,
    _appended_effects,
    _create_objects_window,
    _effect_type_id,
    _effects0,
    _focus_out_every_row,
    _group,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _record_count,
    _row_widget,
    _select_entries,
    _spec_named,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- multi-select entries (GH #60) ---------------------------------------------


def _entry_counts(window, trigger_index: int) -> tuple[int, int]:
    trigger = window.trigger_panel._manager().triggers[trigger_index]
    return (len(trigger.conditions), len(trigger.effects))


def _effect_summary(panel, trigger_index: int, entry_index: int) -> tuple:
    """A comparable summary of one effect, so a delete can be checked to have
    removed the rows it was given rather than merely that many rows."""
    effect = panel._manager().triggers[trigger_index].effects[entry_index]
    return (effect.effect_type, panel._describe("effect", effect))


def test_selected_entry_refs_reads_in_tree_order_and_drops_non_entries() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1), ("condition", 0), ("effect", 0))
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 0), ("effect", 1)]
        assert panel.current_entry_ref() == ("effect", 1), "the first ref is the current row"

        # The trigger row and both group headings are not entries.
        for top in range(3):
            panel.entry_tree.topLevelItem(top).setSelected(True)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 0), ("effect", 1)]

        # A hidden row never reaches a bulk verb.
        panel._entry_item_for("effect", 0).setHidden(True)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 1)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_form_of_two_entries_is_the_fields_they_share() -> None:
    """The armour-split pair decides the rule: both are Modify Attribute, so
    every name matches, but their live cluster slots differ and the cluster
    fields carry different read_only flags. Those must drop out."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 0))
        alone = [s.name for s, *_ in panel._rows]
        cluster = {"quantity", "armour_attack_quantity", "armour_attack_class"}
        assert cluster <= set(alone)

        _select_entries(panel, ("effect", 0), ("effect", 1))
        shared = [s.name for s, *_ in panel._rows]
        assert not cluster & set(shared)
        assert shared == [name for name in alone if name in shared], "the first entry's order"
        assert {"object_list_unit_id", "source_player", "operation", "object_attributes"} <= set(shared)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_two_entry_types_intersect_to_what_they_genuinely_share() -> None:
    """A mixed selection is not refused: it shows what the types really have
    in common, and says so when that is nothing."""
    from PyQt5.QtWidgets import QLabel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)  # Activate Trigger + Deactivate Trigger
        _select_entries(panel, ("effect", 0), ("effect", 1))
        assert [s.name for s, *_ in panel._rows] == ["trigger_id"]
        combo = _row_widget(panel, "trigger_id")
        assert combo.currentText() == "(differs)", "they point at different triggers"

        panel.select_trigger(0)
        _select_entries(panel, ("condition", 0), ("effect", 0))
        assert not panel._rows
        row = panel.property_form.itemAt(0).widget()
        assert isinstance(row, QLabel)
        assert row.text() == "No fields are shared by the 2 selected entries."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_entry_walk_records_nothing_and_saves_byte_identically(tmp_path: Path) -> None:
    """test_browsing_every_row_saves_byte_identically, extended to the set:
    pairing every entry row with every other one builds no model."""
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for i in range(panel.tree.topLevelItemCount()):
            panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
            QApplication.processEvents()
            refs = [
                (kind, c) for kind in ("condition", "effect") for c in range(_group(panel, kind).childCount())
            ]
            for first in refs:
                for second in refs:
                    _select_entries(panel, first, second)

        assert window.trigger_edits is None, "browsing a selection must not build an edit model"
        assert not window.edit_history.is_dirty
        out = tmp_path / "walked.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_selection_change_populates_the_form_once() -> None:
    """A click fires both entry-tree signals; without the latch every click
    builds the form twice, which no widget assertion notices."""
    from PyQt5.QtCore import QItemSelectionModel
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        QApplication.processEvents()
        calls = []
        original = panel._populate_property_form
        panel._populate_property_form = lambda: (calls.append(1), original())[1]

        effects = _group(panel, "effect")
        panel.entry_tree.setCurrentItem(
            effects.child(0), 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        QApplication.processEvents()
        assert len(calls) == 1

        effects.child(1).setSelected(True)
        QApplication.processEvents()
        assert len(calls) == 2, "adding a row to the set rebuilds the form once"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_entry_selection_set_survives_a_rebuild() -> None:
    """show_scenario() rebuilds both trees (every undo does), so the set is
    restored whole, not just its current row."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1), ("condition", 0))
        panel.show_scenario(window.scenario)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 1)]
        assert panel.current_entry_ref() == ("effect", 1)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_and_delete_follow_the_set_while_type_stays_single() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 0))
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_retype_button.isEnabled()

        _select_entries(panel, ("effect", 0), ("effect", 1))
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_delete_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled(), "the picker is one entry's own kind"

        _select_entries(panel)  # nothing resolvable: the selection is unchanged
        panel.entry_tree.clearSelection()
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_delete_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_deleting_two_entries_is_one_record_that_undoes_whole() -> None:
    """Descending removal is what keeps the second index pointing at the row
    the user picked once the first is gone."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        index = 0
        panel.select_trigger(index)
        window.entry_structural_edit("copy", index, "effect", 0, -1)
        window.entry_structural_edit("new", index, "effect", -1, 8)  # ACTIVATE_TRIGGER
        window.edit_history.mark_saved()
        before = _record_count(window)
        conditions, effects = _entry_counts(window, index)
        assert effects == 3
        kept = _effect_summary(panel, index, 1)

        _select_entries(panel, ("effect", 0), ("effect", 2))
        panel.entry_delete_button.click()
        assert _entry_counts(window, index) == (conditions, effects - 2)
        assert _record_count(window) == before + 1, "N deletes are one undo step"
        assert window.edit_history.records[-1].label == "Delete 2 entries"
        assert _effect_summary(panel, index, 0) == kept
        assert panel.selected_entry_refs() == [("effect", 0)], "lands on the lowest removed slot, clamped"

        window.undo()
        assert _entry_counts(window, index) == (conditions, effects)
        window.redo()
        assert _entry_counts(window, index) == (conditions, effects - 2)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copying_two_entries_appends_both_and_selects_them() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        before = _record_count(window)
        conditions, effects = _entry_counts(window, _ARMOUR_TRIGGER)
        originals = [_effect_summary(panel, _ARMOUR_TRIGGER, i) for i in (0, 1)]

        _select_entries(panel, ("effect", 0), ("effect", 1))
        panel.entry_copy_button.click()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects + 2)
        assert _record_count(window) == before + 1, "N copies are one undo step"
        assert window.edit_history.records[-1].label == "Copy 2 entries"
        copies = [_effect_summary(panel, _ARMOUR_TRIGGER, effects + i) for i in (0, 1)]
        assert copies == originals
        assert panel.selected_entry_refs() == [("effect", effects), ("effect", effects + 1)]

        window.undo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects)
        window.redo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects + 2)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_selection_spanning_both_lists_deletes_as_one_record() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        before = _record_count(window)
        conditions, effects = _entry_counts(window, _ARMOUR_TRIGGER)

        _select_entries(panel, ("condition", 0), ("effect", 1))
        panel.entry_delete_button.click()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions - 1, effects - 1)
        assert _record_count(window) == before + 1

        window.undo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_single_entry_delete_keeps_its_old_label_and_landing() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1))
        panel.entry_delete_button.click()
        assert window.edit_history.records[-1].label == "Delete effect"
        assert panel.current_entry_ref() == ("effect", 0), "the last row's delete lands above it"
    finally:
        window.edit_history.mark_saved()
        window.close()


_CHANGE_OWNERSHIP = 18


def _hint_texts(panel) -> list[str]:
    from PyQt5.QtWidgets import QLabel

    return [w.text() for w in panel.property_host.findChildren(QLabel) if "Create Objects" in w.text() and w.isVisible()]


def test_a_create_object_effect_names_the_create_objects_tool() -> None:
    """GH #59: the stamping tool works but is hard to find, so the form says so."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, ("effect", indices[0]))
        assert _hint_texts(panel) == ["Tip: stamp it with Create Objects."]
        _select_entries(panel, *[("effect", i) for i in indices])
        assert len(_hint_texts(panel)) == 1, "a set of Create Objects shares it"

        _select_entries(panel, ("effect", 0))
        assert _effects0(window)[0].effect_type != _CREATE_OBJECT, "fixture assumption"
        assert _hint_texts(panel) == []
        _select_entries(panel, ("effect", 0), ("effect", indices[0]))
        assert _hint_texts(panel) == [], "only when every selected effect is one"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_differing_field_reads_as_differs_and_a_shared_one_shows_its_value() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QComboBox

    from descape.constant_picker import CatalogLineEdit

    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        unit = _row_widget(panel, "object_list_unit_id")
        assert isinstance(unit, CatalogLineEdit) and unit.value() == _ARCHER
        assert unit.line_edit.placeholderText() == "(unset)"
        player = _row_widget(panel, "source_player")
        assert isinstance(player, QComboBox) and player.currentData() == 1
        assert player.findText("(differs)") < 0, "a shared combo gets no (differs) row"
        location = _row_widget(panel, "location_x")
        assert location.is_indeterminate() and location.lineEdit().text() == ""
        item = _row_widget(panel, "item_id")
        shared = _effects0(window)[indices[0]].item_id
        assert not _reads_differs(item) and item.value() == shared, "a shared value is shown as itself"
        sound = _row_widget(panel, "disable_sound")
        assert sound.checkState() != Qt.PartiallyChecked and not sound.isTristate()
    finally:
        window.edit_history.mark_saved()
        window.close()


# TASK-561: the library's Effect.item_id getter is derived from these, so the item id row moves with them.
_HOUSE = 70


_LOOM, _WHEELBARROW = 22, 213


def _modify_attribute_window(count: int = 1):
    window = _triggers_window()
    panel = window.trigger_panel
    indices = _appended_effects(
        window, _effect_type_id(panel, "modify_attribute"), [{"object_list_unit_id": _ARCHER}] * count
    )
    panel.refresh_entries(select=("effect", indices[0]))  # the form was built before the fields were set
    _select_entries(panel, *[("effect", i) for i in indices])
    return window, indices


def _commit_and_settle(widget, value) -> None:
    from PyQt5.QtWidgets import QApplication

    widget.commit_value(value)
    QApplication.processEvents()


def test_a_source_unit_commit_moves_the_item_id_row() -> None:
    """TASK-561: commit_value() is the path Browse, a typed value and the eyedropper share."""
    window, _indices = _modify_attribute_window()
    try:
        panel = window.trigger_panel
        assert _row_widget(panel, "item_id").value() == _ARCHER
        _commit_and_settle(_row_widget(panel, "object_list_unit_id"), _HOUSE)
        assert _row_widget(panel, "item_id").value() == _HOUSE, "the item id row kept the old source unit"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_technology_commit_moves_the_item_id_row() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        index = _appended_effects(window, _effect_type_id(panel, "enable/disable_technology"), [{"technology": _LOOM}])[0]
        panel.refresh_entries(select=("effect", index))
        assert _row_widget(panel, "item_id").value() == _LOOM
        _commit_and_settle(_row_widget(panel, "technology"), _WHEELBARROW)
        assert _row_widget(panel, "item_id").value() == _WHEELBARROW, "the item id row kept the old technology"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_group_source_unit_commit_shows_the_new_shared_item_id() -> None:
    window, _indices = _modify_attribute_window(2)
    try:
        panel = window.trigger_panel
        _commit_and_settle(_row_widget(panel, "object_list_unit_id"), _HOUSE)
        item = _row_widget(panel, "item_id")
        assert not _reads_differs(item) and item.value() == _HOUSE
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_undoing_a_source_unit_commit_shows_the_old_item_id() -> None:
    from PyQt5.QtWidgets import QApplication

    window, _indices = _modify_attribute_window()
    try:
        panel = window.trigger_panel
        _commit_and_settle(_row_widget(panel, "object_list_unit_id"), _HOUSE)
        window.undo()
        QApplication.processEvents()
        assert _row_widget(panel, "object_list_unit_id").value() == _ARCHER
        assert _row_widget(panel, "item_id").value() == _ARCHER
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_rebuild_after_a_source_unit_commit_keeps_focus_on_its_row() -> None:
    from PyQt5.QtWidgets import QApplication

    window, _indices = _modify_attribute_window()
    try:
        panel = window.trigger_panel
        window.activateWindow()
        old = _row_widget(panel, "object_list_unit_id")
        old.line_edit.setFocus()
        QApplication.processEvents()
        assert QApplication.focusWidget() is old.line_edit, "offscreen focus did not take"
        _commit_and_settle(old, _HOUSE)
        new = _row_widget(panel, "object_list_unit_id")
        assert new is not old, "the form was not rebuilt"
        assert QApplication.focusWidget() is new.line_edit, QApplication.focusWidget()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_edit_that_moves_no_other_row_keeps_the_form_widgets() -> None:
    """A source player edit, and a quantity edit whose slot stays (the cluster exemption), patch
    labels only: no rebuild."""
    from PyQt5.QtWidgets import QApplication

    window, _indices = _modify_attribute_window()
    try:
        panel = window.trigger_panel
        before = [w for *_, w in panel._rows]
        player = _row_widget(panel, "source_player")
        player.setCurrentIndex(player.findData(3))
        QApplication.processEvents()
        assert _effects0(window)[_indices[0]].source_player == 3
        assert [w for *_, w in panel._rows] == before, "a source player edit rebuilt the form"

        quantity = _row_widget(panel, "quantity")
        assert not quantity.property("readOnly") and quantity.isEnabled()
        quantity.setValue(7)
        quantity.editingFinished.emit()
        QApplication.processEvents()
        assert _effects0(window)[_indices[0]].quantity == 7
        assert [w for *_, w in panel._rows] == before, "a quantity edit rebuilt the form"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_focusing_every_indeterminate_widget_writes_nothing(tmp_path: Path) -> None:
    """The load-bearing one. Two entries differing in STR, prose, INT, ENUM,
    BOOL, INT_LIST and catalog fields: tab through every row, then save. No
    record, no model, byte-identical output."""
    from PyQt5.QtCore import Qt

    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _DISPLAY_INSTRUCTIONS,
            [
                {"object_list_unit_id": _ARCHER, "source_player": 1, "display_time": 5, "play_sound": 0,
                 "message": "hello", "sound_name": "first"},
                {"object_list_unit_id": _SCOUT, "source_player": 2, "display_time": 9, "play_sound": 1,
                 "message": "goodbye", "sound_name": "second"},
            ],
        )
        indices += _appended_effects(
            window,
            _CHANGE_OWNERSHIP,
            [{"flash_object": 0, "selected_object_ids": [1, 2]}, {"flash_object": 1, "selected_object_ids": [3]}],
        )
        baseline = tmp_path / "baseline.aoe2scenario"
        write_scenario(window.scenario, baseline, triggers=window.trigger_edits)
        records = _record_count(window)

        for pair in (indices[:2], indices[2:]):
            _select_entries(panel, *[("effect", i) for i in pair])
            differs = [s.name for s, _k, _i, w in panel._rows if _reads_differs(w)]
            assert len(differs) >= 2, differs
            _focus_out_every_row(panel)

        assert _record_count(window) == records, "a focus-out on a differing field must not write"
        assert not window.edit_history.is_dirty
        _select_entries(panel, *[("effect", i) for i in indices[2:]])
        assert _row_widget(panel, "flash_object").checkState() == Qt.PartiallyChecked
        out = tmp_path / "focused.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == baseline.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _reads_differs(widget) -> bool:
    """Whether a form widget is in its "(differs)" state, per widget kind."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPlainTextEdit

    from descape.value_picker import ValueLineEdit
    from descape.viewer_common import _IndeterminateMixin

    if isinstance(widget, ValueLineEdit):
        return widget.line_edit.placeholderText() == "(differs)"
    if isinstance(widget, (QLineEdit, QPlainTextEdit)):
        return widget.placeholderText() == "(differs)"
    if isinstance(widget, QComboBox):
        return widget.currentText() == "(differs)"
    if isinstance(widget, QCheckBox):
        return widget.checkState() == Qt.PartiallyChecked
    return isinstance(widget, _IndeterminateMixin) and widget.is_indeterminate()


def test_a_group_edit_writes_every_selected_entry_as_one_record() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        locations = [(_effects0(window)[i].location_x, _effects0(window)[i].location_y) for i in indices]

        unit = _row_widget(panel, "object_list_unit_id")
        unit.line_edit.setText(str(_SCOUT))
        unit.line_edit.editingFinished.emit()

        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_SCOUT] * 3
        assert [(_effects0(window)[i].location_x, _effects0(window)[i].location_y) for i in indices] == locations
        assert _record_count(window) == before + 1, "a group edit is one undo step"
        assert window.edit_history.records[-1].label == "Set source player unit on 3 entries"
        assert panel.selected_entry_refs() == [("effect", i) for i in indices], "the set survives the edit"

        window.undo()
        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_ARCHER] * 3
        window.redo()
        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_SCOUT] * 3
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_group_edit_reaches_the_entries_the_current_row_agrees_with() -> None:
    """"At least one selected entry differs", not "the current one differs":
    a value the current row already holds must still reach the others."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        entry = _effects0(window)[indices[0]]
        window.set_entry_field(0, "effect", indices[0], _spec_named(panel, "effect", entry, "source_player"), 3)
        window.edit_history.mark_saved()
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)

        player = _row_widget(panel, "source_player")
        assert player.currentText() == "(differs)"
        player.setCurrentIndex(player.findData(3))

        assert [_effects0(window)[i].source_player for i in indices] == [3, 3, 3]
        assert _record_count(window) == before + 1
        assert window.edit_history.records[-1].label == "Set source player on 2 entries"
        player.setCurrentIndex(0)  # back to "(differs)": writes nothing
        assert _record_count(window) == before + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_indeterminate_spinbox_commits_its_first_value_to_every_entry() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spin = _row_widget(panel, "location_x")
        assert spin.is_indeterminate() and spin.lineEdit().text() == ""

        spin.setValue(30)
        assert not spin.is_indeterminate()
        assert spin.lineEdit().text() == "30", "the committed value has to become visible"
        assert [_effects0(window)[i].location_x for i in indices] == [30, 30, 30]
        assert [_effects0(window)[i].location_y for i in indices] == [20, 21, 22]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_the_parked_value_into_a_blank_spinbox_still_commits() -> None:
    """The spinbox parks on -1 (UNSET), so typing -1 emits no valueChanged.
    Group-setting a location back to unset must still work."""
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spin = _row_widget(panel, "location_x")
        before = _record_count(window)
        # A bare focus-out first (this offscreen window never moves focus, so
        # it is emitted): nothing typed, nothing written.
        spin.editingFinished.emit()
        assert _record_count(window) == before
        QTest.keyClicks(spin.lineEdit(), "-1")
        spin.editingFinished.emit()
        QApplication.processEvents()
        assert [_effects0(window)[i].location_x for i in indices] == [-1, -1, -1]
        assert spin.lineEdit().text() == "(unset)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_tristate_checkbox_and_a_blank_list_write_only_once_changed() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _CHANGE_OWNERSHIP,
            [{"flash_object": 0, "selected_object_ids": [1, 2]}, {"flash_object": 1, "selected_object_ids": [3]}],
        )
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        flash = _row_widget(panel, "flash_object")
        ids = _row_widget(panel, "selected_object_ids")
        assert flash.checkState() == Qt.PartiallyChecked
        assert ids.text() == "" and ids.placeholderText() == "(differs)"

        # insert(), not setText(): setText clears isModified, the flag that
        # separates "typed in" from "merely focused".
        ids.insert("7, 8")
        ids.editingFinished.emit()
        assert [list(_effects0(window)[i].selected_object_ids) for i in indices] == [[7, 8], [7, 8]]

        flash.setCheckState(Qt.Checked)
        assert [_effects0(window)[i].flash_object for i in indices] == [1, 1]
        assert not flash.isTristate(), "the third state is not a value the user can pick"
        assert _record_count(window) == before + 2
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_blank_text_field_writes_only_once_it_is_typed_in() -> None:
    """Pins the accepted limitation too: a differing text field cannot be
    group-cleared, since an empty box means "leave these alone"."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _DISPLAY_INSTRUCTIONS,
            [{"sound_name": "first", "message": "hello"}, {"sound_name": "second", "message": "goodbye"}],
        )
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        sound = _row_widget(panel, "sound_name")
        message = _row_widget(panel, "message")
        assert sound.text() == "" and sound.placeholderText() == "(differs)"
        assert message.toPlainText() == "" and message.placeholderText() == "(differs)"

        sound.insert("third")
        sound.editingFinished.emit()
        assert [_effects0(window)[i].sound_name for i in indices] == ["third", "third"]

        message.setPlainText("shared")
        message.editingFinished.emit()
        assert [_effects0(window)[i].message for i in indices] == ["shared", "shared"]
        assert _record_count(window) == before + 2
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_single_entry_text_field_still_commits_on_a_plain_focus_out() -> None:
    """N = 1 keeps the pre-#60 rule: the isModified() gate is multi-only."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(window, _DISPLAY_INSTRUCTIONS, [{"sound_name": "first"}])
        _select_entries(panel, ("effect", indices[0]))
        sound = _row_widget(panel, "sound_name")
        sound.setText("typed by the test")  # leaves isModified() False
        sound.editingFinished.emit()
        assert _effects0(window)[indices[0]].sound_name == "typed by the test"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_widget_built_for_another_selection_writes_nothing() -> None:
    """A focus-out can land after the click that moved the selection; the
    widget's captured refs are then stale and must not write anywhere."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spec = _spec_named(panel, "effect", _effects0(window)[indices[0]], "location_x")
        stale = [("effect", i) for i in indices]
        _select_entries(panel, ("effect", indices[0]))
        before = _record_count(window)
        panel._changed(spec, stale, 99)
        assert _record_count(window) == before
        assert [_effects0(window)[i].location_x for i in indices] == [10, 11, 12]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_selects_every_entry_of_that_type() -> None:
    """The issue's literal ask: one Create Object effect in, all of them out,
    the clicked one current. A selection expander only: no model, no record."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        menu = panel.entry_context_menu(panel._entry_item_for("effect", indices[1]))
        action = _action(menu, "Select Same Type")
        assert action.isEnabled()
        action.trigger()

        assert panel.selected_entry_refs() == [("effect", i) for i in indices]
        assert panel.current_entry_ref() == ("effect", indices[1])
        assert _record_count(window) == before
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_skips_other_types_and_hidden_rows() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        panel._entry_item_for("effect", indices[2]).setHidden(True)
        panel.select_same_type_as(panel._entry_item_for("effect", indices[0]))
        # Effect 0 is the fixture's Display Instructions: another type.
        assert panel.selected_entry_refs() == [("effect", indices[0]), ("effect", indices[1])]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_is_offered_only_on_an_entry_row() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        for top in range(3):  # the trigger row and the two group headings
            menu = panel.entry_context_menu(panel.entry_tree.topLevelItem(top))
            assert not _action(menu, "Select Same Type").isEnabled()
        assert _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Select Same Type").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()
