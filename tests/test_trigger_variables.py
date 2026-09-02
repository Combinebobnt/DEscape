"""Phase 4b.6c: the trigger variables dialog and the fifth edit funnel.

Its own file rather than more of tests/test_trigger_panel.py, because the risk
is a different one. The panel's tests protect "browsing is not editing"; these
protect the thing variables have that triggers do not, which is that **nothing
raises when the contract is broken.**

`TriggerEditModel._check_alignment()` compares the trigger count against the
blob list, so a structural trigger edit that bypassed `structural_edit()` fails
loudly at `serialize()`. There is no equivalent check over `manager.variables`.
A variable mutation made outside `structural_edit()` leaves `_variables_dirty`
False, the old variable block splices back verbatim with its old count, and the
edit is gone on the next save with no error anywhere. So the round-trip tests
below are the only thing standing between that defect and a shipped build.

Rename went through the same funnel once `TriggerSnapshot.variables` switched
from a list of live references to a deep copy taken on every snapshot: a
rename mutates a `Variable` object's `name` in place, and a snapshot holding
the same object would have its "before" name overwritten by the very edit it
exists to undo. The deep copy is what makes the undo tests below meaningful
rather than accidentally passing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

# What the shipped fixture starts with, so every count below reads as a delta.
FIXTURE_VARIABLES = [(0, "fixture_var")]


def _triggers_window():
    """A shown, offscreen ViewerWindow in Triggers mode on the fixture.

    show() matters for the same reason it does in tests/test_trigger_panel.py:
    an unshown window's layout is Qt's fallback, not the code under test.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1500, 900)
    window.show()
    window.load_scenario(TRIGGER_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    QApplication.processEvents()
    return window


def _dialog(window):
    """Open the variables dialog the way a user does, through the button."""
    from PyQt5.QtWidgets import QApplication

    window.trigger_panel.variables_button.click()
    QApplication.processEvents()
    dialog = window.trigger_panel._variables_dialog
    assert dialog is not None
    return dialog


def _rows(dialog) -> list[tuple[int, str]]:
    tree = dialog.tree
    return [
        (int(tree.topLevelItem(i).text(0)), tree.topLevelItem(i).text(1))
        for i in range(tree.topLevelItemCount())
    ]


def _add(dialog, name: str) -> None:
    dialog.name_edit.setText(name)
    dialog.add_button.click()


def _rename(dialog, monkeypatch, name: str) -> None:
    """Drive the Rename button through its QInputDialog, the same way
    test_new_map.py fakes QInputDialog.getInt for the new-map-size prompt."""
    import descape.trigger_panel as trigger_panel_module

    monkeypatch.setattr(
        trigger_panel_module.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: (name, True)),
    )
    dialog.rename_button.click()


def _live_variables(window) -> list[tuple[int, str]]:
    from descape.scenario_io import parse_triggers

    manager = parse_triggers(window.scenario)
    return [(v.variable_id, v.name) for v in manager.variables]


# -- browsing is not editing, here too --------------------------------------


def test_opening_the_dialog_builds_no_model_and_records_nothing() -> None:
    """The same guarantee test_browsing_every_row_saves_byte_identically pins
    for the panel. Opening the list is a read; only Add/Remove is an edit."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        assert _rows(dialog) == FIXTURE_VARIABLES
        assert window.trigger_edits is None, "opening the list must not build an edit model"
        assert not window.edit_history.is_dirty
        assert not window.windowTitle().startswith("*")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_add_is_disabled_until_a_name_is_typed() -> None:
    """Add with a blank name would create an unnamed variable that the dialog
    then cannot tell apart from any other. Gated in the widget rather than
    silently substituted for a generated name."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        assert not dialog.add_button.isEnabled()
        dialog.name_edit.setText("  ")
        assert not dialog.add_button.isEnabled(), "whitespace is not a name"
        dialog.name_edit.setText("gold_target")
        assert dialog.add_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_and_rename_are_disabled_with_no_selection() -> None:
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.tree.setCurrentItem(None)
        assert not dialog.remove_button.isEnabled()
        assert not dialog.rename_button.isEnabled()
        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
        assert dialog.remove_button.isEnabled()
        assert dialog.rename_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the edits themselves ----------------------------------------------------


def test_adding_a_variable_shows_up_and_records_one_undo_step() -> None:
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        _add(dialog, "gold_target")

        assert _rows(dialog) == [(0, "fixture_var"), (1, "gold_target")]
        assert _live_variables(window) == [(0, "fixture_var"), (1, "gold_target")]
        assert window.trigger_edits is not None
        assert window.trigger_edits.variables_dirty
        assert window.edit_history.is_dirty
        assert window.edit_history.can_undo
        # No trigger was touched, so no trigger's bytes may have gone stale.
        assert all(blob is not None for blob in window.trigger_edits._blobs)
        # The name field is cleared, so a second Add is not a silent duplicate
        # of the first.
        assert dialog.name_edit.text() == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_removing_a_variable_leaves_the_others_at_their_own_ids() -> None:
    """Removal renumbers nothing. The removed id becomes a gap, which is what
    makes the next add reuse it (see the id-reuse test below)."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        _add(dialog, "second")
        _add(dialog, "third")
        assert _rows(dialog) == [(0, "fixture_var"), (1, "second"), (2, "third")]

        dialog.select_variable(1)
        dialog.remove_button.click()

        assert _rows(dialog) == [(0, "fixture_var"), (2, "third")]
        assert _live_variables(window) == [(0, "fixture_var"), (2, "third")]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_a_variable_updates_the_list_and_records_one_undo_step(monkeypatch) -> None:
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        _rename(dialog, monkeypatch, "gold_target")

        assert _rows(dialog) == [(0, "gold_target")]
        assert _live_variables(window) == [(0, "gold_target")]
        assert window.trigger_edits is not None
        assert window.trigger_edits.variables_dirty
        assert window.edit_history.is_dirty
        assert window.edit_history.can_undo
        # No trigger was touched, so no trigger's bytes may have gone stale.
        assert all(blob is not None for blob in window.trigger_edits._blobs)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_to_the_same_name_is_a_no_op(monkeypatch) -> None:
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        _rename(dialog, monkeypatch, "fixture_var")

        assert window.trigger_edits is None, "a no-op rename must not build an edit model"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_double_clicking_a_row_opens_rename(monkeypatch) -> None:
    """The button is one affordance; double-click is the other, same as any
    renamable list."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        import descape.trigger_panel as trigger_panel_module

        monkeypatch.setattr(
            trigger_panel_module.QInputDialog,
            "getText",
            staticmethod(lambda *a, **k: ("gold_target", True)),
        )
        item = dialog.tree.topLevelItem(0)
        dialog.tree.setCurrentItem(item)
        dialog.tree.itemDoubleClicked.emit(item, 1)

        assert _rows(dialog) == [(0, "gold_target")]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_undoing_an_add_restores_the_list_and_the_dirty_flag() -> None:
    """`variables_dirty` is restored, not just the list. Left True through an
    undo, the next save re-serializes a variable block the document no longer
    edits and drifts every variable name's NUL padding."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        _add(dialog, "gold_target")
        assert window.trigger_edits.variables_dirty

        window.undo()

        assert _live_variables(window) == FIXTURE_VARIABLES
        assert not window.trigger_edits.variables_dirty
        assert _rows(dialog) == FIXTURE_VARIABLES, "the open dialog refreshes on an undo"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_undoing_a_remove_brings_the_variable_back() -> None:
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        dialog.remove_button.click()
        assert _rows(dialog) == []

        window.undo()

        assert _live_variables(window) == FIXTURE_VARIABLES
        assert _rows(dialog) == FIXTURE_VARIABLES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_undoing_a_rename_restores_the_old_name(monkeypatch) -> None:
    """The case TriggerSnapshot's deep copy exists for. Before the fix,
    `variables` held the same Variable objects the rename mutated in place, so
    the "before" side of the record already read the new name by the time
    undo ran."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        _rename(dialog, monkeypatch, "gold_target")
        assert window.trigger_edits.variables_dirty

        window.undo()

        assert _live_variables(window) == FIXTURE_VARIABLES
        assert not window.trigger_edits.variables_dirty
        assert _rows(dialog) == FIXTURE_VARIABLES, "the open dialog refreshes on an undo"

        window.redo()

        assert _live_variables(window) == [(0, "gold_target")]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_added_variable_survives_a_save_and_reload(tmp_path: Path) -> None:
    """The round trip the model has no runtime check for. Variables live
    outside the trigger list, so the blob reconciliation cannot see them and
    nothing raises if this regresses."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        dialog = _dialog(window)
        _add(dialog, "gold_target")

        out = tmp_path / "added.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert [(v.variable_id, v.name) for v in reloaded.variables] == [
            (0, "fixture_var"),
            (1, "gold_target"),
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_removed_variable_stays_removed_across_a_save_and_reload(tmp_path: Path) -> None:
    """The half with no library API behind it: TriggerManager has no
    remove_variable(), so this is a plain list mutation inside
    structural_edit(), and the count in the rewritten block has to shrink with
    it. A count that did not follow would leave the reload reading a stale
    entry, or running off the end of the block."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        dialog.remove_button.click()

        out = tmp_path / "removed.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert [(v.variable_id, v.name) for v in reloaded.variables] == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_renamed_variable_survives_a_save_and_reload(tmp_path: Path, monkeypatch) -> None:
    """The same blind spot as the add/remove round trips, for content rather
    than membership: `_variable_signature()` diffs `.name` too, so a rename
    that skipped structural_edit() would splice the old variable block back
    verbatim with nothing raising."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        _rename(dialog, monkeypatch, "gold_target")

        out = tmp_path / "renamed.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert [(v.variable_id, v.name) for v in reloaded.variables] == [(0, "gold_target")]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_removing_then_adding_reuses_the_freed_id() -> None:
    """Documented, not prevented. add_variable() takes the lowest free id, so
    the new variable inherits every reference the document still holds to the
    removed one. The in-game editor behaves the same way, and renumbering to
    avoid it would break references that are still correct."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        dialog.remove_button.click()
        _add(dialog, "reused")

        assert _live_variables(window) == [(0, "reused")]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the shape of the dialog -------------------------------------------------


def test_the_name_column_is_still_not_inline_editable() -> None:
    """Rename goes through QInputDialog, not an in-place tree edit -- an
    ItemIsEditable flag on the Name column would let a rename bypass the
    single funnel (variable_structural_edit) that keeps it undoable."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        dialog = _dialog(window)
        item = dialog.tree.topLevelItem(0)
        assert not item.flags() & Qt.ItemIsEditable
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_read_only_file_can_browse_variables_but_not_edit_them(monkeypatch) -> None:
    """The button is gated on a document being open, not on write support: the
    list is worth reading either way. Add, Rename, and Remove are gated
    inside."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        dialog.select_variable(0)
        # The panel's own editability flag is what the dialog is handed, so
        # forcing it here exercises the same path a 1.54 file takes.
        window.trigger_panel._editable = False
        window.trigger_panel.refresh_variables()

        assert window.trigger_panel.variables_button.isEnabled()
        assert _rows(dialog) == FIXTURE_VARIABLES
        assert not dialog.add_button.isEnabled()
        assert not dialog.rename_button.isEnabled()
        assert not dialog.remove_button.isEnabled()
        assert not dialog.name_edit.isEnabled()
        # _request_rename() guards on _editable itself, not just the button's
        # enabled state -- belt and braces the same way add/remove do.
        _rename(dialog, monkeypatch, "gold_target")
        assert _live_variables(window) == FIXTURE_VARIABLES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_the_document_closes_the_dialog() -> None:
    """Its contents belong to a document that is no longer open, and an edit
    made against one would go to whichever model _ensure_trigger_edits() built
    next."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        assert dialog.isVisible()

        window.trigger_panel.clear_document()

        assert window.trigger_panel._variables_dialog is None
        assert not window.trigger_panel.variables_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()
