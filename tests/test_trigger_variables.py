"""Phase 4b.6c: the trigger variables dialog and the fifth edit funnel.

Its own file rather than more of the tests/test_trigger_panel_*.py modules, because the risk
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

    show() matters for the same reason it does in tests/trigger_panel_helpers.py:
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
    """Drive the Rename button through its QInputDialog, faked with
    getText returning (name, True)."""
    from PyQt5.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog,
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
        from PyQt5.QtWidgets import QInputDialog

        monkeypatch.setattr(
            QInputDialog,
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


def test_closing_the_window_closes_the_dialog() -> None:
    """TASK-432: close() doesn't destroy the window, so an open dialog stayed up as a stray
    top-level on a shared test QApplication, and a later test counted it as its own."""
    window = _triggers_window()
    try:
        dialog = _dialog(window)
        assert dialog.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()
    assert not dialog.isVisible()
    assert window.trigger_panel._variables_dialog is None


def _bind_variable_shortcuts(window) -> None:
    from PyQt5.QtGui import QKeySequence

    from descape.settings_dialog import SettingsDialog

    dialog = SettingsDialog(window)
    try:
        for aid, key in (
            ("variable_add", "F9"), ("variable_rename", "F10"),
            ("variable_remove", "F11"), ("trigger_variables", "F12"),
        ):
            dialog._keybind_edits[aid].setKeySequence(QKeySequence(f"Ctrl+Alt+{key}"))
    finally:
        dialog.close()


def _variable_key(widget, key) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    top = widget.window()
    top.activateWindow()
    QApplication.setActiveWindow(top)
    widget.setFocus()
    QApplication.processEvents()
    assert top.isActiveWindow() and widget.hasFocus()
    QTest.keyClick(widget, key, Qt.ControlModifier | Qt.AltModifier)
    QApplication.processEvents()


def test_variables_opener_real_key_follows_browse_and_mode_gates() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.settings_dialog import SettingsDialog

    window = conftest.shown_window()
    try:
        _bind_variable_shortcuts(window)
        window.mode_combo.setCurrentText("Triggers")
        action = window._keybind_actions["trigger_variables"]
        assert not action.isEnabled()
        _variable_key(window.trigger_panel.tree, Qt.Key_F12)
        assert window.trigger_panel._variables_dialog is None
        window.load_scenario(TRIGGER_FIXTURE)
        panel = window.trigger_panel
        panel._editable = False
        panel._update_buttons()
        assert action.isEnabled() and panel.variables_button.isEnabled()
        window.mode_combo.setCurrentText("View")
        assert action.shortcut().isEmpty()
        _variable_key(window.map_view, Qt.Key_F12)
        assert panel._variables_dialog is None
        window.mode_combo.setCurrentText("Triggers")
        panel._editable = False
        _variable_key(panel.tree, Qt.Key_F12)
        dialog = panel._variables_dialog
        assert dialog is not None and dialog.isVisible()
        assert _rows(dialog) == FIXTURE_VARIABLES
        assert not dialog.name_edit.isEnabled()
        assert window.trigger_edits is None and not window.edit_history.records
        dialog.close()
        editor = SettingsDialog(window)
        try:
            editor._keybind_edits["trigger_variables"].setKeySequence(QKeySequence("Ctrl+Alt+F8"))
            _variable_key(panel.tree, Qt.Key_F12)
            assert not dialog.isVisible(), "old opener binding remained live"
            _variable_key(panel.tree, Qt.Key_F8)
            assert dialog.isVisible() and panel._variables_dialog is dialog
            dialog.close()
            editor._keybind_edits["trigger_variables"].clear()
            assert settings.get_keybind("trigger_variables") == "" and action.shortcut().isEmpty()
            _variable_key(panel.tree, Qt.Key_F8)
            assert not dialog.isVisible()
        finally:
            editor.close()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("op", ["add", "rename", "remove"])
def test_variable_real_keys_mutate_once_rebind_clear_and_refresh_after_mode_switch(op, monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import QInputDialog

    from descape import settings
    from descape.settings_dialog import SettingsDialog
    from descape.trigger_model import TriggerEditModel

    window = _triggers_window()
    try:
        _bind_variable_shortcuts(window)  # binding exists before the lazy dialog
        dialog = _dialog(window)
        action_id = f"variable_{op}"
        action = window._keybind_actions[action_id]
        assert action not in window.actions() and action not in window.trigger_panel.actions()
        key = {"add": Qt.Key_F9, "rename": Qt.Key_F10, "remove": Qt.Key_F11}[op]
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("renamed", True)))
        structural_calls = []
        structural_edit = TriggerEditModel.structural_edit

        def structural(model, mutate):
            structural_calls.append(model)
            return structural_edit(model, mutate)

        monkeypatch.setattr(TriggerEditModel, "structural_edit", structural)
        window.mode_combo.setCurrentText("View")
        assert not action.shortcut().isEmpty(), "dialog editing is not Triggers-mode scoped"
        dialog.tree.setCurrentItem(None)
        dialog.name_edit.setText("  ")
        assert not action.isEnabled()
        _variable_key(dialog.tree, key)
        assert not window.edit_history.records and window.trigger_edits is None

        dialog.name_edit.setText("new_name")
        dialog.select_variable(0)
        assert action.isEnabled()
        _variable_key(window.map_view, key)
        assert not window.edit_history.records, "focus outside the dialog cannot edit variables"

        editor = SettingsDialog(window)
        try:
            old_key = settings.get_keybind(action_id)
            editor._keybind_edits[action_id].setKeySequence(QKeySequence("Ctrl+Alt+F8"))
            assert settings._load_config()["keybinds"][action_id] == "Ctrl+Alt+F8"
            _variable_key(dialog.tree, key)
            assert not window.edit_history.records, f"old binding {old_key} remained live"
            _variable_key(dialog.tree, Qt.Key_F8)
            expected = {
                "add": [(0, "fixture_var"), (1, "new_name")],
                "rename": [(0, "renamed")], "remove": [],
            }[op]
            assert _live_variables(window) == _rows(dialog) == expected
            assert action.associatedWidgets() == [dialog]
            assert structural_calls == [window.trigger_edits]
            assert len(window.edit_history.records) == window.edit_history.cursor == 1
            assert window.trigger_edits.variables_dirty
            window.undo()
            assert window.edit_history.cursor == 0
            assert _rows(dialog) == _live_variables(window) == FIXTURE_VARIABLES
            window.redo()
            assert window.edit_history.cursor == 1
            assert _rows(dialog) == _live_variables(window) == expected

            editor._keybind_edits[action_id].clear()
            assert action.shortcut().isEmpty() and settings.get_keybind(action_id) == ""
            dialog.name_edit.setText("would_add")
            if expected:
                dialog.select_variable(expected[0][0])
            _variable_key(dialog.tree, Qt.Key_F8)
            assert _live_variables(window) == expected
            assert structural_calls == [window.trigger_edits]
            assert len(window.edit_history.records) == 1
        finally:
            editor.close()
    finally:
        conftest.close_window(window)


def test_variable_shortcuts_follow_read_only_and_individual_button_gates(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QInputDialog

    window = _triggers_window()
    try:
        _bind_variable_shortcuts(window)
        dialog = _dialog(window)
        actions = window.trigger_panel.variable_actions
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: pytest.fail("disabled rename")))
        dialog.tree.setCurrentItem(None)
        dialog.name_edit.setText("new_name")
        assert actions["variable_add"].isEnabled()
        assert not actions["variable_rename"].isEnabled() and not actions["variable_remove"].isEnabled()
        for key in (Qt.Key_F10, Qt.Key_F11):
            _variable_key(dialog.name_edit, key)
        assert window.trigger_edits is None
        dialog.name_edit.clear()
        dialog.select_variable(0)
        assert not actions["variable_add"].isEnabled()
        assert actions["variable_rename"].isEnabled() and actions["variable_remove"].isEnabled()
        _variable_key(dialog.tree, Qt.Key_F9)
        assert window.trigger_edits is None
        dialog.name_edit.setText("new_name")
        window.trigger_panel._editable = False
        window.trigger_panel.refresh_variables()
        assert all(not a.isEnabled() for a in actions.values())
        for key in (Qt.Key_F9, Qt.Key_F10, Qt.Key_F11):
            _variable_key(dialog.tree, key)
        assert _live_variables(window) == FIXTURE_VARIABLES and not window.edit_history.records
        # Dispatch rechecks the button even if an action's enabled state is stale.
        for action in actions.values():
            action.setEnabled(True)
            action.trigger()
        assert window.trigger_edits is None
    finally:
        conftest.close_window(window)


def test_variable_real_key_close_reopen_and_document_recreation_dispatch_once(monkeypatch) -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        _bind_variable_shortcuts(window)
        panel = window.trigger_panel
        actions = dict(panel.variable_actions)
        dispatches = []
        dispatch = panel._dispatch_variable_action

        def counted_dispatch(op):
            dispatches.append(op)
            dispatch(op)

        monkeypatch.setattr(panel, "_dispatch_variable_action", counted_dispatch)
        dialog = _dialog(window)
        for _ in range(3):
            dialog.close()
            dialog.name_edit.setText("hidden_name")
            actions["variable_add"].trigger()
            assert not window.edit_history.records, "hidden dialog must refuse dispatch"
            _variable_key(panel.tree, Qt.Key_F12)
            assert panel._variables_dialog is dialog and dialog.isVisible()
        dispatches.clear()
        dialog.name_edit.setText("first_document")
        _variable_key(dialog.name_edit, Qt.Key_F9)
        assert dispatches == ["add"], "reopening must not reconnect action signals"
        assert _rows(dialog) == [(0, "fixture_var"), (1, "first_document")]
        assert len(window.edit_history.records) == 1

        dialog.name_edit.setText("stale_name")
        assert actions["variable_add"].isEnabled()
        panel.close_variables_dialog()
        assert dialog._keybind_actions == {} and not dialog.actions()
        assert all(not a.isEnabled() and not a.associatedWidgets() for a in actions.values())
        # Before deferred deletion, signals on the old dialog cannot change replacement gates.
        panel._open_variables()  # no event pump until the old dialog's signals are exercised
        replacement = panel._variables_dialog
        assert replacement is not dialog and panel.variable_actions == actions
        replacement.name_edit.clear()
        assert not actions["variable_add"].isEnabled()
        dialog.name_edit.setText("old_deferred_signal")
        assert not actions["variable_add"].isEnabled()
        replacement.name_edit.setText("second_add")
        _variable_key(replacement.name_edit, Qt.Key_F9)
        assert dispatches == ["add", "add"]
        assert _rows(replacement) == [(0, "fixture_var"), (1, "first_document"), (2, "second_add")]
        assert len(window.edit_history.records) == 2

        window.edit_history.mark_saved()
        window.close_scenario()
        assert panel._variables_dialog is None
        assert all(not a.isEnabled() and not a.associatedWidgets() for a in actions.values())
        window.load_scenario(TRIGGER_FIXTURE)
        fresh = _dialog(window)
        assert _rows(fresh) == FIXTURE_VARIABLES
        assert panel.variable_actions == actions
        fresh.name_edit.setText("fresh_document")
        _variable_key(fresh.name_edit, Qt.Key_F9)
        assert dispatches == ["add", "add", "add"]
        assert _rows(fresh) == [(0, "fixture_var"), (1, "fresh_document")]
        assert len(window.edit_history.records) == 1
    finally:
        conftest.close_window(window)
    assert panel._variables_dialog is None
    assert all(not a.isEnabled() and not a.associatedWidgets() for a in actions.values())


def test_standalone_variables_dialog_keeps_existing_constructor_and_buttons() -> None:
    from descape.trigger_panel_variables import VariablesDialog

    conftest.ensure_qapp()
    calls = []
    dialog = VariablesDialog(None, lambda *args: calls.append(args))
    try:
        dialog.set_variables(FIXTURE_VARIABLES, editable=True)
        dialog.name_edit.setText("standalone")
        dialog.add_button.click()
        assert calls == [("add", -1, "standalone")]
        assert dialog._keybind_actions == {} and not dialog.actions()
    finally:
        dialog.close()
