"""Phase 4b.3's viewer wiring: the parts of undo/redo that live in
descape/viewer.py rather than in the model or the history.

Same offscreen-ViewerWindow technique as tests/test_trigger_panel.py and
tests/test_keybinds.py.

Three things here that no headless test of edit_history/trigger_model can
reach, all of them behaviour changes to already-shipped code:

1. **The window's "*" dirty marker.** _update_title() moved out of
   _apply_dirty() and into the undo/redo path, because _apply_dirty()
   early-returns on an empty index list and a trigger undo produces no dirty
   tiles -- so the marker would never update on one. Every other _apply_dirty()
   caller updates the title itself, which is what makes the move safe; nothing
   pinned that before this file.
2. **The empty-history no-op still reports itself**, matching the convention
   paste_tile() and the fill tool spell out: silently doing nothing reads as a
   broken keybind.
3. **trigger_edits is dropped with the document**, or a model would splice one
   file's trigger bytes into the next.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _paint_one_tile(window) -> None:
    """One recorded terrain edit, through the same choke point the tools use."""
    terrain = window.scenario.map_manager.terrain

    def mutate() -> None:
        terrain[0].terrain_id = 1 if terrain[0].terrain_id != 1 else 2

    dirty = window.edit_history.apply("Paint", terrain, mutate)
    window._apply_dirty(dirty)
    window._update_title()


# -- 1. the dirty marker ----------------------------------------------------


def test_the_title_marker_follows_a_tile_edit_and_its_undo() -> None:
    """Pins the _update_title() relocation. A regression here shows up only as
    a window title that stops tracking the document, which no other test in the
    suite would notice."""
    window = conftest.shown_window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        assert not window.windowTitle().startswith("*")

        _paint_one_tile(window)
        assert window.windowTitle().startswith("*"), "an edit must mark the title dirty"

        window.undo()
        assert not window.windowTitle().startswith("*"), "undo back to saved must clear it"

        window.redo()
        assert window.windowTitle().startswith("*")
    finally:
        # mark_saved() before close(): closeEvent -> _confirm_discard_changes
        # raises a modal QMessageBox on a dirty document, which blocks forever
        # offscreen. Same teardown tests/test_trigger_panel.py uses.
        window.edit_history.mark_saved()
        window.close()


def test_undo_and_redo_keep_the_edit_actions_in_step() -> None:
    window = conftest.shown_window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        assert not window.undo_action.isEnabled()

        _paint_one_tile(window)
        window._update_edit_actions()
        assert window.undo_action.isEnabled()
        assert not window.redo_action.isEnabled()

        window.undo()
        assert not window.undo_action.isEnabled()
        assert window.redo_action.isEnabled()
    finally:
        # mark_saved() before close(): closeEvent -> _confirm_discard_changes
        # raises a modal QMessageBox on a dirty document, which blocks forever
        # offscreen. Same teardown tests/test_trigger_panel.py uses.
        window.edit_history.mark_saved()
        window.close()


# -- 2. the empty-history no-op ---------------------------------------------


def test_undo_with_nothing_to_undo_still_reports_itself() -> None:
    """Silently doing nothing reads as a broken keybind, which is why
    paste_tile() and the fill tool log their no-ops too."""
    window = conftest.shown_window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.status_log.clear()

        window.undo()
        assert "undo" in window.status_log.toPlainText().lower(), (
            "an empty undo must say something"
        )

        window.status_log.clear()
        window.redo()
        assert "redo" in window.status_log.toPlainText().lower()
    finally:
        # mark_saved() before close(): closeEvent -> _confirm_discard_changes
        # raises a modal QMessageBox on a dirty document, which blocks forever
        # offscreen. Same teardown tests/test_trigger_panel.py uses.
        window.edit_history.mark_saved()
        window.close()


def test_undo_names_the_edit_it_undid() -> None:
    window = conftest.shown_window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        _paint_one_tile(window)
        window.status_log.clear()
        window.undo()
        assert "Paint" in window.status_log.toPlainText(), (
            "the log names the edit, not just the action"
        )
    finally:
        # mark_saved() before close(): closeEvent -> _confirm_discard_changes
        # raises a modal QMessageBox on a dirty document, which blocks forever
        # offscreen. Same teardown tests/test_trigger_panel.py uses.
        window.edit_history.mark_saved()
        window.close()


# -- 3. the model is dropped with the document ------------------------------


def test_trigger_edits_starts_none_and_is_dropped_with_the_document() -> None:
    """Lazy on purpose: parsing triggers costs seconds on the largest corpus
    files and 4a deliberately kept that off the file-open path."""
    window = conftest.shown_window()
    try:
        assert window.trigger_edits is None
        window.load_scenario(TRIGGER_FIXTURE)
        assert window.trigger_edits is None, "opening a file must not parse triggers"

        from descape.trigger_model import TriggerEditModel

        window.trigger_edits = TriggerEditModel(window.scenario)
        window.load_scenario(BLANK_FIXTURE)
        assert window.trigger_edits is None, "a model must not survive a document switch"

        window.trigger_edits = object()
        window.close_scenario()
        assert window.trigger_edits is None
    finally:
        # mark_saved() before close(): closeEvent -> _confirm_discard_changes
        # raises a modal QMessageBox on a dirty document, which blocks forever
        # offscreen. Same teardown tests/test_trigger_panel.py uses.
        window.edit_history.mark_saved()
        window.close()


def test_saving_with_no_trigger_model_is_byte_identical(tmp_path: Path) -> None:
    """The save wiring lands inert: with no editing UI the model is never
    created, so triggers=None must reproduce exactly the bytes the
    two-argument call produced before 4b.3."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    out = tmp_path / "inert.aoe2scenario"
    write_scenario(loaded, out, triggers=None)
    assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()


# -- 4. every 4b.6b operation survives its own undo (edit -> undo -> save) ----
#
# The shape descape-4b-headless-slice.md's Decision 3 exists for, and the only
# test that proves the UI honours the whole model contract end to end. A record
# that restores the section bytes but not the live object graph, or an operation
# that dirties a blob without pushing a record, both pass everything else in the
# suite and fail here.

_OPERATIONS = [
    ("new trigger", lambda w: w.trigger_structural_edit("new", -1)),
    ("copy trigger", lambda w: w.trigger_structural_edit("copy", 0)),
    # Trigger 2 is "Fixture: references", whose effects activate and deactivate
    # two other triggers. Deleting a referenced trigger is what makes
    # remove_triggers() reset a surviving ce.trigger_id to -1, so this is the
    # case that exercises restore()'s trap 3 rather than only its membership
    # restore. Deleting the referencing trigger itself covers the other side.
    ("delete referenced trigger", lambda w: w.trigger_structural_edit("delete", 0)),
    ("delete referencing trigger", lambda w: w.trigger_structural_edit("delete", 2)),
    ("new condition", lambda w: w.entry_structural_edit("new", 0, "condition", -1, 3)),
    ("new effect", lambda w: w.entry_structural_edit("new", 0, "effect", -1, 55)),
    ("copy condition", lambda w: w.entry_structural_edit("copy", 0, "condition", 0, -1)),
    ("copy effect", lambda w: w.entry_structural_edit("copy", 0, "effect", 0, -1)),
    ("delete condition", lambda w: w.entry_structural_edit("delete", 0, "condition", 0, -1)),
    ("delete effect", lambda w: w.entry_structural_edit("delete", 0, "effect", 0, -1)),
    # An activate-trigger effect is a reference the id remap has to follow, so
    # adding one and undoing it is the content-edit half of the same trap.
    ("new activate-trigger effect", lambda w: w.entry_structural_edit("new", 0, "effect", -1, 53)),
]


@pytest.mark.parametrize("label,operate", _OPERATIONS, ids=[label for label, _ in _OPERATIONS])
def test_a_structural_edit_undone_saves_byte_identically(tmp_path: Path, label, operate) -> None:
    from descape.scenario_write import write_scenario

    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")

        operate(window)
        assert window.trigger_edits is not None, f"{label} did not build an edit model"
        assert window.edit_history.is_dirty, f"{label} recorded no undo step"
        assert window.trigger_edits.has_edits, f"{label} left the model clean"

        window.undo()
        assert not window.edit_history.is_dirty, f"undoing {label} left the history dirty"

        out = tmp_path / "undone.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes(), (
            f"{label} did not restore byte-for-byte through its undo"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("label,operate", _OPERATIONS, ids=[label for label, _ in _OPERATIONS])
def test_a_structural_edit_survives_a_save_and_a_reload(tmp_path: Path, label, operate) -> None:
    """The other direction, and the one the byte-identity test cannot see: an
    edit that is spliced away on save leaves the file byte-identical too, which
    is exactly the silent failure the blob model exists to prevent."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")

        before = _trigger_shape(window.trigger_panel._manager())
        operate(window)
        after = _trigger_shape(window.trigger_edits.manager())
        assert after != before, f"{label} changed nothing in the live document"

        out = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        window.edit_history.mark_saved()
    finally:
        window.edit_history.mark_saved()
        window.close()

    reloaded = parse_triggers(load_map_and_units(out))
    assert reloaded is not None
    assert _trigger_shape(reloaded) == after, f"{label} did not survive the write path"


def _trigger_shape(manager) -> list:
    """Every trigger's name plus its condition and effect types, which is what
    all six operations change and what a splice would silently revert."""
    return [
        (
            trigger.name,
            [c.condition_type for c in trigger.conditions],
            [e.effect_type for e in trigger.effects],
            [e.trigger_id for e in trigger.effects],
        )
        for trigger in manager.triggers
    ]


# -- 5. copying a trigger through the window preserves a custom display order

# The shipped fixture holds an identity display order (4 triggers, 0..3), so
# every "copy trigger" case above is blind to the bug this section exists to
# pin: copy_trigger()'s append_after_source path ends at reorder_triggers(),
# whose triggers-setter resets trigger_display_order to identity. A custom
# order has to be forced in-test, the same technique
# tests/test_trigger_undo.py:test_undo_preserves_a_custom_display_order uses.


def test_copying_a_trigger_through_the_window_preserves_a_custom_display_order() -> None:
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")

        model = window._ensure_trigger_edits()
        manager = model.manager()
        original = list(manager.trigger_display_order)
        custom = list(reversed(range(model.trigger_count)))
        assert custom != original, "the fixture must start at identity for this to bite"
        # A fresh copy, not `custom` itself: the library mutates lists handed
        # to trigger_display_order in place (its getter's list_changed ->
        # update_order_array side effect), so assigning the same list this
        # test still holds a reference to would corrupt it underneath the
        # test once the copy op appends a trigger.
        manager.trigger_display_order = list(custom)
        before_triggers = list(manager.triggers)

        window.trigger_structural_edit("copy", 0)

        after_manager = window.trigger_edits.manager()
        after_order = list(after_manager.trigger_display_order)
        before_ids = {id(t) for t in before_triggers}
        ids_by_slot = [id(after_manager.triggers[i]) for i in after_order]
        without_copy = [i for i in ids_by_slot if i in before_ids]
        expected = [id(before_triggers[old_index]) for old_index in custom]
        assert without_copy == expected, (
            "copying a trigger through the window disturbed the display order "
            "of triggers the user never touched"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()
