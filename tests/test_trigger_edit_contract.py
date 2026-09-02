"""Phase 4b.6's model contract, exercised through ViewerWindow rather than
through trigger_model directly.

Every clause of that contract fails *silently*: an edit that skips mark_dirty()
splices its pre-edit bytes back on save, and a mutation that dirties the model
without pushing a record closes the document with no save prompt. Neither shows
up as an exception, so the tests here are all end-to-end through the write path
or through the history, never assertions that a helper was called.

The load-bearing test is test_a_field_edit_then_undo_saves_byte_identically.
It is the shape descape-4b-headless-slice.md's Decision 3 exists for: one
EditHistory, so an undone trigger edit must leave the file exactly as it was
found, not merely "close".
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

def _name_spec():
    from descape import trigger_fields

    return next(spec for spec in trigger_fields.TRIGGER_FIELDS if spec.name == "name")


def _save(window, path: Path) -> bytes:
    """Save through the same call save_as() makes, minus the file dialog."""
    from descape.scenario_write import write_scenario

    write_scenario(window.scenario, path, triggers=window.trigger_edits)
    return path.read_bytes()


# -- the model is built lazily ----------------------------------------------


def test_no_model_exists_until_a_real_edit_is_made() -> None:
    """Parsing costs seconds on the largest corpus files, and a model that
    exists is a model whose save re-serializes."""
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        assert window.mode == "triggers"
        assert window.trigger_edits is None, "entering Triggers mode must not build a model"

        window.set_trigger_field(0, _name_spec(), "Renamed")
        assert window.trigger_edits is not None
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- the contract ------------------------------------------------------------


def test_a_field_edit_reaches_the_saved_file(tmp_path: Path) -> None:
    """The whole point of 4b: an edit that does not mark its trigger dirty is
    spliced away with no error, so only a reload proves it landed."""
    from descape.scenario_io import load_map_and_units, parse_triggers

    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.set_trigger_field(0, _name_spec(), "Renamed by the test")

        out = tmp_path / "edited.aoe2scenario"
        _save(window, out)
        assert out.read_bytes() != TRIGGER_FIXTURE.read_bytes()

        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].name == "Renamed by the test"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_field_edit_then_undo_saves_byte_identically(tmp_path: Path) -> None:
    """One EditHistory, so an undone trigger edit has to restore the bytes
    exactly. "Close enough" here means a diff against the user's original file
    on every save that ever touched the trigger panel.
    """
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.set_trigger_field(0, _name_spec(), "Renamed by the test")
        window.undo()

        out = tmp_path / "undone.aoe2scenario"
        assert _save(window, out) == TRIGGER_FIXTURE.read_bytes(), (
            "an undone trigger edit must leave the file exactly as found"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_edit_is_undoable_and_redoable() -> None:
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        original = window.trigger_edits
        assert original is None

        window.set_trigger_field(0, _name_spec(), "Renamed")
        assert window.edit_history.can_undo
        assert window.trigger_edits.manager().triggers[0].name == "Renamed"

        window.undo()
        assert window.trigger_edits.manager().triggers[0].name != "Renamed"
        assert window.edit_history.can_redo

        window.redo()
        assert window.trigger_edits.manager().triggers[0].name == "Renamed"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_edit_marks_the_document_dirty() -> None:
    """A model that is dirty while the history is not closes the document with
    no save prompt, which is the failure mode commit_trigger_edit() taking the
    history exists to prevent."""
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        assert not window.windowTitle().startswith("*")

        window.set_trigger_field(0, _name_spec(), "Renamed")
        assert window.windowTitle().startswith("*"), "a trigger edit must mark the title dirty"
        assert window.edit_history.is_dirty
        assert window.trigger_edits.has_edits
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_only_the_edited_trigger_is_marked_dirty() -> None:
    """The minimal-diff guarantee. Every other trigger must still splice
    verbatim, or a one-field edit rewrites the whole section."""
    window = conftest.shown_window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.set_trigger_field(1, _name_spec(), "Only this one")
        assert window.trigger_edits.dirty_indices() == [1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_failed_edit_is_reported_and_still_recorded(monkeypatch) -> None:
    """A raised edit is not a no-op: it can leave every blob dirty, so it needs
    a record to undo and has to tell the user the minimal-diff guarantee is
    gone. Presenting it as a clean no-op is the failure mode."""
    from descape import viewer as viewer_module

    window = conftest.shown_window()
    warned = []
    monkeypatch.setattr(
        viewer_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warned.append(args),
    )
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        model = window._ensure_trigger_edits()
        assert model is not None

        before = window.edit_history.can_undo
        with window._trigger_edit(model, "Deliberate failure"):
            raise RuntimeError("structural edit blew up")

        assert warned, "a failed trigger edit must be reported, not swallowed"
        assert window.edit_history.can_undo and not before, "it still needs a record"
        assert "failed" in window.status_log.toPlainText().lower()
    finally:
        window.edit_history.mark_saved()
        window.close()
