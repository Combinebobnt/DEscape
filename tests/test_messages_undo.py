"""Undo/redo for Messages mode edits, at the model level -- no Qt.

Mirrors tests/test_options_undo.py's shape: every test asserts
model.has_edits alongside the value, because a model dirty while the
history is not closes the document with no save prompt.
"""

from __future__ import annotations

import pytest

from descape.edit_history import EditHistory, MessagesDiffRecord
from descape.messages_fields import STRING_ID_UNSET
from descape.messages_model import MessagesEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units


def _loaded():
    return load_map_and_units(FIXTURE_PATH)


def _set(model, history, field_id, value) -> None:
    before = model.current_value(field_id)
    model.set_value(field_id, value)
    history.push_messages_record(MessagesDiffRecord(f"Set {field_id}", field_id, before, value))


def test_undoing_a_messages_text_edit_restores_the_stored_value() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    history = EditHistory()
    original = model.current_value("hints")

    _set(model, history, "hints", "new hint text")
    assert model.current_value("hints") == "new hint text"
    assert model.has_edits

    history.undo([], None, None, None, model)
    assert model.current_value("hints") == original
    assert not model.has_edits, "an undone Messages edit still reads as dirty"


def test_redoing_a_messages_text_edit_reapplies_it() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    history = EditHistory()

    _set(model, history, "hints", "new hint text")
    history.undo([], None, None, None, model)
    history.redo([], None, None, None, model)
    assert model.current_value("hints") == "new hint text"
    assert model.has_edits


def test_undoing_a_string_id_edit_restores_it() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    history = EditHistory()

    _set(model, history, "hints_id", 999)
    assert model.current_value("hints_id") == 999
    history.undo([], None, None, None, model)
    assert model.current_value("hints_id") == STRING_ID_UNSET
    assert not model.has_edits


def test_undoing_two_messages_edits_unwinds_them_in_order() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    history = EditHistory()

    hints_original = model.current_value("hints")
    scouts_original = model.current_value("scouts")
    _set(model, history, "hints", "hint edit")
    _set(model, history, "scouts", "scout edit")

    history.undo([], None, None, None, model)
    assert model.current_value("scouts") == scouts_original
    assert model.current_value("hints") == "hint edit"
    history.undo([], None, None, None, model)
    assert model.current_value("hints") == hints_original
    assert not model.has_edits


def test_a_messages_record_refuses_to_move_the_cursor_with_no_model() -> None:
    """require_target() raises *before* the cursor moves, so a refused undo
    leaves the history exactly where it was rather than silently desynced."""
    history = EditHistory()
    history.push_messages_record(MessagesDiffRecord("Set hints", "hints", "a", "b"))
    with pytest.raises(RuntimeError):
        history.undo([], None, None, None, None)
    assert history.cursor == 1
    assert history.can_undo


def test_a_messages_record_is_dirty_bookkeeping_like_any_other() -> None:
    history = EditHistory()
    assert not history.is_dirty
    history.push_messages_record(MessagesDiffRecord("Set hints", "hints", "a", "b"))
    assert history.is_dirty
    history.mark_saved()
    assert not history.is_dirty


def test_a_messages_record_truncates_redo_like_any_other() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    history = EditHistory()

    _set(model, history, "hints", "first edit")
    history.undo([], None, None, None, model)
    _set(model, history, "hints", "second edit")

    assert not history.can_redo
    assert model.current_value("hints") == "second edit"
