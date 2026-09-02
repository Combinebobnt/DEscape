"""Undo/redo for Diplomacy grid edits, at the model level -- no Qt.

A stance or allied-victory cell rides the same OptionsDiffRecord every other
OptionsEditModel field does (see options_model.py's module docstring), so
this is test_options_undo.py's own shape applied to the grid's synthetic
ids (stance:i:j, allied_victory:i) rather than a new mechanism -- the whole
point being that a user cannot tell a Diplomacy edit's undo step apart from
a Map Options or Players mode one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.diplomacy_fields import allied_victory_cell_id, stance_cell_id
from descape.edit_history import EditHistory, OptionsDiffRecord
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _loaded():
    return load_map_and_units(FIXTURE_PATH)


def _set(model, history, field_id, value) -> None:
    """The window's own sequence: read the before value, mutate, push. Kept in
    one helper so a test cannot accidentally record a before value captured
    after the mutation -- same helper test_options_undo.py uses."""
    before = model.current_value(field_id)
    model.set_value(field_id, value)
    history.push_options_record(OptionsDiffRecord(f"Set {field_id}", field_id, before, value))


# -- stance cells -------------------------------------------------------------


def test_undoing_a_stance_edit_restores_the_stored_value() -> None:
    loaded = _loaded()
    cell_id = stance_cell_id(1, 2)
    model = OptionsEditModel(loaded)
    stored = model.current_value(cell_id)
    other = next(v for v in (0, 1, 3) if v != stored)
    history = EditHistory()

    _set(model, history, cell_id, other)
    assert model.current_value(cell_id) == other
    assert model.has_edits
    assert model.has_diplomacy_edits

    history.undo([], None, model)
    assert model.current_value(cell_id) == stored
    assert not model.has_edits, "an undone stance edit still reads as dirty"
    assert not model.has_diplomacy_edits


def test_redoing_a_stance_edit_reapplies_it() -> None:
    loaded = _loaded()
    cell_id = stance_cell_id(1, 2)
    model = OptionsEditModel(loaded)
    stored = model.current_value(cell_id)
    other = next(v for v in (0, 1, 3) if v != stored)
    history = EditHistory()

    _set(model, history, cell_id, other)
    history.undo([], None, model)
    history.redo([], None, model)
    assert model.current_value(cell_id) == other
    assert model.has_edits
    assert model.has_diplomacy_edits


def test_editing_one_stance_direction_leaves_the_reverse_pair_untouched_across_undo() -> None:
    """Directionality holds through undo too, not just through the write
    path (tests/test_diplomacy_write_path.py already covers the latter)."""
    loaded = _loaded()
    fwd_id, rev_id = stance_cell_id(1, 2), stance_cell_id(2, 1)
    model = OptionsEditModel(loaded)
    stored_fwd = model.current_value(fwd_id)
    stored_rev = model.current_value(rev_id)
    other = next(v for v in (0, 1, 3) if v != stored_fwd)
    history = EditHistory()

    _set(model, history, fwd_id, other)
    assert model.current_value(rev_id) == stored_rev

    history.undo([], None, model)
    assert model.current_value(fwd_id) == stored_fwd
    assert model.current_value(rev_id) == stored_rev


# -- allied-victory flags ------------------------------------------------


def test_undoing_an_allied_victory_edit_restores_the_stored_value() -> None:
    loaded = _loaded()
    cell_id = allied_victory_cell_id(1)
    model = OptionsEditModel(loaded)
    stored = model.current_value(cell_id)
    history = EditHistory()

    _set(model, history, cell_id, 1 - stored)
    assert model.current_value(cell_id) == 1 - stored
    assert model.has_diplomacy_edits

    history.undo([], None, model)
    assert model.current_value(cell_id) == stored
    assert not model.has_edits
    assert not model.has_diplomacy_edits


# -- mixed with an ordinary Map Options scalar --------------------------------


def test_a_stance_edit_and_a_scalar_edit_undo_independently() -> None:
    """A Diplomacy cell and a Map Options scalar both push an
    OptionsDiffRecord -- both ride the same edit model by design, so
    the two must unwind as separate steps rather than one restoring both."""
    loaded = _loaded()
    cell_id = stance_cell_id(1, 2)
    model = OptionsEditModel(loaded)
    stance_stored = model.current_value(cell_id)
    stance_other = next(v for v in (0, 1, 3) if v != stance_stored)
    lock_teams_stored = model.current_value("lock_teams")
    history = EditHistory()

    _set(model, history, cell_id, stance_other)
    _set(model, history, "lock_teams", 1 - lock_teams_stored)

    history.undo([], None, model)
    assert model.current_value("lock_teams") == lock_teams_stored
    assert model.current_value(cell_id) == stance_other, "undoing the scalar also reverted the cell"
    history.undo([], None, model)
    assert model.current_value(cell_id) == stance_stored
    assert not model.has_edits


def test_a_diplomacy_record_truncates_redo_like_any_other() -> None:
    loaded = _loaded()
    cell_id = stance_cell_id(1, 2)
    model = OptionsEditModel(loaded)
    stored = model.current_value(cell_id)
    other = next(v for v in (0, 1, 3) if v != stored)
    history = EditHistory()

    _set(model, history, cell_id, other)
    history.undo([], None, model)
    _set(model, history, cell_id, other)
    assert not history.can_redo
    assert len(history.records) == 1


def test_a_diplomacy_record_is_dirty_bookkeeping_like_any_other() -> None:
    history = EditHistory()
    assert not history.is_dirty
    history.push_options_record(
        OptionsDiffRecord("Set stance:1:2", stance_cell_id(1, 2), 3, 0)
    )
    assert history.is_dirty
    history.mark_saved()
    assert not history.is_dirty


def test_a_diplomacy_record_refuses_to_move_the_cursor_with_no_model() -> None:
    history = EditHistory()
    history.push_options_record(
        OptionsDiffRecord("Set stance:1:2", stance_cell_id(1, 2), 3, 0)
    )
    with pytest.raises(RuntimeError):
        history.undo([], None, None)
    assert history.cursor == 1
    assert history.can_undo
