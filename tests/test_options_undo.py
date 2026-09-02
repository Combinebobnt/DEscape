"""Undo/redo for Map Options edits, at the model level -- no Qt.

Two record kinds reach this surface, and the whole point of the split is that
a user cannot tell them apart in the panel:

- Every byte-patched scalar produces an `OptionsDiffRecord`, restored by
  replaying the raw before/after value into OptionsEditModel.
- The trigger execution-order row produces a `TriggerDiffRecord`, because it
  lives inside the Triggers region and rides on TriggerEditModel's own
  snapshot machinery. That is also what makes it restore together with
  trigger_display_order.

The invariant both share, and the reason a record exists at all: a model that
is dirty while the history is not closes the document with no save prompt. So
every test here checks the model's `has_edits` alongside the value, not just
the value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import option_fields
from descape.edit_history import EditHistory, OptionsDiffRecord
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.trigger_model import TriggerEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _loaded():
    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)
    return loaded


def _a_flag_spec(loaded):
    """Pinned to lock_teams -- see test_options_write_path.py's own copy of
    this helper for why the Diplomacy-panel move (step 5) does not disturb
    this resolution."""
    for spec in option_fields.specs_for(loaded):
        if spec.kind == option_fields.CHECKBOX and spec.section != "Triggers":
            if option_fields.current_value(loaded, spec) in (0, 1):
                assert spec.field_id == "lock_teams", (
                    f"_a_flag_spec silently re-resolved to {spec.field_id!r}"
                )
                return spec
    raise AssertionError("no writable flag row on this file")


def _set(model, history, field_id, value) -> None:
    """The window's own sequence: read the before value, mutate, push. Kept in
    one helper so a test cannot accidentally record a before value captured
    after the mutation."""
    before = model.current_value(field_id)
    model.set_value(field_id, value)
    history.push_options_record(OptionsDiffRecord(f"Set {field_id}", field_id, before, value))


# -- byte-patched scalars ----------------------------------------------------


def test_undoing_an_option_edit_restores_the_stored_value() -> None:
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)
    model = OptionsEditModel(loaded)
    history = EditHistory()

    _set(model, history, spec.field_id, 1 - stored)
    assert model.current_value(spec.field_id) == 1 - stored
    assert model.has_edits

    history.undo([], None, model)
    assert model.current_value(spec.field_id) == stored
    assert not model.has_edits, "an undone option edit still reads as dirty"


def test_redoing_an_option_edit_reapplies_it() -> None:
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)
    model = OptionsEditModel(loaded)
    history = EditHistory()

    _set(model, history, spec.field_id, 1 - stored)
    history.undo([], None, model)
    history.redo([], None, model)
    assert model.current_value(spec.field_id) == 1 - stored
    assert model.has_edits


def test_undoing_two_option_edits_unwinds_them_in_order() -> None:
    loaded = _loaded()
    specs = [
        spec
        for spec in option_fields.specs_for(loaded)
        if spec.kind == option_fields.CHECKBOX
        and spec.section != "Triggers"
        and option_fields.current_value(loaded, spec) in (0, 1)
    ][:2]
    assert len(specs) == 2
    model = OptionsEditModel(loaded)
    history = EditHistory()

    stored = {s.field_id: option_fields.current_value(loaded, s) for s in specs}
    for spec in specs:
        _set(model, history, spec.field_id, 1 - stored[spec.field_id])

    history.undo([], None, model)
    assert model.current_value(specs[1].field_id) == stored[specs[1].field_id]
    assert model.current_value(specs[0].field_id) == 1 - stored[specs[0].field_id]
    history.undo([], None, model)
    assert not model.has_edits


def test_an_option_record_refuses_to_move_the_cursor_with_no_model() -> None:
    """require_target() raises *before* the cursor moves, so a refused undo
    leaves the history exactly where it was rather than silently desynced."""
    history = EditHistory()
    history.push_options_record(OptionsDiffRecord("Set lock teams", "lock_teams", 0, 1))
    with pytest.raises(RuntimeError):
        history.undo([], None, None)
    assert history.cursor == 1
    assert history.can_undo


def test_an_option_record_is_dirty_bookkeeping_like_any_other() -> None:
    history = EditHistory()
    assert not history.is_dirty
    history.push_options_record(OptionsDiffRecord("Set lock teams", "lock_teams", 0, 1))
    assert history.is_dirty
    history.mark_saved()
    assert not history.is_dirty


def test_an_option_record_truncates_redo_like_any_other() -> None:
    loaded = _loaded()
    spec = _a_flag_spec(loaded)
    stored = option_fields.current_value(loaded, spec)
    model = OptionsEditModel(loaded)
    history = EditHistory()

    _set(model, history, spec.field_id, 1 - stored)
    history.undo([], None, model)
    _set(model, history, spec.field_id, 1 - stored)
    assert not history.can_redo
    assert len(history.records) == 1


# -- the exec-order row, which is a trigger record ---------------------------


def test_undoing_an_exec_order_edit_restores_the_flag_and_the_clean_state() -> None:
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    history = EditHistory()
    stored = model.exec_order

    model.begin_trigger_edit()
    model.set_exec_order(1 - stored)
    record = model.commit_trigger_edit("Set trigger execution order", history)
    assert record.kind == "trigger", "the exec-order row must not produce an options record"
    assert model.exec_order == 1 - stored
    assert model.has_edits

    history.undo(loaded.map_manager.terrain, model)
    assert model.exec_order == stored
    assert not model.has_edits, "an undone exec-order edit still reads as dirty"


def test_redoing_an_exec_order_edit_reapplies_the_flag() -> None:
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    history = EditHistory()
    stored = model.exec_order

    model.begin_trigger_edit()
    model.set_exec_order(1 - stored)
    model.commit_trigger_edit("Set trigger execution order", history)
    history.undo(loaded.map_manager.terrain, model)
    history.redo(loaded.map_manager.terrain, model)
    assert model.exec_order == 1 - stored
    assert model.has_edits


def test_an_exec_order_edit_and_a_trigger_edit_undo_independently() -> None:
    """The flag rides in the same snapshot as trigger_display_order, so the two
    have to unwind as separate steps rather than one restoring the other."""
    loaded = _loaded()
    model = TriggerEditModel(loaded)
    history = EditHistory()
    stored = model.exec_order

    model.begin_trigger_edit()
    model.set_exec_order(1 - stored)
    model.commit_trigger_edit("Set trigger execution order", history)

    model.begin_trigger_edit(content_touched=[0])
    model.manager().triggers[0].name = "renamed by the test"
    model.commit_trigger_edit("Set trigger name", history)

    history.undo(loaded.map_manager.terrain, model)
    assert model.exec_order == 1 - stored, "undoing the name edit also reverted the flag"
    history.undo(loaded.map_manager.terrain, model)
    assert model.exec_order == stored
    assert not model.has_edits
