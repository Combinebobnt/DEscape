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
from descape.edit_history import CompositeDiffRecord, EditHistory, OptionsDiffRecord
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
    # By id, not first-checkbox order: custom_conquest now precedes it in _SPECS.
    for spec in option_fields.specs_for(loaded):
        if spec.field_id == "lock_teams":
            assert spec.kind == option_fields.CHECKBOX
            assert option_fields.current_value(loaded, spec) in (0, 1)
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


def test_undoing_a_custom_victory_edit_restores_the_stored_value() -> None:
    loaded = _loaded()
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == "custom_relics")
    stored = option_fields.current_value(loaded, spec)
    model = OptionsEditModel(loaded)
    history = EditHistory()

    _set(model, history, spec.field_id, stored + 7)
    assert model.current_value(spec.field_id) == stored + 7
    assert model.has_edits

    history.undo([], None, model)
    assert model.current_value(spec.field_id) == stored
    assert not model.has_edits, "an undone custom-victory edit still reads as dirty"


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


# -- per-player disable lists ------------------------------------------------
#
# A fifth additive field set on the same model (descape/disables_fields.py),
# and the only one whose value is a tuple rather than an int or a str. It
# rides the same OptionsDiffRecord, so the tests below are the scalar ones
# above with a tuple in place of the number -- plus the two things only a
# variable-length field can get wrong: leaking into serialize_patches(), and
# a revert that compares a list against the stored tuple and never matches.


_DISABLES_FIELD = "disabled:buildings:2"


def test_a_disables_edit_records_and_undoes_like_a_scalar() -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    assert model.disables_supported, "the shipped fixture should pass the disables gate"
    history = EditHistory()
    stored = model.current_value(_DISABLES_FIELD)

    _set(model, history, _DISABLES_FIELD, (72, 621))
    assert model.current_value(_DISABLES_FIELD) == (72, 621)
    assert model.has_edits
    assert model.has_disables_edits

    history.undo(loaded.map_manager.terrain, None, model)
    assert model.current_value(_DISABLES_FIELD) == stored
    assert not model.has_edits
    assert not model.has_disables_edits

    history.redo(loaded.map_manager.terrain, None, model)
    assert model.current_value(_DISABLES_FIELD) == (72, 621)
    assert model.has_edits


def test_setting_a_disables_list_back_to_its_stored_value_clears_the_edit() -> None:
    """The house rule this whole module exists for: a model dirty while the
    history is not closes the document with no save prompt. A *list* here,
    not a tuple -- [1, 2] == (1, 2) is False, so an un-normalized value would
    leave a phantom pending entry behind."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.current_value(_DISABLES_FIELD)

    model.set_value(_DISABLES_FIELD, (72, 621))
    assert model.has_edits
    model.set_value(_DISABLES_FIELD, list(stored))
    assert not model.has_edits
    assert not model.has_disables_edits


def test_a_disables_edit_never_reaches_serialize_patches() -> None:
    """These lists are variable-length; serialize_patches() asserts a fixed
    one. A leak here would fire that assert inside the write path."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(_DISABLES_FIELD, (72,))
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    patches = model.serialize_patches()
    assert patches, "the scalar edit should still emit its own patch"
    resize = model.serialize_disables_resize()
    assert resize is not None
    start, end, replacement = resize
    assert len(replacement) == (end - start) + 4
    assert all(not (start <= offset < end) for offset, _data in patches)


def test_a_clean_model_emits_no_disables_resize() -> None:
    model = OptionsEditModel(_loaded())
    assert model.serialize_disables_resize() is None


def test_set_value_rejects_a_non_sequence_and_an_out_of_range_id() -> None:
    model = OptionsEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_value(_DISABLES_FIELD, 72)
    with pytest.raises(ValueError):
        model.set_value(_DISABLES_FIELD, [-1])
    with pytest.raises(ValueError):
        model.set_value(_DISABLES_FIELD, [0x1_0000_0000])
    assert not model.has_edits


def test_disables_ids_are_deduped_in_first_seen_order() -> None:
    model = OptionsEditModel(_loaded())
    model.set_value(_DISABLES_FIELD, [621, 72, 621, 10])
    assert model.current_value(_DISABLES_FIELD) == (621, 72, 10)


def test_a_disables_edit_and_a_scalar_edit_undo_independently() -> None:
    """Both are OptionsDiffRecords on one model, so a mis-keyed restore
    would silently unwind the wrong one."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    history = EditHistory()
    spec = _a_flag_spec(loaded)
    stored_flag = model.current_value(spec.field_id)

    _set(model, history, _DISABLES_FIELD, (72,))
    _set(model, history, spec.field_id, 1 - stored_flag)

    history.undo(loaded.map_manager.terrain, None, model)
    assert model.current_value(spec.field_id) == stored_flag
    assert model.current_value(_DISABLES_FIELD) == (72,), "undoing the flag reverted the list too"
    history.undo(loaded.map_manager.terrain, None, model)
    assert not model.has_edits


def test_a_dialog_session_undoes_as_one_step() -> None:
    """What ViewerWindow.set_disabled_ids() pushes: several lists changed by
    one OK, wrapped in a CompositeDiffRecord so it costs one Ctrl+Z."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    history = EditHistory()
    changes = {"disabled:buildings:2": (72,), "disabled:techs:5": (22, 23)}
    records = []
    for field_id, ids in changes.items():
        before = model.current_value(field_id)
        model.set_value(field_id, ids)
        records.append(OptionsDiffRecord(f"Set {field_id}", field_id, before, ids))
    history.push_composite_record(CompositeDiffRecord("Set disabled objects", records))

    history.undo(loaded.map_manager.terrain, None, model)
    assert not model.has_edits
    assert model.current_value("disabled:buildings:2") == ()
    assert model.current_value("disabled:techs:5") == ()

    history.redo(loaded.map_manager.terrain, None, model)
    assert model.current_value("disabled:buildings:2") == (72,)
    assert model.current_value("disabled:techs:5") == (22, 23)
    assert model.has_disables_edits
