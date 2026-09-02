"""Phase 4b.3's undo/redo: descape/trigger_model.py's snapshot/restore pair and
the second record type in descape/edit_history.py.

The claims under test, in the order the risk runs:

1. **A trigger undo is byte-clean.** Edit, undo, save, and the file must be
   byte-identical to the one that was opened. This is the check that fails on
   the design 4b.3 was originally planned with: a record carrying only
   before/after section *bytes* restores neither the live object graph the
   panel reads nor the blob dirty state, so the next save silently takes the
   re-serializing branch and stops being byte-identical.
2. **A structural undo puts the whole graph back.** remove/reorder rewrite
   trigger ids and cross-trigger references on triggers the user never named,
   in place, so restoring list membership alone is not enough. Checked against
   an independently reloaded scenario rather than against remembered values,
   the same technique tests/test_write_path.py's _check_history_invariants uses.
3. **One history, two record types.** Tile and trigger records interleave on a
   single EditHistory, and is_dirty/mark_saved/cursor semantics hold across the
   boundary. A second history for triggers would let a dirty document close
   without a save prompt, which is why there is only one.
4. **Records survive being replayed.** Undo, redo, undo again must reproduce the
   same state each time -- a record that handed its own object to the live list
   gets mutated by the next edit and replays already-edited state.

Every restore trap these exercise is documented on TriggerEditModel.restore().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.edit_history import EditHistory, TriggerDiffRecord
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_REFERENCE_EFFECT_TYPES = (8, 9)  # ACTIVATE_TRIGGER, DEACTIVATE_TRIGGER


class FakeTile:
    """The same duck-typed tile tests/test_edit_history.py uses -- EditHistory
    only ever reads/writes these three attributes."""

    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


def _reference_map(manager) -> list[list[int]]:
    return [
        [
            effect.trigger_id
            for effect in trigger.effects
            if effect.effect_type in _REFERENCE_EFFECT_TYPES
        ]
        for trigger in manager.triggers
    ]


def _graph_state(manager) -> tuple:
    """Everything a remap rewrites, as one comparable value."""
    return (
        [trigger.trigger_id for trigger in manager.triggers],
        [trigger.name for trigger in manager.triggers],
        _reference_map(manager),
        list(manager.trigger_display_order),
    )


def _fresh_graph_state() -> tuple:
    """Read from an independently reloaded scenario, so an undo is compared
    against the file rather than against values this test remembered."""
    loaded = load_map_and_units(FIXTURE_PATH)
    return _graph_state(parse_triggers(loaded))


def _section_bytes(loaded) -> bytes:
    return loaded.decompressed_body[loaded.units_section_end : loaded.triggers_section_end]


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, TriggerEditModel(loaded), EditHistory()


# -- 1. a trigger undo is byte-clean ----------------------------------------


def test_edit_undo_save_is_byte_identical_to_the_original_file(tmp_path: Path) -> None:
    """The test the original 4b.3 design would have failed. A byte-snapshot
    record leaves _blobs[i] = None after an undo, so the save re-serializes a
    trigger the document no longer edits and picks up NUL-trail drift."""
    loaded, model, history = _open()

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed by the test"
    model.commit_trigger_edit("Rename trigger", history)
    assert model.has_edits

    history.undo(loaded.map_manager.terrain, model)

    assert not model.has_edits, "undo must restore blob cleanliness, not just content"
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_restores_the_trigger_content_itself() -> None:
    loaded, model, history = _open()
    original_name = model.manager().triggers[0].name

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed by the test"
    model.commit_trigger_edit("Rename trigger", history)

    history.undo(loaded.map_manager.terrain, model)
    assert model.manager().triggers[0].name == original_name


def test_undo_returns_no_tile_indices() -> None:
    """undo() is always a tile-index list, so viewer.py's _apply_dirty() pipe
    and the five other callers that read it as one keep working untouched."""
    loaded, model, history = _open()
    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "x"
    model.commit_trigger_edit("Rename trigger", history)

    assert history.undo(loaded.map_manager.terrain, model) == []
    assert history.redo(loaded.map_manager.terrain, model) == []


def test_a_trigger_record_without_a_model_raises_before_moving_the_cursor() -> None:
    """A silent no-op here would leave the cursor decremented with the document
    unchanged, desyncing every later undo."""
    loaded, model, history = _open()
    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "x"
    model.commit_trigger_edit("Rename trigger", history)
    cursor_before = history.cursor

    with pytest.raises(RuntimeError, match="no TriggerEditModel"):
        history.undo(loaded.map_manager.terrain)
    assert history.cursor == cursor_before


def test_undo_restores_variables_dirty_false() -> None:
    """Plan verification item 9. A stale _variables_dirty re-serializes the
    variable block of a clean document, reintroducing NUL-trail drift across
    every variable name."""
    loaded, model, history = _open()
    assert not model.variables_dirty

    model.begin_trigger_edit()
    model.structural_edit(lambda m: m.add_variable("undo_test_var"))
    model.commit_trigger_edit("Add variable", history)
    assert model.variables_dirty

    history.undo(loaded.map_manager.terrain, model)
    assert not model.variables_dirty
    assert not model.has_edits
    assert model.serialize() == _section_bytes(loaded)


# -- 2. a structural undo puts the whole graph back -------------------------


def test_undo_of_a_remove_restores_ids_references_and_display_order() -> None:
    """Plan verification item 2. remove_triggers() rewrites trigger_id and
    ce.trigger_id on the *surviving* triggers in place, so this fails on any
    restore that only puts list membership back."""
    loaded, model, history = _open()
    expected = _fresh_graph_state()

    model.begin_trigger_edit()
    model.structural_edit(lambda m: m.remove_trigger(0))
    model.commit_trigger_edit("Remove trigger", history)
    assert _graph_state(model.manager()) != expected

    history.undo(loaded.map_manager.terrain, model)
    assert _graph_state(model.manager()) == expected


def test_undo_of_a_remove_is_byte_identical(tmp_path: Path) -> None:
    loaded, model, history = _open()

    model.begin_trigger_edit()
    model.structural_edit(lambda m: m.remove_trigger(0))
    model.commit_trigger_edit("Remove trigger", history)

    history.undo(loaded.map_manager.terrain, model)
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_of_a_reorder_restores_the_graph() -> None:
    """A reorder leaves the trigger count identical and only object identity
    reveals it, which is the case _check_alignment() exists for."""
    loaded, model, history = _open()
    expected = _fresh_graph_state()
    count = model.trigger_count

    model.begin_trigger_edit()
    model.structural_edit(lambda m: m.reorder_triggers(list(reversed(range(count)))))
    model.commit_trigger_edit("Reorder triggers", history)

    history.undo(loaded.map_manager.terrain, model)
    assert _graph_state(model.manager()) == expected


def test_undo_preserves_a_custom_display_order() -> None:
    """Trap 1. `manager.triggers = [...]` resets trigger_display_order to
    identity in its setter, and restoring the order explicitly is the only
    thing that puts a custom one back.

    Reached through the setter rather than through reorder_triggers(), which
    physically reorders the list and renumbers ids while *leaving display order
    at identity* -- so nothing else in this file can tell the two apart, and a
    reorder-based test would pass with the restore removed entirely. A custom
    display order changes the serialized bytes, so losing it on undo is real
    data loss, not a cosmetic reset.
    """
    loaded, model, history = _open()
    original = list(model.manager().trigger_display_order)
    custom = list(reversed(range(model.trigger_count)))
    assert custom != original, "the fixture must start at identity for this to bite"

    # The undone edit has to be the one that *changes* the order. Undoing some
    # other edit while a custom order merely sits there proves nothing: slice
    # assignment preserves the live order, so restoring the same value it
    # already holds is invisible.
    model.begin_trigger_edit()
    model.structural_edit(lambda m: setattr(m, "trigger_display_order", list(custom)))
    model.commit_trigger_edit("Reorder display", history)
    assert list(model.manager().trigger_display_order) == custom
    # The has_edits hole the trigger-reordering plan measured directly: a pure
    # display-order permutation dirties no blob and changes no trigger
    # identity, so without structural_edit()'s display-order check this reads
    # False for an edit that demonstrably changed the serialized bytes below --
    # and scenario_write.py gates the whole splice on has_edits, so the reorder
    # would be silently dropped on save with nothing here to catch it.
    assert model.has_edits
    assert model.serialize() != _section_bytes(loaded), "display order reaches the bytes"

    history.undo(loaded.map_manager.terrain, model)
    assert list(model.manager().trigger_display_order) == original
    assert model.serialize() == _section_bytes(loaded)


def test_redo_after_undo_of_a_structural_edit() -> None:
    """Plan verification item 4."""
    loaded, model, history = _open()

    model.begin_trigger_edit()
    model.structural_edit(lambda m: m.remove_trigger(0))
    model.commit_trigger_edit("Remove trigger", history)
    after_edit = _graph_state(model.manager())
    edited_bytes = model.serialize()

    history.undo(loaded.map_manager.terrain, model)
    history.redo(loaded.map_manager.terrain, model)

    assert _graph_state(model.manager()) == after_edit
    assert model.serialize() == edited_bytes


# -- 3. one history, two record types ---------------------------------------


def test_tile_and_trigger_records_interleave_on_one_history() -> None:
    loaded, model, history = _open()
    tiles = [FakeTile(), FakeTile()]

    def paint() -> None:
        tiles[0].terrain_id = 5

    history.apply("Paint", tiles, paint)
    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    model.commit_trigger_edit("Rename trigger", history)

    assert [record.kind for record in history.records] == ["tile", "trigger"]

    # Undo the trigger record: the tile edit must survive it.
    assert history.undo(loaded.map_manager.terrain, model) == []
    assert tiles[0].terrain_id == 5
    assert model.manager().triggers[0].name != "Renamed"

    # Undo the tile record: returns tile indices again.
    assert history.undo(tiles, model) == [0]
    assert tiles[0].terrain_id == 0


def test_dirty_and_saved_semantics_hold_across_a_trigger_record() -> None:
    loaded, model, history = _open()
    assert not history.is_dirty

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    model.commit_trigger_edit("Rename trigger", history)
    assert history.is_dirty

    history.mark_saved()
    assert not history.is_dirty

    history.undo(loaded.map_manager.terrain, model)
    assert history.is_dirty, "undoing past the saved cursor is a dirty document again"

    history.redo(loaded.map_manager.terrain, model)
    assert not history.is_dirty


def test_model_dirtiness_always_has_a_matching_record() -> None:
    """The contract the single-history decision rests on. If a mutation can
    make the model dirty without pushing a record, is_dirty stays False and the
    document closes without a save prompt -- the same class of silent bug the
    write path already produced twice."""
    loaded, model, history = _open()

    for label, begin, mutate in (
        ("content", [0], lambda m: setattr(m.triggers[0], "name", "Renamed")),
        ("variable", [], lambda m: m.add_variable("contract_var")),
    ):
        model.begin_trigger_edit(begin)
        mutate(model.manager())
        model.commit_trigger_edit(label, history)
        assert model.has_edits, label
        assert history.is_dirty, f"{label}: model dirty but history is not"

    while history.can_undo:
        history.undo(loaded.map_manager.terrain, model)
    assert not model.has_edits
    assert not history.is_dirty


def test_a_trigger_record_still_passes_the_alignment_check() -> None:
    """Plan verification item 6. restore() puts fresh deepcopies into the live
    list, so _tracked has to be rebuilt from that list rather than from the
    record, or the next serialize() raises."""
    loaded, model, history = _open()

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    model.commit_trigger_edit("Rename trigger", history)

    history.undo(loaded.map_manager.terrain, model)
    model.serialize()  # must not raise
    history.redo(loaded.map_manager.terrain, model)
    model.serialize()


# -- 4. records survive being replayed --------------------------------------


def test_undoing_twice_to_the_same_record_reproduces_the_same_state() -> None:
    """Trap 5. A record that handed its own Trigger object to the live list has
    that object mutated by the next in-place edit, so the second undo replays
    already-edited state instead of the original."""
    loaded, model, history = _open()
    expected = _fresh_graph_state()

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    model.commit_trigger_edit("Rename trigger", history)

    for attempt in range(3):
        history.undo(loaded.map_manager.terrain, model)
        assert _graph_state(model.manager()) == expected, f"attempt {attempt}"
        assert model.serialize() == _section_bytes(loaded), f"attempt {attempt}"
        history.redo(loaded.map_manager.terrain, model)


def test_a_later_edit_does_not_mutate_an_earlier_record() -> None:
    """The same trap from the other side: the record's snapshot must not be a
    live view of a trigger that a subsequent edit rewrites."""
    loaded, model, history = _open()
    expected = _fresh_graph_state()

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "First edit"
    model.commit_trigger_edit("First", history)

    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Second edit"
    model.commit_trigger_edit("Second", history)

    history.undo(loaded.map_manager.terrain, model)
    assert model.manager().triggers[0].name == "First edit"
    history.undo(loaded.map_manager.terrain, model)
    assert _graph_state(model.manager()) == expected


def test_restore_hands_out_a_copy_not_the_records_own_object() -> None:
    """Trap 5, asserted as the invariant rather than through a consequence.

    Deliberately white-box, because no route through today's API reaches the
    corruption it prevents: the object would have to be handed to the live
    list, mutated in place, and then replayed -- but any edit made after an
    undo truncates the very record that held it, and the id-remap class
    self-heals because ref_ids/trigger_ids are stored as values and rewritten
    on every restore. Verified by mutation: removing the copy breaks no
    behavioural test in this file. The copy stays because it stops being
    defensive the moment an in-place variable rename or a "jump to history
    entry N" replay arrives, both of which are already anticipated.
    """
    loaded, model, history = _open()
    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    record = model.commit_trigger_edit("Rename trigger", history)

    history.undo(loaded.map_manager.terrain, model)
    live = model.manager().triggers[0]
    snapshot_object = record.before.states[0]

    assert snapshot_object is not None, "a declared content edit must be deepcopied"
    assert live is not snapshot_object, (
        "the live list is holding the record's own snapshot object -- the next "
        "in-place edit would rewrite the record's idea of the past"
    )
    assert live.name == snapshot_object.name


# -- 5. the begin/commit pair itself -----------------------------------------


def test_commit_after_a_raising_structural_edit_still_undoes_cleanly() -> None:
    """The contract handed to the UI session, tested rather than traced. A
    structural_edit() that raises is not a no-op: it marks every blob dirty,
    forces _variables_dirty True and rolls the display order back, so the state
    change still needs a record. This is the one path where the model is in a
    deliberately-distrusted state at the moment the record is built."""
    loaded, model, history = _open()

    model.begin_trigger_edit()
    with pytest.raises(Exception):
        model.structural_edit(lambda m: m.reorder_triggers([0, 1, 99]))
    record = model.commit_trigger_edit("Failed reorder", history)

    assert model.has_edits, "a failed structural edit distrusts every blob"
    assert record.touched == []

    history.undo(loaded.map_manager.terrain, model)
    assert not model.has_edits, "undo restores the blobs the failure gave up on"
    assert _graph_state(model.manager()) == _fresh_graph_state()
    assert model.serialize() == _section_bytes(loaded)


def test_begin_twice_raises() -> None:
    _loaded, model, _history = _open()
    model.begin_trigger_edit()
    with pytest.raises(RuntimeError, match="already in progress"):
        model.begin_trigger_edit()


def test_commit_without_begin_raises() -> None:
    _loaded, model, history = _open()
    with pytest.raises(RuntimeError, match="no edit in progress"):
        model.commit_trigger_edit("Nothing", history)


def test_abort_discards_without_recording() -> None:
    _loaded, model, history = _open()
    model.begin_trigger_edit([0])
    model.abort_trigger_edit()
    assert history.records == []
    model.begin_trigger_edit([0])  # the pair is free again


def test_begin_rejects_an_out_of_range_index() -> None:
    _loaded, model, _history = _open()
    with pytest.raises(IndexError):
        model.begin_trigger_edit([model.trigger_count])


def test_a_declared_content_edit_is_marked_dirty_at_its_new_position() -> None:
    """content_touched is given in pre-edit numbering and followed by object
    identity, so a declared trigger that a structural edit also moves is still
    marked at the position it ended up in."""
    loaded, model, history = _open()
    count = model.trigger_count

    model.begin_trigger_edit([0])
    model.structural_edit(
        lambda m: (
            setattr(m.triggers[0], "name", "Renamed and moved"),
            m.reorder_triggers(list(reversed(range(count)))),
        )
    )
    record = model.commit_trigger_edit("Rename and reorder", history)

    moved_to = next(
        i for i, t in enumerate(model.manager().triggers) if t.name == "Renamed and moved"
    )
    assert record.touched == [moved_to]
    assert model.is_dirty(moved_to)

    history.undo(loaded.map_manager.terrain, model)
    assert _graph_state(model.manager()) == _fresh_graph_state()


def test_the_record_carries_its_label_and_kind() -> None:
    _loaded, model, history = _open()
    model.begin_trigger_edit([0])
    model.manager().triggers[0].name = "Renamed"
    record = model.commit_trigger_edit("Rename trigger", history)

    assert isinstance(record, TriggerDiffRecord)
    assert record.label == "Rename trigger"
    assert record.kind == "trigger"
    assert history.peek_undo() is record
    assert record.touched_indices() == [], "trigger records touch no tiles"
