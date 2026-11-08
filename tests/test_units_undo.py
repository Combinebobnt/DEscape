"""Phase 3.5a's undo/redo: descape/unit_model.py's snapshot/restore pair and
the fourth record type in descape/edit_history.py.

The claims under test, mirroring tests/test_trigger_undo.py's own ordering:

1. **A unit undo is byte-clean.** Edit, undo, save, and the file must be
   byte-identical to the one that was opened.
2. **The insertion-position rule.** A reassign's naive inverse (remove from
   destination, append to source) restores ownership but not position;
   PlayerListSnapshot captures both affected player lists in full so undo
   restores the exact original index.
3. **One history, four record types.** Tile, trigger, option, and unit
   records interleave on a single EditHistory.
4. **Records survive being replayed.** Undo, redo, undo again reproduces the
   same state each time.

Every restore trap these exercise is documented on UnitEditModel.restore().
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel, merged_field_record

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_TREE_OAK = 100
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded), EditHistory()


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _index_of(loaded, player: int, unit) -> int:
    """By identity, never list.index(). Position is the whole claim these
    tests make, so a value-equal neighbour must not answer for it."""
    return next(i for i, u in enumerate(loaded.unit_manager.units[player]) if u is unit)


# -- 1. a unit undo is byte-clean ---------------------------------------------


def test_edit_undo_save_is_byte_identical_to_the_original_file(tmp_path: Path) -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)

    model.begin_unit_edit([1])
    model.set_position(villager, 40.5, 40.5, 5.0)
    model.commit_unit_edit("Move villager", history)
    assert model.has_edits

    history.undo([], None, None, model)

    assert not model.has_edits, "undo must restore blob cleanliness, not just content"
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_restores_the_unit_content_itself() -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    original = (villager.x, villager.y, villager.z)

    model.begin_unit_edit([1])
    model.set_position(villager, 40.5, 40.5, 5.0)
    model.commit_unit_edit("Move villager", history)

    history.undo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == original


def test_undo_restores_rotation_on_the_live_unit_too(tmp_path: Path) -> None:
    """Rotation is part of UnitState for this reason: without it the blob
    comes back (so the FILE is right) while the live Unit object stays
    rotated, and the inspector and render then disagree with what saving
    would write. Both halves are asserted here. `unit_const` is in UnitState
    for the same reason; see the gate cycle test below."""
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    original = archer.rotation

    model.begin_unit_edit([1])
    model.set_rotation(archer, original + 1.25)
    model.commit_unit_edit("Rotate unit", history)

    history.undo([], None, None, model)

    assert archer.rotation == original
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_restores_a_cycled_variant_on_the_live_unit(tmp_path: Path) -> None:
    loaded, model, history = _open()
    oak = _unit(loaded, _REF_TREE_OAK)
    original = oak.rotation

    model.begin_unit_edit([0])
    model.set_variant(oak, 12.0)
    model.commit_unit_edit("Cycle unit variant", history)

    history.undo([], None, None, model)

    assert oak.rotation == original
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_restores_a_cycled_gates_const_and_anchor_on_the_live_unit(tmp_path: Path) -> None:
    """The other half of the UnitState widening. A gate cycle changes the
    const AND the coordinates (each orientation has its own span), so an undo
    that restored only the bytes would leave a live unit whose footprint no
    longer matches the file. The fixture holds no gate, so the wall is
    re-pointed at one first. That is a plain field edit, invisible to the
    bytes until a commit (finding 4), which is why the byte round-trip holds.
    """
    loaded, model, history = _open()
    gate = _unit(loaded, _REF_WALL)
    wall_fields = (gate.unit_const, gate.x, gate.y)
    gate.unit_const, gate.x, gate.y = 64, 10.0, 5.5
    original = (gate.unit_const, gate.x, gate.y)

    model.begin_unit_edit([0])
    model.set_unit_const(gate, 659)
    model.commit_unit_edit("Rotate unit", history)
    assert (gate.unit_const, gate.x, gate.y) == (659, 10.0, 7.0)

    history.undo([], None, None, model)

    assert (gate.unit_const, gate.x, gate.y) == original
    gate.unit_const, gate.x, gate.y = wall_fields
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_restores_an_unloaded_units_link_on_the_live_unit(tmp_path: Path) -> None:
    """GH #115's UnitState widening: set_garrisoned_in() writes the link in
    place, so an undo that brought back only the blob would leave the live
    villager unloaded while the file says it is inside the House."""
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)

    model.begin_unit_edit([1])
    model.set_garrisoned_in(villager, -1)
    model.commit_unit_edit("Unload units", history)
    assert villager.garrisoned_in_id == -1

    history.undo([], None, None, model)

    assert villager.garrisoned_in_id == _REF_HOUSE
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_undo_returns_no_tile_indices() -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 1.5, 0.0)
    model.commit_unit_edit("Move", history)

    assert history.undo([], None, None, model) == []
    assert history.redo([], None, None, model) == []


def test_a_unit_record_without_a_model_raises_before_moving_the_cursor() -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 1.5, 0.0)
    model.commit_unit_edit("Move", history)

    with pytest.raises(RuntimeError):
        history.undo([])
    assert history.can_undo, "a refused undo must leave the cursor exactly where it was"


def test_add_then_undo_restores_next_unit_id() -> None:
    """Plan verification item 19."""
    _loaded, model, history = _open()
    before_next = model.next_unit_id

    model.begin_unit_edit([2])
    model.add(player=2, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)
    model.commit_unit_edit("Add unit", history)
    assert model.next_unit_id == before_next + 1

    history.undo([], None, None, model)
    assert model.next_unit_id == before_next


# -- 2. the insertion-position rule -------------------------------------------


def test_reassign_then_undo_is_byte_identical(tmp_path: Path) -> None:
    """Plan verification item 8: the naive inverse restores ownership but not
    position; PlayerListSnapshot must restore both."""
    loaded, model, history = _open()
    wall = _unit(loaded, _REF_WALL)
    original_index = loaded.unit_manager.units[0].index(wall)

    model.begin_unit_edit([0, 1])
    model.reassign(wall, 1)
    model.commit_unit_edit("Reassign wall", history)

    history.undo([], None, None, model)

    assert wall in loaded.unit_manager.units[0]
    assert loaded.unit_manager.units[0].index(wall) == original_index
    assert wall not in loaded.unit_manager.units[1]
    assert wall._player == 0

    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_bulk_delete_undo_redo_restores_both_players_exactly(tmp_path: Path) -> None:
    """remove_many() rewrites whole lists across every player holding a
    target, not just the ones begin_unit_edit() named, and it removes from
    mid-list so later units shift down. Undo has to put every one of them
    back at its exact original index, and the model's own derived position
    map with it. The write below goes through _check_alignment(), which is
    what turns a missed rebuild into a failure rather than a wrong save.
    """
    loaded, model, history = _open()
    house = _unit(loaded, _REF_HOUSE)
    archer_p1 = _unit(loaded, _REF_ARCHER_P1)
    villager_p1 = _unit(loaded, _REF_VILLAGER_P1)
    archer_p2 = _unit(loaded, _REF_ARCHER_P2)
    villager_p2 = _unit(loaded, _REF_VILLAGER_P2)
    placed = ((1, house), (1, archer_p1), (1, villager_p1), (2, archer_p2), (2, villager_p2))
    before = {id(u): _index_of(loaded, p, u) for p, u in placed}
    house_pos = (house.x, house.y, house.z)
    assert before[id(archer_p1)] == 1, "the target must be mid-list, or nothing downstream shifts"

    model.begin_unit_edit([1, 2])
    model.remove_many([archer_p1, archer_p2])
    model.commit_unit_edit("Delete 2 units", history)

    assert _index_of(loaded, 1, villager_p1) == 1, "the delete must actually have shifted a later unit"
    assert _index_of(loaded, 2, villager_p2) == 0
    assert model.has_edits

    history.undo([], None, None, model)

    for player, unit in placed:
        assert _index_of(loaded, player, unit) == before[id(unit)]
    assert not model.has_edits
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()

    history.redo([], None, None, model)

    assert archer_p1 not in loaded.unit_manager.units[1]
    assert archer_p2 not in loaded.unit_manager.units[2]
    assert _index_of(loaded, 1, villager_p1) == 1

    # A survivor must still resolve to itself after all that list churn: a
    # stale position entry here would write the house's blob instead.
    model.begin_unit_edit([1])
    model.set_position(villager_p1, 60.5, 61.5, 2.0)
    model.commit_unit_edit("Move villager", history)
    assert (villager_p1.x, villager_p1.y, villager_p1.z) == (60.5, 61.5, 2.0)
    assert (house.x, house.y, house.z) == house_pos


def test_reassign_redo_moves_it_again() -> None:
    loaded, model, history = _open()
    wall = _unit(loaded, _REF_WALL)

    model.begin_unit_edit([0, 1])
    model.reassign(wall, 1)
    model.commit_unit_edit("Reassign wall", history)

    history.undo([], None, None, model)
    history.redo([], None, None, model)

    assert wall in loaded.unit_manager.units[1]
    assert wall not in loaded.unit_manager.units[0]
    assert wall._player == 1


# -- 3. one history, four record types -----------------------------------------


def test_unit_records_interleave_with_tile_records() -> None:
    loaded, model, history = _open()
    tiles = loaded.map_manager.terrain

    def paint():
        tiles[0].terrain_id = 15 if tiles[0].terrain_id != 15 else 2

    history.apply("paint", tiles, paint)

    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 1.5, 0.0)
    model.commit_unit_edit("Move villager", history)

    assert len(history.records) == 2
    assert history.records[0].kind == "tile"
    assert history.records[1].kind == "unit"

    history.undo(tiles, None, None, model)
    history.undo(tiles, None, None, model)
    assert not history.can_undo


def test_is_dirty_tracks_the_cursor_across_a_unit_edit() -> None:
    loaded, model, history = _open()
    assert not history.is_dirty
    villager = _unit(loaded, _REF_VILLAGER_P1)

    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 1.5, 0.0)
    model.commit_unit_edit("Move villager", history)
    assert history.is_dirty

    history.undo([], None, None, model)
    assert not history.is_dirty

    history.redo([], None, None, model)
    assert history.is_dirty

    history.mark_saved()
    assert not history.is_dirty


# -- 4. records survive being replayed -----------------------------------------


def test_undo_redo_undo_reproduces_the_same_state_each_time() -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    original = (villager.x, villager.y, villager.z)

    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 2.5, 3.0)
    model.commit_unit_edit("Move", history)
    moved = (villager.x, villager.y, villager.z)

    history.undo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == original
    history.redo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == moved
    history.undo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == original


def test_a_second_edit_after_undo_does_not_mutate_the_first_records_snapshot() -> None:
    """Trap 5's unit-side equivalent: a record handed a live object reference
    would have its "before" state overwritten by a later edit to the same
    unit, corrupting a still-live undo record."""
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    original = (villager.x, villager.y, villager.z)

    model.begin_unit_edit([1])
    model.set_position(villager, 1.5, 1.5, 0.0)
    model.commit_unit_edit("Move 1", history)

    history.undo([], None, None, model)

    model.begin_unit_edit([1])
    model.set_position(villager, 9.5, 9.5, 0.0)
    model.commit_unit_edit("Move 2", history)

    history.undo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == original


# -- 5. a compass drag folds its spokes into one record (TASK-548) -------------


def _turn(model, history, units, rotation: float, label: str):
    model.begin_unit_edit([1, 2], fields_only=True)
    for unit in units:
        model.set_rotation(unit, rotation)
    return model.commit_unit_edit(label, history, push=False)


def test_merged_field_record_keeps_the_first_before_and_the_last_after() -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    original = archer.rotation

    older = _turn(model, history, [archer], 1.0, "first")
    newer = _turn(model, history, [archer], 2.0, "second")
    history.push_unit_record(merged_field_record(older, newer))

    assert history.peek_undo().label == "second"
    assert len(history.peek_undo().unit_field_entries) == 1
    history.undo([], None, None, model)
    assert archer.rotation == original
    history.redo([], None, None, model)
    assert archer.rotation == 2.0


def test_merged_field_record_keeps_a_key_only_one_side_touched_in_matching_order() -> None:
    loaded, model, history = _open()
    archer, other = _unit(loaded, _REF_ARCHER_P1), _unit(loaded, _REF_ARCHER_P2)
    originals = (archer.rotation, other.rotation)

    older = _turn(model, history, [archer], 1.0, "first")
    newer = _turn(model, history, [other], 2.0, "second")
    merged = merged_field_record(older, newer)
    history.push_unit_record(merged)

    assert list(merged.before.entries) == list(merged.after.entries)
    history.undo([], None, None, model)
    assert (archer.rotation, other.rotation) == originals
    history.redo([], None, None, model)
    assert (archer.rotation, other.rotation) == (1.0, 2.0)


def test_merged_field_record_carries_the_clean_flag_of_the_first_before() -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    assert not model.has_edits

    older = _turn(model, history, [archer], 1.0, "first")
    newer = _turn(model, history, [archer], 2.0, "second")
    history.push_unit_record(merged_field_record(older, newer))

    assert model.has_edits
    history.undo([], None, None, model)
    assert not model.has_edits


# -- 6. records stay cheap for every later full gc (TASK-031.18) --------------


def _add_gaia(loaded, model, history, x: float = 30.5) -> object:
    oak = _unit(loaded, _REF_TREE_OAK)
    model.begin_unit_edit([0])
    unit = model.add(player=0, unit_const=oak.unit_const, x=x, y=30.5, z=0.0, rotation=oak.rotation)
    model.commit_unit_edit("Add tree", history)
    return unit


def test_whole_list_snapshots_hold_untracked_blob_and_state_tuples() -> None:
    """A list is walked by every full collection, an untracked tuple never:
    blobs are bytes or None, so one collect untracks them. States are tuples
    of atoms inside a tuple: the first pass untracks the inner ones only (its
    reachability pass moves them behind the outer), the second the outer.
    `units` holds tracked Units, so it stays tracked."""
    loaded, model, history = _open()
    _add_gaia(loaded, model, history)
    model.begin_unit_edit([1, 2])
    model.reassign(_unit(loaded, _REF_ARCHER_P1), 2)
    model.commit_unit_edit("Reassign archer", history)
    sides = [side for record in history.records for side in (record.before, record.after)]
    snapshots = [pls for side in sides for pls in side.players.values()]
    assert len(snapshots) == 6

    gc.collect()
    for pls in snapshots:
        assert type(pls.units) is tuple and type(pls.blobs) is tuple and type(pls.states) is tuple
        assert pls.blobs and pls.states
        assert not gc.is_tracked(pls.blobs), "blob slots must not cost a later full collection anything"
    gc.collect()
    for pls in snapshots:
        assert not gc.is_tracked(pls.states), "unit states must not cost a later full collection anything"


def test_consecutive_gaia_edits_share_one_units_tuple() -> None:
    """Record 2's before is record 1's after, so the history holds that list
    once, not twice."""
    loaded, model, history = _open()
    _add_gaia(loaded, model, history)
    model.begin_unit_edit([0])
    model.remove(_unit(loaded, _REF_TREE_OAK))
    model.commit_unit_edit("Delete tree", history)

    first, second = history.records
    assert second.before.players[0].units is first.after.players[0].units
    assert second.after.players[0].units is not second.before.players[0].units


def test_a_record_whose_membership_did_not_change_holds_one_units_tuple() -> None:
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.begin_unit_edit([1])
    model.set_position(villager, 40.5, 40.5, 5.0)
    record = model.commit_unit_edit("Move villager", history)

    assert record.before.players[1].units is record.after.players[1].units
    assert record.before.players[1].states != record.after.players[1].states


def test_a_list_changed_outside_an_edit_is_captured_fresh() -> None:
    """The shared tuple is reused only while the live list holds the same
    units in order. A capture that kept a stale one would undo the next edit
    to a list without the unit added in between."""
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.begin_unit_edit([1])
    model.set_position(villager, 40.5, 40.5, 5.0)
    model.commit_unit_edit("Move villager", history)
    added = model.add(player=1, unit_const=83, x=50.5, y=50.5, z=0.0, rotation=0.0)

    model.begin_unit_edit([1])
    model.set_position(villager, 41.5, 41.5, 5.0)
    model.commit_unit_edit("Move villager", history)
    history.undo([], None, None, model)

    assert any(u is added for u in loaded.unit_manager.units[1])
    assert (villager.x, villager.y) == (40.5, 40.5)
    model.serialize()


def test_an_undo_then_a_new_edit_shares_the_restored_tuple() -> None:
    loaded, model, history = _open()
    _add_gaia(loaded, model, history)
    restored = history.records[0].before.players[0].units
    history.undo([], None, None, model)
    _add_gaia(loaded, model, history, x=31.5)

    (record,) = history.records
    assert record.before.players[0].units is restored


def test_a_convert_that_captured_all_nine_keeps_only_the_two_it_touched(tmp_path: Path) -> None:
    """Convert declares every player at press. The record keeps the source
    and destination, and still restores the exact index both ways."""
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    original_index = _index_of(loaded, 1, archer)
    assert 0 < original_index < len(loaded.unit_manager.units[1]) - 1, "the archer must be mid-list"

    model.begin_unit_edit(list(range(9)))
    model.reassign(archer, 2)
    record = model.commit_unit_edit("Convert unit", history)

    assert set(record.before.players) == set(record.after.players) == {1, 2}
    history.undo([], None, None, model)
    assert _index_of(loaded, 1, archer) == original_index
    assert not any(u is archer for u in loaded.unit_manager.units[2])
    assert archer._player == 1
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()

    history.redo([], None, None, model)
    assert loaded.unit_manager.units[2][-1] is archer
    assert not any(u is archer for u in loaded.unit_manager.units[1])
    assert archer._player == 2


def test_a_state_only_change_with_all_nine_declared_keeps_that_player() -> None:
    """Membership alone does not make a player unchanged: a moved unit's
    list keeps its units tuple, and the record must keep it for its state."""
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P2)
    original = (villager.x, villager.y, villager.z)

    model.begin_unit_edit(list(range(9)))
    model.set_position(villager, 40.5, 40.5, 5.0)
    record = model.commit_unit_edit("Move villager", history)

    assert set(record.after.players) == {2}
    history.undo([], None, None, model)
    assert (villager.x, villager.y, villager.z) == original
    assert not model.has_edits


def test_a_same_value_write_keeps_its_player_for_the_blob_slot(tmp_path: Path) -> None:
    """A write of the value a unit already holds changes no state but drops
    its blob slot, which undo must put back so the save stays verbatim."""
    loaded, model, history = _open()
    villager = _unit(loaded, _REF_VILLAGER_P2)

    model.begin_unit_edit(list(range(9)))
    model.set_position(villager, villager.x, villager.y, villager.z)
    record = model.commit_unit_edit("Move villager", history)

    assert set(record.after.players) == {2}
    history.undo([], None, None, model)
    assert all(blob is not None for blob in model._blobs[2])
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()
