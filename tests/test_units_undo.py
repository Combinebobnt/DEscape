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

from pathlib import Path

import pytest

from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_WALL = 102
_REF_VILLAGER_P1 = 203


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded), EditHistory()


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


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
    loaded, model, history = _open()
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
