"""descape.outline_chunks (TASK-031.42): which chunks of a unit-outline overlay
an edit made stale. Qt-free, on fake units with CHUNK_TILES shrunk to 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import outline_chunks, render, unit_pick
from descape.outline_chunks import HeightField, OutlineChunks

_MAP = 40
_VILLAGER = 83
_TOWN_CENTRE = 109
_HOUSE = 70


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch) -> None:
    monkeypatch.setattr(outline_chunks, "CHUNK_TILES", 4)


def _index(units) -> unit_pick.UnitIndex:
    """`units`: (player_id, unit) pairs."""
    index = unit_pick.UnitIndex()
    for player_id, unit in units:
        bounds = render.unit_tile_bounds(unit, _MAP, _MAP)
        assert bounds is not None, unit
        unit_pick._append_entry(index, player_id, unit, bounds)
    return index


def _units():
    return [
        (1, _Unit(10.0, 10.0, _TOWN_CENTRE, 1)),  # own tile (10, 10): chunk (2, 2), rect 8..11
        (1, _Unit(21.0, 5.0, _HOUSE, 2)),  # chunk (5, 1)
        (2, _Unit(2.5, 30.5, _VILLAGER, 3)),  # chunk (0, 7)
        (0, _Unit(33.5, 33.5, _VILLAGER, 4)),  # chunk (8, 8)
    ]


def _plan(planner, units, context=("ctx",), heights=()):
    index = _index(units)
    return planner.plan(index.entries, index, context, heights)


def _primed(units, **kwargs) -> OutlineChunks:
    planner = OutlineChunks()
    first = _plan(planner, units, **kwargs)
    assert first.full
    return planner


def _dirty(plan) -> set:
    assert not plan.full
    return set(plan.dirty) | plan.emptied


def test_a_first_plan_is_full_and_lists_every_chunk_with_its_members() -> None:
    units = _units()
    plan = _plan(OutlineChunks(), units)
    assert plan.full and not plan.emptied
    assert {chunk: [e.unit.reference_id for e in members] for chunk, members in plan.dirty.items()} == {
        (2, 2): [1],
        (5, 1): [2],
        (0, 7): [3],
        (8, 8): [4],
    }


def test_an_unchanged_second_plan_is_empty() -> None:
    units = _units()
    planner = _primed(units)
    assert _dirty(_plan(planner, units)) == set()


def test_an_add_dirties_only_its_chunk() -> None:
    units = _units()
    planner = _primed(units)
    plan = _plan(planner, [*units, (1, _Unit(17.5, 17.5, _VILLAGER, 9))])
    assert _dirty(plan) == {(4, 4)}
    assert [e.unit.reference_id for e in plan.dirty[(4, 4)]] == [9]


def test_a_dirty_chunks_members_come_in_index_order_as_a_full_plan_lists_them() -> None:
    """Qt's cosmetic stroke changes a few pixels with subpath order at low zoom, so an
    incremental chunk must list its members exactly as a full plan does. Here by_tile's
    row scan meets the second-indexed unit first."""
    units = [(1, _Unit(6.5, 6.5, _VILLAGER, 1)), (1, _Unit(5.5, 5.5, _VILLAGER, 2))]
    planner = OutlineChunks()
    full = _plan(planner, units)
    assert [e.unit.reference_id for e in full.dirty[(1, 1)]] == [1, 2]
    units[0][1].x = 6.25
    plan = _plan(planner, units)
    assert [e.unit.reference_id for e in plan.dirty[(1, 1)]] == [1, 2]


def test_a_remove_dirties_its_chunk_and_reports_it_emptied() -> None:
    units = _units()
    planner = _primed(units)
    plan = _plan(planner, units[:-1])
    assert _dirty(plan) == {(8, 8)}
    assert plan.emptied == {(8, 8)}


def test_a_move_within_a_chunk_dirties_that_chunk_only() -> None:
    units = _units()
    planner = _primed(units)
    units[2][1].x = 3.25
    assert _dirty(_plan(planner, units)) == {(0, 7)}


def test_a_move_across_chunks_dirties_both() -> None:
    units = _units()
    planner = _primed(units)
    units[2][1].x = 5.5
    plan = _plan(planner, units)
    assert _dirty(plan) == {(0, 7), (1, 7)}
    assert plan.emptied == {(0, 7)}


def test_a_const_change_dirties_its_chunk() -> None:
    units = _units()
    planner = _primed(units)
    units[1][1].unit_const = _TOWN_CENTRE
    assert _dirty(_plan(planner, units)) == {(5, 1)}


def test_an_owner_change_dirties_its_chunk() -> None:
    units = _units()
    planner = _primed(units)
    units[2] = (3, units[2][1])
    assert _dirty(_plan(planner, units)) == {(0, 7)}


def test_a_stepped_height_change_dirties_the_units_whose_rect_covers_the_tile() -> None:
    """(8, 11) is a corner of the Town Centre's 4x4 rect, not its own tile, and in no other unit's."""
    units = _units()
    elevations = np.zeros((_MAP, _MAP), dtype=np.int32)
    heights = (HeightField(elevations),)
    planner = _primed(units, heights=heights)
    elevations[11, 8] = 2
    assert _dirty(_plan(planner, units, heights=heights)) == {(2, 2)}
    elevations[20, 20] = 1  # under no unit
    assert _dirty(_plan(planner, units, heights=heights)) == set()
    assert _dirty(_plan(planner, units, heights=heights)) == set(), "the snapshot did not advance"


def test_a_sloped_corner_change_dirties_the_units_on_all_four_tiles_around_it() -> None:
    """Corner (16, 16) is where chunks (3,3), (4,3), (3,4) and (4,4) meet; each
    villager's own tile only touches it. The one at (17, 17) does not."""
    units = [
        (1, _Unit(15.5, 15.5, _VILLAGER, 1)),
        (1, _Unit(16.5, 15.5, _VILLAGER, 2)),
        (1, _Unit(15.5, 16.5, _VILLAGER, 3)),
        (1, _Unit(16.5, 16.5, _VILLAGER, 4)),
        (1, _Unit(17.5, 17.5, _VILLAGER, 5)),
        (1, _Unit(21.5, 21.5, _VILLAGER, 6)),
    ]
    corner_rise = np.zeros((_MAP + 1, _MAP + 1), dtype=np.int32)
    fold = np.zeros((_MAP, _MAP), dtype=bool)
    heights = (HeightField(corner_rise, corners=True), HeightField(fold))
    planner = _primed(units, heights=heights)
    corner_rise[16, 16] = 12
    plan = _plan(planner, units, heights=heights)
    assert _dirty(plan) == {(3, 3), (4, 3), (3, 4), (4, 4)}
    assert [e.unit.reference_id for e in plan.dirty[(4, 4)]] == [4, 5], "a dirty chunk lists every member"
    fold[21, 21] = True
    assert _dirty(_plan(planner, units, heights=heights)) == {(5, 5)}


def test_a_corner_on_the_map_edge_maps_only_to_tiles_on_the_map() -> None:
    units = [(1, _Unit(0.5, 0.5, _VILLAGER, 1))]
    corner_rise = np.zeros((_MAP + 1, _MAP + 1), dtype=np.int32)
    heights = (HeightField(corner_rise, corners=True),)
    planner = _primed(units, heights=heights)
    corner_rise[0, 0] = 5
    corner_rise[_MAP, _MAP] = 5
    assert _dirty(_plan(planner, units, heights=heights)) == {(0, 0)}


def test_a_height_change_under_an_out_of_scope_unit_dirties_nothing() -> None:
    """by_tile holds every indexed unit; only the overlay's own entries count."""
    units = _units()
    index = _index(units)
    elevations = np.zeros((_MAP, _MAP), dtype=np.int32)
    heights = (HeightField(elevations),)
    planner = OutlineChunks()
    in_scope = index.entries[1:]
    assert planner.plan(in_scope, index, (), heights).full
    elevations[10, 10] = 3  # the Town Centre's own tile
    plan = planner.plan(in_scope, index, (), heights)
    assert _dirty(plan) == set()


@pytest.mark.parametrize("field", range(7))
def test_each_context_field_changed_alone_dirties_everything(field: int) -> None:
    """MapView's context: style, tile px, map w, map h, projection, drape, dataset generation."""
    units = _units()
    context = ["stepped", 64, _MAP, _MAP, ("proj", 1), False, 3]
    planner = _primed(units, context=tuple(context))
    context[field] = ("changed", context[field])
    plan = _plan(planner, units, context=tuple(context))
    assert plan.full and set(plan.dirty) == {(2, 2), (5, 1), (0, 7), (8, 8)}
    assert _dirty(_plan(planner, units, context=tuple(context))) == set()


def test_a_height_field_changing_shape_or_kind_dirties_everything() -> None:
    units = _units()
    planner = _primed(units, heights=(HeightField(np.zeros((_MAP, _MAP))),))
    assert _plan(planner, units, heights=(HeightField(np.zeros((_MAP + 1, _MAP + 1)), corners=True),)).full
    assert _plan(planner, units, heights=()).full


def test_changing_chunk_tiles_dirties_everything(monkeypatch) -> None:
    units = _units()
    planner = _primed(units)
    monkeypatch.setattr(outline_chunks, "CHUNK_TILES", 8)
    assert _plan(planner, units).full


def test_reset_dirties_everything() -> None:
    units = _units()
    planner = _primed(units)
    planner.reset()
    assert _plan(planner, units).full


def test_a_duplicate_key_at_two_places_draws_both_and_diffs_each_copy() -> None:
    """A corrupt file's player 2 id 3 twice, in chunks (0, 7) and (3, 7)."""
    units = [*_units(), (2, _Unit(13.5, 30.5, _VILLAGER, 3))]
    planner = OutlineChunks()
    plan = _plan(planner, units)
    assert plan.full
    assert [e.unit.x for e in plan.dirty[(0, 7)]] == [2.5]
    assert [e.unit.x for e in plan.dirty[(3, 7)]] == [13.5]
    assert _dirty(_plan(planner, units)) == set()
    units[-1][1].x = 17.5
    plan = _plan(planner, units)
    assert _dirty(plan) == {(3, 7), (4, 7)}
    assert [e.unit.x for e in plan.dirty[(4, 7)]] == [17.5]
    plan = _plan(planner, units[:-1])
    assert _dirty(plan) == {(4, 7)}, "the dropped copy's chunk was not cleared"
    assert plan.emptied == {(4, 7)}


def test_an_exact_duplicate_dirties_its_chunk_every_pass_and_once_after_a_copy_goes() -> None:
    """A set of records cannot count copies, so a chunk holding two equal ones is never trusted."""
    units = [*_units(), (2, _Unit(2.5, 30.5, _VILLAGER, 3))]
    planner = _primed(units)
    plan = _plan(planner, units)
    assert _dirty(plan) == {(0, 7)}
    assert [e.unit.reference_id for e in plan.dirty[(0, 7)]] == [3, 3], "both copies draw"
    assert _dirty(_plan(planner, units)) == {(0, 7)}
    plan = _plan(planner, units[:-1])
    assert _dirty(plan) == {(0, 7)}
    assert [e.unit.x for e in plan.dirty[(0, 7)]] == [2.5]
    assert _dirty(_plan(planner, units[:-1])) == set()


def test_the_in_scope_check_reads_the_same_record_the_diff_does() -> None:
    """plan() spells the record out for speed and checks membership through
    _RECORD: if the two drift, every height-dirtied and gathered member vanishes."""
    units = _units()
    index = _index(units)
    planner = OutlineChunks()
    planner.plan(index.entries, index, ())
    assert planner._records == {outline_chunks._RECORD(e) for e in index.entries}


def test_chunk_of_reads_the_live_chunk_size() -> None:
    assert outline_chunks.chunk_of(9, 3) == (2, 0)
