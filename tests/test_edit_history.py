"""Gaps in EditHistory coverage tools/verify_write_path.py's
check_history_invariants doesn't reach (that one only exercises
apply()-based undo/redo/no-op/redo-truncation against real scenario tiles).
See descape/edit_history.py's own module docstring for the invariant this
whole module exists to enforce.

Uses a fake-tile protocol (plain objects with terrain_id/elevation/layer)
rather than real AoE2ScenarioParser TerrainTiles, on purpose: EditHistory
only ever reads/writes those three attributes (see tile_state() and
undo()/redo()), and staying duck-typed here is what lets a future widened
protocol (v3.5's unit add/remove/move) arrive as a new
failing test against a wider fake, not a rewrite of every test in this
file.
"""

from __future__ import annotations

import pytest

from descape.edit_history import EditHistory


class FakeTile:
    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


def _paint(tiles, i: int, terrain_id: int):
    def mutate():
        tiles[i].terrain_id = terrain_id

    return mutate


def test_fresh_history_is_not_dirty() -> None:
    assert not EditHistory().is_dirty


def test_apply_marks_dirty() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    assert hist.is_dirty


def test_mark_saved_clears_dirty() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.mark_saved()
    assert not hist.is_dirty


def test_undo_after_save_is_dirty_again() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.mark_saved()
    hist.undo(tiles)
    assert hist.is_dirty


def test_commit_stroke_trims_at_max_records() -> None:
    hist = EditHistory(max_records=3)
    tiles = [FakeTile()]
    for terrain_id in range(1, 6):  # 5 distinct edits, cap is 3
        hist.apply("paint", tiles, _paint(tiles, 0, terrain_id))
    assert len(hist.records) == 3
    assert hist.cursor == 3
    # The two oldest records (terrain_id 1, 2) were dropped -- only the
    # last three edits (3, 4, 5) survive, oldest surviving first.
    assert [r.changes[0][2][0] for r in hist.records] == [3, 4, 5]


def test_overflow_past_saved_cursor_marks_saved_at_cursor_none() -> None:
    hist = EditHistory(max_records=3)
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 1))
    hist.mark_saved()  # saved_at_cursor = 1
    for terrain_id in (2, 3, 4, 5):
        hist.apply("paint", tiles, _paint(tiles, 0, terrain_id))
    # 4 more edits on a max_records=3 history: the 4th and 5th each overflow
    # by 1, shifting saved_at_cursor from 1 -> 0 -> -1. Going negative is
    # what flips it to None -- the saved state was among the evicted
    # records, unreachable from any cursor position now.
    assert hist.saved_at_cursor is None


def test_saved_state_evicted_stays_dirty_regardless_of_cursor() -> None:
    hist = EditHistory(max_records=2)
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 1))
    hist.mark_saved()
    for terrain_id in (2, 3, 4):
        hist.apply("paint", tiles, _paint(tiles, 0, terrain_id))
    assert hist.saved_at_cursor is None
    assert hist.is_dirty
    hist.undo(tiles)
    assert hist.is_dirty  # still dirty -- no cursor position can be "saved" anymore
    hist.redo(tiles)
    assert hist.is_dirty


def test_begin_stroke_twice_raises() -> None:
    hist = EditHistory()
    hist.begin_stroke([FakeTile()])
    with pytest.raises(RuntimeError):
        hist.begin_stroke([FakeTile()])


def test_stroke_dirty_indices_without_stroke_raises() -> None:
    hist = EditHistory()
    with pytest.raises(RuntimeError):
        hist.stroke_dirty_indices([FakeTile()])


def test_commit_stroke_without_stroke_raises() -> None:
    hist = EditHistory()
    with pytest.raises(RuntimeError):
        hist.commit_stroke("paint", [FakeTile()])


def test_abort_stroke_discards_without_recording() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.begin_stroke(tiles)
    tiles[0].terrain_id = 9
    hist.abort_stroke()
    assert hist.records == []
    assert hist.cursor == 0


def test_abort_stroke_allows_a_fresh_begin_stroke() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.begin_stroke(tiles)
    hist.abort_stroke()
    hist.begin_stroke(tiles)  # would raise RuntimeError if abort_stroke left state behind
    assert hist.stroke_dirty_indices(tiles) == []


def test_reset_clears_everything() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.begin_stroke(tiles)  # leave an in-progress stroke too
    hist.reset()
    assert hist.records == []
    assert hist.cursor == 0
    assert hist.saved_at_cursor == 0
    assert not hist.is_dirty
    hist.begin_stroke(tiles)  # would raise if reset() didn't clear _stroke_before
