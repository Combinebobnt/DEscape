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

import gc
import inspect
import weakref

import pytest

from descape.edit_history import (
    CompositeDiffRecord,
    DiffRecord,
    EditHistory,
    TileDiffRecord,
    TriggerDiffRecord,
    UnitDiffRecord,
)


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


def _stroke_both_ways(tiles, mutate, touched):
    """The same stroke recorded with and without `touched`, from the same
    start state; returns (full, fast)."""
    start = [(t.terrain_id, t.elevation, t.layer) for t in tiles]
    records = []
    for arg in (None, touched):
        for t, (terrain_id, elevation, layer) in zip(tiles, start, strict=True):
            t.terrain_id, t.elevation, t.layer = terrain_id, elevation, layer
        hist = EditHistory()
        hist.begin_stroke(tiles)
        mutate()
        records.append(hist.build_stroke_record("paint", tiles, arg))
    return records


def test_a_touched_set_records_exactly_the_full_diff() -> None:
    """Includes a tile changed then restored within the stroke (index 3),
    which must not appear, and one written with its own value (index 5)."""
    tiles = [FakeTile(terrain_id=i % 3, elevation=i % 2) for i in range(12)]

    def mutate():
        tiles[9].terrain_id = 7
        tiles[3].elevation = 4
        tiles[1].layer = 2
        tiles[3].elevation = 1
        tiles[5].terrain_id = tiles[5].terrain_id

    full, fast = _stroke_both_ways(tiles, mutate, {9, 3, 1, 5})
    assert [i for i, _o, _n in full.changes] == [1, 9]
    assert fast.changes == full.changes


def test_a_touched_set_with_only_restored_tiles_records_nothing() -> None:
    tiles = [FakeTile() for _ in range(4)]

    def mutate():
        tiles[2].terrain_id = 6
        tiles[2].terrain_id = 0

    assert _stroke_both_ways(tiles, mutate, {2}) == [None, None]


def test_no_touched_set_falls_back_to_the_full_diff() -> None:
    """Region paste, fill and mirror pass no set: every changed tile must
    still land in the record, or undoing them silently loses tiles."""
    tiles = [FakeTile() for _ in range(6)]
    hist = EditHistory()
    hist.begin_stroke(tiles)
    tiles[0].terrain_id = 1
    tiles[5].elevation = 2
    assert hist.commit_stroke("paste", tiles) == [0, 5]
    hist.undo(tiles)
    assert (tiles[0].terrain_id, tiles[5].elevation) == (0, 0)


def test_an_empty_touched_set_is_not_the_fallback() -> None:
    """[] means "wrote nothing", not "unknown": only None scans the map."""
    tiles = [FakeTile()]
    hist = EditHistory()
    hist.begin_stroke(tiles)
    tiles[0].terrain_id = 1
    assert hist.build_stroke_record("paint", tiles, []) is None


def test_a_scoped_stroke_with_captures_records_exactly_the_full_diff() -> None:
    """Scope {2, 3, 4} up front, capture 9 and 10 just before writing them
    (10 twice: the first capture must win). Index 3 is written then restored,
    4 is scoped but never written."""
    tiles = [FakeTile(terrain_id=i % 3, elevation=i % 2) for i in range(12)]
    start = [(t.terrain_id, t.elevation, t.layer) for t in tiles]
    hist = EditHistory()
    hist.begin_stroke(tiles, indices=[2, 3, 4])
    tiles[2].terrain_id = 7
    tiles[3].elevation = 5
    tiles[3].elevation = start[3][1]
    hist.stroke_capture(tiles, 10)
    tiles[10].elevation = 3
    hist.stroke_capture(tiles, 9)
    tiles[9].layer = 1
    hist.stroke_capture(tiles, 10)
    tiles[10].elevation = 4
    assert hist.stroke_dirty_indices(tiles) == [2, 9, 10]
    assert hist.stroke_start_state(10) == start[10]
    record = hist.build_stroke_record("paste", tiles)
    end = [(t.terrain_id, t.elevation, t.layer) for t in tiles]
    oracle = [(i, start[i], end[i]) for i in range(len(tiles)) if end[i] != start[i]]
    assert [i for i, _o, _n in oracle] == [2, 9, 10]
    assert record.changes == oracle
    assert not hist.in_stroke


def test_stroke_capture_raises_on_an_unscoped_stroke_and_with_no_stroke() -> None:
    tiles = [FakeTile() for _ in range(3)]
    hist = EditHistory()
    with pytest.raises(RuntimeError, match="no stroke"):
        hist.stroke_capture(tiles, 0)
    hist.begin_stroke(tiles)
    with pytest.raises(RuntimeError, match="unscoped"):
        hist.stroke_capture(tiles, 0)


def test_stroke_new_dirty_on_a_scoped_stroke_reads_the_capture_only_when_seen_has_none() -> None:
    """Tile 2 was never captured but is in `seen`, so it compares against that
    and must not reach for the absent capture."""
    tiles = [FakeTile() for _ in range(4)]
    hist = EditHistory()
    hist.begin_stroke(tiles, indices=[1])
    seen = {2: (0, 0, -1)}
    tiles[1].terrain_id = 5
    tiles[2].terrain_id = 6
    assert hist.stroke_new_dirty([1, 2], tiles, seen) == {1, 2}
    assert seen == {1: (5, 0, -1), 2: (6, 0, -1)}


def test_stroke_new_dirty_raises_on_a_scoped_index_neither_seen_nor_captured() -> None:
    tiles = [FakeTile() for _ in range(4)]
    hist = EditHistory()
    hist.begin_stroke(tiles, indices=[1])
    tiles[3].terrain_id = 2
    with pytest.raises(RuntimeError, match="never captured"):
        hist.stroke_new_dirty([3], tiles, {})


def test_a_touched_set_on_a_scoped_stroke_raises() -> None:
    """The captures are the write set; a `touched` alongside them could only
    disagree with it."""
    tiles = [FakeTile() for _ in range(3)]
    hist = EditHistory()
    hist.begin_stroke(tiles, indices=[0])
    tiles[0].terrain_id = 1
    with pytest.raises(ValueError, match="scoped"):
        hist.build_stroke_record("paste", tiles, [0])


def test_a_scoped_tile_captured_then_restored_records_nothing() -> None:
    tiles = [FakeTile() for _ in range(4)]
    hist = EditHistory()
    hist.begin_stroke(tiles, indices=[1])
    tiles[1].terrain_id = 6
    hist.stroke_capture(tiles, 3)
    tiles[3].elevation = 2
    tiles[1].terrain_id = 0
    tiles[3].elevation = 0
    assert hist.build_stroke_record("paste", tiles) is None
    assert not hist.in_stroke


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


# -- the second record type -------------------------------------------------
#
# Still duck-typed, and deliberately only one method wide: restore() is the
# single entry point EditHistory uses on the trigger side, which is what keeps
# this module free of AoE2ScenarioParser. A fake needing five private
# attributes would mean edit_history.py had reached into the model itself.
# Behaviour against the real library lives in tests/test_trigger_undo.py.


class FakeTriggerModel:
    def __init__(self):
        self.restored: list[str] = []

    def restore(self, snapshot) -> None:
        self.restored.append(snapshot)


def _trigger_record(label: str = "trigger edit") -> TriggerDiffRecord:
    return TriggerDiffRecord(label, before=f"{label}:before", after=f"{label}:after", touched=[3])


def test_trigger_record_undo_restores_the_before_snapshot() -> None:
    hist = EditHistory()
    model = FakeTriggerModel()
    hist.push_trigger_record(_trigger_record())

    assert hist.undo([], model) == [], "trigger records yield no tile indices"
    assert model.restored == ["trigger edit:before"]
    assert hist.redo([], model) == []
    assert model.restored == ["trigger edit:before", "trigger edit:after"]


def test_trigger_record_without_a_model_raises_before_moving_the_cursor() -> None:
    hist = EditHistory()
    hist.push_trigger_record(_trigger_record())
    with pytest.raises(RuntimeError, match="no TriggerEditModel"):
        hist.undo([])
    assert hist.cursor == 1, "a refused undo must not move the cursor"
    assert hist.can_undo


def test_pushing_a_trigger_record_marks_the_document_dirty() -> None:
    hist = EditHistory()
    hist.push_trigger_record(_trigger_record())
    assert hist.is_dirty
    hist.mark_saved()
    assert not hist.is_dirty


def test_a_trigger_push_truncates_the_redo_tail() -> None:
    """The shared _push() path, reached from the trigger side: pushing while
    the cursor is behind the tip must discard the records after it, exactly as
    commit_stroke does."""
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.apply("paint", tiles, _paint(tiles, 0, 6))
    hist.undo(tiles)
    assert hist.can_redo

    hist.push_trigger_record(_trigger_record())
    assert not hist.can_redo
    assert [record.kind for record in hist.records] == ["tile", "trigger"]


def test_a_trigger_push_overflows_the_cap_like_a_tile_push() -> None:
    """_push()'s saved_at_cursor arithmetic is the part a second push path
    would get subtly wrong -- the symptom (a file that stops reading as dirty)
    surfaces nowhere near the bug."""
    hist = EditHistory(max_records=3)
    for i in range(5):
        hist.push_trigger_record(_trigger_record(f"edit {i}"))

    assert len(hist.records) == 3
    assert hist.cursor == 3
    assert [record.label for record in hist.records] == ["edit 2", "edit 3", "edit 4"]
    assert hist.saved_at_cursor is None, "the saved state fell off the front"
    assert hist.is_dirty


def test_mixed_records_keep_their_own_undo_behaviour() -> None:
    hist = EditHistory()
    model = FakeTriggerModel()
    tiles = [FakeTile()]

    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.push_trigger_record(_trigger_record())

    assert hist.peek_undo().kind == "trigger"
    assert hist.undo(tiles, model) == []
    assert tiles[0].terrain_id == 5, "the tile edit must survive a trigger undo"

    assert hist.peek_undo().kind == "tile"
    assert hist.undo(tiles, model) == [0]
    assert tiles[0].terrain_id == 0
    assert model.restored == ["trigger edit:before"]


def test_peek_does_not_move_the_cursor() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    assert hist.peek_undo() is None
    assert hist.peek_redo() is None

    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    assert hist.peek_undo().label == "paint"
    assert hist.peek_redo() is None
    assert hist.cursor == 1

    hist.undo(tiles)
    assert hist.peek_undo() is None
    assert hist.peek_redo().label == "paint"
    assert hist.cursor == 0


# -- CompositeDiffRecord (phase 2.8's one-undo-step region paste) -----------
#
# FakeUnitModel mirrors FakeTriggerModel above: only the one method
# UnitDiffRecord.undo()/redo() actually calls, so this stays duck-typed and
# free of AoE2ScenarioParser like the rest of this file.


class FakeUnitModel:
    def __init__(self):
        self.restored: list[str] = []

    def restore(self, snapshot) -> None:
        self.restored.append(snapshot)


def _unit_record(label: str = "unit edit") -> UnitDiffRecord:
    return UnitDiffRecord(label, before=f"{label}:before", after=f"{label}:after")


def test_composite_kinds_unions_its_children() -> None:
    composite = CompositeDiffRecord("paste", children=[_tile_record(), _unit_record()])
    assert composite.kinds() == frozenset({"tile", "unit"})


def test_composite_require_target_checks_every_child_before_any_undo() -> None:
    tiles = [FakeTile(terrain_id=5)]
    composite = CompositeDiffRecord("paste", children=[_tile_record(), _unit_record()])
    with pytest.raises(RuntimeError, match="no UnitEditModel"):
        composite.require_target(tiles, None, None, None)
    assert tiles[0].terrain_id == 5, "a refused require_target must not have run any child's undo"


def test_composite_undo_reverses_children_redo_replays_forward() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    model = FakeUnitModel()
    hist.begin_stroke(tiles)
    tiles[0].terrain_id = 7
    tile_record = hist.build_stroke_record("paste", tiles)
    unit_record = _unit_record()
    composite = CompositeDiffRecord("paste", children=[tile_record, unit_record])
    hist.push_composite_record(composite)

    assert hist.peek_undo().kind == "composite"
    assert hist.undo(tiles, None, None, model) == [0]
    assert tiles[0].terrain_id == 0
    assert model.restored == ["unit edit:before"]

    assert hist.redo(tiles, None, None, model) == [0]
    assert tiles[0].terrain_id == 7
    assert model.restored == ["unit edit:before", "unit edit:after"]


def test_pushing_an_empty_composite_raises() -> None:
    hist = EditHistory()
    with pytest.raises(ValueError, match="no children"):
        hist.push_composite_record(CompositeDiffRecord("paste", children=[]))
    assert hist.records == []


def _tile_record() -> TileDiffRecord:
    return TileDiffRecord("paint", changes=[(0, (0, 0, -1), (5, 0, -1))])


def test_every_diffrecord_subclass_accepts_the_four_parameter_shape() -> None:
    """Threading `units` as a fourth target through
    require_target/undo/redo means a missed subclass
    fails at runtime, not at import -- this is what actually catches it, and
    it survives a fifth record kind arriving later without another edit
    here."""
    subclasses = DiffRecord.__subclasses__()
    assert UnitDiffRecord in subclasses
    assert TriggerDiffRecord in subclasses
    for cls in subclasses:
        for name in ("require_target", "undo", "redo"):
            params = list(inspect.signature(getattr(cls, name)).parameters)
            assert "units" in params, f"{cls.__name__}.{name} has no `units` parameter: {params}"


def test_a_push_that_strands_the_saved_marker_reads_dirty() -> None:
    """Undo, then make a DIFFERENT edit: _push()'s truncation destroys the
    record the saved marker pointed at, so leaving the marker alone would let
    is_dirty read clean at a cursor that no longer reconstructs the file on
    disk. A region move is undo-then-push by construction, which is what makes
    this reachable in one gesture (save right after a paste, then drag it)."""
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.mark_saved()
    assert not hist.is_dirty
    hist.undo(tiles)
    hist.apply("paint again", tiles, _paint(tiles, 0, 9))
    assert hist.is_dirty


def test_a_push_at_the_tip_leaves_the_saved_marker_alone() -> None:
    """The no-false-positive half: nothing was truncated, so the marker still
    names a live record and only the cursor moving off it makes us dirty."""
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.mark_saved()
    saved = hist.saved_at_cursor
    hist.apply("paint again", tiles, _paint(tiles, 0, 9))
    assert hist.saved_at_cursor == saved
    assert hist.is_dirty
    hist.undo(tiles)
    assert not hist.is_dirty, "undoing back onto the saved cursor is clean again"


# -- jump_to / span_kinds / on_change (GH #30's History window) ---------------


def _three_paints() -> tuple[EditHistory, list[FakeTile]]:
    """Three single-tile records, so the tile's terrain_id reads back the
    cursor position directly: cursor 0 -> 0, cursor 1 -> 1, and so on."""
    hist = EditHistory()
    tiles = [FakeTile()]
    for value in (1, 2, 3):
        hist.apply(f"paint {value}", tiles, _paint(tiles, 0, value))
    return hist, tiles


def test_jump_to_zero_replays_every_undo() -> None:
    hist, tiles = _three_paints()
    assert hist.jump_to(0, tiles) == [0]
    assert hist.cursor == 0
    assert tiles[0].terrain_id == 0


def test_jump_to_a_middle_entry_lands_exactly_there() -> None:
    hist, tiles = _three_paints()
    hist.jump_to(1, tiles)
    assert hist.cursor == 1
    assert tiles[0].terrain_id == 1


def test_jump_forward_replays_the_redos() -> None:
    hist, tiles = _three_paints()
    hist.jump_to(0, tiles)
    assert hist.jump_to(3, tiles) == [0]
    assert hist.cursor == 3
    assert tiles[0].terrain_id == 3


def test_jump_to_the_current_cursor_is_a_no_op() -> None:
    hist, tiles = _three_paints()
    assert hist.jump_to(3, tiles) == []
    assert hist.cursor == 3
    assert tiles[0].terrain_id == 3


def test_jump_out_of_range_raises_without_moving() -> None:
    hist, tiles = _three_paints()
    for target in (-1, 4):
        with pytest.raises(ValueError, match="out of range"):
            hist.jump_to(target, tiles)
    assert hist.cursor == 3
    assert tiles[0].terrain_id == 3


def test_jump_dedupes_tile_indices_across_the_span() -> None:
    """All three records touch tile 0, and _apply_dirty() branches on the
    count of what it is handed -- one tile repainted three times is one
    repaint, not three."""
    hist, tiles = _three_paints()
    assert hist.jump_to(0, tiles) == [0]


def test_jump_onto_the_saved_cursor_reads_clean_again() -> None:
    hist, tiles = _three_paints()
    hist.jump_to(1, tiles)
    hist.mark_saved()
    saved = hist.saved_at_cursor
    hist.jump_to(3, tiles)
    assert hist.is_dirty
    hist.jump_to(1, tiles)
    assert hist.saved_at_cursor == saved, "a jump never touches the saved marker"
    assert not hist.is_dirty


def test_a_refused_record_mid_span_leaves_the_cursor_and_the_tiles_alone() -> None:
    """require_target() runs over the WHOLE span before anything moves, so a
    jump past an unrestorable record is all-or-nothing, exactly like undo()."""
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.push_trigger_record(_trigger_record())
    hist.apply("paint again", tiles, _paint(tiles, 0, 9))
    with pytest.raises(RuntimeError, match="no TriggerEditModel"):
        hist.jump_to(0, tiles)
    assert hist.cursor == 3
    assert tiles[0].terrain_id == 9, "no record in the span may have been applied"


def test_span_kinds_unions_the_span_including_a_composite() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.push_composite_record(
        CompositeDiffRecord("paste", children=[_tile_record(), _unit_record()])
    )
    assert hist.span_kinds(0) == frozenset({"tile", "unit"})
    assert hist.span_kinds(1) == frozenset({"tile", "unit"}), "only the composite"
    assert hist.span_kinds(2) == frozenset(), "the cursor is already there"


def test_span_kinds_out_of_range_raises() -> None:
    hist, _tiles = _three_paints()
    with pytest.raises(ValueError, match="out of range"):
        hist.span_kinds(9)


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def test_on_change_fires_once_per_mutation() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    counter = _Counter()
    hist.on_change = counter
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    assert counter.calls == 1
    hist.apply("paint again", tiles, _paint(tiles, 0, 9))
    assert counter.calls == 2
    hist.undo(tiles)
    assert counter.calls == 3
    hist.redo(tiles)
    assert counter.calls == 4
    hist.mark_saved()
    assert counter.calls == 5
    hist.reset()
    assert counter.calls == 6


def test_on_change_fires_once_for_a_whole_jump() -> None:
    """Not once per step: the History window would otherwise rebuild its
    whole tree N times for one double-click."""
    hist, tiles = _three_paints()
    counter = _Counter()
    hist.on_change = counter
    hist.jump_to(0, tiles)
    assert counter.calls == 1


def test_on_change_does_not_fire_on_a_no_op() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    counter = _Counter()
    hist.on_change = counter
    hist.undo(tiles)
    hist.redo(tiles)
    hist.apply("nothing", tiles, lambda: None)
    hist.jump_to(0, tiles)
    assert counter.calls == 0


# -- view-only ruler records (GH #108) ----------------------------------------


def _m(n: int):
    from descape.ruler import Measurement

    return Measurement((0, 0), (n + 1, n))


def _ruler(hist, pinned, m) -> None:
    """A user ruler add, recorded the way MapView and the viewer record it."""
    from descape.edit_history import RulerDiffRecord

    index = len(pinned)
    assert pinned.insert(index, m)
    hist.push_ruler_record(RulerDiffRecord("Add ruler", ((index, m),), (), pinned.apply_delta))


def _clear(hist, pinned) -> None:
    from descape.edit_history import RulerDiffRecord

    removed = tuple(enumerate(pinned))
    pinned.clear()
    hist.push_ruler_record(RulerDiffRecord("Clear rulers", (), removed, pinned.apply_delta))


def _pinned():
    from descape.ruler import PinnedRulers

    return PinnedRulers()


def test_a_ruler_record_undoes_and_redoes_without_dirtying() -> None:
    hist = EditHistory()
    view = _pinned()
    a, b = _m(1), _m(2)
    _ruler(hist, view, a)
    _ruler(hist, view, b)
    assert not hist.is_dirty
    hist.undo([])
    assert list(view) == [a] and not hist.is_dirty
    hist.redo([])
    assert list(view) == [a, b] and not hist.is_dirty
    hist.jump_to(0, [])
    assert list(view) == [] and not hist.is_dirty


def test_an_unrecorded_ruler_survives_undo_and_redo_of_an_older_one() -> None:
    """The delta rule: a ruler drawn unrecorded while a real edit's redo was
    pending is not part of any record, so no ruler undo or redo may drop it."""
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    a, b = _m(1), _m(2)
    _ruler(hist, view, a)
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.undo(tiles)
    view.insert(len(view), b)  # unrecorded: a file edit is ahead of the cursor
    hist.undo(tiles)
    assert list(view) == [b]
    hist.redo(tiles)
    assert list(view) == [a, b], "the redone ruler goes back to its own slot"
    hist.redo(tiles)
    assert tiles[0].terrain_id == 5


def test_undoing_clear_rulers_keeps_rulers_drawn_since_and_the_old_order() -> None:
    hist = EditHistory()
    view = _pinned()
    a, b, c, d = _m(1), _m(2), _m(3), _m(4)
    for m in (a, b, c):
        _ruler(hist, view, m)
    _clear(hist, view)
    view.insert(0, d)
    hist.undo([])
    assert list(view) == [a, b, c, d]
    hist.redo([])
    assert list(view) == [d]


def test_a_ruler_delta_tolerates_a_live_set_that_already_changed() -> None:
    """Undo of an add whose ruler is already gone, and redo of one already
    back, are no-ops on the set; the cursor still moves and nothing raises."""
    hist = EditHistory()
    view = _pinned()
    a = _m(1)
    _ruler(hist, view, a)
    view.discard(a)
    hist.undo([])
    assert list(view) == [] and hist.cursor == 0
    view.insert(0, a)
    hist.redo([])
    assert list(view) == [a] and hist.cursor == 1


def test_equal_rulers_are_separate_entries_through_undo_and_redo() -> None:
    hist = EditHistory()
    view = _pinned()
    first, second = _m(1), _m(1)
    _ruler(hist, view, first)
    _ruler(hist, view, second)
    hist.undo([])
    assert len(view) == 1 and view.newest is first
    hist.redo([])
    assert len(view) == 2 and view.newest is second


def test_is_dirty_still_counts_a_real_record_beside_ruler_records() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    _ruler(hist, view, _m(1))
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    _ruler(hist, view, _m(2))
    assert hist.is_dirty
    hist.mark_saved()
    hist.undo(tiles)
    assert not hist.is_dirty, "undoing a ruler past the save point dirtied the file"
    hist.undo(tiles)
    assert hist.is_dirty
    hist.redo(tiles)
    assert not hist.is_dirty


def test_a_push_over_only_ruler_redo_keeps_the_save_point() -> None:
    """Save, undo a ruler, then edit: the truncated tail was view-only, so
    undoing that edit lands back on the saved file and must read clean."""
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    _ruler(hist, view, _m(1))
    hist.mark_saved()
    hist.undo(tiles)
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    assert hist.is_dirty
    hist.undo(tiles)
    assert not hist.is_dirty


def test_rulers_overflowing_past_the_save_point_keep_the_file_clean() -> None:
    """Save, then more ruler records than max_records: only view-only records
    were evicted past the save point, so the file still matches it."""
    hist = EditHistory(max_records=3)
    tiles = [FakeTile()]
    view = _pinned()
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.mark_saved()
    for n in range(1, 6):
        _ruler(hist, view, _m(n))
    assert hist.saved_at_cursor == 0 and not hist.is_dirty
    hist.apply("paint", tiles, _paint(tiles, 0, 6))
    assert hist.is_dirty


def test_a_real_edit_evicted_past_the_save_point_still_dirties() -> None:
    hist = EditHistory(max_records=3)
    tiles = [FakeTile()]
    view = _pinned()
    hist.mark_saved()
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    for n in range(1, 4):
        _ruler(hist, view, _m(n))
    assert hist.saved_at_cursor is None and hist.is_dirty


def test_a_ruler_push_with_a_file_edit_to_redo_raises_without_touching_history() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    _ruler(hist, view, _m(1))
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    hist.undo(tiles)
    hist.undo(tiles)
    records = list(hist.records)
    assert hist.redo_has_file_edits
    with pytest.raises(RuntimeError):
        _ruler(hist, view, _m(2))
    assert hist.records == records and hist.can_redo


def test_a_ruler_push_over_only_ruler_redo_is_recorded_and_keeps_the_save_point() -> None:
    """Loosened from "refuse whenever redo is non-empty": discarding a ruler's
    own redo costs no real edit, so the new ruler goes on the stack."""
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    a, b = _m(1), _m(2)
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    _ruler(hist, view, a)
    hist.mark_saved()
    hist.undo(tiles)
    assert hist.can_redo and not hist.redo_has_file_edits
    _ruler(hist, view, b)
    assert hist.records[-1].added == ((0, b),) and not hist.can_redo
    assert hist.saved_at_cursor == 1 and not hist.is_dirty
    hist.undo(tiles)
    assert list(view) == [] and not hist.is_dirty


# -- content_key: what autosave compares (ruler-only moves are no change) ------


def test_content_key_ignores_ruler_pushes_undos_and_redos() -> None:
    hist = EditHistory()
    tiles = [FakeTile()]
    view = _pinned()
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    key = hist.content_key
    _ruler(hist, view, _m(1))
    assert hist.content_key == key
    hist.undo(tiles)
    assert hist.content_key == key
    hist.redo(tiles)
    assert hist.content_key == key
    hist.apply("paint", tiles, _paint(tiles, 0, 6))
    assert hist.content_key != key


def test_content_key_tells_apart_two_edits_at_the_same_cursor() -> None:
    hist = EditHistory()
    tiles = [FakeTile(), FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    key = hist.content_key
    hist.undo(tiles)
    hist.apply("paint", tiles, _paint(tiles, 1, 5))
    assert hist.cursor == 1 and hist.content_key != key


def test_content_key_origin_changes_when_the_cap_evicts_a_real_edit_or_on_reset() -> None:
    hist = EditHistory(max_records=2)
    tiles = [FakeTile()]
    view = _pinned()
    origin = hist.content_key
    _ruler(hist, view, _m(1))
    _ruler(hist, view, _m(2))
    _ruler(hist, view, _m(3))
    assert hist.content_key == origin, "evicting only rulers changed the content"
    hist.apply("paint", tiles, _paint(tiles, 0, 6))
    _ruler(hist, view, _m(6))
    _ruler(hist, view, _m(7))
    assert all(r.view_only for r in hist.records)
    assert hist.content_key != origin, "an evicted paint left the old origin in place"
    evicted = hist.content_key
    hist.reset()
    assert hist.content_key != evicted


def test_a_held_content_key_does_not_pin_a_truncated_record() -> None:
    """Autosave holds content_key between ticks. Holding it must not keep a
    record alive once undo-then-edit truncates it: a fill can be tens of MB."""
    hist = EditHistory()
    tiles = [FakeTile(), FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    held = hist.content_key
    ref = weakref.ref(hist.records[-1])
    hist.undo(tiles)
    hist.apply("paint", tiles, _paint(tiles, 1, 5))
    gc.collect()
    assert ref() is None, "the held content_key kept the truncated record alive"
    assert hist.content_key != held


def test_a_held_content_key_does_not_pin_an_evicted_record() -> None:
    hist = EditHistory(max_records=1)
    tiles = [FakeTile()]
    hist.apply("paint", tiles, _paint(tiles, 0, 5))
    held = hist.content_key
    ref = weakref.ref(hist.records[-1])
    hist.apply("paint", tiles, _paint(tiles, 0, 6))
    assert len(hist.records) == 1
    gc.collect()
    assert ref() is None, "the held content_key kept the evicted record alive"
    assert hist.content_key != held
