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

import inspect

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
