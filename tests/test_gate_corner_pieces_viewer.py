"""A moved gate carries its corner pieces (TASK-531), through a real
offscreen ViewerWindow: every translation path (single drag, arrow nudge, a
group of gate plus pieces, a typed coordinate, Scatter's move mode) moves
them by the gate's own delta in the gate's undo record. The lookup itself is
tested headless in test_placement_rules.py.

A turned, deleted or cut gate takes them too (TASK-532): Rotate, the Rotation
field and a drag's turn land them on the gate's new ends as the new corner
sibling (placement_rules.companion_landing()); Delete, Cut and Find's Delete
remove them with it. Flip and Rotate Group are in test_selection_transform_viewer.py.

The window is test_wall_run_viewer's GH #194 one: a Stone Gate (64) on
tiles 25..28 of a wall run, with its corner pieces (81) added on tiles 25
and 28, as real files store them."""

from __future__ import annotations

import numpy as np
import pytest
from test_wall_run_viewer import (  # noqa: F401 -- the two underscored fixtures are autouse
    _GRAB,
    _HELD,
    _NEW,
    _OLD,
    _ROW,
    GATE,
    PLAYER,
    VILLAGER,
    WALL,
    _add_unit,
    _arrow,
    _close,
    _drag_gate,
    _gate_window,
    _hold_the_synthetic_button,
    _is_field_delta,
    _last_status,
    _mouse_event,
    _moved_gate_status,
    _point_pos,
    _select_units,
    _unit_count,
    _wall_consts_intact,
    _walls_on,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

CORNER = 81  # Stone Gate 64's corner piece
GATE_Y, CORNER_Y = 88, 95  # GATE's (1, 4) sibling, a quarter turn along, and its corner
GATE_C, CORNER_C = 659, 663  # GATE's +x+y diagonal, one step along, and its corner
_Y = _ROW + 0.5
# GATE on tiles 25..28 has its low corner at (25, _ROW): turned there, its pieces land on these.
_TURNED_Y = [(CORNER_Y, 25.5, _Y), (CORNER_Y, 25.5, _Y + 3)]
_TURNED_C = [(CORNER_C, 25.5, _Y), (CORNER_C, 28.5, _Y + 3)]
# On _GRAB's tile but off its point, past the drag threshold once _zoom_in() has run.
_HOME = (26.2, _ROW + 0.8)


def _zoom_in(window) -> None:
    """test_unit_drag_preview's _zoomed(): a drag can end on the tile it began on."""
    view = window.map_view
    view.scale(12, 12)
    assert view.transform().m11() * view._tile_pixels > 2 * view.UNIT_DRAG_THRESHOLD_PX


def _pieces_window(box: bool = True, free: bool = False):
    """(window, gate, [west piece, east piece]): _gate_window() plus the two pieces."""
    from descape import gate_orientation

    assert gate_orientation.corner_piece(GATE) == CORNER
    window, gate = _gate_window(box=box, free=free)
    pieces = [_add_unit(window, CORNER, 25.5, _Y), _add_unit(window, CORNER, 28.5, _Y)]
    window._rebuild_unit_index()
    return window, gate, pieces


def _points(units) -> list[tuple[float, float]]:
    return [(u.x, u.y) for u in units]


def test_box_off_a_dragged_gate_carries_its_pieces_in_one_record():
    window, gate, pieces = _pieces_window(box=False, free=True)
    try:
        records = len(window.edit_history.records)

        _drag_gate(window, 5, 0)

        assert (gate.x, gate.y) == (32.0, _Y), _last_status(window)
        assert _points(pieces) == [(30.5, _Y), (33.5, _Y)]
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        assert _moved_gate_status(window) == f"to (32, {_Y:g})"

        window.undo()
        assert (gate.x, _points(pieces)) == (27.0, [(25.5, _Y), (28.5, _Y)])
        window.redo()
        assert (gate.x, _points(pieces)) == (32.0, [(30.5, _Y), (33.5, _Y)])
    finally:
        _close(window)


def test_box_on_a_nudge_carries_the_trailing_piece_so_the_fill_lands():
    """Without the carry the piece left on tile 25 stands where the fill goes,
    and the fill skips it."""
    from PyQt5.QtCore import Qt

    window, gate, pieces = _pieces_window()
    try:
        _select_units(window, [gate])
        before = _unit_count(window)
        records = len(window.edit_history.records)

        _arrow(window, Qt.Key_Right, Qt.ShiftModifier)

        assert gate.x == 28.0
        assert _points(pieces) == [(26.5, _Y), (29.5, _Y)]
        assert _walls_on(window, {(29, _ROW)}) == []
        assert [u.unit_const for u in _walls_on(window, {(25, _ROW)})] == [WALL]
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records + 1
        status = _moved_gate_status(window)
        assert status == f"to (28, {_Y:g}), replacing 1 wall, filling 1", status

        window.undo()
        assert (gate.x, _points(pieces)) == (27.0, [(25.5, _Y), (28.5, _Y)])
        assert len(_walls_on(window, {(29, _ROW)})) == 1 and _walls_on(window, {(25, _ROW)}) == []
    finally:
        _close(window)


def test_a_gate_selected_with_its_pieces_moves_each_piece_once():
    from PyQt5.QtCore import Qt

    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate, *pieces])
        records = len(window.edit_history.records)

        _arrow(window, Qt.Key_Down, Qt.ShiftModifier)

        assert (gate.x, gate.y) == (27.0, _Y + 1)
        assert _points(pieces) == [(25.5, _Y + 1), (28.5, _Y + 1)]
        assert len(window.edit_history.records) == records + 1
        assert _last_status(window) == "Moved 3 units"
    finally:
        _close(window)


def test_a_selected_piece_alone_moves_alone():
    from PyQt5.QtCore import Qt

    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [pieces[0]])

        _arrow(window, Qt.Key_Down, Qt.ShiftModifier)

        assert _points(pieces) == [(25.5, _Y + 1), (28.5, _Y)]
        assert (gate.x, gate.y) == (27.0, _Y)
    finally:
        _close(window)


def test_a_typed_y_on_the_gate_carries_its_pieces():
    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate])
        records = len(window.edit_history.records)

        window.units_panel.unit_field_editors["y"].setValue(50.5)

        assert (gate.x, gate.y) == (27.0, 50.5)
        assert _points(pieces) == [(25.5, 50.5), (28.5, 50.5)]
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert (gate.y, _points(pieces)) == (_Y, [(25.5, _Y), (28.5, _Y)])
    finally:
        _close(window)


def test_a_typed_x_on_the_gate_and_its_selected_pieces_keeps_their_offset():
    """The selected pieces ride with the gate rather than stacking on the typed value."""
    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate, *pieces])

        window.units_panel.unit_field_editors["x"].setValue(10.0)

        assert (gate.x, gate.y) == (10.0, _Y)
        assert _points(pieces) == [(8.5, _Y), (11.5, _Y)]
    finally:
        _close(window)


def test_scatter_move_mode_puts_the_pieces_at_the_gates_new_point_plus_offset():
    from descape.scatter_dialog import ScatterParams

    window, gate, pieces = _pieces_window(box=False)
    try:
        keys = _select_units(window, [gate, *pieces])
        entries = [window.map_view._unit_index.entry_for_key(key) for key in keys]
        area = [(x, y) for x in range(50, 60) for y in range(50, 60)]
        params = ScatterParams(
            tiles=area, unit_const=None, player=1, count=None, density=None, seed=7, jitter=0.0,
            min_spacing=0, move_selected=True,
        )
        records = len(window.edit_history.records)

        window._scatter_selected_units(entries, params)

        assert (int(gate.x), int(gate.y)) in area, _last_status(window)
        assert _points(pieces) == [(gate.x - 1.5, gate.y), (gate.x + 1.5, gate.y)]
        assert len(window.edit_history.records) == records + 1
        assert _last_status(window) == "Scattered 1 selected units across the area"
    finally:
        _close(window)


def test_the_pick_index_follows_a_dragged_gates_pieces():
    """The canvas cannot tell (the gate's own splice covers its pieces' tiles); the index patch can."""
    from descape import unit_pick

    window, gate, pieces = _pieces_window(box=False, free=True)
    try:
        _drag_gate(window, 0, 6)

        assert gate.y == _Y + 6, _last_status(window)
        index = window.map_view._unit_index
        own = [index.entry_for_key(unit_pick.unit_key(1, p)) for p in pieces]
        assert [(e.own_x, e.own_y) for e in own] == [(25, _ROW + 6), (28, _ROW + 6)]
    finally:
        _close(window)


def test_a_dragged_gates_piece_splices_repaint_like_a_fresh_render():
    from unit_edit_viewer_helpers import _rebuilt_canvas, _whole_canvas

    window, gate, pieces = _pieces_window(box=False, free=True)
    try:
        before = _whole_canvas(window)

        _drag_gate(window, 0, 6)

        assert (gate.y, _points(pieces)) == (_Y + 6, [(25.5, _Y + 6), (28.5, _Y + 6)])
        assert _is_field_delta(window)
        moved = _whole_canvas(window)
        assert not np.array_equal(before, moved)
        assert np.array_equal(moved, _rebuilt_canvas(window))
        assert len(_walls_on(window, _NEW)) == 4 and _walls_on(window, _OLD) == []
    finally:
        _close(window)


# --- TASK-532: a turned, deleted or cut gate takes its pieces --------------------


def _states(units) -> list[tuple[int, float, float]]:
    return [(u.unit_const, u.x, u.y) for u in units]


def _turned_drag(window, *points, cw: int = 2) -> None:
    """A real press at points[0], a move to each later point with the Rotate
    presses after the first move, and a release at the last (GH #280)."""
    from PyQt5.QtCore import QEvent, Qt

    view = window.map_view
    _HELD["value"] = True
    try:
        view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, _point_pos(window, *points[0]), Qt.LeftButton, Qt.LeftButton))
        for i, point in enumerate(points[1:]):
            view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, _point_pos(window, *point), Qt.NoButton, Qt.LeftButton))
            for _ in range(cw if i == 0 else 0):
                window.rotate_cw_action.trigger()
        end = _point_pos(window, *points[-1])
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
    finally:
        _HELD["value"] = False


@pytest.mark.parametrize("with_pieces", [False, True])
def test_coarse_rotate_turns_the_pieces_onto_the_gates_new_ends_in_one_record(with_pieces):
    """Selected or not, the pieces ride with their gate: the status names the one gate."""
    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate, *pieces] if with_pieces else [gate])
        before = _states([gate, *pieces])
        records = len(window.edit_history.records)

        window.on_unit_rotate_coarse(1)

        after = [(GATE_Y, 25.5, _ROW + 2.0), *_TURNED_Y]
        assert _states([gate, *pieces]) == after, _last_status(window)
        assert len(window.edit_history.records) == records + 1
        status = _last_status(window)
        assert status.startswith("Cycled ") and f" to (25.5, {_ROW + 2:g})" in status, status
        window.undo()
        assert _states([gate, *pieces]) == before
        window.redo()
        assert _states([gate, *pieces]) == after
    finally:
        _close(window)


def test_fine_rotate_lands_the_far_pieces_on_the_diagonals_far_tip():
    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate])
        window.on_unit_rotate(1)
        assert _states([gate, *pieces]) == [(GATE_C, 27.0, _ROW + 2.0), *_TURNED_C]
    finally:
        _close(window)


def test_rotate_on_a_lone_selected_piece_cycles_it_in_place():
    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [pieces[0]])
        window.on_unit_rotate_coarse(1)
        assert _states([gate, *pieces]) == [(GATE, 27.0, _Y), (CORNER_Y, 25.5, _Y), (CORNER, 28.5, _Y)]
    finally:
        _close(window)


def test_the_rotation_fields_typed_index_and_its_step_turn_the_pieces():
    from descape import unit_pick

    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate])
        window.units_panel.unit_field_editors["rotation"].setValue(2)
        assert _states([gate, *pieces]) == [(GATE_Y, 25.5, _ROW + 2.0), *_TURNED_Y]

        window.undo()
        _select_units(window, [gate, *pieces])
        entries = [unit_pick.UnitEntry(PLAYER, u, int(u.x), int(u.y), -1) for u in (gate, *pieces)]
        window._step_rotation_field(window._ensure_unit_edits(), entries, 1)
        assert _states([gate, *pieces]) == [(GATE_C, 27.0, _ROW + 2.0), *_TURNED_C]
        assert _last_status(window).startswith("Turned 1 unit by +1"), _last_status(window)
    finally:
        _close(window)


def test_a_gate_blocked_at_the_map_edge_leaves_its_selected_pieces_unturned_too():
    window, _gate, _pieces = _pieces_window(box=False)
    try:
        # The fixture is 120 tiles square: a (1, 4) turn from row 119 would hang off it.
        edge = [_add_unit(window, GATE, 60.0, 119.5), _add_unit(window, CORNER, 58.5, 119.5), _add_unit(window, CORNER, 61.5, 119.5)]
        window._rebuild_unit_index()
        before = _states(edge)
        _select_units(window, edge)
        window.on_unit_rotate_coarse(1)
        assert _states(edge) == before
        assert _last_status(window) == "Rotate: 1 gate(s) would hang off the map edge"
    finally:
        _close(window)


def test_a_single_drag_with_a_turn_lands_the_pieces_on_the_turned_ends_plus_the_delta():
    window, gate, pieces = _pieces_window(box=False, free=True)
    try:
        before = _states([gate, *pieces])
        _turned_drag(window, _GRAB, (50.5, _ROW + 10.1))
        assert (gate.unit_const, gate.x, gate.y) == (GATE_Y, 50.5, _ROW + 10.0), _last_status(window)
        # The stand-in stood at (25.5, _ROW + 2): a delta of (25, 8).
        assert _states(pieces) == [(CORNER_Y, 50.5, _Y + 8), (CORNER_Y, 50.5, _Y + 11)]
        window.undo()
        assert _states([gate, *pieces]) == before
    finally:
        _close(window)


def test_a_single_drag_turned_and_dropped_on_its_own_point_turns_the_pieces():
    window, gate, pieces = _pieces_window(box=False)
    try:
        _turned_drag(window, _GRAB, (40.5, _ROW + 8.1), (25.6, _ROW + 2.1))
        assert _states([gate, *pieces]) == [(GATE_Y, 25.5, _ROW + 2.0), *_TURNED_Y], _last_status(window)
        assert window.edit_history.peek_undo().label == "Rotate unit"
    finally:
        _close(window)


@pytest.mark.parametrize("box", [False, True], ids=["splice", "wholesale"])
def test_a_group_drag_with_a_turn_lands_the_pieces_on_the_turned_ends_plus_the_delta(box):
    """Ticked, the gap the gate leaves fills, so the record is wholesale."""
    window, gate, pieces = _pieces_window(box=box, free=True)
    try:
        villager = _add_unit(window, VILLAGER, 60.5, 60.5)
        window._rebuild_unit_index()
        _select_units(window, [gate, villager])
        _turned_drag(window, _GRAB, (36.75, _ROW + 6.4))
        assert (gate.unit_const, gate.x, gate.y) == (GATE_Y, 35.5, _ROW + 8.0), _last_status(window)
        assert _states(pieces) == [(CORNER_Y, 35.5, _Y + 6), (CORNER_Y, 35.5, _Y + 9)]
        assert (villager.x, villager.y) == (70.5, 66.5)
        assert _is_field_delta(window) != box
        assert ("filling 4" in _last_status(window)) == box, _last_status(window)
    finally:
        _close(window)


def test_a_group_turn_in_place_lands_the_pieces_on_the_turned_ends():
    window, gate, pieces = _pieces_window(box=False)
    try:
        villager = _add_unit(window, VILLAGER, 60.5, 60.5)
        window._rebuild_unit_index()
        _select_units(window, [gate, villager])
        _zoom_in(window)
        _turned_drag(window, _GRAB, _HOME)
        assert _states([gate, *pieces]) == [(GATE_Y, 25.5, _ROW + 2.0), *_TURNED_Y], _last_status(window)
        assert (villager.x, villager.y) == (60.5, 60.5)
        assert _last_status(window) == "Rotated 1 unit, cycled 1 gate"
    finally:
        _close(window)


def test_a_group_turn_of_a_gate_and_one_of_two_stacked_pieces_lands_all_three_unrefused():
    """The selected piece rides with its gate: never turned on its own spot, never judged against its unselected twin."""
    window, gate, pieces = _pieces_window(box=False)
    try:
        twin = _add_unit(window, CORNER, 25.5, _Y)
        window._rebuild_unit_index()
        _select_units(window, [gate, pieces[0]])
        _zoom_in(window)
        _turned_drag(window, _GRAB, _HOME)
        assert _states([gate, *pieces]) == [(GATE_Y, 25.5, _ROW + 2.0), *_TURNED_Y], _last_status(window)
        assert _states([twin]) == _TURNED_Y[:1]
        status = _last_status(window)
        assert status.startswith("Cycled ") and f" to (25.5, {_ROW + 2:g})" in status, status
    finally:
        _close(window)


@pytest.mark.parametrize("path", ["delete", "find"])
def test_delete_takes_the_gates_pieces_in_one_record(path):
    from PyQt5.QtCore import Qt

    from descape import unit_pick
    from descape.viewer_common import _unit_name

    window, gate, pieces = _pieces_window(box=False)
    try:
        before = _states([gate, *pieces])
        count = _unit_count(window)
        records = len(window.edit_history.records)
        if path == "delete":
            _select_units(window, [gate])
            window.on_unit_delete(Qt.NoModifier)
        else:
            assert window._delete_units([unit_pick.UnitEntry(PLAYER, gate, 27, _ROW, -1)]) == (3, 0, 0)

        alive = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert not alive & {id(gate), *map(id, pieces)}
        assert _unit_count(window) == count - 3
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Delete unit"
        assert _last_status(window).startswith(f"Deleted {_unit_name(GATE)} (2 corner pieces)"), _last_status(window)
        window.undo()
        assert _unit_count(window) == count and _states([gate, *pieces]) == before
    finally:
        _close(window)


def test_the_eraser_over_the_gates_middle_leaves_its_pieces():
    """The carry is opt-in: a stroke erases only what is under the brush."""
    window, gate, pieces = _pieces_window(box=False)
    try:
        window.eraser_action.setChecked(True)
        window.brush_size_spin.setValue(1)
        for name, action in window.eraser_target_actions.items():
            action.setChecked(name == "buildings")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(26, _ROW)], 0)
        window.on_edit_stroke_end()
        alive = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert id(gate) not in alive, _last_status(window)
        assert {id(p) for p in pieces} <= alive
    finally:
        _close(window)


def test_cut_holds_the_pieces_and_copy_units_holds_them_only_when_selected():
    from descape.viewer_common import _unit_name

    window, gate, pieces = _pieces_window(box=False)
    try:
        _select_units(window, [gate])
        window.copy_units()
        assert [u.unit_const for u in window._clipboard_history.active_block.units] == [GATE]
        _select_units(window, [gate, *pieces])
        window.copy_units()
        assert sorted(u.unit_const for u in window._clipboard_history.active_block.units) == [GATE, CORNER, CORNER]

        _select_units(window, [gate])
        count = _unit_count(window)
        ids = sorted(u.reference_id for u in (gate, *pieces))
        tile = (int(gate.x), int(gate.y))
        window.cut_units()
        assert _unit_count(window) == count - 3
        assert sorted(u.unit_const for u in window._clipboard_history.active_block.units) == [GATE, CORNER, CORNER]
        assert window.edit_history.peek_undo().label == "Cut unit"
        cut_line = f"Cut {_unit_name(GATE)} (2 corner pieces) to the clipboard"
        assert _last_status(window).startswith(cut_line), _last_status(window)

        # TASK-341 meets TASK-532: the pasted pieces keep their ids and belong to their gate again.
        from descape.placement_rules import gate_companions

        window.on_hover(tile)
        window.paste_region()
        assert "IDs kept" in _last_status(window), _last_status(window)
        live = {u.reference_id: u for units in window.scenario.unit_manager.units for u in units}
        assert all(i in live for i in ids)
        (pasted,) = [live[i] for i in ids if live[i].unit_const == GATE]
        assert len(gate_companions(window.scenario, pasted)) == 2
    finally:
        _close(window)


def test_a_rotated_gates_piece_splices_repaint_like_a_fresh_render():
    """Off the wall run, so the cycle touches no wall and keeps the fields-only splice record."""
    from unit_edit_viewer_helpers import _rebuilt_canvas, _whole_canvas

    window, _gate, _pieces = _pieces_window(box=False)
    try:
        lone = [_add_unit(window, GATE, 60.0, 60.5), _add_unit(window, CORNER, 58.5, 60.5), _add_unit(window, CORNER, 61.5, 60.5)]
        window._rebuild_unit_index()
        _select_units(window, [lone[0]])
        before = _whole_canvas(window)

        window.on_unit_rotate(1)

        assert _states(lone) == [(GATE_C, 60.0, 62.0), (CORNER_C, 58.5, 60.5), (CORNER_C, 61.5, 63.5)]
        assert _is_field_delta(window)
        turned = _whole_canvas(window)
        assert not np.array_equal(before, turned)
        assert np.array_equal(turned, _rebuilt_canvas(window))
    finally:
        _close(window)
