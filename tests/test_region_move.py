"""Moving an already-pasted region: press inside the selection and drag it,
committing on release, with the whole paste-plus-moves gesture collapsing to
one undo step.

Same offscreen technique tests/test_fill_tool.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().

Built on tests/test_region_select.py's style-aware drag helpers rather than a
second copy of them -- see that module for why a viewport position has to be
derived from _tile_polygon rather than computed.
"""

from __future__ import annotations

import pytest
from test_region_select import _press, _release, _select_window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _paint(window, sx0, sy0, sx1, sy1, terrain_id):
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(terrain_id)
    for y in range(sy0, sy1):
        for x in range(sx0, sx1):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    window._on_tool_selected("select")


def _copy_and_paste(window, at=(20, 20), size=2, terrain_id=_TERRAIN_A):
    """Paints a distinct patch at the origin, copies it, and pastes it at
    `at`. Returns the paste anchor."""
    _paint(window, 0, 0, size, size, terrain_id)
    window.on_region_selected((0, 0, size, size))
    window.copy_region()
    window.on_hover(at)
    window.paste_region()
    return at


def _drag_region(window, start_tile, end_tile):
    map_view = window.map_view
    _press(map_view, start_tile)
    _move_to(map_view, end_tile)
    _release(map_view, end_tile)


def _move_to(map_view, tile):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QMouseEvent

    map_view.mouseMoveEvent(
        QMouseEvent(
            QEvent.MouseMove,
            conftest.polygon_viewport_pos(map_view, *tile),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )


def _terrain_at(window, x, y):
    return window.scenario.map_manager.get_tile(x, y).terrain_id


# -- the gesture -------------------------------------------------------------


def test_a_drag_from_inside_the_selection_moves_the_pasted_block() -> None:
    window = _select_window()
    try:
        before = _terrain_at(window, 20, 20)
        _copy_and_paste(window, at=(20, 20))
        assert _terrain_at(window, 20, 20) == _TERRAIN_A

        _drag_region(window, (20, 20), (24, 23))
        assert _terrain_at(window, 24, 23) == _TERRAIN_A
        # The original paste site is back to what it was BEFORE the paste --
        # a move is not an extra copy.
        assert _terrain_at(window, 20, 20) == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_plus_three_moves_is_one_undo_step() -> None:
    """Fact the whole design rests on: EditHistory._push() opens with
    del records[cursor:], so undo-then-push replaces the top record rather
    than stacking a second one."""
    window = _select_window()
    try:
        _paint(window, 0, 0, 2, 2, _TERRAIN_A)
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()
        snapshot = [
            [_terrain_at(window, x, y) for x in range(18, 30)] for y in range(18, 30)
        ]
        records_before = len(window.edit_history.records)

        window.on_hover((20, 20))
        window.paste_region()
        _drag_region(window, (20, 20), (22, 20))
        _drag_region(window, (22, 20), (24, 22))
        _drag_region(window, (24, 22), (26, 24))
        assert len(window.edit_history.records) == records_before + 1

        window.undo()
        after = [[_terrain_at(window, x, y) for x in range(18, 30)] for y in range(18, 30)]
        assert after == snapshot
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_press_outside_the_selection_starts_a_new_selection() -> None:
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        map_view = window.map_view
        _press(map_view, (30, 30))
        assert map_view._move_anchor is None
        assert map_view._select_anchor == (30, 30)
        _move_to(map_view, (32, 32))
        _release(map_view, (32, 32))
        assert window._region == (30, 30, 33, 33)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_zero_delta_press_release_inside_keeps_the_region() -> None:
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        region = window._region
        records = len(window.edit_history.records)
        _drag_region(window, (20, 20), (20, 20))
        assert window._region == region
        assert len(window.edit_history.records) == records
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_escape_mid_move_writes_nothing_and_restores_the_overlay() -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        map_view = window.map_view
        region = window._region
        records = len(window.edit_history.records)
        terrain = _terrain_at(window, 26, 26)

        _press(map_view, (20, 20))
        _move_to(map_view, (26, 26))
        map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))

        assert map_view._move_anchor is None
        assert window._region == region, "Escape mid-move must not destroy the region"
        assert len(window.edit_history.records) == records
        assert _terrain_at(window, 26, 26) == terrain
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_block_dragged_off_the_map_and_back_lands_whole() -> None:
    """The property that says the snapshot-plus-unclipped-anchor model is
    right: it fails on any design that re-reads the map between steps."""
    window = _select_window()
    try:
        _paint(window, 0, 0, 3, 3, _TERRAIN_A)
        window.on_region_selected((0, 0, 3, 3))
        window.copy_region()
        window.on_hover((20, 20))
        window.paste_region()

        map_view = window.map_view
        _press(map_view, (20, 20))
        _move_to(map_view, (1, 1))  # most of the block now hangs off the edge
        _move_to(map_view, (26, 26))
        _release(map_view, (26, 26))

        for y in range(26, 29):
            for x in range(26, 29):
                assert _terrain_at(window, x, y) == _TERRAIN_A, (x, y)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_move_replays_the_pastes_own_categories_not_the_current_boxes() -> None:
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        # Unchecking terrain AFTER the paste must not change what a move of
        # that paste re-places: a move replays the paste that happened.
        window.paste_terrain_check.setChecked(False)
        _drag_region(window, (20, 20), (24, 24))
        assert _terrain_at(window, 24, 24) == _TERRAIN_A
    finally:
        window.paste_terrain_check.setChecked(True)
        window.edit_history.mark_saved()
        window.close()


# -- when the state lapses ---------------------------------------------------


def _press_inside_starts_a_selection(window, tile=(20, 20)) -> bool:
    map_view = window.map_view
    _press(map_view, tile)
    started_selection = map_view._move_anchor is None and map_view._select_anchor is not None
    map_view._select_anchor = None
    map_view._select_current = None
    map_view._move_anchor = None
    map_view._move_current = None
    return started_selection


def test_an_unrelated_edit_lapses_the_move_state() -> None:
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        assert window._paste_move is not None
        _paint(window, 40, 40, 41, 41, _TERRAIN_B)
        assert window._paste_move is None
        assert _press_inside_starts_a_selection(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_undo_lapses_the_move_state_and_a_redo_does_not_rearm_it() -> None:
    """The redo case is the sweep working, not the identity check misfiring:
    the state was already cleared at undo time and nothing re-arms it on the
    way back."""
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        window.undo()
        assert window._paste_move is None
        window.redo()
        assert window._paste_move is None
        assert _press_inside_starts_a_selection(window, (20, 20))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("lapse", ["select_all", "fresh_drag", "copy", "deselect"])
def test_a_selection_or_clipboard_change_lapses_the_move_state(lapse: str) -> None:
    window = _select_window()
    try:
        _copy_and_paste(window, at=(20, 20))
        assert window._paste_move is not None
        if lapse == "select_all":
            window.select_all()
        elif lapse == "deselect":
            window.deselect()
        elif lapse == "copy":
            window.copy_region()
        else:
            map_view = window.map_view
            _press(map_view, (30, 30))
            _release(map_view, (32, 32))
        assert window._paste_move is None
        assert not window.map_view._region_movable
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_activating_another_clipboard_entry_lapses_the_move_state() -> None:
    """Every clipboard-history mutation clears the move state, so dragging
    never moves the PREVIOUSLY pasted content after the user has shifted
    clipboard intent."""
    window = _select_window()
    try:
        _paint(window, 0, 0, 2, 2, _TERRAIN_A)
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()
        entry_a = window._clipboard_history.entries[0]
        _paint(window, 5, 0, 7, 2, _TERRAIN_B)
        window.on_region_selected((5, 0, 7, 2))
        window.copy_region()

        window.on_hover((20, 20))
        window.paste_region()
        assert window._paste_move is not None

        window._on_clipboard_activate(entry_a.entry_id)
        assert window._paste_move is None
        assert _press_inside_starts_a_selection(window, (20, 20))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_that_wrote_nothing_arms_no_move() -> None:
    window = _select_window()
    try:
        _paint(window, 0, 0, 2, 2, _TERRAIN_A)
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()
        for check in (
            window.paste_terrain_check,
            window.paste_elevation_check,
            window.paste_units_check,
        ):
            check.setChecked(False)
        window.on_hover((20, 20))
        window.paste_region()
        assert window._paste_move is None
        assert not window.map_view._region_movable
    finally:
        for check in (
            window.paste_terrain_check,
            window.paste_elevation_check,
            window.paste_units_check,
        ):
            check.setChecked(True)
        window.edit_history.mark_saved()
        window.close()
