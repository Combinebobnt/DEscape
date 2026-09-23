"""A raise inside an open EditHistory stroke must not leave the stroke open.

MapManager.get_tile() raises on a non-square map, and before
ViewerWindow._close_stroke_on_error() a raise there left
EditHistory._stroke_before set, so every later begin_stroke() raised too and
one click wedged the document. Each test stubs get_tile to raise partway
through, then checks the stroke closed, the tiles already written are one
undoable record, and the next stroke works. Same offscreen technique as
tests/test_fill_tool.py; every window calls edit_history.mark_saved() before
close().
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _draw_window():
    window = conftest.terrain_edit_window()
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(_TERRAIN)
    window.brush_size_spin.setValue(1)
    # No tree/eye-candy planting: its large-edit QMessageBox would hang offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _raise_after(monkeypatch, mm, good_calls: int):
    """get_tile works `good_calls` times, then raises what a non-square map raises."""
    real = mm.get_tile
    calls = {"n": 0}

    def get_tile(x, y):
        calls["n"] += 1
        if calls["n"] > good_calls:
            raise ValueError("Map is not a square")
        return real(x, y)

    monkeypatch.setattr(mm, "get_tile", get_tile)
    return real


def _terrain_at(mm, x: int, y: int) -> int:
    return mm.terrain[y * mm.map_width + x].terrain_id


def _assert_next_stroke_works(window, monkeypatch, real_get_tile, x: int, y: int) -> None:
    """Not monkeypatch.undo(): that would also undo conftest's settings isolation."""
    mm = window.scenario.map_manager
    monkeypatch.setattr(mm, "get_tile", real_get_tile)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()
    assert _terrain_at(mm, x, y) == _TERRAIN


def test_a_raise_mid_drag_closes_the_stroke_and_the_release_is_harmless(monkeypatch) -> None:
    window = _draw_window()
    try:
        mm = window.scenario.map_manager
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(10, 10, 0)
        real = _raise_after(monkeypatch, mm, good_calls=0)
        with pytest.raises(ValueError):
            window.on_edit_stroke_tile(12, 10, 0)

        assert not window.edit_history.in_stroke
        assert len(window.edit_history.records) == 1, "the tile painted before the raise was not recorded"
        # MapView keeps delivering the drag and then the release; neither may write or raise.
        window.on_edit_stroke_tile(14, 10, 0)
        window.on_edit_stroke_end()
        assert _terrain_at(mm, 14, 10) != _TERRAIN
        assert len(window.edit_history.records) == 1

        monkeypatch.setattr(mm, "get_tile", real)  # undo's own repaint reads it
        window.undo()
        assert _terrain_at(mm, 10, 10) != _TERRAIN
        _assert_next_stroke_works(window, monkeypatch, real, 20, 20)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_raise_inside_a_shape_commit_closes_the_stroke(monkeypatch) -> None:
    window = _draw_window()
    try:
        mm = window.scenario.map_manager
        real = _raise_after(monkeypatch, mm, good_calls=1)
        with pytest.raises(ValueError):
            window.on_shape_commit([(8, 8), (9, 8), (10, 8)])

        assert not window.edit_history.in_stroke
        assert not window._busy and window.isEnabled()
        assert _terrain_at(mm, 8, 8) == _TERRAIN
        assert len(window.edit_history.records) == 1
        _assert_next_stroke_works(window, monkeypatch, real, 20, 20)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_raise_inside_a_region_paste_closes_the_stroke(monkeypatch) -> None:
    window = _draw_window()
    try:
        mm = window.scenario.map_manager
        for x in (0, 1):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, 0, 0)
            window.on_edit_stroke_end()
        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 2, 1))
        window.copy_region()
        block = window._clipboard_history.active_block
        assert block is not None
        records_before = len(window.edit_history.records)

        # paste_terrain indexes mm.terrain directly; set_tiles_elevation is what calls get_tile.
        real = _raise_after(monkeypatch, mm, good_calls=0)
        with pytest.raises(ValueError):
            window._paste_block_at(block, 30, 30, True, True, False)

        assert not window.edit_history.in_stroke
        assert _terrain_at(mm, 30, 30) == _TERRAIN
        assert len(window.edit_history.records) == records_before + 1
        window._on_tool_selected("draw")
        _assert_next_stroke_works(window, monkeypatch, real, 40, 40)
    finally:
        window.edit_history.mark_saved()
        window.close()
