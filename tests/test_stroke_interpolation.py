"""Cursor-tile interpolation for the drag-stroke tools. Qt compresses queued
motion events, so a slow stroke handler sees cursor tiles several apart;
MapView._touch_tile fills the gap with brush.line_tiles and hands the whole
path to ViewerWindow.on_edit_stroke_tiles in one call, which mutates tile by
tile but scans and applies the dirty set once. tests/test_brush.py pins the
path geometry itself.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.brush import BRUSH_SHAPE_SQUARE, line_tiles

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _edit_window(tool: str):
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_TERRAIN)
    # Trees default on and would pop a large-edit confirm modal offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _shown_flat(window) -> None:
    from PyQt5.QtWidgets import QApplication

    # Unchecked before the style switch -- see tests/test_brush_stroke.py's _shown_flat.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show()
    QApplication.processEvents()


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _press(map_view, tile, button=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    pos = conftest.polygon_viewport_pos(map_view, *tile)
    map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, button, button))


def _move(map_view, tile, button=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    pos = conftest.polygon_viewport_pos(map_view, *tile)
    map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, button))


def _release(map_view, tile, button=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    pos = conftest.polygon_viewport_pos(map_view, *tile)
    map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, button, Qt.NoButton))


def _painted(window) -> set[tuple[int, int]]:
    return {(t.x, t.y) for t in window.scenario.map_manager.terrain if t.terrain_id == _TERRAIN}


def test_a_far_jump_paints_every_tile_on_the_line_with_one_apply() -> None:
    """Brush 1 is the case the gaps were visible at: without interpolation a
    5-tile jump paints only its two end tiles."""
    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        start, end = (10, 10), (15, 12)
        calls = []
        original = window._apply_dirty

        def counting(dirty_indices) -> None:
            calls.append(set(dirty_indices))
            original(dirty_indices)

        window._apply_dirty = counting
        try:
            _press(map_view, start)
            calls.clear()
            _move(map_view, end)
            assert len(calls) == 1, f"one mouse event ran _apply_dirty {len(calls)} times"
            _release(map_view, end)
        finally:
            del window._apply_dirty

        assert _painted(window) == {start, *line_tiles(*start, *end)}
        assert len(window.edit_history.records) == 1
    finally:
        _close(window)


@pytest.mark.parametrize("tool", ["elevation", "set_level"])
def test_a_batched_path_matches_per_tile_calls_exactly(tool: str) -> None:
    """The propagation-order guard: set_tiles_elevation runs per path tile,
    in path order, so one batched event must leave the same terrain as a
    slow drag delivering the same tiles one at a time. Brush 3 so
    footprints overlap and propagation re-touches tiles mid-path."""
    path = [(20, 20), *line_tiles(20, 20, 27, 24), *line_tiles(27, 24, 22, 29)]

    def run(batched: bool):
        window = _edit_window(tool)
        try:
            window.brush_size_spin.setValue(3)
            window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
            window.elevation_level_spin.setValue(5)
            window.on_edit_stroke_start()
            if batched:
                window.on_edit_stroke_tiles(path[:1], 0)
                window.on_edit_stroke_tiles(path[1:], 0)
            else:
                for x, y in path:
                    window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
            assert len(window.edit_history.records) == 1
            return [int(t.elevation) for t in window.scenario.map_manager.terrain]
        finally:
            _close(window)

    per_tile = run(batched=False)
    assert any(per_tile), "the stroke raised nothing, so the comparison is vacuous"
    assert run(batched=True) == per_tile


def test_the_batched_elevate_undo_restores_everything_in_one_step() -> None:
    window = _edit_window("elevation")
    try:
        _shown_flat(window)
        map_view = window.map_view
        _press(map_view, (10, 10))
        _move(map_view, (16, 14))
        _release(map_view, (16, 14))
        mm = window.scenario.map_manager
        raised = {(t.x, t.y) for t in mm.terrain if t.elevation == 1}
        assert raised == {(10, 10), *line_tiles(10, 10, 16, 14)}
        window.undo()
        assert not any(t.elevation for t in mm.terrain)
    finally:
        _close(window)


def _record_deliveries(map_view) -> list[list[tuple[int, int]]]:
    """Swaps MapView's stroke callbacks for fakes, so only its routing runs."""
    delivered: list[list[tuple[int, int]]] = []
    map_view._on_stroke_start = lambda: None
    map_view._on_stroke_end = lambda: None
    map_view._on_stroke_tiles = lambda tiles, modifiers: delivered.append(list(tiles))
    return delivered


def test_mapview_delivers_the_gap_filled_path_per_event() -> None:
    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        delivered = _record_deliveries(map_view)
        _press(map_view, (10, 10))
        _move(map_view, (14, 12))
        _move(map_view, (14, 12))  # same cursor tile: nothing new
        _move(map_view, (14, 15))
        _move(map_view, (14, 12))  # back over touched tiles only: nothing new
        _move(map_view, (11, 12))  # the path starts at (14, 12), the last cursor tile
        _release(map_view, (11, 12))
        assert delivered == [
            [(10, 10)],
            line_tiles(10, 10, 14, 12),
            [(14, 13), (14, 14), (14, 15)],
            [(12, 12), (11, 12)],  # (13, 12) is already on the first path
        ]
    finally:
        _close(window)


@pytest.mark.parametrize(("mode", "tool"), [("Terrain", "cliff"), ("Units", "convert")])
def test_cliff_and_convert_get_the_gap_filled_path(mode: str, tool: str) -> None:
    window = _edit_window("draw")
    try:
        window.mode_combo.setCurrentText(mode)
        window._on_tool_selected(tool)
        _shown_flat(window)
        map_view = window.map_view
        assert map_view._tool == tool
        delivered = _record_deliveries(map_view)
        _press(map_view, (10, 10))
        _move(map_view, (15, 12))
        _release(map_view, (15, 12))
        assert delivered == [[(10, 10)], line_tiles(10, 10, 15, 12)], f"{tool}: {delivered}"
    finally:
        _close(window)


_UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _units_window(mode: str, tool: str):
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(_UNITS_FIXTURE)
    assert window.scenario is not None, "fixture failed to load"
    window.show_garrisoned_action.setChecked(True)
    window.mode_combo.setCurrentText(mode)
    window._on_tool_selected(tool)
    _shown_flat(window)
    assert window.map_view._tool == tool
    return window


def _drag(map_view, tiles) -> None:
    """Press on the first tile, one move per remaining tile, release on the last."""
    _press(map_view, tiles[0])
    for tile in tiles[1:]:
        _move(map_view, tile)
    _release(map_view, tiles[-1])


def test_a_compressed_convert_drag_converts_what_a_tile_by_tile_drag_does() -> None:
    """The fixture's three player-1 units sit on (10..12, 10), strictly inside
    this drag: without gap-filling, a two-event drag converts none of them."""
    start, end = (9, 10), (13, 10)

    def run(tiles) -> tuple[set, set]:
        window = _units_window("Units", "convert")
        try:
            window.units_panel.select_owner(2)
            window.brush_size_spin.setValue(1)
            units = window.scenario.unit_manager.units
            player_1 = {(u.unit_const, u.x, u.y) for u in units[1]}
            _drag(window.map_view, tiles)
            return player_1, player_1 & {(u.unit_const, u.x, u.y) for u in units[2]}
        finally:
            _close(window)

    player_1, tile_by_tile = run([start, *line_tiles(*start, *end)])
    assert len(player_1) == 3
    assert tile_by_tile == player_1, "the slow drag missed a unit, so the comparison is vacuous"
    assert run([start, end])[1] == tile_by_tile


def test_a_right_button_path_lowers_every_tile() -> None:
    from PyQt5.QtCore import Qt

    window = _edit_window("elevation")
    try:
        mm = window.scenario.map_manager
        for t in mm.terrain:
            t.elevation = 2
        _shown_flat(window)
        map_view = window.map_view
        _press(map_view, (10, 10), Qt.RightButton)
        _move(map_view, (14, 10), Qt.RightButton)
        _release(map_view, (14, 10), Qt.RightButton)
        lowered = {(t.x, t.y) for t in mm.terrain if t.elevation < 2}
        assert {(x, 10) for x in range(10, 15)} <= lowered
    finally:
        _close(window)


# n_cliff_default_x1's 01 piece: 3x3, so a diagonal drag crosses several lattice cells.
_DEFAULT_CLIFF = 264
_CLIFF_FRAME = 3


def _cliff_window():
    from descape import cliff_catalog

    window = _units_window("Terrain", "cliff")
    families = cliff_catalog.families()
    file_name = next(n for n, pieces in families.items() if any(p.unit_const == _DEFAULT_CLIFF for p in pieces))
    window.cliff_family_combo.setCurrentIndex(window.cliff_family_combo.findData(file_name))
    window.cliff_piece_combo.setCurrentIndex(window.cliff_piece_combo.findData(_DEFAULT_CLIFF))
    window.cliff_frame_spin.setValue(_CLIFF_FRAME)
    return window


def _cliff_drag(tiles) -> list[tuple[int, float, float, float]]:
    """The cliffs one drag placed, in placement order. A fresh window per drag:
    a previous run's cliffs would feed the next one's existing-cliff scan."""
    window = _cliff_window()
    try:
        gaia = window.scenario.unit_manager.units[0]
        before = len(gaia)
        _drag(window.map_view, tiles)
        return [(u.unit_const, u.x, u.y, u.rotation) for u in gaia[before:]]
    finally:
        _close(window)


def test_a_compressed_diagonal_cliff_drag_lays_the_tile_by_tile_chain() -> None:
    """The chain walk alone fills a two-event drag without holes but cuts the
    corner (diagonal steps, then straight); the gap-filled path follows the line."""
    from descape import cliff_chain

    start, end = (30, 30), (48, 36)
    tile_by_tile = _cliff_drag([start, *line_tiles(*start, *end)])
    assert len(tile_by_tile) > 2
    assert _cliff_drag([start, end]) == tile_by_tile

    # What the endpoints alone give, so the equality above isn't vacuous.
    stroke = cliff_chain.ChainStroke(
        map_width=120, map_height=120, existing=[], fallback_const=_DEFAULT_CLIFF, fallback_frame=_CLIFF_FRAME
    )
    stroke.add_cursor_tile(*start)
    stroke.add_cursor_tile(*end)
    corner_cut = [(n.x, n.y) for n in stroke.nodes()]
    assert corner_cut != [(x, y) for _, x, y, _ in tile_by_tile]


def test_a_one_tile_cliff_click_is_unchanged() -> None:
    assert _cliff_drag([(30, 30)]) == [(_DEFAULT_CLIFF, 31.5, 31.5, float(_CLIFF_FRAME))]
