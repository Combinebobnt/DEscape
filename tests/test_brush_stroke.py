"""Brush size/shape wiring for the drag-stroke edit tools (Draw, Elevate,
Set Elevation), driven through a real offscreen ViewerWindow -- same
technique and default-tier rationale as tests/test_fill_tool.py and
tests/test_toolbar_params.py. tests/test_brush.py covers the pure geometry;
tests/test_elevation_tools.py covers the batched-elevation correctness fix;
this module covers the actual stroke wiring in descape/viewer.py (the
footprint loop inside on_edit_stroke_tile, the painted-tile dedupe, and the
hover preview).

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape.brush import BRUSH_SHAPE_CIRCLE, BRUSH_SHAPE_SQUARE
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _edit_window(tool: str = "draw"):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected(tool)
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_TERRAIN))
    # This file tests brush/stroke mechanics, not descape/terrain_units.py --
    # Trees defaults on and would otherwise pop an unpatched large-fill
    # confirm QMessageBox for the whole-map fills below, hanging offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _shown_flat(window) -> None:
    from PyQt5.QtWidgets import QApplication

    # Unchecked BEFORE the style switch, while still in Stepped -- iso_action
    # defaults checked (MapView._isometric's own default), so Flat would
    # otherwise render through the Flat+Isometric plan's real-iso path
    # instead of the plain top-down canvas this module means to exercise.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show()
    QApplication.processEvents()


def _mouse_event(kind, pos, button, buttons):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)


def _stroke(window, cx: int, cy: int, modifiers: int = 0) -> None:
    """A single-cursor-tile stroke, called directly rather than through real
    mouse events -- matches tests/test_fill_tool.py's
    test_mid_drag_tool_switch...'s own direct-call style for the cases that
    don't need to exercise MapView's own event routing."""
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(cx, cy, modifiers)
    window.on_edit_stroke_end()


def test_square_brush_paints_whole_footprint_in_one_undo_record() -> None:
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(3)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        mm = window.scenario.map_manager

        _stroke(window, 10, 10)

        painted = [t for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert len(painted) == 9
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_circle_brush_paints_exactly_the_circle_footprint() -> None:
    from descape.brush import brush_tiles

    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
        mm = window.scenario.map_manager

        _stroke(window, 40, 40)

        expected = set(brush_tiles(40, 40, 5, BRUSH_SHAPE_CIRCLE, mm.map_width, mm.map_height))
        painted = {(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN}
        assert painted == expected
        # The 4 corners of the bounding 5x5 square are excluded by the
        # circle shape -- confirm they're specifically untouched, not just
        # that the count matches.
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            assert mm.get_tile(40 + dx, 40 + dy).terrain_id != _TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_elevate_drag_raises_each_tile_at_most_once_per_stroke() -> None:
    """The load-bearing regression test. Cursor-tile dedupe
    (MapView._stroke_touched) is NOT the same as painted-tile dedupe
    (ViewerWindow._stroke_painted) once a brush is bigger than one tile: a
    3x3 brush dragged across several cursor tiles has painted tiles that
    fall under more than one cursor position. Without _stroke_painted,
    Elevate's accumulating +1 would raise those overlapping tiles more than
    once in a single stroke. Driven through real QMouseEvents (not direct
    on_edit_stroke_tile calls) so MapView's own _touch_tile dedupe is
    genuinely exercised, matching test_fill_tool.py's
    test_drag_after_click_fills_only_once."""
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("elevation")
    try:
        window.brush_size_spin.setValue(3)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        _shown_flat(window)
        map_view = window.map_view
        mm = window.scenario.map_manager

        cx, cy = 10, 10
        press_pos = conftest.viewport_pos(map_view, cx, cy)
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, press_pos, Qt.LeftButton, Qt.LeftButton))

        # Drag right one cursor tile at a time -- each step's 3x3 footprint
        # overlaps the previous step's by two columns.
        for step in range(1, 5):
            move_pos = conftest.viewport_pos(map_view, cx + step, cy)
            map_view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, move_pos, Qt.NoButton, Qt.LeftButton))

        map_view.mouseReleaseEvent(
            _mouse_event(QEvent.MouseButtonRelease, press_pos, Qt.LeftButton, Qt.NoButton)
        )

        touched = [
            t.elevation
            for y in range(cy - 2, cy + 3)
            for x in range(cx - 2, cx + 7)
            for t in [mm.get_tile(x, y)]
            if t.elevation != 0
        ]
        assert touched, "expected the drag to have raised at least one tile"
        assert max(touched) == 1, f"a tile was raised more than once in a single stroke: elevations {touched}"
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_footprint_clipped_at_map_corner_does_not_raise() -> None:
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(9)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))
        mm = window.scenario.map_manager

        _stroke(window, 0, 0)  # no exception -- footprint clips to the map

        painted = [t for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert 0 < len(painted) < 81
        for t in mm.terrain:
            if t.terrain_id == _TERRAIN:
                assert 0 <= t.x < mm.map_width
                assert 0 <= t.y < mm.map_height
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_size_1_matches_pre_brush_single_tile_behavior() -> None:
    window = _edit_window("draw")
    try:
        assert window.brush_size_spin.value() == 1  # the default
        mm = window.scenario.map_manager

        _stroke(window, 20, 20)

        painted = [(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN]
        assert painted == [(20, 20)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_ignores_brush_size() -> None:
    window = _edit_window("fill")
    try:
        window.brush_size_spin.setValue(9)  # left over from a prior tool selection
        mm = window.scenario.map_manager

        window.on_fill(0, 0, 0)

        # A flood fill, not a 9x9 patch -- covers the whole uniform map.
        assert all(t.terrain_id == _TERRAIN for t in mm.terrain)
        assert not window.brush_size_spin_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_hover_preview_matches_the_stroke_footprint() -> None:
    """Checks the actual QPainterPath geometry, not just the memo key --
    Flat mode's axis-aligned _tile_polygon makes tile-center containment a
    reliable, simple check. Also confirms the painted set (after a real
    stroke at the same cursor tile) equals the same expected footprint, so
    the preview and the edit are shown to agree, not just each independently
    match descape.brush's own output."""
    from PyQt5.QtCore import QPointF

    from descape.brush import brush_tiles

    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
        _shown_flat(window)
        mm = window.scenario.map_manager
        map_view = window.map_view
        tp = map_view._tile_pixels

        map_view._update_highlight(40, 40)
        expected = set(brush_tiles(40, 40, 5, BRUSH_SHAPE_CIRCLE, mm.map_width, mm.map_height))
        assert map_view._highlight_key == (40, 40, 5, BRUSH_SHAPE_CIRCLE)
        assert map_view._highlight_outline_item is not None

        path = map_view._highlight_outline_item.path()
        for tx, ty in expected:
            center = QPointF((tx + 0.5) * tp, (ty + 0.5) * tp)
            assert path.contains(center), f"expected highlight to cover tile ({tx}, {ty})"
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:  # circle-excluded bounding-box corners
            corner = QPointF((40 + dx + 0.5) * tp, (40 + dy + 0.5) * tp)
            assert not path.contains(corner), f"highlight should exclude the corner at offset ({dx}, {dy})"

        _stroke(window, 40, 40)
        painted = {(t.x, t.y) for t in mm.terrain if t.terrain_id == _TERRAIN}
        assert painted == expected
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_size_change_refreshes_preview_without_a_mouse_move() -> None:
    window = _edit_window("draw")
    try:
        window.on_hover((40, 40))
        map_view = window.map_view

        map_view._update_highlight(40, 40)
        assert map_view._highlight_key == (40, 40, 1, BRUSH_SHAPE_SQUARE)

        window.brush_size_spin.setValue(5)  # no mouse move in between
        assert map_view._highlight_key == (40, 40, 5, BRUSH_SHAPE_SQUARE)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_resets_to_size_1_square_on_a_fresh_window() -> None:
    """Guards the no-persistence requirement: brush state is session-only,
    with no settings.py config key, so a fresh window always starts at
    size 1 / square regardless of what a previous window in the same
    process left the spinbox/combo at."""
    window = _edit_window("draw")
    try:
        window.brush_size_spin.setValue(9)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_CIRCLE))
    finally:
        window.edit_history.mark_saved()
        window.close()

    fresh = _edit_window("draw")
    try:
        assert fresh.brush_size_spin.value() == 1
        assert fresh.brush_shape_combo.currentData() == BRUSH_SHAPE_SQUARE
    finally:
        fresh.edit_history.mark_saved()
        fresh.close()
