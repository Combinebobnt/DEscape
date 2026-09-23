"""Draw Line / Draw Rectangle wiring, driven through a real offscreen
ViewerWindow -- the Qt half that tests/test_shape_tools.py's pure geometry
cannot reach: tool registration, the press-drag-release routing in MapView,
the single-record commit, the preview's lifetime, and every cancel path.

Same offscreen technique tests/test_ruler_viewer.py documents, and the same
drag helpers tests/test_region_select.py uses: MapView's own handlers are
called directly with real QMouseEvents rather than posted through the
session.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

from descape import settings, shape_tools, viewer_common
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _shape_window(tool: str = "draw_line", style: str = "Flat"):
    """A shown window with the blank template loaded and `tool` active.
    Caller must edit_history.mark_saved() + close()."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    # Unchecked before the style switch, while still in Stepped -- see
    # tests/test_brush_stroke.py's _shown_flat for why.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_TERRAIN)
    # This file tests shape mechanics, not descape/terrain_units.py -- Trees
    # defaults on and would pop an unpatched large-edit confirm QMessageBox
    # for the bigger rectangles below, hanging the offscreen run.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.show()
    QApplication.processEvents()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _mouse_event(kind, pos, button, buttons, modifiers=None):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier if modifiers is None else modifiers)


def _press(map_view, tile, button=None, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    map_view.mousePressEvent(
        _mouse_event(QEvent.MouseButtonPress, conftest.polygon_viewport_pos(map_view, *tile), button, button, modifiers)
    )


def _move(map_view, tile, buttons=None, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    buttons = Qt.LeftButton if buttons is None else buttons
    map_view.mouseMoveEvent(
        _mouse_event(QEvent.MouseMove, conftest.polygon_viewport_pos(map_view, *tile), Qt.NoButton, buttons, modifiers)
    )


def _release(map_view, tile, button=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    map_view.mouseReleaseEvent(
        _mouse_event(QEvent.MouseButtonRelease, conftest.polygon_viewport_pos(map_view, *tile), button, Qt.NoButton)
    )


def _drag(map_view, a, b, *, modifiers=None) -> None:
    _press(map_view, a, modifiers=modifiers)
    _move(map_view, b, modifiers=modifiers)
    _release(map_view, b)


def _painted(window) -> set[tuple[int, int]]:
    """Every tile now carrying _TERRAIN. The blank template is uniformly
    terrain_id 0, so this is exactly what the gesture painted."""
    mm = window.scenario.map_manager
    return {
        (x, y)
        for y in range(mm.map_height)
        for x in range(mm.map_width)
        if mm.get_tile(x, y).terrain_id == _TERRAIN
    }


# -- registration ------------------------------------------------------------


def test_both_shape_tools_are_registered_terrain_tools() -> None:
    by_id = {t.tool_id: t for t in settings.TOOLS}
    for tool_id, shape in (("draw_line", "line"), ("draw_rect", "rect")):
        tool = by_id[tool_id]
        assert tool.drag_shape == shape
        assert tool.modes == ("terrain",)
        assert tool.is_edit_tool and tool.param_widget == "terrain"
        # Both ship unbound: every bare letter is taken, and a duplicate
        # QKeySequence silently kills both actions.
        assert tool.default_key == ""
        assert tool_id in viewer_common.SHAPE_TOOLS


def test_the_terrain_keybind_rows_stay_contiguous() -> None:
    """_build_keybinds_tab only compares against the previous row, so a
    terrain tool inserted outside the block would grow a second header."""
    tool_rows = [a for a, _, _ in settings.REBINDABLE_ACTIONS if a.startswith("tool_")]
    terrain_ids = [t.tool_id for t in settings.TOOLS if t.modes == ("terrain",)]
    positions = [tool_rows.index(f"tool_{t}") for t in terrain_ids]
    assert positions == sorted(positions)
    assert positions == list(range(min(positions), max(positions) + 1))


# -- the gesture -------------------------------------------------------------


# Every style, not just Flat: the preview's path build runs _tile_polygon()
# per tile, which in Sloped reads four corner_rise values off the chunk
# cache, and _pick_tile answers from a pick plane rather than by division.
# Neither is exercised at all by a Flat-only drag.
@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_a_line_drag_paints_exactly_the_rasterized_line(style: str) -> None:
    window = _shape_window("draw_line", style)
    try:
        mm = window.scenario.map_manager
        _drag(window.map_view, (4, 4), (12, 8))
        assert _painted(window) == set(
            shape_tools.line_tiles(4, 4, 12, 8, mm.map_width, mm.map_height)
        )
    finally:
        _close(window)


def test_a_line_drag_is_one_undo_record() -> None:
    window = _shape_window("draw_line")
    try:
        before = len(window.edit_history.records)
        _drag(window.map_view, (4, 4), (20, 12))
        assert len(window.edit_history.records) == before + 1
        window.undo()
        assert _painted(window) == set()
    finally:
        _close(window)


def test_a_filled_rectangle_paints_its_whole_area() -> None:
    window = _shape_window("draw_rect")
    try:
        window.rect_fill_combo.setCurrentText("Filled")
        _drag(window.map_view, (3, 5), (8, 9))
        assert _painted(window) == {(x, y) for y in range(5, 10) for x in range(3, 9)}
    finally:
        _close(window)


def test_an_outlined_rectangle_paints_its_border_only() -> None:
    window = _shape_window("draw_rect")
    try:
        window.rect_fill_combo.setCurrentText("Outline")
        _drag(window.map_view, (3, 5), (8, 9))
        painted = _painted(window)
        assert (3, 5) in painted and (8, 9) in painted
        assert (5, 7) not in painted  # interior stays untouched
        assert len(painted) == 2 * (6 + 5) - 4
    finally:
        _close(window)


def test_a_reversed_drag_normalizes_the_rectangle() -> None:
    window = _shape_window("draw_rect")
    try:
        window.rect_fill_combo.setCurrentText("Filled")
        _drag(window.map_view, (8, 9), (3, 5))
        assert _painted(window) == {(x, y) for y in range(5, 10) for x in range(3, 9)}
    finally:
        _close(window)


def test_a_zero_length_drag_paints_one_brush_footprint() -> None:
    """A click that never moves must still paint -- "click does nothing" is
    the worse surprise for a mutating tool. Deliberately NOT the Ruler's
    stay-armed behaviour."""
    window = _shape_window("draw_line")
    try:
        _drag(window.map_view, (6, 6), (6, 6))
        assert _painted(window) == {(6, 6)}
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_a_shrinking_rubber_band_leaves_no_residue(style: str) -> None:
    """The property the per-tile stroke path structurally cannot have: a
    stroke paints as it goes and cannot un-paint, so dragging back over the
    anchor would strand everything the band has since left behind."""
    window = _shape_window("draw_rect", style)
    try:
        window.rect_fill_combo.setCurrentText("Filled")
        map_view = window.map_view
        _press(map_view, (10, 10))
        _move(map_view, (20, 20))
        _move(map_view, (14, 14))
        _move(map_view, (11, 11))
        _release(map_view, (11, 11))
        assert _painted(window) == {(10, 10), (11, 10), (10, 11), (11, 11)}
    finally:
        _close(window)


def test_nothing_is_written_until_release() -> None:
    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        before = len(window.edit_history.records)
        _press(map_view, (4, 4))
        _move(map_view, (14, 4))
        assert _painted(window) == set()
        assert len(window.edit_history.records) == before
        _release(map_view, (14, 4))
        assert _painted(window)
    finally:
        _close(window)


def test_a_line_honours_the_brush() -> None:
    window = _shape_window("draw_line")
    try:
        window.brush_size_spin.setValue(3)
        _drag(window.map_view, (6, 6), (10, 6))
        painted = _painted(window)
        # A size-3 square brush along a 5-tile horizontal run: 3 rows deep.
        assert painted == {(x, y) for y in range(5, 8) for x in range(5, 12)}
    finally:
        _close(window)


def test_a_filled_rectangle_ignores_the_brush() -> None:
    """Filled mode hides the brush params, so the painted set must not be
    dilated by whatever size the spinbox was last left at."""
    window = _shape_window("draw_rect")
    try:
        window.brush_size_spin.setValue(5)
        window.rect_fill_combo.setCurrentText("Filled")
        _drag(window.map_view, (10, 10), (13, 13))
        assert _painted(window) == {(x, y) for y in range(10, 14) for x in range(10, 14)}
    finally:
        _close(window)


# -- Shift -------------------------------------------------------------------


def test_shift_snaps_a_line_to_a_primitive_direction() -> None:
    from PyQt5.QtCore import Qt

    window = _shape_window("draw_line")
    try:
        mm = window.scenario.map_manager
        # (10, 3) off a (20, 20) anchor is nearest the 2:1 primitive, and
        # snaps to a whole multiple of it rather than to the raw endpoint.
        _drag(window.map_view, (20, 20), (30, 23), modifiers=Qt.ShiftModifier)
        dx, dy = shape_tools.snap_line_delta(10, 3)
        assert _painted(window) == set(
            shape_tools.line_tiles(20, 20, 20 + dx, 20 + dy, mm.map_width, mm.map_height)
        )
    finally:
        _close(window)


def test_shift_squares_a_rectangle_toward_the_drag() -> None:
    from PyQt5.QtCore import Qt

    window = _shape_window("draw_rect")
    try:
        window.rect_fill_combo.setCurrentText("Filled")
        _drag(window.map_view, (30, 30), (24, 27), modifiers=Qt.ShiftModifier)
        # max(|dx|, |dy|) == 6, each axis keeping its own (negative) sign.
        assert _painted(window) == {(x, y) for y in range(24, 31) for x in range(24, 31)}
    finally:
        _close(window)


def test_shift_pressed_mid_drag_updates_the_preview_without_a_move() -> None:
    """Stage 5's whole point: mouseMoveEvent already carries modifiers, so
    Shift-then-move works for free -- pressing Shift and holding still is
    the case that needs the key handlers."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        _press(map_view, (20, 20))
        _move(map_view, (30, 23))
        unsnapped = map_view._shape_span()
        map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Shift, Qt.ShiftModifier))
        snapped = map_view._shape_span()
        assert snapped != unsnapped
        assert snapped == (20, 20, *[20 + d for d in shape_tools.snap_line_delta(10, 3)])
        map_view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Shift, Qt.NoModifier))
        assert map_view._shape_span() == unsnapped
    finally:
        _close(window)


# -- preview and cancel paths ------------------------------------------------


def test_the_preview_appears_mid_drag_and_is_gone_after_release() -> None:
    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 15))
        assert map_view._highlight_outline_item is not None
        assert len(map_view._shape_tiles(preview=True)) > 1
        _release(map_view, (15, 15))
        assert map_view._shape_anchor is None
        assert map_view._highlight_outline_item is None
    finally:
        _close(window)


def test_a_filled_rectangle_previews_its_perimeter_only() -> None:
    """Flag B: one rebuild of a large filled rectangle is hundreds of
    thousands of _tile_polygon() calls, which freezes the drag. The ring is
    what reads as "the rectangle" anyway; the COMMITTED set stays exact."""
    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        window.rect_fill_combo.setCurrentText("Filled")
        _press(map_view, (10, 10))
        _move(map_view, (40, 40))
        preview = map_view._shape_tiles(preview=True)
        committed = map_view._shape_tiles(preview=False)
        assert len(preview) == 2 * (31 + 31) - 4
        assert len(committed) == 31 * 31
        assert set(preview) < set(committed)
        _release(map_view, (40, 40))
    finally:
        _close(window)


def test_escape_cancels_a_drag_without_painting() -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 15))
        map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert map_view._shape_anchor is None
        _release(map_view, (15, 15))
        assert _painted(window) == set()
    finally:
        _close(window)


def test_the_right_button_cancels_a_drag() -> None:
    from PyQt5.QtCore import Qt

    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 5))
        _press(map_view, (15, 5), button=Qt.RightButton)
        assert map_view._shape_anchor is None
        _release(map_view, (15, 5))
        assert _painted(window) == set()
    finally:
        _close(window)


def test_a_right_button_press_never_starts_a_shape() -> None:
    """_touch_tile ORs ShiftModifier in for right-button strokes, so a
    right-drag reaching this path would read as a phantom constrain."""
    from PyQt5.QtCore import Qt

    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5), button=Qt.RightButton)
        assert map_view._shape_anchor is None
        _move(map_view, (15, 15), buttons=Qt.RightButton)
        _release(map_view, (15, 15), button=Qt.RightButton)
        assert _painted(window) == set()
    finally:
        _close(window)


def test_a_tool_switch_mid_drag_cancels_rather_than_commits() -> None:
    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 15))
        window._on_tool_selected("draw")
        assert map_view._shape_anchor is None
        assert _painted(window) == set()
    finally:
        _close(window)


def test_a_brush_change_mid_drag_keeps_the_preview() -> None:
    """The easiest cancel path to miss: ]/[ mid-drag reaches
    refresh_highlight(), whose EDIT_TOOLS branch would replace the rubber
    band with a plain brush footprint."""
    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        _press(map_view, (6, 6))
        _move(map_view, (16, 6))
        window.brush_size_spin.setValue(3)
        assert map_view._shape_anchor == (6, 6)
        assert map_view._shape_end == (16, 6)
        _release(map_view, (16, 6))
        # The new brush is honoured by the commit, not discarded.
        assert (6, 5) in _painted(window)
    finally:
        _close(window)


def test_leaving_the_viewport_with_no_button_held_cancels() -> None:
    """leaveEvent gets a bare QEvent with no button state, so the guard asks
    QApplication.mouseButtons(). Under a synthesized press nothing is
    physically held, which is exactly the focus-lost case the cancel exists
    for. The other half -- a real held button keeping the drag alive while
    the cursor runs past the viewport edge -- is not synthesizable here and
    is in the in-app list instead."""
    from PyQt5.QtCore import QEvent

    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 5))
        assert map_view._shape_anchor is not None
        map_view.leaveEvent(QEvent(QEvent.Leave))
        assert map_view._shape_anchor is None
        _release(map_view, (15, 5))
        assert _painted(window) == set()
    finally:
        _close(window)


def test_closing_the_map_mid_drag_drops_the_shape() -> None:
    window = _shape_window("draw_rect")
    try:
        map_view = window.map_view
        _press(map_view, (5, 5))
        _move(map_view, (15, 15))
        window.edit_history.mark_saved()
        window.close_scenario()
        assert map_view._shape_anchor is None
    finally:
        _close(window)


# -- undo --------------------------------------------------------------------


def test_a_mid_drag_undo_is_safe() -> None:
    """A free property of committing at release: nothing is mutated until
    then, and apply() is begin+commit in one call, so there is no open
    snapshot for undo() to corrupt. The stroke tools do not have this."""
    window = _shape_window("draw_line")
    try:
        map_view = window.map_view
        # One committed edit to undo into.
        _drag(map_view, (2, 2), (6, 2))
        painted_first = _painted(window)
        assert painted_first

        _press(map_view, (20, 20))
        _move(map_view, (30, 20))
        window.undo()
        assert _painted(window) == set()
        _release(map_view, (30, 20))
        # The interrupted drag still commits cleanly on its own terms.
        mm = window.scenario.map_manager
        assert _painted(window) == set(
            shape_tools.line_tiles(20, 20, 30, 20, mm.map_width, mm.map_height)
        )
    finally:
        _close(window)
