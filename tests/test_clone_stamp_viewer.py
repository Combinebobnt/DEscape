"""The Clone Stamp (GH #285) through a real offscreen ViewerWindow: the Ctrl+press
source, the stroke in descape/viewer_clone_stamp.py, its Tool Options row, its
marker and its undo record. The copy and the seam are tests/test_clone_stamp.py.

Every ViewerWindow here closes through conftest.close_window().
"""

from __future__ import annotations

import random

import pytest
from test_brush_stroke import _shown_flat
from test_clone_stamp import _steep_legal
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- tree_sprites requests it
from test_terrain_units_viewer import (
    _check_against_fresh,
    _render_levels,
    _splice_any_batch,
    _spy,
    _styled_window,
    tree_sprites,  # noqa: F401 -- fixture, requested by name
)

from descape import brush
from descape.edit_history import CompositeDiffRecord, TileDiffRecord, tile_state
from descape.mirror_tools import _elevation_violations
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.viewer_clone_stamp import ILLEGAL_SOURCE_WARNING

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_FOREST_OAK = 10  # density 1000/1000: one tree per tile when painted as a base
_FOREST_OAK_CONST = 411


def _window(size: int = 1):
    window = conftest.terrain_edit_window()
    window.clone_action.setChecked(True)
    assert window._current_tool == "clone"
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.brush_size_spin.setValue(size)
    window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(brush.BRUSH_SHAPE_SQUARE))
    return window


def _tile(window, x: int, y: int):
    mm = window.scenario.map_manager
    return mm.terrain[y * mm.map_width + x]


def _states(window) -> list:
    return [tile_state(t) for t in window.scenario.map_manager.terrain]


def _elevations(window) -> list[int]:
    return [t.elevation for t in window.scenario.map_manager.terrain]


def _stroke(window, tiles) -> None:
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def _lines(window) -> list[str]:
    return window.status_log.toPlainText().splitlines()


def _hill(window, cx: int, cy: int, top: int, terrain_id: int | None = None, radius: int = 2) -> None:
    """A legal Chebyshev hill: `top` on a (2*radius+1) square plateau, falling one level a tile."""
    mm = window.scenario.map_manager
    for i, tile in enumerate(mm.terrain):
        x, y = i % mm.map_width, i // mm.map_width
        d = max(abs(x - cx), abs(y - cy))
        tile.elevation = max(tile.elevation, top - max(0, d - radius))
        if terrain_id is not None and d <= radius:
            tile.terrain_id = terrain_id


def _ctrl_press(map_view, tile, button) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QMouseEvent

    pos = conftest.viewport_pos(map_view, *tile)
    map_view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, button, button, Qt.ControlModifier))


def _mouse(map_view, kind: str, tile, button) -> None:
    from PyQt5.QtCore import QEvent, Qt

    pos = conftest.viewport_pos(map_view, *tile)
    if kind == "press":
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, button, button))
    elif kind == "move":
        map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, button))
    else:
        map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, button, Qt.NoButton))


# -- the gestures -------------------------------------------------------------


@pytest.mark.parametrize("button", ["left", "right"])
def test_a_ctrl_press_sets_the_source_and_records_nothing(button: str) -> None:
    from PyQt5.QtCore import Qt

    qt_button = Qt.LeftButton if button == "left" else Qt.RightButton
    window = _window()
    try:
        _shown_flat(window)
        before = _states(window)
        _ctrl_press(window.map_view, (12, 7), qt_button)
        assert window._clone.anchor == (12, 7)
        assert not window.map_view._stroke_active and not window.edit_history.in_stroke
        _mouse(window.map_view, "release", (12, 7), qt_button)
        assert window.edit_history.records == []
        assert _states(window) == before
        assert _lines(window)[-1] == "Clone source set at (12, 7)"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("button", ["left", "right"])
def test_a_drag_through_mapview_clones_at_the_offset(button: str) -> None:
    """Ctrl+click (10, 6), then drag from (30, 8) to (36, 8) in one jump: the gap fills too."""
    from PyQt5.QtCore import Qt

    qt_button = Qt.LeftButton if button == "left" else Qt.RightButton
    window = _window()
    try:
        _shown_flat(window)
        for x in range(10, 17):
            _tile(window, x, 6).terrain_id = 40 + x
            _tile(window, x, 8).terrain_id = 20 + x  # under the drag's row offset, never the source
        _ctrl_press(window.map_view, (10, 6), Qt.LeftButton)
        _mouse(window.map_view, "press", (30, 8), qt_button)
        _mouse(window.map_view, "move", (36, 8), qt_button)
        _mouse(window.map_view, "release", (36, 8), qt_button)
        assert window._clone.offset == (20, 2)
        assert [_tile(window, x, 8).terrain_id for x in range(30, 37)] == [40 + x for x in range(10, 17)]
        assert len(window.edit_history.records) == 1
    finally:
        conftest.close_window(window)


def test_a_press_with_no_source_opens_no_stroke_and_says_how_to_set_one() -> None:
    window = _window()
    try:
        _stroke(window, [(20, 20), (21, 20)])
        assert window.edit_history.records == [] and not window.edit_history.in_stroke
        assert _lines(window)[-1] == "Ctrl+click a tile to set the clone source"
    finally:
        conftest.close_window(window)


def test_nothing_ticked_opens_no_stroke_and_fixes_no_offset() -> None:
    """Terrain unticked on a non-square map: Elevation stays ticked but cannot apply."""
    window = _window()
    try:
        window.scenario.map_is_square = False
        window._update_tool_enabled()
        assert window.clone_action.isEnabled()
        assert window.clone_elevation_check.isChecked() and not window.clone_elevation_check.isEnabled()
        window.clone_terrain_check.setChecked(False)
        window.set_clone_source((5, 5))
        _stroke(window, [(40, 40)])
        assert window.edit_history.records == [] and not window.edit_history.in_stroke
        assert _lines(window)[-1] == "Tick Terrain or Elevation to clone"
        assert window._clone.offset is None, "a refused press fixed the offset"
        window.clone_terrain_check.setChecked(True)
        _tile(window, 6, 5).terrain_id = 9
        _stroke(window, [(50, 50), (51, 50)])
        assert window._clone.offset == (45, 45)
        assert _tile(window, 51, 50).terrain_id == 9
    finally:
        conftest.close_window(window)


def test_the_first_press_fixes_the_offset_and_a_second_stroke_keeps_it() -> None:
    """The first event carries two tiles, so the press is its first, not its last."""
    window = _window()
    try:
        for x in range(30):
            _tile(window, x, 3).terrain_id = 1 + x % 40
        window.set_clone_source((2, 3))
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 9), (21, 9)], 0)
        window.on_edit_stroke_end()
        assert window._clone.offset == (18, 6)
        assert [_tile(window, x, 9).terrain_id for x in (20, 21)] == [3, 4]
        _stroke(window, [(30, 9), (31, 9)])
        assert window._clone.offset == (18, 6)
        assert [_tile(window, x, 9).terrain_id for x in (30, 31)] == [13, 14]
        assert len(window.edit_history.records) == 2
        assert _lines(window)[-1] == "Clone Stamp: 2 tiles from offset (18, 6)"
    finally:
        conftest.close_window(window)


def test_an_overlapping_stroke_copies_the_map_as_it_was_at_the_press() -> None:
    """Offset (1, 0) dragged east: each dab's source is the tile the last dab wrote."""
    window = _window()
    try:
        for x in range(4, 16):
            _tile(window, x, 10).terrain_id = 1 + x
        window.set_clone_source((4, 10))
        _stroke(window, [(x, 10) for x in range(5, 16)])
        assert [_tile(window, x, 10).terrain_id for x in range(5, 16)] == [1 + x - 1 for x in range(5, 16)]
    finally:
        conftest.close_window(window)


def test_a_layered_tile_copies_its_pair_verbatim() -> None:
    window = _window()
    try:
        source = _tile(window, 8, 8)
        source.terrain_id, source.layer = 9, 4
        window.set_clone_source((8, 8))
        _stroke(window, [(30, 30)])
        assert (_tile(window, 30, 30).terrain_id, _tile(window, 30, 30).layer) == (9, 4)
    finally:
        conftest.close_window(window)


# -- elevation ----------------------------------------------------------------


def test_one_stroke_is_one_undo_record_and_undo_restores_every_tile_its_ring_included() -> None:
    window = _window(size=5)
    try:
        _hill(window, 20, 20, 8, terrain_id=3)
        before = _states(window)
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60), (61, 60), (62, 60)])
        after = _states(window)
        assert len(window.edit_history.records) == 1
        record = window.edit_history.records[0]
        assert isinstance(record, TileDiffRecord) and record.label == "Clone Stamp"
        # (65, 60) is one tile past the dab, which cloned (64, 60) from (24, 20) at 6.
        assert _tile(window, 61, 60).elevation == 8 and _tile(window, 65, 60).elevation == 5
        ring = [i for i, (old, new) in enumerate(zip(before, after, strict=True)) if old != new and old[0] == new[0]]
        assert len(ring) > 20, "no clamp ring"
        window.undo()
        assert _states(window) == before
        window.redo()
        assert _states(window) == after
    finally:
        conftest.close_window(window)


def test_an_elevation_clone_onto_uneven_ground_leaves_no_illegal_step() -> None:
    window = _window(size=5)
    try:
        rng = random.Random(7)
        e0 = _steep_legal(rng, 120, 120)
        for tile, value in zip(window.scenario.map_manager.terrain, e0.ravel().tolist(), strict=True):
            tile.elevation = value
        assert not _elevation_violations(_elevations(window), 120, 120)
        window.set_clone_source((30, 30))
        _stroke(window, [(70 + i, 64 + i // 2) for i in range(14)])
        assert _elevations(window) != e0.ravel().tolist()
        assert not _elevation_violations(_elevations(window), 120, 120)
        assert ILLEGAL_SOURCE_WARNING not in _lines(window)
    finally:
        conftest.close_window(window)


def test_elevation_unticked_leaves_every_elevation_alone() -> None:
    window = _window(size=3)
    try:
        _hill(window, 20, 20, 6, terrain_id=3)
        elevations = _elevations(window)
        window.clone_elevation_check.setChecked(False)
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60)])
        assert _elevations(window) == elevations
        assert _tile(window, 60, 60).terrain_id == 3
    finally:
        conftest.close_window(window)


def test_on_a_non_square_map_the_elevation_box_is_off_and_the_stroke_clones_terrain_only() -> None:
    window = _window(size=3)
    try:
        _hill(window, 20, 20, 6, terrain_id=3)
        elevations = _elevations(window)
        window.scenario.map_is_square = False
        window._update_tool_enabled()
        assert window.clone_elevation_check.isChecked() and not window.clone_elevation_check.isEnabled()
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60)])
        assert _elevations(window) == elevations
        assert _tile(window, 60, 60).terrain_id == 3
    finally:
        conftest.close_window(window)


def test_an_illegal_source_step_is_copied_and_warned_about() -> None:
    window = _window(size=3)
    try:
        _tile(window, 20, 20).elevation = 3
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60)])
        assert _tile(window, 60, 60).elevation == 3
        assert _lines(window)[-1] == ILLEGAL_SOURCE_WARNING
        assert len(window.edit_history.records) == 1, "the edit should stay, one undo away"
    finally:
        conftest.close_window(window)


# -- trees and the row --------------------------------------------------------


def test_a_forest_clone_with_trees_ticked_is_one_composite_record() -> None:
    window = _window(size=3)
    try:
        window.paint_trees_check.setChecked(True)
        _hill(window, 20, 20, 4, terrain_id=_FOREST_OAK, radius=1)
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60)])
        (record,) = window.edit_history.records
        assert isinstance(record, CompositeDiffRecord)
        trees = [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
        assert {(int(u.x), int(u.y)) for u in trees} == {(x, y) for x in range(59, 62) for y in range(59, 62)}
        window.undo()
        assert not window.scenario.unit_manager.units[0]
    finally:
        conftest.close_window(window)


def test_trees_and_eye_candy_follow_the_terrain_tick_and_the_draw_only_options_stay_hidden() -> None:
    window = conftest.terrain_edit_window()
    try:
        window.draw_action.setChecked(True)
        window.terrain_target_combo.setCurrentIndex(1)  # Layer: Draw's own Trees box hides
        assert not window.paint_trees_param_action.isVisible()
        window.clone_action.setChecked(True)
        assert window.paint_trees_param_action.isVisible() and window.paint_eye_candy_param_action.isVisible()
        assert window.paint_trees_check.isEnabled()
        for hidden in (
            window.terrain_target_param_action, window.auto_beach_param_action, window.spray_flow_param_action,
            window.replace_flood_param_action, window.level_param_spin_action,
        ):
            assert not hidden.isVisible()
        assert window.clone_terrain_param_action.isVisible() and window.clone_elevation_param_action.isVisible()
        assert window.brush_size_spin_action.isVisible() and window.tool_param_separator_action.isVisible()
        window.clone_terrain_check.setChecked(False)
        assert not window.paint_trees_param_action.isVisible() and not window.paint_eye_candy_param_action.isVisible()
        window.clone_terrain_check.setChecked(True)
        assert window.paint_trees_param_action.isVisible()
        window.draw_action.setChecked(True)
        assert not window.clone_terrain_param_action.isVisible() and not window.clone_elevation_param_action.isVisible()
    finally:
        conftest.close_window(window)


def test_the_tool_greys_with_draw_on_a_locked_document(monkeypatch) -> None:
    window = _window()
    try:
        monkeypatch.setattr(window, "terrain_lock_reason", lambda family=None: "locked")
        window._update_tool_enabled()
        assert not window.clone_action.isEnabled() and not window.draw_action.isEnabled()
        assert window._current_tool == "pan"
        assert "Ctrl+click a tile to set where to copy from" in window.clone_action.toolTip()
    finally:
        conftest.close_window(window)


def test_load_and_close_drop_the_source() -> None:
    window = _window()
    try:
        window.set_clone_source((5, 5))
        window._clone.fix_offset((9, 9))
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert (window._clone.anchor, window._clone.offset) == (None, None)
        window.set_clone_source((5, 5))
        window.close_scenario()
        assert window._clone.anchor is None
    finally:
        conftest.close_window(window)


def test_an_open_mid_stroke_leaves_no_clone_state_for_the_next_stroke() -> None:
    """File > Open mid-drag drops the release; the next Draw stroke must end as Draw's, not as a clone."""
    window = _window()
    try:
        window.set_clone_source((5, 5))
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(8, 8, 0)
        assert window._stroke_clone is not None
        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.draw_action.setChecked(True)
        before = sum(line.startswith("Clone Stamp:") for line in _lines(window))
        _stroke(window, [(3, 3)])
        assert sum(line.startswith("Clone Stamp:") for line in _lines(window)) == before
        assert window.edit_history.records[-1].label == "Paint terrain"
        assert window._stroke_clone is None and window._clone_stroke is None
    finally:
        conftest.close_window(window)


# -- the marker ---------------------------------------------------------------


def _marker_rect(window):
    item = window.map_view._clone_marker_item
    if item is None or not item.isVisible():
        return None
    return item.path().boundingRect()


def _tiles_rect(window, tiles):
    from PyQt5.QtCore import QRectF

    rect = QRectF()
    for x, y in tiles:
        rect = rect.united(window.map_view._tile_polygon(x, y, coarse=True).boundingRect())
    return rect


def test_the_marker_sits_on_the_anchor_then_follows_the_hover_by_the_offset() -> None:
    window = _window(size=3)
    try:
        map_view = window.map_view
        map_view.refresh_highlight((50, 50))
        assert _marker_rect(window) is None, "a marker with no source"
        window.set_clone_source((10, 12))
        map_view.refresh_highlight((50, 50))
        assert _marker_rect(window) == _tiles_rect(window, window._brush_footprint(10, 12))
        pen = map_view._clone_marker_item.pen()
        assert pen.dashPattern() and pen.color() == map_view._highlight_outline_pen.color()
        assert map_view._clone_marker_item.brush().style() == 0  # Qt.NoBrush
        _stroke(window, [(30, 30)])
        map_view.refresh_highlight((40, 35))
        assert _marker_rect(window) == _tiles_rect(window, window._brush_footprint(20, 17))
        map_view.refresh_highlight(None)
        assert _marker_rect(window) is None
        window.draw_action.setChecked(True)
        map_view.refresh_highlight((40, 35))
        assert _marker_rect(window) is None, "the marker outlived its tool"
    finally:
        conftest.close_window(window)


def test_clear_image_drops_the_marker_reference() -> None:
    """File > Close's scene().clear(), called alone so nothing touches the deleted item first."""
    window = _window(size=3)
    try:
        window.set_clone_source((10, 12))
        window.map_view.refresh_highlight((50, 50))
        assert window.map_view._clone_marker_item is not None
        window.map_view.clear_image()
        nulled = window.map_view._clone_marker_item is None
        assert nulled
    finally:
        conftest.close_window(window)


def test_the_marker_survives_a_view_style_switch_and_a_close_and_reopen() -> None:
    """scene().clear() deletes the item: a kept reference would raise RuntimeError on its next use."""
    window = _window(size=3)
    try:
        map_view = window.map_view
        window.set_clone_source((10, 12))
        map_view.refresh_highlight((50, 50))
        assert _marker_rect(window) is not None
        window.terrain_style_combo.setCurrentText("Sloped")
        assert map_view._clone_marker_item is None
        map_view.refresh_highlight((51, 50))
        assert _marker_rect(window) == _tiles_rect(window, window._brush_footprint(10, 12))

        window.close_scenario()
        # A use of a deleted item need not raise (sip can't see a scene-owned item go), so pin the null itself.
        assert map_view._clone_marker_item is None
        # With no file open: every hide path runs on the closed scene.
        window.mode_combo.setCurrentText("View")
        map_view.refresh_clone_marker(None)
        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.mode_combo.setCurrentText("Terrain")
        window.clone_action.setChecked(True)
        map_view.refresh_highlight((50, 50))
        assert _marker_rect(window) is None
        window.set_clone_source((7, 7))
        map_view.refresh_highlight((50, 50))
        assert _marker_rect(window) == _tiles_rect(window, window._brush_footprint(7, 7))
    finally:
        conftest.close_window(window)


# -- Stepped: trees planted on a raised clone ---------------------------------


def test_stepped_an_elevation_clone_of_a_forest_hill_matches_a_fresh_render(tree_sprites, monkeypatch) -> None:  # noqa: F811
    """Design 5's splice point: Draw's live-painted record carries the elevation
    and the trees in one push. The cache's units and every resident level's
    pixels equal a fresh build's, and the trees stand on raised tiles."""
    _splice_any_batch(monkeypatch)
    window = _styled_window("Stepped")
    try:
        window.show_sprites_action.setChecked(True)
        window.clone_action.setChecked(True)
        assert window._current_tool == "clone" and window.paint_trees_check.isChecked()
        window.brush_size_spin.setValue(3)
        _hill(window, 20, 20, 5, terrain_id=_FOREST_OAK, radius=1)
        window.refresh_map()
        # Every oracle level resident at once, mip 0 shown whole (test_terrain_units_viewer's _over_cap_window).
        window._cache.max_bytes *= 8
        _cx0, _cy0, cx1, cy1 = window._cache.chunk_index_range(0, 0, 0, *window._cache.canvas_dims(0))
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, cx1, cy1))
        _render_levels(window)
        assert window._cache.sprites_enabled
        calls = _spy(window, monkeypatch)
        window.set_clone_source((20, 20))
        _stroke(window, [(60, 60), (61, 60)])
        (changed,) = calls["changed"]
        assert changed, "the stroke end took the wholesale unit invalidation"
        assert calls["wholesale"] == 0, "the cache fell back to a wholesale source rebuild"
        (record,) = window.edit_history.records
        assert isinstance(record, CompositeDiffRecord)
        trees = [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FOREST_OAK_CONST]
        assert len(trees) == 9 and all(_tile(window, int(u.x), int(u.y)).elevation == 5 for u in trees)
        _check_against_fresh(window, "clone")
        # The history's own path for a record carrying elevation and units (viewer_history.py).
        window.undo()
        assert not window.scenario.unit_manager.units[0] and _tile(window, 60, 60).elevation == 0
        _render_levels(window)
        _check_against_fresh(window, "undo")
        window.redo()
        _check_against_fresh(window, "redo")
    finally:
        conftest.close_window(window)
