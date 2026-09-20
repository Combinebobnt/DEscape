"""GridItem and MapView's View > Grid lifecycle, driven through real offscreen
paint dispatch. Structured after tests/test_edge_ticks_viewer.py.

Unlike the ticks, the grid is supposed to change interior pixels, so the
leak detector here compares a hidden grid against a grid removed from the
scene outright, not on against off.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from descape import grid_overlay, settings
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

MAP_W, MAP_H, TILE_PX = 20, 16, 8


def _bare_item(blend: int = grid_overlay.BLEND_DEFAULT, thickness: int = grid_overlay.THICKNESS_DEFAULT):
    conftest.ensure_qapp()
    from PyQt5.QtCore import QRectF

    from descape.viewer_canvas import GridItem

    map_rect = QRectF(0, 0, MAP_W * TILE_PX, MAP_H * TILE_PX)
    return GridItem(MAP_W, MAP_H, map_rect, tile_px=TILE_PX, blend=blend, thickness=thickness)


def _paint(
    item,
    scale: float,
    rotate_deg: float = 0.0,
    scale_y: float | None = None,
    size: int = 400,
    fill: int = 0,
):
    """One direct paint() at a chosen world transform, onto a flat `fill`
    grey. Black by default, since most tests only ask whether anything was
    drawn; a blend that darkens needs a mid-grey to be visible at all."""
    from PyQt5.QtGui import QColor, QImage, QPainter, QTransform

    from testkit.qt_capture import qimage_rgb888_to_array

    image = QImage(size, size, QImage.Format_RGB888)
    image.fill(QColor(fill, fill, fill))
    painter = QPainter(image)
    transform = QTransform()
    transform.translate(20, 20)
    transform.scale(scale, scale if scale_y is None else scale_y)
    if rotate_deg:
        transform.rotate(rotate_deg)
    painter.setWorldTransform(transform)
    item.paint(painter, None)
    left_over = painter.worldTransform()
    painter.end()
    return qimage_rgb888_to_array(image), left_over, transform


# --- the item ----------------------------------------------------------------


def test_paint_leaves_the_painter_transform_untouched() -> None:
    item = _bare_item()
    _pixels, left_over, transform = _paint(item, 2.0)
    assert left_over == transform


@pytest.mark.parametrize("zoom", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("thickness", grid_overlay.THICKNESS_STOPS)
def test_thickness_is_device_pixels_at_any_zoom(zoom: float, thickness: int) -> None:
    """A column crossing only horizontal lines: each line's lit run is the
    pen width, whatever the zoom. Antialiasing is off on a bare QPainter."""
    item = _bare_item(blend=grid_overlay.BLEND_MAX, thickness=thickness)
    pixels, _l, _t = _paint(item, zoom)
    column = 20 + round(TILE_PX * zoom * 1.5)  # mid-tile, clear of every vertical line
    lit = pixels[:, column].max(axis=-1) > 0
    runs = []
    length = 0
    for on in lit:
        if on:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    assert runs, "nothing drawn"
    assert set(runs) == {thickness}


def test_minors_drop_before_the_grid_as_the_view_zooms_out() -> None:
    item = _bare_item()
    seen = []
    for scale in (2.0, 0.5, 0.15, 0.01):
        _paint(item, scale)
        seen.append(item.last_lod["x"])
    assert seen[0] == grid_overlay.GridLod(True, True)
    assert seen[-1] == grid_overlay.GridLod(False, False)
    for earlier, later in pairwise(seen):
        assert later.draw_grid <= earlier.draw_grid
        assert later.draw_minors <= earlier.draw_minors


def test_a_dropped_grid_paints_nothing() -> None:
    item = _bare_item(blend=grid_overlay.BLEND_MAX)
    pixels, _l, _t = _paint(item, 0.01)
    assert not pixels.any()


def test_an_anisotropic_transform_gives_each_axis_its_own_verdict() -> None:
    item = _bare_item()
    _paint(item, 1.0, scale_y=0.1)
    assert item.last_lod["x"] != item.last_lod["y"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"blend": 25}, {"blend": grid_overlay.BLEND_MAX}),
        ({"thickness": grid_overlay.THICKNESS_STOPS[0]}, {"thickness": grid_overlay.THICKNESS_STOPS[-1]}),
    ],
)
def test_appearance_reaches_pixels(first: dict, second: dict) -> None:
    item = _bare_item()
    item.set_appearance(first.get("blend", grid_overlay.BLEND_MAX), first.get("thickness", 1))
    a, _l, _t = _paint(item, 2.0)
    item.set_appearance(second.get("blend", grid_overlay.BLEND_MAX), second.get("thickness", 1))
    b, _l, _t = _paint(item, 2.0)
    assert a.any() and b.any()
    assert not np.array_equal(a, b)


def test_each_side_of_the_blend_moves_the_terrain_its_own_way() -> None:
    """The whole point of the slider: a negative blend darkens what is under
    the line, a positive one lightens it, so both are measured against a
    mid-grey ground rather than black."""
    mid = 128
    item = _bare_item(blend=grid_overlay.BLEND_MIN)
    dark, _l, _t = _paint(item, 2.0, fill=mid)
    item.set_appearance(grid_overlay.BLEND_MAX, grid_overlay.THICKNESS_DEFAULT)
    light, _l, _t = _paint(item, 2.0, fill=mid)
    assert dark.min() < mid < light.max()
    assert dark.max() == light.min() == mid  # untouched ground, both ways


def test_a_centred_blend_paints_nothing_and_skips_the_pass() -> None:
    item = _bare_item(blend=0)
    pixels, _l, _t = _paint(item, 2.0, fill=128)
    assert (pixels == 128).all()
    assert item.paint_count == 1  # paint() ran, and returned before drawing
    assert item.last_lod == {}


def test_the_pad_covers_the_thickest_pen_at_the_minimum_scale() -> None:
    item = _bare_item()
    scale = 0.01
    item.set_min_view_scale(scale)
    rect = item.boundingRect()
    overhang = min(rect.right() - MAP_W * TILE_PX, rect.bottom() - MAP_H * TILE_PX, -rect.left(), -rect.top())
    assert overhang * scale * math.sqrt(0.5) >= max(grid_overlay.THICKNESS_STOPS) / 2.0


def test_a_non_positive_scale_keeps_the_existing_pad() -> None:
    item = _bare_item()
    item.set_min_view_scale(0.01)
    rect = item.boundingRect()
    item.set_min_view_scale(0.0)
    item.set_min_view_scale(None)
    assert item.boundingRect() == rect


# --- MapView lifecycle -------------------------------------------------------


def _window():
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    window.map_view.set_grid_overlay(True)
    return window


def test_the_item_is_built_in_every_terrain_style_and_keeps_its_state() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.map_view.set_grid_appearance(-60, 3)
        expected_alpha = grid_overlay.grid_colors(-60)[0][3]
        for style in ("Flat", "Sloped", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            item = window.map_view._grid_item
            assert item is not None, style
            assert item.isVisible(), style
            minor_pen, _major = item.pens()
            assert minor_pen.color().alpha() == expected_alpha, style
            assert minor_pen.width() == 3, style
            assert minor_pen.isCosmetic(), style
    finally:
        conftest.close_window(window)


def test_the_brush_highlight_stacks_above_the_grid() -> None:
    """No Z value on the grid, so the lazily created gold highlight, added
    later at the same Z, paints over it by insertion order."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        window.mode_terrain_action.trigger()
        view._update_highlight(40, 40)
        highlight = view._highlight_outline_item
        assert highlight is not None
        assert view._grid_item.zValue() == 0.0
        stacked = view.scene().items(Qt.DescendingOrder)
        assert stacked.index(highlight) < stacked.index(view._grid_item)
    finally:
        conftest.close_window(window)


def test_a_hidden_grid_is_byte_identical_to_no_grid_and_never_paints() -> None:
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry as ig
    from testkit.qt_capture import scene_rect_to_array

    window = _window()
    try:
        view = window.map_view
        proj = view._iso_proj
        sx, sy = ig.tile_screen_origin(60, 60, 0, proj)
        rect = QRectF(sx, sy, 4 * proj.half_w, 4 * proj.half_h)
        item = view._grid_item

        shown = scene_rect_to_array(view.scene(), rect)
        view.set_grid_overlay(False)
        before = item.paint_count
        hidden = scene_rect_to_array(view.scene(), rect)
        assert item.paint_count == before
        view.scene().removeItem(item)
        removed = scene_rect_to_array(view.scene(), rect)
        view.scene().addItem(item)

        assert np.array_equal(hidden, removed)
        assert not np.array_equal(shown, hidden), "the shown grid drew nothing, so the check is vacuous"
    finally:
        conftest.close_window(window)


def test_a_resize_while_zoomed_out_keeps_the_pad_valid() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        view = window.map_view
        item = view._grid_item
        view.resetTransform()
        view.scale(view._min_linear_scale, view._min_linear_scale)
        view._capture_zoom_baseline()
        current = abs(view.transform().determinant()) ** 0.5
        window.resize(window.width() * 2, window.height() * 2)
        QApplication.processEvents()
        view._capture_zoom_baseline()
        rect = item.boundingRect()
        overhang = min(-rect.left(), -rect.top())
        assert overhang * current >= max(grid_overlay.THICKNESS_STOPS) / 2.0
    finally:
        conftest.close_window(window)


def test_closing_a_map_then_resizing_and_restyling_does_not_touch_a_deleted_item() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.map_view.clear_image()
        assert window.map_view._grid_item is None
        window.resize(window.width() + 120, window.height() + 90)
        QApplication.processEvents()
        window.map_view._capture_zoom_baseline()
        window.map_view.set_grid_overlay(False)
        window.map_view.set_grid_appearance(-30, 2)
    finally:
        conftest.close_window(window)


# --- View > Grid ---------------------------------------------------------------


def _plain_window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    return ViewerWindow()


def test_building_the_window_writes_no_config_file(tmp_path) -> None:
    window = _plain_window()
    try:
        assert not (tmp_path / "config.yaml").exists()
        assert not window.grid_action.isChecked()
    finally:
        conftest.close_window(window)


def test_the_action_opens_at_the_persisted_state(monkeypatch) -> None:
    monkeypatch.setattr(settings, "_grid_overlay", True)
    window = _plain_window()
    try:
        assert window.grid_action.isChecked()
    finally:
        conftest.close_window(window)


def test_the_toggle_persists_and_reaches_the_view() -> None:
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.grid_action.setChecked(True)
        assert settings.get_grid_overlay() is True
        assert window.map_view._grid_item.isVisible()
        window.grid_action.setChecked(False)
        assert settings.get_grid_overlay() is False
        assert not window.map_view._grid_item.isVisible()
    finally:
        conftest.close_window(window)


def test_the_action_is_never_greyed_out() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        for mode_action in (window.mode_view_action, window.mode_terrain_action, window.mode_units_action):
            mode_action.trigger()
            QApplication.processEvents()
            assert window.grid_action.isEnabled()
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            assert window.grid_action.isEnabled(), style
        window.map_view.clear_image()
        assert window.grid_action.isEnabled()
    finally:
        conftest.close_window(window)


def test_the_keybind_row_ships_unbound_and_stays_with_the_view_rows() -> None:
    ids = [action_id for action_id, _label, _key in settings.REBINDABLE_ACTIONS]
    row = next(r for r in settings.REBINDABLE_ACTIONS if r[0] == "view_grid_overlay")
    assert row[2] == ""
    view_positions = [i for i, action_id in enumerate(ids) if action_id.startswith("view_")]
    assert view_positions == list(range(view_positions[0], view_positions[-1] + 1))


# --- Settings > Appearance sliders -------------------------------------------


def _dialog_and_window(path=None):
    from descape.viewer import SettingsDialog

    window = conftest.stepped_window(path) if path else _plain_window()
    return SettingsDialog(window), window


def test_the_sliders_open_at_the_persisted_values(monkeypatch) -> None:
    """Blend is a value-space slider (every position reachable), thickness a
    stop-indexed one."""
    monkeypatch.setattr(settings, "_grid_blend", -60)
    monkeypatch.setattr(settings, "_grid_thickness", 3)
    dialog, window = _dialog_and_window()
    try:
        blend, thick = dialog.grid_blend_slider, dialog.grid_thickness_slider
        assert (blend.minimum(), blend.maximum()) == (grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX)
        assert (thick.minimum(), thick.maximum()) == (1, len(grid_overlay.THICKNESS_STOPS))
        assert blend.value() == -60
        assert blend.singleStep() == 1  # smooth, not a stop ladder
        assert dialog.grid_blend_value_label.text() == "dark 60"
        assert grid_overlay.thickness_for_index(thick.value()) == 3
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.parametrize(("blend", "label"), [(-100, "dark 100"), (0, "off"), (72, "light 72")])
def test_the_blend_label_names_the_side(blend: int, label: str) -> None:
    from descape.viewer import SettingsDialog

    assert SettingsDialog._blend_label(blend) == label


def test_dragging_a_slider_persists_and_reaches_the_live_item_synchronously() -> None:
    """No apply timer, deliberately: the handler must have run by the time
    setValue() returns."""
    dialog, window = _dialog_and_window(BLANK_TEMPLATE_PATH)
    try:
        dialog.grid_blend_slider.setValue(-30)
        dialog.grid_thickness_slider.setValue(grid_overlay.thickness_index(4))
        assert settings.get_grid_blend() == -30
        assert settings.get_grid_thickness() == 4
        minor_pen, major_pen = window.map_view._grid_item.pens()
        minor, major = grid_overlay.grid_colors(-30)
        assert minor_pen.color().red() == major_pen.color().red() == 0  # darkening
        assert (minor_pen.color().alpha(), major_pen.color().alpha()) == (minor[3], major[3])
        assert minor_pen.width() == major_pen.width() == 4
    finally:
        dialog.close()
        conftest.close_window(window)


# --- Follow Terrain Elevation --------------------------------------------------


def _ramped_window(style: str = "Stepped"):
    """A window whose map climbs one level across its width, so draping has
    something to follow. Same fixture shape as tests/test_sloped_edit.py's."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    mm = window.scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("elevation")
    window.terrain_style_combo.setCurrentText(style)
    # The tiles were mutated behind the render's back, and a setCurrentText to
    # the style already showing is a no-op, so re-render explicitly.
    window.refresh_map()
    QApplication.processEvents()
    window.map_view.set_grid_overlay(True)
    window.map_view.set_grid_follow_elevation(True)
    return window


# The tile the geometry checks below both capture around and edit: a windowed
# grid only rebuilds what the visible rect covers, so an edit elsewhere on the
# map legitimately leaves its segments untouched.
_PROBE = (60, 60)


def _interior_capture(view, tile=_PROBE):
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry as ig
    from testkit.qt_capture import scene_rect_to_array

    proj = view._iso_proj
    sx, sy = ig.tile_screen_origin(tile[0], tile[1], 0, proj)
    return scene_rect_to_array(view.scene(), QRectF(sx - 4 * proj.half_w, sy - 8 * proj.half_h,
                                                    12 * proj.half_w, 16 * proj.half_h))


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_draping_reaches_pixels_on_a_raised_map(style: str) -> None:
    window = _ramped_window(style)
    try:
        view = window.map_view
        # A probe astride the ramp, where the two halves differ in height.
        mm = window.scenario.map_manager
        probe = (mm.map_width // 2, mm.map_height // 2)
        draped = _interior_capture(view, probe)
        view.set_grid_follow_elevation(False)
        flat = _interior_capture(view, probe)
        assert draped.any()
        assert not np.array_equal(draped, flat)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_on_a_flat_map_the_mode_switch_changes_nothing_on_screen(style: str) -> None:
    """The sharper half: draping moves lines, it never changes which are
    drawn."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.terrain_style_combo.setCurrentText(style)
        QApplication.processEvents()
        view = window.map_view
        view.set_grid_overlay(True)
        view.set_grid_follow_elevation(True)
        draped = _interior_capture(view)
        view.set_grid_follow_elevation(False)
        flat = _interior_capture(view)
        assert draped.any()
        assert np.array_equal(draped, flat)
    finally:
        conftest.close_window(window)


def test_a_hidden_grid_is_byte_identical_to_no_grid_while_draping() -> None:
    from testkit.qt_capture import scene_rect_to_array

    window = _ramped_window("Sloped")
    try:
        view = window.map_view
        item = view._grid_item
        view.set_grid_overlay(False)
        hidden = _interior_capture(view)
        view.scene().removeItem(item)
        removed = _interior_capture(view)
        view.scene().addItem(item)
        del scene_rect_to_array
        assert np.array_equal(hidden, removed)
    finally:
        conftest.close_window(window)


def _raise_tile(window, x: int, y: int) -> None:
    """One real elevation stroke through the same path a drag takes."""
    from PyQt5.QtCore import Qt

    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, Qt.NoModifier)
    window.on_edit_stroke_end()


def _live_segments(view) -> set:
    """Every segment the grid item is currently drawing from."""
    _interior_capture(view)
    return {tuple(row) for segments in view._grid_item._draped for row in segments.tolist()}


def _expected_segments(view) -> set:
    """The same, recomputed from the map's live height field. A capture-based
    comparison cannot serve here: an elevation edit moves terrain pixels too,
    so a stale grid still yields a changed image."""
    item = view._grid_item
    tiles = np.array(
        [(x, y) for y in range(view._map_height) for x in range(view._map_width)], dtype=np.int64
    )
    source = (
        {"elevations": view._iso_elevations}
        if view._terrain_style == "stepped"
        else {"corner_rise": view._sloped_cache().corner_rise}
    )
    minor, major = grid_overlay.draped_lines(tiles, view._terrain_style, view._iso_proj, **source)
    del item
    return {tuple(row) for segments in (minor, major) for row in segments.tolist()}


def _assert_grid_matches_the_terrain(view) -> None:
    """Merged runs make a windowed segment a sub-run of a full-map one, so
    compare the per-tile steps rather than whole segments."""
    half_w = view._iso_proj.half_w
    live = _unit_steps(_live_segments(view), half_w)
    assert live, "the grid drew nothing, so the check is vacuous"
    assert live <= _unit_steps(_expected_segments(view), half_w)


def _unit_steps(segments: set, half_w: int) -> set:
    steps = set()
    for x0, y0, x1, y1 in segments:
        count = abs(x1 - x0) // half_w
        dx, dy = (x1 - x0) // count, (y1 - y0) // count
        for k in range(count):
            a = (x0 + k * dx, y0 + k * dy)
            steps.add((*min(a, (a[0] + dx, a[1] + dy)), *max(a, (a[0] + dx, a[1] + dy))))
    return steps


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevation_edit_moves_the_grid_within_the_same_frame(style: str) -> None:
    """In Sloped this is the regression for SlopedChunkCache.patch() rebinding
    corner_rise: a reference held from set_source(), or a push that never
    happens, leaves the grid drawing the pre-edit height field forever."""
    window = _ramped_window(style)
    try:
        view = window.map_view
        _assert_grid_matches_the_terrain(view)
        before = _live_segments(view)
        _raise_tile(window, *_PROBE)
        assert _live_segments(view) != before, "the edit did not reach the grid"
        _assert_grid_matches_the_terrain(view)
    finally:
        conftest.close_window(window)


def test_an_edit_made_while_the_grid_is_hidden_still_shows_up_when_it_returns() -> None:
    window = _ramped_window("Sloped")
    try:
        view = window.map_view
        _assert_grid_matches_the_terrain(view)
        view.set_grid_overlay(False)
        _raise_tile(window, *_PROBE)
        view.set_grid_overlay(True)
        _assert_grid_matches_the_terrain(view)
    finally:
        conftest.close_window(window)


def test_a_terrain_only_stroke_rebuilds_no_grid_geometry() -> None:
    from PyQt5.QtCore import Qt

    window = _ramped_window("Sloped")
    try:
        view = window.map_view
        _interior_capture(view)  # force one rebuild
        window._on_tool_selected("draw")
        before = view._grid_item.rebuild_count
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(30, 30, Qt.NoModifier)
        window.on_edit_stroke_end()
        _interior_capture(view)
        assert view._grid_item.rebuild_count == before
    finally:
        conftest.close_window(window)


def test_an_elevation_stroke_rebuilds_no_more_often_than_it_repaints() -> None:
    """Qt coalesces the update() calls, so a multi-touch stroke costs a
    handful of rebuilds, not one per touched tile."""
    from PyQt5.QtCore import Qt

    window = _ramped_window("Stepped")
    try:
        view = window.map_view
        _interior_capture(view)
        before = view._grid_item.rebuild_count
        window.on_edit_stroke_start()
        for step in range(8):
            window.on_edit_stroke_tile(20 + step, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        _interior_capture(view)
        rebuilds = view._grid_item.rebuild_count - before
        assert 1 <= rebuilds <= 2
    finally:
        conftest.close_window(window)


def test_the_bounding_rect_covers_every_draped_corner() -> None:
    window = _ramped_window("Sloped")
    try:
        view = window.map_view
        item = view._grid_item
        item.set_min_view_scale(1.0)
        rect = item.boundingRect()
        _interior_capture(view)
        for segments in item._draped:
            for x0, y0, x1, y1 in segments.tolist():
                assert rect.contains(x0, y0) and rect.contains(x1, y1)
    finally:
        conftest.close_window(window)


def test_both_toggles_survive_a_terrain_style_round_trip() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _ramped_window("Stepped")
    try:
        window.terrain_style_combo.setCurrentText("Flat")
        QApplication.processEvents()
        window.terrain_style_combo.setCurrentText("Sloped")
        QApplication.processEvents()
        item = window.map_view._grid_item
        assert item.isVisible()
        assert item._follow is True
        assert item._corner_rise is not None
    finally:
        conftest.close_window(window)


def test_the_follow_action_persists_reaches_the_view_and_is_never_greyed() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        assert window.grid_follow_action.isChecked() is True  # on by default
        window.grid_follow_action.setChecked(False)
        assert settings.get_grid_follow_elevation() is False
        assert window.map_view._grid_item._follow is False
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            assert window.grid_follow_action.isEnabled(), style
    finally:
        conftest.close_window(window)


def test_flat_ignores_the_follow_toggle() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        QApplication.processEvents()
        view = window.map_view
        view.set_grid_overlay(True)
        view.set_grid_follow_elevation(True)
        assert view._grid_item._drapes() is False
    finally:
        conftest.close_window(window)
