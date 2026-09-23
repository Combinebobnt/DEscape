"""GridItem and MapView's View > Grid lifecycle, driven through real offscreen
paint dispatch. Structured after tests/test_edge_ticks_viewer.py.

The grid has two halves. With Follow Terrain Elevation on (and always in
Flat) it is baked into the chunk cache, checked here through the live cache's
own spec and pixels; GridItem only draws the follow-off lattice and the
Settings > Appearance slider preview. Unlike the ticks, the overlay is
supposed to change interior pixels, so its leak detector compares a hidden
item against one removed from the scene outright, not on against off.
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


def _overlay_window():
    """Follow Terrain Elevation off, the one iso mode GridItem still draws."""
    window = _window()
    window.map_view.set_grid_follow_elevation(False)
    return window


def _probe_rect(view, tile=(60, 60), tiles: int = 4):
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry as ig

    proj = view._iso_proj
    sx, sy = ig.tile_screen_origin(tile[0], tile[1], 0, proj)
    return QRectF(sx, sy, tiles * proj.half_w, tiles * proj.half_h)


def _cache_pixels(window, tile=(60, 60)):
    """The live cache's own pixels around `tile`, no scene items involved."""
    rect = _probe_rect(window.map_view, tile, 8)
    x0, y0 = int(rect.left()), int(rect.top())
    return window._cache.render_rect(x0, y0, x0 + int(rect.width()), y0 + int(rect.height())).copy()


def test_the_bake_follows_every_terrain_style_and_keeps_its_state() -> None:
    """A style switch builds a fresh cache in set_source(), which must get the
    spec re-applied or the grid silently drops until the next toggle."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.map_view.set_grid_appearance(-60, 3)
        expected = grid_overlay.grid_bake(True, -60, 3)
        for style in ("Flat", "Sloped", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            item = window.map_view._grid_item
            assert item is not None, style
            assert not item.isVisible(), f"{style}: the bake is live, so the overlay must be hidden"
            assert window._cache.grid == expected, style
            minor_pen, _major = item.pens()
            assert minor_pen.color().alpha() == expected.minor[3], style
            assert minor_pen.width() == 3, style
            assert minor_pen.isCosmetic(), style
    finally:
        conftest.close_window(window)


def test_the_brush_highlight_stacks_above_the_grid() -> None:
    """No Z value on the grid, so the lazily created gold highlight, added
    later at the same Z, paints over it by insertion order."""
    from PyQt5.QtCore import Qt

    window = _overlay_window()
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


def test_a_hidden_overlay_is_byte_identical_to_no_overlay_and_never_paints() -> None:
    from testkit.qt_capture import scene_rect_to_array

    window = _overlay_window()
    try:
        view = window.map_view
        rect = _probe_rect(view)
        item = view._grid_item
        assert window._cache.grid == grid_overlay.DEFAULT_GRID, "follow off must not bake"

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


def test_turning_the_baked_grid_off_restores_the_exact_pixels() -> None:
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        before = _cache_pixels(window)
        window.grid_action.setChecked(True)
        on = _cache_pixels(window)
        window.grid_action.setChecked(False)
        assert not np.array_equal(before, on), "the baked grid drew nothing, so the check is vacuous"
        assert np.array_equal(before, _cache_pixels(window))
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
        window.map_view.begin_grid_preview()
        window.map_view.end_grid_preview()
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


def test_the_toggle_persists_and_reaches_the_cache() -> None:
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.grid_action.setChecked(True)
        assert settings.get_grid_overlay() is True
        assert window._cache.grid.paints
        assert not window.map_view._grid_item.isVisible()
        window.grid_action.setChecked(False)
        assert settings.get_grid_overlay() is False
        assert window._cache.grid == grid_overlay.DEFAULT_GRID
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


def _count_spec_changes(cache) -> list:
    changes = []
    original = cache.set_grid

    def counting(spec):
        if spec != cache.grid:
            changes.append(spec)
        original(spec)

    cache.set_grid = counting
    return changes


def test_a_drag_evicts_twice_per_gesture_not_per_tick() -> None:
    """The first tick pulls the bake and shows GridItem's preview; every tick
    after is pens only; the release re-bakes at the final appearance."""
    dialog, window = _dialog_and_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        view.set_grid_overlay(True)
        changes = _count_spec_changes(window._cache)
        slider = dialog.grid_blend_slider
        slider.setSliderDown(True)
        for value in (-10, -20, -30, -40):
            slider.setValue(value)
        assert changes == [grid_overlay.DEFAULT_GRID]
        assert view.grid_previewing()
        assert view._grid_item.isVisible()
        minor_pen, _ = view._grid_item.pens()
        assert minor_pen.color().alpha() == grid_overlay.grid_colors(-40)[0][3]
        slider.setSliderDown(False)
        applied = grid_overlay.grid_bake(True, -40, settings.get_grid_thickness())
        assert changes == [grid_overlay.DEFAULT_GRID, applied]
        assert window._cache.grid == applied
        assert not view.grid_previewing()
        assert not view._grid_item.isVisible()
        assert settings.get_grid_blend() == -40
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_keyboard_change_is_applied_when_the_dialog_closes() -> None:
    """No press/release pair, so a timer applies it; closing the dialog
    first must not leave the grid stuck in its unbaked preview."""
    dialog, window = _dialog_and_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        view.set_grid_overlay(True)
        dialog.grid_thickness_slider.setValue(grid_overlay.thickness_index(4))
        assert view.grid_previewing()
        assert dialog._grid_apply_timer.isActive()
        dialog.reject()  # what Close, Escape and the title-bar X all reach
        assert not view.grid_previewing()
        assert window._cache.grid.thickness == 4
    finally:
        conftest.close_window(window)


def test_the_overlay_mode_never_evicts_from_the_sliders() -> None:
    dialog, window = _dialog_and_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        view.set_grid_overlay(True)
        view.set_grid_follow_elevation(False)
        changes = _count_spec_changes(window._cache)
        dialog.grid_blend_slider.setValue(-30)
        dialog.grid_thickness_slider.setValue(grid_overlay.thickness_index(4))
        assert changes == []
        assert not view.grid_previewing()
        minor_pen, major_pen = view._grid_item.pens()
        minor, major = grid_overlay.grid_colors(-30)
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
    window._on_grid_overlay_toggled(True)
    window._on_grid_follow_toggled(True)
    return window


_PROBE = (60, 60)


def _raise_tile(window, x: int, y: int) -> None:
    """One real elevation stroke through the same path a drag takes."""
    from PyQt5.QtCore import Qt

    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, Qt.NoModifier)
    window.on_edit_stroke_end()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_follow_off_swaps_the_bake_for_the_overlay(style: str) -> None:
    window = _ramped_window(style)
    try:
        view = window.map_view
        assert window._cache.grid.paints
        assert not view._grid_item.isVisible()
        window.grid_follow_action.setChecked(False)
        assert window._cache.grid == grid_overlay.DEFAULT_GRID
        assert view._grid_item.isVisible()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevation_edit_redrapes_the_grid_with_no_refresh(style: str) -> None:
    """The edit's own cache.patch() recomposites through self.grid, so the
    patched pixels must equal a from-scratch render of the edited map with
    the same spec, which refresh_map() rebuilds through set_source()."""
    window = _ramped_window(style)
    try:
        before = _cache_pixels(window, _PROBE)
        _raise_tile(window, *_PROBE)
        patched = _cache_pixels(window, _PROBE)
        assert not np.array_equal(before, patched), "the edit changed nothing, so the check is vacuous"
        window.refresh_map()
        assert window._cache.grid.paints
        assert np.array_equal(patched, _cache_pixels(window, _PROBE))
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
        assert window.map_view.grid_bake_live()
        assert window._cache.grid.paints
    finally:
        conftest.close_window(window)


def test_the_follow_action_persists_reaches_the_view_and_is_never_greyed() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        assert window.grid_follow_action.isChecked() is True  # on by default
        window.grid_follow_action.setChecked(False)
        assert settings.get_grid_follow_elevation() is False
        assert window.map_view._grid_follow_elevation is False
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            assert window.grid_follow_action.isEnabled(), style
    finally:
        conftest.close_window(window)


def test_flat_ignores_the_follow_toggle_and_always_bakes() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        QApplication.processEvents()
        view = window.map_view
        view.set_grid_overlay(True)
        view.set_grid_follow_elevation(False)
        assert view.grid_bake_live()
        assert window._cache.grid.paints
        assert not view._grid_item.isVisible()
    finally:
        conftest.close_window(window)
