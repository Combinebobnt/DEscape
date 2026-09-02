"""EdgeTickItem and MapView's tick lifecycle, driven through a real
offscreen QGraphicsScene paint dispatch rather than an off-engine call.

Structured after tests/test_sprite_toggle_viewer.py. Two levels are used
deliberately:

- A bare scene holding only the item, for anything measuring what paint()
  puts on screen. Nothing else can contribute a pixel, so a colour match IS
  the tick, and the item's own transform can be set exactly.
- A real ViewerWindow, for the lifecycle questions a bare item cannot ask:
  the bounding-rect pad after a resize, and the deleted-C++-object path
  through clear_image.

The load-bearing ones are
test_a_resize_while_zoomed_out_keeps_the_bounding_rect_valid (the case
min(_min_linear_scale, current) exists for; using the floor alone passes
every static test here) and
test_an_interior_capture_is_byte_identical_with_ticks_on_and_off (the leak
detector: if that moves, the overlay reached the canvas).

Every ViewerWindow() constructed here calls edit_history.mark_saved() before
close(); see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import conftest
from descape import edge_ticks
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Interior probe, matching tests/test_seam_viewer.py's own _ANCHOR on this
# same 120x120 template and for the same reason: the map-extent outline cuts
# through any probe placed near a corner.
_ANCHOR = (60, 60)

MAP_W, MAP_H, TILE_PX = 20, 16, 8


def _bare_scene(interval: int = edge_ticks.TICK_INTERVAL_DEFAULT, scale: float = 1.0):
    """A scene holding nothing but one flat-mode EdgeTickItem."""
    conftest.ensure_qapp()
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QGraphicsScene

    from descape.viewer_canvas import EdgeTickItem

    map_rect = QRectF(0, 0, MAP_W * TILE_PX, MAP_H * TILE_PX)
    item = EdgeTickItem(MAP_W, MAP_H, interval, map_rect, tile_px=TILE_PX)
    item.set_min_view_scale(scale)
    scene = QGraphicsScene()
    scene.addItem(item)
    return scene, item


def _paint_at(item, scale_x: float, scale_y: float = None, rotate_deg: float = 0.0):
    """Drives exactly ONE paint() at a chosen world transform and hands back
    item.stats. Calls paint() directly: TickPaintStats records the last paint
    only, and Qt is free to split a scene render into several."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter, QTransform

    image = QImage(400, 400, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    transform = QTransform()
    transform.scale(scale_x, scale_x if scale_y is None else scale_y)
    if rotate_deg:
        transform.rotate(rotate_deg)
    painter.setWorldTransform(transform)
    item.paint(painter, None)
    left_over = painter.worldTransform()
    painter.end()
    return item.stats, left_over, QRectF()


# --- device-space geometry -------------------------------------------------


def _column_run(pixels, column: int, color: tuple[int, int, int]) -> int:
    """How many rows of `column` are exactly `color`."""
    match = np.all(pixels[:, column] == np.array(color, dtype=np.uint8), axis=-1)
    return int(match.sum())


def test_tick_length_is_constant_in_device_pixels() -> None:
    """Flat and non-isometric, where y0's outward direction is exactly
    (0, -1): an axis-aligned, antialiasing-free vertical run whose length is
    trivially countable. A major is twice a minor."""
    from PyQt5.QtCore import QRectF

    from testkit.qt_capture import scene_rect_to_array
    from descape.viewer_canvas import EdgeTickItem

    scene, _item = _bare_scene()
    # Above the y0 edge, which sits at scene y == 0, so the ticks run up into
    # negative y.
    rect = QRectF(-4, -40, 120, 44)
    pixels = scene_rect_to_array(scene, rect)

    major = EdgeTickItem.MAJOR_PEN.color().getRgb()[:3]
    minor = EdgeTickItem.MINOR_PEN.color().getRgb()[:3]
    # Scene x == 0 is a major (index 0); scene x ==
    # TICK_INTERVAL_DEFAULT*TILE_PX is the first minor. Capture x is scene x
    # minus the rect's own left edge.
    major_run = _column_run(pixels, int(0 - rect.left()), major)
    minor_run = _column_run(
        pixels, int(edge_ticks.TICK_INTERVAL_DEFAULT * TILE_PX - rect.left()), minor
    )

    assert minor_run == pytest.approx(edge_ticks.MINOR_TICK_PX, abs=1)
    assert major_run == pytest.approx(edge_ticks.MAJOR_TICK_PX, abs=1)
    assert major_run > minor_run


def _render_zoomed(scene, rect, zoom: float):
    """The same scene RECT rasterized `zoom` times larger, which is what a
    real zoom-in is. testkit.scene_rect_to_array is deliberately 1:1 only
    (see its docstring), so it cannot express this."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    from testkit.qt_capture import qimage_rgb888_to_array

    width, height = int(rect.width() * zoom), int(rect.height() * zoom)
    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    scene.render(painter, QRectF(0, 0, width, height), rect)
    painter.end()
    return qimage_rgb888_to_array(image)


@pytest.mark.parametrize("zoom", [1, 2, 4])
def test_the_tick_length_does_not_follow_the_zoom(zoom: int) -> None:
    """The whole reason marks are drawn under a reset transform. A
    scene-space length would grow with the view; a device-space one does
    not, so the SAME run length must come back at every zoom."""
    from PyQt5.QtCore import QRectF

    from descape.viewer_canvas import EdgeTickItem

    scene, _item = _bare_scene()
    rect = QRectF(-4, -40, 120, 44)
    pixels = _render_zoomed(scene, rect, zoom)
    column = int(round((edge_ticks.TICK_INTERVAL_DEFAULT * TILE_PX - rect.left()) * zoom))
    run = _column_run(pixels, column, EdgeTickItem.MINOR_PEN.color().getRgb()[:3])
    assert run == pytest.approx(edge_ticks.MINOR_TICK_PX, abs=1)


def test_paint_leaves_the_painter_transform_untouched() -> None:
    """paint() resets the transform to draw in device space, so it must put
    it back. A leak would corrupt whatever paints after this item, and the
    symptom looks exactly like the stale-boundingRect one."""
    from PyQt5.QtGui import QTransform

    _scene, item = _bare_scene()
    _stats, left_over, _ = _paint_at(item, 1.5)
    expected = QTransform()
    expected.scale(1.5, 1.5)
    assert left_over == expected


# --- the LOD ladder --------------------------------------------------------


def test_the_ladder_drops_labels_before_minors_as_the_view_zooms_out() -> None:
    """Asserted on the item's own recorded verdict, not by counting pixels:
    the thresholds are numbers, and measuring them as pixels would only add
    a rasterizer to the failure modes."""
    _scene, item = _bare_scene()
    seen = []
    for scale in (2.0, 0.2, 0.12, 0.02):
        stats, _t, _r = _paint_at(item, scale)
        lod = stats.lod["y0"]
        seen.append((lod.draw_edge, lod.draw_minors, lod.draw_labels))
    assert seen[0] == (True, True, True)
    # Monotone: nothing ever comes back as the view zooms further out.
    for earlier, later in zip(seen, seen[1:]):
        assert all(b <= a for a, b in zip(earlier, later))
    assert seen[-1][0] is False


def test_a_dropped_edge_draws_nothing_at_all() -> None:
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 0.005)
    assert all(not lod.draw_edge for lod in stats.lod.values())
    assert stats.ticks_drawn == 0
    assert stats.labels_drawn == 0


def test_only_majors_survive_once_minors_are_dropped() -> None:
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 0.1)
    assert stats.lod["y0"].draw_minors is False
    assert stats.lod["y0"].draw_edge is True
    expected = sum(sum(run.majors) for run in item._runs)
    assert stats.ticks_drawn == expected


def test_every_tick_is_drawn_when_there_is_room() -> None:
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 4.0)
    assert stats.ticks_drawn == sum(len(run.anchors) for run in item._runs)
    assert stats.labels_drawn == sum(sum(run.majors) for run in item._runs)


def test_an_anisotropic_transform_gives_each_edge_its_own_verdict() -> None:
    """Flat + isometric squashes one axis, so the two edge pairs genuinely
    differ in on-screen spacing. A single global LOD would be wrong for one
    of them."""
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 1.0, scale_y=0.02)
    assert stats.spacing_px["y0"] != pytest.approx(stats.spacing_px["x0"])


# --- the interval ----------------------------------------------------------


def test_changing_the_interval_rebuilds_the_runs() -> None:
    _scene, item = _bare_scene(interval=4)
    before = tuple(len(run.anchors) for run in item._runs)
    item.set_interval(5)
    after = tuple(len(run.anchors) for run in item._runs)
    assert before != after
    assert item._runs[0].tiles == tuple(range(0, MAP_W + 1, 5))


def test_setting_the_same_interval_rebuilds_nothing() -> None:
    _scene, item = _bare_scene(interval=4)
    runs = item._runs
    item.set_interval(4)
    assert item._runs is runs


# --- the bounding-rect pad -------------------------------------------------


def test_the_pad_contains_the_device_reach_at_the_minimum_scale() -> None:
    _scene, item = _bare_scene(scale=0.01)
    rect = item.boundingRect()
    overhang = min(
        rect.right() - MAP_W * TILE_PX,
        rect.bottom() - MAP_H * TILE_PX,
        -rect.left(),
        -rect.top(),
    )
    assert overhang * 0.01 >= math.sqrt(2) * edge_ticks.DEVICE_REACH_PX


def test_the_bounding_rect_covers_every_anchor_even_below_the_canvas() -> None:
    """A map whose minimum elevation is above 0 puts the elevation-0 anchors
    slightly past canvas_h, so the base rect is the union of the map rect and
    the anchors, not the map rect alone."""
    conftest.ensure_qapp()
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry
    from descape.viewer_canvas import EdgeTickItem

    proj = iso_geometry.canvas_size_and_origin(24, 24, 32, min_elev=3, max_elev=6)
    map_rect = QRectF(0, 0, proj.canvas_w, proj.canvas_h)
    item = EdgeTickItem(24, 24, 4, map_rect, proj=proj)
    item.set_min_view_scale(1.0)
    for run in item._runs:
        for anchor in run.anchors:
            assert item.boundingRect().contains(*anchor), f"{run.edge} anchor {anchor} outside"


def test_a_non_positive_scale_keeps_the_existing_pad() -> None:
    _scene, item = _bare_scene(scale=0.01)
    rect = item.boundingRect()
    item.set_min_view_scale(0.0)
    item.set_min_view_scale(None)
    assert item.boundingRect() == rect


# --- MapView lifecycle -----------------------------------------------------


def _window():
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    window.map_view.set_edge_ticks(True)
    return window


def test_a_resize_while_zoomed_out_keeps_the_bounding_rect_valid() -> None:
    """The case min(_min_linear_scale, current) exists for. resizeEvent
    re-runs _capture_zoom_baseline, and a larger viewport RAISES
    _min_linear_scale without rescaling the transform, so the current scale
    ends up below the new floor. Padding from the floor alone passes every
    static test in this file and under-pads exactly here."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        view = window.map_view
        item = view._edge_tick_item
        # Zoom to the floor, then grow the viewport without rescaling.
        view.resetTransform()
        view.scale(view._min_linear_scale, view._min_linear_scale)
        view._capture_zoom_baseline()
        current = abs(view.transform().determinant()) ** 0.5
        window.resize(window.width() * 2, window.height() * 2)
        QApplication.processEvents()
        view._capture_zoom_baseline()

        assert current < view._min_linear_scale, "the resize did not raise the floor"
        rect = item.boundingRect()
        overhang = min(-rect.left(), -rect.top())
        assert overhang * current >= math.sqrt(2) * edge_ticks.DEVICE_REACH_PX
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_a_map_then_resizing_does_not_touch_a_deleted_item() -> None:
    """scene().clear() destroys the C++ object, and resizeEvent still reaches
    _capture_zoom_baseline with no map loaded. Without the null-out this
    raises RuntimeError on a wrapped-C-object-deleted call."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.map_view.clear_image()
        assert window.map_view._edge_tick_item is None
        window.resize(window.width() + 120, window.height() + 90)
        QApplication.processEvents()
        window.map_view._capture_zoom_baseline()  # must not raise
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_item_is_built_in_every_terrain_style() -> None:
    """Not gated by mode or style: the overlay needs only map dimensions
    plus, for the iso styles, the projection. Pinned rather than left as an
    assumption, since nothing greys it out to explain otherwise."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            item = window.map_view._edge_tick_item
            assert item is not None, style
            assert item.isVisible(), style
            assert sum(len(run.anchors) for run in item._runs) > 0, style
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_toggle_is_off_by_default_and_hides_the_item() -> None:
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        item = window.map_view._edge_tick_item
        assert item is not None
        assert not item.isVisible()
        window.map_view.set_edge_ticks(True)
        assert item.isVisible()
        window.map_view.set_edge_ticks(False)
        assert not item.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_interval_reaches_the_live_item() -> None:
    window = _window()
    try:
        other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
        window.map_view.set_edge_tick_interval(other)
        assert window.map_view._edge_tick_item._interval == other
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_tick_state_survives_a_terrain_style_round_trip() -> None:
    """State lives on MapView, which outlives every set_source(), so unlike
    Show sprites it needs no re-application in _render_current()."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
        window.map_view.set_edge_tick_interval(other)
        window.terrain_style_combo.setCurrentText("Flat")
        QApplication.processEvents()
        window.terrain_style_combo.setCurrentText("Stepped")
        QApplication.processEvents()
        item = window.map_view._edge_tick_item
        assert item.isVisible()
        assert item._interval == other
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- the leak detector -----------------------------------------------------


def test_an_interior_capture_is_byte_identical_with_ticks_on_and_off() -> None:
    """The fixed bar for this whole feature: the overlay must never reach the
    canvas. An interior probe sees only terrain, so if these two differ by a
    byte the marks are being composited into map pixels rather than drawn
    over them."""
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry as ig
    from testkit.qt_capture import scene_rect_to_array

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        proj = view._iso_proj
        sx, sy = ig.tile_screen_origin(_ANCHOR[0], _ANCHOR[1], 0, proj)
        rect = QRectF(sx, sy, 4 * proj.half_w, 4 * proj.half_h)

        view.set_edge_ticks(False)
        off = scene_rect_to_array(view.scene(), rect)
        view.set_edge_ticks(True)
        on = scene_rect_to_array(view.scene(), rect)

        assert np.array_equal(off, on)
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- the View > Distance Ticks submenu -------------------------------------


def _plain_window():
    """A ViewerWindow with no scenario loaded. The submenu is not gated on
    having a map, so most menu questions need nothing loaded."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    return ViewerWindow()


def test_building_the_menu_writes_no_config_file(tmp_path) -> None:
    """checked= is set in each QAction's constructor and the handlers are
    connected only afterwards. Reversed, every ViewerWindow() in the gui tier
    would persist to disk during _build_menu_bar."""
    window = _plain_window()
    try:
        assert not (tmp_path / "config.yaml").exists()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_menu_opens_at_the_persisted_state(monkeypatch) -> None:
    """Pinned via the module globals, never settings.set_*(), which would
    persist to disk from a test."""
    from descape import settings

    other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
    monkeypatch.setattr(settings, "_distance_ticks", True)
    monkeypatch.setattr(settings, "_distance_tick_interval", other)
    window = _plain_window()
    try:
        assert window.distance_ticks_action.isChecked()
        assert window.distance_tick_interval_actions[other].isChecked()
        assert not window.distance_tick_interval_actions[edge_ticks.TICK_INTERVAL_DEFAULT].isChecked()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_menu_defaults_to_off_at_the_default_interval() -> None:
    window = _plain_window()
    try:
        assert not window.distance_ticks_action.isChecked()
        assert window.distance_tick_interval_actions[edge_ticks.TICK_INTERVAL_DEFAULT].isChecked()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_switching_interval_calls_the_handler_exactly_once() -> None:
    """The discriminating form. QActionGroup unchecks the outgoing action
    before it checks the incoming one, so without the `on and` guard the
    handler fires twice and the FIRST call carries the stale interval.
    Asserting only the final value passes either way, because the last call
    is correct in both. The call count is what separates them."""
    window = _plain_window()
    try:
        calls: list[int] = []
        window._on_distance_tick_interval = calls.append
        other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
        window.distance_tick_interval_actions[other].setChecked(True)
        assert calls == [other]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_interval_actions_are_mutually_exclusive() -> None:
    window = _plain_window()
    try:
        other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
        window.distance_tick_interval_actions[other].setChecked(True)
        checked = [n for n, a in window.distance_tick_interval_actions.items() if a.isChecked()]
        assert checked == [other]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_toggle_persists_and_reaches_the_view() -> None:
    from descape import settings

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.distance_ticks_action.setChecked(True)
        assert settings.get_distance_ticks() is True
        assert window.map_view._edge_tick_item.isVisible()
        window.distance_ticks_action.setChecked(False)
        assert settings.get_distance_ticks() is False
        assert not window.map_view._edge_tick_item.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_interval_choice_persists_and_reaches_the_view() -> None:
    from descape import settings

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
        window.distance_tick_interval_actions[other].setChecked(True)
        assert settings.get_distance_tick_interval() == other
        assert window.map_view._edge_tick_item._interval == other
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_submenu_is_never_greyed_out() -> None:
    """Not gated by mode, style, or even having a map, unlike its two View
    neighbours. Pinned rather than assumed, since nothing greys it out to
    explain otherwise."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        for mode_action in (window.mode_view_action, window.mode_terrain_action, window.mode_triggers_action):
            mode_action.trigger()
            QApplication.processEvents()
            assert window.distance_ticks_action.isEnabled()
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            assert window.distance_ticks_action.isEnabled(), style
        window.map_view.clear_image()
        assert window.distance_ticks_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_keybind_row_ships_unbound() -> None:
    """Unbound needs no collision audit, and leaves the obvious mnemonic to
    the separately backlogged Ruler tool."""
    from descape import settings

    assert settings.get_default_keybind("view_distance_ticks") == ""
    window = _plain_window()
    try:
        assert window._keybind_actions["view_distance_ticks"] is window.distance_ticks_action
        assert window.distance_ticks_action.shortcut().isEmpty()
    finally:
        window.edit_history.mark_saved()
        window.close()
