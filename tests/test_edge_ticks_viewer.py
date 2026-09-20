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
from itertools import pairwise

import numpy as np
import pytest

from descape import edge_ticks, settings
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Interior probe, matching tests/test_seam_viewer.py's own _ANCHOR on this
# same 120x120 template and for the same reason: the map-extent outline cuts
# through any probe placed near a corner.
_ANCHOR = (60, 60)

MAP_W, MAP_H, TILE_PX = 20, 16, 8


def _bare_scene(
    interval: int = edge_ticks.TICK_INTERVAL_DEFAULT,
    scale: float = 1.0,
    font_px: int = edge_ticks.LABEL_FONT_PX,
):
    """A scene holding nothing but one flat-mode EdgeTickItem."""
    conftest.ensure_qapp()
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QGraphicsScene

    from descape.viewer_canvas import EdgeTickItem

    map_rect = QRectF(0, 0, MAP_W * TILE_PX, MAP_H * TILE_PX)
    item = EdgeTickItem(MAP_W, MAP_H, interval, map_rect, tile_px=TILE_PX)
    # Font set BEFORE the scale, since set_min_view_scale's pad computation
    # reads self._font.pixelSize() at call time.
    item.set_label_font_px(font_px)
    item.set_min_view_scale(scale)
    scene = QGraphicsScene()
    scene.addItem(item)
    return scene, item


def _paint_at(item, scale_x: float, scale_y: float | None = None, rotate_deg: float = 0.0):
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

    from descape.viewer_canvas import EdgeTickItem
    from testkit.qt_capture import scene_rect_to_array

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
    column = round((edge_ticks.TICK_INTERVAL_DEFAULT * TILE_PX - rect.left()) * zoom)
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


# --- exposedRect culling ---------------------------------------------------

# Device-space origin of the scene in _paint_exposed's image, far enough in
# that the outward-running y0/x0 ticks land inside it.
_CULL_IMG_PX = 400
_CULL_ORIGIN = 100.0
_CULL_CASES = ("interior", "tick_tips", "labels_only", "axis_letter_only")


class _ExposedOption:
    """Stand-in for QStyleOptionGraphicsItem: paint() reads exposedRect and
    nothing else off it, and a real one cannot be handed an arbitrary rect."""

    def __init__(self, rect):
        self.exposedRect = rect


def _paint_exposed(item, exposed, clip):
    """One paint() clipped to `clip` (scene coords), told `exposed` is the
    exposed area. Clipping both renders the same way is what makes the two
    comparable: Qt's own paint dispatch clips to the exposed region too, so
    the only thing `exposed` may change is which draw calls are issued.

    Hands back (pixels, stats)."""
    from PyQt5.QtGui import QImage, QPainter, QTransform

    from testkit.qt_capture import qimage_rgb888_to_array

    image = QImage(_CULL_IMG_PX, _CULL_IMG_PX, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    transform = QTransform()
    transform.translate(_CULL_ORIGIN, _CULL_ORIGIN)
    painter.setWorldTransform(transform)
    painter.setClipRect(clip)
    item.paint(painter, _ExposedOption(exposed))
    stats = item.stats
    painter.end()
    return qimage_rgb888_to_array(image), stats


def _cull_rect(name: str):
    """The scene-space rect for one _CULL_CASES exposure.

    `interior` holds whole marks, anchors included. `tick_tips` is scene y in
    [-12, -3], where every anchor (y == 0) is outside while the outer part of
    each y0 tick is inside: the partial-visibility case a naive
    anchor-in-rect cull drops. `labels_only` is scene y in [-32, -18], past
    even a major tick's 12px reach, so only label boxes can land there."""
    from PyQt5.QtCore import QRectF

    return {
        "interior": QRectF(30, -34, 60, 60),
        "tick_tips": QRectF(-10, -12, 200, 9),
        "labels_only": QRectF(-10, -32, 200, 14),
        "axis_letter_only": QRectF(60, -68, 40, 20),
    }[name]


@pytest.mark.parametrize("name", _CULL_CASES)
def test_culling_to_the_exposed_rect_is_pixel_identical(name: str) -> None:
    """The B9 bar: what reaches the screen must not depend on how much of
    the item Qt says is exposed. Each rect is rendered twice, once with the
    cull disarmed (exposed == the whole bounding rect) and once with it
    armed, and the two must be byte-identical."""
    _scene, item = _bare_scene()
    clip = _cull_rect(name)
    unculled, unculled_stats = _paint_exposed(item, item.boundingRect(), clip)
    culled, culled_stats = _paint_exposed(item, clip, clip)

    assert unculled.any(), f"{name} rendered nothing, so identity is vacuous"
    assert np.array_equal(unculled, culled)
    # ... and the cull really did fire, rather than passing by doing nothing.
    assert culled_stats.ticks_drawn < unculled_stats.ticks_drawn


def test_a_full_exposed_rect_culls_nothing() -> None:
    _scene, item = _bare_scene()
    rect = item.boundingRect()
    full, full_stats = _paint_exposed(item, rect, rect)
    _pixels, stats = _paint_exposed(item, item.boundingRect(), rect)
    assert full.any()
    assert full_stats.ticks_drawn == stats.ticks_drawn
    assert full_stats.ticks_drawn == sum(len(run.anchors) for run in item._runs)
    assert full_stats.labels_drawn == sum(sum(run.majors) for run in item._runs)


def test_labels_survive_an_exposure_that_misses_their_own_tick() -> None:
    """Labels are batched separately from the lines, so they need their own
    proof that the cull's margin covers a label box whose anchor is well
    outside the exposed rect."""
    _scene, item = _bare_scene()
    clip = _cull_rect("labels_only")
    _pixels, stats = _paint_exposed(item, clip, clip)
    assert stats.labels_drawn > 0


def test_the_axis_letter_survives_an_exposure_past_every_number() -> None:
    """`axis_letter_only` sits beyond every numeric label's reach, around
    the y0 run's middle anchor (tile 10, scene x 80), so only the letter can
    paint there and the cull margin must still keep its anchor."""
    _scene, item = _bare_scene()
    clip = _cull_rect("axis_letter_only")
    assert clip.bottom() < -edge_ticks.label_reach_px(edge_ticks.LABEL_FONT_PX)
    pixels, stats = _paint_exposed(item, clip, clip)
    assert stats.axis_labels_drawn >= 1
    assert pixels.any()


def test_a_real_view_paint_narrows_the_exposed_rect_and_culls() -> None:
    """ItemUsesExtendedStyleOption is what makes the cull reachable at all:
    without it Qt reports the full boundingRect as exposed on every repaint
    and every cull test above passes while production culls nothing.

    Driven through a real QGraphicsView, not QGraphicsScene.render(): that
    API hands every item its whole boundingRect as the exposed area
    regardless of the flag (measured), so the capture helpers this file uses
    elsewhere cannot ask this question. Same spy shape as
    tests/test_lazy_viewport.py's own exposedRect check."""
    from PyQt5.QtWidgets import QApplication, QGraphicsItem, QGraphicsView

    scene, item = _bare_scene()
    assert item.flags() & QGraphicsItem.ItemUsesExtendedStyleOption
    total = sum(len(run.anchors) for run in item._runs)
    bounding = item.boundingRect()
    seen = []
    original = item.paint

    def spy(painter, option, widget=None):
        original(painter, option, widget)
        rect = option.exposedRect
        seen.append((rect.width() * rect.height(), item.stats.ticks_drawn))

    item.paint = spy
    view = QGraphicsView(scene)
    try:
        view.resize(200, 200)
        view.show()
        QApplication.processEvents()
        QApplication.processEvents()
        view.scale(3.0, 3.0)
        QApplication.processEvents()
    finally:
        view.close()

    assert seen, "the item was never painted"
    full_area = bounding.width() * bounding.height()
    assert min(area for area, _n in seen) < full_area / 4
    assert min(n for _area, n in seen) < total


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
    for earlier, later in pairwise(seen):
        assert all(b <= a for a, b in zip(earlier, later, strict=True))
    assert seen[-1][0] is False


def test_a_dropped_edge_draws_nothing_at_all() -> None:
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 0.005)
    assert all(not lod.draw_edge for lod in stats.lod.values())
    assert stats.ticks_drawn == 0
    assert stats.labels_drawn == 0
    assert stats.axis_labels_drawn == 0


def test_the_axis_letter_drops_out_with_the_numbers() -> None:
    _scene, item = _bare_scene()
    stats, _t, _r = _paint_at(item, 0.1)
    assert stats.lod["y0"].draw_labels is False
    assert stats.labels_drawn == 0
    assert stats.axis_labels_drawn == 0


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
    assert stats.axis_labels_drawn == len(item._runs) == 4


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


@pytest.mark.parametrize("font_px", range(8, 25))  # settings.DISTANCE_TICK_FONT_PX_MIN..MAX
def test_the_pad_contains_the_device_reach_at_the_minimum_scale(font_px: int) -> None:
    _scene, item = _bare_scene(scale=0.01, font_px=font_px)
    rect = item.boundingRect()
    overhang = min(
        rect.right() - MAP_W * TILE_PX,
        rect.bottom() - MAP_H * TILE_PX,
        -rect.left(),
        -rect.top(),
    )
    assert overhang * 0.01 >= math.sqrt(2) * edge_ticks.device_reach_px(font_px)


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


@pytest.mark.parametrize("font_px", [8, edge_ticks.LABEL_FONT_PX, 24])
def test_a_resize_while_zoomed_out_keeps_the_bounding_rect_valid(font_px: int) -> None:
    """The case min(_min_linear_scale, current) exists for. resizeEvent
    re-runs _capture_zoom_baseline, and a larger viewport RAISES
    _min_linear_scale without rescaling the transform, so the current scale
    ends up below the new floor. Padding from the floor alone passes every
    static test in this file and under-pads exactly here.

    Parametrized over the font range's endpoints and the default rather than
    every legal value -- this drives a real ViewerWindow, and the pure
    per-font-px geometry is already exhaustively checked by the bare-scene
    and Qt-free tests above/in test_edge_ticks.py; this one only needs to
    confirm the min()-of-floor-and-current path still holds with a
    non-default font in the mix."""
    from PyQt5.QtWidgets import QApplication

    settings.set_distance_tick_font_px(font_px)
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
        assert overhang * current >= math.sqrt(2) * edge_ticks.device_reach_px(font_px)
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
        other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
        window.map_view.set_edge_tick_interval(other)
        assert window.map_view._edge_tick_item._interval == other
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_font_change_repaints_and_repads_live_with_no_item_churn() -> None:
    """apply_distance_tick_font()'s live-apply path -- test_overlay_colors_
    viewer.py's test_live_apply_repaints_without_replacing_the_item's shape:
    same item identity, new font actually applied, and the bounding rect
    grows with it since scene_pad depends on font_px too."""
    window = _window()
    try:
        map_view = window.map_view
        item = map_view._edge_tick_item
        item_id_before = id(item)
        rect_before = item.boundingRect()

        bigger = edge_ticks.LABEL_FONT_PX + 8
        settings.set_distance_tick_font_px(bigger)
        map_view.apply_distance_tick_font()

        assert id(map_view._edge_tick_item) == item_id_before
        assert item._font.pixelSize() == bigger
        rect_after = item.boundingRect()
        assert rect_after.width() > rect_before.width()
        assert rect_after.height() > rect_before.height()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_apply_distance_tick_font_with_no_live_item_is_a_noop() -> None:
    """The deleted-item guard: File > Close nulls _edge_tick_item (same path
    _repad_edge_ticks and set_edge_tick_interval already guard), so the
    Appearance spinbox reaching this with no map open must not raise."""
    window = _window()
    try:
        window.map_view.clear_image()
        assert window.map_view._edge_tick_item is None
        settings.set_distance_tick_font_px(edge_ticks.LABEL_FONT_PX + 4)
        window.map_view.apply_distance_tick_font()  # must not raise
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_tick_state_survives_a_terrain_style_round_trip() -> None:
    """State lives on MapView, which outlives every set_source(), so unlike
    Show sprites it needs no re-application in _render_current()."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
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

    other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
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
        other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
        window.distance_tick_interval_actions[other].setChecked(True)
        assert calls == [other]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_interval_actions_are_mutually_exclusive() -> None:
    window = _plain_window()
    try:
        other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
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
        other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
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
