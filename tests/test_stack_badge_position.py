"""GH #100: the stacked-unit badge's position setting and background colour.

Two layers. StackBadgeItem's box placement is checked synthetically (identity
transform, QImage painter), and MapView's anchors through a real offscreen
window in Flat, Flat+Isometric View, Stepped and Sloped, mapped to viewport
coordinates so the Flat-iso view transform is part of what is under test.
"""

from __future__ import annotations

import pytest

from descape import settings
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_VILLAGER = 83
_STACK_TILE = (40, 40)


# --- StackBadgeItem, synthetic ----------------------------------------------


def _paint(item):
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtWidgets import QStyleOptionGraphicsItem

    image = QImage(400, 400, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        item.paint(painter, QStyleOptionGraphicsItem())
    finally:
        painter.end()
    return image


def _item(anchor, centre, *, tucked: bool = True, background=None):
    from PyQt5.QtGui import QColor

    from descape.viewer_canvas import StackBadgeItem

    conftest.ensure_qapp()
    item = StackBadgeItem(40.0, QColor("#ffffff"), background)
    item.set_placement(tucked)
    item.set_badges([(anchor, centre, 3)])
    return item


def test_a_tucked_box_sits_gap_inside_an_anchor_whose_centre_is_up_left() -> None:
    from PyQt5.QtCore import QPointF

    item = _item(QPointF(100, 100), QPointF(80, 80))
    _paint(item)
    (box,) = item.drawn_boxes
    gap = item.GAP_PX
    assert box.right() == pytest.approx(100 - gap)
    assert box.bottom() == pytest.approx(100 - gap)


def test_a_tucked_box_mirrors_for_a_centre_down_right() -> None:
    from PyQt5.QtCore import QPointF

    item = _item(QPointF(100, 100), QPointF(120, 120))
    _paint(item)
    (box,) = item.drawn_boxes
    assert box.left() == pytest.approx(100 + item.GAP_PX)
    assert box.top() == pytest.approx(100 + item.GAP_PX)


def test_a_shared_axis_centres_the_box_on_it() -> None:
    from PyQt5.QtCore import QPointF

    item = _item(QPointF(100, 100), QPointF(100.3, 80))
    _paint(item)
    (box,) = item.drawn_boxes
    assert box.center().x() == pytest.approx(100)
    assert box.bottom() == pytest.approx(100 - item.GAP_PX)


def test_above_keeps_the_original_box_exactly() -> None:
    """Today's box: bottom-centre GAP_PX above the anchor, whatever the centre says."""
    from PyQt5.QtCore import QPointF

    item = _item(QPointF(100, 100), QPointF(120, 120), tucked=False)
    _paint(item)
    (box,) = item.drawn_boxes
    assert box.center().x() == pytest.approx(100)
    assert box.bottom() == pytest.approx(100 - item.GAP_PX)


def test_the_background_keeps_its_alpha_after_a_colour_change() -> None:
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QColor

    item = _item(QPointF(100, 100), QPointF(80, 80), background=QColor("#2040c0"))
    assert item._background.alpha() == item.BACKGROUND_ALPHA == 200
    item.set_background_color(QColor(255, 0, 0, 17))
    assert (item._background.red(), item._background.green(), item._background.blue()) == (255, 0, 0)
    assert item._background.alpha() == 200


def test_a_custom_background_paints_at_the_same_translucency_as_black() -> None:
    """Must-show 5, counted: the box's premultiplied alpha is the same for both colours."""
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QColor

    alphas = []
    for colour in ("#000000", "#c02040"):
        item = _item(QPointF(100, 100), QPointF(80, 80), background=QColor(colour))
        image = _paint(item)
        (box,) = item.drawn_boxes
        # A corner-ish interior pixel, clear of the rounded corner and the digit.
        pixel = QColor.fromRgba(image.pixel(int(box.left()) + 3, int(box.center().y())))
        alphas.append(pixel.alpha())
    assert alphas[0] == alphas[1] == 200


# --- MapView anchors, real window --------------------------------------------


def _window(style: str, *, iso: bool = False, position: str = "bottom_right"):
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window()
    if style == "Flat":
        window.iso_action.setChecked(iso)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Stack", [1]):
        for _ in range(3):
            model.add(1, _VILLAGER, _STACK_TILE[0] + 0.5, _STACK_TILE[1] + 0.5)
    window.map_view.set_stack_badge_position(position)
    window.resize(1000, 800)
    window.show()
    QApplication.processEvents()
    # Flat + Isometric View renders through a real iso cache: MapView is "stepped" there.
    assert window.map_view._terrain_style == window._render_style
    assert window.map_view._terrain_style == ("stepped" if iso else style.lower())
    assert _STACK_TILE in window.map_view._stack_groups
    _zoom_on_stack(window.map_view)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _to_view(map_view, point):
    return map_view.viewportTransform().map(point)


def _anchor_and_centre(map_view):
    (pair,) = map_view._stack_badge_item.badge_anchors()
    return _to_view(map_view, pair[0]), _to_view(map_view, pair[1])


def _outline(map_view):
    polygon = map_view._tile_polygon(*_STACK_TILE)
    return [_to_view(map_view, polygon[i]) for i in range(polygon.count())]


def _extreme(outline, key, pick):
    """Mean of the outline points within 0.5 px of the extreme: Sloped's
    staircase silhouette has a flat run of pixels at each vertex, not one point."""
    from PyQt5.QtCore import QPointF

    best = pick(key(p) for p in outline)
    near = [p for p in outline if abs(key(p) - best) <= 0.5]
    return QPointF(sum(p.x() for p in near) / len(near), sum(p.y() for p in near) / len(near))


def _lower_right_midpoint(outline):
    lowest = _extreme(outline, lambda p: p.y(), max)
    rightmost = _extreme(outline, lambda p: p.x(), max)
    return (lowest + rightmost) / 2.0


def _top_vertex(outline):
    return _extreme(outline, lambda p: p.y(), min)


def _tolerance(map_view) -> float:
    """1 px, or one scene pixel in Sloped, whose silhouette is a 1-scene-px staircase."""
    if map_view._terrain_style != "sloped":
        return 1.0
    t = map_view.viewportTransform()
    return max(1.0, abs(t.m11()) + abs(t.m21())) + 0.01


_ISO_CASES = [("Flat", True), ("Stepped", False), ("Sloped", False)]
_ALL_CASES = [("Flat", False), *_ISO_CASES]


@pytest.mark.parametrize(("style", "iso"), _ALL_CASES)
def test_bottom_right_anchors_down_right_of_the_centre(style: str, iso: bool) -> None:
    window = _window(style, iso=iso)
    try:
        anchor, centre = _anchor_and_centre(window.map_view)
        assert anchor.x() > centre.x() + 1
        assert anchor.y() > centre.y() + 1
    finally:
        _close(window)


@pytest.mark.parametrize(("style", "iso"), _ISO_CASES)
def test_iso_bottom_right_is_the_lower_right_edge_midpoint(style: str, iso: bool) -> None:
    window = _window(style, iso=iso)
    try:
        map_view = window.map_view
        anchor, _centre = _anchor_and_centre(map_view)
        outline = _outline(map_view)
        expected = _lower_right_midpoint(outline)
        tol = _tolerance(map_view)
        assert abs(anchor.x() - expected.x()) <= tol
        assert abs(anchor.y() - expected.y()) <= tol
    finally:
        _close(window)


def test_square_bottom_right_is_the_bottom_right_corner() -> None:
    window = _window("Flat")
    try:
        anchor, _centre = _anchor_and_centre(window.map_view)
        outline = _outline(window.map_view)
        assert anchor.x() == pytest.approx(max(p.x() for p in outline), abs=1e-6)
        assert anchor.y() == pytest.approx(max(p.y() for p in outline), abs=1e-6)
    finally:
        _close(window)


@pytest.mark.parametrize(("style", "iso"), _ISO_CASES)
def test_above_in_iso_is_the_top_vertex(style: str, iso: bool) -> None:
    """Pins the Flat-iso fix: the old bounding-rect top-centre landed on the
    screen's upper-left edge there, not the top vertex."""
    window = _window(style, iso=iso, position="above")
    try:
        map_view = window.map_view
        anchor, _centre = _anchor_and_centre(map_view)
        top = _top_vertex(_outline(map_view))
        tol = _tolerance(map_view)
        assert abs(anchor.x() - top.x()) <= tol
        assert abs(anchor.y() - top.y()) <= tol
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("position", "sx", "sy"),
    [("bottom_right", 1, 1), ("bottom_left", -1, 1), ("top_right", 1, -1), ("top_left", -1, -1)],
)
@pytest.mark.parametrize(("style", "iso"), _ALL_CASES)
def test_each_tucked_position_names_its_screen_quadrant(style: str, iso: bool, position: str, sx: int, sy: int) -> None:
    window = _window(style, iso=iso, position=position)
    try:
        anchor, centre = _anchor_and_centre(window.map_view)
        assert (anchor.x() - centre.x()) * sx > 1
        assert (anchor.y() - centre.y()) * sy > 1
    finally:
        _close(window)


def test_mapview_set_isometric_re_anchors_the_same_item() -> None:
    """MapView's own transform-based iso path (unreachable from the View menu,
    which re-renders Flat+Iso as a real iso cache): re-anchored in place."""
    window = _window("Flat")
    try:
        map_view = window.map_view
        item = map_view._stack_badge_item
        square = item.badge_anchors()[0][0]
        map_view.set_isometric(True)
        assert map_view._stack_badge_item is item
        assert item.badge_anchors()[0][0] != square
        _zoom_on_stack(map_view)
        anchor, _centre = _anchor_and_centre(map_view)
        outline = _outline(map_view)
        expected = _lower_right_midpoint(outline)
        tol = _tolerance(map_view)
        assert abs(anchor.x() - expected.x()) <= tol
        assert abs(anchor.y() - expected.y()) <= tol
    finally:
        _close(window)


def test_mapview_transform_iso_above_is_the_top_vertex() -> None:
    """The old bounding-rect top-centre lands on the screen's upper-left edge
    under MapView's scale + rotate iso transform, not the top vertex."""
    window = _window("Flat", position="above")
    try:
        map_view = window.map_view
        map_view.set_isometric(True)
        _zoom_on_stack(map_view)
        anchor, _centre = _anchor_and_centre(map_view)
        top = _top_vertex(_outline(map_view))
        assert abs(anchor.x() - top.x()) <= 1
        assert abs(anchor.y() - top.y()) <= 1
    finally:
        _close(window)


def test_toggling_isometric_view_in_the_menu_re_anchors_the_badge() -> None:
    window = _window("Flat")
    try:
        window.iso_action.setChecked(True)
        map_view = window.map_view
        _zoom_on_stack(map_view)
        anchor, _centre = _anchor_and_centre(map_view)
        outline = _outline(map_view)
        assert len(outline) == 4
        expected = _lower_right_midpoint(outline)
        tol = _tolerance(map_view)
        assert abs(anchor.x() - expected.x()) <= tol
        assert abs(anchor.y() - expected.y()) <= tol
    finally:
        _close(window)


def test_an_elevation_edit_re_anchors_the_badge_in_stepped() -> None:
    window = _window("Stepped")
    try:
        map_view = window.map_view
        before = map_view._stack_badge_item.badge_anchors()[0][0]
        raised = map_view._iso_elevations.copy()
        raised[_STACK_TILE[1], _STACK_TILE[0]] += 4
        map_view._iso_elevations = raised
        map_view.refresh_elevation_overlays()
        after = map_view._stack_badge_item.badge_anchors()[0][0]
        assert after.y() < before.y()
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_hidden_badge_layer_skips_the_elevation_rebuild_and_catches_up_on_show(style) -> None:
    window = _window(style)
    try:
        map_view = window.map_view
        item = map_view._stack_badge_item
        before = item.badge_anchors()[0][0]
        window.mode_combo.setCurrentText("Terrain")
        assert not item.isVisible()
        calls = []
        real = map_view._stack_badge_anchor
        map_view._stack_badge_anchor = lambda *a: calls.append(a) or real(*a)
        mm = window.scenario.map_manager
        history = EditHistory()
        history.begin_stroke(mm.terrain)
        set_tile_elevation(mm, *_STACK_TILE, mm.get_tile(*_STACK_TILE).elevation + 4)
        # The real stroke path, so Sloped's corner_rise follows the edit too.
        window._apply_dirty(history.commit_stroke("raise", mm.terrain))
        assert calls == []
        assert item.badge_anchors()[0][0] == before
        window.mode_combo.setCurrentText("Units")
        assert item.isVisible()
        assert calls
        assert item.badge_anchors()[0][0].y() < before.y()
    finally:
        window.map_view.__dict__.pop("_stack_badge_anchor", None)
        _close(window)


def _zoom_on_stack(map_view) -> None:
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtWidgets import QApplication

    rect = QRectF()
    for dx in (-2, 2):
        for dy in (-2, 2):
            rect = rect.united(map_view._tile_polygon(_STACK_TILE[0] + dx, _STACK_TILE[1] + dy).boundingRect())
    map_view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()
    map_view.viewport().grab()


@pytest.mark.parametrize("position", ["bottom_right", "bottom_left", "top_right", "top_left"])
@pytest.mark.parametrize(("style", "iso"), _ALL_CASES)
def test_a_tucked_badge_is_drawn_inside_its_own_tile(style: str, iso: bool, position: str) -> None:
    """Must-show 1, counted: at a zoom where the badge fits, its box centre is inside its tile."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPolygonF

    window = _window(style, iso=iso, position=position)
    try:
        map_view = window.map_view
        _zoom_on_stack(map_view)
        (box,) = map_view._stack_badge_item.drawn_boxes
        tile = QPolygonF(_outline(map_view))
        assert tile.containsPoint(box.center(), Qt.OddEvenFill), f"{box} outside {tile.boundingRect()}"
    finally:
        _close(window)


@pytest.mark.parametrize(("style", "iso"), _ALL_CASES)
def test_badges_on_neighbouring_tiles_land_on_three_distinct_tiles(style: str, iso: bool) -> None:
    """Must-show 2, counted: three stacks on touching tiles (the Castle-footprint
    case) draw three boxes, each inside exactly one of the three tiles."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPolygonF

    window = _window(style, iso=iso)
    try:
        map_view = window.map_view
        tiles = [(_STACK_TILE[0] + 1, _STACK_TILE[1]), (_STACK_TILE[0], _STACK_TILE[1] + 1)]
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Stack", [1]):
            for tx, ty in tiles:
                for _ in range(2):
                    model.add(1, _VILLAGER, tx + 0.5, ty + 0.5)
        _zoom_on_stack(map_view)
        tiles.append(_STACK_TILE)
        boxes = map_view._stack_badge_item.drawn_boxes
        assert len(boxes) == 3
        hit_tiles = []
        for box in boxes:
            inside = [
                t for t in tiles
                if QPolygonF([_to_view(map_view, p) for p in map_view._tile_polygon(*t)]).containsPoint(
                    box.center(), Qt.OddEvenFill
                )
            ]
            assert len(inside) == 1, (box, inside)
            hit_tiles.append(inside[0])
        assert sorted(hit_tiles) == sorted(tiles)
    finally:
        _close(window)


def test_the_badge_device_size_is_identical_across_styles() -> None:
    """Must-show 3, counted."""
    sizes = set()
    for style, iso in _ALL_CASES:
        window = _window(style, iso=iso)
        try:
            _zoom_on_stack(window.map_view)
            (box,) = window.map_view._stack_badge_item.drawn_boxes
            sizes.add((round(box.width(), 3), round(box.height(), 3)))
        finally:
            _close(window)
    assert len(sizes) == 1, sizes


# --- Settings > Appearance ------------------------------------------------------


def test_the_appearance_combo_persists_and_re_anchors_the_live_item(monkeypatch) -> None:
    from descape.viewer import SettingsDialog

    window = _window("Stepped")
    dialog = SettingsDialog(window)
    try:
        map_view = window.map_view
        item = map_view._stack_badge_item
        combo = dialog.stack_badge_position_combo
        assert combo.currentData() == "bottom_right"
        before = item.badge_anchors()[0][0]
        combo.setCurrentIndex(combo.findData("top_left"))
        assert settings.get_stack_badge_position() == "top_left"
        monkeypatch.setattr(settings, "_stack_badge_position", None)
        assert settings.get_stack_badge_position() == "top_left", "not persisted to the config file"
        assert map_view._stack_badge_item is item
        after, centre = _anchor_and_centre(map_view)
        assert item.badge_anchors()[0][0] != before
        assert after.x() < centre.x() and after.y() < centre.y(), "top_left is not up-left of the centre"
        assert "Stacked-unit badge position: Top-left" in window.status_log.toPlainText()
        combo.setCurrentIndex(combo.findData("above"))
        assert not item._tucked
    finally:
        dialog.close()
        _close(window)


def test_apply_overlay_colors_re_inks_the_background_in_place() -> None:
    from PyQt5.QtGui import QColor

    window = _window("Flat")
    try:
        map_view = window.map_view
        item = map_view._stack_badge_item
        settings.set_overlay_color("unit_stack_background", "#3060a0")
        map_view.apply_overlay_colors()
        assert map_view._stack_badge_item is item
        assert item._background == QColor(0x30, 0x60, 0xA0, item.BACKGROUND_ALPHA)
    finally:
        _close(window)
