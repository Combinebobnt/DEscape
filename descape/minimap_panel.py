"""The Minimap dock's widget (GH #54): minimap.minimap_rgb()'s image fitted
in the main view's orientation, with the main view's viewport outlined.

The image is one pixel per tile. The window hands over tile_to_view, the
main view's map-point-to-viewport transform, on every pan, zoom and resize;
the fit divides out its translation and scale and keeps its shape, so the
map draws as a diamond or a square, as the main view does, at a size that
never follows the main view's zoom.

The panel's own zoom (GH #404) scales that fit about a map-point centre,
clamped so no off-map band shows on an axis the zoomed map overflows. It is
relative to the fit, so a panel resize keeps it, and a new document resets it.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QMenu, QWidget

MIN_HEIGHT = 64
DEFAULT_HEIGHT = 180
DEFAULT_WIDTH = 2 * DEFAULT_HEIGHT
MARGIN = 4
MAX_ZOOM = 8.0
ZOOM_STEP = math.sqrt(2)
WHEEL_NOTCH_UNITS = 120
# A zoomed minimap scrolls along once the viewport box's centre leaves the panel inset by this share a side.
FOLLOW_INSET = 0.15
# MapView.OUTSIDE_MAP_COLOR, so the letterbox reads as off-map.
FILL_COLOR = QColor(35, 35, 35)
BORDER_COLOR = QColor(150, 150, 150)
BOX_SHADOW_COLOR = QColor(0, 0, 0, 150)
BOX_COLOR = QColor(255, 255, 255)
TOOLTIP = (
    "The whole map, with the visible area outlined. Click or drag to move the view; "
    "wheel to zoom; right-click for zoom options."
)


def _quad(rect: QRectF) -> QPolygonF:
    return QPolygonF([rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()])


def _clamp_axis(lo: float, hi: float, extent: float, centre: float, scale: float) -> float:
    """The offset on one axis putting view coord `centre` at the panel's middle,
    with no band past MARGIN outside [lo, hi] once the scaled span overflows."""
    if (hi - lo) * scale <= extent - 2 * MARGIN:
        return (extent - (hi - lo) * scale) / 2 - lo * scale
    offset = extent / 2 - centre * scale
    return min(max(offset, extent - MARGIN - hi * scale), MARGIN - lo * scale)


def clamped_offset(box: QRectF, width: float, height: float, scale: float, centre: QPointF) -> QPointF:
    """The translation for view-space `box` drawn at `scale`, centred on view point `centre` where it can be."""
    return QPointF(
        _clamp_axis(box.left(), box.right(), width, centre.x(), scale),
        _clamp_axis(box.top(), box.bottom(), height, centre.y(), scale),
    )


class MinimapPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(MIN_HEIGHT)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(TOOLTIP)
        self._image: QImage | None = None
        self._tile_to_view: QTransform | None = None
        self._viewport = QRectF()
        self._dragging = False
        self._zoom = 1.0
        # The map point shown at the panel's middle while zoomed, None for the map's centre.
        self._centre: QPointF | None = None
        self._wheel_accum = 0
        self._last_box_centre: QPointF | None = None
        self._follow_pending = False
        # Map point to centre the main view on, and the end of a click, drag or menu. Assigned by the window.
        self.on_pan = lambda mx, my: None
        self.on_released = lambda: None

    def sizeHint(self) -> QSize:
        return QSize(DEFAULT_WIDTH, DEFAULT_HEIGHT)

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_image(self, rgb: np.ndarray | None) -> None:
        """An (h, w, 3) uint8 array, pixel (y, x) = tile (x, y), or None for no map (which resets the zoom)."""
        if rgb is None:
            self._image = None
            self._zoom, self._centre, self._wheel_accum = 1.0, None, 0
            self._last_box_centre, self._follow_pending = None, False
        else:
            h, w = rgb.shape[:2]
            data = np.ascontiguousarray(rgb, dtype=np.uint8)
            self._image = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_view(self, tile_to_view: QTransform | None, viewport: QRectF) -> None:
        """The main view's map-point-to-viewport transform and its viewport rect.
        Also called after each rebuild, so a zoomed window follows only a moved box."""
        self._tile_to_view = tile_to_view
        self._viewport = QRectF(viewport)
        centre = self._box_centre()
        moved = centre is not None and (
            self._last_box_centre is None
            or abs(centre.x() - self._last_box_centre.x()) > 1e-6
            or abs(centre.y() - self._last_box_centre.y()) > 1e-6
        )
        self._last_box_centre = centre
        if moved:
            if self._dragging:
                self._follow_pending = True
            else:
                self._follow()
        self.update()

    def _map_quad(self) -> QPolygonF:
        return _quad(QRectF(0, 0, self._image.width(), self._image.height()))

    def _base(self):
        """(view-space map box, the whole-map fit scale), or None while there is nothing to draw."""
        if self._image is None or self._tile_to_view is None:
            return None
        box = self._tile_to_view.map(self._map_quad()).boundingRect()
        if box.width() <= 0 or box.height() <= 0:
            return None
        scale = min((self.width() - 2 * MARGIN) / box.width(), (self.height() - 2 * MARGIN) / box.height())
        if scale <= 0:
            return None
        return box, scale

    def _view_inverse(self) -> QTransform | None:
        inverse, ok = self._tile_to_view.inverted()
        return inverse if ok else None

    def _box_centre(self) -> QPointF | None:
        """The main view's viewport centre as a map point."""
        if self._tile_to_view is None or self._viewport.isEmpty():
            return None
        inverse = self._view_inverse()
        return None if inverse is None else inverse.map(self._viewport.center())

    def _placement(self):
        """(offset, scale) of the zoomed, clamped fit in view space, or None while there is nothing to draw.
        `_centre` stays the requested point, so a resize that briefly fits an axis keeps the position."""
        base = self._base()
        if base is None:
            return None
        box, scale = base
        scale *= self._zoom
        centre = self._centre or QPointF(self._image.width() / 2, self._image.height() / 2)
        return clamped_offset(box, self.width(), self.height(), scale, self._tile_to_view.map(centre)), scale

    def fit(self) -> QTransform | None:
        """Map point to widget pixel, or None while there is nothing to draw."""
        placement = self._placement()
        if placement is None:
            return None
        offset, scale = placement
        return self._tile_to_view * QTransform(scale, 0, 0, scale, offset.x(), offset.y())

    def _shift_centre(self, dx: float, dy: float) -> None:
        """Move the shown map by (dx, dy) widget pixels' worth of view space, starting from what is shown."""
        placement = self._placement()
        inverse = self._view_inverse() if placement is not None else None
        if inverse is None:
            return
        offset, scale = placement
        shown_x, shown_y = (self.width() / 2 - offset.x()) / scale, (self.height() / 2 - offset.y()) / scale
        self._centre = inverse.map(QPointF(shown_x + dx / scale, shown_y + dy / scale))

    def _follow(self) -> None:
        """Scroll a zoomed minimap the least amount that brings the box's centre back inside the inner area."""
        if self._zoom == 1.0:
            return
        fit, centre = self.fit(), self._box_centre()
        if fit is None or centre is None:
            return
        pos = fit.map(centre)
        inset_x, inset_y = self.width() * FOLLOW_INSET, self.height() * FOLLOW_INSET
        dx = pos.x() - min(max(pos.x(), inset_x), self.width() - inset_x)
        dy = pos.y() - min(max(pos.y(), inset_y), self.height() - inset_y)
        if dx or dy:
            self._shift_centre(dx, dy)

    def zoom_by(self, notches: int, anchor: QPointF | None = None) -> None:
        """Zoom `notches` ZOOM_STEPs (negative out), keeping the map point under widget pixel `anchor` there."""
        fit = self.fit()
        if fit is None or notches == 0:
            return
        zoom = min(max(self._zoom * ZOOM_STEP ** notches, 1.0), MAX_ZOOM)
        if abs(zoom - 1.0) < 1e-6:
            zoom = 1.0
        elif abs(zoom - MAX_ZOOM) < 1e-6:
            zoom = MAX_ZOOM
        if zoom == self._zoom:
            return
        anchor = QPointF(self.width() / 2, self.height() / 2) if anchor is None else QPointF(anchor)
        inverse, ok = fit.inverted()
        view_inverse = self._view_inverse()
        self._zoom = zoom
        if ok and view_inverse is not None:
            # The centre that puts the point under `anchor` back there at the new scale; the fit clamps it.
            under = self._tile_to_view.map(inverse.map(anchor))
            scale = self._base()[1] * zoom
            middle = QPointF(under.x() + (self.width() / 2 - anchor.x()) / scale, under.y() + (self.height() / 2 - anchor.y()) / scale)
            self._centre = view_inverse.map(middle)
        self.update()

    def fit_whole_map(self) -> None:
        self._zoom, self._centre = 1.0, None
        self.update()

    def viewport_polygon(self) -> QPolygonF | None:
        """The main view's viewport in widget pixels, or None."""
        fit = self.fit()
        if fit is None:
            return None
        inverse = self._view_inverse()
        if inverse is None:
            return None
        return fit.map(inverse.map(_quad(self._viewport)))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), FILL_COLOR)
        fit = self.fit()
        if fit is not None:
            painter.setClipRect(self.rect())
            painter.setTransform(fit)
            painter.drawImage(QRectF(0, 0, self._image.width(), self._image.height()), self._image)
            painter.resetTransform()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(BORDER_COLOR, 1))
            painter.drawPolygon(fit.map(self._map_quad()))
            box = self.viewport_polygon()
            if box is not None:
                painter.setPen(QPen(BOX_SHADOW_COLOR, 3))
                painter.drawPolygon(box)
                painter.setPen(QPen(BOX_COLOR, 1.5))
                painter.drawPolygon(box)
        painter.end()

    def _pan_to(self, pos) -> None:
        fit = self.fit()
        if fit is None:
            return
        inverse, ok = fit.inverted()
        if ok:
            point = inverse.map(QPointF(pos))
            self.on_pan(point.x(), point.y())

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._pan_to(event.pos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and event.buttons() & Qt.LeftButton:
            self._pan_to(event.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _end_drag(self) -> None:
        """Ends a left drag and applies the follow it held back."""
        self._dragging = False
        if self._follow_pending:
            self._follow_pending = False
            self._follow()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._dragging:
            super().mouseReleaseEvent(event)
            return
        self._end_drag()
        event.accept()
        self.on_released()

    def hideEvent(self, event) -> None:
        # A drag cut short by a close, float or tab switch never sees its release.
        self._end_drag()
        super().hideEvent(event)

    def wheelEvent(self, event) -> None:
        # Always accepted, so a wheel over the minimap never scrolls a parent or zooms the map.
        event.accept()
        if event.modifiers() != Qt.NoModifier:
            return
        self._wheel_accum += event.angleDelta().y()
        notches = int(self._wheel_accum / WHEEL_NOTCH_UNITS)
        if notches:
            self._wheel_accum -= notches * WHEEL_NOTCH_UNITS
            self.zoom_by(notches, event.position())

    def contextMenuEvent(self, event) -> None:
        event.accept()
        # The menu's grab swallows a held left button's release.
        self._end_drag()
        if self.fit() is None:
            return
        menu = QMenu(self)
        zoom_in = menu.addAction("Zoom In", lambda: self.zoom_by(1))
        zoom_in.setEnabled(self._zoom < MAX_ZOOM)
        zoom_out = menu.addAction("Zoom Out", lambda: self.zoom_by(-1))
        zoom_out.setEnabled(self._zoom > 1.0)
        menu.addAction("Fit Whole Map", self.fit_whole_map)
        menu.exec_(event.globalPos())
        self.on_released()
