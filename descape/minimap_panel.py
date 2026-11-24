"""The Minimap dock's widget (GH #54): minimap.minimap_rgb()'s image fitted
in the main view's orientation, with the main view's viewport outlined.

The image is one pixel per tile. The window hands over tile_to_view, the
main view's map-point-to-viewport transform, on every pan, zoom and resize;
the fit divides out its translation and scale and keeps its shape, so the
map draws as a diamond or a square, as the main view does, at a size that
never follows the main view's zoom.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QWidget

MIN_HEIGHT = 180
DEFAULT_WIDTH = 2 * MIN_HEIGHT
MARGIN = 4
# MapView.OUTSIDE_MAP_COLOR, so the letterbox reads as off-map.
FILL_COLOR = QColor(35, 35, 35)
BORDER_COLOR = QColor(150, 150, 150)
BOX_SHADOW_COLOR = QColor(0, 0, 0, 150)
BOX_COLOR = QColor(255, 255, 255)


def _quad(rect: QRectF) -> QPolygonF:
    return QPolygonF([rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()])


class MinimapPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(MIN_HEIGHT)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("The whole map, with the visible area outlined. Click or drag to move the view.")
        self._image: QImage | None = None
        self._tile_to_view: QTransform | None = None
        self._viewport = QRectF()
        self._dragging = False
        # Map point to centre the main view on, and the end of a click or drag. Assigned by the window.
        self.on_pan = lambda mx, my: None
        self.on_released = lambda: None

    def sizeHint(self) -> QSize:
        return QSize(DEFAULT_WIDTH, MIN_HEIGHT)

    def set_image(self, rgb: np.ndarray | None) -> None:
        """An (h, w, 3) uint8 array, pixel (y, x) = tile (x, y), or None for no map."""
        if rgb is None:
            self._image = None
        else:
            h, w = rgb.shape[:2]
            data = np.ascontiguousarray(rgb, dtype=np.uint8)
            self._image = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_view(self, tile_to_view: QTransform | None, viewport: QRectF) -> None:
        """The main view's map-point-to-viewport transform and its viewport rect."""
        self._tile_to_view = tile_to_view
        self._viewport = QRectF(viewport)
        self.update()

    def _map_quad(self) -> QPolygonF:
        return _quad(QRectF(0, 0, self._image.width(), self._image.height()))

    def fit(self) -> QTransform | None:
        """Map point to widget pixel, or None while there is nothing to draw."""
        if self._image is None or self._tile_to_view is None:
            return None
        box = self._tile_to_view.map(self._map_quad()).boundingRect()
        if box.width() <= 0 or box.height() <= 0:
            return None
        scale = min((self.width() - 2 * MARGIN) / box.width(), (self.height() - 2 * MARGIN) / box.height())
        if scale <= 0:
            return None
        dx = (self.width() - box.width() * scale) / 2 - box.x() * scale
        dy = (self.height() - box.height() * scale) / 2 - box.y() * scale
        return self._tile_to_view * QTransform(scale, 0, 0, scale, dx, dy)

    def viewport_polygon(self) -> QPolygonF | None:
        """The main view's viewport in widget pixels, or None."""
        fit = self.fit()
        if fit is None:
            return None
        inverse, ok = self._tile_to_view.inverted()
        if not ok:
            return None
        return fit.map(inverse.map(_quad(self._viewport)))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), FILL_COLOR)
        fit = self.fit()
        if fit is not None:
            painter.setTransform(fit)
            painter.drawImage(QRectF(0, 0, self._image.width(), self._image.height()), self._image)
            painter.resetTransform()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(BORDER_COLOR, 1))
            painter.drawPolygon(fit.map(self._map_quad()))
            box = self.viewport_polygon()
            if box is not None:
                painter.setClipRect(self.rect())
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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._dragging:
            super().mouseReleaseEvent(event)
            return
        self._dragging = False
        event.accept()
        self.on_released()

    def wheelEvent(self, event) -> None:
        # Swallowed, so a wheel over the minimap never scrolls a parent or zooms the map.
        event.accept()
