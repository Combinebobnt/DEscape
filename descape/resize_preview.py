"""Map > Resize Map…'s live preview (TASK-547, GH #45): the open map's terrain
with the new bounds placed by the plan's own dx/dy, so the picture cannot
disagree with what the resize will do. Read-only, and no dry run of
resize_scenario(): a spin step only moves rectangles over one cached image.

One tile-space image is drawn under one of two transforms, following the
current view: a 2:1 diamond with tile (0, 0) at the west tip, x running
up-right and y down-right (iso_geometry's convention), or a top-down square.
Nothing here assumes a square map (TASK-032).

Red marks only say what the darkened zone doesn't: see deleted_marks().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QWidget

from descape import render
from descape.find_objects import is_garrisoned
from descape.scenario_new import BLANK_TERRAIN_STRUCT
from descape.terrain_palette import color_for_terrain_id

PREVIEW_WIDTH = 300
PREVIEW_HEIGHT = 220
_MARGIN = 8
_RING_RADIUS = 5.0
_MARK_PX = 2
_OLD_MAP_DASH = [4, 3]
_BACKGROUND = QColor(24, 28, 22)  # fixed, so the white outlines read on every theme
_NEW_ONLY = QColor(255, 255, 255, 90)
_OLD_ONLY = QColor(0, 0, 0, 170)
_OUTLINE = QColor(255, 255, 255)
_MARK = QColor(255, 48, 48)
_RING = QColor(255, 214, 64)


@dataclass(frozen=True, eq=False)
class PreviewSource:
    """What the preview reads from the document, gathered once per dialog."""

    rgb: np.ndarray  # (h, w, 3) uint8, one pixel per tile; the widget's QImage wraps this memory
    new_ground: tuple[int, int, int]
    isometric: bool
    gaia_ids: frozenset[int]  # id() of unit_manager.units[0]'s units


def terrain_rgb(scenario) -> np.ndarray:
    """One colour per tile, pixel (y, x) being tile (x, y). Reads MapManager.terrain
    live, the source build_patched() writes from, so pending terrain edits show."""
    mm = scenario.map_manager
    ids = render.tile_field_mirror(scenario, "terrain_id").reshape(mm.map_height, mm.map_width)
    unique, inverse = np.unique(ids, return_inverse=True)
    lut = np.array([color_for_terrain_id(int(t)) for t in unique], dtype=np.uint8)
    return np.ascontiguousarray(lut[inverse.reshape(ids.shape)])


def preview_source(scenario, *, isometric: bool) -> PreviewSource:
    return PreviewSource(
        rgb=terrain_rgb(scenario),
        new_ground=color_for_terrain_id(BLANK_TERRAIN_STRUCT[0]),  # what remap_terrain() writes there
        isometric=isometric,
        gaia_ids=frozenset(id(unit) for unit in scenario.unit_manager.units[0]),
    )


def old_rect(plan) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the old map in new-map tiles. A refused plan's dx/dy
    describe a resize that won't happen, so it shows the old map alone."""
    if plan.refusal is not None:
        return 0, 0, plan.old_w, plan.old_h
    return plan.dx, plan.dy, plan.old_w, plan.old_h


def preview_extent(plan) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the union of the new map and the old map, in new-map tiles."""
    x, y, w, h = old_rect(plan)
    if plan.refusal is not None:
        return x, y, x + w, y + h
    return min(0, x), min(0, y), max(plan.new_w, x + w), max(plan.new_h, y + h)


def deleted_marks(plan, gaia_ids) -> list[tuple[float, float]]:
    """New-map tile points worth a red mark: deleted units that are not garrison
    occupants (they have no reliable position, their host is marked) and are
    player-owned or still anchored on the new map. Every other GAIA loss sits
    in the dark zone already, and marking those floods it red."""
    marks = []
    for unit in plan.removed:
        if is_garrisoned(unit):
            continue
        x, y = unit.x + plan.dx, unit.y + plan.dy
        anchored = 0 <= math.floor(x) < plan.new_w and 0 <= math.floor(y) < plan.new_h
        if anchored or id(unit) not in gaia_ids:
            marks.append((x, y))
    return marks


def anchor_point(plan) -> tuple[float, float]:
    """The anchored spot in new-map tiles."""
    column, row = plan.anchor.value
    return column / 2 * plan.new_w, row / 2 * plan.new_h


def tile_transform(extent, isometric: bool, width: int, height: int) -> QTransform:
    """Tile space to widget pixels: `extent` fitted inside the margin and centred."""
    x0, y0, x1, y1 = extent
    room_w, room_h = width - 2 * _MARGIN, height - 2 * _MARGIN
    if isometric:
        span = (x1 - x0) + (y1 - y0)
        scale = min(room_w / span, 2 * room_h / span)
        base = QTransform(scale, -scale / 2, scale, scale / 2, 0, 0)
    else:
        scale = min(room_w / (x1 - x0), room_h / (y1 - y0))
        base = QTransform(scale, 0, 0, scale, 0, 0)
    centre = base.map(QPointF((x0 + x1) / 2, (y0 + y1) / 2))
    return base * QTransform.fromTranslate(width / 2 - centre.x(), height / 2 - centre.y())


def _rect_path(x: float, y: float, w: float, h: float) -> QPainterPath:
    path = QPainterPath()
    path.addRect(QRectF(x, y, w, h))
    return path


def _cosmetic_pen(color: QColor, width: int = 1) -> QPen:
    pen = QPen(color)
    pen.setWidth(width)
    pen.setCosmetic(True)  # widths ignore the transform
    return pen


class ResizePreview(QWidget):
    """The old map as a thumbnail, with the plan's new bounds over it."""

    def __init__(self, source: PreviewSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.setAccessibleName("Resize preview")
        self.source = source
        self.plan = None
        self._marks = QPolygonF()
        h, w = source.rgb.shape[:2]
        self._image = QImage(source.rgb.data, w, h, 3 * w, QImage.Format_RGB888)

    def set_plan(self, plan) -> None:
        self.plan = plan
        self._marks = QPolygonF([QPointF(x, y) for x, y in deleted_marks(plan, self.source.gaia_ids)])
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        plan = self.plan
        if plan is None or plan.refusal is not None:
            old_w, old_h = self._image.width(), self._image.height()
            painter.setTransform(self._transform((0, 0, old_w, old_h)))
            painter.drawImage(QRectF(0, 0, old_w, old_h), self._image)
            return
        transform = self._transform(preview_extent(plan))
        painter.setTransform(transform)
        new_path = _rect_path(0, 0, plan.new_w, plan.new_h)
        old_path = _rect_path(*old_rect(plan))
        painter.fillPath(new_path, QColor(*self.source.new_ground))
        painter.drawImage(old_path.boundingRect(), self._image)  # nearest sampling: smoothing stays off
        painter.fillPath(new_path.subtracted(old_path), _NEW_ONLY)
        painter.fillPath(old_path.subtracted(new_path), _OLD_ONLY)
        painter.setPen(_cosmetic_pen(_MARK, _MARK_PX))
        painter.drawPoints(self._marks)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_cosmetic_pen(_OUTLINE))
        painter.drawPath(new_path)
        dashed = _cosmetic_pen(_OUTLINE)
        dashed.setDashPattern(_OLD_MAP_DASH)
        painter.setPen(dashed)
        painter.drawPath(old_path)
        # In widget pixels, so the ring stays round on the diamond.
        centre = transform.map(QPointF(*anchor_point(plan)))
        painter.resetTransform()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(_cosmetic_pen(_RING, 2))
        painter.drawEllipse(centre, _RING_RADIUS, _RING_RADIUS)

    def _transform(self, extent) -> QTransform:
        return tile_transform(extent, self.source.isometric, self.width(), self.height())
