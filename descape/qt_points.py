"""numpy (N, 2) float64 point arrays to and from Qt polygons and paths, in one
buffer copy each way (TASK-355). numpy and PyQt5 only, so any module may import it."""

from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QPainterPath, QPolygonF


def polygon_from_points(points: np.ndarray) -> QPolygonF:
    """A QPolygonF holding a float64 (N, 2) array's rows, written in one copy
    through the polygon's own buffer (the sip voidptr route zoom_preframe.py
    uses for QImage bits). Relies on qreal being double, as on every desktop
    PyQt5 build."""
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = QPolygonF(len(points))
    if len(points):
        buffer = polygon.data()
        buffer.setsize(points.nbytes)
        np.frombuffer(buffer, dtype=np.float64).reshape(-1, 2)[:] = points
    return polygon


def points_from_polygon(polygon: QPolygonF) -> np.ndarray:
    """The reverse of polygon_from_points: a float64 (N, 2) copy of the points."""
    count = polygon.size()
    if not count:
        return np.zeros((0, 2), dtype=np.float64)
    buffer = polygon.data()
    buffer.setsize(count * 2 * np.dtype(np.float64).itemsize)
    return np.frombuffer(buffer, dtype=np.float64).reshape(count, 2).copy()


def ring_path_from_points(points: np.ndarray) -> QPainterPath:
    """One closed subpath through `points`, consecutive duplicates dropped as
    the old per-point loop's currentPosition() check did. No points, empty path."""
    path = QPainterPath()
    if not len(points):
        return path
    # One complex per row: an exact (x, y) compare in a single ufunc call.
    rows = np.ascontiguousarray(points, dtype=np.float64).view(np.complex128).ravel()
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = rows[1:] != rows[:-1]
    path.addPolygon(polygon_from_points(points[keep]))
    path.closeSubpath()
    return path
