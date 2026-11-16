"""Trigger overlay rings that paint only what is on screen, at a detail the zoom can show (TASK-508).

Qt strokes a wide cosmetic pen's whole path before clipping, and a plain
QGraphicsPathItem's boundingRect() and shape() stroke it too, so a Sloped ring
union of tens of thousands of staircase points cost tens of ms per frame at
any zoom and about 100 ms in the first paint after every rebuild.

`RingSet` holds one overlay bucket's rings and hands paint the path to draw:

- **Levels.** Exact; for Sloped, the chord (each tile edge's two end points),
  off the exact ring by at most `deviation`; then Douglas-Peucker of the chord
  at each `RING_DP_TOLERANCES` step t, off by at most `deviation + t`. Every
  bound is point-to-segment both ways (a Hausdorff bound). Paint takes the
  coarsest level whose bound is within `RING_LOD_TOLERANCE_PX` device px at
  the transform's largest singular value; levels build on first use.
- **Culling by clamping.** A level's rings split into chunks of
  `RING_CHUNK_POINTS` points that overlap the next by one point, so every edge
  lies in one chunk's box. Against the paint rect (the visible scene rect
  grown by the pen plus 2 device px), a ring whose box misses is dropped, and
  every point belonging only to missing chunks is clamped onto the rect. Such
  a chunk stays in a closed half-plane outside the rect, so no clamped edge
  crosses its interior: winding (the fill) and every stroke within the
  margin are unchanged, pixel for pixel.
- **Memo.** The rect is snapped outward to a quarter of the visible rect's
  longer side; (level, snapped rect) returns the last path, so the fill and
  outline items sharing a set, hover repaints and small pans reuse it.

`RingPathItem` keeps `path()` as the exact union, returns precomputed bounds
and an unstroked `shape()`, and times its paint as Perf Trace's `rings`
repaint phase. A set lives exactly as long as its items: every rebuild makes
new ones, so nothing here is invalidated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QPainterPath, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsPathItem

from descape import perf_trace
from descape.qt_points import polygon_from_points, ring_path_from_points
from descape.viewer_canvas import _max_axis_scale

# Largest departure from the exact ring a coarser level may show, in device px (user decision, 2026-10-28).
RING_LOD_TOLERANCE_PX = 0.5
# Points per culling chunk: small enough to cull a mostly off-screen ring, large enough to keep numpy calls few.
RING_CHUNK_POINTS = 64
# Douglas-Peucker tolerances over the chord ring, in scene px, finest first.
RING_DP_TOLERANCES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


class RingGeometry(NamedTuple):
    """One ring: its exact float64 (N, 2) points (consecutive duplicates
    allowed), its chord points (the exact ones outside Sloped), and the
    largest distance from an exact point to its chord segment."""

    points: np.ndarray
    chord: np.ndarray
    deviation: float


def min_axis_scale(transform) -> float:
    """The smallest singular value of `transform`'s 2x2 linear part: the least
    any direction is stretched, so one device px covers up to 1/this scene px."""
    det = abs(transform.m11() * transform.m22() - transform.m12() * transform.m21())
    largest = _max_axis_scale(transform)
    return det / largest if largest > 0 else 0.0


def _split_deduped(points: np.ndarray, owner: np.ndarray, count: int) -> list[np.ndarray]:
    """`points` (each row owned by ring `owner`, ascending) split into `count`
    rings, consecutive exact duplicates within a ring dropped, as
    qt_points.ring_path_from_points does."""
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = (owner[1:] != owner[:-1]) | np.any(points[1:] != points[:-1], axis=1)
    points, owner = points[keep], owner[keep]
    return np.split(points, np.searchsorted(owner, np.arange(1, count)))


def _dedupe_rings(rings: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Each float64 ring with its consecutive exact duplicates dropped, in one numpy pass."""
    rings = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in rings]
    if not rings:
        return []
    sizes = [len(r) for r in rings]
    return _split_deduped(np.vstack(rings), np.repeat(np.arange(len(rings)), sizes), len(rings))


def _closed_path(points: np.ndarray) -> QPainterPath:
    path = QPainterPath()
    if len(points):
        path.addPolygon(polygon_from_points(points))
        path.closeSubpath()
    return path


def simplify_rings(rings: Sequence[np.ndarray], tolerance: float) -> list[np.ndarray]:
    """Douglas-Peucker over each closed ring, from and back to its first point,
    which it keeps. Distances are to the segment, never its line, so a
    hairpin past a segment's end counts at full length. Every open span of
    every ring splits in one numpy pass per recursion depth. A ring always
    keeps its farthest point, so a sub-pixel ring still draws as a dot."""
    closed = [np.vstack([r, r[:1]]) for r in rings if len(r)]
    if not closed:
        return list(rings)
    sizes = np.array([len(c) for c in closed])
    starts = np.cumsum(sizes) - sizes
    points = np.vstack(closed)
    keep = np.zeros(len(points), dtype=bool)
    keep[starts] = keep[starts + sizes - 1] = True
    a, b = starts, starts + sizes - 1
    bar = 0.0  # the first pass splits every ring at its farthest point
    while True:
        inner = b - a - 1
        open_ = inner > 0
        a, b, inner = a[open_], b[open_], inner[open_]
        if not len(a):
            break
        span = np.repeat(np.arange(len(a)), inner)
        first = np.cumsum(inner) - inner
        index = a[span] + 1 + (np.arange(len(span)) - first[span])
        d = (points[b] - points[a])[span]
        rel = points[index] - points[a][span]
        length_sq = np.einsum("ij,ij->i", d, d)
        u = np.clip(np.einsum("ij,ij->i", rel, d) / np.where(length_sq > 0, length_sq, 1.0), 0.0, 1.0)
        dist = np.hypot(rel[:, 0] - u * d[:, 0], rel[:, 1] - u * d[:, 1])
        peak = np.maximum.reduceat(dist, first)
        # The first index at each span's peak, as np.argmax picks.
        at_peak = np.flatnonzero(dist == peak[span])
        _spans, firsts = np.unique(span[at_peak], return_index=True)
        k = index[at_peak[firsts]]
        split = peak > bar
        bar = tolerance
        keep[k[split]] = True
        a, b = np.concatenate([a[split], k[split]]), np.concatenate([k[split], b[split]])
    keep[starts + sizes - 1] = False  # each ring's closing copy of its first point
    owner = np.repeat(np.arange(len(closed)), sizes)
    out = iter(_split_deduped(points[keep], owner[keep], len(closed)))
    return [next(out) if len(r) else r for r in rings]


class _Level:
    """One representation of a set's rings, chunked for culling."""

    __slots__ = ("boxes", "path", "ring_boxes", "ring_paths", "rings", "spans")

    def __init__(self, rings: list[np.ndarray], ring_paths: list[QPainterPath] | None = None) -> None:
        self.rings = rings
        self.ring_paths = ring_paths if ring_paths is not None else [_closed_path(r) for r in rings]
        self.path = QPainterPath()
        for ring_path in self.ring_paths:
            self.path.addPath(ring_path)
        self.path.setFillRule(Qt.WindingFill)
        # Every ring's chunks in one numpy pass: a per-ring loop cost 5 ms per level on a large union.
        k = RING_CHUNK_POINTS
        sizes = np.array([len(r) for r in rings], dtype=np.int64)
        counts = (sizes + k - 1) // k
        ends = np.cumsum(counts)
        self.spans = list(zip((ends - counts).tolist(), ends.tolist(), strict=True))
        self.ring_boxes = np.tile([np.inf, np.inf, -np.inf, -np.inf], (len(rings), 1))
        self.boxes = np.zeros((0, 4))
        if counts.sum():
            points = np.vstack([r for r in rings if len(r)])
            offsets = np.cumsum(sizes) - sizes
            owner = np.repeat(np.arange(len(rings)), counts)
            local = (np.arange(len(owner)) - (ends - counts)[owner]) * k
            starts = offsets[owner] + local
            # Each chunk runs through the next chunk's first point (the last back to point 0).
            after = points[offsets[owner] + np.where(local + k < sizes[owner], local + k, 0)]
            lo = np.minimum(np.minimum.reduceat(points, starts, axis=0), after)
            hi = np.maximum(np.maximum.reduceat(points, starts, axis=0), after)
            self.boxes = np.hstack([lo, hi])
            filled = np.flatnonzero(sizes)
            self.ring_boxes[filled, :2] = np.minimum.reduceat(points, offsets[filled], axis=0)
            self.ring_boxes[filled, 2:] = np.maximum.reduceat(points, offsets[filled], axis=0)

    def clamped(self, x0: float, y0: float, x1: float, y1: float) -> QPainterPath:
        b = self.boxes
        visible = (b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1)
        if visible.all():
            return self.path
        rb = self.ring_boxes
        hit = (rb[:, 2] >= x0) & (rb[:, 0] <= x1) & (rb[:, 3] >= y0) & (rb[:, 1] <= y1)
        path = QPainterPath()
        k = RING_CHUNK_POINTS
        for r in np.flatnonzero(hit):
            lo, hi = self.spans[r]
            chunks = visible[lo:hi]
            if chunks.all():
                path.addPath(self.ring_paths[r])
                continue
            ring = self.rings[r]
            index = np.arange(len(ring))
            keep = chunks[index // k]
            # A chunk's first point is also the previous chunk's last; point 0 closes the last chunk.
            firsts = index[::k]
            keep[firsts] |= chunks[(firsts // k - 1) % len(chunks)]
            out = ring.copy()
            out[~keep, 0] = np.clip(out[~keep, 0], x0, x1)
            out[~keep, 1] = np.clip(out[~keep, 1], y0, y1)
            path.addPath(ring_path_from_points(out))
        path.setFillRule(Qt.WindingFill)
        return path


class RingSet:
    """One overlay bucket's rings (module docstring), shared by its fill and outline items."""

    def __init__(
        self, geometries: Sequence[RingGeometry], margin_px: float, exact_paths: Sequence[QPainterPath] | None = None
    ) -> None:
        self._geometries = list(geometries)
        self.margin_px = margin_px
        exact = _dedupe_rings([g.points for g in self._geometries])
        paths = list(exact_paths) if exact_paths is not None else [ring_path_from_points(r) for r in exact]
        self._levels: dict[int, _Level] = {0: _Level(exact, paths)}
        self.deviation = max((g.deviation for g in self._geometries), default=0.0)
        # (level index, error in scene px); the chord level only where it differs from exact.
        errors = [(0, 0.0)]
        if self.deviation > 0:
            errors.append((1, self.deviation))
        errors += [(2 + i, self.deviation + t) for i, t in enumerate(RING_DP_TOLERANCES)]
        self.errors = errors
        points = [r for r in exact if len(r)]
        if points:
            stacked = np.vstack(points)
            (x0, y0), (x1, y1) = stacked.min(axis=0), stacked.max(axis=0)
            self.bounds = QRectF(x0, y0, x1 - x0, y1 - y0)
        else:
            self.bounds = QRectF()
        self._chord: list[np.ndarray] | None = None
        self._memo: tuple[tuple, QPainterPath] | None = None
        self.memo_hits = 0
        self.memo_misses = 0

    @property
    def exact_path(self) -> QPainterPath:
        return self._levels[0].path

    def level_index_for(self, scale: float) -> int:
        """The coarsest level whose error is within RING_LOD_TOLERANCE_PX at `scale` device px per scene px."""
        index = 0
        for level, error in self.errors:
            if error * scale <= RING_LOD_TOLERANCE_PX:
                index = level
        return index

    def level_rings(self, index: int) -> list[np.ndarray]:
        return self._level(index).rings

    def _level(self, index: int) -> _Level:
        level = self._levels.get(index)
        if level is None:
            if self._chord is None:
                self._chord = _dedupe_rings([g.chord for g in self._geometries])
            if index == 1:
                level = _Level(self._chord)
            else:
                tolerance = RING_DP_TOLERANCES[index - 2]
                level = _Level(simplify_rings(self._chord, tolerance))
            self._levels[index] = level
        return level

    def paint_key(self, world, exposed: QRectF, widget=None) -> tuple[int, tuple[float, float, float, float] | None]:
        """(level index, snapped clamp rect x0, y0, x1, y1) for a paint, the
        rect None when the transform is singular."""
        index = self.level_index_for(_max_axis_scale(world))
        s_min = min_axis_scale(world)
        if s_min <= 0:
            return index, None
        basis = exposed
        if widget is not None:
            inverse, ok = world.inverted()
            if ok:
                view = inverse.mapRect(QRectF(widget.rect()))
                if view.contains(exposed):
                    basis = view
        grow = self.margin_px / s_min
        # A power of two near a quarter of the longer side, so a small zoom can keep the same cells.
        cell = 2.0 ** round(np.log2(max(basis.width(), basis.height(), 1e-6) / 4.0))
        rect = (
            float(np.floor((basis.left() - grow) / cell) * cell),
            float(np.floor((basis.top() - grow) / cell) * cell),
            float(np.ceil((basis.right() + grow) / cell) * cell),
            float(np.ceil((basis.bottom() + grow) / cell) * cell),
        )
        return index, rect

    def path_for(self, world, exposed: QRectF, widget=None) -> QPainterPath:
        """The path to draw under painter world transform `world` for item-space `exposed`."""
        key = self.paint_key(world, exposed, widget)
        index, rect = key
        if rect is None:
            return self._level(index).path
        if self._memo is not None and self._memo[0] == key:
            self.memo_hits += 1
            return self._memo[1]
        self.memo_misses += 1
        path = self._level(index).clamped(*rect)
        self._memo = (key, path)
        return path


class RingPathItem(QGraphicsPathItem):
    """A ring bucket's fill or outline, drawn through its RingSet (module docstring)."""

    def __init__(self, ring_set: RingSet, pen: QPen, brush: QBrush) -> None:
        super().__init__()
        self._rings = ring_set
        # Assigned before setPath() and addItem(); a later change must call prepareGeometryChange() first.
        width = 0.0 if pen.style() == Qt.NoPen else max(pen.widthF(), 1.0) * max(1.0, pen.miterLimit() / 2.0)
        self._bounds = ring_set.bounds.adjusted(-width, -width, width, width)
        self.setPath(ring_set.exact_path)
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def ring_set(self) -> RingSet:
        return self._rings

    def boundingRect(self) -> QRectF:
        return self._bounds

    def shape(self) -> QPainterPath:
        return self._rings.exact_path

    def paint(self, painter, option, widget=None) -> None:
        with perf_trace.phase("repaint"), perf_trace.repaint_phase("rings"):
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            # combinedTransform() includes any window/viewport mapping (a print), so the scales stay true.
            painter.drawPath(self._rings.path_for(painter.combinedTransform(), option.exposedRect, widget))
