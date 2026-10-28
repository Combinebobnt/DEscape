"""Pre-rendered canvas frames for the next zoom notch (TASK-031.71).

After a zoom settles, PreframeWarmer renders the canvas as it will look one
notch in and one notch out, in idle ticks, through the same block loop
MapCanvasItem.paint() runs. When the next notch's paint has exactly the key a
frame was rendered under, paint() copies the frame 1:1 instead of fetching,
converting and smooth-drawing its blocks. Anything else is a miss and paints
as before, so a wrong prediction costs speed, never pixels.

The key is exact, not approximate, because Qt's smooth drawImage() is not
shift-invariant: the same image drawn under a device transform translated by
a whole pixel differs by 1-2 levels in about half the cases measured. A frame
is byte-identical only under the exact device transform it was drawn with,
which includes the viewport's offset in its window and the device pixel ratio.

The predicted transform comes from a twin: a hidden QGraphicsView with an
empty scene, synced to the real view's scene rect, size, alignment, scrollbar
policies, transform and scrollbar values, which then runs the same
zoom_about() wheelEvent runs. Qt's own scroll and indent math produces the
prediction, so there is no copy of it here to drift."""

from __future__ import annotations

import os
import time

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PyQt5.QtGui import QImage, QPainter, QTransform
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView

from descape import debug_log, gc_hold, perf_trace
from descape.level_warm import _IdleTimerDriver

# Device rows one strip tick draws; Step 1c of the plan sized it.
STRIP_ROWS = 256
# Quiet time after the last zoom, scroll, resize or (zoom on cursor) cursor move before
# frames render: WHEEL_GESTURE_GAP_S, so the predicted notch starts a fresh gesture.
SETTLE_MS = 250
# DESCAPE_PREFRAME_CHECK=1: every hit renders a frame from the live paint's state and the cache
# as it is now, recording any byte difference in check_failures (the suite turns it on, conftest.py).
CHECK = os.environ.get("DESCAPE_PREFRAME_CHECK") == "1"
check_failures: list[str] = []


class Frame:
    """A finished pre-rendered frame: the viewport's device pixels as the
    canvas item would paint them under `key`, transparent where it paints
    nothing (an RGB32 image where it covers the whole viewport). It serves
    only while the cache still holds level `mip`'s chunks with `stamps`
    (render_cache chunk_stamps())."""

    __slots__ = ("blit_format", "image", "key", "mip", "reduction", "stamps")

    def __init__(self, key: tuple, image: QImage, blit_format, reduction: int, mip=None, stamps=None) -> None:
        self.key = key
        self.image = image
        # What paint()'s normal path leaves in _last_blit_format and _last_reduction.
        self.blit_format = blit_format
        self.reduction = reduction
        self.mip = mip
        self.stamps = stamps


def paint_key(
    device: QTransform, rect: QRectF, viewport: QSize, offset: QPoint, hints, dpr: float, bounding: QRectF
) -> tuple:
    """Everything a live canvas paint's pixels depend on besides the cache's
    content: the device transform's nine terms, the exposed scene rect, the
    viewport's size and its offset in its window, the render hints, the
    device pixel ratio and the item's bounds. Compared exactly; a hit draws
    nothing paint() would not. The offset is there for blit_frame(), which
    draws at the live offset: an offset change cancelled by a scroll keeps
    the device transform, and with the whole map in view the rect too."""
    return (
        transform_terms(device),
        (rect.x(), rect.y(), rect.width(), rect.height()),
        (viewport.width(), viewport.height()),
        (offset.x(), offset.y()),
        int(hints),
        dpr,
        (bounding.width(), bounding.height()),
    )


def viewport_offset(viewport) -> QPoint:
    """The viewport's offset in its window, in logical pixels."""
    return viewport.mapTo(viewport.window(), QPoint())


def blit_frame(painter: QPainter, frame: Frame) -> None:
    """Copies `frame` 1:1 onto the viewport under `painter`: with the world
    transform reset, logical (0, 0) is the viewport's device origin, and the
    image's device pixel ratio makes one image pixel one device pixel."""
    painter.save()
    painter.resetTransform()
    painter.drawImage(QPointF(0, 0), frame.image)
    painter.restore()


class FrameJob:
    """Renders one frame in steps: each step() fetches one block (through the
    item's own _level_blocks(), so with exactly the paint's block rects,
    conversion and clip) or, once every block is held, draws one STRIP_ROWS strip of
    them, clipped. The blocks are fetched once for the whole frame."""

    def __init__(
        self, item, key: tuple, device: QTransform, rect: QRectF, plan, viewport_device: QRect, hints, dpr: float,
        stamps=None,
    ) -> None:
        mip, k, _level_rect = plan
        self.key = key
        self.mip = mip
        self.stamps = stamps
        self._device = device
        self._hints = hints
        self._dpr = dpr
        self._reduction = k
        self._viewport = viewport_device
        self._blocks_iter = item._level_blocks(device, mip, rect, k)
        self._blocks: list = []
        self._fetched = False
        # Device pixels from (0, 0): the device transform includes the viewport's offset in its window.
        self._image = QImage(viewport_device.right() + 1, viewport_device.bottom() + 1, QImage.Format_ARGB32_Premultiplied)
        self._image.fill(Qt.transparent)
        self._next_row = viewport_device.top()
        self.frame: Frame | None = None

    @property
    def done(self) -> bool:
        return self.frame is not None

    @property
    def next_step(self) -> str:
        """What step() does next, for Perf Trace: `fetch`, `strip` or `finish`."""
        if not self._fetched:
            return "fetch"
        return "strip" if self._next_row <= self._viewport.bottom() else "finish"

    def step(self) -> bool:
        """One unit of work; False once the frame is finished."""
        if self.frame is not None:
            return False
        if not self._fetched:
            block = next(self._blocks_iter, None)
            if block is None:
                self._fetched = True
            else:
                self._blocks.append(block)
            return True
        vp = self._viewport
        if self._next_row <= vp.bottom():
            rows = min(STRIP_ROWS, vp.bottom() + 1 - self._next_row)
            self.draw_strip(QRect(vp.left(), self._next_row, vp.width(), rows))
            self._next_row += rows
            return True
        self._finish()
        return False

    def draw_strip(self, clip: QRect) -> None:
        painter = QPainter(self._image)
        try:
            painter.setRenderHints(self._hints)
            # Clipped before the transform is set, so the clip is in device pixels.
            painter.setClipRect(clip)
            painter.setWorldTransform(self._device)
            for draw in self._blocks:
                draw(painter)
        finally:
            painter.end()

    def _finish(self) -> None:
        image = self._image.copy(self._viewport)
        self._image = None
        bits = image.constBits()
        bits.setsize(image.byteCount())
        alpha = np.frombuffer(bits, np.uint8)[3::4]
        if alpha.min() == 255:
            # Premultiplied ARGB with every alpha 255 is RGB32 bit for bit: drawn as a plain copy.
            image.reinterpretAsFormat(QImage.Format_RGB32)
        image.setDevicePixelRatio(self._dpr)
        blit_format = self._blocks[-1].format if self._blocks else None
        self._blocks = []
        self.frame = Frame(self.key, image, blit_format, self._reduction, self.mip, self.stamps)

    def run_to_completion(self) -> Frame:
        while self.step():
            pass
        return self.frame


def scroll_view_by(view: QGraphicsView, dx: int, dy: int) -> None:
    """Moves `view`'s scrollbars by (dx, dy): MapView._scroll_by()'s body,
    shared with the twin so a prediction scrolls the way a real zoom does."""
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + dx)
    view.verticalScrollBar().setValue(view.verticalScrollBar().value() + dy)


def zoom_about(view: QGraphicsView, factor: float, anchor: QPoint) -> None:
    """Scales `view` by `factor` keeping the scene point under the viewport
    point `anchor` where it was: the wheel zoom's transform change, run on the
    real view by MapView.wheelEvent and on the twin by ZoomTwin.predict()."""
    before = view.mapToScene(anchor)
    view.scale(factor, factor)
    drift = view.mapFromScene(before) - anchor
    scroll_view_by(view, drift.x(), drift.y())


def device_transform(world: QTransform, offset: QPoint, dpr: float) -> QTransform:
    """The painter device transform a paint of the viewport gets under
    `world`, composed in QPainter's own order (world, then the widget's
    redirection offset in its window, then the HiDPI scale), with the same
    QTransform operators, so the terms come out bit-equal."""
    t = QTransform(world)
    t *= QTransform.fromTranslate(offset.x(), offset.y())
    if dpr != 1.0:
        t *= QTransform.fromScale(dpr, dpr)
    return t


def exposed_scene_rect(world: QTransform, viewport_rect: QRect, bounding: QRectF) -> QRectF:
    """The option.exposedRect Qt hands an item with ItemUsesExtendedStyleOption
    on a full-viewport paint under `world`: the viewport rect mapped back
    through the inverted world transform, intersected with the item's bounds
    (QGraphicsItemPrivate::initStyleOption())."""
    inverse, _ok = world.inverted()
    rect = QRectF()
    rect |= inverse.mapRect(QRectF(viewport_rect))
    rect &= bounding
    return rect


def transform_terms(t: QTransform) -> tuple[float, ...]:
    """All nine terms of `t`, for an exact comparison."""
    return (t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), t.m31(), t.m32(), t.m33())


class ZoomTwin:
    """A hidden copy of a QGraphicsView's geometry, for predicting the
    viewport transform a zoom will produce without touching the real view.

    A child of the real view, so a style sheet that changes the scrollbar
    extent reaches it too. Never shown: QAbstractScrollArea computes its
    scroll ranges and indents from size(), which a hidden widget's resize()
    updates at once."""

    def __init__(self, real: QGraphicsView) -> None:
        self._real = real
        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene, real)
        self._view.hide()

    def sync(self) -> None:
        real, twin = self._real, self._view
        twin.setFrameStyle(real.frameStyle())
        twin.setLineWidth(real.lineWidth())
        twin.setHorizontalScrollBarPolicy(real.horizontalScrollBarPolicy())
        twin.setVerticalScrollBarPolicy(real.verticalScrollBarPolicy())
        twin.setAlignment(real.alignment())
        twin.setTransformationAnchor(QGraphicsView.NoAnchor)
        twin.setResizeAnchor(QGraphicsView.NoAnchor)
        twin.setLayoutDirection(real.layoutDirection())
        margins = real.viewportMargins()
        twin.setViewportMargins(margins.left(), margins.top(), margins.right(), margins.bottom())
        twin.resize(real.size())
        twin.setSceneRect(real.sceneRect())
        # A hidden widget gets no resizeEvent, so a transform change is what
        # recomputes its ranges and indents; reset first so one always happens.
        twin.resetTransform()
        twin.setTransform(real.transform())
        twin.horizontalScrollBar().setValue(real.horizontalScrollBar().value())
        twin.verticalScrollBar().setValue(real.verticalScrollBar().value())

    def predict(self, factor: float, anchor: QPoint) -> tuple[QTransform, QRect]:
        """(viewportTransform(), viewport rect) the real view will have after
        zoom_about(real, factor, anchor), computed on the twin. The real view
        is not touched."""
        self.sync()
        zoom_about(self._view, factor, anchor)
        return self._view.viewportTransform(), self._viewport_rect()

    def _viewport_rect(self) -> QRect:
        """The real viewport's rect once its as-needed scrollbars follow the
        twin's new ranges: a zoom can show or hide one, which resizes the
        viewport by its extent (QAbstractScrollArea's layoutChildren(), run
        from a queued range-change slot, so after the zoom returns)."""
        real, twin = self._real, self._view
        size = real.maximumViewportSize()
        w, h = size.width(), size.height()
        # maximumViewportSize() already subtracts an always-on bar; only an as-needed one moves.
        if self._as_needed_shown(real.horizontalScrollBarPolicy(), twin.horizontalScrollBar()):
            h -= real.horizontalScrollBar().sizeHint().height()
        if self._as_needed_shown(real.verticalScrollBarPolicy(), twin.verticalScrollBar()):
            w -= real.verticalScrollBar().sizeHint().width()
        return QRect(0, 0, w, h)

    @staticmethod
    def _as_needed_shown(policy, bar) -> bool:
        return policy == Qt.ScrollBarAsNeeded and bar.minimum() < bar.maximum()

    def synced_transform(self) -> QTransform:
        """The twin's viewportTransform() right after sync(): equal to the
        real view's when the twin tracks it (the probe's sanity check)."""
        self.sync()
        return self._view.viewportTransform()

    def map_to_scene(self, point: QPoint) -> QPointF:
        return self._view.mapToScene(point)


def viewport_device_rect(viewport, size: QSize) -> QRect:
    """The device rect a viewport of `size` covers in its window's backing
    store: its offset in the window and its size, both times the DPR."""
    dpr = viewport.devicePixelRatioF()
    offset = viewport_offset(viewport)
    return QRect(round(offset.x() * dpr), round(offset.y() * dpr), round(size.width() * dpr), round(size.height() * dpr))


def device_aligned(offset: QPoint, size: QSize, dpr: float) -> bool:
    """Whether a viewport at `offset` in its window with `size` lands on whole
    device pixels at `dpr`. blit_frame() draws at logical (0, 0), device
    offset * dpr, so at a fractional DPR (1.25, 1.5) an odd offset would put
    the 1:1 copy between pixels: no frame is rendered there."""
    return all(float(v * dpr).is_integer() for v in (offset.x(), offset.y(), size.width(), size.height()))


def check_hit(item, frame: Frame, device: QTransform, rect: QRectF, viewport_device: QRect, hints, dpr: float) -> None:
    """The check mode's test of one hit: a frame rendered now from the live
    paint's own transform, rect, viewport, hints and DPR, over the cache as it
    is, must match `frame` byte for byte. A mismatch means an edit changed
    pixels without re-stamping or removing a chunk, or the key leaves out
    something the paint depends on. Logged and recorded in check_failures
    rather than raised, since paint() runs inside Qt."""
    plan = item.frame_plan(device, rect)
    same = plan is not None
    if same:
        fresh = FrameJob(item, frame.key, device, rect, plan, viewport_device, hints, dpr).run_to_completion().image
        img = frame.image
        same = img.format() == fresh.format() and img.size() == fresh.size()
    if same:
        a, b = img.constBits(), fresh.constBits()
        a.setsize(img.byteCount())
        b.setsize(fresh.byteCount())
        same = bytes(a) == bytes(b)
    if not same:
        message = f"preframe check: level {frame.mip} frame differs from a render of the live paint"
        check_failures.append(message)
        debug_log.log(message)


class PreframeWarmer(_IdleTimerDriver):
    """Renders the frames for the next zoom notch in idle ticks: one
    `make_job(direction)` per queued direction (+1 in, -1 out), then one
    FrameJob step per tick, each frame handed to the canvas item once done.
    A job is dropped once its chunks' stamps go stale (an edit landed), so a
    tick never composites a chunk, and the frame is handed over only while
    they are current. Pauses while a mouse button is held, as the warms do.
    A direction make_job refused (its chunks not cached yet, say) waits in
    `refused` for retry_refused()."""

    def __init__(self) -> None:
        super().__init__()
        self._item = None
        self._make_job = None
        self._pending: list[int] = []
        self._job: FrameJob | None = None
        self.refused: list[int] = []

    @property
    def is_active(self) -> bool:
        return self._job is not None or bool(self._pending)

    def start(self, item, directions, make_job) -> None:
        self.cancel()
        self._item = item
        self._make_job = make_job
        self._pending = list(directions)
        self._schedule()

    def retry_refused(self) -> bool:
        """Queues the refused directions again, behind any still pending;
        True if there were any."""
        if not self.refused or self._item is None:
            return False
        self._pending += [d for d in self.refused if d not in self._pending]
        self.refused = []
        self._schedule()
        return True

    def cancel(self) -> None:
        """Drops the queue, the job and the item: a cancelled warmer holds no
        canvas item, so a set_source() can release the old one."""
        was_active = self.is_active
        self._pending = []
        self._job = None
        self.refused = []
        self._item = None
        self._make_job = None
        self._stop_timer()
        if was_active:
            gc_hold.idle_soon()

    def tick(self) -> bool:
        if self._held_backoff() or self._overdue_collect():
            return True
        t0 = time.perf_counter()
        cache = self._item._cache if self._item is not None else None
        job = self._job
        if job is None:
            if not self._pending:
                self._stop_timer()
                gc_hold.idle_soon()
                return False
            direction = self._pending.pop(0)
            self._job = self._make_job(direction)
            if self._job is None:
                self.refused.append(direction)
            perf_trace.preframe_tick((time.perf_counter() - t0) * 1000, "predict")
            return True
        if not cache.stamps_current(job.stamps):
            self._job = None
            return True
        kind = job.next_step
        job.step()
        if job.done:
            self._job = None
            self._item.add_preframe(job.frame)
        perf_trace.preframe_tick((time.perf_counter() - t0) * 1000, kind)
        return True
