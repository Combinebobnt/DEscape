"""The QGraphicsItems the map scene is drawn from, plus the QTransform
readers they and MapView size themselves by.

MapCanvasItem paints the composited map; EdgeTickItem paints the
distance-tick strip outside its border. Both are driven by MapView
(descape.map_view), which imports this module one way."""

from __future__ import annotations


import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QTransform,
)
from PyQt5.QtWidgets import (
    QGraphicsItem,
)


from descape import (
    edge_ticks,
    iso_geometry,
    perf_trace,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)


def _max_axis_scale(transform) -> float:
    """The largest singular value of `transform`'s 2x2 linear part -- i.e.
    the greatest factor by which it stretches ANY direction, which is the
    scale a mip level has to keep up with. Phase B-D-c's LOD metric.

    Deliberately NOT QStyleOptionGraphicsItem.levelOfDetailFromTransform():
    measured on Flat's own scale(1, 0.5) + rotate(-45) at 2x zoom, that
    returns 1.5811 where the true max stretch is 2.0 -- under the
    floor(log2(...)) rule this project used until 2026-08-28, that was 0
    vs 1, a different level, one octave of blur along Flat's stretched
    axis. Both numbers are reproduced as an oracle in
    tests/test_mip_viewer.py, alongside a second transform that still
    straddles a level boundary under the current ceil(log2(...)) rule (this
    specific 2x-zoom one no longer does: both readings now land on level 1).

    Qt's m12/m21 naming can't silently transpose this: singular values are
    invariant under transpose, so the metric is identical either way (also
    pinned in that test file)."""
    a, b = transform.m11(), transform.m21()
    c, d = transform.m12(), transform.m22()
    half_sum_sq = (a * a + b * b + c * c + d * d) / 2.0
    det = a * d - b * c
    # max(0, ...) only guards float error at the equal-singular-value
    # (pure uniform scale/rotation) case, where the discriminant is 0.
    disc = math.sqrt(max(0.0, half_sum_sq * half_sum_sq - det * det))
    return math.sqrt(half_sum_sq + disc)


class MapCanvasItem(QGraphicsItem):
    """Qt-side wrapper around a descape.render chunk cache -- Phase B-C of
    Track B, replacing a mode's single monolithic
    QPixmap/array (the whole canvas rendered up front, ~1-2s and hundreds of
    MB for this project's bigger real maps) with a QGraphicsItem whose
    paint() composites only the chunks its own exposedRect actually needs,
    via the Qt-free chunk cache built in Phase B-B. Introduced Stepped-only
    (IsoChunkCache); Phase B-E gives Flat mode the same treatment
    (FlatChunkCache) via this SAME class, unchanged -- its paint() has
    nothing style-specific in it (both caches expose the same canvas_dims()/
    render_rect() contract, see descape.render_cache._ChunkCacheBase), only which
    cache MapView.set_source() constructs differs.

    ItemUsesExtendedStyleOption is the load-bearing flag here: without it
    Qt always reports the item's FULL boundingRect as "exposed" regardless
    of what's actually visible, which would make paint() composite every
    chunk on every repaint -- silently degrading back to full-canvas cost
    with none of the memory/latency upside a chunk cache exists for. See
    tests/test_lazy_viewport.py for this specifically (option.exposedRect
    must come back smaller than boundingRect() under a real Qt paint
    cycle, not just in principle)."""

    # Phase B-E: painting a padded superset per block, tiled at PAINT_BLOCK_PX,
    # rather than one drawImage() over the whole exposedRect -- see paint()'s
    # own comment for why. Both apply identically to Stepped and Flat (this
    # class has nothing style-specific in it -- see the class docstring), and
    # land AFTER Flat was wired to this class (Phase B-E's earlier steps) so a
    # later human-reported Stepped visual artifact still bisects to either
    # Phase B-C or this specific change, not ambiguously to "B-E" as a whole.
    #
    # PAINT_PAD_PX: under Flat's rotated view transform (scale(1,0.5) then
    # rotate(-45), see MapView.set_isometric()), option.exposedRect is the
    # axis-aligned bbox of a ROTATED region, so drawImage()'s destination
    # rect is sampled by QPainter.SmoothPixmapTransform under that rotation
    # -- texels just outside a bare exposedRect-sized block don't exist,
    # which can show as a 1px seam at block edges. Compositing a small
    # padded superset instead gives the filter real neighbor pixels on both
    # sides. Under an identity/axis-aligned transform (Stepped, or Flat
    # unrotated) this padding is a no-op superset -- correctness is
    # unaffected either way, only clipped to canvas bounds like the
    # unpadded block always was.
    #
    # Phase B-D-d note: 2px was justified above (and in PLAN_MIPS.md) partly
    # by "residual magnification in [1, 2) means bilinear reaches at most 1
    # source texel past a block edge", which mip-down's clamp at the coarsest
    # level makes no longer universally true -- below that level the residual
    # is a MINIFICATION. Measured across residuals 0.3/0.5/0.8/0.95 (see
    # tests/test_mip_viewer.py's minifying-band A/B), tiled and single-draw
    # renders still come back byte-identical, so 2px stands unchanged for the
    # minifying band too: bilinear samples a 2x2 texel neighbourhood whatever
    # the ratio, so its reach past a block edge is set by the filter, not by
    # the scale factor.
    #
    # 2026-08-28: mip_for_scale's rule flipped from floor(log2(scale)) to
    # ceil, so the UNCLAMPED band is now (0.5, 1.0] rather than [1, 2) --
    # entirely inside the residuals already measured above, so this filter-
    # reach argument (not the old [1,2) one) was always the real reason 2px
    # is enough, and no value here needs to change.
    #
    # PAINT_BLOCK_PX: at fit-to-view (the whole map exposed at once, e.g.
    # right after opening a file), a single render_rect() call over the
    # entire exposedRect would allocate one full-canvas-sized stitched
    # scratch array (675 MiB on this project's biggest real map) ON TOP OF
    # the persistent warm cache -- a real transient memory regression
    # against the old single-buffer approach this phase is supposed to
    # avoid reintroducing. Tiling a large exposedRect into PAINT_BLOCK_PX
    # sub-blocks (each independently padded and drawn) bounds that scratch
    # to a small, fixed size regardless of how much of the canvas is
    # exposed at once.
    PAINT_PAD_PX = 2
    PAINT_BLOCK_PX = 2048

    def __init__(self, cache: "IsoChunkCache | FlatChunkCache | SlopedChunkCache"):
        super().__init__()
        self._cache = cache
        # Scene space is pinned to the REFERENCE level permanently (plan
        # decision D2), so boundingRect never follows the selected mip --
        # that's what leaves _pick_tile/_pos_on_map/the overscroll scene
        # rect/set_isometric/wheelEvent untouched by this whole track.
        canvas_w, canvas_h = cache.canvas_dims()
        self._bounding_rect = QRectF(0, 0, canvas_w, canvas_h)
        # Write-only, for tests/debugging to see what paint() selected.
        # paint() never reads it back: selection stays a pure function of
        # the painter transform (see _select_mip). It records the LAST
        # paint only, so a reader treating it as "the level this render
        # chose" has to have driven exactly one paint (Phase B-D-d's tests
        # do, via a single scene.render() -- worth keeping in mind before
        # reusing it after a repaint that Qt may have split).
        self._last_mip = 0
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def _select_mip(self, painter: QPainter) -> int:
        """The mip level to paint from, as a pure function of `painter`'s
        transform -- no hysteresis, no stored state. Same view state must
        always yield the same pixels, or every scene-render byte-identity
        check in this track becomes history-dependent.

        Reads deviceTransform(), not worldTransform(): measured, at
        QT_SCALE_FACTOR=2 the device transform's m11 is 6.0 where world's
        is 3.0, so selecting from device gets HiDPI right for free. The
        blit itself stays in logical coordinates, so DPR never enters the
        destination-rect math.

        The rule itself (clamp(ceil(log2(scale)), coarsest, finest), plus
        the degenerate scale <= 0 guard) lives on the cache as
        mip_for_scale() and is deliberately not restated here -- Phase
        B-D-c carried a local copy only because it clamped the floor at the
        reference level; Phase B-D-d, which extends selection below the
        reference, has nothing left to add to the cache's own rule.

        ceil(log2(scale)) is derived, not tuned (2026-08-28: was floor,
        see git history for the [1, 2)-residual rule this replaced): one
        level-L pixel covers mip_scale(L) * scale device pixels, and
        requiring that be <= 1 (never magnify) is exactly L >= log2(scale).
        Residual magnification then lands in (0.5, 1.0] whenever the
        derived level is actually available -- below the coarsest
        enumerated level the clamp binds instead and the residual becomes
        a minification, bounded only by how far MapView's zoom-out floor
        lets the view go; above the finest enumerated level the clamp
        binds the other way and the residual becomes a magnification,
        bounded only by how far MapView's zoom-in ceiling lets the view go
        (see _capture_zoom_baseline for both)."""
        return self._cache.mip_for_scale(_max_axis_scale(painter.deviceTransform()))

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        with perf_trace.phase("repaint"):
            rect = option.exposedRect.intersected(self._bounding_rect)
            if rect.isEmpty():
                return
            # Everything below tiles and pads in LEVEL pixels, deriving each
            # destination rect in scene units by multiplying back by `scale`.
            # Doing it the other way (tiling in scene units) would round block
            # edges independently per block, so adjacent blocks' destination
            # rects could fail to abut -- `scale` is an exact power of two, so
            # this way they always do.
            mip = self._select_mip(painter)
            self._last_mip = mip
            scale = self._cache.mip_scale(mip)
            # Floor/ceil, not round -- the composited block must fully cover
            # exposedRect (a gap at the edge would leave a visible unpainted
            # sliver), matching iso_geometry's own "whole pixels only" integer
            # discipline throughout this project's rendering code.
            x0, y0 = int(math.floor(rect.left() / scale)), int(math.floor(rect.top() / scale))
            x1, y1 = int(math.ceil(rect.right() / scale)), int(math.ceil(rect.bottom() / scale))
            canvas_w, canvas_h = self._cache.canvas_dims(mip)
            x1, y1 = min(x1, canvas_w), min(y1, canvas_h)
            if x1 <= x0 or y1 <= y0:
                return

            # Tile [x0,x1)x[y0,y1) into PAINT_BLOCK_PX-aligned sub-blocks (grid
            # anchored at the canvas origin, not at x0/y0, so repeated partial
            # exposures of the same region always request the same block
            # boundaries -- irrelevant to correctness, since each block is
            # independently correct, but keeps cache chunk-fetch patterns
            # stable across paint calls). Overlapping padded draws between
            # adjacent blocks are harmless: each is an opaque, independently
            # correct overwrite of its own destination rect.
            block_x0 = (x0 // self.PAINT_BLOCK_PX) * self.PAINT_BLOCK_PX
            block_y0 = (y0 // self.PAINT_BLOCK_PX) * self.PAINT_BLOCK_PX
            for by in range(block_y0, y1, self.PAINT_BLOCK_PX):
                for bx in range(block_x0, x1, self.PAINT_BLOCK_PX):
                    cx0, cy0 = max(bx, x0), max(by, y0)
                    cx1 = min(bx + self.PAINT_BLOCK_PX, x1)
                    cy1 = min(by + self.PAINT_BLOCK_PX, y1)
                    if cx1 <= cx0 or cy1 <= cy0:
                        continue
                    px0 = max(0, cx0 - self.PAINT_PAD_PX)
                    py0 = max(0, cy0 - self.PAINT_PAD_PX)
                    px1 = min(canvas_w, cx1 + self.PAINT_PAD_PX)
                    py1 = min(canvas_h, cy1 + self.PAINT_PAD_PX)
                    block = self._cache.render_rect(px0, py0, px1, py1, mip=mip)
                    if block.size == 0:
                        continue
                    h, w = block.shape[:2]
                    contiguous = np.ascontiguousarray(block)
                    qimg = QImage(contiguous.data, w, h, 3 * w, QImage.Format_RGB888)
                    if scale == 1.0:
                        # The point overload is a different QPainter code path
                        # from the scaled one and is not guaranteed
                        # bit-identical to it at unit scale. Keeping it for the
                        # reference level is what makes this phase structurally
                        # incapable of regressing any existing byte-identity
                        # bar, rather than merely empirically not doing so.
                        painter.drawImage(px0, py0, qimg)
                    else:
                        painter.drawImage(
                            QRectF(px0 * scale, py0 * scale, w * scale, h * scale),
                            qimg,
                            QRectF(0, 0, w, h),
                        )

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Schedules a Qt repaint for exactly the given canvas-pixel bbox --
        the Qt-side half of an edit's redraw. Does NOT touch self._cache
        itself (the caller must already have called cache.patch(bbox) --
        see ViewerWindow._apply_dirty); this is purely "please repaint this
        area," so the next paint() call pulls the already-patched pixels.

        Distinct from IsoChunkCache.invalidate_region() (Phase B-B): that
        one evicts cache entries to force a real recomposite; this one
        never touches the cache at all, just Qt's own dirty-region
        tracking. Same name, different class, different layer."""
        x0, y0, x1, y1 = bbox
        self.update(QRectF(x0, y0, x1 - x0, y1 - y0))

    def refresh_canvas_dims(self) -> None:
        """Re-reads canvas_dims() from self._cache and grows boundingRect if
        it changed -- Track P3-g6. SlopedChunkCache.canvas_dims() can now
        change AFTER construction (sprites toggled on at a low
        elev_step_pct stop widens it, see that method's own docstring),
        which nothing before this needed: every other cache's canvas_dims()
        answer is fixed for the cache's whole lifetime, and this class's
        own __init__ only ever read it once on that assumption.

        prepareGeometryChange() MUST precede the assignment -- see
        EdgeTickItem._apply_geometry's own comment for the failure mode if
        reversed (a stale Qt scene-index entry, not a clean failure)."""
        canvas_w, canvas_h = self._cache.canvas_dims()
        rect = QRectF(0, 0, canvas_w, canvas_h)
        if rect == self._bounding_rect:
            return
        self.prepareGeometryChange()
        self._bounding_rect = rect


def _mapped_delta(transform: QTransform, dx: float, dy: float) -> tuple[float, float]:
    """`(dx, dy)` through `transform`'s 2x2 linear part only, dropping the
    translation. A direction is not a point: mapping one with
    QTransform.map() would add the translation and give a vector that
    changes whenever the view scrolls.

    Sits beside _max_axis_scale above, which reads a scale off a QTransform
    the same way."""
    return (
        dx * transform.m11() + dy * transform.m21(),
        dx * transform.m12() + dy * transform.m22(),
    )


class TickPaintStats:
    """What the LAST paint() actually drew. Write-only, the same convention
    MapCanvasItem._last_mip already uses: paint() never reads it back, so
    nothing here can make rendering history-dependent.

    It records ONE paint, so a reader treating it as "what this render
    chose" has to have driven exactly one paint, and Qt may split a repaint
    into several. Its purpose is letting the tests assert device tick length
    and the LOD ladder as numbers rather than by measuring pixels.

    A plain class rather than a dataclass only because viewer.py already
    binds `field` as a loop variable, so dataclasses.field cannot be
    imported here without shadowing it."""

    def __init__(self) -> None:
        self.spacing_px: dict[str, float] = {}
        self.lod: dict[str, edge_ticks.TickLod] = {}
        self.ticks_drawn = 0
        self.labels_drawn = 0
        self.minor_len_px = 0.0
        self.major_len_px = 0.0


class EdgeTickItem(QGraphicsItem):
    """The ruler strip of distance ticks just outside the map's own border,
    View > Distance Ticks. Anchors come from the Qt-free descape.edge_ticks;
    everything visual is decided here.

    **self._runs is built once per (dims, interval) and never inside
    paint().** That is load-bearing, not an optimisation: the padded
    bounding rect spans the viewport, so Qt treats this item as exposed on
    essentially every repaint, including every invalidate_region during a
    drag-paint stroke, and a 480x480 at interval 4 is 484 ticks. Do not
    "simplify" the rebuild back into paint().

    Marks are drawn in DEVICE space, under a reset transform, so a tick is a
    fixed pixel length and a label stays upright and readable at any zoom.
    Scene-space drawing would be sheared and rotated by Flat's
    scale(1, 0.5) + rotate(-45), and sub-pixel at fit-to-view on a 480x480.

    Colours deliberately ignore settings.get_dark_mode: MapView paints its
    scene background unconditionally to OUTSIDE_MAP_COLOR, so there is no
    light variant of the surface these sit on. Gold is not used either, since
    HIGHLIGHT_OUTLINE_PEN reserves it for "live and about to paint"."""

    # Majors match the existing map-extent outline; minors are dimmer so the
    # two ranks read apart by brightness as well as by length.
    MAJOR_PEN = QPen(QColor(200, 200, 200), 0)
    MINOR_PEN = QPen(QColor(140, 140, 140), 0)
    LABEL_COLOR = QColor(200, 200, 200)

    def __init__(
        self,
        map_w: int,
        map_h: int,
        interval: int,
        map_rect: QRectF,
        proj: "iso_geometry.IsoProjection | None" = None,
        tile_px: int | None = None,
    ) -> None:
        super().__init__()
        self._map_w = map_w
        self._map_h = map_h
        self._interval = interval
        self._proj = proj
        self._tile_px = tile_px
        self._map_rect = QRectF(map_rect)
        self._pad = 0.0
        self._runs: tuple[edge_ticks.EdgeRun, ...] = ()
        self._base_rect = QRectF(map_rect)
        self._bounding_rect = QRectF(map_rect)
        self._font = QFont()
        self._font.setPixelSize(edge_ticks.LABEL_FONT_PX)
        self.stats = TickPaintStats()
        self._rebuild()

    def _rebuild(self) -> None:
        self._runs = edge_ticks.edge_runs(
            self._map_w, self._map_h, self._interval, proj=self._proj, tile_px=self._tile_px
        )
        # Unioned with the anchor extent rather than taken as the map rect
        # alone: on a map whose minimum elevation is above 0 the elevation-0
        # anchors sit slightly below canvas_h. The union costs one pass over
        # already-built anchors and removes the question entirely.
        xs = [x for run in self._runs for x, _y in run.anchors]
        ys = [y for run in self._runs for _x, y in run.anchors]
        anchors_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self._base_rect = self._map_rect.united(anchors_rect)
        self._apply_geometry()

    def _apply_geometry(self) -> None:
        rect = self._base_rect.adjusted(-self._pad, -self._pad, self._pad, self._pad)
        if rect == self._bounding_rect:
            return
        # prepareGeometryChange() MUST precede the assignment. Reversed, Qt
        # keeps the stale index entry and the symptom is intermittent
        # leftover tick fragments after a scroll, not a clean failure.
        self.prepareGeometryChange()
        self._bounding_rect = rect

    def set_interval(self, interval: int) -> None:
        if interval == self._interval:
            return
        self._interval = interval
        self._rebuild()
        self.update()

    def set_min_view_scale(self, scale: float | None) -> None:
        """Re-pads the bounding rect for the SMALLEST scale the view can
        reach. A constant device tick length means the scene-unit overhang
        grows without bound as the view zooms out, so the pad cannot be a
        fixed fraction of the map: at the coarsest mip on a narrow viewport
        the real overhang exceeds MapView.OVERSCROLL_FRACTION outright."""
        if scale is None or scale <= 0:
            return
        pad = edge_ticks.scene_pad(scale)
        if pad == self._pad:
            return
        self._pad = pad
        self._apply_geometry()

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        with perf_trace.phase("repaint"):
            # worldTransform(), NOT deviceTransform(): resetTransform() below
            # puts drawing in the painter's logical device coordinates, and
            # worldTransform() maps scene into those same coordinates.
            # deviceTransform() folds in the device pixel ratio, which would
            # halve every tick on a HiDPI screen. MapCanvasItem._select_mip
            # reads deviceTransform() deliberately, for texel density, which is
            # a different question.
            world = painter.worldTransform()
            stats = TickPaintStats()
            self.stats = stats

            painter.save()
            try:
                painter.resetTransform()
                painter.setFont(self._font)
                for run in self._runs:
                    out_x, out_y = _mapped_delta(world, *run.outward)
                    reach = math.hypot(out_x, out_y)
                    if reach <= 0:
                        continue
                    unit_x, unit_y = out_x / reach, out_y / reach
                    step_x, step_y = _mapped_delta(world, *run.minor_step)
                    spacing = math.hypot(step_x, step_y)
                    lod = edge_ticks.tick_lod(spacing)
                    stats.spacing_px[run.edge] = spacing
                    stats.lod[run.edge] = lod
                    if not lod.draw_edge:
                        continue
                    self._paint_run(painter, world, run, lod, unit_x, unit_y, stats)
            finally:
                # Restored even on an exception: a leaked identity transform
                # would corrupt whatever paints after this item, and the symptom
                # (stale, misplaced marks) looks exactly like the boundingRect
                # bug above.
                painter.restore()

    def _paint_run(self, painter, world, run, lod, unit_x, unit_y, stats) -> None:
        for index, (anchor_x, anchor_y) in enumerate(run.anchors):
            major = run.majors[index]
            if not major and not lod.draw_minors:
                continue
            point = world.map(QPointF(anchor_x, anchor_y))
            length = edge_ticks.MAJOR_TICK_PX if major else edge_ticks.MINOR_TICK_PX
            painter.setPen(self.MAJOR_PEN if major else self.MINOR_PEN)
            painter.drawLine(
                QPointF(point.x(), point.y()),
                QPointF(point.x() + unit_x * length, point.y() + unit_y * length),
            )
            stats.ticks_drawn += 1
            if major:
                stats.major_len_px = length
            else:
                stats.minor_len_px = length
            if not (major and lod.draw_labels):
                continue
            center_x = point.x() + unit_x * edge_ticks.LABEL_CENTER_PX
            center_y = point.y() + unit_y * edge_ticks.LABEL_CENTER_PX
            box = QRectF(
                center_x - edge_ticks.LABEL_BOX_W_PX / 2.0,
                center_y - edge_ticks.LABEL_BOX_H_PX / 2.0,
                edge_ticks.LABEL_BOX_W_PX,
                edge_ticks.LABEL_BOX_H_PX,
            )
            painter.setPen(self.LABEL_COLOR)
            painter.drawText(box, Qt.AlignCenter, str(run.tiles[index]))
            stats.labels_drawn += 1
