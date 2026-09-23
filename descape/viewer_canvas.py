"""The QGraphicsItems the map scene is drawn from, plus the QTransform
readers they and MapView size themselves by.

MapCanvasItem paints the composited map; EdgeTickItem paints the
distance-tick strip outside its border. Both are driven by MapView
(descape.map_view), which imports this module one way."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from PyQt5.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QGraphicsItem,
)

from descape import (
    edge_ticks,
    grid_overlay,
    iso_geometry,
    perf_trace,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)


def map_overlay_font(px: int) -> QFont:
    """A font for map-overlay text (ruler readout, distance-tick numbers,
    stacked-unit badges), deliberately NOT following the app chrome font
    the way viewer_dialogs.apply_ui_font() sets it.

    Same carve-out rationale apply_theme()'s docstring gives for colors:
    this text sits on game data and is sized in device pixels against fixed
    boxes, so scaling it with chrome would break the LOD ladder and the
    label boxes it is measured against. systemFont(GeneralFont) is the base
    rather than a bare QFont() because a bare one inherits whatever
    app.setFont() was last called with at construction time -- stale, not
    fixed -- and rather than a hardcoded family because this stays portable."""
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    font.setPixelSize(px)
    return font


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


def level_rect_for(cache, mip: int, scene_rect: QRectF) -> tuple[int, int, int, int] | None:
    """The LEVEL-`mip`-pixel chunk-cache rect a scene-space rect covers --
    MapCanvasItem.paint()'s own arithmetic (floor the low edge, ceil the
    high edge, clamp to canvas_dims(mip)), extracted so
    MapView.viewport_chunk_target() (2026-09-07 plan's A2.3) doesn't grow a
    second copy that can silently drift from what a real paint selects.
    Floor/ceil, not round: the level rect must fully cover scene_rect,
    matching paint()'s own "whole pixels only" discipline -- a gap at the
    edge would leave a real caller's request under-covered.

    `descape.render_cache._ChunkCacheBase._bbox_to_level()` is deliberately
    NOT reused here even though it does a similar floor/ceil conversion: it
    takes an integer REFERENCE-pixel bbox, where both of this function's
    callers start from a float scene-space QRectF, and mixing the two
    integer-vs-float conventions in one function would be more confusing
    than the small amount of shared arithmetic is worth.

    None for a degenerate or empty rect once clamped to canvas bounds --
    both callers already treat that as "nothing to do" and return early."""
    scale = cache.mip_scale(mip)
    x0, y0 = math.floor(scene_rect.left() / scale), math.floor(scene_rect.top() / scale)
    x1, y1 = math.ceil(scene_rect.right() / scale), math.ceil(scene_rect.bottom() / scale)
    canvas_w, canvas_h = cache.canvas_dims(mip)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(x1, canvas_w), min(y1, canvas_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


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
    # Phase B-D-d note: 2px was justified above partly
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

    def __init__(self, cache: IsoChunkCache | FlatChunkCache | SlopedChunkCache):
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
        # Cleared once, on this item's first paint only -- one new
        # MapCanvasItem is built per MapView.set_source(), so this is
        # exactly "first paint of the current canvas item" with no
        # plumbing back to the load path that constructed it.
        self._first_paint_pending = True
        # Optional (elapsed_seconds, mip) callback, called on EVERY paint
        # while installed. Plain attribute: a QGraphicsItem is not a QObject.
        self._on_paint_timed: Callable[[float, int], None] | None = None
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
        first_paint = self._first_paint_pending
        self._first_paint_pending = False
        # Read once into a local: the viewer drops the callback as soon as it
        # reports, and the finally below must see what this paint started with.
        on_paint_timed = self._on_paint_timed
        paint_t0 = time.perf_counter() if on_paint_timed is not None else 0.0
        try:
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
                level_rect = level_rect_for(self._cache, mip, rect)
                if level_rect is None:
                    return
                x0, y0, x1, y1 = level_rect
                canvas_w, canvas_h = self._cache.canvas_dims(mip)

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
        finally:
            # Every paint, not just the first, and including the two early
            # returns above: whoever installed the callback is summing them.
            if on_paint_timed is not None:
                on_paint_timed(time.perf_counter() - paint_t0, self._last_mip)
            if first_paint:
                perf_trace.first_paint_done()

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
        self.axis_labels_drawn = 0
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
    "simplify" the rebuild back into paint(). paint() culls that fixed run
    list against option.exposedRect and batches the survivors per pen, which
    is what keeps a small repaint cheap.

    Marks are drawn in DEVICE space, under a reset transform, so a tick is a
    fixed pixel length and a label stays upright and readable at any zoom.
    Scene-space drawing would be sheared and rotated by Flat's
    scale(1, 0.5) + rotate(-45), and sub-pixel at fit-to-view on a 480x480.

    Colours deliberately ignore settings.get_dark_mode: MapView paints its
    scene background unconditionally to OUTSIDE_MAP_COLOR, so there is no
    light variant of the surface these sit on. Gold is not used either, since
    that's the edit highlight's default color ("live and about to paint"),
    now user-settable via settings.OVERLAY_COLORS."""

    # Majors match the existing map-extent outline; minors are dimmer so the
    # two ranks read apart by brightness as well as by length.
    MAJOR_PEN = QPen(QColor(200, 200, 200), 0)
    MINOR_PEN = QPen(QColor(140, 140, 140), 0)
    LABEL_COLOR = QColor(200, 200, 200)

    # Device-pixel slack added on top of edge_ticks.device_reach_px() before
    # culling: cosmetic pen width plus antialiasing spill.
    CULL_MARGIN_PX = 2.0

    def __init__(
        self,
        map_w: int,
        map_h: int,
        interval: int,
        map_rect: QRectF,
        proj: iso_geometry.IsoProjection | None = None,
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
        self._font = map_overlay_font(edge_ticks.LABEL_FONT_PX)
        self.stats = TickPaintStats()
        # Same flag, same reason as MapCanvasItem: without it Qt reports the
        # whole boundingRect as exposed and paint()'s cull can never fire.
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)
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

    def set_label_font_px(self, font_px: int) -> None:
        """Rebuilds the label font. Does NOT recompute the pad itself --
        the pad depends on both font_px and the view's min scale, and
        MapView.apply_distance_tick_font() already knows the latter via
        _repad_edge_ticks(), so it calls that right after this rather than
        this item duplicating that floor/current-scale min() logic."""
        if font_px == self._font.pixelSize():
            return
        self._font = map_overlay_font(font_px)
        self.update()

    def set_min_view_scale(self, scale: float | None) -> None:
        """Re-pads the bounding rect for the SMALLEST scale the view can
        reach. A constant device tick length means the scene-unit overhang
        grows without bound as the view zooms out, so the pad cannot be a
        fixed fraction of the map: at the coarsest mip on a narrow viewport
        the real overhang exceeds MapView.OVERSCROLL_FRACTION outright."""
        if scale is None or scale <= 0:
            return
        pad = edge_ticks.scene_pad(scale, self._font.pixelSize())
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
            exposed = self._exposed_device_rect(world, option)

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
                    self._paint_run(painter, world, run, lod, unit_x, unit_y, stats, exposed)
            finally:
                # Restored even on an exception: a leaked identity transform
                # would corrupt whatever paints after this item, and the symptom
                # (stale, misplaced marks) looks exactly like the boundingRect
                # bug above.
                painter.restore()

    def _exposed_device_rect(self, world, option) -> QRectF | None:
        """`option`'s exposed area in DEVICE space, grown by the furthest any
        mark can land from its anchor, or None for "cull nothing".

        None covers both a caller that passes no option at all (the tests'
        direct paint() calls) and an empty exposed rect. Grown by
        edge_ticks.device_reach_px, which is that module's own bound on the
        outermost drawn pixel and so covers a label as well as its tick:
        an anchor outside this rect cannot paint inside the exposed one.
        world.mapRect gives the bounding rect of the mapped quad, which under
        Flat's rotate-and-squash is a superset. Over-covering is free."""
        rect = getattr(option, "exposedRect", None)
        if rect is None or rect.isEmpty():
            return None
        margin = edge_ticks.device_reach_px(self._font.pixelSize()) + self.CULL_MARGIN_PX
        return world.mapRect(QRectF(rect)).adjusted(-margin, -margin, margin, margin)

    def _paint_run(self, painter, world, run, lod, unit_x, unit_y, stats, exposed=None) -> None:
        minors: list[QLineF] = []
        majors: list[QLineF] = []
        labels: list[tuple[QRectF, str]] = []
        for index, (anchor_x, anchor_y) in enumerate(run.anchors):
            major = run.majors[index]
            if not major and not lod.draw_minors:
                continue
            point = world.map(QPointF(anchor_x, anchor_y))
            if exposed is not None and not exposed.contains(point):
                continue
            length = edge_ticks.MAJOR_TICK_PX if major else edge_ticks.MINOR_TICK_PX
            (majors if major else minors).append(
                QLineF(
                    point.x(),
                    point.y(),
                    point.x() + unit_x * length,
                    point.y() + unit_y * length,
                )
            )
            stats.ticks_drawn += 1
            if major:
                stats.major_len_px = length
            else:
                stats.minor_len_px = length
            if not (major and lod.draw_labels):
                continue
            font_px = self._font.pixelSize()
            label_center = edge_ticks.label_center_px(font_px)
            box_w, box_h = edge_ticks.label_box_px(font_px)
            center_x = point.x() + unit_x * label_center
            center_y = point.y() + unit_y * label_center
            box = QRectF(
                center_x - box_w / 2.0,
                center_y - box_h / 2.0,
                box_w,
                box_h,
            )
            labels.append((box, str(run.tiles[index])))
            stats.labels_drawn += 1
        if lod.draw_labels and run.anchors:
            # One letter per edge, at the middle anchor, outboard of the numbers.
            anchor_x, anchor_y = run.anchors[len(run.anchors) // 2]
            point = world.map(QPointF(anchor_x, anchor_y))
            if exposed is None or exposed.contains(point):
                font_px = self._font.pixelSize()
                center = edge_ticks.axis_label_center_px(font_px)
                box_w, box_h = edge_ticks.axis_label_box_px(font_px)
                center_x = point.x() + unit_x * center
                center_y = point.y() + unit_y * center
                box = QRectF(center_x - box_w / 2.0, center_y - box_h / 2.0, box_w, box_h)
                labels.append((box, edge_ticks.axis_letter(run.edge)))
                stats.axis_labels_drawn += 1
        # One drawLines per pen instead of one drawLine per tick. Adjacent
        # ticks are >= 4 device px apart, so grouping cannot share a pixel.
        if minors:
            painter.setPen(self.MINOR_PEN)
            painter.drawLines(minors)
        if majors:
            painter.setPen(self.MAJOR_PEN)
            painter.drawLines(majors)
        if labels:
            painter.setPen(self.LABEL_COLOR)
            for box, text in labels:
                painter.drawText(box, Qt.AlignCenter, text)


class GridItem(QGraphicsItem):
    """View > Grid's ground-plane lattice: one line per tile boundary over the
    whole map, with every fourth line a major. Geometry comes from the
    Qt-free descape.grid_overlay.

    Not the usual grid. With Follow Terrain Elevation on (and always in Flat)
    the grid is baked into the chunk composite under the sprites, and this
    item is hidden. It serves only the follow-off elevation-0 lattice in
    Stepped and Sloped, which is not per-tile geometry and so draws over
    sprites, and the Settings > Appearance slider preview, where re-baking
    per tick would evict the whole canvas each time.

    **Lines are prebuilt once per (dims, style) in _rebuild(), never inside
    paint().** Same reason as EdgeTickItem: the bounding rect spans the map,
    so Qt treats this item as exposed on nearly every repaint, including each
    invalidate_region of a drag-paint stroke. paint() is then at most four
    batched drawLines() calls.

    Lines stay in scene space; only the LOD measurement is device-space.
    Cosmetic pens keep the user's thickness in device pixels, and the same
    weight on both axes under Flat's rotate-and-squash."""

    def __init__(
        self,
        map_w: int,
        map_h: int,
        map_rect: QRectF,
        proj: iso_geometry.IsoProjection | None = None,
        tile_px: int | None = None,
        blend: int = grid_overlay.BLEND_DEFAULT,
        thickness: int = grid_overlay.THICKNESS_DEFAULT,
    ) -> None:
        super().__init__()
        self._map_w = map_w
        self._map_h = map_h
        self._proj = proj
        self._tile_px = tile_px
        self._map_rect = QRectF(map_rect)
        self._pad = 0.0
        self._axes: tuple[grid_overlay.GridAxis, ...] = ()
        # Per axis: (minor lines, major lines).
        self._batches: list[tuple[tuple[QLineF, ...], tuple[QLineF, ...]]] = []
        self._base_rect = QRectF(map_rect)
        self._bounding_rect = QRectF(map_rect)
        self._minor_pen = QPen()
        self._major_pen = QPen()
        self.paint_count = 0
        self.last_lod: dict[str, grid_overlay.GridLod] = {}
        self.set_appearance(blend, thickness)
        self._rebuild()

    def _rebuild(self) -> None:
        self._axes = grid_overlay.grid_axes(self._map_w, self._map_h, proj=self._proj, tile_px=self._tile_px)
        self._batches = []
        xs: list[int] = []
        ys: list[int] = []
        for axis in self._axes:
            minors: list[QLineF] = []
            majors: list[QLineF] = []
            for ((x0, y0), (x1, y1)), major in zip(axis.lines, axis.majors, strict=True):
                (majors if major else minors).append(QLineF(x0, y0, x1, y1))
                xs += (x0, x1)
                ys += (y0, y1)
            self._batches.append((tuple(minors), tuple(majors)))
        lines_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self._base_rect = self._map_rect.united(lines_rect)
        self._apply_geometry()

    def _apply_geometry(self) -> None:
        rect = self._base_rect.adjusted(-self._pad, -self._pad, self._pad, self._pad)
        if rect == self._bounding_rect:
            return
        # prepareGeometryChange() MUST precede the assignment; see EdgeTickItem.
        self.prepareGeometryChange()
        self._bounding_rect = rect

    def set_appearance(self, blend: int, thickness: int) -> None:
        """Rebuilds the two pens only. The pad already covers the thickest
        stop, so no geometry change is needed."""
        minor, major = grid_overlay.grid_colors(blend)
        width = grid_overlay.snap_thickness(thickness)
        self._minor_pen = QPen(QColor(*minor), width)
        self._minor_pen.setCosmetic(True)
        self._major_pen = QPen(QColor(*major), width)
        self._major_pen.setCosmetic(True)
        self.update()

    def pens(self) -> tuple[QPen, QPen]:
        return self._minor_pen, self._major_pen

    def set_min_view_scale(self, scale: float | None) -> None:
        """Pads for the thickest pen's device-space half-width at the
        smallest scale the view can reach, like EdgeTickItem.set_min_view_scale."""
        if scale is None or scale <= 0:
            return
        pad = edge_ticks.PAD_SAFETY * (max(grid_overlay.THICKNESS_STOPS) / 2.0 + 1.0) / scale
        if pad == self._pad:
            return
        self._pad = pad
        self._apply_geometry()

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        self.paint_count += 1
        # A centred blend slider is invisible, so skip the whole pass.
        if not self._minor_pen.color().alpha() and not self._major_pen.color().alpha():
            return
        with perf_trace.phase("grid_repaint"):
            world = painter.worldTransform()
            painter.save()
            try:
                for axis, (minors, majors) in zip(self._axes, self._batches, strict=True):
                    lod = grid_overlay.grid_lod(math.hypot(*_mapped_delta(world, *axis.minor_step)))
                    self.last_lod[axis.axis] = lod
                    if not lod.draw_grid:
                        continue
                    if lod.draw_minors and minors:
                        painter.setPen(self._minor_pen)
                        painter.drawLines(list(minors))
                    if majors:
                        painter.setPen(self._major_pen)
                        painter.drawLines(list(majors))
            finally:
                painter.restore()


class StackBadgeItem(QGraphicsItem):
    """The stacked-unit count badges, View > Show Stacked-Unit Badges: one
    item for the whole layer, drawing a small count above every tile where
    unit_pick.stack_groups() found a unit hidden under another.

    Anchors are scene points handed in by MapView (each group tile's top
    vertex, via its own _tile_polygon), so this item knows nothing about
    terrain styles. Drawn in device space like EdgeTickItem, so a badge
    stays one size at every zoom; the whole layer drops out once a tile is
    too small on screen for a badge to say which tile it belongs to."""

    FONT_PX = 11
    PAD_PX = 3.0
    GAP_PX = 2.0
    BACKGROUND_ALPHA = 200
    # Below this many device pixels per tile the badges would pile onto
    # their neighbours' tiles, so none are drawn at all.
    MIN_TILE_DEVICE_PX = 12.0

    def __init__(self, tile_extent: float, color: QColor) -> None:
        super().__init__()
        self._tile_extent = float(tile_extent)
        self._badges: list[tuple[QPointF, str]] = []
        self._bounding_rect = QRectF()
        self._font = map_overlay_font(self.FONT_PX)
        self._font.setBold(True)
        self._pen = QPen(color)
        self._background = QColor(0, 0, 0, self.BACKGROUND_ALPHA)
        self.badges_drawn = 0
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def set_badges(self, badges: list[tuple[QPointF, int]]) -> None:
        self.prepareGeometryChange()
        self._badges = [(QPointF(point), str(count)) for point, count in badges]
        if not self._badges:
            self._bounding_rect = QRectF()
        else:
            xs = [p.x() for p, _text in self._badges]
            ys = [p.y() for p, _text in self._badges]
            # The badge's device size in scene units is largest at the LOD
            # floor, where one tile is MIN_TILE_DEVICE_PX; 4 tiles covers it.
            pad = 4 * self._tile_extent
            self._bounding_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)).adjusted(
                -pad, -pad, pad, pad
            )
        self.update()

    def badge_texts(self) -> list[str]:
        return [text for _point, text in self._badges]

    def set_color(self, color: QColor) -> None:
        self._pen = QPen(color)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        self.badges_drawn = 0
        if not self._badges:
            return
        world = painter.worldTransform()
        tile_dx, tile_dy = _mapped_delta(world, self._tile_extent, 0.0)
        if math.hypot(tile_dx, tile_dy) < self.MIN_TILE_DEVICE_PX:
            return
        exposed = getattr(option, "exposedRect", None)
        exposed_device = None
        if exposed is not None and not exposed.isEmpty():
            margin = self.FONT_PX * 4
            exposed_device = world.mapRect(QRectF(exposed)).adjusted(-margin, -margin, margin, margin)
        painter.save()
        try:
            painter.resetTransform()
            painter.setFont(self._font)
            metrics = painter.fontMetrics()
            for point, text in self._badges:
                device = world.map(point)
                if exposed_device is not None and not exposed_device.contains(device):
                    continue
                w = metrics.horizontalAdvance(text) + 2 * self.PAD_PX
                h = metrics.height() + self.PAD_PX
                box = QRectF(device.x() - w / 2.0, device.y() - self.GAP_PX - h, w, h)
                painter.setPen(Qt.NoPen)
                painter.setBrush(self._background)
                painter.drawRoundedRect(box, 3.0, 3.0)
                painter.setPen(self._pen)
                painter.drawText(box, Qt.AlignCenter, text)
                self.badges_drawn += 1
        finally:
            painter.restore()


class AnalysisMarkerItem(QGraphicsItem):
    """Tools > Map Analysis' on-map markers: one item for the whole layer,
    drawing a severity glyph over every tile map_analysis.marker_anchors()
    returned, while the results dialog is open.

    StackBadgeItem's shape (scene-point anchors handed in by MapView, drawn
    in device space so a marker stays one size at every zoom) minus its LOD
    floor: seeing where findings cluster at fit-to-view is the point, so
    markers are drawn at every zoom and allowed to overlap when far out.
    Without that floor the scene-unit overhang grows as the view zooms out,
    so the bounding rect is padded from the smallest reachable scale, the
    way EdgeTickItem.set_min_view_scale() pads the tick strip."""

    FONT_PX = 11
    PAD_PX = 3.0
    GAP_PX = 2.0
    BACKGROUND_ALPHA = 200
    FOCUS_RING_PX = 2.0
    # Furthest a marker box reaches from its anchor, in device pixels: the
    # widest glyph plus count ("!! x200") with padding and the focus ring.
    DEVICE_REACH_PX = 64.0
    GLYPHS: ClassVar[dict[str, str]] = {"error": "!!", "warning": "!", "info": "i"}

    def __init__(self, color: QColor) -> None:
        super().__init__()
        self._markers: list[tuple[QPointF, str, str, int]] = []
        self._focus: QPointF | None = None
        self._pad = 0.0
        self._bounding_rect = QRectF()
        self._font = map_overlay_font(self.FONT_PX)
        self._font.setBold(True)
        self._color = QColor(color)
        self._background = QColor(0, 0, 0, self.BACKGROUND_ALPHA)
        self.markers_drawn = 0
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)
        # A map-wide layer: it must never swallow a Units-mode click.
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_markers(self, markers: list[tuple[QPointF, str, int]]) -> None:
        """(scene anchor, severity, finding count) per marked tile."""
        self._markers = [
            (QPointF(point), severity, self._label(severity, count), int(count)) for point, severity, count in markers
        ]
        self._apply_geometry()

    def set_focus(self, point: QPointF | None) -> None:
        """The anchor of the marker to ring, or None."""
        self._focus = None if point is None else QPointF(point)
        self.update()

    def set_min_view_scale(self, scale: float | None) -> None:
        if scale is None or scale <= 0:
            return
        pad = edge_ticks.PAD_SAFETY * self.DEVICE_REACH_PX / scale
        if pad == self._pad:
            return
        self._pad = pad
        self._apply_geometry()

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def color(self) -> QColor:
        return QColor(self._color)

    def marker_points(self) -> list[tuple[QPointF, str, int]]:
        return [(QPointF(point), severity, count) for point, severity, _label, count in self._markers]

    @classmethod
    def _label(cls, severity: str, count: int) -> str:
        glyph = cls.GLYPHS.get(severity, "?")
        return f"{glyph} x{count}" if count > 1 else glyph

    def _apply_geometry(self) -> None:
        self.prepareGeometryChange()
        if not self._markers:
            self._bounding_rect = QRectF()
        else:
            xs = [p.x() for p, *_rest in self._markers]
            ys = [p.y() for p, *_rest in self._markers]
            pad = self._pad
            self._bounding_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)).adjusted(
                -pad, -pad, pad, pad
            )
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        self.markers_drawn = 0
        if not self._markers:
            return
        world = painter.worldTransform()
        exposed = getattr(option, "exposedRect", None)
        exposed_device = None
        if exposed is not None and not exposed.isEmpty():
            margin = self.DEVICE_REACH_PX
            exposed_device = world.mapRect(QRectF(exposed)).adjusted(-margin, -margin, margin, margin)
        painter.save()
        try:
            painter.resetTransform()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setFont(self._font)
            metrics = painter.fontMetrics()
            text_pen = QPen(self._color)
            ring_pen = QPen(self._color, self.FOCUS_RING_PX)
            focused_box = None
            for point, _severity, label, _count in self._markers:
                device = world.map(point)
                if exposed_device is not None and not exposed_device.contains(device):
                    continue
                w = metrics.horizontalAdvance(label) + 2 * self.PAD_PX
                h = metrics.height() + self.PAD_PX
                box = QRectF(device.x() - w / 2.0, device.y() - self.GAP_PX - h, w, h)
                painter.setPen(Qt.NoPen)
                painter.setBrush(self._background)
                painter.drawRoundedRect(box, 3.0, 3.0)
                painter.setPen(text_pen)
                painter.drawText(box, Qt.AlignCenter, label)
                if self._focus is not None and point == self._focus:
                    focused_box = box
                self.markers_drawn += 1
            # Last, so a neighbour drawn later can't cover the ring.
            if focused_box is not None:
                painter.setPen(ring_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(focused_box.adjusted(-1.0, -1.0, 1.0, 1.0), 4.0, 4.0)
        finally:
            painter.restore()


class UnitGhostItem(QGraphicsItem):
    """The translucent copy of a unit that follows the cursor while it is
    being dragged (the mid-drag move preview).

    A paint()-based item rather than a path/rect one because the ghost has to
    be able to show real sprite art: MapCanvasItem and EdgeTickItem are the
    only two precedents here, and every other overlay in this module draws
    geometry alone.

    **It touches no render-cache state at all**, which is the whole design.
    Every unit mutation routes through ViewerWindow._after_unit_mutation(),
    whose three steps are whole-canvas (invalidate_units' unvectorized walk
    over every unit, a whole-canvas invalidate_region, and SlopedChunkCache.
    patch()'s wholesale pick-plane clear) -- one to three orders of magnitude
    past a frame. An overlay costs none of that, and leaves the pick planes
    warm so the per-move _pick_tile stays cheap.

    The honest boundary, and the first thing an in-app pass notices: the
    original unit stays drawn at the source tile for the whole drag. Unit
    pixels are baked into composited chunk arrays, so an overlay can draw on
    top of them but cannot erase them."""

    # Translucent enough to read as a preview rather than as the real thing,
    # opaque enough that a sprite's own art is still identifiable.
    GHOST_OPACITY = 0.6

    def __init__(self) -> None:
        super().__init__()
        # The numpy arrays are held alongside their QImages deliberately:
        # QImage does not own the buffer it is constructed over, so dropping
        # the array would leave the item painting freed memory.
        self._arrays: list[np.ndarray] = []
        self._images: list[tuple[QImage, float, float]] = []
        # (polygon, brush, pen) per mark polygon: a group ghost (GH #75) mixes owners.
        self._marks: list[tuple[QPolygonF, QBrush, QPen]] = []
        self._bounding_rect = QRectF()
        self.setOpacity(self.GHOST_OPACITY)

    def set_sprites(self, draws) -> bool:
        """draws: render.unit_sprite_draws_at()'s (draw, px, py) triples, in
        canvas pixels -- which ARE scene coordinates, since scene space is
        pinned to the reference mip (MapCanvasItem's plan decision D2).

        Returns whether anything is left to show, so the caller can fall back
        to the coloured mark on an empty resolve rather than leaving a stale
        ghost on screen."""
        return self.set_content(draws, ())

    def set_mark(self, polygons, color: tuple[int, int, int]) -> bool:
        """polygons: unit_pick.unit_polygons()' plain (x, y) point lists, in
        the same canvas/scene pixels as set_sprites(). The coloured-mark
        fallback, for Flat (which has no sprite compositor at all) and for any
        unit whose sprite does not resolve."""
        return self.set_content((), [(polygons, color)])

    def set_content(self, draws, marks) -> bool:
        """Sprite draws and coloured marks together, for a group ghost whose
        members resolve differently. `marks` is a list of (polygons, color)."""
        self.prepareGeometryChange()
        self._arrays = []
        self._images = []
        self._marks = []
        rect = QRectF()
        for polygons, color in marks:
            qcolor = QColor(*color)
            brush, pen = QBrush(qcolor), QPen(qcolor)
            for points in polygons:
                polygon = QPolygonF([QPointF(px, py) for px, py in points])
                self._marks.append((polygon, brush, pen))
                rect = rect.united(polygon.boundingRect())
        for draw, ax, ay in draws:
            rgba = np.ascontiguousarray(draw.rgba)
            h, w = rgba.shape[:2]
            if h == 0 or w == 0:
                continue
            # The same top-left _clipped_paint_rgba() composites at, and the
            # same RGB channel order the canvas itself is built in -- the
            # first three channels blend straight into an RGB888 canvas.
            x0, y0 = ax - draw.hotspot_x, ay - draw.hotspot_y
            image = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
            self._arrays.append(rgba)
            self._images.append((image, float(x0), float(y0)))
            rect = rect.united(QRectF(x0, y0, w, h))
        self._bounding_rect = rect
        self.update()
        return bool(self._images or self._marks)

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        for polygon, brush, pen in self._marks:
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPolygon(polygon)
        for image, x, y in self._images:
            painter.drawImage(QPointF(x, y), image)


class CameraMarkerItem(QGraphicsItem):
    """One player's starting camera (Map.initial_player_views), View >
    Player Cameras: a camera glyph standing on the tile that player's view
    is set to, filled in the player's own colour and carrying its number.

    Device-space like StackBadgeItem, but through
    ItemIgnoresTransformations rather than a resetTransform() in paint():
    this is one item per player rather than one for the whole layer, so the
    flag's own position-anchored device frame is exactly what is wanted --
    the glyph keeps its size and stays upright through zoom, the Flat squash
    and the isometric rotation, with the ruler label as the precedent.

    The anchor (a scene point handed in by MapView, from its own
    _ruler_anchor) is the glyph's BOTTOM CENTRE, so the marker stands on the
    tile at that tile's own elevation rather than hanging over it.
    """

    GLYPH_W_PX = 20.0
    # An emphasised marker (the player selected in Players mode) is drawn
    # larger as well as outlined lighter: size survives a busy map where a
    # one-pixel outline colour change does not.
    EMPHASIS_SCALE = 1.3
    NUMBER_FONT_PX = 9
    INK = QColor(20, 20, 20)
    EMPHASIS_INK = QColor(255, 255, 255)
    INK_WIDTH = 1.0
    EMPHASIS_INK_WIDTH = 2.0

    def __init__(self, player_id: int, color: QColor, tile: tuple[int, int]) -> None:
        super().__init__()
        self.player_id = int(player_id)
        self._color = QColor(color)
        self._tile = (int(tile[0]), int(tile[1]))
        self._emphasised = False
        self._font = map_overlay_font(self.NUMBER_FONT_PX)
        self._font.setBold(True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        # The marker is chrome, not a target: without this it would swallow
        # a Units-mode click on whatever sits under it.
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setToolTip(f"P{self.player_id} camera view ({self._tile[0]}, {self._tile[1]})")
        self._rebuild()

    # -- state ---------------------------------------------------------

    def set_emphasised(self, emphasised: bool) -> None:
        if emphasised == self._emphasised:
            return
        self._emphasised = emphasised
        self._rebuild()

    def is_emphasised(self) -> bool:
        return self._emphasised

    def tile(self) -> tuple[int, int]:
        return self._tile

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def _rebuild(self) -> None:
        """The glyph, in device pixels around a bottom-centre origin: a
        camera body with a viewfinder bump and a lens, plus a pin down to
        the anchor point. Rebuilt rather than scaled in paint() so the
        bounding rect follows the emphasis size."""
        self.prepareGeometryChange()
        w = self.GLYPH_W_PX * (self.EMPHASIS_SCALE if self._emphasised else 1.0)
        body_h, pin_h, bump_h = 0.58 * w, 0.34 * w, 0.17 * w
        body_top = -(body_h + pin_h)
        body = QRectF(-w / 2.0, body_top, w, body_h)
        path = QPainterPath()
        path.addRoundedRect(body, 0.12 * w, 0.12 * w)
        path.addRect(QRectF(body.left() + 0.16 * w, body_top - bump_h, 0.28 * w, bump_h + 1.0))
        pin = QPolygonF([
            QPointF(-0.17 * w, body.bottom()),
            QPointF(0.17 * w, body.bottom()),
            QPointF(0.0, 0.0),
        ])
        path.addPolygon(pin)
        path.closeSubpath()
        self._path = path.simplified()
        self._lens = QRectF(0.0, 0.0, 0.34 * w, 0.34 * w)
        self._lens.moveCenter(QPointF(body.center().x() - 0.22 * w, body.center().y()))
        self._number_rect = QRectF(body.center().x() + 0.02 * w, body_top, 0.44 * w, body_h)
        pad = self.EMPHASIS_INK_WIDTH + 1.0
        self._bounding_rect = QRectF(
            body.left(), body_top - bump_h, w, body_h + bump_h + pin_h
        ).adjusted(-pad, -pad, pad, pad)
        self.update()

    # -- painting ------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        ink = self.EMPHASIS_INK if self._emphasised else self.INK
        width = self.EMPHASIS_INK_WIDTH if self._emphasised else self.INK_WIDTH
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(ink, width))
        painter.setBrush(QBrush(self._color))
        painter.drawPath(self._path)
        # The lens reads as a lens only against the body, so it takes the
        # body's outline ink at a fixed hairline width rather than the
        # emphasis one, which at 2 px would fill it in.
        painter.setPen(QPen(ink, self.INK_WIDTH))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(self._lens)
        painter.setPen(QPen(_contrast_ink(self._color)))
        painter.setFont(self._font)
        painter.drawText(self._number_rect, Qt.AlignCenter, str(self.player_id))


def _contrast_ink(color: QColor) -> QColor:
    """Black or white, whichever reads on `color` -- the player number sits
    on the player's own fill, which ranges from near-black to yellow."""
    luma = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return QColor(0, 0, 0) if luma > 140 else QColor(255, 255, 255)
