"""MapView's viewport: zoom anchor and stops, preframes, chunk targets, view-state capture and restore, resize and scroll."""

from __future__ import annotations

import math

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QCursor,
    QTransform,
)
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
)

from descape import (
    iso_geometry,
    perf_trace,
    zoom_preframe,
    zoom_stops,
)
from descape.render_cache_common import MIP_SCALE_SLACK

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- render.py imports no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.viewer_canvas import (
    _max_axis_scale,
    level_rect_for,
)


def _scroll_clamp(start: float, low: float, extent: float, size: float) -> float:
    """One axis of QGraphicsView's scroll clamp: the visible span's start,
    centred on a scene narrower than the viewport, else kept inside it."""
    if extent <= size:
        return low - (size - extent) / 2
    return min(max(start, low), low + extent - size)


class _MapViewViewportMixin:
    """MapView's zoom, scroll and view state (map_view.py's docstring has the split's rules)."""

    def set_zoom_anchor_mode(self, centered_on_cursor: bool) -> None:
        # wheelEvent anchors by hand; Qt's AnchorUnderMouse reads a stale point after a pan (GH #146).
        self._zoom_on_cursor = centered_on_cursor
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    def zoom_anchor_scene(self) -> QPointF:
        """The scene point the next wheel zoom is expected to anchor at: the
        last wheel zoom's viewport point while zoom-on-cursor is on (a roll
        keeps the cursor still), else the viewport centre. TASK-031.25's
        finer-neighbour warm places its patch by it."""
        anchor = self._last_zoom_anchor
        if not self._zoom_on_cursor or anchor is None:
            anchor = self.viewport().rect().center()
        return self.mapToScene(anchor)

    # --- pre-rendered zoom frames (TASK-031.71) ---------------------------

    def set_preframes_allowed(self, allowed: bool) -> None:
        """Opts this view in or out of pre-rendered zoom frames; off drops them."""
        self._preframes_allowed = allowed
        if not allowed:
            self._preframe_timer.stop()
            self._preframe_warmer.cancel()
            if self._canvas_item is not None:
                self._canvas_item.drop_preframes()
        self.retarget_preframes()

    def retarget_preframes(self) -> None:
        """Cancels the frame being rendered and restarts the settle wait: the
        view moved, resized or (zoom on cursor) the cursor moved, or the window
        is about to change the scene. Finished frames stay, since the paint
        that follows a zoom is the one that uses them; a stale key or stamp
        only misses."""
        if not self._preframes_allowed:
            return
        self._preframe_warmer.cancel()
        if self._canvas_item is not None:
            self._preframe_timer.start()

    def retry_refused_preframes(self) -> None:
        """A margin or load warm drained: renders the notches the last settle
        refused, whose chunks it may have cached since. Not while a settle is
        pending, which renders both anyway, nor for a hidden view."""
        if self._preframes_allowed and not self._preframe_timer.isActive() and self.isVisible():
            self._preframe_warmer.retry_refused()

    @property
    def preframe_warmer(self) -> zoom_preframe.PreframeWarmer:
        return self._preframe_warmer

    def _on_preframe_settle(self) -> None:
        item = self._canvas_item
        if item is None or not self._preframes_allowed or not self.isVisible():
            return
        if QApplication.mouseButtons():
            self._preframe_timer.start()
            return
        item.drop_preframes()
        self._preframe_warmer.start(item, (self._preframe_dir, -self._preframe_dir), self._preframe_job)

    def _preframe_job(self, direction: int) -> zoom_preframe.FrameJob | None:
        """The FrameJob for the paint one wheel notch in `direction` (+1 in)
        would make now, as a fresh gesture, anchored at the cursor (or the
        viewport centre with zoom on cursor off), or None where that notch
        is refused, the cursor is off the viewport, the viewport is not on
        whole device pixels or the paint would not blit its own level from
        cached chunks alone."""
        item = self._canvas_item
        if item is None or item is not self._preframe_warmer._item:
            return None
        saved = self._wheel_gesture_mip
        self._wheel_gesture_mip = self._mip_for_zoom_factor(1.0)
        try:
            factor = self._wheel_zoom_factor(direction)
        finally:
            self._wheel_gesture_mip = saved
        if factor is None:
            return None
        vp = self.viewport()
        anchor = vp.mapFromGlobal(QCursor.pos()) if self._zoom_on_cursor else vp.rect().center()
        if not vp.rect().contains(anchor):
            return None
        if self._preframe_twin is None:
            self._preframe_twin = zoom_preframe.ZoomTwin(self)
        world, vp_rect = self._preframe_twin.predict(factor, anchor, self._twin_snap(factor))
        dpr = vp.devicePixelRatioF()
        offset = zoom_preframe.viewport_offset(vp)
        if not zoom_preframe.device_aligned(offset, vp_rect.size(), dpr):
            return None
        device = zoom_preframe.device_transform(world, offset, dpr)
        bounding = item.boundingRect()
        rect = zoom_preframe.exposed_scene_rect(world, vp_rect, bounding)
        plan = item.frame_plan(device, rect)
        if plan is None:
            return None
        stamps = item.frame_stamps(plan[0], rect, _max_axis_scale(device))
        if stamps is None:
            return None
        key = zoom_preframe.paint_key(device, rect, vp_rect.size(), offset, self.renderHints(), dpr, bounding)
        viewport_device = zoom_preframe.viewport_device_rect(vp, vp_rect.size())
        return zoom_preframe.FrameJob(item, key, device, rect, plan, viewport_device, self.renderHints(), dpr, stamps)

    def _twin_snap(self, factor: float) -> tuple[QPoint, float] | None:
        """ZoomTwin.predict()'s `snap` for a notch by `factor`: wheelEvent()
        snaps a notch that lands on any level's stop (_snap_to_device_px(),
        TASK-600), so the twin must too, else that notch never hits its frame."""
        if not any(abs(factor / f - 1.0) <= MIP_SCALE_SLACK for f in self._level_stop_factors()):
            return None
        return zoom_preframe.viewport_offset(self.viewport()), self.devicePixelRatioF()

    def _scroll_by(self, dx: int, dy: int) -> None:
        """The one place that moves the scrollbars. Shared by middle-drag and
        the held-key pan, so both get setSceneRect()'s overscroll clamping and
        zoom-relative feel identically."""
        zoom_preframe.scroll_view_by(self, dx, dy)

    def center_on_tile(self, tile_x: int, tile_y: int) -> None:
        """Scroll so tile (tile_x, tile_y) is centred, in every terrain style
        (routed through _tile_polygon for exactly that reason)."""
        poly = self._tile_polygon(tile_x, tile_y)
        if poly is not None:
            self.centerOn(poly.boundingRect().center())

    def view_device_scale(self) -> float:
        """The max-axis device scale a paint of the canvas item reads now
        (viewport_chunk_target()'s reading), or 0.0 with no item."""
        item = self._canvas_item
        if item is None:
            return 0.0
        return _max_axis_scale(item.deviceTransform(self.viewportTransform())) * self.devicePixelRatioF()

    def refresh_canvas_dims(self) -> None:
        """Re-syncs the canvas item's boundingRect AND this view's own scene
        rect/overscroll margin to the cache's CURRENT canvas_dims() -- Track
        P3-g6. Needed only when a cache's canvas_dims() answer can change
        after set_source() already ran; today that is SlopedChunkCache
        toggling sprites at a low elev_step_pct stop (see its canvas_dims()
        docstring). Every other caller's canvas_dims() never moves once
        set_source() has run, so nothing before this needed a resync path --
        invalidate_region() alone was always enough. Skipping this after
        such a toggle leaves the newly-widened strip composited in the
        cache but never reaching the screen: MapCanvasItem.boundingRect()
        and this view's own setSceneRect() both stay at their set_source()-
        time size regardless of what the cache reports afterwards."""
        if self._canvas_item is None:
            return
        self._canvas_item.refresh_canvas_dims()
        w, h = self._canvas_item._cache.canvas_dims()
        self._map_rect = QRectF(0, 0, w, h)
        margin_x = w * self.OVERSCROLL_FRACTION
        margin_y = h * self.OVERSCROLL_FRACTION
        self.setSceneRect(self._map_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))

    def viewport_chunk_target(self) -> tuple[int, int, int, int, int] | None:
        """(mip, cx0, cy0, cx1, cy1) -- the mip a real paint would select
        right now, and the inclusive chunk-index range the current viewport
        covers at it -- or None with no canvas item or a degenerate
        viewport. The paint-free equivalent of what MapCanvasItem.paint()
        knows: A2's hook for driving a margin warm (descape.margin_warm)
        off the viewport between paints, since option.exposedRect only
        exists inside an actual Qt paint call.

        Mip selection mirrors MapCanvasItem._select_mip() exactly but reads
        the ITEM's deviceTransform(viewportTransform) rather than a live
        QPainter's deviceTransform() (there is none here).
        QGraphicsItem.deviceTransform(viewportTransform) is used rather
        than self.viewportTransform() alone so a future item-level
        transform can't silently desynchronise this reading from what a
        real paint selects; multiplying by devicePixelRatioF() folds in
        DPR the same way ViewerWindow._start_level_warm already does, for
        the same reason -- scaling a transform by a uniform `s` scales its
        singular values by `s`, so this is exact, not an approximation.

        Scene rect: the viewport mapped to scene space via
        mapToScene(...).boundingRect(), intersected with the item's own
        boundingRect(). Under Flat's rotate+squash view transform, the
        bounding box of that mapped polygon is precisely the shape a real
        option.exposedRect already is.

        Correctness is pinned by a test, not by inspection (tests/
        test_margin_warm.py): after a real paint cycle, this method's mip
        must equal MapCanvasItem._last_mip, and its chunk range must
        contain every chunk that paint's own exposed rect resolved to."""
        item = self._canvas_item
        if item is None:
            return None
        device_transform = item.deviceTransform(self.viewportTransform())
        scale = _max_axis_scale(device_transform) * self.devicePixelRatioF()
        cache = item._cache
        mip = cache.mip_for_scale(scale)
        scene_rect = self._current_scene_rect()
        if scene_rect is None:
            return None
        level_rect = level_rect_for(cache, mip, scene_rect)
        if level_rect is None:
            return None
        x0, y0, x1, y1 = level_rect
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, x0, y0, x1, y1)
        return mip, cx0, cy0, cx1, cy1

    def _current_scene_rect(self) -> QRectF | None:
        """The scene-space rect the viewport currently covers, intersected
        with the canvas item's own boundingRect() -- the mip-independent
        half of viewport_chunk_target()'s projection, shared with
        viewport_chunk_target_at() below (2026-09-07 plan's load-time
        margin warm, Step 1) so there is only one copy of this
        mapToScene(...).boundingRect().intersected(...) line. None with no
        canvas item or a degenerate (empty) viewport."""
        item = self._canvas_item
        if item is None:
            return None
        rect = self.mapToScene(self.viewport().rect()).boundingRect().intersected(item.boundingRect())
        return None if rect.isEmpty() else rect

    def viewport_chunk_target_at(self, mip: int) -> tuple[int, int, int, int] | None:
        """(cx0, cy0, cx1, cy1) -- the chunk range level `mip` would show if
        the viewport were looking at it right now, projecting the SAME
        on-screen scene rect viewport_chunk_target() uses for the live mip
        (2026-09-07 plan's load-time margin warm, Step 1). Unlike that
        method, this never reads the device transform or picks a mip
        itself -- `mip` is the caller's, so this is safe to call for a
        level the viewport isn't actually showing (a neighbour level
        that hasn't been visited yet).

        None with no canvas item or a degenerate/empty projection at that
        mip. `mip` itself must be one this cache's ladder actually
        enumerates (mip_scale()/canvas_dims() raise KeyError otherwise) --
        every real caller gets `mip` from level_warm.neighbour_mips(),
        which already clamps to the enumerated ladder, so this is never
        asked about a mip outside it."""
        item = self._canvas_item
        if item is None:
            return None
        cache = item._cache
        scene_rect = self._current_scene_rect()
        if scene_rect is None:
            return None
        level_rect = level_rect_for(cache, mip, scene_rect)
        if level_rect is None:
            return None
        x0, y0, x1, y1 = level_rect
        return cache.chunk_index_range(mip, x0, y0, x1, y1)

    def zoomed_scene_rect(self, factor: float, anchor: QPointF) -> QRectF:
        """The scene rect the viewport would show after a wheel zoom by
        `factor` about scene point `anchor` (zoom_preframe.zoom_about()),
        unclipped. The scrollbars' clamp is modelled where Qt applies it: in
        the transformed space, against the mapped sceneRect(), centred when
        the scene is narrower (AlignCenter). Sized at maximumViewportSize(),
        so a scrollbar the zoom hides cannot widen the view past it."""
        size = self.maximumViewportSize()
        width, height = size.width(), size.height()
        scaled = QTransform(self.transform())
        scaled.scale(factor, factor)
        origin = scaled.map(anchor) - self.viewportTransform().map(anchor)
        bounds = scaled.mapRect(self.sceneRect())
        x = _scroll_clamp(origin.x(), bounds.left(), bounds.width(), width)
        y = _scroll_clamp(origin.y(), bounds.top(), bounds.height(), height)
        inverse, _ok = scaled.inverted()
        return inverse.mapRect(QRectF(x, y, width, height))

    def viewport_reach_target(self, mip: int) -> tuple[int, int, int, int] | None:
        """(cx0, cy0, cx1, cy1), the zoom-out reach of level `mip` (TASK-031.11
        D1): the chunks a paint reads (MapCanvasItem.view_chunks()) at the
        widest view `mip` shows, a zoom about zoom_anchor_scene() down to the
        device scale where the next coarser level takes over. The current
        view for a level finer than the view's. None for the coarsest level
        (the fit paint composites all of it), with no item or an empty rect."""
        item = self._canvas_item
        if item is None:
            return None
        cache = item._cache
        levels = cache.mip_levels()
        if mip not in levels or mip == levels[0]:
            return None
        scale = self.view_device_scale()
        if scale <= 0:
            return None
        widest = 1.0 / cache.mip_scale(mip - 1)
        rect = self.zoomed_scene_rect(min(1.0, widest / scale), self.zoom_anchor_scene())
        rect = rect.intersected(item.boundingRect())
        if rect.isEmpty():
            return None
        return item.view_chunks(mip, rect, widest)

    def viewport_chunk_span(self, cache, mip: int) -> tuple[int, int]:
        """(chunks_wide, chunks_tall) -- how many chunks a REAL viewport at
        `mip` would cover, sized from THIS view's own on-screen dimensions
        (device pixels) rather than from whatever scene-space rect happens
        to be visible right now. At least 1x1.

        This is deliberately independent of the live zoom/mip:
        viewport_chunk_target_at()'s raw projection reuses the CURRENT
        scene rect verbatim, which at a fit-to-view zoom (every
        load_scenario() open) is the whole map -- margin_warm.
        bounded_chunk_range() is what turns that raw projection plus this
        span into an actual viewport-sized patch; see its own docstring
        for why the two are split rather than done in one method here."""
        dpr = self.devicePixelRatioF()
        rect = self.viewport().rect()
        chunks_w = max(1, math.ceil(rect.width() * dpr / cache.chunk_px))
        chunks_h = max(1, math.ceil(rect.height() * dpr / cache.chunk_px))
        return chunks_w, chunks_h

    def _note_viewport_changed(self) -> None:
        """Starts the viewport-changed poll if it isn't already running --
        NEVER restarts an already-active one. That restart guard is the
        entire mechanism a restart-on-every-move debounce would be missing:
        see VIEWPORT_POLL_MS's own comment and _on_viewport_poll_tick() for
        why a restarted timer would starve during a continuous drag.

        Called from every path that can change what viewport_chunk_target()
        would return: scrollContentsBy (every pan source at once -- left
        ScrollHandDrag, middle-drag's direct scrollbar writes, keyboard
        scrolling, centerOn, and the scrollbar shifts a scale() call
        induces), wheelEvent (a zoom whose scrollbars happen not to move
        still changes the mip), _capture_zoom_baseline (covers
        resizeEvent), and set_source()/clear_image() (a new document).

        Also the funnel for the region ants' on-screen gate, since that is
        the same set of paths that changes what is visible. That includes
        the resize on show(), which is what starts the ants for a region set
        while the window had no real viewport yet. And for the pre-rendered
        zoom frames' settle wait (retarget_preframes())."""
        self._sync_region_ant_timer()
        self.retarget_preframes()
        if not self._viewport_poll_timer.isActive():
            self._viewport_poll_timer.start()

    def _on_viewport_poll_tick(self) -> None:
        """One poll fire: notify on_viewport_changed() if the target has
        moved since the last fire, else stop -- the "self-stopping poll,
        not a restart-on-change debounce" shape A2.4 spends real space
        justifying. A REPEATING timer (not single-shot) firing at a bounded
        cadence throughout a drag, delivering the final target one fire
        after motion ends, then switching itself off: no trailing-edge
        special case, and a static view costs nothing once it stops."""
        target = self.viewport_chunk_target()
        if target == self._last_viewport_target:
            self._viewport_poll_timer.stop()
            return
        self._last_viewport_target = target
        self.on_viewport_changed()

    def stop_viewport_poll(self) -> None:
        """Drops a pending poll fire: the owning window is closing, and a fire
        after that would re-arm its warms on a shared event loop. The
        pre-rendered zoom frames' settle wait and warm stop with it."""
        self._viewport_poll_timer.stop()
        self._preframe_timer.stop()
        self._preframe_warmer.cancel()

    def _capture_view_state(self) -> tuple[float, QPointF] | None:
        """Snapshot of the current view, as a zoom multiple of the CURRENT
        fit-to-view baseline plus the viewport's centered scene point
        expressed as a FRACTION of the current _map_rect, for
        set_isometric() to restore instead of resetting to fit -- see
        set_source()'s reset_view parameter. Both readings are relative
        (fit multiple, map-rect fraction), not absolute (scale, scene
        point): reset_view=False callers include a graphics-quality change,
        which rescales tile_pixels and therefore the WHOLE scene coordinate
        system (a smaller/larger map_rect in the same map, still 0..1
        across it) -- an absolute center point captured before that rescale
        would land somewhere else on the new, differently-scaled map.

        None when there's no established fit yet (nothing on screen to be
        relative to) -- reset_view=False callers only ever call this while a
        map is already showing, so this is a defensive fallback, not an
        expected path.

        The centre is the exact float one (_viewport_center_scene()), and
        _restore_view() puts it back with _center_on_exact(), so a
        same-geometry re-render round-trips to the same scrollbar values
        (TASK-386: QRect.center() and centerOn()'s truncation used to creep
        the view 1-2 px up-left per re-render)."""
        fit = self._fit_baseline_scale()
        if not fit or self._map_rect is None:
            return None
        if self._map_rect.width() <= 0 or self._map_rect.height() <= 0:
            return None
        current_scale = abs(self.transform().determinant()) ** 0.5
        center = self._viewport_center_scene()
        frac = QPointF(
            (center.x() - self._map_rect.left()) / self._map_rect.width(),
            (center.y() - self._map_rect.top()) / self._map_rect.height(),
        )
        return current_scale / fit, frac

    def _restore_view(self, restore: tuple[float, QPointF]) -> None:
        """Applies a captured _capture_view_state() reading on top of the
        transform set_isometric() has already established at the call site
        (squash+rotate for isometric Flat, untouched/identity otherwise),
        instead of that method's usual fit-to-view. Falls back to leaving
        the current transform alone if no fit baseline can be computed (e.g.
        a zero-size viewport) -- same "nothing sensible to do" case
        _capture_view_state() itself declines to capture."""
        relative_scale, frac = restore
        new_fit = self._fit_baseline_scale()
        if new_fit is None or self._map_rect is None:
            return
        target = relative_scale * new_fit
        if self._terrain_style in _ELEVATED_STYLES or not self._isometric:
            self.scale(target, target)
        else:
            k = target / math.sqrt(self.ISO_VERTICAL_SQUASH)
            self.scale(k, k)
        center = QPointF(
            self._map_rect.left() + frac.x() * self._map_rect.width(),
            self._map_rect.top() + frac.y() * self._map_rect.height(),
        )
        self._center_on_exact(center)

    def _viewport_center_scene(self) -> QPointF:
        """The scene point under the viewport's true centre (w/2, h/2), as a
        float: QRect.center() is ((w-1)//2, (h-1)//2), half a pixel up-left."""
        viewport = self.viewport()
        inverse, _ok = self.viewportTransform().inverted()
        return inverse.map(QPointF(viewport.width() / 2.0, viewport.height() / 2.0))

    def _center_on_exact(self, point: QPointF) -> None:
        """centerOn(), then each scrollbar corrected by the rounded offset of
        `point` from the viewport's true centre (centerOn() truncates). The
        scrollbars clamp at their limits; a zero range makes it a no-op."""
        self.centerOn(point)
        viewport = self.viewport()
        mapped = self.viewportTransform().map(point)
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() + round(mapped.x() - viewport.width() / 2.0))
        v.setValue(v.value() + round(mapped.y() - viewport.height() / 2.0))

    def set_isometric(self, enabled: bool) -> None:
        self._isometric = enabled
        # GH #100: Flat's badge anchors differ between square and iso views.
        self._rebuild_stack_badges()
        self.resetTransform()
        restore = self._pending_view_restore
        self._pending_view_restore = None
        # Stepped's (and Sloped's) projection is already baked into the
        # pixels -- this whole method (below this point) is Flat mode's own
        # QTransform trick (see the class comment above) and has nothing
        # left to do for a Stepped/Sloped image beyond the same plain
        # "whole canvas fits the viewport" fit Flat's own "unchecked" case
        # already uses. ViewerWindow greys out the View > Isometric View
        # checkbox whenever Terrain Style != Flat, so this method only ever
        # reaches the enabled=True rotate/squash path below while Flat is
        # active.
        if self._terrain_style in _ELEVATED_STYLES or not enabled:
            if self._map_rect is not None:
                if restore is not None:
                    self._restore_view(restore)
                else:
                    self.fitInView(self._map_rect, Qt.KeepAspectRatio)
                self._capture_zoom_baseline()
            return

        # Order matters: QTransform.scale() called before rotate() applies the
        # squash to already-rotated (diamond) coordinates, giving the correct
        # 2:1 bounding box; the other order gives a square bounding box instead
        # (verified directly against QTransform.mapRect()).
        self.scale(1.0, self.ISO_VERTICAL_SQUASH)
        self.rotate(self.ISO_ROTATION_DEGREES)

        if self._map_rect is None:
            return
        if restore is not None:
            self._restore_view(restore)
            self._capture_zoom_baseline()
            return
        # fitInView() is intentionally not used here: its internal normalization
        # step (dividing out the transformed unit square's own bounding box)
        # interacts badly with an anisotropic transform and silently squashes
        # the 2:1 diamond back down to ~1:1 (verified empirically). Computing
        # the fit scale by hand and applying it as one final *uniform* scale
        # sidesteps that -- a uniform scale can't introduce shear, so it can't
        # distort the ratio no matter how Qt composes it with what came before.
        mapped_rect = self.transform().mapRect(self._map_rect)
        viewport_rect = self.viewport().rect()
        if mapped_rect.width() <= 0 or mapped_rect.height() <= 0:
            return
        fit_scale = self.ISO_FIT_MARGIN * min(
            viewport_rect.width() / mapped_rect.width(),
            viewport_rect.height() / mapped_rect.height(),
        )
        self.scale(fit_scale, fit_scale)
        self.centerOn(self._map_rect.center())
        self._capture_zoom_baseline()

    def _coarsest_mip_scale(self) -> float:
        """How many scene units one pixel of the COARSEST enumerated mip
        level covers -- i.e. how much coarser than the reference the level
        set actually goes. Always >= 1.0, and exactly 1.0 both when no
        canvas item exists yet and when the level set enumerates nothing
        below the reference (a normal case, e.g. elev_step_pct=10, where it
        makes _capture_zoom_baseline's floor identical to the pre-mip
        one)."""
        if self._canvas_item is None:
            return 1.0
        cache = self._canvas_item._cache
        return cache.mip_scale(cache.mip_levels()[0])

    def _finest_mip_scale(self) -> float:
        """Mirrors _coarsest_mip_scale() at the other end of the ladder: how
        many scene units one pixel of the FINEST enumerated mip level covers
        -- i.e. how much finer than the reference the level set actually
        goes. Always <= 1.0, and exactly 1.0 both when no canvas item exists
        yet and when the level set enumerates nothing above the reference."""
        if self._canvas_item is None:
            return 1.0
        cache = self._canvas_item._cache
        return cache.mip_scale(cache.mip_levels()[-1])

    def _fit_baseline_scale(self) -> float | None:
        """The sqrt-determinant scale this view WOULD have at fit-to-view,
        derived from geometry rather than from the current transform.

        Returns None when no source is set: with no map there is no fit, so
        the quantity this computes does not exist to be computed.

        The transform reading this replaces was only ever a fit baseline
        because both of _capture_zoom_baseline's callers ran immediately
        after a fit. Recomputing on resize needs a form that doesn't depend
        on that -- otherwise a resize taken while zoomed to 3x fit re-anchors
        the bounds to that zoom and strands the user above fit, unable to
        zoom back out to the whole map.
        """
        if self._map_rect is None:
            return None
        viewport_rect = self.viewport().rect()
        if viewport_rect.width() <= 0 or viewport_rect.height() <= 0:
            return None
        if self._map_rect.width() <= 0 or self._map_rect.height() <= 0:
            return None
        if self._terrain_style in _ELEVATED_STYLES or not self._isometric:
            # fitInView's own arithmetic, including its hardcoded 2 px per side
            # (measured against the real transform, three aspect ratios).
            return min(
                (viewport_rect.width() - 4) / self._map_rect.width(),
                (viewport_rect.height() - 4) / self._map_rect.height(),
            )
        # Flat + isometric: squash, rotate, then one uniform fit scale -- see
        # set_isometric(). Rotation's determinant is 1, so the baseline is that
        # uniform scale times sqrt(ISO_VERTICAL_SQUASH).
        iso = QTransform()
        iso.scale(1.0, self.ISO_VERTICAL_SQUASH)
        iso.rotate(self.ISO_ROTATION_DEGREES)
        mapped_rect = iso.mapRect(self._map_rect)
        if mapped_rect.width() <= 0 or mapped_rect.height() <= 0:
            return None
        fit_scale = self.ISO_FIT_MARGIN * min(
            viewport_rect.width() / mapped_rect.width(),
            viewport_rect.height() / mapped_rect.height(),
        )
        return fit_scale * math.sqrt(self.ISO_VERTICAL_SQUASH)

    def zoom_percent_of_fit(self) -> float | None:
        """Current zoom as a percentage of fit-to-view -- the same quantity
        wheelEvent's ceiling/floor checks compare against, times 100. Backs
        the status-bar readout; reuses both expressions rather than
        re-deriving either, since the readout is worthless for calibrating
        MAX_ZOOM_MULTIPLE_OF_FIT if its arithmetic can drift from the clamp's.

        None means "no map loaded" (_fit_baseline_scale's own None case),
        not the identity-transform fallback _capture_zoom_baseline uses --
        reporting a fabricated 100% for an empty view would be wrong.
        """
        fit = self._fit_baseline_scale()
        if fit is None or fit <= 0:
            return None
        current = abs(self.transform().determinant()) ** 0.5
        return current / fit * 100.0

    def _capture_zoom_baseline(self) -> None:
        # Real per-tile texture detail (see render.py) is high-frequency
        # enough that zooming out past a few times the "whole map fits in
        # view" scale minifies it beyond what QPainter.SmoothPixmapTransform's
        # plain bilinear filtering can represent without mipmapping -- the
        # result is visible moire/banding artifacts, not a rendering bug in
        # the usual sense. Zooming in past a similar multiple just upscales
        # that same finite per-tile texture resolution further, revealing no
        # more real detail, only blur/blockiness. Rather than fix either with
        # real mipmapping (a much bigger rendering change), wheelEvent() just
        # refuses to zoom past floor/ceiling values relative to this
        # fit-to-view baseline -- which is also just more usable regardless
        # (nothing useful past "whole map visible" in one direction or "one
        # tile fills the view" in the other).
        # determinant() gives the transform's area scale factor independent
        # of rotation, so this stays valid even under the isometric
        # transform's rotation + anisotropic squash.
        #
        # Phase B-D-d: mipmapping now exists below the reference level, so
        # the zoom-out floor gets divided by exactly how much coarser the
        # ladder goes. The property that buys, stated so it can't drift into
        # an overclaim: at the floor the residual minification is UNCHANGED
        # from the pre-mip one (the floor scales by 1/S while the level
        # painted there is S times coarser), and the reachable zoom-out range
        # grows by exactly the mip depth. Both halves are one line of algebra
        # and neither depends on window size. At a fixed scale the coarser
        # level does alias S times less, but that is not what this floor is
        # claiming -- an earlier version of this comparison got the
        # direction backwards.
        #
        # 2026-08-27: the zoom-in ceiling gets the same treatment, divided by
        # _finest_mip_scale() (<= 1.0, so dividing raises the ceiling). B-D-d
        # left this half alone on the theory that mip levels finer than the
        # reference were a pure detail win with no reachability question --
        # wrong in practice: MAX_ZOOM_MULTIPLE_OF_FIT was calibrated against
        # the pre-mip reference tile size, and on real maps that ceiling sat
        # BELOW the scale needed to ever select the finest enumerated level,
        # so the sharper level the mip track added was dead code behind the
        # wheel-zoom limit (measured: reachable at fit-multiple ~48, capped
        # at 40). Same algebra as the floor, mirrored: at the new ceiling the
        # residual magnification is unchanged from the pre-mip one, and the
        # reachable zoom-in range grows by exactly the mip depth.
        #
        # 2026-08-28: mip_for_scale's rule flipped from floor(log2(scale)) to
        # ceil, which does not change either formula below -- both derivations
        # above only used mip_scale()/_coarsest_mip_scale()/_finest_mip_scale(),
        # never the selection rule's rounding direction. What it does change:
        # the floor's clamp-binding and residual-unchanged property (the
        # paragraph above this one) survives verbatim; the ceiling's
        # reachability property survives too and gets slightly MORE robust
        # (the finest level now engages at half the scale it used to, so
        # "reachable at fit-multiple ~48" above is now reachable at ~24). Net
        # effect on the two clamped-beyond-the-ladder bands: the over-
        # minification band below the floor shrinks by one octave (less blur
        # at heavy zoom-out) and the over-magnification band above the
        # ceiling grows by one octave (one more octave of zoom-in past native
        # density before the ceiling has to intervene). Deliberately not
        # compensated further -- same reasoning as the note above about why
        # this isn't an absolute bound.
        #
        # Fit-RELATIVE rather than an absolute scale bound, which is what the
        # plan originally specified: an absolute floor has no notion of how
        # big the canvas is in the window, so it lands above fit-to-view on a
        # big map (measured, 480x480 at a 480px window: fit is scale 0.0413,
        # an absolute floor of 1/(4 * 4) = 0.0625 would refuse zoom-out
        # everywhere and strand the user above fit). mip_scale >= 1 always,
        # so dividing by it is never tighter than the pre-mip floor, and a
        # level set with nothing below the reference (elev_step_pct=10
        # enumerates {0, 1}) reduces it to exactly the pre-mip expression
        # rather than needing a special case.
        #
        # 2026-08-30: the widened value is now capped back to the pre-mip
        # MIN_ZOOM_FRACTION_OF_FIT/MAX_ZOOM_MULTIPLE_OF_FIT bound with
        # max()/min() below, reversing the "never tighter than the pre-mip
        # bound" property above on user request -- a fixed 50%-6400%-of-fit
        # range was wanted more than always being able to wheel-zoom to the
        # sharpest/coarsest enumerated mip level. Concretely, on Flat/Stepped's
        # typical 4-level ladder this pulled the reachable range in from
        # 12.5%-12800% (measured on the 120x120 fixture) back to 50%-6400%.
        # The mip-scale division above still runs and stays correct on its
        # own terms -- it is what this clamp is capping -- so the two
        # invariants coexist rather than one being deleted in favour of the
        # other; only the enumerated-level-reachability guarantee is gone.
        #
        # resizeEvent below re-runs this on every viewport size change, so
        # neither bound goes stale after a resize any more. That is only safe
        # because the baseline comes from _fit_baseline_scale() -- geometry --
        # rather than from the current transform: reading the transform while
        # the user is zoomed in would re-anchor both bounds to that zoom.
        fit = self._fit_baseline_scale()
        # No map set means no fit scale exists to derive, and the current
        # transform is the only reading available. Not a compatibility shim:
        # it is the defined behaviour for a view with no source. resizeEvent
        # reaches this before any scenario is loaded, where it sets the bounds
        # from the identity transform (0.5 and 64) rather than leaving them
        # None -- a no-op either way, since there is nothing on screen to zoom,
        # and set_isometric() recomputes both the moment a source arrives.
        baseline = fit if fit is not None else abs(self.transform().determinant()) ** 0.5
        raw_min = self.MIN_ZOOM_FRACTION_OF_FIT * baseline
        raw_max = self.MAX_ZOOM_MULTIPLE_OF_FIT * baseline
        self._min_linear_scale = max(raw_min / self._coarsest_mip_scale(), raw_min)
        self._max_linear_scale = min(raw_max / self._finest_mip_scale(), raw_max)
        # TASK-599, TASK-600: never below the finest level's stop, which 64x fit misses on a big map in a small window.
        finest = self._level_stop_factors()
        if finest:
            self._max_linear_scale = max(self._max_linear_scale, abs(self.transform().determinant()) ** 0.5 * finest[-1])
        self._wheel_lattice_cache = None
        self._repad_edge_ticks()
        # Covers every path that reaches here: set_isometric()/set_source()
        # resetting the transform (readout returns to ~100%), and resizeEvent
        # -- which changes the percentage with no zoom action at all, since
        # _fit_baseline_scale() reads the viewport rect.
        self._on_zoom_changed()
        # Same coverage argument for the viewport-changed poll: resizeEvent
        # changes what viewport_chunk_target() would return with no scroll or
        # zoom action of its own, so it needs its own fire here rather than
        # relying on scrollContentsBy/wheelEvent.
        self._note_viewport_changed()

    def viewport_centre_tile(self) -> tuple[int, int] | None:
        """The tile at the centre of what is on screen, for Players mode's
        Set View button. None when that centre is not on the map.

        Flat's own division is always defined (it does no bounds check at
        all), so the bounds check happens here. Stepped and Sloped return
        None off-map and, in Stepped, on a skirt face -- so those fall back
        to the continuous _pick_map_point() solve before giving up."""
        if self._map_width is None or self._map_height is None:
            return None
        pos = self.mapToScene(self.viewport().rect().center())
        tile = self._pick_tile(pos)
        if tile is None:
            point = self._pick_map_point(pos)
            if point is None:
                return None
            tile = (math.floor(point[0]), math.floor(point[1]))
        x, y = tile
        if not (0 <= x < self._map_width and 0 <= y < self._map_height):
            return None
        return x, y

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recompute the bounds, but deliberately no re-fit: that would fight
        # the zoom and pan the user chose. The new log-splitter handle makes a
        # viewport resize a one-drag action rather than a window-resize-only
        # one, which is what made the stale bounds worth fixing.
        self._capture_zoom_baseline()
        self._instruction_overlay.place(self.viewport().geometry())

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """The one Qt override that catches every pan source at once: left
        ScrollHandDrag, middle-drag's direct scrollbar writes (see
        mouseMoveEvent), keyboard scrolling, centerOn(), and the scrollbar
        shifts a scale() call induces -- all of them move the scrollbars,
        and this is what Qt calls whenever they do. A pair of scrollbar
        valueChanged connections would be equivalent but two objects
        instead of one."""
        super().scrollContentsBy(dx, dy)
        self._note_viewport_changed()

    def _mip_for_zoom_factor(self, factor: float) -> int | None:
        """The mip level a paint would select if this view were scaled by
        `factor` right now; None with no canvas item.

        Same reading as viewport_chunk_target(), and exact rather than
        approximate for a uniform factor: scaling a transform by `s` scales
        its singular values by `s`, which is the same argument that method
        already makes for folding in devicePixelRatioF()."""
        item = self._canvas_item
        if item is None:
            return None
        device_transform = item.deviceTransform(self.viewportTransform())
        scale = _max_axis_scale(device_transform) * self.devicePixelRatioF()
        return item._cache.mip_for_scale(scale * factor)

    def level_zoom_factor(self, level: int) -> float | None:
        """The zoom factor that draws `level` at exactly one device px per
        level px (its stop, TASK-600), DPR folded in as _mip_for_zoom_factor()
        does. None with no canvas item or a level the ladder lacks."""
        item = self._canvas_item
        if item is None or level not in item._cache.mip_levels():
            return None
        scale = self.view_device_scale()
        if scale <= 0:
            return None
        return 1.0 / item._cache.mip_scale(level) / scale

    def native_zoom_factor(self) -> float | None:
        """level_zoom_factor() of the native sprite level (TASK-599: one tile
        spans iso_geometry.NATIVE_TILE_PIXELS device px, so x1 sprites are
        1:1). None where the ladder has no such level: no canvas item, Potato
        and below, top-down Flat."""
        item = self._canvas_item
        if item is None:
            return None
        cache = item._cache
        native = next(
            (lvl for lvl in cache.mip_levels() if cache.mip_tile_px(lvl) == iso_geometry.NATIVE_TILE_PIXELS), None
        )
        return None if native is None else self.level_zoom_factor(native)

    def _level_stop_factors(self) -> list[float]:
        """level_zoom_factor() of every level in the ladder, coarsest first."""
        item = self._canvas_item
        if item is None:
            return []
        factors = (self.level_zoom_factor(level) for level in item._cache.mip_levels())
        return [f for f in factors if f is not None]

    def at_level_stop(self) -> bool:
        """Whether the view sits on some level's stop: within mip_for_scale()'s
        own slack of it, so the level it picks there is that one, drawn 1:1."""
        return any(abs(f - 1.0) <= MIP_SCALE_SLACK for f in self._level_stop_factors())

    def at_native_stop(self) -> bool:
        """Whether the view sits on the native stop (View > Zoom to Native
        Sprite Size's target), with at_level_stop()'s slack."""
        native = self.native_zoom_factor()
        return native is not None and abs(native - 1.0) <= MIP_SCALE_SLACK

    def _wheel_lattice(self) -> tuple[list[float], float] | None:
        """(the notch lattice in device scale, device scale per transform
        scale), TASK-600's zoom_stops lattice over this ladder's level stops
        and the floor/ceiling. None with no canvas item. The ratio is constant
        under a uniform zoom (an iso Flat squash and the DPR folded in)."""
        item = self._canvas_item
        device = self.view_device_scale()
        current = abs(self.transform().determinant()) ** 0.5
        if item is None or device <= 0 or current <= 0:
            return None
        per = device / current
        cache = item._cache
        ratios = tuple(1.0 / cache.mip_scale(level) for level in cache.mip_levels())
        lo = per * self._min_linear_scale if self._min_linear_scale is not None else min(ratios) / 2.0**8
        hi = per * self._max_linear_scale if self._max_linear_scale is not None else max(ratios) * 2.0**8
        # Rounded: `per` differs by ulps between calls, which must not miss the cache.
        key = (ratios, float(f"{lo:.12g}"), float(f"{hi:.12g}"))
        if self._wheel_lattice_cache is None or self._wheel_lattice_cache[0] != key:
            self._wheel_lattice_cache = (key, zoom_stops.lattice(zoom_stops.level_stops(ratios), lo, hi))
        return self._wheel_lattice_cache[1], per

    def zoom_to_native(self) -> bool:
        """View > Zoom to Native Sprite Size (TASK-599): zooms about the
        viewport centre onto the native stop. False, and nothing changes,
        where the ladder has no native level."""
        factor = self.native_zoom_factor()
        if factor is None:
            return False
        self._end_wheel_gesture()
        with perf_trace.op("zoom"):
            zoom_preframe.zoom_about(self, factor, self.viewport().rect().center())
            self._snap_to_device_px()
            self._preframe_dir = 1 if factor > 1.0 else -1
            self._note_viewport_changed()
            self._on_zoom_changed()
        return True

    def _snap_to_device_px(self) -> None:
        """Keeps a level stop pixel-exact: zoom_preframe.snap_to_device_px()
        with this viewport's offset in its window. A later pan at a fractional
        DPR is not re-snapped."""
        if self._canvas_item is None:
            return
        offset = zoom_preframe.viewport_offset(self.viewport())
        zoom_preframe.snap_to_device_px(self, offset, self.devicePixelRatioF())
