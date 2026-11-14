"""ViewerWindow's render pipeline: dirty patches, warms, placeholders, reach, viewport changes, sprite
prefetch, _render_current and refresh_map."""

from __future__ import annotations

import math
import time

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
)

from descape import (
    asset_source,
    debug_log,
    level_warm,
    margin_warm,
    perf_trace,
    settings,
    sprite_prefetch,
    unit_sprites,
)
from descape.render_bbox import dirty_screen_bbox_iso, dirty_screen_bbox_sloped
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)
from descape.render_iso import elevations_and_proj
from descape.render_paint import flat_dirty_rects, tile_pixels_for_map
from descape.render_sloped import sloped_elevations_and_proj

# How long a placeholder may wait with no warm active before the watchdog settles it.
PLACEHOLDER_STUCK_MS = 1500


# Above this many dirty tiles, Stepped mode's own _apply_dirty() path (per-
# tile dilation into a Python set, then one patch() over what ends up being
# effectively the whole canvas -- see dirty_screen_bbox_iso's own docstring)
# is slower than just re-rendering: measured with tools/bench_fill_latency.py
# against a full-map Paint Can fill (this project's own worst case for dirty
# tile count) -- 14,400 tiles (blank_120x120) stayed under the dilation path
# at ~105ms, but 230,400 tiles (blank_480x480) cost ~2.3s there versus ~35ms
# for a full _render_current() (Phase B-C's lazy chunk cache means "full
# re-render" no longer means "recomposite every pixel up front" -- see that
# method's own docstring). Set well above the fine case and well below the
# slow one; the exact crossover between them hasn't been measured. Past it the
# cache is now refreshed in place (TASK-031.61 A5), not rebuilt. Top-down Flat
# reads it as a pre-check too: past it the dirty set is never dilated into
# rects, and the cached region under its bbox is evicted (_evict_flat_region()).
STEPPED_FULL_RERENDER_THRESHOLD = 20_000


# Top-down Flat patches a terrain edit's rects at the visible level only, and evicts their bbox instead once
# they cover more tiles than this multiple of the tiles an eviction recomposites: the visible-level chunks
# under the bbox that the viewport shows. tools/bench_stroke_end.py --flat-ratio.
_FLAT_EVICT_RATIO = 1.0


def _rects_bbox(rects) -> tuple[int, int, int, int]:
    return min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)


class _ViewerRenderMixin:
    """ViewerWindow's render pipeline (viewer.py's docstring has the split's rules)."""

    def _apply_dirty(self, dirty_indices, deferred: list | None = None) -> bool:
        """Repaints exactly the given tile indices -- the incremental path
        used by strokes (above) and undo/redo, instead of a full
        refresh_map() (which reallocates and recomposites the entire map;
        fine once per file open, far too slow per brush touch).

        Stepped mode (Phase B-C, on top of Phase 4/5's terrain+unit
        interleaving): dirty_screen_bbox_iso() plays refresh_region_iso()'s
        old "half 1" role -- mutates self._iso_elevations in place and
        returns the invalidated bbox, without needing an img array at all
        (see descape.render_iso's own docstring for that split, Phase B-B) --
        then self._cache.patch(bbox) recomposites just the touched chunks
        and self.map_view.invalidate_region(bbox) schedules the actual Qt
        repaint, which pulls the freshly-patched pixels straight from the
        cache. dirty_screen_bbox_iso() can decline (return None) if an edit
        pushed a tile's elevation outside the range self._iso_proj was
        sized for -- see that function's own docstring for why this is a
        defensive fallback, not an expected path given the viewer's own
        elevation tools already clamp to that same range. Falling back to a
        full _render_current() keeps correctness even if that assumption is
        ever violated (e.g. a hand-edited file loaded with an elevation
        already out of range), at the cost of one full re-render instead of
        an instant patch for that one stroke tile. Only the level in view is
        patched (and the level a placeholder draws, while one is pending);
        every other resident level drops its chunks under the bbox, and the
        load warm refills the neighbours once the mouse button is released
        (TASK-031.73). With no viewport target every level is patched.

        Sloped mode (Track C4's Step 3): structurally the same two-part path
        as Stepped, through dirty_screen_bbox_sloped() instead. The one thing
        that differs is invisible from here and lives on the cache: patch()
        re-derives corner_rise from the elevations the bbox call just mutated,
        so the height field the patch composites against is the post-edit one.

        Every Sloped edit route lands in that branch: the four Terrain tools
        (released by Track C4's Step 4), Paint Can via on_fill(), Paste via
        paste_region(), and Undo/Redo. Undo is worth naming because it was
        reachable in Sloped BEFORE Step 4 -- _update_tool_enabled()'s gate
        never covered it, since _update_edit_actions() enables it from
        edit_history alone -- so "edit in Stepped, switch to Sloped, Ctrl+Z"
        hit this branch while every tool was still greyed out. Before the
        branch existed that undo fell through to the Flat branch below --
        whose assert passed, since Sloped carried no snapshot -- and patched
        axis-aligned tile squares onto an isometric canvas. See
        tests/test_sloped_edit.py's checks 7 and 8.

        Flat mode (Phase B-E): no elevation term and no fallback case --
        the dirty rects cover the dirty tiles' own pixel squares, and their
        ring of neighbours while terrain blending is effective. Unlike
        Stepped, there's no footprint-expansion step here
        either: composite_rect_flat() redraws terrain AND every overlapping
        unit in full stacking order within whatever rect it's given (see
        that function's own docstring for why that makes the old
        refresh_units_over() fixed-point dirty-tile expansion unnecessary).
        Patches render_paint.flat_dirty_rects()' rects (one per 16-tile block the
        set touches, one a tile where a block is sparse), not one union
        bbox -- a union over a scattered undo set could span the whole map,
        turning patch() into a full recomposite. Only the level in view is
        patched; every other resident level drops its chunks under the rects,
        which the re-armed level warm refills at idle for the levels next to
        the one in view, the rest on demand (TASK-338). With no
        viewport target every level is patched. Two
        cases evict the cached region under the set's bbox instead
        (_evict_flat_region()), leaving the next paint to recomposite what is
        on screen: more than STEPPED_FULL_RERENDER_THRESHOLD dirty tiles, and
        rects that cover more tiles than that recomposite would
        (_flat_patch_exceeds_viewport()).

        Re-arms the level warm at its tail (Batch B step B1). Every branch
        below cancels the warms first, and _start_level_warm() had exactly one
        caller (load_scenario), so before this the first stroke, paste or undo
        of a session left the neighbour mips cold for the rest of it, and the
        next zoom to one paid its 0.6-4.3s level build inside paint(). The
        body lives in _apply_dirty_render() so the re-arm covers its early
        returns too (both full-rerender thresholds, both out-of-range bbox
        fallbacks, the no-cache guards), not just the patch path. An exception
        propagates past the re-arm, so no warm ever starts against a
        half-patched cache.

        Returns True when a full branch ran (the threshold's in-place refresh,
        _refresh_wholesale(), or an out-of-range re-render): the unit step
        must then run wholesale, not apply a history move's splices
        (_refresh_after_history_move). Flat's eviction is not one: it moves
        no unit source, so the splices still apply.

        `deferred` (TASK-031.61 A4, a terrain + unit history move): a list
        that, when the Stepped/Sloped patch branch runs for an edit that moved
        no elevation, gets the terrain bbox appended instead of a composite.
        Only patch_sources() runs here; the caller's unit step then repaints
        that bbox with its own (_patch_unit_edit_cache()). Every other branch
        leaves it empty and repaints as usual."""
        if not dirty_indices or self.scenario is None:
            return False
        # The place-ghost memo keys on id(_iso_elevations), which an in-place edit keeps.
        self._place_ghost_memo = None
        # The drag previews' rule answers read terrain too (TASK-327).
        self._reset_drag_rule_memos(grid=False)
        self._cliff_hover_memo = None
        # Its cached ghost draws were placed at the old elevations.
        self._cliff_hover_look = None
        full = self._apply_dirty_render(dirty_indices, deferred)
        # Re-arms what the body's _cancel_warms() dropped. Kept below the
        # guard above, whose early return cancelled nothing to re-arm.
        with perf_trace.phase("warm_restart"):
            self._start_level_warm()
        return full

    def _apply_dirty_render(self, dirty_indices, deferred: list | None = None) -> bool:
        """_apply_dirty's body, split out so that every early return below
        still reaches that method's tail re-arm of the level warm. True when
        a full branch ran. `deferred`: see _apply_dirty()."""
        # Before any of the elevation-array mutation below (and before
        # patch(), which rebuilds source caches off it) -- see
        # _cancel_warms's docstring. One call here covers every stroke,
        # Paint Can, Paste and Undo/Redo route into the three style branches.
        with perf_trace.phase("warm_restart"):
            self._cancel_warms()
        mm = self.scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        if self._render_style == "stepped":
            if self._cache is None:
                return False
            # Above STEPPED_FULL_RERENDER_THRESHOLD dirty tiles (a Paint Can
            # fill spanning a large fraction of the map, measured with
            # tools/bench_fill_latency.py -- see that constant's own
            # comment), dirty_screen_bbox_iso's per-tile dilation is slower
            # than just re-rendering. Skips the dilation call entirely
            # rather than running it and discarding the result -- the
            # dilation cost is exactly what this branch exists to avoid.
            # Refreshed in place since TASK-031.61 A5, keeping the sprite memos.
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                if self._refresh_wholesale(dirty_indices, "Elevation view"):
                    return True
                self._rerender_past_threshold(dirty_indices, "Elevation view")
                return True
            # with_sprites MUST be the live cache's own flag, never a
            # constant: the widening this asks for and the sprites _level()
            # actually draws have to agree, or an elevation stroke beside a
            # large building leaves a stale sprite fragment behind (P3-g5).
            # tests/test_sprite_toggle_viewer.py pins this exact argument.
            #
            # elevation_changed (draw-perf plan Step 3): populated in place,
            # then threaded into patch() so a terrain-paint-only edit skips
            # rebuilding units_by_tile/building_bboxes/sprites entirely --
            # see IsoChunkCache._refresh_source_caches's own docstring.
            elevation_changed: set = set()
            with perf_trace.phase("bbox"):
                bbox = dirty_screen_bbox_iso(
                    self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                    with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                    flatten_elevations=(self._terrain_style == "flat"),
                )
            if bbox is None:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Elevation view: edit exceeded the cached elevation range, "
                    f"re-rendered full map (prepared in {elapsed:.2f}s)"
                )
            else:
                # Only the level in view patches or rebuilds; the others evict the bbox (TASK-031.73).
                target = self.map_view.viewport_chunk_target()
                rebuild_levels = None if target is None else (target[0],)
                if deferred is not None and not elevation_changed:
                    # The unit step composites this bbox with its own (A4).
                    with perf_trace.phase("patch_sources"):
                        self._cache.patch_sources(bbox, elevation_changed=elevation_changed)
                    deferred.append(bbox)
                    return False
                with perf_trace.phase("patch"):
                    self._cache.patch(
                        bbox, elevation_changed=elevation_changed, levels=self._visible_levels(target),
                        rebuild_levels=rebuild_levels,
                    )
                # The draped grid reads the elevations the bbox call above
                # mutated. Only on an edit that moved one: a terrain-only
                # stroke cannot move a grid line. Qt coalesces the resulting
                # update() calls, so a stroke needs no debounce of its own.
                if elevation_changed:
                    with perf_trace.phase("overlays"):
                        self.map_view.refresh_elevation_overlays()
                with perf_trace.phase("invalidate"):
                    self.map_view.invalidate_region(bbox)
            full = bbox is None
        elif self._render_style == "sloped":
            if self._cache is None:
                return False
            # Same two-part shape as Stepped, and the same threshold: the
            # dilation this guards is a per-tile Python set walk, so its cost
            # doesn't depend on the terrain style at all, and Sloped's own
            # _render_current() fallback is only ~1.8-2.0x Stepped's
            # (tools/bench_sloped_patch.py) -- so 20,000 stays conservative
            # here rather than needing its own measured constant.
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                if self._refresh_wholesale(dirty_indices, "Sloped view"):
                    return True
                self._rerender_past_threshold(dirty_indices, "Sloped view")
                return True
            # with_sprites MUST be the live cache's own flag, same load-bearing
            # warning the Stepped branch above carries -- tests/
            # test_sprite_toggle_viewer.py pins this exact argument too.
            #
            # elevation_changed: same draw-perf plan Step 3 wiring as the
            # Stepped branch above -- see SlopedChunkCache._refresh_source_
            # caches's own docstring: corner_rise rebuilds on any elevation
            # change, the unit layers splice over the corner-changed seeds.
            elevation_changed: set = set()
            with perf_trace.phase("bbox"):
                bbox = dirty_screen_bbox_sloped(
                    self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                    with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                )
            if bbox is None:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Sloped view: edit exceeded the cached elevation range, "
                    f"re-rendered full map (prepared in {elapsed:.2f}s)"
                )
            else:
                # patch() rebuilds corner_rise from the elevations the call
                # above just mutated, before compositing anything -- see
                # SlopedChunkCache._refresh_source_caches(). It REBINDS the
                # array, so the grid overlay has to be handed the new one.
                # As in the Stepped branch: only the level in view patches or rebuilds (TASK-031.73).
                target = self.map_view.viewport_chunk_target()
                rebuild_levels = None if target is None else (target[0],)
                if deferred is not None and not elevation_changed:
                    # The unit step composites this bbox with its own (A4).
                    with perf_trace.phase("patch_sources"):
                        self._cache.patch_sources(bbox, elevation_changed=elevation_changed)
                    deferred.append(bbox)
                    return False
                with perf_trace.phase("patch"):
                    self._cache.patch(
                        bbox, elevation_changed=elevation_changed, levels=self._visible_levels(target),
                        rebuild_levels=rebuild_levels,
                    )
                if elevation_changed:
                    with perf_trace.phase("overlays"):
                        self.map_view.refresh_elevation_overlays()
                with perf_trace.phase("invalidate"):
                    self.map_view.invalidate_region(bbox)
            full = bbox is None
        else:
            assert self._render_style == "flat" and self._iso_elevations is None, (
                "Flat (non-isometric) mode must never carry an elevation snapshot"
            )
            if self._cache is None:
                return False
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                # Sized from the indices alone and grown one tile for the blend ring: no dilation, no rects.
                with perf_trace.phase("bbox"):
                    idx = np.fromiter(dirty_indices, dtype=np.int64, count=len(dirty_indices))
                    ys, xs = np.divmod(idx, mm.map_width)
                    bbox = (
                        max(int(xs.min()) - 1, 0) * tile_px, max(int(ys.min()) - 1, 0) * tile_px,
                        min(int(xs.max()) + 2, mm.map_width) * tile_px, min(int(ys.max()) + 2, mm.map_height) * tile_px,
                    )
                self._evict_flat_region(bbox, len(dirty_indices))
                return False
            with perf_trace.phase("bbox"):
                coords = [(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices]
                rects = flat_dirty_rects(coords, mm.map_width, mm.map_height, tile_px, self._layers)
                target = self.map_view.viewport_chunk_target()
                evict = self._flat_patch_exceeds_viewport(rects, tile_px, target)
            if evict:
                self._evict_flat_region(_rects_bbox(rects))
                return False
            with perf_trace.phase("patch"):
                self._cache.patch_rects(rects, levels=None if target is None else (target[0],))
            with perf_trace.phase("invalidate"):
                for rect in rects:
                    self.map_view.invalidate_region(rect)
            full = False
        return full

    def _flat_patch_exceeds_viewport(self, rects, tile_px: int, target) -> bool:
        """Top-down Flat: whether the tiles `rects` cover, which
        patch_rects() composites at the visible level, outnumber
        _FLAT_EVICT_RATIO times the tiles an eviction of their bbox
        recomposites: the visible-level chunks the bbox reaches that the
        viewport shows, a ragged edge chunk at its real size (TASK-338). Both
        sides price one level, since the patch evicts the others too. A bbox
        wholly off screen evicts at any finite ratio: nothing is composited
        now. `target` is viewport_chunk_target(); False with none, or no rects."""
        if target is None or not rects:
            return False
        mip, vx0, vy0, vx1, vy1 = target
        cache = self._cache
        covered = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) / tile_px**2
        evicted = 0.0
        under = cache.chunk_range_under(_rects_bbox(rects), mip)
        if under is not None:
            cx0, cy0 = max(under[0], vx0), max(under[1], vy0)
            cx1, cy1 = min(under[2], vx1), min(under[3], vy1)
            if cx1 >= cx0 and cy1 >= cy0:
                canvas_w, canvas_h = cache.canvas_dims(mip)
                size = cache.chunk_px
                px = (min((cx1 + 1) * size, canvas_w) - cx0 * size) * (min((cy1 + 1) * size, canvas_h) - cy0 * size)
                evicted = px / cache.mip_tile_px(mip) ** 2
        if evicted == 0:
            return not math.isinf(_FLAT_EVICT_RATIO)
        return covered > _FLAT_EVICT_RATIO * evicted

    def _evict_flat_region(self, bbox: tuple[int, int, int, int], n_dirty: int | None = None) -> None:
        """Top-down Flat's answer to a terrain edit too large to patch:
        evicts every cached chunk under bbox (REFERENCE canvas pixels) at
        every resident level, so the next paint recomposites the ones on
        screen. One invalidate_region() call, never one per rect (it scans
        every cached key), with terrain_changed left True so both terrain
        mirrors are re-read over bbox. Logs a status line only with n_dirty,
        the count-threshold branch's: a Draw step past the ratio rule must not
        fill the log."""
        with perf_trace.phase("patch"):
            self._cache.invalidate_region(bbox)
        with perf_trace.phase("invalidate"):
            self.map_view.invalidate_region(bbox)
        if n_dirty is not None:
            self._log_status(f"Flat view: {n_dirty} tiles dirty, evicted the cached region instead of patching")

    def _refresh_wholesale(self, dirty_indices, label: str) -> bool:
        """_apply_dirty_render()'s threshold branch (TASK-031.61 A5): the
        cache's refresh_wholesale() in place of a new cache, so the sprite
        memos survive. Then what the patch path does for the whole canvas:
        Qt's repaint and, when an elevation moved, the overlays (camera
        markers, rulers, trigger rings, badges); plus _render_current()'s
        texture prefetch. False when the cache refused (an elevation out of
        range): the caller re-renders."""
        t0 = time.perf_counter()
        with perf_trace.phase("refresh"):
            moved = self._cache.refresh_wholesale(
                dirty_indices, flatten=self._render_style == "stepped" and self._terrain_style == "flat"
            )
            if moved is None:
                return False
            if self._layers.terrain_textures:
                asset_source.prefetch_terrain_textures(self._texture_prefetch_ids())
            self.map_view.invalidate_region((0, 0, *self._cache.canvas_dims(0)))
            if moved:
                self.map_view.refresh_elevation_overlays()
        self._log_status(
            f"{label}: {len(dirty_indices)} tiles dirty, refreshed the full map in place instead of patching "
            f"(in {time.perf_counter() - t0:.2f}s)"
        )
        return True

    def _rerender_past_threshold(self, dirty_indices, label: str) -> None:
        """The threshold branch's fallback when refresh_wholesale() refused.
        Keeps the `N tiles dirty, re-rendered` shape that
        tools/bench_gui_drag.py reads as its `rerender` path."""
        elapsed, _ = self._render_current(reset_view=False)
        self._log_status(
            f"{label}: {len(dirty_indices)} tiles dirty, re-rendered full map: the edit exceeded the cached "
            f"elevation range (prepared in {elapsed:.2f}s)"
        )

    def _cancel_warms(self) -> None:
        """Drops any in-flight level warm AND margin-ring chunk warm --
        called by every path that mutates the scenario, mutates the
        elevations array, or replaces the cache, BEFORE it mutates.

        That ordering is the whole race argument: a warm slice/chunk only
        ever resumes between event-loop iterations, so if every mutating
        path cancels first, a slice/chunk and a mutation can never
        interleave, and none of a threaded design's locks/snapshots/tokens
        are needed. The install predicates in render_cache's
        level_warm_job() are the second line of defence for a level-warm
        path nobody enumerated, not the first; the margin warm has no
        install step to revalidate (get_chunk() populates the cache
        directly), so its own second line of defence is
        is_level_resident() at start() time instead.

        One body with three cancels (rather than nine new call sites
        enumerated separately) is what keeps this correct as the set of
        warms grows: the correctness argument is "every mutating path
        cancels first," and that can't drift out of sync with itself. Also
        resets the margin warm's pan-direction baseline (see
        self._last_viewport_chunk_target's own comment) -- a stale one just
        means the next ring after this mutation starts with no direction
        preference, never a wrong answer. Drops _load_warm_queue too: a
        pending (mip, chunks) pair that never got to _load_warmer.start()
        is exactly as stale as one already ticking.

        Cheap and idempotent, so an over-broad call site costs at most a warm
        that has to be restarted on the next open -- always the right side to
        err on here.

        The Eraser, the Convert brush and the Unit Brush mutate the model
        mid-drag and reach the cache only at their flush or release (TASK-326),
        so they cancel at the mutation itself (_erase_units(), _convert_one(),
        _unit_brush_stroke_tiles()'s add callback), not in that
        later _after_unit_mutation() alone. Only where a unit really changes,
        not per drag event: a stroke that changed nothing has no flush to
        re-arm what a cancel dropped. _start_level_warm() then arms nothing
        until the flush (_unit_flush_owed())."""
        self._level_warmer.cancel()
        self._margin_warmer.cancel()
        self._load_warmer.cancel()
        self._load_warm_queue = []
        self._reach_queue = []
        self._reach_running = None
        self._reach_keys = ()
        self._last_viewport_chunk_target = None
        # A pre-rendered zoom frame's own stamps void it; this re-renders it once the edit settles.
        self.map_view.retarget_preframes()

    def _unit_flush_owed(self) -> bool:
        """Whether an Eraser, Convert or Unit Brush stroke has changed the model
        since its last flush, so the cache's unit sources are behind it (TASK-326)."""
        return bool(self._unit_stroke_pending_splices) or (
            self._erase is not None and bool(self._erase.pending_splices)
        )

    # --- placeholder paints (TASK-031.25) ---------------------------------

    def _placeholders_on(self) -> bool:
        """Placeholders need a warm to resolve them, so they follow Preload
        zoom levels (D7), and only a process that allows them gets any (D8)."""
        return self._placeholders_allowed and settings.get_preload_zoom_levels()

    def set_placeholders_allowed(self, allowed: bool) -> None:
        """Opts this window in or out of placeholder paints (D8: main() opts
        in; tests and tools stay out unless they ask)."""
        self._placeholders_allowed = allowed
        self.sync_canvas_placeholders()

    def set_reach_warm_allowed(self, allowed: bool) -> None:
        """Opts this window in or out of the zoom-out reach warm (TASK-031.11
        D9: the app opts in; tests and tools stay out unless they ask)."""
        self._reach_warm_allowed = allowed
        if not allowed:
            self._drop_reach()

    def sync_canvas_placeholders(self) -> None:
        """Pushes _placeholders_on() to the canvas; off settles a pending level."""
        self.map_view.set_canvas_placeholders(self._placeholders_on())
        if not self._placeholders_on():
            self._stop_placeholder_timers()

    def _pending_level(self) -> int | None:
        item = self.map_view._canvas_item
        return None if item is None else item.pending_mip()

    def _on_canvas_placeholder(self, mip: int) -> None:
        """MapCanvasItem's on_placeholder, called inside paint() once per new
        pending level: everything else waits for the next event-loop turn."""
        self._placeholder_item = self.map_view._canvas_item
        self._placeholder_pending_timer.start()

    def _on_placeholder_pending(self) -> None:
        """Re-anchors the warm on the pending level now rather than at the
        next viewport poll fire, and arms the watchdog."""
        item = self.map_view._canvas_item
        if item is None or item is not self._placeholder_item:
            return
        mip = item.pending_mip()
        if mip is None:
            return
        last = self._last_viewport_chunk_target
        if last is None or last[0] != mip:
            self._on_viewport_changed()
        else:
            self._start_level_warm()
        self._placeholder_watchdog.start(PLACEHOLDER_STUCK_MS)

    def _on_placeholder_watchdog(self) -> None:
        """Settles a placeholder that no warm will resolve: some path dropped
        its warm. Re-arms instead while a warm is active, a handler sits
        between its _cancel_warms() and its processEvents() (_busy), or an
        Eraser or Convert flush is owed (_start_level_warm() arms nothing then)."""
        item = self.map_view._canvas_item
        if item is None or item is not self._placeholder_item:
            return
        self._maybe_refine()
        mip = item.pending_mip()
        if mip is None:
            return
        warming = any(w.is_active for w in (self._level_warmer, self._load_warmer, self._margin_warmer))
        if warming or self._busy or self._unit_flush_owed():
            self._placeholder_watchdog.start(PLACEHOLDER_STUCK_MS)
            return
        item.force_settle()
        debug_log.log(f"placeholder: force-settled level {mip}")

    def _stop_placeholder_timers(self) -> None:
        self._placeholder_pending_timer.stop()
        self._placeholder_watchdog.stop()

    def _rearm_pending_warm(self) -> None:
        """Re-arms the warm a _cancel_warms() caller dropped, once its change
        is applied, if a placeholder waits on a level; else a no-op. For the
        callers that do not reach _start_level_warm() themselves."""
        if self._pending_level() is not None:
            self._start_level_warm()

    def _maybe_refine(self) -> None:
        """Settles the pending level once it is the view's level and every
        chunk its paint reads is cached (D2), then starts the neighbour warm
        it held back (D6). Called when a load or margin warm drains, when the
        pending view entry has nothing to warm, and by the watchdog."""
        item = self.map_view._canvas_item
        mip = None if item is None else item.pending_mip()
        if mip is None:
            return
        target = self.map_view.viewport_chunk_target()
        rect = self.map_view._current_scene_rect()
        if target is None or target[0] != mip or rect is None:
            return
        if not item.view_ready(mip, rect, self.map_view.view_device_scale()):
            return
        item.refine(mip)
        perf_trace.level_event("refine", mip, 0.0)
        self._placeholder_watchdog.stop()
        # The settled level's ring entry outlives the re-anchor, ahead of the neighbours' entries.
        self._start_level_warm(keep=[entry for entry in self._load_warm_queue if entry[0] == mip])
        self._pump_load_warm()

    def _start_level_warm(self, keep=()) -> None:
        """Queues the neighbouring mip levels' sprite layers for an idle-time
        warm -- the 2026-09-04 plan's hook point. `keep`: load-warm entries
        that survive the re-anchor at the queue's head (_maybe_refine()'s
        settled ring); an early return leaves the queue as it was.

        Called from load_scenario AFTER _render_current(), not from inside
        it: _render_current is also the re-render path for every terrain-
        style/quality/slider change, and those already cancel a warm rather
        than starting one.

        Also called at the tail of _apply_dirty() and _after_unit_mutation()
        (Batch B step B1), the deliberate exception to that: an edit cancels
        the warm and then re-arms it, so the neighbour mips do not stay cold
        for the rest of the session once the first edit lands. A stroke
        therefore pays a start/cancel pair per step, which the plan accepts
        pending a Perf Trace pass. The style/quality re-render paths reached
        through _render_current only cancel; their re-arm is the first
        viewport fire on the new cache (_on_viewport_changed).

        And from _on_viewport_changed whenever the viewport's mip changes
        (zoom plan 2026-09-29): the set is anchored on the level in view,
        viewport_chunk_target()[0], because the next zoom leaves from it. The
        fit level is the anchor only when there is no viewport target.

        Silent no-op with no map, no viewport (every headless test, and a
        window not yet shown) or a single-level ladder -- _fit_baseline_scale
        returns None for the first two, which is why the level is derived
        from it rather than from mip_for_scale, whose clamp would happily
        return a fabricated level 0 instead.

        Also a no-op while an Eraser or Convert mutation waits for its flush
        (_unit_flush_owed(), TASK-326): the model is post-edit and the cache's
        buckets pre-edit, so a warm armed there, by a viewport fire after the
        mutation's cancel say, would walk lists the cache has not heard of.
        Every such window closes through _after_unit_mutation(), whose call
        here re-arms once the cache has caught up.

        While the canvas waits on a placeholder's level and that level is the
        anchor (TASK-031.25 D6), the warm is that level's job alone, so no
        neighbour walk ticks before it; a resident one has no job, and the
        loop at the end queues its load warm at once (no re-queued prime,
        TASK-031.82 D6). The neighbours are warmed through _maybe_refine(),
        which calls this again once the level has settled. The anchor itself
        is otherwise only derived and primed, last (pack_only; see
        level_warm.neighbour_mips_of())."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        if self._unit_flush_owed():
            return
        fit = self.map_view._fit_baseline_scale()
        if fit is None:
            return
        target = self.map_view.viewport_chunk_target()
        # devicePixelRatio for the same reason CanvasItem._mip_for_painter
        # reads deviceTransform() rather than worldTransform(): the level a
        # paint actually selects is the DEVICE-space one, so a HiDPI window
        # would otherwise warm the neighbours of a level it never paints.
        # viewport_chunk_target() already folds it in.
        if target is not None:
            anchor = target[0]
        else:
            anchor = self._cache.mip_for_scale(fit * self.map_view.devicePixelRatioF())
        mips = level_warm.neighbour_mips_of(self._cache, anchor)
        self._warm_set = list(mips)
        if self._pending_level() == anchor:
            # D6: the level a placeholder waits on warms first and alone; refine re-arms the rest.
            mips, pack_only = [anchor], ()
        else:
            # The current mip's pack derive too: a wholesale rebuild drops its
            # pack, and a single-level ladder has no neighbours at all.
            pack_only = (target[0],) if target is not None else ()
            # A gen-bump edit rebuilds only the visible level (patch's rebuild_levels); warm the
            # other stale resident levels too, or the next zoom to one rebuilds it inside paint.
            visible = target[0] if target is not None else None
            mips += [
                mip for mip in self._cache.resident_levels()
                if mip != visible and mip not in mips and not self._cache.is_level_resident(mip)
            ]
        # A re-anchor's old (mip, chunks) are stale. Safe before start(): its cancel never fires on_job_done.
        self._load_warm_queue = list(keep)
        # TASK-031.11 D3: a re-anchor preempts a running reach entry; a pending anchor's reach comes after refine.
        if self._pending_level() == anchor:
            self._drop_reach()
        else:
            self._preempt_reach()
        notifying = self._level_warmer.start(
            self._cache, mips, pack_only=pack_only, on_job_done=self._queue_load_warm,
            on_drained=self._on_level_drained, requeue_prime=self._pending_level() != anchor,
        )
        # A mip with no queued job fires no on_job_done at all -- check its
        # residency here, once, rather than the callback ever having to
        # distinguish "no job" from "job not done yet". A mip that will
        # notify must not start here too: its load warm would run before its
        # pack derive, and be queued twice.
        for mip in mips:
            if mip not in notifying:
                self._queue_load_warm(mip)
        # A pending anchor gets its reach once refine calls this again (D2, D6 of TASK-031.25).
        if self._pending_level() != anchor:
            self._queue_reach()

    def _queue_load_warm(self, mip: int) -> None:
        """Queues the load-time chunk warm for one neighbour mip once its
        sprite layer is actually resident (2026-09-07 plan's load-time
        margin warm, Step 2/3) -- _start_level_warm's LevelWarmer.start()
        on_job_done callback, and also called directly from there for a mip
        already resident when the level warm started.

        Re-checks is_level_resident(mip) itself rather than trusting "the
        job finished" (on_job_done fires on a dropped job too, and a job's
        own install() can return False when a real paint got there first --
        in that second case the level IS resident, just not because of
        this warm, and this must still proceed). Silent no-op with no
        cache, the setting off, a still-not-resident mip, or a degenerate/
        off-grid projection at that mip -- viewport_chunk_target_at()
        returns None for the last case, mirroring _on_viewport_changed's
        own guard shape.

        viewport_chunk_target_at()'s raw range is bounded to a real
        viewport's worth (margin_warm.bounded_chunk_range(), sized via
        MapView.viewport_chunk_span()) before it ever reaches
        load_warm_chunks() -- load_scenario() always calls
        _start_level_warm() right after a fit-to-view render, so the raw
        range is the WHOLE neighbour-mip grid, not the ~20-chunk patch this
        feature is sized for. See bounded_chunk_range's own docstring.

        The bound is the view a crossing lands on, not a residual-1.0
        viewport (TASK-031.25 D3b, _neighbour_warm_range()). A notch onto a
        warmed neighbour then lands sharp with nothing composited in its paint.

        The level a placeholder waits on is not bounded at all: its warm is
        the view's own chunk range at it (_queue_pending_load_warm(), D3)."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        if not self._cache.is_level_resident(mip):
            return
        if mip == self._pending_level():
            self._queue_pending_load_warm(mip)
            return
        target = self.map_view.viewport_chunk_target_at(mip)
        if target is None:
            return
        view = self.map_view.viewport_chunk_target()
        cx0, cy0, cx1, cy1 = self._neighbour_warm_range(mip, target, view)
        chunks = margin_warm.load_warm_chunks(self._cache, mip, cx0, cy0, cx1, cy1)
        if not chunks:
            return
        self._load_warm_queue.append((mip, chunks))
        self._preempt_reach()
        self._pump_load_warm()

    def _neighbour_warm_range(self, mip: int, target, view) -> tuple[int, int, int, int]:
        """The chunk range a neighbour's load warm takes inside its raw
        projection `target` (TASK-031.25 D3b, review fold), `view` being
        viewport_chunk_target() or None.

        Finer than the view: twice the viewport span plus one per axis. Just
        past a zoom-in crossing the residual is 0.5-0.625, so the view covers
        1.6-2 spans, and a view edge late in a chunk touches one more index.
        Placed at the fraction of the range where the next zoom anchors
        (MapView.zoom_anchor_scene(), the cursor while zoom-on-cursor is on):
        zooming in about a point keeps it at the same fraction of the view.
        Only when it fits the chunk LRU beside the settled view and the
        coarser neighbour's patch (_fits_chunk_budget()); else the plain span,
        so the warm never evicts the view's own chunks.

        Coarser than the view: the span plus one (chunk alignment on a
        zoom-out), centred. Anything else: the span, centred."""
        cache = self._cache
        span_w, span_h = self.map_view.viewport_chunk_span(cache, mip)
        if view is None or mip == view[0]:
            return margin_warm.bounded_chunk_range(*target, span_w, span_h)
        if mip < view[0]:
            return margin_warm.bounded_chunk_range(*target, span_w + 1, span_h + 1)
        anchor = self.map_view.zoom_anchor_scene()
        size = cache.chunk_px * cache.mip_scale(mip)
        cx0, cy0, cx1, cy1 = target
        frac = ((anchor.x() / size - cx0) / (cx1 - cx0 + 1), (anchor.y() / size - cy0) / (cy1 - cy0 + 1))
        wide = margin_warm.bounded_chunk_range(*target, 2 * span_w + 1, 2 * span_h + 1, anchor=frac)
        if self._fits_chunk_budget(view, mip, wide):
            return wide
        return margin_warm.bounded_chunk_range(*target, span_w, span_h, anchor=frac)

    def _patch_footprint(self, mip: int, cx0: int, cy0: int, cx1: int, cy1: int) -> int:
        """Chunks a load warm of [cx0, cx1] x [cy0, cy1] can cache at `mip`:
        the range plus its depth-1 ring, clipped to the level's grid."""
        x0, y0, x1, y1 = self._footprint_rect(mip, cx0, cy0, cx1, cy1)
        return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)

    def _footprint_rect(self, mip: int, cx0: int, cy0: int, cx1: int, cy1: int) -> tuple[int, int, int, int]:
        """_patch_footprint()'s inclusive chunk rect; empty when x1 < x0 or y1 < y0."""
        cache = self._cache
        canvas_w, canvas_h = cache.canvas_dims(mip)
        max_cx, max_cy = (canvas_w - 1) // cache.chunk_px, (canvas_h - 1) // cache.chunk_px
        return max(cx0 - 1, 0), max(cy0 - 1, 0), min(cx1 + 1, max_cx), min(cy1 + 1, max_cy)

    def _fits_chunk_budget(self, view, mip: int, rng) -> bool:
        """Whether a finer neighbour's warm of `rng` fits the chunk LRU beside
        the settled view (and its ring) and the coarser neighbour's patch,
        both warmed or painted no earlier, so it cannot evict the view's chunks."""
        budget = self._chunk_budget()
        if budget is None:
            return True
        used = self._patch_footprint(view[0], *view[1:]) + self._patch_footprint(mip, *rng)
        return used + self._coarser_patch_footprint(view[0]) <= budget

    def _chunk_budget(self) -> int | None:
        """The chunk LRU's size in whole chunks, or None when unbounded."""
        cache = self._cache
        if cache.max_chunks is not None:
            return cache.max_chunks
        if cache.max_bytes is not None:
            return cache.max_bytes // (cache.chunk_px * cache.chunk_px * 3)
        return None

    def _coarser_patch_footprint(self, mip: int) -> int:
        """_patch_footprint() of the coarser neighbour's crossing patch (the
        span plus one, centred) for a view at `mip`; 0 with none."""
        rng = self._coarser_patch_range(mip)
        return 0 if rng is None else self._patch_footprint(mip - 1, *rng)

    def _coarser_patch_range(self, mip: int) -> tuple[int, int, int, int] | None:
        """The coarser neighbour's crossing patch range for a view at `mip`, or None."""
        cache = self._cache
        coarser = mip - 1
        if coarser < cache.mip_levels()[0]:
            return None
        raw = self.map_view.viewport_chunk_target_at(coarser)
        if raw is None:
            return None
        span_w, span_h = self.map_view.viewport_chunk_span(cache, coarser)
        return margin_warm.bounded_chunk_range(*raw, span_w + 1, span_h + 1)

    def _queue_pending_load_warm(self, mip: int) -> None:
        """The load warm of the level a placeholder waits on (TASK-031.25 D3,
        D6): the view's chunks at it (MapCanvasItem.view_chunks(), the range
        its paint and view_ready() read), then their depth-1 ring, at the head
        of _load_warm_queue in place of any entry already queued for it. A
        load warm in flight is cancelled; if it was another level of the
        current warm set it is queued again behind (load_warm_chunks()
        recomputes what is still uncached). With nothing in view to warm it
        refines at once."""
        item = self.map_view._canvas_item
        rect = self.map_view._current_scene_rect()
        if item is None or rect is None:
            return
        view = item.view_chunks(mip, rect, self.map_view.view_device_scale())
        if view is None:
            return
        centre, ring = margin_warm.load_warm_parts(self._cache, mip, *view)
        self._load_warm_queue = [entry for entry in self._load_warm_queue if entry[0] != mip]
        self._load_warm_queue[:0] = [(mip, chunks) for chunks in (centre, ring) if chunks]
        # A busy reach entry goes back to _reach_queue, not through _queue_load_warm() (TASK-031.11 D3).
        self._preempt_reach()
        busy = self._load_warmer.mip if self._load_warmer.is_active else None
        if busy is not None:
            self._load_warmer.cancel()
            if busy != mip and busy in self._warm_set:
                self._queue_load_warm(busy)
        if not centre:
            self._maybe_refine()
        self._pump_load_warm()

    def _pump_load_warm(self) -> None:
        """Starts the next queued (mip, chunks) pair on _load_warmer, if it
        isn't already busy with one -- the sequencing _queue_load_warm needs
        because up to two neighbour mips can each become ready at different
        times, and _load_warmer.start() (MarginWarmer's own contract)
        replaces whatever is in flight rather than queuing alongside it.
        Chained off _load_warmer's own on_drained callback, so the second
        mip's chunks start the moment the first mip's queue empties, with
        no polling. A placeholder's level refines here once its view entry
        has drained (_maybe_refine(), TASK-031.25).

        The zoom-out reach (TASK-031.11 D3) comes last: an entry of
        _reach_queue starts only once no load entry is queued or running, no
        level is pending and the level and margin warmers are idle."""
        self._maybe_refine()
        if self._load_warmer.is_active:
            return
        if not self._load_warm_queue:
            # Every queued level is warm: a zoom frame refused across a level change may render now.
            self.map_view.retry_refused_preframes()
            self._start_reach()
            return
        mip, chunks = self._load_warm_queue.pop(0)
        # The pending level's entry defers too while input is live (TASK-031.82 D3): its placeholder sharpens after.
        self._load_warmer.start(self._cache, mip, chunks, on_drained=self._pump_load_warm)

    # --- the zoom-out reach (TASK-031.11) ----------------------------------

    def _reach_on(self) -> bool:
        """Whether this window warms the reach: allowed (D9), preload on, and
        a cache whose chunks composite on a worker (D1)."""
        cache = self._cache
        return (
            self._reach_warm_allowed and cache is not None and settings.get_preload_zoom_levels()
            and cache.pools_chunks()
        )

    def _queue_reach(self) -> None:
        """Arms the reach (D7): both entries recomputed (D2, D5), replacing
        any queued ones, then a pump, since a re-anchor whose other warms
        queue nothing would otherwise never start them."""
        entries = self._reach_entries() if self._reach_on() else []
        self._reach_queue = list(entries)
        self._reach_keys = tuple(entries)
        if entries:
            self._pump_load_warm()

    def _reach_entries(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """The settled level's reach, then the coarser neighbour's (D2), each
        taken whole while the total fits the chunk LRU (D5): the settled view
        or its reach, the finer neighbour's chosen D3b range, the coarser
        neighbour's crossing patch or its reach, and the coarsest level's
        resident chunks. Stops at the first that does not fit."""
        cache = self._cache
        view = self.map_view.viewport_chunk_target()
        if view is None:
            return []
        mip = view[0]
        levels = cache.mip_levels()
        budget = self._chunk_budget()
        finer = 0
        raw = self.map_view.viewport_chunk_target_at(mip + 1) if mip + 1 <= levels[-1] else None
        if raw is not None:
            finer = self._patch_footprint(mip + 1, *self._neighbour_warm_range(mip + 1, raw, view))
        coarser = self._coarser_patch_footprint(mip)
        # A coarser patch at the coarsest level is already counted: its chunks are not counted twice.
        patch = self._coarser_patch_range(mip) if mip - 1 == levels[0] else None
        outside = None if patch is None else self._footprint_rect(mip - 1, *patch)
        coarsest = cache.level_chunk_count(levels[0], outside=outside)
        settled = self._patch_footprint(mip, *view[1:])
        entries = []
        for level in (mip, mip - 1):
            rng = self.map_view.viewport_reach_target(level) if level >= levels[0] else None
            if rng is None:
                break
            footprint = self._patch_footprint(level, *rng)
            if level == mip:
                total = footprint + finer + coarser + coarsest
            else:
                total = settled + finer + footprint + coarsest
            if budget is not None and total > budget:
                break
            if level == mip:
                settled = footprint
            entries.append((level, rng))
        return entries

    def _reach_chunks(self, mip: int, rng) -> list[tuple[int, int]]:
        """An entry's queue, recomputed whenever it starts (D3): the settled
        level's less the view and its depth-1 ring (the paint's and the margin
        warmer's), ordered about the zoom anchor (D4)."""
        view_map = self.map_view
        target = view_map.viewport_chunk_target()
        exclude = None
        if target is not None and target[0] == mip:
            _mip, cx0, cy0, cx1, cy1 = target
            exclude = {(cx, cy) for cx in range(cx0 - 1, cx1 + 2) for cy in range(cy0 - 1, cy1 + 2)}
        scale = self._cache.mip_scale(mip)
        rect = view_map.mapToScene(view_map.viewport().rect()).boundingRect()
        anchor = view_map.zoom_anchor_scene()
        view = (rect.left() / scale, rect.top() / scale, rect.right() / scale, rect.bottom() / scale)
        return margin_warm.reach_chunks(
            self._cache, mip, rng, view, (anchor.x() / scale, anchor.y() / scale), exclude=exclude,
        )

    def _start_reach(self) -> None:
        """D3's gate, then the first reach entry with anything left to warm."""
        if self._reach_running is not None or self._load_warmer.is_active or self._load_warm_queue:
            return
        if self._level_warmer.is_active or self._margin_warmer.is_active or self._pending_level() is not None:
            return
        while self._reach_queue:
            entry = self._reach_queue.pop(0)
            chunks = self._reach_chunks(*entry)
            if chunks:
                # D5: the coarsest level's chunks, the LRU's oldest after a roll in, must outlive the reach.
                self._cache.touch_level(self._cache.mip_levels()[0])
                self._reach_running = entry
                self._load_warmer.start(self._cache, entry[0], chunks, on_drained=self._on_reach_drained)
                if self._load_warmer.is_active:
                    return
                self._reach_running = None

    def _on_reach_drained(self) -> None:
        """A reach entry cached its chunks: the one-notch-out frame refused for
        them may render now (D10, the pump's empty branch retries it); then the
        next entry. Not recomputed (D7)."""
        self._reach_running = None
        self._pump_load_warm()

    def _preempt_reach(self) -> None:
        """Cancels a running reach entry and puts it back at the head of
        _reach_queue, to be recomputed when it resumes (D3)."""
        if self._reach_running is None:
            return
        self._load_warmer.cancel()
        self._reach_queue.insert(0, self._reach_running)
        self._reach_running = None

    def _drop_reach(self) -> None:
        """Cancels a running reach entry and clears the reach's state."""
        if self._reach_running is not None:
            self._load_warmer.cancel()
        self._reach_queue = []
        self._reach_running = None
        self._reach_keys = ()

    def _on_level_drained(self) -> None:
        """LevelWarmer's on_drained (D8): the reach may start now. Only with
        one queued, so a window without the reach pumps as before (D9)."""
        if self._reach_queue:
            self._pump_load_warm()

    def _refresh_reach(self) -> None:
        """D6 on a same-level poll fire: recomputes both entries and re-arms
        only when their (mip, range) keys changed, so a roll about a still
        cursor keeps the warm's progress."""
        entries = self._reach_entries() if self._reach_on() else []
        if tuple(entries) == self._reach_keys:
            return
        self._drop_reach()
        self._reach_queue = list(entries)
        self._reach_keys = tuple(entries)
        if entries:
            self._pump_load_warm()

    def _on_viewport_changed(self) -> None:
        """The margin warm's entry point (2026-09-07 plan's A2.5/A3) -- wired
        to self.map_view.on_viewport_changed, called whenever a viewport
        poll fire finds a new viewport_chunk_target().

        Silent no-op with no cache, the setting off, or a degenerate
        viewport (viewport_chunk_target() returns None) -- mirroring
        _start_level_warm's own guard shape so a headless/never-shown
        window (every default-tier test) stays safe.

        `lead` is the sign pair of the viewport origin's movement since the
        LAST fire this method itself recorded -- not MapView's own
        _last_viewport_target, which exists only to decide when the poll
        should stop and would go stale the moment this method's fire
        target catches up to it. Falls back to (0, 0) (see
        margin_warm.ring_chunks' own docstring for why that's a correct
        answer, not a missing one) on the first fire, right after a cancel,
        or whenever the mip itself changed -- a pan direction from a
        DIFFERENT level's chunk grid means nothing here.

        A changed mip, or no previous fire, also re-anchors the level warm on
        the new level first (zoom plan 2026-09-29). LevelWarmer.start cancels
        the in-flight job, whose scaled sprites stay in the process-global
        _scaled_cache, so a cancelled partial warm is not wasted. No previous
        fire means _cancel_warms() just ran: a zoom right after an edit would
        otherwise keep the pre-zoom anchor, and a re-render's first fire warms
        the new cache's neighbours. A fire inside _render_current's
        processEvents() re-arms on the outgoing cache, which its second
        _cancel_warms() drops before the swap."""
        with perf_trace.phase("viewport_changed"):
            self._on_viewport_changed_traced()

    def _on_viewport_changed_traced(self) -> None:
        """_on_viewport_changed's body, under its perf phase."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        target = self.map_view.viewport_chunk_target()
        if target is None:
            return
        mip, cx0, cy0, cx1, cy1 = target
        prev = self._last_viewport_chunk_target
        self._last_viewport_chunk_target = target
        if prev is None or prev[0] != mip:
            self._start_level_warm()
        elif mip == self._pending_level():
            # A pan while a placeholder waits: its view entry follows the view (replacing the old one).
            self._queue_load_warm(mip)
        lead = (0, 0)
        if prev is not None and prev[0] == mip:
            prev_cx0, prev_cy0 = prev[1], prev[2]
            lead = (
                (cx0 > prev_cx0) - (cx0 < prev_cx0),
                (cy0 > prev_cy0) - (cy0 < prev_cy0),
            )
        ring = margin_warm.ring_chunks(self._cache, mip, cx0, cy0, cx1, cy1, lead=lead)
        self._margin_warmer.start(self._cache, mip, ring, on_drained=self._on_margin_drained)
        # TASK-031.11 D3: the reach never ticks beside a ring; it resumes once the ring drains.
        if self._margin_warmer.is_active:
            self._preempt_reach()
        if prev is not None and prev[0] == mip and mip != self._pending_level():
            self._refresh_reach()
        # A fire whose ring is cached cancels the old ring and its drain callback: resume a parked reach here.
        if self._reach_queue and self._reach_running is None and not self._margin_warmer.is_active:
            self._pump_load_warm()

    def _on_margin_drained(self) -> None:
        """The margin warm cached its ring: the pending level may settle, and
        a zoom frame refused for uncached chunks may render now. The reach
        may start (TASK-031.11 D3)."""
        self._maybe_refine()
        self.map_view.retry_refused_preframes()
        if self._reach_queue:
            self._pump_load_warm()

    def _on_canvas_paint_timed(self, elapsed: float, mip: int) -> None:
        """MapCanvasItem's per-paint stopwatch callback (installed by
        _render_current, reported on only by load_scenario).

        Every paint counts, and none is filtered: no property of a single
        paint says "this was the real one". render_rect() returns pixels
        whether it composited them now or hit the LRU, and a paint of an 8px
        sliver is indistinguishable up front from the fit-to-view composite.
        Summing until the event loop goes idle needs no such proxy: the
        sliver contributes its 0.01s, the composite contributes its 5.10s,
        and the total is right in either order.

        The timer is armed only once load_scenario() has marked the report
        ready, so a paint that fires synchronously inside _render_current()'s
        own processEvents() (which does dispatch zero-delay timers) still
        accumulates but cannot print ahead of the load's own line."""
        # The opening level is built by now: drop what the sprite prefetch has left (TASK-031.47 D6).
        sprite_prefetch.discard()
        report = self._pending_paint_report
        if report is None:
            return
        report["paint"] += elapsed
        report["paints"] += 1
        # The mip reported is the costliest paint's, not the last one's: a
        # trailing sliver repaint can select a different level.
        if elapsed >= report["max_paint"]:
            report["max_paint"] = elapsed
            report["mip"] = mip
        if report["ready"]:
            self._paint_report_timer.start()

    def _emit_paint_report(self) -> None:
        """Prints the deferred-composite follow-up to a load's `Loaded ...`/
        `Created ...` line, once painting has stopped, then uninstalls the
        stopwatch so steady-state painting is back to one `is not None`
        check.

        `total` is (New Map's generate +) parse + prepare + summed paint
        time, i.e. work actually done, not wall clock from load start to
        here, which would fold in however long Qt sat idle and make the
        number a reading of how busy the machine was rather than of the file."""
        report = self._pending_paint_report
        self._pending_paint_report = None
        self.map_view.set_paint_timed_callback(None)
        if report is None or not report["paints"]:
            return
        total = report["generate"] + report["parse"] + report["prepare"] + report["paint"]
        count = report["paints"]
        counted = "" if count == 1 else f", {count} paints"
        self._log_status(
            f"First paint composited in {report['paint']:.2f}s "
            f"(total {total:.2f}s, mip {report['mip']}{counted})"
        )

    def _texture_prefetch_ids(self) -> list[int]:
        """The terrain ids a render of self.scenario textures: every tile's,
        every layered terrain's while blending draws them, plus the farm
        overlay's where it draws (sprites on, every style)."""
        from descape import native_composite, render_paint, render_sprites

        mm = self.scenario.map_manager
        tiles = np.arange(mm.map_width * mm.map_height)
        ids = np.unique(native_composite._terrain_ids(self.scenario, tiles)).tolist()
        if render_paint.blend_effective(self._layers):
            # A terrain used only as a layer has no base tile to list it above.
            layer_ids = np.unique(render_paint.tile_field_mirror(self.scenario, "layer"))
            ids.extend(layer_ids[layer_ids >= 0].tolist())
        if self._sprites_enabled and self._layers.farm_overlay:
            consts = {unit.unit_const for units in self.scenario.unit_manager.units for unit in units}
            ids.extend(tid for tid in map(render_sprites._terrain_overlay_for, consts) if tid is not None)
        return ids

    def _start_sprite_prefetch(self) -> None:
        """A load's sprite prefetch (TASK-031.47): the opening Stepped or Sloped
        level's frames, decoded on a pool from right after the parse, since the
        level builds 260-540 ms later (Step 0). Off with sprites off or in
        top-down Flat. The fit's tile half-width is at most the viewport's width
        over the map's diamond span, which bounds whether x2 can draw."""
        if not (self._sprites_enabled and self._render_style in ("stepped", "sloped")):
            sprite_prefetch.discard()
            return
        mm = self.scenario.map_manager
        span = mm.map_width + mm.map_height
        max_fit_half_w = max(0, self.map_view.viewport().width() - 4) / span if span > 0 else None
        sprite_prefetch.prefetch_level_sprites(self.scenario, self._unit_filter, max_fit_half_w)

    def _render_current(self, *, reset_view: bool = True, op_label: str | None = None) -> tuple[float, int]:
        # op_label: the style switches name themselves; otherwise load or re-render.
        with perf_trace.op(op_label or ("load" if reset_view else "re-render")):
            return self._render_current_traced(reset_view=reset_view)

    def _render_current_traced(self, *, reset_view: bool = True) -> tuple[float, int]:
        """Renders/prepares self.scenario at the currently selected Terrain
        Style and pushes it to self.map_view -- the render+display step
        shared by load_scenario/refresh_map/on_terrain_style_changed/
        _apply_dirty's full-redraw fallback, each of which wants a
        differently-worded status log line, so that part stays theirs.
        Returns (elapsed_seconds, tile_px).

        reset_view is threaded straight through to MapView.set_source() --
        see that parameter's own docstring. Default True matches only
        load_scenario() (a genuinely new document); _apply_dirty's
        fallback, refresh_map(), and on_terrain_style_changed() all pass
        False to keep the zoom/pan the user already had.

        Neither style blocks on the full canvas anymore as of Phase B-E
        (Flat was the last one still doing so -- render_scenario() always
        compositing the whole map up front, ~1-2s for a real map). Stepped
        (Phase B-C): elevations_and_proj() reads elevations and sizes the
        projection WITHOUT compositing a single pixel. Flat (Phase B-E):
        FlatChunkCache's own construction is similarly cheap -- it only
        precomputes unit_draws (_flat_unit_draws()), not any pixels. Both
        chunk caches then composite lazily, chunk by chunk, only for
        whatever MapCanvasItem's paint() actually asks for.
        This is the whole point of Track B:
        opening/switching to either view no longer pays for the full
        canvas up front.

        setEnabled(False) for the same span, not just the cursor: processEvents()
        re-enters the event loop, which would otherwise let a second click
        (Open, the Elevation View combo, a toolbar tool) dispatch a
        reentrant call into this same render path mid-flight -- e.g. a
        second render starting while self._cache/self.map_view are still
        being written by the first. Now purely a reentrancy guard rather
        than also covering a visible freeze -- the wait cursor still shows
        briefly for the cache construction above, but there's no multi-
        second compositing pass left for the window to actually appear
        frozen during. A disabled window still repaints (the cursor/log
        message stay visible), it just stops accepting input."""
        # Unconditional, and FIRST: this method replaces self._cache outright
        # on every path below, so anything still warming the old one is
        # warming a cache nothing will ever paint from. Covers refresh_map,
        # the terrain-style combo, the graphics-quality and elev-step
        # sliders, the Isometric View toggle and a re-entrant load in one
        # place, rather than one call per caller.
        self._cancel_warms()
        # A re-render's caches are not the load's: its sprite prefetch is load_scenario()'s alone.
        if not reset_view:
            sprite_prefetch.discard()
        self._stop_placeholder_timers()
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        # Again: a viewport poll fired in that turn can re-arm any warm on the outgoing cache.
        self._cancel_warms()
        try:
            t0 = time.perf_counter()
            # Armed before building the cache, not after: the deferred
            # composite cost this exists to capture happens on the FIRST
            # paint of the new MapCanvasItem set_source() below constructs,
            # which can fire before this method returns (a synchronous
            # processEvents() elsewhere in the call stack). reset_view is
            # True only for load_scenario's genuinely-new-document call;
            # every other caller re-renders the same document. The
            # _on_canvas_paint_timed stopwatch every set_source() below
            # installs is unconditional for the same reason: it no-ops
            # unless load_scenario() left a report pending.
            perf_trace.arm("load" if reset_view else "re-render")
            # Loads the map's textures on a pool while the first paint's level
            # builds; files already cached are skipped, so a re-render pays a scan.
            if self._layers.terrain_textures:
                asset_source.prefetch_terrain_textures(self._texture_prefetch_ids())
            mm = self.scenario.map_manager
            tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
            # Before set_source(): the new item's first paint can fire inside it.
            self._refresh_colour_mood_preview()
            if self._render_style == "stepped":
                elevations, proj = elevations_and_proj(self.scenario)
                # Flat + Isometric View (Flat+Isometric plan): force every
                # tile's elevation to 0 so terrain paints as a real iso
                # projection at elevation 0 rather than the scenario's real
                # heights -- proj is sized from the fixed legal constants
                # (MIN_ELEVATION/MAX_ELEVATION), never from observed data, so
                # an all-zero array yields a proj field-identical to real
                # Stepped's. self._terrain_style, not self._render_style: this
                # branch is also real Stepped, which must NOT flatten.
                if self._terrain_style == "flat":
                    elevations = np.zeros_like(elevations)
                self._iso_elevations, self._iso_proj = elevations, proj
                # sprites= at CONSTRUCTION rather than a set_sprites_enabled()
                # call afterwards: the setter's job is to rebuild and evict a
                # live cache, all of which this fresh one would be paying for
                # nothing. Same reason unit_filter is passed here too.
                self._cache = IsoChunkCache(
                    self.scenario, elevations, proj, tile_px,
                    unit_filter=self._unit_filter, sprites=self._sprites_enabled,
                    layers=self._layers,
                )
                # Before set_source(): the new item's first paint can fire inside it (TASK-031.64).
                self._cache.set_chunk_executor(*margin_warm.chunk_pool())
                self.map_view.set_source(
                    tile_px, terrain_style="stepped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view, on_paint_timed=self._on_canvas_paint_timed,
                    placeholders=self._placeholders_on(), on_placeholder=self._on_canvas_placeholder,
                )
            elif self._render_style == "sloped":
                # Same snapshot contract as Stepped above, and it must be the
                # SAME array object the cache and MapView hold (Risk #6): the
                # sloped patch path mutates it in place, so a copy here would
                # let the pick plane drift from the pixels.
                elevations, corner_rise, proj = sloped_elevations_and_proj(self.scenario)
                self._iso_elevations, self._iso_proj = elevations, proj
                # sprites= at CONSTRUCTION (Track P3-g6), same reason the
                # Stepped branch above passes it there rather than calling
                # set_sprites_enabled() afterwards: that setter's job is to
                # rebuild and evict a LIVE cache, all of which this fresh one
                # would be paying for nothing.
                self._cache = SlopedChunkCache(
                    self.scenario, elevations, corner_rise, proj, tile_px, unit_filter=self._unit_filter,
                    sprites=self._sprites_enabled, layers=self._layers,
                )
                self._cache.set_chunk_executor(*margin_warm.chunk_pool())
                self.map_view.set_source(
                    tile_px, terrain_style="sloped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view, on_paint_timed=self._on_canvas_paint_timed,
                    placeholders=self._placeholders_on(), on_placeholder=self._on_canvas_placeholder,
                )
            else:
                self._iso_elevations, self._iso_proj = None, None
                # sprites= at CONSTRUCTION (Track P3-g7), same reason the two
                # branches above pass it there rather than calling
                # set_sprites_enabled() afterwards.
                self._cache = FlatChunkCache(
                    self.scenario, tile_px, unit_filter=self._unit_filter,
                    sprites=self._sprites_enabled, layers=self._layers,
                )
                # Flat has no job path, so this stays serial; set for one rule across styles.
                self._cache.set_chunk_executor(*margin_warm.chunk_pool())
                self.map_view.set_source(
                    tile_px, terrain_style="flat", cache=self._cache, reset_view=reset_view,
                    on_paint_timed=self._on_canvas_paint_timed,
                    placeholders=self._placeholders_on(), on_placeholder=self._on_canvas_placeholder,
                )
            # set_source() drops the pick index along with every other scene
            # item, so a style switch made while in Units mode has to rebuild
            # it -- otherwise selection silently stops working until the mode
            # is toggled off and back on.
            if self._needs_unit_index():
                self._rebuild_unit_index()
            # set_source() dropped MapView's copy of the selection too; re-push it where live (TASK-359).
            if self._selection:
                self._reconcile_selection()
            # set_source() rebuilt the markers from MapView's own stored
            # list, which a NEW document's has not replaced yet -- so push
            # this scenario's. One call covers load, New Map, a map resize
            # and a style switch.
            self._refresh_camera_markers()
            # A quality change or style switch can add or remove the native level.
            self._update_zoom_native_action()
            return time.perf_counter() - t0, tile_px
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def refresh_map(self) -> None:
        # Re-render the currently open map, if any, so a settings change (e.g.
        # a newly-configured install path) takes effect immediately instead of
        # only on the next file open. self._busy guard: see load_scenario's
        # own comment on it -- same reentrancy concern, one shared flag.
        # reset_view=False: same document, same style, just picking up a
        # setting -- nothing about that should yank the user back to fit.
        if self.scenario is not None and not self._busy:
            self._busy = True
            try:
                elapsed, tile_px = self._render_current(reset_view=False)
                self._log_status(
                    f"Re-rendered map (tile_px={tile_px}, style={self._style_log_label}) "
                    f"prepared in {elapsed:.2f}s"
                )
            finally:
                self._busy = False
            unit_sprites.sprite_reach()  # an install or mod-art change resets it; see load_scenario()
            # Re-gate too, not just re-render. "Show sprites" is the one action
            # whose enablement depends on the install path, and its greyed-out
            # tooltip tells the user to go configure one -- so without this,
            # following that instruction re-renders the map and leaves the
            # toggle stubbornly greyed until the next file open.
            self._update_tool_enabled()
