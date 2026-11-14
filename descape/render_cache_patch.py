"""_ChunkCacheBase's patch path: patch() through invalidate_region(), the in-place recomposite of cached chunks."""

from __future__ import annotations

import numpy as np

from descape import (
    composite_backend,
    debug_log,
    perf_trace,
    render_paint,
)
from descape.render_cache_common import (
    RectJob,
    join_chunk_jobs,
)
from descape.render_cache_splice import (
    _union_bbox,
)


class _ChunkCachePatchMixin:
    """_ChunkCacheBase's patch path (render_cache.py's docstring has the split's rules)."""

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None, levels=None, rebuild_levels=None
    ) -> None:
        """"Patch, don't drop": for every chunk CURRENTLY cached, at every
        RESIDENT mip level (only those in `levels`, when given, and every
        other resident level drops its chunks under bbox: the viewer passes
        the level in view, TASK-031.73), that bbox (REFERENCE canvas pixels) overlaps
        once converted to that level, recomposites just the intersected
        sub-rect via self._composite_rect() and writes it into the
        existing chunk array in place -- the cached chunk stays valid
        immediately, without paying a full chunk recomposite (or leaving a
        stale one on screen until the next get_chunk() eviction). Chunks
        NOT currently cached need no action: get_chunk() always composites
        fresh against the current state, so there's nothing stale to fix
        for those.

        Iterates RESIDENT levels ({key[0] for key in self._cache}), not
        every ENUMERATED level (mip_levels()) -- a level holding zero
        cached chunks is never entered, which is what keeps a single edit's
        cost from scaling with the total level count rather than with how
        many levels are actually warm.

        Looks up the overlapping chunk-grid range directly (same index math
        as invalidate_region()) rather than scanning every cached entry --
        a real cost difference once max_chunks is in the hundreds and only
        a handful of chunks overlap a single edit's bbox.

        bbox must already reflect the edit; this method does not mutate any
        underlying source state itself (elevations, terrain) -- callers
        that need to (dirty_screen_bbox_iso() for Stepped) do so before
        calling this. Refreshes this cache's own per-edit derived state
        first (_refresh_terrain_ids(bbox), then self._refresh_source_caches()) -- see each subclass's own
        docstring for what that means to it.

        elevation_changed (draw-perf plan Step 3): the subset of this edit's
        dirty tiles whose elevation actually moved (see
        render_bbox._dirty_screen_bbox()'s own docstring for where this comes
        from) -- passed straight through to _refresh_source_caches(). None
        (the default) means "unknown", which every _refresh_source_caches()
        override treats as "assume the worst and rebuild everything", i.e.
        today's behavior -- every call site that doesn't yet compute this
        set keeps its exact prior cost and correctness. The degenerate-bbox
        check runs BEFORE this call now (it used to run after, paying a full
        rebuild for a bbox with nothing to patch).

        rebuild_levels: None (every level may) or the levels allowed to do
        per-level source work (rebuild or splice) inside this call. A resident
        level outside it that the refresh left stale (not is_level_resident())
        evicts bbox's chunks instead of patching them, so a Stepped elevation
        step that bumps the gen rebuilds the visible level only, not every
        resident one; the stale level rebuilds when next composited or warmed.
        A current level outside it defers its elevation splice instead
        (_LeveledChunkCache._splice_levels), and any level with deferred work
        (_defers_patch) evicts too, so no cached chunk shows an unflushed layer.

        The two halves are patch_sources() and patch_composite(). With a chunk
        pool set (set_chunk_executor()), a large enough patch composites its
        slices on it (TASK-031.67), to the same pixels, the same chunk
        objects and the same LRU order."""
        if self.patch_sources(bbox, elevation_changed, rebuild_levels):
            self.patch_composite(bbox, levels, rebuild_levels)

    def patch_sources(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None, rebuild_levels=None
    ) -> bool:
        """patch()'s source half: the terrain-id mirror over bbox, then
        _refresh_source_caches(). False, refreshing nothing, for an empty bbox.

        Run alone by a terrain + unit history move (TASK-031.61 A4), whose
        unit step then repaints the union of this bbox and its own, so the
        terrain is composited once rather than patched here and evicted after.
        Such a caller owns the composite of bbox at every resident level, and
        passes an empty elevation_changed only: with one, a level may go stale
        and SlopedChunkCache.patch() also rewrites the pick planes."""
        px0, py0, px1, py1 = bbox
        # Before the early return too: the caller may have written elevations in place.
        self._mutation_epoch += 1
        if px1 <= px0 or py1 <= py0:
            return False
        self._refresh_terrain_ids(bbox)
        self._refresh_source_caches(elevation_changed, splice_levels=rebuild_levels)
        return True

    def patch_composite(self, bbox: tuple[int, int, int, int], levels=None, rebuild_levels=None) -> None:
        """patch()'s composite half, after patch_sources(): recomposites bbox
        in every cached chunk of the resident levels, or evicts it where a
        level outside rebuild_levels went stale or has deferred work. With a
        chunk pool set, enough slices composite on it (_composite_patch_targets()).

        `levels` given: only those levels are patched, and every other
        resident level drops its chunks under bbox first (_evict_rects(),
        TASK-031.73), so no level is left stale. It relies on the caller's
        patch_sources() epoch bump to void a margin-warm job prepared before
        the edit, as patch_composite_rects() does."""
        if levels is not None:
            self._evict_rects([bbox], [mip for mip in self.resident_levels() if mip not in levels])
        self._composite_patch_targets(list(self._patch_targets(bbox, levels, rebuild_levels)))

    def _patch_targets(self, bbox: tuple[int, int, int, int], levels=None, rebuild_levels=None):
        """patch_composite()'s walk, compositing nothing: evicts bbox's chunks
        at a level that must evict, and yields (mip, cx, cy, chunk, ix0, iy0,
        ix1, iy1) for each other cached chunk bbox reaches, the slice in LEVEL
        pixels, in the serial loop's order.

        A caller may collect every walk before compositing any slice (a
        pooled patch must, to price the batch): a composite changes no level's
        evict test. It can build or flush only the level it composites, and
        that level evicts nothing either way: one inside rebuild_levels (or
        with none given) never takes the stale branch, one outside it is
        composited only while resident, so _level() does nothing there, and
        one with deferred work evicts and is never composited, so never flushed."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0 = lx0 // self.chunk_px, ly0 // self.chunk_px
            cx1, cy1 = (lx1 - 1) // self.chunk_px, (ly1 - 1) // self.chunk_px
            evict = (
                rebuild_levels is not None and mip not in rebuild_levels and not self.is_level_resident(mip)
            ) or self._defers_patch(mip)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self._cache.get((mip, cx, cy))
                    if chunk is None:
                        continue
                    if evict:
                        self._cache_bytes -= chunk.nbytes
                        del self._cache[(mip, cx, cy)]
                        continue
                    chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                    chunk_x1, chunk_y1 = chunk_x0 + chunk.shape[1], chunk_y0 + chunk.shape[0]
                    ix0, iy0 = max(lx0, chunk_x0), max(ly0, chunk_y0)
                    ix1, iy1 = min(lx1, chunk_x1), min(ly1, chunk_y1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    yield mip, cx, cy, chunk, ix0, iy0, ix1, iy1

    def _composite_patch_targets(self, targets: list) -> None:
        """Composites each _patch_targets() slice into its chunk: pooled when
        _pools_patch() says so (a zero-length `patch_pool=<jobs>/<targets>`
        Perf Trace phase marks it), else one _composite_rect() per slice in order."""
        if self._pools_patch(targets):
            jobs = self._pooled_patch(targets)
            with perf_trace.phase(f"patch_pool={jobs}/{len(targets)}"):
                pass
            return
        for mip, cx, cy, chunk, ix0, iy0, ix1, iy1 in targets:
            self._write_slice(chunk, mip, cx, cy, ix0, iy0, ix1, iy1, self._composite_rect(mip, ix0, iy0, ix1, iy1))

    def _pools_patch(self, targets: list) -> bool:
        """Whether a patch takes _pooled_patch(): render_rect()'s _pools()
        conditions, at least POOL_MIN_MISSING slices, and at least
        POOL_MIN_PATCH_TILE_AREA tiles summed over them. The area is in tiles
        of each slice's own level, so the gate is the same at every zoom: a
        typical brush-9 Draw step (about 344 tiles) stays serial, and the rare
        larger step pools by area like any other patch. Pooled only beat
        serial by 5% from about 1100 (TASK-031.67 step 0)."""
        if self._chunk_executor is None or not self._has_unit_pack or composite_backend.native is None:
            return False
        if len(targets) < self.POOL_MIN_MISSING:
            return False
        tiles = 0.0
        for mip, _cx, _cy, _chunk, ix0, iy0, ix1, iy1 in targets:
            tiles += (ix1 - ix0) * (iy1 - iy0) / self.mip_tile_px(mip) ** 2
        return tiles >= self.POOL_MIN_PATCH_TILE_AREA

    def _pooled_patch(self, targets: list) -> int:
        """Composites `targets` (_patch_targets() slices) on the chunk pool
        (TASK-031.67) and returns how many jobs it submitted. Per POOL_BATCH
        slices:

        1. Derive each level's unit pack (_derive_pack_for_slices(), which
           also builds or flushes the level as its first prepare would) and
           take one heights snapshot.
        2. Prepare each slice in walk order (prepare_rect_job()) and submit
           its kernel at once.
        3. Join (join_chunk_jobs()).
        4. In walk order, a painted job goes through install_slice(); any
           other slice (a declined prepare, a declined or raising kernel, a
           refused install) composites synchronously into the walked chunk,
           exactly as the serial loop does.

        Why that is the serial loop's result: nothing pumps the event loop
        between the first prepare and the last install, so no mutation
        interleaves (install_slice() still checks the epoch and the chunk's
        identity). Nothing is inserted, so _evict() never runs and every
        target chunk outlives its job, the same array object and LRU order.
        Slices of one chunk are disjoint unless the caller passed overlapping
        rects; those paint the same pixels from the same sources, and the
        installs run in walk order, so the last write is the serial one's. A
        level built by a later prepare voids no earlier job: _commit_level()
        rebinds that level's fields only and bumps no epoch, and each job
        holds its own arrays (`holds`). Workers run RectJob.run() alone, as
        in _pooled_chunks()."""
        executor = self._chunk_executor
        total = 0
        for start in range(0, len(targets), self.POOL_BATCH):
            batch = targets[start : start + self.POOL_BATCH]
            by_level: dict[int, list] = {}
            for mip, _cx, _cy, _chunk, ix0, iy0, ix1, iy1 in batch:
                by_level.setdefault(mip, []).append((ix0, iy0, ix1, iy1))
            for mip, rects in by_level.items():
                self._derive_pack_for_slices(mip, rects)
            # One per batch: elevations is not per level, and Sloped returns None.
            snapshot = self._heights_snapshot(batch[0][0])
            jobs: list[RectJob | None] = [None] * len(batch)
            painted: list[bool] = [False] * len(batch)
            submitted = []
            try:
                for i, (mip, cx, cy, _chunk, ix0, iy0, ix1, iy1) in enumerate(batch):
                    job = self.prepare_rect_job(mip, cx, cy, ix0, iy0, ix1, iy1, heights=snapshot)
                    if job is None:
                        continue
                    try:
                        future = executor.submit(job.run)
                    except Exception as exc:  # noqa: BLE001
                        debug_log.log(f"pooled patch: submit failed for chunk {job.key} ({exc!r})", level=debug_log.ERROR)
                        future = None
                    jobs[i] = job
                    submitted.append((i, job, future))
            except BaseException:
                # A prepare raised: don't leave this batch's kernels queued ahead of the next paint.
                for _i, _job, future in submitted:
                    if future is not None:
                        future.cancel()
                # Nothing from here on is patched, so evict it: the next paint composites it fresh.
                for target in targets[start:]:
                    key, chunk = target[:3], target[3]
                    if self._cache.get(key) is chunk:
                        del self._cache[key]
                        self._cache_bytes -= chunk.nbytes
                raise
            del snapshot
            with perf_trace.repaint_phase("join"):
                outcomes = join_chunk_jobs([(job, future) for _i, job, future in submitted])
            for (i, _job, _future), ok in zip(submitted, outcomes, strict=True):
                painted[i] = ok
            total += len(submitted)
            del submitted
            for i, (mip, cx, cy, chunk, ix0, iy0, ix1, iy1) in enumerate(batch):
                job, jobs[i] = jobs[i], None
                if job is not None and self.install_slice(job, painted[i]):
                    continue
                self._write_slice(chunk, mip, cx, cy, ix0, iy0, ix1, iy1, self._composite_rect(mip, ix0, iy0, ix1, iy1))
        return total

    def patch_area(self, bbox: tuple[int, int, int, int], levels=None) -> int:
        """Level pixels patch(bbox, levels=levels) would recomposite: bbox
        clipped to each cached chunk, summed over the resident levels. Mirrors
        patch()'s loop without compositing, so a caller can price the eager patch first."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return 0
        area = 0
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0, cx1, cy1 = self.chunk_index_range(mip, lx0, ly0, lx1, ly1)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self._cache.get((mip, cx, cy))
                    if chunk is None:
                        continue
                    chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                    w = min(lx1, chunk_x0 + chunk.shape[1]) - max(lx0, chunk_x0)
                    h = min(ly1, chunk_y0 + chunk.shape[0]) - max(ly0, chunk_y0)
                    if w > 0 and h > 0:
                        area += w * h
        return area

    def patch_rects(self, rects, levels=None) -> None:
        """patch() for each rect in rects -- Phase B-E's Flat edits patch
        render_paint.flat_tile_rects()' disjoint rects (one per block of tiles an
        edit touches, one a tile in a sparse block) rather than one union
        bbox (a union over a scattered undo set can span the whole map,
        turning patch() into a full recomposite; see
        ViewerWindow._apply_dirty's Flat branch).

        No elevation_changed parameter: this is Flat's only patch_rects()
        caller today, and FlatChunkCache._refresh_source_caches() is a
        documented no-op regardless of what's passed to it -- there is
        nothing here for that set to gate.

        Two passes, every rect's sources before any composite: a blended
        tile reads its neighbours' mirror entries, which a later rect may
        be the one refreshing. Between them the blend keys under every rect
        are derived in one call, since a sparse block's rects are one tile each.

        levels: None patches every resident level. Given (top-down Flat
        passes the level in view, TASK-338), only those are patched, and every
        other resident level drops its cached chunks under each rect, so no
        level is left stale. The level warm refills the neighbour levels at
        idle; the others recomposite on demand. The eviction and its
        resident_levels() scan run once per call, not once per rect
        (TASK-031.63's 12,100 one-tile rects), so the rects composite through
        _patch_targets() directly rather than patch_composite()."""
        live = [rect for rect in rects if self.patch_sources(rect)]
        self._fill_blend_keys(live)
        if levels is not None:
            self._evict_rects(live, [mip for mip in self.resident_levels() if mip not in levels])
        for rect in live:
            self._composite_patch_targets(list(self._patch_targets(rect, levels)))

    def _evict_rects(self, rects, mips) -> None:
        """Drops the cached chunks at each of `mips` that any of `rects`
        (REFERENCE canvas pixels) reaches: per rect, not under their union,
        so a scattered set does not drop a whole level."""
        for mip in mips:
            for rect in rects:
                rng = self.chunk_range_under(rect, mip)
                if rng is None:
                    continue
                cx0, cy0, cx1, cy1 = rng
                for cy in range(cy0, cy1 + 1):
                    for cx in range(cx0, cx1 + 1):
                        chunk = self._cache.pop((mip, cx, cy), None)
                        if chunk is not None:
                            self._cache_bytes -= chunk.nbytes

    def chunk_range_under(self, bbox: tuple[int, int, int, int], mip: int) -> tuple[int, int, int, int] | None:
        """(cx0, cy0, cx1, cy1), the inclusive chunk range a REFERENCE bbox
        reaches at level `mip`, clipped to that level's canvas; None when the
        clipped bbox is empty."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
        lx0, ly0, lx1, ly1 = max(lx0, 0), max(ly0, 0), min(lx1, canvas_w), min(ly1, canvas_h)
        if lx1 <= lx0 or ly1 <= ly0:
            return None
        return self.chunk_index_range(mip, lx0, ly0, lx1, ly1)

    def _fill_blend_keys(self, rects) -> None:
        """Derives every blend key Flat's patch rects `rects` (REFERENCE
        canvas pixels) will read, in one influences() call instead of one
        per rect (TASK-031.63: 12,100 one-tile rects cost 0.5 s more than
        the per-composite derivation they replaced). The composites then
        only gather. Through render_paint._rect_blend_keys(), the composites' own
        path, so the keys cannot differ."""
        if self._blend_key_map is None or self.style != "flat" or not rects or not render_paint.blend_effective(self.layers):
            return
        mm = self.scenario.map_manager
        w, h = mm.map_width, mm.map_height
        t = self.tile_px
        under = np.zeros((h, w), dtype=bool)
        for px0, py0, px1, py1 in rects:
            # Stops clamped too: a negative one would wrap and derive most of the map.
            under[max(py0 // t, 0) : max(-(-py1 // t), 0), max(px0 // t, 0) : max(-(-px1 // t), 0)] = True
        ys, xs = np.nonzero(under)
        render_paint._rect_blend_keys(
            self.scenario, np.stack([xs, ys], axis=1), self.layers, self._terrain_ids, self._layer_ids,
            self._blend_key_map,
        )

    def chunk_rects(self, rects, mip: int) -> list[tuple[int, int, int, int]]:
        """`rects` (REFERENCE canvas pixels) in per-chunk form at level `mip`:
        one REFERENCE rect per chunk any of them reaches, the bbox of the
        rects clipped to that chunk, cached or not. Rects that overlap inside
        a chunk become one composite there instead of one each (TASK-031.61
        C1's probe: 165 block rects patched one by one cost more than evicting
        their 22 chunks, the per-chunk form 55 ms less)."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        size = self.chunk_px
        per_chunk: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for rect in rects:
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, rect)
            lx0, ly0, lx1, ly1 = max(lx0, 0), max(ly0, 0), min(lx1, canvas_w), min(ly1, canvas_h)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0, cx1, cy1 = self.chunk_index_range(mip, lx0, ly0, lx1, ly1)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    clipped = (
                        max(lx0, cx * size), max(ly0, cy * size), min(lx1, (cx + 1) * size), min(ly1, (cy + 1) * size)
                    )
                    per_chunk[(cx, cy)] = _union_bbox(per_chunk.get((cx, cy)), clipped)
        return [self._level_bbox_to_reference(mip, box) for _key, box in sorted(per_chunk.items())]

    def patch_composite_rects(self, rects, levels=None, rebuild_levels=None, terrain_current: bool = False) -> None:
        """patch_composite() for each rect in `rects` (REFERENCE canvas
        pixels): one _composite_rect() per rect-chunk intersection, written
        into the cached chunk in place. For a batch unit edit whose union bbox
        is mostly pixels that did not change (TASK-031.61 C1:
        ViewerWindow._repaint_unit_edit_split() passes chunk_rects()).

        Runs no unit or elevation source work: the caller's invalidate_units()
        did, and this relies on that call's _mutation_epoch bump to void a
        margin-warm chunk job prepared before the edit (install_chunk()). A
        caller with no such bump behind it must not use this.

        terrain_current: the caller's promise that both terrain mirrors match
        every tile, so nothing is re-read. Without it (the default) the
        mirrors are re-read over every rect before any rect is composited,
        patch_rects()' two passes: a blended tile reads its neighbours' mirror
        entries, which a later rect may be the one refreshing.

        levels and rebuild_levels as in patch_composite(), except that
        `levels` here evicts nothing at the other levels: its one caller
        evicts those under the larger reach bbox first. A level outside
        rebuild_levels that went stale, or one with deferred work, evicts the
        chunks under each rect instead. A stale level inside rebuild_levels
        rebuilds here, in the first composite, not at the next paint. A chunk
        that is not cached is skipped, never composited: get_chunk() builds
        it fresh.

        Every rect's slices are walked first and composited as one set, so a
        chunk pool takes them in one fan-out (TASK-031.67: undo 1's rects are
        one chunk each, so a fan-out per rect would never reach two jobs)."""
        live = [rect for rect in rects if rect[2] > rect[0] and rect[3] > rect[1]]
        if not terrain_current:
            for rect in live:
                self._refresh_terrain_ids(rect)
        targets = [target for rect in live for target in self._patch_targets(rect, levels, rebuild_levels)]
        self._composite_patch_targets(targets)

    def invalidate_region(self, bbox: tuple[int, int, int, int], levels=None, terrain_changed: bool = True) -> None:
        """Evicts every cached chunk, at every RESIDENT mip level (only those
        in `levels`, when given), whose
        grid cell (once bbox is converted to that level) intersects it --
        forces a full recomposite from get_chunk() next time that chunk is
        requested, rather than trusting whatever's cached. Distinct from
        patch(): patch() keeps a cached chunk valid immediately at the cost
        of only the touched sub-rect; this drops it outright. Exists as its
        own primitive -- separate from patch() -- so a correctness bug
        elsewhere can't be masked by patch() quietly papering over it;
        tools/verify_iso_chunks.py's "invalidate_region round-trips" check
        calls this directly, forces a real recomposite via get_chunk(), and
        compares against a fresh full render.

        Per-level conversion fixes a real latent bug the old mip-agnostic
        chunk-index match had: a finer level's canvas is LARGER, so its
        chunk grid extends further -- a reference-derived index range used
        unconverted would under-evict a finer level's chunks near the
        canvas high edge, leaving stale pixels there. Coarser levels were
        merely over-evicted (harmless); this fixes both directions the same
        way, by computing each resident level's own true chunk-index
        range instead of reusing the reference's.

        terrain_changed=False skips the terrain-id mirror re-read, for a
        caller whose change is not terrain (units, layers, grid, sprites):
        the whole-canvas re-read costs 2.9 ms at 240x240 and 11 ms at 480x480."""
        _px0, _py0, _px1, _py1 = bbox
        self._mutation_epoch += 1
        if terrain_changed:
            self._refresh_terrain_ids(bbox)
        ranges: dict[int, tuple[int, int, int, int]] = {}
        for mip in self._select_levels(levels):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            ranges[mip] = (
                lx0 // self.chunk_px,
                ly0 // self.chunk_px,
                (lx1 - 1) // self.chunk_px,
                (ly1 - 1) // self.chunk_px,
            )
        for key in list(self._cache):
            mip, cx, cy = key
            rng = ranges.get(mip)
            if rng is None:
                continue
            cx0, cy0, cx1, cy1 = rng
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                self._cache_bytes -= self._cache[key].nbytes
                del self._cache[key]
